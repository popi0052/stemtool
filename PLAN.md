# stemtool — plan

Herramienta personal: arrastras un audio → BPM + key al instante → separación en
vocals / drums / bass / other → descargas los que marques.

## Decisiones tomadas

| Decisión | Elegido | Por qué |
|---|---|---|
| Forma | **Webapp local** (servidor en tu PC + UI en el navegador) | Usa la RTX 4060 por CUDA. Drag&drop, audio player y descargas son nativos del navegador: cero librerías. Sin instalador, sin Rust, sin empaquetar Python. |
| Motor | **Sidecar Python + CUDA**, TS para todo lo demás | `audio-separator` 0.48 ya resuelve Demucs/RoFormer/MDX con CUDA. Reimplementar STFT/chunking en TS cuesta semanas y da peor calidad. |
| Pipeline | **Dos presets seleccionables** | Rápido para oír algo, Máximo cuando importa. Es un `if` en el runner, no dos arquitecturas. |
| Extras | Reproductor solo/mute, export wav/mp3, cola | Los tres pedidos. |

### Hardware y toolchain verificados en esta máquina
- RTX 4060 Laptop, **8188 MiB VRAM**, driver 580.97 → CUDA 12.x disponible.
- i7-13620H, 15.7 GB RAM.
- Node **v22.13.1** → corre `.ts` directo con `--experimental-strip-types`. Sin bundler para el servidor.
- Python **3.12.8** (Anaconda) → wheels de torch cu12x OK.
- **ffmpeg: falta.** Hay que instalarlo (Phase 0).

## Arquitectura

```
stemtool/
  package.json              node --experimental-strip-types, 0 deps runtime
  server/
    index.ts                http + static + /api/jobs + SSE de progreso
    queue.ts                1 worker secuencial, estado en jobs.json
    worker.ts               habla con el proceso Python por stdin/stdout JSON-lines
  py/
    worker.py               bucle: lee job JSON por stdin, emite progreso por stdout
    separate.py             audio-separator (1 o 2 etapas según preset)
    analyze.py              BPM + key (librosa)
  web/
    index.html              dropzone, lista de jobs, tarjeta por canción
    app.ts → dist/app.js    tsc, sin bundler
  data/<jobId>/             mix.flac + stems + analysis.json
```

**Un solo proceso Python vivo**, no uno por canción: cargar un checkpoint de RoFormer
cuesta 5–20 s y con una cola eso se paga en cada archivo. El worker mantiene el
`Separator` en memoria entre jobs.

**Por qué TypeScript manda y Python es un binario tonto:** todo lo que es estado,
cola, API, tipos y UI vive en TS. Python solo recibe `{in, out, preset}` y escribe
líneas de progreso. Si mañana aparece un motor mejor, se cambia un archivo.

## Pipeline de calidad (esto es lo que decide el resultado)

### 0. Ingesta — decodificar una vez, nunca re-codificar
```
ffmpeg -i entrada.mp3 -af aresample=resampler=soxr:precision=28 -ar 44100 -c:a flac mix.flac
```
- Los modelos están entrenados a **44.1 kHz**: resamplear ahí con soxr, no con el
  resampler por defecto de ffmpeg.
- Todos los intermedios en **FLAC o WAV float32**. Un mp3 intermedio arruina el
  trabajo del modelo y es el error más común.
- **No volver a subir a 48 kHz** la salida: no recupera nada, solo pesa más.

### 1. Análisis inmediato (mientras el usuario mira la pantalla)
- **BPM**: `librosa.beat.beat_track` sobre el mix, con guarda de octava —
  plegar al rango 70–180 y elegir la octava que concuerde con la autocorrelación
  de onsets. El error de doble/mitad de tempo es el fallo dominante.
- **Key**: Krumhansl–Schmuckler/Temperley sobre `chroma_cqt` de la parte armónica
  (`librosa.effects.hpss`). Devolver **las 2 mejores candidatas con confianza**, no
  una sola respuesta: la detección de key es honestamente el punto débil de todo el
  estado del arte (los papers de 2026 rondan 73–82 de weighted error en GiantSteps).
- Tarda ~3–8 s en una canción. Se muestra antes de empezar a separar.

### 2. Refinado gratis después de separar
Esto es la mejor práctica que casi ninguna herramienta aplica:
- **BPM recalculado sobre el stem de drums** → onsets limpios, mucho más fiable.
- **Key recalculada sobre bass + other** (sin voz ni percusión) → chroma mucho más
  limpio.
- Si el refinado discrepa del estimado inicial, se muestra el refinado y se marca
  que cambió. Coste: dos llamadas más a librosa sobre audio que ya existe.

### 3. Separación

**Preset Rápido** (~1–2 min/canción en la 4060):
```
htdemucs_ft, overlap 0.25, shifts 1 → vocals, drums, bass, other
```

**Preset Máximo** (~4–6 min/canción) — cascada de dos etapas:
```
1. Mel-Band / BS-RoFormer   mix → vocals + instrumental
2. htdemucs_ft              instrumental → drums, bass, other
```
- RoFormer saca ~1–2 dB más de SDR en voces que Demucs; Demucs sigue siendo la
  mejor opción para bass y drums (las bandas mel de RoFormer modelan peor el bajo).
  La cascada toma lo mejor de cada uno.
- `shifts 2–5`: el shift trick promedia predicciones sobre desplazamientos
  aleatorios, ~+0.2 dB SDR a cambio de ser N veces más lento. `overlap 0.25` es el
  default y está bien.
- **VRAM**: 8 GB alcanza, pero RoFormer es el que aprieta. Si hay OOM, bajar
  `segment_size` — exponerlo como constante, no como opción de UI.

### 4. Salida
- **FLAC 44.1 kHz por defecto** (lossless, ~mitad del tamaño de WAV).
- Export opcional a **WAV 24-bit** y **MP3 320** vía ffmpeg, desde el FLAC.
- **No normalizar los stems por separado**: cambia el balance entre ellos y rompe
  la posibilidad de volver a sumarlos. Los stems pueden pasar de 0 dBFS
  individualmente y eso es correcto; FLAC/WAV 24-bit lo aguanta. Solo el export a
  MP3 lleva un limitador de seguridad.

### 5. Null test automático (QC)
Sumar los 4 stems y restar el mix. El residual en dB es una medida objetiva y
gratuita de si la separación salió bien. Se guarda en `analysis.json` y se muestra
en la tarjeta. Si el residual es alto, algo falló y lo sabes sin tener que escuchar.

## Fases

**Phase 0 — entorno** (1 sesión)
Instalar ffmpeg. Crear venv, `pip install "audio-separator[gpu]" librosa`, torch cu12x.
Verificar `torch.cuda.is_available()` y separar una canción por CLI. **No escribir
nada de la app hasta que esto funcione**: aquí es donde se rompen estos proyectos.

**Phase 1 — worker Python**
`worker.py` + `separate.py` + `analyze.py`, manejados a mano por stdin. Preset
rápido y cascada. Null test. Se valida con 3 canciones de géneros distintos antes
de seguir.

**Phase 2 — servidor TS**
`node:http` sirviendo estáticos, `POST /api/jobs` con el upload, cola secuencial,
SSE para progreso, `GET /api/jobs/:id/stems.zip?pick=vocals,drums` (zip con
`zipfile` de la stdlib de Python, sin dependencia nueva).

**Phase 3 — UI**
Una página. Dropzone (evento `drop` nativo), una tarjeta por canción con BPM/key/
progreso, checkboxes por stem, botón de descarga. Sin framework.

**Phase 4 — reproductor**
Web Audio API: `decodeAudioData` de los 4 stems, 4 `AudioBufferSourceNode`
arrancados en el mismo `ctx.currentTime` (sample-accurate, a diferencia de 4
elementos `<audio>` que derivan), un `GainNode` por stem para mute/solo. ~40 líneas.

**Phase 5 — exports**
WAV 24-bit y MP3 320 por ffmpeg, bajo demanda y cacheados.

## Lo que NO se construye

- **Cuentas, auth, multiusuario**: corre en localhost para ti.
- **Base de datos**: `jobs.json` y una carpeta por job. Cuando moleste, SQLite.
- **Docker**: añade una capa de GPU passthrough por cero beneficio en local.
- **Separación en el navegador** (WASM/WebGPU): deja la GPU sin usar por CUDA y
  topa a 2–4 GB de memoria. Se reconsidera solo si quieres usarlo desde el móvil.
- **6 stems (guitar/piano)**: `htdemucs_6s` existe y es un cambio de un string.
  Se añade cuando lo pidas, no antes.
- **Input de YouTube**: no lo pediste.

## Riesgos reales

1. **Instalación de torch + CUDA en Windows/Anaconda** es la parte más frágil de
   todo el proyecto. Phase 0 existe para que falle temprano y aislado.
2. **Calidad de la detección de key**: es mala en todas las herramientas, no solo
   aquí. Mitigación: top-2 con confianza y recálculo sobre stems armónicos. Si no
   basta, el siguiente paso es un modelo CNN dedicado, no afinar los templates.
3. **BPM de doble/mitad**: la guarda de octava cubre el 90%; medir contra tus
   propias canciones etiquetadas antes de confiar.
4. **VRAM en la cascada**: 8 GB es suficiente pero sin holgura. `segment_size`
   ajustable en código.

## Estado real (2026-11-07)

### Verificado con ejecucion, no supuesto
- **venv persistente** en `~/.venvs/stemtool` (fuera de la carpeta
  de sesion, no se borra con ella). torch **2.14.1+cu126**, CUDA activa,
  comprobada con un matmul real en GPU.
- Elegido cu126 y no cu130 porque onnxruntime-gpu se compila contra CUDA 12 y
  mezclar runtimes en un proceso da problemas. Driver 580.97 lo soporta.
- **No hace falta el extra `[gpu]`**: Demucs y RoFormer son PyTorch; ONNX solo
  lo usan los MDX-Net, que no usamos. Ahorra ~2 GB.
- `audio-separator` **0.48.0** con dos deps no declaradas que hay que instalar a
  mano: `onnxruntime` (la CPU basta) y `audioread` (lo importa `spec_utils.py`
  pero librosa 0.11 ya no lo arrastra).
- **ffmpeg SI hace falta**, al contrario de lo que supuse: nuestro I/O va entero
  por soundfile+soxr, pero `audio-separator` ejecuta `ffmpeg -version` en su
  constructor y revienta sin el. Instalado 9.0.2 por winget.
- soundfile **0.14.0** / libsndfile **1.2.2**: WAV/FLAC/MP3 lectura y escritura
  comprobadas con roundtrip real. soxr 1.1.0 para 44.1k.
- Modelos reales (el catalogo trae 97 MDXC y 4 Demucs):
  `htdemucs_ft.yaml` y `model_bs_roformer_ep_317_sdr_12.9755.ckpt`.
  htdemucs_ft ya descargado en `~/.cache/stemtool-models` (4 x 80 MB).
- `normalization_threshold` es `if peak > umbral: escalar`. El default **0.9 da a
  cada stem una ganancia distinta**: rompe el balance y el null test. Puesto a 1.0.

### Analisis BPM/key: funciona y esta medido
Contra el fixture sintetico (120.0 BPM, A menor, `py/make_test_audio.py`):
- BPM **120.2** en mix y en drums; la guarda de octava descarta 80.1 y 180.3.
- Key **A minor** correcta en mix y en bass+other.
- Dos arreglos con fundamento, no tanteo:
  1. `np.median` sobre el cromagrama estaba mal: KS mide energia total por clase
     de altura y la mediana aplana el enfasis tonal. Ahora normaliza por frame
     y promedia.
  2. Perfiles de **Temperley**, no Krumhansl-Kessler: KK salio de probe-tone
     sobre menor natural y confunde menor con mayor en cuanto suena una
     dominante con su sensible. El `key_margin` subio de 0.013 a 0.108.

### Dos trampas de la API que costaron el rato

**1. `separate()` devuelve nombres relativos.** Son `"s_vocals.flac"`, no rutas.
El `shutil.move` los buscaba en el CWD y daba un `FileNotFoundError [WinError 2]`
que parecia un ffmpeg sin resolver. La separacion llevaba funcionando desde el
principio. Se resuelven contra el `output_dir`.

**2. `output_dir` se fija al construir el Separator y no se relee.** Como el
Separator se reutiliza entre canciones (para no recargar el checkpoint), un
directorio de staging por job hace que a partir del segundo el modelo siga
escribiendo en el del primero. El sintoma es desconcertante: las rutas que
devuelve la libreria existen, pero los ficheros no estan ahi. Solucion: **un
solo `<data>/.staging` para todo el proceso**, que se vacia entre jobs en vez
de borrarse. La cola es secuencial, asi que no hay riesgo de solape.

Las dos se manifestaron como errores de fichero no encontrado apuntando en la
direccion equivocada; por eso el test de regresion son los dos presets seguidos
en el mismo proceso, no cada uno por su cuenta.

### Probado de punta a punta
- Servidor arriba, subida PUT con cuerpo crudo, cola, SSE, UI.
- `prepare -> analyze -> separate -> refine` completo en los **dos presets**,
  encolados seguidos en el mismo proceso worker (el caso que destapo el bug 2):

  | preset | modelos | null test |
  |---|---|---|
  | fast | htdemucs_ft | **-32.6 dB** |
  | max | BS-RoFormer -> htdemucs_ft | **-31.6 dB** |

  Esa es la referencia para el null test: por encima de unos -25 dB conviene
  mirar que paso. Los dos dan 120.2 BPM y A minor, y el refinado sobre los
  stems sube la confianza del BPM de 0.848 a 0.873.
- Reproductor: los 4 stems decodificados y sincronizados, transporte avanzando,
  mute y solo correctos (al quitar el solo recuerda lo que estaba silenciado).
- Las 4 rutas de descarga: FLAC suelto, zip multi-FLAC, MP3 320, zip WAV 24.
- `npm run check` limpio en frontend y servidor.

Lo que **no** esta probado es la calidad de la separacion: el fixture es
sintetico y solo valida que el pipeline cuadra. Para eso hace falta una cancion
real y escucharla.

## Como se arranca

```
npm install
npm run build        # compila web/app.ts -> web/dist/web/app.js
npm start            # http://localhost:4747
```

El venv de Python en `~/.venvs/stemtool` ya esta montado; `server/worker.ts` lo
busca ahi solo, o en `STEMTOOL_PYTHON` si se quiere otro. Los modelos se cachean
en `~/.cache/stemtool-models` y se descargan la primera vez que se usa cada
preset (htdemucs_ft ~320 MB, BS-RoFormer ~1 GB).

## Fuentes

- [MVSEP algorithm leaderboard](https://mvsep.com/en/algorithms) — ranking SDR actual
- [audio-separator (PyPI)](https://pypi.org/project/audio-separator/) — v0.48.0, CUDA/CoreML/DirectML
- [Mel-Band RoFormer paper](https://arxiv.org/pdf/2310.01809) — mel bands mejoran voz, no bajo
- [MSST benchmark framework](https://arxiv.org/pdf/2607.23395)
- [htdemucs vs BS-RoFormer vs Spleeter 2026](https://aistemsplitter.org/blog/htdemucs-vs-bs-roformer-vs-spleeter-2026-benchmark) — ~+1.3 dB a favor de RoFormer
- [Guía RoFormer en producción 2026](https://tomodahinata.com/en/blog/bs-roformer-mel-band-roformer-vocal-separation-guide)
- [UVR: mejor modelo para instrumental](https://github.com/Anjok07/ultimatevocalremovergui/discussions/444) — práctica de la cascada
- [demucs separate.py](https://huggingface.co/spaces/IvyGlenn/RVC_HFv3/blob/main/demucs/separate.py) — shifts, overlap, float32
- [BPM/key en navegador con essentia.js](https://dev.to/tagmybeat/i-built-a-free-music-tool-i4l) — 42%→85% exacto combinando detectores
- [KeyMyna, key detection SOTA 2026](https://arxiv.org/pdf/2604.10021)
- [ONNX Runtime Web: large models](https://onnxruntime.ai/docs/tutorials/web/large-models.html) — techos de 2/4 GB
- [Tauri sidecar](https://v2.tauri.app/zh-cn/develop/sidecar/) — lo que nos ahorramos

## Interfaz (2026-11-16)

Rediseño con estética de Mirror's Edge: fondo blanco, geometría recta sin
esquinas redondeadas y corchetes de visor en la zona de soltar. El negro de ME
es el modo oscuro, no el tema por defecto.

El acento es turquesa `#0fb9ab`, no el rojo del juego, y está reservado para lo
que se toca: botón de descarga, preajuste activo, M/S encendidos, barra viva de
la tarjeta. Como texto sobre blanco el turquesa brillante no llega a contraste,
así que hay un segundo tono `--tq-ink` solo para eso. Lo único que sigue siendo
rojo es el estado de error, que es semántico y no decorativo; la duda (key poco
clara, BPM que no cuadra con el del mix) va en ámbar.

### La fuente, solo en los títulos

`web/fonts/bmspace.ttf` es BM Space (BitmapMania / cooz, 2001), fuente de mapa
de bits. Se usa en dos sitios y nada más: el logotipo y el titular de la zona
de soltar. Las dos razones son restricciones de la propia fuente:

- **unitsPerEm = 900**, es decir 9 px por em. Solo se ve nítida en múltiplos de
  9, y de ahí los dos únicos tamaños que aparecen: 45 px el logotipo, 27 px el
  titular. En 14 px o 16 px el navegador interpola y los píxeles salen sucios,
  que es justo lo que pasaría si se usara para texto corrido.
- **96 glifos, sin acentos.** No existen á é í ó ú ñ ¿ ¡. Mientras se quede en
  títulos en inglés no molesta; en cuanto toca interfaz en español, cada
  palabra con tilde se parte en dos tipografías a media palabra.

El resto va en la sans del sistema. Las versalitas las pone el CSS con
`text-transform: uppercase`, no el texto fuente, porque así "duración" se
convierte en "DURACIÓN" con su acento en vez de quedarse escrito sin él.

Licencia: gratuita, uso personal permitido, prohibido adjuntarla a un producto
comercial sin permiso del autor (`web/fonts/LICENSE-bmspace.txt`, en japonés).

### Lo que cambió en el uso, no solo en el aspecto

- Medidor de nivel por stem, **antes de la ganancia**: un stem silenciado sigue
  enseñando señal, que es como se detecta una fuga entre pistas.
- `espacio` reproduce o pausa; `M` y `S` sobre la fila de un stem enfocada.
- El `dragleave` lleva contador: con una bandera simple el marco parpadea al
  pasar el cursor por encima de un hijo de la zona de soltar.
- La zona de soltar dice el límite real (600 MB) y lleva botón visible, que es
  el respaldo para quien no arrastra.
- El total de la canción sale del análisis, así el transporte muestra
  `0:00 / 3:38` antes de descargar los stems.

### Fuentes del rediseño

- [BM Space en dafont](https://www.dafont.com/es/bm-space.font)
- [Paleta y color de Mirror's Edge](https://www.worldofleveldesign.com/categories/game_environments_design/mirrors-edge-color.php) — blanco neutro de base, rojo como señal de navegación
- [Style and gameplay in Mirror's Edge (DICE)](https://de.slideshare.net/DICEStudio/style-and-gameplay-in-the-mirrors-edge)
- [stemplayer-js](https://unpkg.com/@stemplayer-js/stemplayer-js@4.1.0-beta.4/README.md) — fila por stem con solo/mute/volumen
- [Demo stem separator de WebNN](https://microsoft.github.io/webnn-developer-preview/demos/stem-separator/) — atajos M y S
- [File uploader UX best practices](https://uploadcare.com/blog/file-uploader-ux-best-practices/) — formatos y límite en el estado vacío, progreso real
- [Upload file UI: componentes, estados y errores](https://blog.filestack.com/upload-file-ui-design-components-states-and-errors/)
