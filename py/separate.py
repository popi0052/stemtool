"""Separacion en 4 stems. Dos presets.

  fast: htdemucs_ft en una pasada -> vocals, drums, bass, other
  max:  BS-RoFormer saca la voz (SDR 12.97) y htdemucs_ft parte el
        instrumental. Cada modelo en lo que gana: las bandas mel de RoFormer
        modelan peor el bajo, y Demucs es el mejor en bass/drums.
"""
import logging
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from audio_separator.separator import Separator

CACHE = Path.home() / ".cache" / "stemtool-models"
STEMS = ("vocals", "drums", "bass", "other")

DEMUCS = "htdemucs_ft.yaml"  # el fine-tuned: 4x mas lento que htdemucs, mejor SDR
ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

# Pesos para la barra de progreso. ponytail: progreso por etapa, no real.
# audio-separator reporta via tqdm y no acepta callback; interceptarlo es
# monkeypatching de una libreria ajena. Si molesta, ahi esta el camino.
WEIGHTS = {"fast": [("htdemucs_ft", 0.0, 0.92)],
           "max": [("BS-RoFormer (voces)", 0.0, 0.5),
                   ("htdemucs_ft (resto)", 0.5, 0.92)]}

_sep = None
_loaded: str | None = None
_staging: Path | None = None


def _get_staging(out: Path) -> Path:
    """UN solo directorio de staging para todo el proceso, no uno por job.

    `output_dir` se fija al construir el Separator y la libreria no lo relee
    despues. Como el Separator se reutiliza entre canciones, un staging por job
    hace que a partir del segundo el modelo siga escribiendo en el directorio
    del primero: las rutas que devuelve existen, pero los ficheros no estan
    donde se los espera. Todos los jobs cuelgan del mismo padre, asi que un
    unico `<data>/.staging` vale y es estable.
    """
    global _staging
    if _staging is None:
        _staging = out.parent / ".staging"
    _staging.mkdir(parents=True, exist_ok=True)
    return _staging


def _clear(d: Path):
    for p in d.glob("*"):
        p.unlink(missing_ok=True)


def _separator(staging: Path):
    """Un solo Separator para todo el proceso; el modelo se cachea aparte."""
    global _sep
    if _sep is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        _sep = Separator(
            log_level=logging.WARNING,
            model_file_dir=str(CACHE),      # el default es /tmp, inservible en Windows
            output_dir=str(staging),
            output_format="FLAC",           # lossless y la mitad que WAV
            # 1.0 = solo evita clipping al escribir entero. El default 0.9 da a
            # cada stem una ganancia DISTINTA: rompe el balance y el null test.
            normalization_threshold=1.0,
            use_soundfile=True,             # evita pydub (y su dependencia de ffmpeg)
            # ponytail: fp32. use_autocast=True va ~1.5-2x mas rapido pero el
            # objetivo declarado es calidad; es un flag si hace falta velocidad.
            use_autocast=False,
            sample_rate=44100,
            # shifts=2 (default) es el punto medio. Subirlo da ~+0.2 dB por 2x
            # de tiempo; el salto grande esta en la cascada, no aqui.
            demucs_params={"segment_size": "Default", "shifts": 2,
                           "overlap": 0.25, "segments_enabled": True},
            # override_model_segment_size=False -> usa el del modelo, que es con
            # el que se entreno. Si hay OOM en 8 GB, bajarlo aqui.
            mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                         "batch_size": None, "overlap": None, "pitch_shift": 0},
        )
    return _sep


def _run_model(staging: Path, model: str, src: Path, names: dict[str, str]):
    """Carga el modelo si cambio y separa. Devuelve {stem: Path}."""
    global _loaded
    sep = _separator(staging)
    if _loaded != model:
        sep.load_model(model)  # 5-20s: por eso el proceso no muere entre jobs
        _loaded = model
    written = sep.separate(str(src), custom_output_names=names)
    out = {}
    for name in written:
        # separate() devuelve nombres RELATIVOS ("s_vocals.flac"), no rutas:
        # hay que resolverlos contra el output_dir o no se encuentran.
        p = staging / Path(name).name
        for stem, base in names.items():
            if p.stem == base:
                out[stem] = p
    return out


def _null_test(mix: Path, stems: dict[str, Path]) -> float:
    """Suma de stems menos mix, en dB relativos al mix.

    Medida objetiva y gratis de si la separacion cuadra. Mas negativo = mejor.
    No es SDR: solo dice si se perdio o se duplico energia por el camino.
    """
    ref, _ = sf.read(mix, always_2d=True, dtype="float32")
    total = np.zeros_like(ref)
    for p in stems.values():
        s, _ = sf.read(p, always_2d=True, dtype="float32")
        n = min(len(s), len(total))
        total[:n] += s[:n, : total.shape[1]]
    resid = np.sqrt(np.mean((ref - total) ** 2))
    base = np.sqrt(np.mean(ref ** 2))
    return float(20 * np.log10(resid / base + 1e-12))


def run(path, out, preset="fast", progress=lambda *_: None):
    """Separa `path` dejando <out>/<stem>.flac. Devuelve rutas y null test."""
    mix, out = Path(path), Path(out)
    staging = _get_staging(out)
    _clear(staging)  # restos de un job anterior que fallara a medias
    stages = WEIGHTS[preset]

    try:
        if preset == "fast":
            progress(0.02, stages[0][0])
            got = _run_model(staging, DEMUCS, mix,
                             {s: f"s_{s}" for s in STEMS})
        else:
            progress(0.02, stages[0][0])
            voc = _run_model(staging, ROFORMER, mix,
                             {"vocals": "s_vocals", "instrumental": "s_instrumental"})
            if "vocals" not in voc:
                raise RuntimeError(f"RoFormer no dio voces, solo {list(voc)}")
            # El segundo stem deberia llamarse "instrumental", pero si el modelo
            # lo nombra de otra forma se coge el que no sea la voz, en vez de
            # reventar con un KeyError que no dice nada.
            inst = voc.get("instrumental") or next(
                (p for s, p in voc.items() if s != "vocals"), None)
            if inst is None:
                raise RuntimeError("RoFormer no dejo pista instrumental")

            progress(0.5, stages[1][0])
            rest = _run_model(staging, DEMUCS, inst,
                              {"drums": "s_drums", "bass": "s_bass",
                               "other": "s_other", "vocals": "s_dropped"})
            got = {"vocals": voc["vocals"],
                   **{k: rest[k] for k in ("drums", "bass", "other") if k in rest}}

        missing = [s for s in STEMS if s not in got]
        if missing:
            raise RuntimeError(f"el modelo no produjo: {missing}")

        final = {}
        for stem in STEMS:
            dest = out / f"{stem}.flac"
            shutil.move(str(got[stem]), dest)
            final[stem] = str(dest)

        progress(0.94, "null test")
        residual = _null_test(mix, {k: Path(v) for k, v in final.items()})
        progress(1.0, None)
        return {"stems": final, "residual_db": round(residual, 1)}
    finally:
        # Se vacia, no se borra: el Separator guarda la ruta y tiene que seguir
        # existiendo para el siguiente job.
        _clear(staging)
