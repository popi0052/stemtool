"""Worker Python: un proceso vivo, una linea JSON por mensaje.

stdout es SOLO el protocolo. Todo lo demas (logs, tqdm, warnings, prints de
terceros) va a stderr.

Modo one-shot, para tareas baratas que no deben esperar a que la GPU acabe:
    python worker.py zip    <out.zip> <f1> <f2> ...
    python worker.py export <src> <out> <wav24|mp3>
"""
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path


def _fix_path():
    """winget deja ffmpeg en el PATH del registro, pero los procesos ya
    arrancados heredan el viejo. audio-separator ejecuta `ffmpeg -version`
    en su constructor y revienta sin el."""
    if shutil.which("ffmpeg") or platform.system() != "Windows":
        return
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            os.environ["PATH"] = (winreg.QueryValueEx(k, "PATH")[0]
                                  + os.pathsep + os.environ["PATH"])
    except OSError:
        pass


_fix_path()

# El protocolo sale por el stdout real; se sustituye sys.stdout por stderr para
# que ningun print() de una libreria pueda corromper el canal.
_proto = sys.stdout
sys.stdout = sys.stderr
logging.basicConfig(stream=sys.stderr, level=logging.WARNING,
                    format="%(levelname)s %(name)s: %(message)s")

import numpy as np          # noqa: E402
import soundfile as sf      # noqa: E402
import soxr                 # noqa: E402

SR = 44100


def emit(**kw):
    _proto.write(json.dumps(kw) + "\n")
    _proto.flush()


# --- lectura y preparacion ---------------------------------------------------

def _read_audio(src):
    """soundfile cubre wav/flac/mp3/ogg; ffmpeg el resto (m4a, aac, wma)."""
    try:
        data, sr = sf.read(src, always_2d=True, dtype="float32")
        if len(data):
            return data, sr
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "decoded.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-i", str(src), "-c:a", "pcm_f32le", str(tmp)],
                       check=True)
        return sf.read(tmp, always_2d=True, dtype="float32")


def h_prepare(src, out):
    """Decodifica UNA vez a 44.1 kHz lossless. Todo el pipeline parte de aqui.

    44.1k porque es la frecuencia a la que se entrenaron los modelos, y soxr
    VHQ porque el resampler barato mete aliasing que el modelo luego intenta
    separar como si fuera musica.
    """
    data, sr = _read_audio(src)
    if sr != SR:
        data = soxr.resample(data, sr, SR, quality="VHQ")
    sf.write(out, data, SR, subtype="PCM_24")
    return {"duration": round(len(data) / SR, 2), "sr_in": sr,
            "channels": int(data.shape[1])}


# --- handlers del bucle ------------------------------------------------------

def click_track(out, bpm, offset, duration, sr=SR):
    """Metronomo de comprobacion: clics a tempo CONSTANTE desde el primer beat.

    Constante a proposito, y ahi esta toda la gracia. Un metronomo pegado a
    los beats que detecto el tracker sonaria bien aunque el numero fuese
    falso; este empieza donde empieza la musica y sigue a ritmo fijo, asi que
    si el BPM esta mal el clic se va separando y se oye solo. Es la diferencia
    entre comprobar el dato y decorarlo.

    Mono: es una pista de control, no material. Entra en cualquier DAW igual.
    """
    period = 60.0 / bpm
    n = int(round(duration * sr))
    k = np.arange(int(0.03 * sr)) / sr
    # Seno de 1 kHz con caida exponencial: pega fuerte en el transitorio, que
    # es lo que deja ver si el clic cae sobre el golpe o al lado.
    click = (np.sin(2 * np.pi * 1000 * k) * np.exp(-k * 90) * 0.6)
    track = np.zeros(n + len(click), dtype="float32")
    times = np.arange(offset, duration, period)
    for t in times:
        i = int(round(t * sr))
        track[i:i + len(click)] += click
    sf.write(out, track[:n], sr, subtype="PCM_24")
    return {"clicks": len(times), "bpm": bpm}


def h_analyze(path):
    import analyze
    return analyze.analyze(path)


def h_analyze_stems(stems):
    import analyze
    return analyze.analyze_stems(stems)


def h_separate(path, out, preset):
    import separate
    return separate.run(path, out, preset,
                        lambda p, stage=None: emit(progress=p, stage=stage))


HANDLERS = {
    "prepare": lambda r: h_prepare(r["src"], r["out"]),
    "analyze": lambda r: h_analyze(r["path"]),
    "analyze_stems": lambda r: h_analyze_stems(r["stems"]),
    "separate": lambda r: h_separate(r["path"], r["out"], r.get("preset", "fast")),
}


# --- one-shot ----------------------------------------------------------------

def _selfcheck():
    """python worker.py --selfcheck: que el clic caiga donde dice que cae."""
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "click.flac")
        bpm, offset, dur = 170.0, 0.137, 12.0
        info = click_track(out, bpm, offset, dur)
        y, sr = sf.read(out, dtype="float32")
        assert sr == SR and abs(len(y) / sr - dur) < 0.01, "duracion"
        # Un clic son varios lobulos de seno por encima del umbral; se agrupan
        # los que esten a menos de 20 ms y cada grupo cuenta como un golpe.
        loud = np.flatnonzero(y > 0.3)
        starts = loud[np.r_[True, np.diff(loud) > sr // 50]] / sr
        assert len(starts) == info["clicks"], f"{len(starts)} != {info['clicks']}"
        assert abs(starts[0] - offset) < 0.002, f"fase: {starts[0]:.4f}"
        drift = float(np.abs(np.diff(starts) - 60.0 / bpm).max())
        assert drift < 0.002, f"deriva entre clics: {drift * 1000:.1f} ms"
    print(f"selfcheck ok: {info['clicks']} clics a {bpm} BPM, "
          f"primero en {offset}s, deriva < 2 ms")


def one_shot(argv):
    cmd = argv[0]
    if cmd == "zip":
        out, rest = argv[1], argv[2:]
        # Pares nombre/ruta alternados en vez de "nombre=ruta": el nombre lo
        # pone quien llama (lleva BPM y key) y no hay delimitador que un titulo
        # de cancion pueda romper.
        # ZIP_STORED: FLAC ya viene comprimido, deflate solo gastaria CPU.
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
            for arc, src in zip(rest[::2], rest[1::2]):
                z.write(src, arc)
    elif cmd == "click":
        out, bpm, offset, duration = argv[1], *map(float, argv[2:5])
        click_track(out, bpm, offset, duration)
    elif cmd == "--selfcheck":
        _selfcheck()
    elif cmd == "export":
        src, out, fmt = argv[1], argv[2], argv[3]
        if fmt == "wav24":
            data, sr = sf.read(src, always_2d=True, dtype="float32")
            sf.write(out, data, sr, subtype="PCM_24")
        elif fmt == "mp3":
            # ffmpeg y no soundfile: el bitrate es inequivoco. Sin limitador,
            # porque la fuente es FLAC entero y no puede pasar de fondo escala.
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                            "-i", src, "-c:a", "libmp3lame", "-b:a", "320k", out],
                           check=True)
        else:
            raise ValueError(f"formato no soportado: {fmt}")
    else:
        raise ValueError(f"comando one-shot desconocido: {cmd}")


def main():
    emit(ready=True)
    for line in sys.stdin:
        # El ﻿ es por si alguien alimenta el worker con un fichero que
        # lleva BOM; sin esto el primer mensaje revienta y no se ve por que.
        line = line.strip().lstrip("﻿")
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.pop("cmd")
            emit(result=HANDLERS[cmd](req))
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            emit(error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    if len(sys.argv) > 1:
        one_shot(sys.argv[1:])
    else:
        main()
