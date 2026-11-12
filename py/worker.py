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
