"""Phase 0: verifica que el entorno sirve ANTES de escribir la app.

Responde tres preguntas:
  1. Hay CUDA y cuanta VRAM?
  2. soundfile sabe leer mp3 y escribir mp3/wav24?  -> hace falta ffmpeg o no?
  3. Esta soxr para resamplear a 44.1k con calidad?
"""
import io
import tempfile
from pathlib import Path

import numpy as np

ok = True


def check(label, fn):
    global ok
    try:
        print(f"[ok]   {label}: {fn()}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")


# --- 1. CUDA -----------------------------------------------------------------
def cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} sin CUDA (build CPU?)")
    p = torch.cuda.get_device_properties(0)
    # una multiplicacion real en GPU, no solo is_available()
    a = torch.randn(2048, 2048, device="cuda")
    (a @ a).sum().item()
    return (f"torch {torch.__version__} | {p.name} | "
            f"{p.total_memory / 2**30:.1f} GiB | matmul GPU ok")


check("CUDA", cuda)

# --- 2. formatos de audio: necesitamos ffmpeg? -------------------------------
SR = 44100
sig = (0.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)


def formats():
    import soundfile as sf
    fmts = set(sf.available_formats())
    need = {"WAV", "FLAC", "MP3"}
    missing = need - fmts
    if missing:
        raise RuntimeError(f"libsndfile sin soporte para {missing}")
    return (f"soundfile {sf.__version__} | libsndfile "
            f"{sf.__libsndfile_version__} | WAV/FLAC/MP3 presentes")


def roundtrip():
    """Lo que de verdad importa: escribir mp3 y volver a leerlo."""
    import soundfile as sf
    results = []
    with tempfile.TemporaryDirectory() as d:
        for name, kw in [("a.wav", {"subtype": "PCM_24"}),
                         ("a.flac", {}),
                         ("a.mp3", {})]:
            p = Path(d) / name
            sf.write(p, sig, SR, **kw)
            back, sr = sf.read(p)
            assert sr == SR and len(back) > SR * 0.9, name
            results.append(f"{name} {p.stat().st_size // 1024}KB")
    return " | ".join(results)


def soxr():
    import soxr
    out = soxr.resample(sig, 48000, 44100, quality="VHQ")
    return f"soxr {soxr.__version__} | 48k->44.1k VHQ ok ({len(out)} muestras)"


def separator():
    from audio_separator.separator import Separator
    import audio_separator
    return f"audio-separator {audio_separator.__version__}"


check("audio formats", formats)
check("write/read roundtrip", roundtrip)
check("resample soxr", soxr)
check("audio-separator", separator)

print()
if ok:
    print("Phase 0 OK: todo in-process en Python, ffmpeg NO hace falta.")
else:
    print("Phase 0 con fallos -> arreglar antes de escribir la app.")
    raise SystemExit(1)
