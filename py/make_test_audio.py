"""Genera un mix sintetico con ground truth conocido, para validar el pipeline.

120.0 BPM exactos, A menor, progresion Am-Dm-E-Am.

La progresion es i-iv-V-i a proposito: el E MAYOR mete un G#, que no existe en
Do mayor, y eso rompe la ambiguedad de tonalidad relativa. Un bucle tipo
Am-F-C-G seria vi-IV-I-V de Do mayor y no tendria respuesta unica, asi que no
serviria como test.

Escribe mix.wav + los 4 stems de referencia, asi se puede medir el null test
y comprobar el analisis contra una respuesta que si es defendible.

Uso: python make_test_audio.py <carpeta_salida>
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM  # 0.5 s
BARS = 8
N = int(BARS * 4 * BEAT * SR)
T = np.arange(N) / SR

# (root del bajo, triada) -> Am - Dm - E - Am
PROG = [
    (55.00, [220.00, 261.63, 329.63]),  # Am : A  C  E
    (73.42, [293.66, 349.23, 440.00]),  # Dm : D  F  A
    (41.20, [329.63, 415.30, 493.88]),  # E  : E  G# B   <- el G# decide
    (55.00, [220.00, 261.63, 329.63]),  # Am
]
SCALE = [440.00, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99]  # A4..G5
rng = np.random.default_rng(0)


def env(start, length, attack=0.004, decay=None):
    """Envolvente percusiva sobre el buffer completo."""
    e = np.zeros(N)
    i0, i1 = int(start * SR), min(N, int((start + length) * SR))
    if i0 >= N:
        return e
    k = np.arange(i1 - i0) / SR
    e[i0:i1] = np.minimum(k / attack, 1.0) * np.exp(-k / (decay or length / 3))
    return e


def tone(freq, start, length, harmonics=(1.0,), vibrato=0.0, **kw):
    """Suma de armonicos con vibrato opcional, recortada por envolvente."""
    phase = 2 * np.pi * freq * T
    if vibrato:
        phase += vibrato * np.sin(2 * np.pi * 5.0 * T)
    sig = sum(a * np.sin((i + 1) * phase) for i, a in enumerate(harmonics))
    return sig * env(start, length, **kw)


def noise(start, length, hp=False, **kw):
    n = rng.standard_normal(N)
    if hp:  # diferencia de primer orden = paso alto barato
        n = np.diff(n, prepend=0.0)
    return n * env(start, length, **kw)


bass = np.zeros(N)
other = np.zeros(N)
drums = np.zeros(N)
vocals = np.zeros(N)

for bar in range(BARS):
    root, triad = PROG[bar % len(PROG)]
    t0 = bar * 4 * BEAT

    # bajo: negras sobre la fundamental, con 2o armonico para que tenga cuerpo
    for b in range(4):
        bass += tone(root, t0 + b * BEAT, BEAT * 0.9, harmonics=(1.0, 0.25),
                     decay=BEAT * 0.5)

    # pad armonico: triada sostenida todo el compas
    for f in triad:
        other += tone(f, t0, 4 * BEAT, harmonics=(1.0, 0.4, 0.2),
                      attack=0.05, decay=4 * BEAT)

    # bateria: bombo en 1 y 3, caja en 2 y 4, charles en corcheas
    for b in (0, 2):
        drums += tone(58.0, t0 + b * BEAT, 0.25, decay=0.06) * 1.4
    for b in (1, 3):
        drums += noise(t0 + b * BEAT, 0.18, decay=0.05) * 0.8
    for b in range(8):
        drums += noise(t0 + b * BEAT / 2, 0.05, hp=True, decay=0.015) * 0.35

    # melodia: corcheas de la escala, con vibrato para que parezca voz
    for b in range(8):
        f = SCALE[(bar * 3 + b * 2) % len(SCALE)]
        vocals += tone(f, t0 + b * BEAT / 2, BEAT * 0.45,
                       harmonics=(1.0, 0.35, 0.15), vibrato=0.03,
                       attack=0.02, decay=BEAT * 0.3)

stems = {"bass": bass, "drums": drums, "other": other, "vocals": vocals}

# normalizar cada stem a un nivel razonable, luego el mix a -3 dBFS.
# Importante: se escala el mix, NO cada stem por separado, para que la suma
# de los stems de referencia siga siendo exactamente el mix.
for k in stems:
    stems[k] /= np.abs(stems[k]).max()
levels = {"bass": 0.55, "drums": 0.60, "other": 0.40, "vocals": 0.50}
stems = {k: v * levels[k] for k, v in stems.items()}

mix = sum(stems.values())
g = 10 ** (-3 / 20) / np.abs(mix).max()
mix *= g
stems = {k: v * g for k, v in stems.items()}

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
sf.write(out / "mix.wav", mix.astype(np.float32), SR, subtype="FLOAT")
for k, v in stems.items():
    sf.write(out / f"ref_{k}.wav", v.astype(np.float32), SR, subtype="FLOAT")

residual = 20 * np.log10(np.abs(mix - sum(stems.values())).max() + 1e-12)
print(f"escrito en {out}")
print(f"  {BARS * 4 * BEAT:.1f}s  {BPM:g} BPM  A minor")
print(f"  null test de referencia: {residual:.1f} dB (debe ser < -100)")
