"""BPM y key. Dos usos:

  - al ingerir: sobre el mix, rapido, para mostrar algo ya.
  - tras separar: BPM sobre el stem de drums y key sobre bass+other.
    Mismo codigo, entrada mas limpia, resultado bastante mejor.

Self-check: python analyze.py --demo <carpeta_del_fixture>
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np

SR = 44100
HOP = 512
BPM_LO, BPM_HI = 65.0, 190.0

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Perfiles de Temperley (corpus Kostka-Payne), no los de Krumhansl-Kessler.
# Motivo concreto: KK salio de experimentos de probe-tone sobre menor NATURAL y
# penaliza la septima elevada. Pero casi toda la musica en menor usa el V mayor
# con su sensible (en A menor, el G# del acorde de E), asi que KK confunde menor
# con mayor en cuanto aparece una dominante. Temperley sale de musica real y da
# a la sensible 0.330 frente a 0.133 de la septima menor.
#   indice 0 = tonica, 11 = sensible
PROFILE_MAJOR = np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460,
                          0.096, 0.715, 0.104, 0.366, 0.057, 0.400])
PROFILE_MINOR = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460,
                          0.105, 0.747, 0.404, 0.067, 0.133, 0.330])


def _fold(bpm):
    """Lleva un tempo al rango util doblando o partiendo por la mitad."""
    while bpm < BPM_LO:
        bpm *= 2
    while bpm > BPM_HI:
        bpm /= 2
    return bpm


def estimate_bpm(y, sr=SR):
    """Tempo con guarda de octava: el fallo dominante es el doble/mitad.

    librosa da un candidato; se generan sus octavas y variantes de tresillo,
    y se elige la que mejor explica la envolvente de onsets, pesada por un
    prior log-normal centrado en 120 BPM.
    """
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    raw = librosa.beat.beat_track(onset_envelope=onset, sr=sr,
                                  hop_length=HOP)[0]
    raw = float(np.atleast_1d(raw)[0])

    ac = librosa.autocorrelate(onset, max_size=len(onset) // 2)
    ac = ac / (ac[0] + 1e-9)
    lags = np.arange(len(ac))

    def evidence(bpm):
        """Autocorrelacion en el periodo del pulso y sus multiplos."""
        beat_lag = 60.0 / bpm * sr / HOP
        return float(np.mean([np.interp(beat_lag * m, lags, ac)
                              for m in (1, 2, 4)]))

    def prior(bpm):
        return float(np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.6) ** 2))

    cands = {round(_fold(raw * r), 3)
             for r in (1 / 3, 1 / 2, 2 / 3, 1.0, 3 / 2, 2.0, 3.0)}
    scored = sorted(((evidence(b) * prior(b), b) for b in cands), reverse=True)
    best_score, best = scored[0]
    return {
        "bpm": round(best, 1),
        "bpm_raw": round(raw, 1),
        "bpm_confidence": round(best_score, 3),
        "bpm_alternatives": [round(b, 1) for _, b in scored[1:3]],
    }


def estimate_key(y, sr=SR, already_harmonic=False):
    """Krumhansl-Schmuckler sobre chroma. Devuelve top-2, no una respuesta.

    La deteccion de key es el punto debil del estado del arte entero, asi que
    se reporta el margen entre candidatas en vez de fingir certeza.
    """
    if not already_harmonic:
        y = librosa.effects.hpss(y)[0]

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    # Normalizar CADA frame y luego promediar: mide "cuanto suena esta nota"
    # en vez de "cuanto grita". Sin esto, el estribillo mas fuerte decide la
    # tonalidad de la cancion entera.
    chroma = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    profile = chroma.mean(axis=1)
    if profile.sum() < 1e-6:
        return {"key": None, "key_confidence": 0.0, "key_alternatives": []}
    profile = (profile - profile.mean()) / (profile.std() + 1e-9)

    scored = []
    for tonic in range(12):
        for name, ref in (("major", PROFILE_MAJOR), ("minor", PROFILE_MINOR)):
            r = np.roll(ref, tonic)
            r = (r - r.mean()) / r.std()
            scored.append((float(np.dot(profile, r) / 12),
                           f"{PITCHES[tonic]} {name}"))
    scored.sort(reverse=True)
    return {
        "key": scored[0][1],
        "key_confidence": round(scored[0][0], 3),
        "key_margin": round(scored[0][0] - scored[1][0], 3),
        "key_alternatives": [k for _, k in scored[1:3]],
    }


def analyze(path, bpm=True, key=True, already_harmonic=False):
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    out = {"duration": round(len(y) / sr, 2)}
    if bpm:
        out |= estimate_bpm(y, sr)
    if key:
        out |= estimate_key(y, sr, already_harmonic)
    return out


def analyze_stems(stems):
    """Refinado tras separar: drums manda en BPM, bass+other en key."""
    out = {}
    if "drums" in stems:
        y, _ = librosa.load(str(stems["drums"]), sr=SR, mono=True)
        out |= estimate_bpm(y, SR)
    harmonic = [stems[k] for k in ("bass", "other") if k in stems]
    if harmonic:
        ys = [librosa.load(str(p), sr=SR, mono=True)[0] for p in harmonic]
        n = min(len(y) for y in ys)
        out |= estimate_key(sum(y[:n] for y in ys), SR, already_harmonic=True)
    return out


def _demo(fixture):
    """Ground truth del fixture: 120.0 BPM, A minor."""
    d = Path(fixture)
    mix = analyze(d / "mix.wav")
    print("mix      ", json.dumps(mix))
    assert abs(mix["bpm"] - 120.0) < 2.0, f"BPM del mix: {mix['bpm']} != 120"

    # En el mix se tolera la confusion de tonalidad relativa (A menor / C mayor
    # comparten las notas): basta que la correcta este entre las dos primeras.
    top2 = [mix["key"], *mix["key_alternatives"][:1]]
    assert "A minor" in top2, f"key del mix: A minor no esta en {top2}"

    ref = analyze_stems({k: d / f"ref_{k}.wav"
                         for k in ("drums", "bass", "other")})
    print("ref stems", json.dumps(ref))
    assert abs(ref["bpm"] - 120.0) < 2.0, f"BPM de drums: {ref['bpm']} != 120"
    # Sobre bass+other, sin voz ni percusion, si se exige el acierto exacto.
    assert ref["key"] == "A minor", f"key de bass+other: {ref['key']} != A minor"
    print("\ndemo ok: BPM y key correctos contra ground truth")


if __name__ == "__main__":
    if sys.argv[1] == "--demo":
        _demo(sys.argv[2])
    else:
        print(json.dumps(analyze(sys.argv[1]), indent=2))
