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
    """Tempo en dos pasos: elegir el nivel metrico y despues afinarlo.

    El fallo dominante no es fallar el pulso, es acertarlo y contarlo al
    doble, a la mitad o a dos tercios. librosa propone un candidato; aqui se
    generan sus octavas y variantes de tresillo y se elige la que mas energia
    tiene de verdad en la cancion.
    """
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    raw = librosa.beat.beat_track(onset_envelope=onset, sr=sr,
                                  hop_length=HOP)[0]
    raw = float(np.atleast_1d(raw)[0])

    # Tempograma: autocorrelacion de la envolvente ventana a ventana, o sea
    # cuanta energia hay EN cada periodo de pulso.
    #
    # Antes se usaba la autocorrelacion global medida en el periodo y en sus
    # multiplos (x1, x2, x4), y eso premia sistematicamente a los tempos
    # lentos: los multiplos de un pulso lento caen en el compas y en la frase,
    # que en musica de loop son periodicos de sobra. El candidato lento cobraba
    # una prima que el rapido no tenia como cobrar.
    tg = librosa.feature.tempogram(onset_envelope=onset, sr=sr, hop_length=HOP)
    strength = tg.mean(axis=1)
    lags = np.arange(len(strength))

    def evidence(bpm):
        return float(np.interp(60.0 / bpm * sr / HOP, lags, strength))

    # Prior ancho (una octava de sigma): desempata entre candidatos parejos y
    # nada mas. Con el anterior (0.6) la cercania a 120 pesaba mas que la
    # evidencia, y un trap de 170 salia a 113 -- que es 170*2/3, un pulso que
    # no toca nadie en la cancion.
    def prior(bpm):
        return float(np.exp(-0.5 * (np.log2(bpm / 120.0) / 1.0) ** 2))

    cands = {round(_fold(raw * r), 3)
             for r in (1 / 3, 1 / 2, 2 / 3, 1.0, 3 / 2, 2.0, 3.0)}
    scored = sorted(((evidence(b) * prior(b), b) for b in cands), reverse=True)
    best_score, level = scored[0]

    bpm, offset = level, 0.0
    beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr,
                                    hop_length=HOP, start_bpm=level,
                                    trim=False)[1]
    if len(beats) >= 8:
        # Recta sobre los tiempos de los beats: la pendiente promedia cientos
        # de pulsos, asi que el decimal del BPM significa algo (el nivel solo
        # tiene la resolucion del frame). El corte da la fase, que es lo que
        # necesita el metronomo para no entrar a destiempo.
        t = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
        slope, intercept = np.polyfit(np.arange(len(t)), t, 1)
        # Si la recta se aleja mas de un 5% del nivel elegido no es un
        # afinado, es una cancion que cambia de tempo: mejor el nivel pelado.
        if slope > 0 and abs(60.0 / slope - level) < level * 0.05:
            bpm = 60.0 / slope
            offset = float(intercept % slope)
    return {
        "bpm": round(bpm, 1),
        # Tiempo del primer beat dentro del compas, en segundos. Sin esto el
        # metronomo clava el tempo pero entra a contratiempo.
        "beat_offset": round(offset, 3),
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


def _trap(bpm=170.0, secs=20.0, sr=SR):
    """Patron de trap sintetico: el caso que rompia el detector.

    Bombo y caja marcan la mitad y el charles subdivide. El viejo metodo
    contestaba 113.0 a esto, que es bpm*2/3: un pulso que no toca nadie, ni
    el charles ni la caja. Es el mismo fallo que daba 113 en una cancion real
    de 170.
    """
    n = int(secs * sr)
    y = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    beat = 60.0 / bpm

    def hit(at, length, freq, amp):
        i = int(at * sr)
        k = np.arange(min(int(length * sr), n - i)) / sr
        if len(k) <= 0:
            return
        e = np.exp(-k / (length / 4)) * amp
        wave = (np.sin(2 * np.pi * freq * k) if freq
                else rng.standard_normal(len(k)))
        y[i:i + len(k)] += (wave * e).astype(np.float32)

    for s in range(int(secs / (beat / 2))):        # charles en octavos
        hit(s * beat / 2, 0.025, 9000, 0.35 if s % 2 == 0 else 0.21)
    for b in range(int(secs / beat)):
        if b % 4 == 0:
            hit(b * beat, 0.18, 55, 1.0)           # bombo
        if b % 4 == 2:
            hit(b * beat, 0.12, 0, 0.8)            # caja, el contragolpe
    return y


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

    # Regresion del nivel metrico. Se exige caer en la rejilla, no un numero:
    # 170 y 85 son la misma cuadricula leida a dos velocidades y las dos son
    # defendibles. 113 no lo es, y era justo lo que salia antes.
    trap = estimate_bpm(_trap(170.0))
    print("trap 170 ", json.dumps(trap))
    off = min(abs(trap["bpm"] - x) for x in (85.0, 170.0))
    assert off < 2.0, f"trap: {trap['bpm']} no es 85 ni 170"
    # La fase cae dentro de un pulso: si no, el metronomo entra a destiempo.
    assert 0 <= trap["beat_offset"] < 60.0 / trap["bpm"], trap["beat_offset"]
    print("\ndemo ok: BPM y key correctos contra ground truth")


if __name__ == "__main__":
    if sys.argv[1] == "--demo":
        _demo(sys.argv[2])
    else:
        print(json.dumps(analyze(sys.argv[1]), indent=2))
