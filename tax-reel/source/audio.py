"""Original soundtrack for the tax reel: synthesized music bed + SFX placed on the
animation cues exported from index.html (cues.json). Usage: python audio.py cues.json out.wav"""
import json
import sys

import numpy as np
from pedalboard import Compressor, HighpassFilter, Limiter, LowpassFilter, Pedalboard, Reverb
from scipy.signal import istft, stft

SR = 48000
DUR = 45.0
N = int(SR * (DUR + 1.5))
rng = np.random.default_rng(2026)


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def place(bus, sig, t, gain=1.0, pan=0.0):
    """Mix mono (equal-power panned) or stereo `sig` into `bus` at time t."""
    if sig.ndim == 1:
        a = (pan + 1) * np.pi / 4
        sig = np.vstack([sig * np.cos(a), sig * np.sin(a)]) * np.sqrt(2)
    i = int(round(t * SR))
    if i < 0:
        sig, i = sig[:, -i:], 0
    end = min(bus.shape[1], i + sig.shape[1])
    if end > i:
        bus[:, i:end] += gain * sig[:, : end - i]


def fx(sig, board):
    st = sig if sig.ndim == 2 else np.vstack([sig, sig])
    return board(st.astype(np.float32), SR).astype(np.float64)


def norm(x):
    return x / (np.max(np.abs(x)) + 1e-9)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------------------------------------------------------------- instruments
def ep_note(f, dur, vel=1.0):
    t = tt(dur)
    idx = 2.0 * np.exp(-t / 0.3) + 0.2
    car = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t))
    tine = 0.12 * np.sin(2 * np.pi * f * 7.0 * t) * np.exp(-t / 0.04)
    amp = (1 - np.exp(-t / 0.003)) * np.exp(-t / (1.3 if f < 300 else 0.95))
    sig = (car + tine) * amp * (1 + 0.12 * np.sin(2 * np.pi * 4.2 * t))
    r = int(0.08 * SR)
    sig[-r:] *= np.linspace(1, 0, r)
    return sig * vel


def pad_note(f, dur, att=0.5, rel=0.7):
    t = tt(dur + rel)
    sig = np.zeros_like(t)
    for det in (-0.09, 0.0, 0.08):
        ff = f * 2 ** (det / 12)
        ph = rng.random() * 2 * np.pi
        for h in range(1, int(min(14, 6000 / ff)) + 1):
            sig += np.sin(2 * np.pi * ff * h * t + ph * h) / h * 0.82 ** h
    env = np.minimum(1, t / att) * np.where(t > dur, np.exp(-(t - dur) / (rel / 3)), 1.0)
    return sig * env / 6


def kick(vel=1.0):
    t = tt(0.45)
    f = 48 + 95 * np.exp(-t / 0.04)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    click = rng.standard_normal(len(t)) * np.exp(-t / 0.002) * 0.25
    return np.tanh(1.6 * (body + click)) * vel


def clap(vel=1.0):
    t = tt(0.35)
    n = rng.standard_normal(len(t))
    env = np.zeros_like(t)
    for d in (0, 0.011, 0.022):
        env += np.where(t >= d, np.exp(-(t - d) / (0.008 if d < 0.02 else 0.12)), 0)
    body = 0.35 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    return (n * env + body) * vel


def hat(vel=1.0, open_=False):
    t = tt(0.4 if open_ else 0.08)
    return rng.standard_normal(len(t)) * np.exp(-t / (0.12 if open_ else 0.018)) * vel


def bass_note(f, dur):
    t = tt(dur)
    sig = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
    env = (1 - np.exp(-t / 0.006)) * np.exp(-t / 0.9)
    r = int(0.05 * SR)
    sig = np.tanh(1.4 * sig) * env
    sig[-r:] *= np.linspace(1, 0, r)
    return sig


def crash():
    t = tt(2.2)
    return rng.standard_normal(len(t)) * np.exp(-t / 0.55)


# ---------------------------------------------------------------- SFX
def sweep_noise(dur, f0, f1, f2, peak=0.6, bw=0.55, shape=2.0):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    fr, ts, Z = stft(x, fs=SR, nperseg=1024, noverlap=768)
    p = np.clip(ts / dur, 0, 1)
    lf = np.where(p < peak, np.log2(f0) + (np.log2(f1) - np.log2(f0)) * (p / peak),
                  np.log2(f1) + (np.log2(f2) - np.log2(f1)) * ((p - peak) / (1 - peak)))
    g = np.exp(-0.5 * ((np.log2(np.maximum(fr, 20))[:, None] - lf[None, :]) / bw) ** 2)
    _, y = istft(Z * g, fs=SR, nperseg=1024, noverlap=768)
    y = norm(y[:n])
    q = np.arange(n) / n
    env = np.where(q < peak, (q / peak) ** shape, np.exp(-(q - peak) / (1 - peak) * 4.5))
    return y * env


def panned(sig, p0, p1):
    q = np.linspace(0, 1, len(sig))
    a = ((p0 + (p1 - p0) * q) + 1) * np.pi / 4
    return np.vstack([sig * np.cos(a), sig * np.sin(a)]) * np.sqrt(2)


def bell(f, dur=1.2, bright=1.0):
    t = tt(dur)
    parts = [(1, 1, 0.9), (2.0, .45, .5), (3.01, .25 * bright, .25), (4.2, .18 * bright, .14), (5.4, .1 * bright, .08)]
    sig = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d) for r, a, d in parts)
    return sig * (1 - np.exp(-t / 0.002))


def sub_hit(f0=90, f1=40, dur=1.3, tau=0.45):
    t = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-t / 0.12)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / tau)


def blip(f0, f1, dur=0.08):
    t = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-t / (dur / 4))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (dur / 3)) * (1 - np.exp(-t / 0.001))


def click_snd():
    t = tt(0.05)
    a = rng.standard_normal(len(t)) * np.exp(-t / 0.0015)
    b = np.zeros_like(t)
    i = int(0.045 * SR * 0.2)
    b[i:] = rng.standard_normal(len(t) - i) * np.exp(-(t[i:] - t[i]) / 0.0012) * 0.6
    return (a + b + 0.4 * np.sin(2 * np.pi * 2600 * t) * np.exp(-t / 0.004))


def tick_snd():
    t = tt(0.035)
    f = 1800 + rng.random() * 1600
    return (rng.standard_normal(len(t)) * 0.6 + np.sin(2 * np.pi * f * t)) * np.exp(-t / 0.006)


def wood(f=1500):
    t = tt(0.06)
    return np.sin(2 * np.pi * f * t) * np.exp(-t / 0.012)


def buzz_snd():
    t = tt(0.5)
    tone = np.sign(np.sin(2 * np.pi * 150 * t)) * 0.5 + np.sin(2 * np.pi * 150 * t)
    gate = ((t % 0.17) < 0.1).astype(float) * (t < 0.45)
    return tone * gate * 0.5


# ---------------------------------------------------------------- music
BPM = 96
BEAT = 60 / BPM
BAR = 4 * BEAT
CH = [  # (EP voicing, bass root)
    ([62, 65, 69, 72, 76], 38),   # Dm9
    ([58, 65, 69, 72, 74], 34),   # Bbmaj9
    ([53, 60, 64, 67, 69], 41),   # Fmaj9
    ([60, 67, 74, 76], 36),       # Cadd9
]


def build_music():
    keys = np.zeros((2, N))
    pad = np.zeros((2, N))
    drums = np.zeros((2, N))
    bass = np.zeros((2, N))
    duck = np.ones(N)
    kicks = []

    def bar_block(t0, k, full=True, last=False):
        notes, root = CH[k % 4]
        # pad
        for j, m in enumerate(notes[:4]):
            place(pad, pad_note(midi(m - 12 if j == 0 else m), BAR * (1.6 if last else 1.0)), t0, 0.5, pan=-0.5 + j / 3)
        # electric piano: stab on 1 and on the "and" of 2
        for hit_t, vel in ((0.0, 1.0), (1.5 * BEAT, 0.7)) if not last else ((0.0, 1.0),):
            for j, m in enumerate(notes):
                place(keys, ep_note(midi(m), 2.6 if last else 1.6, vel * (0.9 - 0.08 * j)), t0 + hit_t + 0.006 * j, 0.32, pan=-0.35 + 0.7 * j / max(1, len(notes) - 1))
        if not full:
            return
        # bass
        for bt, ln, off in ((0, 1.4, 0), (1.5, 0.45, 0), (3.0, 0.8, 7)):
            place(bass, bass_note(midi(root + off), ln * BEAT), t0 + bt * BEAT, 0.55)
        # drums
        for b in (0, 2, 2.5):
            if b == 2.5 and k % 2 == 0:
                continue
            place(drums, kick(1.0 if b != 2.5 else 0.7), t0 + b * BEAT, 0.9)
            kicks.append(t0 + b * BEAT)
        for b in (1, 3):
            place(drums, clap(), t0 + b * BEAT, 0.32, pan=0.05)
        for e in range(8):
            open_ = (e == 7 and k % 4 == 3)
            place(drums, hat(0.9 if e % 2 else 0.55, open_), t0 + e * BEAT / 2 + (0.012 if e % 2 else 0), 0.16 if not open_ else 0.11, pan=0.25)

    # section 1: bars on a grid offset so the downbeat lands on the dark-screen cut (37.6)
    off = 0.1
    for k in range(15):
        bar_block(off + k * BAR, k, full=k > 0)
    # breakdown 37.6 – 40.85: filtered pad + heartbeat kicks
    bd = np.zeros((2, N))
    for j, m in enumerate(CH[0][0][:4]):
        place(bd, pad_note(midi(m - 12), 3.0, att=0.15, rel=0.6), 37.6, 0.38, pan=-0.4 + j / 4)
    bd = fx(bd, Pedalboard([LowpassFilter(cutoff_frequency_hz=520)]))
    pad += bd
    for t in (38.85, 39.48, 40.1, 40.42):
        place(drums, kick(0.55), t, 0.6)
    # outro from 40.85: groove + final ring-out chord
    place(drums, crash(), 40.85, 0.12, pan=-0.2)
    bar_block(40.85, 0, full=True)
    bar_block(40.85 + BAR, 1, full=False, last=True)

    for k in kicks:
        i = int(k * SR)
        seg = np.arange(N - i) / SR
        duck[i:] = np.minimum(duck[i:], 1 - 0.42 * np.exp(-seg / 0.16))

    keys = fx(keys, Pedalboard([HighpassFilter(140), LowpassFilter(5200), Reverb(room_size=0.55, damping=0.5, wet_level=0.22, dry_level=0.85, width=1.0)]))
    pad = fx(pad, Pedalboard([HighpassFilter(180), LowpassFilter(2600), Reverb(room_size=0.8, damping=0.6, wet_level=0.35, dry_level=0.7, width=1.0)]))
    drums = fx(drums, Pedalboard([HighpassFilter(30), Compressor(threshold_db=-14, ratio=3, attack_ms=8, release_ms=90)]))
    bass = fx(bass, Pedalboard([LowpassFilter(900)]))
    music = (keys + pad) * duck + drums + bass * duck ** 0.5
    return music


# ---------------------------------------------------------------- SFX from cues
def build_sfx(cues):
    sfx = np.zeros((2, N))
    verb = np.zeros((2, N))
    dings = [midi(81), midi(84), midi(86), midi(89)]
    di = 0
    for c in cues:
        t, k = c["t"], c["type"]
        if k == "start":
            place(sfx, sweep_noise(0.36, 400, 3000, 3000, peak=0.98, shape=3), t, 0.25)
        elif k == "impact":
            place(sfx, sub_hit(95, 42, 1.4, 0.5), t, 0.95)
            place(verb, norm(sweep_noise(0.5, 900, 300, 120, peak=0.03, shape=1)), t, 0.35)
        elif k == "impactSoft":
            place(sfx, sub_hit(85, 45, 0.9, 0.3), t, 0.5)
            place(verb, bell(midi(74), 1.2, 0.6), t, 0.10)
        elif k == "hit":
            place(sfx, kick(0.7), t, 0.45)
            place(verb, bell(midi(86), 1.0, 1.0), t, 0.08, pan=0.2)
        elif k == "shimmer":
            for j, m in enumerate((81, 86, 93, 98)):
                place(verb, bell(midi(m), 1.4, 0.7), t + 0.05 * j, 0.06, pan=-0.4 + 0.27 * j)
        elif k == "whoosh":
            place(sfx, panned(sweep_noise(0.55, 350, 2600, 700, peak=0.55), -0.6, 0.6), t - 0.12, 0.42)
        elif k == "whooshBig":
            w = sweep_noise(0.75, 220, 2200, 400, peak=0.5, bw=0.7)
            place(sfx, panned(w, -0.8, 0.8), t - 0.18, 0.6)
            place(sfx, sub_hit(70, 38, 0.8, 0.25), t + 0.2, 0.35)
        elif k == "whooshIn":
            place(sfx, panned(sweep_noise(0.6, 300, 3200, 3200, peak=0.92, shape=2.6), 0.0, 0.0), t, 0.45)
        elif k == "swish":
            place(sfx, panned(sweep_noise(0.32, 900, 4200, 1600, peak=0.5, bw=0.45), 0.5, -0.5), t - 0.06, 0.22)
        elif k == "swishSoft":
            place(sfx, panned(sweep_noise(0.4, 700, 2500, 1200, peak=0.5, bw=0.45), -0.3, 0.3), t, 0.12)
        elif k == "pop":
            place(sfx, blip(1150, 420, 0.07), t, 0.3, pan=(rng.random() - 0.5) * 0.6)
        elif k == "ding":
            place(verb, bell(dings[di % len(dings)], 1.5), t, 0.16)
            place(sfx, bell(dings[di % len(dings)], 0.5), t, 0.10)
            di += 1
        elif k == "click":
            place(sfx, click_snd(), t, 0.5)
        elif k == "tick":
            place(sfx, tick_snd(), t, 0.10 + rng.random() * 0.05, pan=(rng.random() - 0.5) * 0.3)
        elif k == "count":
            place(sfx, wood(1450 + rng.random() * 120), t, 0.10)
        elif k == "thud":
            place(sfx, sub_hit(75, 48, 0.6, 0.18), t, 0.7)
            place(sfx, blip(240, 170, 0.25), t, 0.18)
        elif k == "drop":
            place(sfx, sub_hit(110, 28, 2.0, 0.6), t, 0.75)
            place(verb, norm(sweep_noise(1.0, 2400, 300, 120, peak=0.02, shape=1)), t, 0.25)
        elif k == "notif":
            place(verb, bell(midi(88), 0.6, 0.4), t, 0.18)
            place(verb, bell(midi(93), 0.9, 0.4), t + 0.13, 0.18)
        elif k == "buzz":
            place(sfx, buzz_snd(), t, 0.16)
        elif k == "final":
            place(sfx, sub_hit(90, 42, 1.6, 0.6), t, 0.7)
            for j, m in enumerate((74, 77, 81, 86, 89)):
                place(verb, bell(midi(m), 2.0, 0.6), t + 0.03 * j, 0.07, pan=-0.4 + 0.2 * j)
    # riser into the outro
    place(sfx, panned(sweep_noise(1.25, 250, 5000, 5000, peak=0.97, shape=2.5), -0.3, 0.3), 39.6, 0.22)
    verb = fx(verb, Pedalboard([Reverb(room_size=0.75, damping=0.4, wet_level=0.45, dry_level=0.8, width=1.0)]))
    return sfx + verb


def main():
    cues = json.load(open(sys.argv[1]))
    music = build_music()
    sfx = build_sfx(cues)
    mix = 0.55 * music / (np.max(np.abs(music)) + 1e-9) + 0.75 * sfx / (np.max(np.abs(sfx)) + 1e-9)
    n = int(DUR * SR)
    mix = mix[:, :n]
    fade = int(0.6 * SR)
    mix[:, -fade:] *= np.linspace(1, 0, fade) ** 1.5
    mix[:, :int(0.01 * SR)] *= np.linspace(0, 1, int(0.01 * SR))
    mix = fx(mix, Pedalboard([HighpassFilter(25), Compressor(threshold_db=-10, ratio=2.5, attack_ms=10, release_ms=120), Limiter(threshold_db=-2.0, release_ms=80)]))
    mix = 0.89 * mix / (np.max(np.abs(mix)) + 1e-9)
    pcm = (np.clip(mix.T, -1, 1) * 32767).astype(np.int16)
    import wave
    with wave.open(sys.argv[2], "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", sys.argv[2], pcm.shape)


if __name__ == "__main__":
    main()
