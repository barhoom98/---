"""Final voiceover: synthesize chosen variants, split the badge list, compute the video time-warp so every
phrase lands on its visual cue, and render the processed voice track (48 kHz).  Usage: build_vo.py outdir"""
import json, os, sys, wave
import numpy as np, sherpa_onnx
from scipy.signal import resample_poly
from pedalboard import Pedalboard, HighpassFilter, PeakFilter, HighShelfFilter, Compressor, Reverb
B = os.path.dirname(os.path.abspath(__file__))
M = f'{B}/model/vits-piper-ar_JO-kareem-medium'
out = sys.argv[1]; SPEED, GAP, SR = 1.2, 0.14, 48000
segs = json.load(open(f'{B}/segments.json')); best = json.load(open(f'{B}/best.json'))
OVR = {'2b': ('بَسْ، مُمْكِنْ تِنْرِفِضْ!', 'n67'), '3e': ('التَّوْسِيقِ الصَّحّ، بِيِحْمِي مَشْرُوعَكْ مِنِ الْغَرَامَاتْ.', 'n45'),
       '6b': ('وِمْوَسَّقْ بِفَاتُورَة.', 'n45'), '6c': ('أَمَّا الشَّخْصِي، مَا بِيِنْخِصِمْ.', 'n67')}
NOISE = {'n67': (0.667, 0.8), 'n45': (0.45, 0.7)}
E = {k: sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
    model=f'{M}/ar_JO-kareem-medium.onnx', tokens=f'{M}/tokens.txt', data_dir=f'{M}/espeak-ng-data', noise_scale=a, noise_scale_w=b), num_threads=4))) for k, (a, b) in NOISE.items()}

def synth(text, nk):
    a = E[nk].generate(text, sid=0, speed=SPEED); x = np.array(a.samples, np.float64)
    x = resample_poly(x, 320, 147)  # 22050 -> 48000
    thr = 0.012 * np.abs(x).max(); nz = np.where(np.abs(x) > thr)[0]
    x = x[max(0, nz[0] - int(.03 * SR)): nz[-1] + int(.15 * SR)]
    f = int(.008 * SR); x[:f] *= np.linspace(0, 1, f); x[-int(.05*SR):] *= np.linspace(1, 0, int(.05*SR))
    return x

def split3(x):
    """split a 3-word list clip at its two longest internal pauses"""
    hop = int(.01 * SR); e = np.array([np.sqrt(np.mean(x[i:i+hop]**2)) for i in range(0, len(x) - hop, hop)])
    quiet = e < 0.04 * e.max(); runs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]: j += 1
            if i > 5 and j < len(quiet) - 5: runs.append((j - i, i, j))
            i = j
        else: i += 1
    cuts = sorted(sorted(runs, reverse=True)[:2], key=lambda r: r[1])
    pts = [0] + [((a + b) // 2) * hop for _, a, b in cuts] + [len(x)]
    parts = []
    for a, b in zip(pts[:-1], pts[1:]):
        p = x[a:b]; nz = np.where(np.abs(p) > 0.012 * np.abs(x).max())[0]
        p = p[max(0, nz[0] - int(.02*SR)): nz[-1] + int(.12*SR)].copy()
        p[-int(.03*SR):] *= np.linspace(1, 0, int(.03*SR)); parts.append(p)
    return parts, [round(r[0] * .01, 2) for r in cuts]

items = []
for s in segs:
    text, nk = OVR.get(s['id'], (best[s['id']]['text'], best[s['id']]['noise']))
    x = synth(text, nk)
    if 'sub' in s:
        parts, gaps = split3(x); print('3bcd pauses (s):', gaps, 'parts:', [round(len(p)/SR, 2) for p in parts])
        for k, (p, a, w) in enumerate(zip(parts, s['sub'], [s['w']] + s['subw'][1:])):
            items.append({'id': f"3{'bcd'[k]}", 'a': a, 'w': w, 'x': p, 'text': text})
    else:
        items.append({'id': s['id'], 'a': s['a'], 'w': s['w'], 'x': x, 'text': text})

# ---- time-warp (new <- old): slow-motion windows wherever the voice needs more room
knots, S, prev_end, holds = [[0.0, 0.0]], 0.0, -1e9, []
for it in items:
    dur = len(it['x']) / SR; want = it['a'] + S
    if want < prev_end + GAP:
        d = prev_end + GAP - want; w0, w1 = it['w']
        assert w0 >= knots[-1][1] - 1e-9 and w1 <= it['a'] + 1e-9, (it['id'], w0, w1, knots[-1])
        knots.append([w0 + S, w0]); S += d; knots.append([w1 + S, w1]); holds.append((it['id'], round(d, 2)))
        want = it['a'] + S
    it['v'] = want; prev_end = want + dur
end_new = 45.0 + S
if prev_end + 1.4 > end_new:  # let the outro breathe after the last line
    d = prev_end + 1.4 - end_new; knots.append([44.0 + S, 44.0]); S += d; knots.append([45.0 + S, 45.0]); end_new = 45.0 + S
end_new = round(end_new, 3)
print('holds:', holds); print(f'new duration {end_new:.2f}s (extra {S:.2f}s)')

# ---- voice track
vo = np.zeros(int((end_new + 1) * SR))
for it in items:
    i = int(round(it['v'] * SR)); vo[i:i + len(it['x'])] += it['x'] / (np.abs(it['x']).max() + 1e-9)
chain = Pedalboard([HighpassFilter(85), PeakFilter(260, -2.5, 0.9), PeakFilter(3300, 3.0, 1.0), HighShelfFilter(9000, 1.5),
                    Compressor(threshold_db=-20, ratio=3.2, attack_ms=4, release_ms=90), Reverb(room_size=0.18, damping=0.7, wet_level=0.06, dry_level=1.0)])
vo = chain(vo[None, :].astype(np.float32), SR)[0].astype(np.float64)
vo = vo[:int(end_new * SR)]; vo = 0.9 * vo / np.abs(vo).max()
with wave.open(f'{out}/vo.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((vo * 32767).astype(np.int16).tobytes())
json.dump({'knots': knots, 'duration': end_new}, open(f'{out}/warp.json', 'w'), indent=1)
json.dump([{'id': it['id'], 'start': round(it['v'], 3), 'dur': round(len(it['x']) / SR, 3), 'text': it['text']} for it in items],
          open(f'{out}/vo_items.json', 'w'), ensure_ascii=False, indent=1)
for it in items: print(f"{it['id']:>4} at {it['v']:6.2f}s ({len(it['x'])/SR:4.2f}s)  [visual cue old {it['a']:.2f}]")
