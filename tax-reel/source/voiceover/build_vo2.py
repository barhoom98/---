"""Best-of-N voice takes: keep the previous schedule (starts/durations -> video stays in sync), regenerate each phrase
N times, keep the clearest take (Whisper CER) that fits its slot, then rebuild the processed voice track."""
import json, os, wave, re, unicodedata
import numpy as np, sherpa_onnx
from scipy.signal import resample_poly
from pedalboard import Pedalboard, HighpassFilter, PeakFilter, HighShelfFilter, Compressor, Reverb
B = os.path.dirname(os.path.abspath(__file__))
M, A = f'{B}/model/vits-piper-ar_JO-kareem-medium', f'{B}/asr/sherpa-onnx-whisper-small'
SR, SPEED, NTAKES, MINGAP = 48000, 1.2, 8, 0.07
items = json.load(open(f'{B}/vo/vo_items.json')); warp = json.load(open(f'{B}/vo/warp.json'))
best = json.load(open(f'{B}/best.json'))
OVR = {'2b': 'n67', '3e': 'n45', '6b': 'n45', '6c': 'n67'}
NOISE = {'n67': (0.667, 0.8), 'n45': (0.45, 0.7)}
E = {k: sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
    model=f'{M}/ar_JO-kareem-medium.onnx', tokens=f'{M}/tokens.txt', data_dir=f'{M}/espeak-ng-data', noise_scale=a, noise_scale_w=b), num_threads=4))) for k, (a, b) in NOISE.items()}
asr = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=f'{A}/small-encoder.int8.onnx', decoder=f'{A}/small-decoder.int8.onnx', tokens=f'{A}/small-tokens.txt', language='ar', task='transcribe', num_threads=4)
def norm(s):
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = re.sub('[أإآ]', 'ا', s).replace('ة', 'ه').replace('ى', 'ي').replace('ث', 'س')
    return re.sub(r'[^ء-ي]', '', s)
def cer(r, h):
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1): d[j] = min(prev[j] + 1, d[j-1] + 1, prev[j-1] + (r[i-1] != h[j-1]))
    return d[-1] / max(1, len(r))
def hear(x):
    p = np.zeros(int(.4 * SR)); st = asr.create_stream(); st.accept_waveform(SR, np.concatenate([p, x, p]).astype(np.float32)); asr.decode_stream(st); return st.result.text.strip()
def synth(text, nk, speed=SPEED):
    a = E[nk].generate(text, sid=0, speed=speed); x = resample_poly(np.array(a.samples, np.float64), 320, 147)
    thr = 0.012 * np.abs(x).max(); nz = np.where(np.abs(x) > thr)[0]
    x = x[max(0, nz[0] - int(.03 * SR)): nz[-1] + int(.15 * SR)]
    f = int(.008 * SR); x[:f] *= np.linspace(0, 1, f); x[-int(.05*SR):] *= np.linspace(1, 0, int(.05*SR)); return x
def split3(x):
    hop = int(.01 * SR); e = np.array([np.sqrt(np.mean(x[i:i+hop]**2)) for i in range(0, len(x) - hop, hop)])
    quiet = e < 0.04 * e.max(); runs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]: j += 1
            if i > 5 and j < len(quiet) - 5: runs.append((j - i, i, j))
            i = j
        else: i += 1
    if len(runs) < 2: return None
    cuts = sorted(sorted(runs, reverse=True)[:2], key=lambda r: r[1]); pts = [0] + [((a + b) // 2) * hop for _, a, b in cuts] + [len(x)]
    parts = []
    for a, b in zip(pts[:-1], pts[1:]):
        p = x[a:b]; nz = np.where(np.abs(p) > 0.012 * np.abs(x).max())[0]
        p = p[max(0, nz[0] - int(.02*SR)): nz[-1] + int(.12*SR)].copy(); p[-int(.03*SR):] *= np.linspace(1, 0, int(.03*SR)); parts.append(p)
    return parts
chosen = {}
groups = [['3b', '3c', '3d']] + [[it['id']] for it in items if it['id'] not in ('3b', '3c', '3d')]
by_id = {it['id']: it for it in items}
order = [it['id'] for it in items]
room = {}
for k, it in enumerate(items):
    nxt = items[k + 1]['start'] if k + 1 < len(items) else warp['duration'] - 0.8
    room[it['id']] = nxt - it['start'] - MINGAP
for g in groups:
    it0 = by_id[g[0]]; text = it0['text']; key = '3bcd' if len(g) == 3 else g[0]
    nk = OVR.get(key, best[key]['noise']); ref = norm(text); cands = []
    for sp in (SPEED, 1.26, 1.32):
      if cands: break
      for _ in range(NTAKES):
        x = synth(text, nk, sp)
        if len(g) == 3:
            parts = split3(x)
            if parts is None or any(len(p) / SR > room[k] for p, k in zip(parts, g)): continue
        elif len(x) / SR > room[g[0]]: continue
        h = hear(x); cands.append((cer(ref, norm(h)), h, x, parts if len(g) == 3 else [x], sp))
    assert cands, f'no take fits for {key} (room {room[g[0]]:.2f}s)'
    cands.sort(key=lambda c: c[0]); c = cands[0]
    for p, k in zip(c[3], g): chosen[k] = p
    print(f"{key:>4} speed {c[4]:.2f} takes {len(cands)}  best CER {c[0]:.2f} (worst {cands[-1][0]:.2f}) | {c[1]}", flush=True)
dur = warp['duration']; vo = np.zeros(int((dur + 1) * SR))
for it in items:
    x = chosen[it['id']]; i = int(round(it['start'] * SR)); vo[i:i + len(x)] += x / (np.abs(x).max() + 1e-9)
chain = Pedalboard([HighpassFilter(85), PeakFilter(260, -2.5, 0.9), PeakFilter(3300, 3.0, 1.0), HighShelfFilter(9000, 1.5),
                    Compressor(threshold_db=-20, ratio=3.2, attack_ms=4, release_ms=90), Reverb(room_size=0.18, damping=0.7, wet_level=0.06, dry_level=1.0)])
vo = chain(vo[None, :].astype(np.float32), SR)[0].astype(np.float64)[:int(dur * SR)]; vo = 0.9 * vo / np.abs(vo).max()
with wave.open(f'{B}/vo/vo.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((vo * 32767).astype(np.int16).tobytes())
print('rebuilt vo.wav', round(len(vo) / SR, 2), 's')
