// Usage: node render.mjs <out> [mode]
//   mode "frames:t1,t2,..." -> PNG stills at those times
//   mode "video:start:end"  -> pipes JPEG frames [start,end) to ffmpeg as an H.264 segment
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { writeFileSync, readFileSync } from 'node:fs';
import path from 'node:path';
const [, , out, mode = 'frames:0'] = process.argv;
const FPS = 30;
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--force-color-profile=srgb', '--font-render-hinting=none'] }).catch(() => chromium.launch());
const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
page.on('console', m => { if (m.type() === 'error') console.error('[page]', m.text()); });
page.on('pageerror', e => console.error('[pageerror]', e.message));
await page.goto('file://' + path.resolve('index.html'));
await page.waitForFunction(() => window.__ready === true, null, { timeout: 30000 });
if (process.env.WARP_FILE) { const w = JSON.parse(readFileSync(process.env.WARP_FILE, 'utf8')); await page.evaluate(k => window.setWarp(k), w.knots); }
if (mode.startsWith('cues')) {
  writeFileSync(out, JSON.stringify(await page.evaluate(() => window.__cues), null, 0));
} else if (mode.startsWith('frames:')) {
  const ts = mode.slice(7).split(',').map(Number);
  for (const t of ts) {
    await page.evaluate(t => window.seek(t), t);
    await page.screenshot({ path: `${out}/f_${t.toFixed(2)}.png`, type: 'png' });
  }
} else if (mode.startsWith('video:')) {
  const [a, b] = mode.slice(6).split(':').map(Number);
  const f0 = Math.round(a * FPS), f1 = Math.round(b * FPS);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-tune', 'film', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  for (let f = f0; f < f1; f++) {
    await page.evaluate(t => window.seek(t), f / FPS);
    const buf = await page.screenshot({ type: 'jpeg', quality: 96 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if ((f - f0) % 60 === 0) console.error(`${out}: frame ${f - f0}/${f1 - f0} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
}
await browser.close();
