# ريل المحاسبة الضريبية

| الملف | الوصف |
|---|---|
| `tax-reel-voice.mp4` | **النسخة مع التعليق الصوتي**: 65 ثانية، صوت عربي أردني بالذكاء الاصطناعي، والموسيقى بتوطى تحت الصوت |
| `tax-reel.mp4` | النسخة بدون صوت: 1080×1920، ‏30 فريم/ث، ‏45 ثانية، H.264 + صوت AAC بمستوى ‎-14 LUFS (مناسب لإنستغرام وتيك توك) |
| `cover.jpg` | صورة غلاف من مشهد الهوك |
| `source/` | ملفات المصدر لو بدك تعدّل وتعيد التصدير |

## المشاهد
1. **الهوك** (0–2.6ث): «مش كل مصروف **بينخصم** ضريبياً!»
2. **المقارنة**: كارت «مقبول ✓» مقابل «مرفوض ✗»
3. **الفكرة الأساسية**: عدسة مكبّرة، وبعدها ٣ شارات: إثبات، فاتورة، موعد
4. **٣ نقاط بتحميك من الغرامات**، وبعدها كروت «؟»
5. **أولاً: الفاتورة الرسمية** (كلمات طايرة)
6. **ثانياً: المصروف المقبول** (زر تشغيل وقبل/بعد)
7. **ثالثاً: المواعيد** (عدّاد وكتابة)
8. **الخلاصة** بالنقاط الذهبية
9. **الشاشة الداكنة**: إشعار آخر موعد للإقرار
10. **الختام**: كورس المحاسبة الضريبية، سؤال التشويق، ودعوة للتعليق

الموسيقى والمؤثرات مولّدة برمجياً بالكامل (`audio.py`)، فما عليها حقوق نشر.

## التعليق الصوتي
- **الصوت:** نموذج Piper مفتوح المصدر ومجاني، صوت «kareem» (عربي أردني)، بشتغل محلياً عبر sherpa-onnx بدون أي خدمة مدفوعة.
- **النص:** السكربت المشكّل باللهجة الأردنية (`source/voiceover/segments.json`)، مقسوم لـ26 جملة قصيرة. كل جملة مربوطة بلحظتها بالفيديو، يعني «إثبات، فاتورة، وموعد» بتنحكى مع ظهور كل شارة.
- **الجودة:** كل جملة انولدت 8 مرات، وانختارت أوضحها بالتعرّف على الكلام (Whisper)، وبعض الكلمات انكتبت متل ما بتنلفظ بالأردني (توسيق، مْوَسَّق، إسبات) عشان تطلع أوضح.
- **التوقيت:** الفيديو صار 65 ثانية لأنه المشاهد بتستنى الصوت بأماكن ثابتة (`warp.json`)، بدل ما نسرّع الصوت لدرجة يصير آلي.
- **`source/voiceover/voice-only.wav`:** الصوت لحاله، إذا بدك تستخدمه بـ CapCut أو تستبدله بتسجيلك.
- **الرخصة:** كود Piper مرخّص MIT، بس رخصة بيانات تدريب صوت kareem مش موضّحة بشكل صريح. للإعلانات المدفوعة، الأسلم تسجّل بصوتك أو تستخدم خدمة برخصة تجارية.

## التعديل وإعادة التصدير
- النصوص والحركة: `source/index.html` (مكتبة GSAP، والتوقيتات بالثواني داخل `build()`).
- الصوت: `source/audio.py` بيقرأ توقيتات المؤثرات من `cues.json`.

```bash
cd source
npm install                      # gsap + playwright
node render.mjs video.mp4 video:0:45      # تصدير الفريمات (Chromium + ffmpeg)
node render.mjs cues.json cues            # توقيتات المؤثرات
pip install numpy scipy pedalboard
python audio.py cues.json audio.wav
ffmpeg -i video.mp4 -i audio.wav -map 0:v -map 1:a -c:v copy -af volume=-1.9dB -c:a aac -b:a 192k -shortest -movflags +faststart tax-reel.mp4
```

نسخة الصوت: نزّل النموذج `vits-piper-ar_JO-kareem-medium` (ومعه `sherpa-onnx-whisper-small` للتقييم) من إصدارات sherpa-onnx على GitHub داخل `source/voiceover/model` و `source/voiceover/asr`، وبعدين:
```bash
cd source/voiceover && pip install sherpa-onnx scipy pedalboard
python build_vo.py vo && python build_vo2.py           # الصوت + خريطة التوقيت
cd .. && WARP_FILE=voiceover/vo/warp.json node render.mjs video.mp4 video:0:65
python audio.py cues.json mix.wav voiceover/vo/warp.json voiceover/vo/vo.wav
```

الخطوط: Thmanyah Sans (من نفس المستودع)، Playfair Display، Great Vibes، IBM Plex Sans Arabic (رخصة OFL).

> تنبيه: المحتوى توعوي عام، والقواعد والمواعيد الضريبية بتختلف من بلد لبلد.
