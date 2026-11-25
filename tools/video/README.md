# Demo video sources

The 2-minute product video `reports/Ayas_demo_1080p.mp4` (1920 × 1080, 30 fps, H.264 + AAC) is built from:

| File | Job |
|---|---|
| `blender_shots.py` | Blender 5.2 product shots (hero, exploded view, bursts, UV); `Blender -b -P blender_shots.py -- <shot> <out>` writes JPEG frames |
| `prep_assets.py` | Real pipeline assets for the cards (twin bursts, overlays, height maps) |
| `compose.py` | The compositor: 12 scenes, neural-network animations, captions; pipes frames to ffmpeg |
| `narration.json` | The narration script and scene timing |
| `voice/tts_parler.py` | Indian-English male narration with ai4bharat Indic Parler-TTS on a GPU (Whisper WER picks the best take); `voice/narration_tts.json` has the phonetic spellings |
