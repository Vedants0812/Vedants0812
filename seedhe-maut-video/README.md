# claude × seedhe maut

A terminal/HUD remix of a Seedhe Maut (Azadi Records) music video, styled after the
"Claude loves Seedhe Maut" reference clip.

| file | what |
|---|---|
| `claude_x_seedhe_maut_16x9.mp4` | 1280×720, 30 fps, 64 s: the screen render |
| `claude_x_seedhe_maut_9x16.mp4` | 1080×1920 reel cut with the "Claude loves Seedhe Maut 🐉" caption |
| `render.py` | the renderer (numpy + OpenCV + Pillow + ffmpeg, beat analysis with librosa) |

## Timeline (synced to the track's real beat grid, 161 BPM)

| time | scene |
|---|---|
| 0:00 | terminal boot: `claude: open ~/seedhe_maut/`, `encore abj × calm · azadi records` |
| 0:03 | glowing **म** glyph on a red scanline |
| 0:06 | `ls ~/seedhe_maut/ --sort hype`: 2 Ka Pahada → Bayaan → न → Nayaab → Lunch Break → `azadi.mp4` overflows at 999% |
| 0:11 | `play azadi.mp4 --from 0:12 --mode chaos`: glitch tear into the drop |
| 0:12–0:48 | MV through 12 render modes, switching every 2 bars: raw, glyph_न, sobel, neon, halftone, bayer_8x8, thermal, mirror, glitch/stutter, echo trails, glyph_म, blood. The HUD shows the live onset counter, BPM, an audio scope and the timecode, plus terminal jokes (`sudo rm -rf ~/fake_rappers`, `git commit -m 'seedhe maut never miss'` …) |
| 0:48 | `spell 'seedhe maut'`: **स से सीधे · म से मौत**, letters filled with the footage |
| 0:54 | `render crowd --mode glyph_म` + neon stick figures bouncing on the beat |
| 0:57 | **SEEDHE MAUT.** / तेरा भाई सीधे मौत |
| 1:00 | `for @seedhemaut · स × म × claude` |

## Re-render

```bash
pip install opencv-python-headless librosa scipy pillow numpy
# fonts: JetBrains Mono, Inter Display, Noto Sans Devanagari (apt: fonts-noto-core)
mkdir work
ffmpeg -i source.mp4 -vn -ac 1 -ar 22050 work/audio.wav
ffmpeg -i source.mp4 -vn -c:a pcm_s16le -ar 44100 work/audio_st.wav
python3 render.py source.mp4 out.mp4 work                  # full render (~2 min on 4 cores)
python3 render.py source.mp4 previews/ work --preview 3,14,50   # stills
```

Footage and music © Seedhe Maut / Azadi Records. Fan edit.
