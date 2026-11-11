# MADIRA: 60-second reel

The first minute of the Seedhe Maut **MADIRA** re-edit (prod. Hurricane, DL91 FM), in the
"Archive of Forgotten Moments" art direction. It is 1080 × 1920 at 24 fps and covers 0:22.0 to 1:22.5 of the song.

| Song time | Shot | What happens |
|---|---|---|
| 0:22.0 | S02 The last tick | The pocket watch reaches 12:48 and stops on the 1-second inhale; a gold seam cracks the crystal |
| 0:23.6 | S03 The dial opens | On the drop the dial splits like doors, the numerals burst out, and मदिरा fills with gold |
| 0:30.8 | S04 The movement | The camera dives through the counter of म into a cathedral of gears; holds on every micro-pause; a gold drop falls |
| 0:48.1 | S05 The first drop | The drop hits as the vocals enter; ripples, a crown splash, a photo rises and इंतज़ार arrives letter by letter |
| 0:58.5 | S06 Floating photographs | Memories drift past the camera, the rose hangs among them, and one photo dissolves into gold every 2 bars |
| 1:10.5 | S07 Cast in gold | The particles converge into मदिरा, freeze through the pause, then slump |
| 1:15.0 | S08 The letter | The seal cracks, the letter unfolds one fold per bar, and ink writes in time with the vocals |
| 1:21.0 | S09 The held breath | Ink droplets freeze mid-air, the light dims, then the end mark |

The letter text and the archive words are placeholders written for this edit, not song lyrics.

`render_reel.py` re-renders it. It needs Python with numpy, opencv, pillow and librosa, plus ffmpeg and the
Google Fonts listed in the script (Rozha One, Tiro Devanagari Hindi, Kalam, Cormorant Garamond, IBM Plex Mono):

```bash
python3 render_reel.py song.wav out.mp4 work/ fonts/                 # about 10 minutes on 4 cores
python3 render_reel.py song.wav previews/ work/ fonts/ --preview 27.5,51,71
```

Music © Seedhe Maut. Fan edit.
