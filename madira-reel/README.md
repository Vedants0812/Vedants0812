# MADIRA: 60-second lyric reel

**Final cut: `madira_lyric_video_final_9x16.mp4`** (song 0:59.3 to 1:48, about 49 s: the hook with echoes, then Calm's verse, one scene per line). Earlier cut: `madira_lyric_reel_60s_9x16.mp4`. It is 1080 × 1920 at 24 fps, uses the song's audio from 0:59.3 (the hook's second pass) to 2:00.3,
and has 26 scenes, one per sung line, with the lyrics as kinetic Devanagari type. Its theme, "Ink of the Night", uses
midnight indigo, moon-silver, wine and crimson. Line timings follow the song's 12-second blocks, and each word is snapped to
a detected vocal onset.

`render_lyrics.py` re-renders it (about 10 minutes on 4 cores). The lyric text is not stored in this repo; pass it with
`--lyrics LYRICS.txt` (26 lines, in order).

---

# MADIRA: 60-second reel, "Ink of the Night"

**Main cut: `madira_ink_reel_60s_9x16.mp4`.** It is 1080 × 1920 at 24 fps and uses the song's audio from **0:48.1, where the vocals
enter, to 1:48.1**, the silence at the hand-off. Nothing before 0:48 is used.

Its theme is our own, not the reference's: there are no clocks, gears or gold. It is ink smoke under a moon, in midnight indigo,
moon-silver, wine and crimson. The ink is a small fluid simulation keyed to the song's beats, vocal onsets and bass hits.

| Song time | What happens |
|---|---|
| 0:48.1 | The vocals enter and a drop of wine ink blooms over the moon; मदिरा forms out of ink, then dissolves (the bass is still out) |
| 0:58.5 | The bass enters. Two ink silhouettes face each other, moon-silver and wine; their smoke reaches across and mixes on every downbeat while petals fall |
| 1:10.5 | The pause: everything holds, with नशा |
| 1:11.9 | The ink spirals into a vortex |
| 1:15.0 | Calm's verse bursts in: an ink splash on every strong vocal hit, and shockwaves from the moon on bass hits |
| 1:21.0 | The 3-second break: a total freeze, with ठहरो |
| 1:24.0 | A crimson rose sheds a petal on every downbeat, each petal bleeding into ink, while the moon slowly turns blood-red |
| 1:46.1 | The hand-off silence: a freeze and the end mark |

`render_ink.py` re-renders it (about 8 minutes on 4 cores; `--preview 52,64,92` renders stills).
The on-screen words (मदिरा, नशा, ठहरो) are thematic words, not lyrics. Real lyric lines can be added once the lyrics are supplied.

---

## Earlier cut (archive theme, 0:22–1:22.5)

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
