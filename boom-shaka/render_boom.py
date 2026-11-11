#!/usr/bin/env python3
"""
KRSNA - BOOM SHAKA : kinetic-type lyric video, 1920x1080 @ 24 fps, "BOOM" style.

Blast orange, bone white, black. Each lyric line is a justified type stack
(every row scaled to the same width, Devanagari + condensed Latin mixed), words
slam in on their measured syllable onsets, the key word burns orange. The
cover art sits opposite the type as a posterized duotone that glitches on the
beat. Backgrounds rotate per line; the chorus explodes.

The lyric text and the line/word timing are supplied at run time:
  python3 render_boom.py SONG.mp3 OUT.mp4 WORKDIR FONTDIR --lyrics L.txt --timing T.json --cover cover.png
  ... --preview 15,30,82          (song times)
"""
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "madira-reel"))
import render_reel as RR  # noqa: E402
from render_reel import over, over_c, affine, draw_sprite, blur, sm, eout, tsprite  # noqa: E402

W, H, FPS = 1920, 1080, 24
T0, T1 = 13.6, 97.2
NF = int(round((T1 - T0) * FPS))


def hexc(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)


BLACK = hexc("#070707")
EMBER = hexc("#3A0D02")
BLAZE = hexc("#FF5A1F")
BONE = hexc("#F2EDE4")
HAZARD = hexc("#FFC21A")
F = {}
AU = {}
LINES = []
KEY = {"goat", "bm'", "m'", "dm", "gm", "tn", "pm", "aura", "दबदबा", "अप्सरा", "rap", "whack", "bitter", "better", "bag",
       "fact", "trigger,", "finger,", "tapper\"", "factor", "masked", "kill", "bars", "राजधानी", "hashmi", "lv", "shelby",
       "boom", "shakalaka", "room", "बंदूक", "तबाह?", "कटघरा", "ज़ुबान,", "ज़हर", "जान-बाज़", "retire", "stove", "blocked,"}


# ------------------------------------------------------------------- data --
def analyse(audio):
    import librosa
    y, sr = librosa.load(audio, sr=22050)
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    _, bf = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
    bt = librosa.frames_to_time(bf, sr=sr)
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    lo = librosa.onset.onset_strength(S=librosa.power_to_db(M[:4]), sr=sr)
    lt = librosa.times_like(lo, sr=sr)
    pk = librosa.util.peak_pick(lo, pre_max=6, post_max=6, pre_avg=20, post_avg=20, delta=lo.std() * 1.3, wait=8)
    AU.update(beats=bt, bass=lt[pk], downs=bt[::4])


def load(lyrics, timing):
    lyr = [l.strip() for l in open(lyrics, encoding="utf-8") if l.strip()]
    tm = json.load(open(timing))
    for d in tm:
        raw = lyr[d["idx"]]
        ads = re.findall(r"\(([^)]*)\)", raw)
        txt = re.sub(r"(\s*\([^)]*\))+\s*$", "", raw)
        words = txt.split()
        LINES.append(dict(idx=d["idx"], t0=d["t0"], te=d["t1"], wt=d["words"], words=words, ads=ads,
                          chorus=d["idx"] >= len(lyr) - 4))
    for a, b in zip(LINES, LINES[1:]):
        a["t1"] = b["t0"]
    LINES[-1]["t1"] = T1 + 1


def last(arr, t):
    i = int(np.searchsorted(arr, t, "right")) - 1
    return (i, t - arr[i]) if i >= 0 else (-1, 9.0)


def pulse(arr, t, tau):
    i, dt = last(arr, t)
    return math.exp(-dt / tau) if i >= 0 else 0.0


def line_at(t):
    for ln in LINES:
        if ln["t0"] <= t < ln["t1"]:
            return ln
    return None


# ------------------------------------------------------------------- type --
def is_deva(s):
    return any("ऀ" <= c <= "ॿ" for c in s)


def word_sprite(w, size):
    size = int(max(30, min(300, round(size / 6) * 6)))
    if is_deva(w):
        return tsprite(w, F["deva"], size, pad=int(size * 0.18))
    return tsprite(w.upper(), F["anton"], size, pad=int(size * 0.18))


def key_word(words):
    for w in words:
        if w.lower().strip(".,?\"'") in KEY or w.lower() in KEY:
            return w
    return max(words, key=lambda w: len(w.strip(",?.\"'")))


def layout(ln, box_w):
    """rows of (word, size, x, y) for a justified stack, centred on (0, 0)"""
    if "rows" in ln and ln.get("box") == box_w:
        return ln["rows"]
    words = ln["words"]
    n = len(words)
    nrows = 1 if n <= 2 else (2 if n <= 5 else (3 if n <= 8 else 4))
    per = math.ceil(n / nrows)
    groups = [words[i:i + per] for i in range(0, n, per)]
    rows, y, k = [], 0.0, 0
    for g in groups:
        base = 120
        sps = [word_sprite(w, base) for w in g]
        pads = [int(base * 0.18) for _ in g]
        gap = base * 0.16
        nat = sum(s.shape[1] - 2 * p for s, p in zip(sps, pads)) + gap * (len(g) - 1)
        size = max(46, min(240, base * box_w / max(nat, 1)))
        sps = [word_sprite(w, size) for w in g]
        pads = [int(round(size / 6) * 6 * 0.18) for _ in g]
        gap = size * 0.16
        ws = [s.shape[1] - 2 * p for s, p in zip(sps, pads)]
        tot = sum(ws) + gap * (len(g) - 1)
        rh = max(s.shape[0] - 2 * p for s, p in zip(sps, pads)) * 1.02
        x = -tot / 2
        for w, s, wd in zip(g, sps, ws):
            rows.append(dict(w=w, k=k, size=size, x=x + wd / 2, y=y + rh / 2))
            x += wd + gap
            k += 1
        y += rh + size * 0.12
    for r in rows:
        r["y"] -= y / 2
    ln["rows"], ln["box"] = rows, box_w
    return rows


def draw_stack(img, ln, t, cx, cy, box_w, hot=BLAZE):
    rows = layout(ln, box_w)
    kw = key_word(ln["words"])
    for r in rows:
        tw = ln["wt"][r["k"]] if r["k"] < len(ln["wt"]) else ln["t0"]
        dt = t - tw
        if dt < -0.02:
            continue
        a = sm((dt + 0.02) / 0.07)
        s = 1 + 0.35 * math.exp(-max(dt, 0) / 0.07)
        sp = word_sprite(r["w"], r["size"])
        col = hot if r["w"] == kw else BONE
        M = affine(cx + r["x"], cy + r["y"] - 18 * (1 - a), s, s, 0, sp.shape[1], sp.shape[0])
        if r["w"] == kw:
            draw_sprite(img, hot, blur(sp, 14), M, add=True, mul=0.55 * a)
        draw_sprite(img, col, sp, M, mul=a)


# ----------------------------------------------------------------- assets --
A = {}


def build_assets(cover):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    A["xx"], A["yy"] = xx, yy
    A["vig"] = np.clip(1 - 0.45 * (((xx - W / 2) / (W * 0.7)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2) ** 1.4, 0.3, 1)
    A["grain"] = [cv2.resize(np.random.default_rng(k).normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H)) for k in range(6)]
    c = np.asarray(Image.open(cover).convert("RGB"), np.float32) / 255
    c = cv2.resize(c, (720, 720), interpolation=cv2.INTER_CUBIC)
    g = c.mean(-1)
    g = cv2.createCLAHE(2.0, (6, 6)).apply((g * 255).astype(np.uint8)).astype(np.float32) / 255
    post = np.round(g * 4) / 4
    stops = [(0, BLACK), (0.35, EMBER), (0.65, BLAZE), (1.0, BONE)]
    x = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(x, [s[0] for s in stops], [s[1][i] for s in stops]) for i in range(3)], -1).astype(np.float32)
    A["cover"] = lut[(post * 255).astype(np.uint8)]
    m = np.ones((720, 720), np.float32)
    A["cover_a"] = m
    r = np.random.default_rng(4)
    A["sparks"] = r.uniform(0, 1, (220, 4)).astype(np.float32)


def cover(img, t, side, ln):
    cx = W * (0.74 if side > 0 else 0.26)
    i, dt = last(AU["beats"], t)
    s = 0.95 + 0.04 * math.exp(-dt / 0.1)
    rot = 4 * side + 1.5 * math.sin(t * 1.3)
    rgb = A["cover"].copy()
    if pulse(AU["bass"], t, 0.08) > 0.4:
        rr = np.random.default_rng(int(t * 24))
        for _ in range(6):
            y0 = int(rr.integers(0, 680))
            h = int(rr.integers(8, 50))
            rgb[y0:y0 + h] = np.roll(rgb[y0:y0 + h], int(rr.integers(-60, 60)), 1)
    M = affine(cx, H / 2, s, s, rot, 720, 720)
    sh = affine(cx + 22 * side, H / 2 + 26, s, s, rot, 720, 720)
    draw_sprite(img, BLACK * 0.0, A["cover_a"] * 0.6, sh)
    draw_sprite(img, rgb, A["cover_a"], M)
    fr = np.zeros((720, 720), np.float32)
    cv2.rectangle(fr, (4, 4), (715, 715), 1.0, 8)
    draw_sprite(img, BONE, fr, M)


# ------------------------------------------------------------ backgrounds --
def bg_speed(img, t, cx):
    xx, yy = A["xx"], A["yy"]
    ang = np.arctan2(yy - H / 2, xx - cx)
    rays = (np.sin(ang * 36 + t * 3) > 0.86).astype(np.float32)
    rays *= np.clip(np.hypot(xx - cx, yy - H / 2) / 900, 0, 1)
    img += rays[..., None] * EMBER * 1.6


def bg_rings(img, t, cx):
    m = np.zeros((H, W), np.float32)
    for b in AU["bass"][(AU["bass"] <= t) & (AU["bass"] > t - 1.2)]:
        r = (t - b) * 1600
        cv2.circle(m, (int(cx), H // 2), int(r), max(0.0, 1 - (t - b) / 1.2), 10, cv2.LINE_AA)
    img += (blur(m, 2) + blur(m, 18) * 0.6)[..., None] * BLAZE * 0.7


def bg_halftone(img, t, cx):
    xx, yy = A["xx"], A["yy"]
    cell = 26
    gx = (xx + t * 60) % cell - cell / 2
    gy = (yy + t * 30) % cell - cell / 2
    rad = 3 + 6 * (0.5 + 0.5 * np.sin(xx / 260 + t))
    dots = np.clip(rad - np.hypot(gx, gy), 0, 1)
    img += dots[..., None] * EMBER * 1.4


def bg_outline(img, t, cx, ln):
    w = key_word(ln["words"])
    sp = word_sprite(w, 300)
    big = cv2.resize(sp, (int(sp.shape[1] * 2.2), int(sp.shape[0] * 2.2)))
    e = np.clip(big - cv2.erode(big, np.ones((7, 7), np.uint8)), 0, 1)
    x = W / 2 - big.shape[1] / 2 + ((t - ln["t0"]) * -220)
    over(img, BLAZE * 0.55, e, x, H / 2 - big.shape[0] / 2)


def bg_sparks(img, t, cx, ln):
    age = t - ln["t0"]
    s = A["sparks"]
    ang = s[:, 0] * 6.283
    sp = 300 + s[:, 1] * 1100
    x = cx + np.cos(ang) * sp * age
    y = H / 2 + np.sin(ang) * sp * age + 400 * age * age
    a = np.clip(1 - age / 1.0, 0, 1)
    m = np.zeros((H, W), np.float32)
    for xi, yi, vx, vy in zip(x, y, np.cos(ang), np.sin(ang)):
        cv2.line(m, (int(xi), int(yi)), (int(xi - vx * 30), int(yi - vy * 30)), float(a), 2, cv2.LINE_AA)
    img += (m + blur(m, 6))[..., None] * HAZARD


def bg_hazard(img, t, cx):
    xx, yy = A["xx"], A["yy"]
    st = (((xx + yy + t * 240) // 70) % 2).astype(np.float32)
    band = (np.abs(yy - H * 0.92) < 40).astype(np.float32) + (np.abs(yy - H * 0.08) < 40).astype(np.float32)
    img += (st * band)[..., None] * HAZARD * 0.55


BGS = ["speed", "rings", "halftone", "outline", "sparks", "hazard"]


# ------------------------------------------------------------------ scenes --
def verse(img, t, ln):
    side = 1 if ln["idx"] % 2 == 0 else -1
    cxc = W * (0.74 if side > 0 else 0.26)
    kind = BGS[ln["idx"] % len(BGS)]
    if kind == "speed":
        bg_speed(img, t, cxc)
    elif kind == "rings":
        bg_rings(img, t, cxc)
    elif kind == "halftone":
        bg_halftone(img, t, cxc)
    elif kind == "outline":
        bg_outline(img, t, cxc, ln)
    elif kind == "sparks":
        bg_sparks(img, t, cxc, ln)
    else:
        bg_hazard(img, t, cxc)
    cover(img, t, side, ln)
    tx = W * (0.285 if side > 0 else 0.715)
    draw_stack(img, ln, t, tx, H / 2, 800)


def chorus(img, t, ln):
    k = ln["idx"]
    bg_rings(img, t, W / 2)
    bg_speed(img, t, W / 2)
    booms = [ln["wt"][i] for i, w in enumerate(ln["words"]) if "boom" in w.lower()]
    for b in booms:
        if 0 <= t - b < 0.18:
            img += (1 - (t - b) / 0.18) * 0.55 * BONE
        if 0 <= t - b < 0.9:
            age = t - b
            s = A["sparks"]
            ang = s[:, 0] * 6.283
            sp = 500 + s[:, 1] * 1500
            x = W / 2 + np.cos(ang) * sp * age
            y = H / 2 + np.sin(ang) * sp * age
            m = np.zeros((H, W), np.float32)
            for xi, yi, vx, vy in zip(x, y, np.cos(ang), np.sin(ang)):
                cv2.line(m, (int(xi), int(yi)), (int(xi - vx * 40), int(yi - vy * 40)), float(1 - age / 0.9), 3, cv2.LINE_AA)
            img += (m + blur(m, 8))[..., None] * HAZARD
    for i, w in enumerate(ln["words"]):
        if "बंदूक" in w and 0 <= t - ln["wt"][i] < 0.5:
            age = t - ln["wt"][i]
            m = np.zeros((H, W), np.float32)
            for a in range(8):
                th = a * math.pi / 4
                r = 80 + 500 * age
                cv2.line(m, (W // 2, H // 2), (int(W / 2 + math.cos(th) * r), int(H / 2 + math.sin(th) * r)), 1 - age / 0.5, 6, cv2.LINE_AA)
            img += (blur(m, 2) + blur(m, 20))[..., None] * BLAZE
    draw_stack(img, ln, t, W / 2, H / 2, 1400, hot=HAZARD)
    for j, ad in enumerate(ln["ads"]):
        ta = ln["te"] - 0.5 + 0.25 * j
        if t >= ta:
            a = sm((t - ta) / 0.08)
            sp = word_sprite(ad, 110)
            M = affine(W * (0.86 - 0.12 * j), H * (0.18 + 0.1 * j), 1, 1, -12 + 8 * j, sp.shape[1], sp.shape[0])
            draw_sprite(img, HAZARD, sp, M, mul=a)


def intro(img, t):
    a = sm((t - T0) / 0.25)
    sp = tsprite("BOOM SHAKA", F["anton"], 230, pad=30)
    over_c(img, BLAZE, sp * a, W / 2, H / 2 - 40)
    tg = RR.tracked("KRSNA", F["mono"], 34, 22)
    over_c(img, BONE, tg * a, W / 2, H / 2 + 130)


def outro(img, t):
    f = sm((t - 96.6) / 0.4)
    img *= 1 - 0.97 * f
    sp = tsprite("BOOM SHAKA", F["anton"], 230, pad=30)
    over_c(img, BLAZE, sp * f, W / 2, H / 2 - 40)
    tg = RR.tracked("KRSNA", F["mono"], 34, 22)
    over_c(img, BONE, tg * f, W / 2, H / 2 + 130)


def render_frame(fi):
    t = T0 + fi / FPS
    img = np.zeros((H, W, 3), np.float32) + BLACK
    ln = line_at(t)
    if ln is None:
        intro(img, t)
    elif ln["chorus"]:
        chorus(img, t, ln)
    else:
        verse(img, t, ln)
    if ln is not None and 0 <= t - ln["t0"] < 2 / FPS:
        img = img * 0.5 + BLAZE * 0.5
    if t > 96.6:
        outro(img, t)
    # camera: beat punch + bass shake (+ hard shake on "room" lines)
    z = 1 + 0.025 * pulse(AU["beats"], t, 0.09) + 0.03 * pulse(AU["downs"], t, 0.12)
    sh = pulse(AU["bass"], t, 0.07) * (14 if ln is not None and ln["chorus"] else 7)
    rot = 0.0
    if ln is not None and ln["chorus"] and any("room" in w for w in ln["words"]):
        ir = [ln["wt"][i] for i, w in enumerate(ln["words"]) if "room" in w]
        if ir and 0 <= t - ir[0] < 0.8:
            rot = 4 * math.sin((t - ir[0]) * 40) * (1 - (t - ir[0]) / 0.8)
    if z > 1.001 or sh > 0.5 or rot:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, z)
        M[0, 2] += sh * math.sin(fi * 2.7)
        M[1, 2] += sh * math.cos(fi * 1.9)
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    lum = img.max(-1)
    br = np.clip(lum - 0.7, 0, None)
    img = img + (blur(br, 16) * 0.4 + blur(br, 50) * 0.3)[..., None] * BLAZE
    img = img * A["vig"][..., None] + A["grain"][fi % 6][..., None] * 0.02
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# -------------------------------------------------------------------- main --
CFG = {}


def setup(audio, fontdir, argv):
    j = lambda n: os.path.join(fontdir, n)
    F.update(anton=j("Anton-Regular.ttf"), deva="/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
             mono=j("IBMPlexMono-Medium.ttf"))
    analyse(audio)
    load(argv[argv.index("--lyrics") + 1], argv[argv.index("--timing") + 1])
    build_assets(argv[argv.index("--cover") + 1])


def worker(job):
    k, f0, f1, out = job
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        enc.stdin.write(render_frame(fi).tobytes())
        if (fi - f0) % 96 == 0:
            print(f"[w{k}] {fi}/{f1}", flush=True)
    enc.stdin.close()
    enc.wait()
    return out


def main():
    audio, out, work, fontdir = sys.argv[1:5]
    os.makedirs(work, exist_ok=True)
    setup(audio, fontdir, sys.argv)
    if "--preview" in sys.argv:
        os.makedirs(out, exist_ok=True)
        for s in sys.argv[sys.argv.index("--preview") + 1].split(","):
            t = float(s)
            Image.fromarray(render_frame(int(round((t - T0) * FPS)))).save(os.path.join(out, f"b_{t:05.2f}.png"))
        return
    nw = int(os.environ.get("WORKERS", "4"))
    b = np.linspace(0, NF, nw + 1).astype(int)
    jobs = [(k, int(b[k]), int(b[k + 1]), os.path.join(work, f"boom_chunk{k}.mp4")) for k in range(nw)]
    with mp.get_context("fork").Pool(nw) as pool:
        parts = pool.map(worker, jobs)
    lst = os.path.join(work, "boom_chunks.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", str(T0), "-t", str(T1 - T0), "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", f"afade=t=in:st=0:d=0.05,afade=t=out:st={T1 - T0 - 0.6}:d=0.6", "-c:a", "aac", "-b:a", "256k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
