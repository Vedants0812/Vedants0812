#!/usr/bin/env python3
"""
SEEDHE MAUT - RED : lyric remix, 1920x1080 @ 24 fps.

Line times come from the video's own burned-in subtitles (read frame by frame),
word times from the isolated vocal track. The footage fills the frame (cropped
above the burned subtitles) and changes treatment per line; the lyric is a big
kinetic word stack whose words slam in on their syllables. Lyric text and timing
are read from local files at run time.

usage:
  python3 render_red_lyric.py SOURCE.mp4 OUT.mp4 WORKDIR FONTDIR --lyrics L.txt --timing T.json
  ... --preview 2,12,33
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
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "madira-reel"))
import render_red16 as R16  # noqa: E402  (render modes)
from render_reel import over_c, affine, draw_sprite, blur, sm, tsprite  # noqa: E402

W, H, FPS = 1920, 1080, 24
DUR = 70.7
NF = int(DUR * FPS)
CROP = (0, 40, 1526, 858)       # x, y, w, h: 16:9, above the burned-in subtitles
RED, WHITE, INK = R16.RED, R16.WHITE, R16.INK
SRC = None
LINES = []
FN = {}

# per line: footage mode, text placement, key word, extra effect
PLAN = [
    ("thermal", "c", "जानेजान", None), ("echo", "l", "खाली", "sparks"), ("sobel", "r", "प्राण", None),
    ("raw", "c", "हनुमान", "fire"), ("news", "l", "भालू", None), ("news", "r", "Sarojini", None),
    ("strobe", "c", "hypocrisy", "redact"), ("mirror", "c", "democracy", None), ("raw", "l", "भूरी", None),
    ("news", "names", "Azad", None), ("glyph", "r", "देश", None), ("thermal", "l", "औकात", None),
    ("split", "c", "anti-government,", None), ("thermal", "l", "Anti-establishment", None),
    ("split2", "r", "chemical,", None), ("mosh", "c", "cancel", "strike"), ("strobe", "c", "ज़ोर", "shake"),
    ("bayer", "l", "Mumbai,", "shake"), ("echo", "r", "जात", None), ("raw", "hindi", "हिंदी", None),
    ("raw", "c", "आशा", "flood"),
]
SPLIT_MODES = {"split": ["raw", "sobel", "bayer"], "split2": ["sobel", "glyph", "thermal"]}


# ------------------------------------------------------------------- data --
def load(lyrics, timing):
    lyr = [l.strip() for l in open(lyrics, encoding="utf-8") if l.strip()]
    tm = json.load(open(timing))
    for d in tm:
        i = d["idx"]
        mode, place, key, fx = PLAN[i]
        LINES.append(dict(idx=i, t0=d["t0"], te=d["t1"], wt=d["words"], words=lyr[i].split(), anchors=d.get("anchors", []),
                          mode=mode, place=place, key=key, fx=fx))
    for a, b in zip(LINES, LINES[1:]):
        a["t1"] = b["t0"]
    LINES[-1]["t1"] = DUR + 1
    R16.SEG[:] = [(ln["t0"], ln["t1"], ln["mode"]) for ln in LINES]


def line_at(t):
    for ln in LINES:
        if ln["t0"] <= t < ln["t1"]:
            return ln
    return None


def pulse(arr, t, tau):
    return R16.pulse(arr, t, tau)


# ----------------------------------------------------------------- source --
class Src:
    def __init__(self, f0):
        self.f0 = max(0, f0)
        x, y, w, h = CROP
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{self.f0 / FPS:.4f}", "-i", SRC, "-vf",
                                   f"fps={FPS},crop={w}:{h}:{x}:{y},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                  stdout=subprocess.PIPE)
        self.n = self.f0 - 1
        self.buf = {}

    def get(self, fi):
        fi = max(0, fi)
        while self.n < fi:
            b = self.p.stdout.read(W * H * 3)
            if len(b) < W * H * 3:
                break
            self.n += 1
            self.buf[self.n] = np.frombuffer(b, np.uint8).reshape(H, W, 3)
            self.buf.pop(self.n - 40, None)
        fi = min(fi, self.n)
        while fi not in self.buf and fi < self.n:
            fi += 1
        return self.buf[fi].astype(np.float32) / 255

    def close(self):
        self.p.kill()


# ---------------------------------------------------------------- footage --
def footage(ln, src, fi, t):
    mode = ln["mode"] if ln else "raw"
    if mode in SPLIT_MODES:
        k = sum(1 for a in ln["anchors"] if t >= a)
        mode = SPLIT_MODES[mode][min(k, 2)]
    if mode == "strobe":
        step = 3
        hold = fi - (fi % step)
        img, m = R16.bw_red(src.get(hold))
        if (fi // step) % 3 == 1:
            img = 1 - img.mean(-1, keepdims=True) * np.ones(3)
        elif (fi // step) % 3 == 2:
            img = img.mean(-1, keepdims=True) * RED * 1.5 + img * 0.1
        return img
    return R16.mode_frame(mode, src, fi, t)


# ------------------------------------------------------------------- type --
def is_deva(s):
    return any("ऀ" <= c <= "ॿ" for c in s)


def wsprite(w, size):
    size = int(max(36, min(320, round(size / 6) * 6)))
    if is_deva(w):
        return tsprite(w, FN["deva"], size, pad=int(size * 0.18))
    return tsprite(w.upper(), FN["anton"], size, pad=int(size * 0.18))


def layout(ln, box_w):
    if ln.get("box") == box_w:
        return ln["rows"]
    words = ln["words"]
    n = len(words)
    nrows = 1 if n <= 2 else (2 if n <= 5 else (3 if n <= 8 else 4))
    per = math.ceil(n / nrows)
    rows, y, k = [], 0.0, 0
    for g in [words[i:i + per] for i in range(0, n, per)]:
        base = 120
        nat = sum(wsprite(w, base).shape[1] - 2 * int(base * 0.18) for w in g) + base * 0.16 * (len(g) - 1)
        size = max(50, min(220, base * box_w / max(nat, 1)))
        sps = [wsprite(w, size) for w in g]
        pad = int(round(size / 6) * 6 * 0.18)
        ws = [s.shape[1] - 2 * pad for s in sps]
        gap = size * 0.16
        rh = max(s.shape[0] - 2 * pad for s in sps) * 1.02
        x = -(sum(ws) + gap * (len(g) - 1)) / 2
        for w, wd in zip(g, ws):
            rows.append(dict(w=w, k=k, size=size, x=x + wd / 2, y=y + rh / 2))
            x += wd + gap
            k += 1
        y += rh + size * 0.12
    for r in rows:
        r["y"] -= y / 2
    ln["rows"], ln["box"] = rows, box_w
    return rows


def stack(img, ln, t, cx, cy, box_w):
    for r in layout(ln, box_w):
        tw = ln["wt"][r["k"]] if r["k"] < len(ln["wt"]) else ln["t0"]
        dt = t - tw
        if dt < -0.02:
            continue
        a = sm((dt + 0.02) / 0.06)
        s = 1 + 0.4 * math.exp(-max(dt, 0) / 0.06)
        sp = wsprite(r["w"], r["size"])
        hot = ln["key"] and ln["key"].lower() in r["w"].lower()
        M = affine(cx + r["x"], cy + r["y"] - 16 * (1 - a), s, s, 0, sp.shape[1], sp.shape[0])
        sh = affine(cx + r["x"] + 6, cy + r["y"] + 8, s, s, 0, sp.shape[1], sp.shape[0])
        draw_sprite(img, INK * 0.2, blur(sp, 4), sh, mul=0.85 * a)
        if hot:
            draw_sprite(img, RED, blur(sp, 16), M, add=True, mul=0.6 * a)
        draw_sprite(img, RED if hot else WHITE, sp, M, mul=a)


def knock(img, word, t, t0, frame, cy=H / 2, size=300):
    sp = wsprite(word, size)
    k = t - t0
    s = 1 + 0.5 * math.exp(-max(k, 0) / 0.07)
    mask = np.zeros((H, W), np.float32)
    M = affine(W / 2, cy, s, s, 0, sp.shape[1], sp.shape[0])
    draw_sprite(mask[..., None], np.ones(1, np.float32), sp, M)
    a = sm(k / 0.05)
    edge = np.clip(cv2.dilate(mask, np.ones((15, 15), np.uint8)) - mask, 0, 1)
    fill = np.clip(frame * 1.9 + 0.18, 0, 1)
    img[:] = img * (1 - mask[..., None] * a) + fill * mask[..., None] * a
    img[:] = img * (1 - edge[..., None] * a) + RED * edge[..., None] * a
    img += blur(edge, 14)[..., None] * RED * 0.5 * a


# --------------------------------------------------------------- effects --
def fx_layer(img, ln, t, fi, src):
    fx = ln["fx"]
    age = t - ln["t0"]
    if fx in ("sparks", "fire"):
        r = np.random.default_rng(ln["idx"])
        n = 160
        ang = r.uniform(-math.pi, 0, n) if fx == "fire" else r.uniform(0, 2 * math.pi, n)
        sp = r.uniform(300, 1300, n)
        x = W / 2 + np.cos(ang) * sp * age
        y = (H * 0.95 if fx == "fire" else H / 2) + np.sin(ang) * sp * age + 300 * age * age
        a = max(0.0, 1 - age / 1.6)
        m = np.zeros((H, W), np.float32)
        for xi, yi, c, s_ in zip(x, y, np.cos(ang), np.sin(ang)):
            cv2.line(m, (int(xi), int(yi)), (int(xi - c * 26), int(yi - s_ * 26)), a, 3, cv2.LINE_AA)
        img += (m + blur(m, 8))[..., None] * RED * 1.2
    if fx == "redact":
        rr = np.random.default_rng(fi // 3)
        for _ in range(3):
            y0 = int(rr.uniform(0.1, 0.85) * H)
            x0 = int(rr.uniform(0, 0.5) * W)
            img[y0:y0 + int(rr.uniform(30, 70)), x0:x0 + int(rr.uniform(0.3, 0.6) * W)] = RED
    if fx == "strike":
        ks = [ln["wt"][i] for i, w in enumerate(ln["words"]) if "cancel" in w.lower()]
        if ks and t >= ks[0]:
            p = sm((t - ks[0]) / 0.25)
            m = np.zeros((H, W), np.float32)
            cv2.line(m, (int(W * 0.12), int(H * 0.55)), (int(W * 0.12 + W * 0.76 * p), int(H * 0.47)), 1.0, 26, cv2.LINE_AA)
            img[:] = img * (1 - m[..., None]) + RED * m[..., None]
    if fx == "flood":
        u = sm(age / 1.4)
        img[:] = img * (1 - 0.55 * u) + RED * 0.55 * u * img.mean(-1, keepdims=True) * 2.2


NAMES = ["APJ", "BOSE", "BHAGAT SINGH", "CHANDRA SHEKHAR AZAD"]


def names_scene(img, ln, t, frame):
    starts = []
    words = ln["words"]
    idx = [0, 1, 2, 4]
    for k, i in enumerate(idx):
        starts.append(ln["wt"][min(i, len(ln["wt"]) - 1)])
    k = sum(1 for s in starts if t >= s) - 1
    if k < 0:
        return
    img *= 0.35
    size = 330 if k < 2 else (200 if k == 2 else 130)
    knock(img, NAMES[k], t, starts[k], frame, H / 2, size)
    small = tsprite("  ·  ".join(NAMES[:k + 1]), FN["mono"], 30, pad=10)
    over_c(img, WHITE, small * 0.85, W / 2, H * 0.86)


def hindi_scene(img, ln, t, frame):
    words = ln["words"]
    hi = next((i for i, w in enumerate(words) if "हिंदी" in w), len(words) - 3)
    th = ln["wt"][hi]
    if t < th:
        part = dict(ln, words=words[:hi], wt=ln["wt"][:hi], rows=None, box=None)
        stack(img, part, t, W / 2, H / 2, 1300)
        return
    img *= 0.35
    knock(img, "हिंदी", t, th, frame, H * 0.44, 380)
    rest = " ".join(words[hi + 1:])
    if rest and t >= ln["wt"][min(hi + 1, len(ln["wt"]) - 1)]:
        sp = wsprite(rest, 90)
        over_c(img, WHITE, sp, W / 2, H * 0.82)


# --------------------------------------------------------------------- HUD --
def hud(img, t, ln):
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil, "RGBA")
    f1 = ImageFont.truetype(FN["anton"], 46)
    f2 = ImageFont.truetype(FN["mono"], 20)
    d.text((48, 42), "RED", font=f1, fill=(226, 18, 26))
    d.text((140, 62), "SEEDHE MAUT", font=f2, fill=(238, 238, 232, 220))
    if int(t * 2) % 2 == 0:
        d.ellipse([W - 250, 52, W - 228, 74], fill=(226, 18, 26))
    d.text((W - 216, 50), f"REC  {int(t // 60):02d}:{int(t % 60):02d}:{int((t % 1) * 24):02d}", font=f2, fill=(238, 238, 232, 220))
    if ln is not None:
        mode = ln["mode"]
        if mode in SPLIT_MODES:
            mode = SPLIT_MODES[mode][min(sum(1 for a in ln["anchors"] if t >= a), 2)]
        lab = f"render red.mp4 --mode {mode}"
        n = int((t - ln["t0"]) * 45) + 1
        d.rectangle([44, H - 78, 44 + 14 + f2.getlength(lab) + 20, H - 44], fill=(10, 10, 12, 190))
        d.text((54, H - 72), ">", font=f2, fill=(226, 18, 26))
        d.text((74, H - 72), lab[:n], font=f2, fill=(238, 238, 232))
    return np.asarray(pil, np.float32) / 255


# ------------------------------------------------------------------ frame --
VIG = None


def render_frame(fi, src):
    global VIG
    t = fi / FPS
    ln = line_at(t)
    frame_raw = R16.bw_red(src.get(fi))[0]
    img = footage(ln, src, fi, t).copy() if ln is not None else frame_raw * 0.5
    # dim the footage behind the type
    img = img * 0.62
    # bass: redaction bars + flood pulse
    for b in R16.AU["bass"][(R16.AU["bass"] <= t) & (R16.AU["bass"] > t - 0.16)]:
        r = np.random.default_rng(int(b * 1000))
        y0 = int(r.uniform(0.05, 0.9) * H)
        img[y0:y0 + int(r.uniform(20, 46)), int(r.uniform(0, 0.4) * W):int(r.uniform(0.6, 1.0) * W)] = RED * 0.9
    if ln is None:
        bts = [b for b in R16.AU["beats"] if b < 1.1]
        if t > 0.15:
            k = t - 0.15
            sp = tsprite("RED", FN["anton"], 520, pad=40)
            s = 1 + 0.4 * math.exp(-k / 0.08)
            M = affine(W / 2, H / 2, s, s, 0, sp.shape[1], sp.shape[0])
            mask = np.zeros((H, W), np.float32)
            draw_sprite(mask[..., None], np.ones(1, np.float32), sp, M)
            img = img * 0.3 * (1 - mask[..., None]) + np.clip(frame_raw * 1.4, 0, 1) * mask[..., None]
            img += blur(mask, 30)[..., None] * RED * 0.6
    else:
        if ln["fx"]:
            fx_layer(img, ln, t, fi, src)
        if ln["place"] == "names":
            names_scene(img, ln, t, frame_raw)
        elif ln["place"] == "hindi":
            hindi_scene(img, ln, t, frame_raw)
        else:
            cx = {"c": W / 2, "l": W * 0.34, "r": W * 0.66}[ln["place"]]
            stack(img, ln, t, cx, H * 0.5, 1300 if ln["place"] == "c" else 1000)
        if t - ln["t0"] < 2 / FPS:
            img = img * 0.5 + RED * 0.5
    # end card
    if t > 70.1:
        f = sm((t - 70.1) / 0.3)
        img = img * (1 - 0.9 * f)
        sp = tsprite("RED", FN["anton"], 420, pad=40)
        over_c(img, RED, sp * f, W / 2, H / 2 - 40)
        sb = tsprite("SEEDHE MAUT", FN["mono"], 40, pad=10)
        over_c(img, WHITE, sb * f, W / 2, H / 2 + 200)
    # camera: beat punch, bass shake, extra shake on marked lines
    z = 1 + 0.03 * pulse(R16.AU["beats"], t, 0.08) + 0.03 * pulse(R16.AU["downs"], t, 0.12)
    sh = pulse(R16.AU["bass"], t, 0.07) * 10
    rot = 0.0
    if ln is not None and ln["fx"] == "shake":
        sh += 8
        rot = 1.6 * math.sin(fi * 1.7)
    if z > 1.002 or sh > 0.5 or rot:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, z)
        M[0, 2] += sh * math.sin(fi * 2.6)
        M[1, 2] += sh * math.cos(fi * 1.9)
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    img = hud(img, t, ln) if t < 70.1 else img
    if VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        VIG = np.clip(1 - 0.45 * (((xx - W / 2) / (W * 0.7)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2) ** 1.4, 0.3, 1)
    lum = img.max(-1)
    img = img + blur(np.clip(lum - 0.75, 0, None), 18)[..., None] * RED * 0.35
    g = np.random.default_rng(fi % 8).normal(0, 0.022, (H // 2, W // 2)).astype(np.float32)
    img = img * VIG[..., None] + cv2.resize(g, (W, H))[..., None]
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------- main --
def setup(src, fontdir, argv):
    global SRC
    SRC = src
    R16.SRC = src
    R16.MW, R16.MH = W, H
    j = lambda n: os.path.join(fontdir, n)
    FN.update(anton=j("Anton-Regular.ttf"), mono=j("IBMPlexMono-Medium.ttf"),
              deva="/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf")
    R16.FN.update(mono=j("IBMPlexMono-Regular.ttf"), bold=j("IBMPlexMono-Medium.ttf"), anton=FN["anton"], deva=FN["deva"],
                  deva_r="/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf")
    R16.analyse(os.environ.get("RED_AUDIO", src))
    load(argv[argv.index("--lyrics") + 1], argv[argv.index("--timing") + 1])


def worker(job):
    k, f0, f1, out = job
    src = Src(f0 - 30)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        enc.stdin.write(render_frame(fi, src).tobytes())
        if (fi - f0) % 96 == 0:
            print(f"[w{k}] {fi}/{f1}", flush=True)
    enc.stdin.close()
    enc.wait()
    src.close()
    return out


def main():
    src, out, work, fontdir = sys.argv[1:5]
    os.makedirs(work, exist_ok=True)
    setup(src, fontdir, sys.argv)
    if "--preview" in sys.argv:
        os.makedirs(out, exist_ok=True)
        for s in sys.argv[sys.argv.index("--preview") + 1].split(","):
            t = float(s)
            fi = int(round(t * FPS))
            sr = Src(fi - 30)
            R16.MO.clear()
            Image.fromarray(render_frame(fi, sr)).save(os.path.join(out, f"l_{t:05.2f}.png"))
            sr.close()
        return
    nw = int(os.environ.get("WORKERS", "4"))
    b = np.linspace(0, NF, nw + 1).astype(int)
    jobs = [(k, int(b[k]), int(b[k + 1]), os.path.join(work, f"redlyr_chunk{k}.mp4")) for k in range(nw)]
    with mp.get_context("fork").Pool(nw) as pool:
        parts = pool.map(worker, jobs)
    lst = os.path.join(work, "redlyr_chunks.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", src,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest",
                    "-movflags", "+faststart", out], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
