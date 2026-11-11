#!/usr/bin/env python3
"""
SEEDHE MAUT - RED : "red is the only thing alive" remix, 1080x1920 @ 24 fps.

Built from the RED music video footage. Everything is black and white except
red, and the red misbehaves: it floods on bass hits, is cut out and pops as a
sticker, smears through time, prints as halftone, melts (datamosh), stutters
and finally drowns the frame. Every switch is on the track's bar grid (136 BPM).

usage:
  python3 render_red.py SOURCE.mp4 OUT.mp4 WORKDIR FONTDIR
  python3 render_red.py SOURCE.mp4 OUTDIR  WORKDIR FONTDIR --preview 1,5,12
"""
import math
import multiprocessing as mp
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "madira-reel"))
import render_reel as RR  # noqa: E402
from render_reel import over_c, affine, draw_sprite, blur, sm, eout, tsprite  # noqa: E402

W, H, FPS = 1080, 1920, 24
SW, SH = 1526, 1080
RED = np.array([0.86, 0.06, 0.10], np.float32)
DEEP = np.array([0.35, 0.0, 0.03], np.float32)
WHITE = np.array([0.95, 0.95, 0.95], np.float32)
SRC = None
DUR = 70.7
NF = int(DUR * FPS)
AU = {}
FONT = {}
SEG = []


# ------------------------------------------------------------------ audio --
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


def last(arr, t):
    i = int(np.searchsorted(arr, t, "right")) - 1
    return (i, t - arr[i]) if i >= 0 else (-1, 9.0)


def pulse(arr, t, tau):
    i, dt = last(arr, t)
    return math.exp(-dt / tau) if i >= 0 else 0.0


# ------------------------------------------------------------- schedule ----
def build_schedule():
    d = AU["downs"]

    def bars_from(a, b, every, modes):
        out = []
        ds = [x for x in d if a - 0.05 <= x < b]
        if not ds or ds[0] > a + 0.2:
            ds = [a] + ds
        k = 0
        for i in range(0, len(ds), every):
            s0 = ds[i] if i else a
            s1 = ds[i + every] if i + every < len(ds) else b
            out.append((s0, s1, modes[k % len(modes)]))
            k += 1
        return out

    S = [(0.0, 2.3, "intro")]
    S += [(2.3, 7.6, "flood")]
    S += [(7.6, 10.3, "triptych")]
    S += bars_from(10.3, 22.8, 2, ["sticker", "smear", "halftone", "flood"])
    S += bars_from(22.8, 35.1, 2, ["kaleido", "triptych", "mosh", "strobe"])
    S += [(35.1, 37.8, "laal")]
    S += bars_from(37.8, 58.7, 1, ["sticker", "kaleido", "smear", "halftone", "triptych", "flood", "strobe", "mosh"])
    S += [(58.7, 62.6, "mosh")]
    S += [(62.6, 65.1, "strobe")]
    S += [(65.1, DUR + 1, "drown")]
    SEG[:] = S


def seg_at(t):
    for s in SEG:
        if s[0] <= t < s[1]:
            return s
    return SEG[-1]


# --------------------------------------------------------------- source ----
class Src:
    """sequential decoder with a ring buffer (frames as float32 0..1, source size)"""

    def __init__(self, f0):
        self.f0 = max(0, f0)
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{self.f0 / FPS:.4f}", "-i", SRC, "-vf", f"fps={FPS}",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.n = self.f0 - 1
        self.buf = {}

    def get(self, fi):
        fi = max(0, fi)
        while self.n < fi:
            b = self.p.stdout.read(SW * SH * 3)
            if len(b) < SW * SH * 3:
                break
            self.n += 1
            self.buf[self.n] = np.frombuffer(b, np.uint8).reshape(SH, SW, 3)
            self.buf.pop(self.n - 60, None)
        fi = min(fi, self.n)
        while fi not in self.buf and fi < self.n:
            fi += 1
        return self.buf[fi].astype(np.float32) / 255

    def close(self):
        self.p.kill()


def redmask(f):
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    m = np.clip((r - np.maximum(g, b) - 0.10) * 5, 0, 1) * (r > 0.22)
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)


def bw_red(f, boost=1.4):
    """grey everywhere, red kept and pushed"""
    m = redmask(f)[..., None]
    g = f.mean(-1, keepdims=True)
    g = np.clip((g - 0.04) * 1.15, 0, 1)
    red = np.clip(f * np.array([boost, 0.55, 0.6], np.float32), 0, 1)
    return g * (1 - m) + red * m, m[..., 0]


def full_crop(img, pan=0.0, zoom=1.0):
    """16:9 source -> 9:16 full-bleed crop"""
    s = H / SH * zoom
    w = int(SW * s)
    x0 = (w - W) / 2 + pan * (w - W) / 2
    M = np.float32([[s, 0, -x0], [0, s, -(SH * s - H) / 2]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def band(img, h=None):
    """16:9 source scaled to full width"""
    bh = int(SH * W / SW)
    out = cv2.resize(img, (W, bh), interpolation=cv2.INTER_AREA)
    if h and h < bh:
        y0 = (bh - h) // 2
        out = out[y0:y0 + h]
    return out


def punch(img, t, amt=0.05):
    z = 1 + amt * pulse(AU["beats"], t, 0.09) + amt * 0.8 * pulse(AU["downs"], t, 0.12)
    if z < 1.002:
        return img
    M = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- modes ----
def m_flood(src, fi, t, amount=None):
    f = src.get(fi)
    img, m = bw_red(f)
    img = full_crop(img, pan=0.25 * math.sin(t * 0.4))
    m = full_crop(m)
    a = 0.55 * pulse(AU["bass"], t, 0.22) if amount is None else amount
    if a > 0.01:
        k = int(10 + 160 * a)
        fl = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        fl = blur(fl, 10 + 30 * a) * min(1.0, 0.4 + a)
        lum = img.mean(-1, keepdims=True)
        img = img * (1 - fl[..., None]) + (RED * (0.35 + 1.1 * lum)) * fl[..., None]
    img += blur(m, 24)[..., None] * RED * 0.45
    return punch(img, t)


def m_triptych(src, fi, t):
    out = np.zeros((H, W, 3), np.float32)
    ph = H // 3
    lag = int(round(0.441 / 2 * FPS))
    for k in range(3):
        f = src.get(fi - k * lag)
        img, m = bw_red(f)
        b = band(img, ph)
        if k == 1:
            b = b.mean(-1, keepdims=True) * RED * 1.6 + b * 0.15
        if k == 2:
            b = 1 - b
            b = b.mean(-1, keepdims=True) * np.array([0.9, 0.9, 0.9])
        out[k * ph:k * ph + b.shape[0]] = b
        out[k * ph:k * ph + 4] = RED
    return out


def m_sticker(src, fi, t):
    f = src.get(fi)
    img, m = bw_red(f)
    img = full_crop(img)
    m = full_crop(m)
    base = img * 0.35
    big = cv2.dilate((m > 0.3).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    n, lab, stats, cents = cv2.connectedComponentsWithStats((big > 0).astype(np.uint8))
    sel = np.zeros_like(big)
    if n > 1:
        idx = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:3] + 1
        for i in idx:
            if stats[i, cv2.CC_STAT_AREA] > 1500:
                sel[lab == i] = 1
    if sel.sum() < 4000:
        sel = np.zeros_like(big)
        rw, rh = int(W * 0.72), int(H * 0.46)
        cv2.rectangle(sel, (W // 2 - rw // 2, H // 2 - rh // 2), (W // 2 + rw // 2, H // 2 + rh // 2), 1.0, -1)
    sel = cv2.GaussianBlur(sel, (0, 0), 1.0)
    ys, xs = np.nonzero(sel > 0.5)
    cx, cy = (xs.mean(), ys.mean()) if len(xs) else (W / 2, H / 2)
    s = 1.0 + 0.22 * pulse(AU["beats"], t, 0.12) + 0.12
    M = np.float32([[s, 0, (1 - s) * cx], [0, s, (1 - s) * cy]])
    cut = cv2.warpAffine(img, M, (W, H))
    sm_ = cv2.warpAffine(sel, M, (W, H))
    outline = np.clip(cv2.dilate(sm_, np.ones((15, 15), np.uint8)) - sm_, 0, 1)
    shadow = cv2.GaussianBlur(np.roll(np.roll(sm_, 18, 0), 12, 1), (0, 0), 10)
    out = base * (1 - 0.6 * shadow[..., None])
    out = out * (1 - outline[..., None]) + WHITE * outline[..., None]
    out = out * (1 - sm_[..., None]) + np.clip(cut * 1.15, 0, 1) * sm_[..., None]
    return out


def m_smear(src, fi, t):
    out = np.zeros((H, W, 3), np.float32)
    blk = 24
    cache = {}
    for y0 in range(0, H, blk):
        d = int(10 + 9 * math.sin(y0 / 170 + t * 3.0))
        if d not in cache:
            img, _ = bw_red(src.get(fi - d))
            cache[d] = full_crop(img)
        out[y0:y0 + blk] = cache[d][y0:y0 + blk]
    cur, m = bw_red(src.get(fi))
    cur = full_crop(cur)
    mm = full_crop(m)[..., None]
    out = out * (1 - mm) + cur * mm
    return out


DOT = None


def m_halftone(src, fi, t):
    global DOT
    f = src.get(fi)
    img, m = bw_red(f)
    img = full_crop(img, zoom=1.0 + 0.04 * pulse(AU["downs"], t, 0.2))
    m = full_crop(m)
    cell = 12
    gh, gw = H // cell, W // cell
    g = cv2.resize(img.mean(-1), (gw, gh), interpolation=cv2.INTER_AREA)
    mr = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)
    if DOT is None:
        lv = []
        for k in range(12):
            d = np.zeros((cell * 4, cell * 4), np.float32)
            r = k / 11 * cell * 0.62 * 4
            if r > 0:
                cv2.circle(d, (cell * 2, cell * 2), int(r), 1.0, -1, cv2.LINE_AA)
            lv.append(cv2.resize(d, (cell, cell), interpolation=cv2.INTER_AREA))
        DOT = np.stack(lv)
    idx = np.clip(((1 - g) * 11.99), 0, 11).astype(int)
    dots = DOT[idx].transpose(0, 2, 1, 3).reshape(gh * cell, gw * cell)
    mr_up = cv2.resize(mr, (gw * cell, gh * cell), interpolation=cv2.INTER_NEAREST)
    paper = np.array([0.93, 0.91, 0.86], np.float32)
    ink = np.where(mr_up[..., None] > 0.3, RED, np.array([0.05, 0.05, 0.06], np.float32))
    out = np.ones((H, W, 3), np.float32) * paper
    hh, ww = dots.shape
    out[:hh, :ww] = paper * (1 - dots[..., None]) + ink * dots[..., None]
    return punch(out, t, 0.03)


def m_kaleido(src, fi, t):
    img, m = bw_red(src.get(fi))
    b = band(img)
    sq = min(b.shape[0], W // 2 + 200)
    x0 = int((b.shape[1] - sq) / 2 + 160 * math.sin(t * 0.7))
    tile = b[:sq, max(0, x0):max(0, x0) + sq]
    tile = cv2.resize(tile, (W // 2, W // 2))
    q = np.vstack([np.hstack([tile, tile[:, ::-1]]), np.hstack([tile[::-1], tile[::-1, ::-1]])])
    reps = int(math.ceil(H / q.shape[0]))
    col = np.vstack([q if k % 2 == 0 else q[::-1] for k in range(reps)])[:H]
    ang = 8 * math.sin(t * 0.9) + 25 * pulse(AU["downs"], t, 0.25)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.25)
    return cv2.warpAffine(col, M, (W, H), borderMode=cv2.BORDER_REFLECT)


MOSH = {}


def flow_of(a, b):
    ga = cv2.resize((a.mean(-1) * 255).astype(np.uint8), (W // 4, H // 4))
    gb = cv2.resize((b.mean(-1) * 255).astype(np.uint8), (W // 4, H // 4))
    fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(fl, (W, H)) * 4


def m_mosh(src, fi, t, seg):
    """datamosh: keep the old picture and drag it along the motion of the new one"""
    s0 = int(round(seg[0] * FPS))
    if MOSH.get("seg") != seg[0] or MOSH.get("fi", -9) != fi - 1:
        start = s0 if MOSH.get("seg") != seg[0] else fi
        img0, _ = bw_red(src.get(start))
        MOSH.update(seg=seg[0], acc=full_crop(img0), prev=full_crop(img0), fi=start)
        for k in range(start + 1, fi):
            mosh_step(src, k)
    mosh_step(src, fi)
    out = MOSH["acc"].copy()
    return out


def mosh_step(src, k):
    cur, m = bw_red(src.get(k))
    cur = full_crop(cur)
    mm = full_crop(m)
    fl = flow_of(MOSH["prev"], cur)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    acc = cv2.remap(MOSH["acc"], gx - fl[..., 0] * 1.4, gy - fl[..., 1] * 1.4, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    t = k / FPS
    refresh = 0.06 + 0.9 * (pulse(AU["downs"], t, 0.05) > 0.9)
    acc = acc * (1 - refresh) + cur * refresh
    acc = acc * (1 - mm[..., None] * 0.6) + cur * mm[..., None] * 0.6
    MOSH.update(acc=acc, prev=cur, fi=k)


def m_strobe(src, fi, t):
    step = int(round(0.441 / 2 * FPS))
    i, _ = last(AU["beats"], t)
    b0 = int(round(AU["beats"][max(0, i)] * FPS))
    hold = b0 + ((fi - b0) // max(1, step)) * step
    img, m = bw_red(src.get(hold))
    img = full_crop(img, zoom=1.0 + 0.08 * ((fi // step) % 3))
    m = full_crop(m)
    ph = (fi // 2) % 4
    if ph == 0:
        img = 1 - img.mean(-1, keepdims=True) * np.ones(3)
    elif ph == 2:
        img = img.mean(-1, keepdims=True) * RED * 1.5 + img * 0.1
    return punch(img, t, 0.07)


def knockout(src, fi, t, word, font, size, bg=RED):
    img, m = bw_red(src.get(fi))
    img = full_crop(img)
    sp = tsprite(word, FONT[font], size, pad=30)
    mask = np.zeros((H, W), np.float32)
    i, dt = last(AU["downs"], t)
    s = 1 + 0.35 * math.exp(-dt / 0.08)
    M = affine(W / 2, H * 0.5, s, s, 0, sp.shape[1], sp.shape[0])
    draw_sprite(mask[..., None], np.ones(1, np.float32), sp, M)
    out = np.ones((H, W, 3), np.float32) * bg
    out = out * (1 - mask[..., None]) + np.clip(img * 1.2, 0, 1) * mask[..., None]
    return out


def m_intro(src, fi, t):
    out = np.zeros((H, W, 3), np.float32)
    bts = [b for b in AU["beats"] if b < 2.4][:4]
    letters = "RED"
    img, _ = bw_red(src.get(fi))
    img = full_crop(img)
    n = sum(1 for b in bts[:3] if t >= b)
    if n:
        s = tsprite(letters[:n], FONT["anton"], 560, pad=40)
        k = t - bts[n - 1]
        z = 1 + 0.4 * math.exp(-k / 0.07)
        mask = np.zeros((H, W), np.float32)
        M = affine(W / 2, H * 0.5, z, z, 0, s.shape[1], s.shape[0])
        draw_sprite(mask[..., None], np.ones(1, np.float32), s, M)
        out = out * (1 - mask[..., None]) + np.clip(img * 1.3, 0, 1) * mask[..., None]
        out += blur(mask, 30)[..., None] * RED * 0.5
    if len(bts) > 3 and t >= bts[3] - 0.1:
        u = sm((t - bts[3] + 0.1) / 0.4)
        out = out * (1 - u) + img * u
    return out


def m_drown(src, fi, t, seg):
    u = (t - seg[0]) / (DUR - seg[0])
    a = min(1.0, 0.25 + u * 1.3 + 0.3 * pulse(AU["bass"], t, 0.2))
    img = m_flood(src, fi, t, amount=a)
    if t > DUR - 1.6:
        f = sm((t - (DUR - 1.6)) / 0.5)
        img = img * (1 - f) + RED * f * 0.9
        sp = tsprite("RED", FONT["anton"], 520, pad=40)
        over_c(img, np.zeros(3, np.float32), sp * f, W / 2, H * 0.47)
        tg = RR.tracked("SEEDHE MAUT", FONT["mono"], 34, 18)
        over_c(img, np.zeros(3, np.float32), tg * f, W / 2, H * 0.47 + 330)
    return img


# ------------------------------------------------------------------ frame --
GRAIN = None
VIG = None


def render_frame(fi, src):
    global GRAIN, VIG
    t = fi / FPS
    seg = seg_at(t)
    mode = seg[2]
    if mode == "intro":
        img = m_intro(src, fi, t)
    elif mode == "flood":
        img = m_flood(src, fi, t)
    elif mode == "triptych":
        img = m_triptych(src, fi, t)
    elif mode == "sticker":
        img = m_sticker(src, fi, t)
    elif mode == "smear":
        img = m_smear(src, fi, t)
    elif mode == "halftone":
        img = m_halftone(src, fi, t)
    elif mode == "kaleido":
        img = m_kaleido(src, fi, t)
    elif mode == "mosh":
        img = m_mosh(src, fi, t, seg)
    elif mode == "strobe":
        img = m_strobe(src, fi, t)
    elif mode == "laal":
        img = knockout(src, fi, t, "लाल", "rozha", 520)
    else:
        img = m_drown(src, fi, t, seg)
    if mode != "mosh":
        MOSH.clear()
    # cut flash on mode change
    k = t - seg[0]
    if 0 < seg[0] and k < 2 / FPS:
        img = img * 0.4 + RED * 0.6
    # bass shake
    sh = pulse(AU["bass"], t, 0.06)
    if sh > 0.3 and mode not in ("intro",):
        dx = int(10 * sh * math.sin(fi * 2.3))
        img = np.roll(img, dx, 1)
    if GRAIN is None:
        GRAIN = [cv2.resize(np.random.default_rng(k).normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H)) for k in range(6)]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        VIG = np.clip(1 - 0.5 * (((xx - W / 2) / (W * 0.7)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2) ** 1.4, 0.3, 1)
    lum = img.max(-1)
    br = np.clip(lum - 0.75, 0, None)
    img = img + blur(br, 18)[..., None] * 0.35
    img = img * VIG[..., None] + GRAIN[fi % 6][..., None] * 0.025
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# -------------------------------------------------------------------- main --
def setup(src, fontdir):
    global SRC
    SRC = src
    FONT.update(anton=os.path.join(fontdir, "Anton-Regular.ttf"), rozha=os.path.join(fontdir, "RozhaOne-Regular.ttf"),
                mono=os.path.join(fontdir, "IBMPlexMono-Medium.ttf"))
    analyse(os.environ.get("RED_AUDIO", src))
    build_schedule()


def worker(job):
    k, f0, f1, out = job
    src = Src(f0 - 50)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        enc.stdin.write(render_frame(fi, src).tobytes())
        if (fi - f0) % 72 == 0:
            print(f"[w{k}] {fi}/{f1}", flush=True)
    enc.stdin.close()
    enc.wait()
    src.close()
    return out


def main():
    src, out, work, fontdir = sys.argv[1:5]
    os.makedirs(work, exist_ok=True)
    setup(src, fontdir)
    print("segments", [(round(a, 2), m) for a, b, m in SEG], flush=True)
    if "--preview" in sys.argv:
        os.makedirs(out, exist_ok=True)
        for s in sys.argv[sys.argv.index("--preview") + 1].split(","):
            t = float(s)
            fi = int(round(t * FPS))
            sr = Src(fi - 50)
            MOSH.clear()
            Image.fromarray(render_frame(fi, sr)).save(os.path.join(out, f"r_{t:05.2f}.png"))
            sr.close()
        return
    nw = int(os.environ.get("WORKERS", "4"))
    b = np.linspace(0, NF, nw + 1).astype(int)
    jobs = [(k, int(b[k]), int(b[k + 1]), os.path.join(work, f"red_chunk{k}.mp4")) for k in range(nw)]
    with mp.get_context("fork").Pool(nw) as pool:
        parts = pool.map(worker, jobs)
    lst = os.path.join(work, "red_chunks.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", src,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest",
                    "-movflags", "+faststart", out], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
