#!/usr/bin/env python3
"""
MADIRA (Seedhe Maut, prod. Hurricane, DL91 FM) - 60 s vertical reel.
"The Archive of Forgotten Moments": the stopped clock, the drop and the title,
the watch movement, the first gold drop when the vocals enter (0:48), floating
photographs, the hook cast in gold, the letter, and the held breath at 1:21.

Procedural 2.5D motion design (numpy + OpenCV + Pillow), 1080x1920 @ 24 fps,
every move keyed to the song's measured beat grid. It is the animatic/look
reference for the Blender build described in the creative-direction doc.

usage:
  python3 render_reel.py SONG_AUDIO OUT.mp4 WORKDIR FONTDIR
  python3 render_reel.py SONG_AUDIO OUTDIR  WORKDIR FONTDIR --preview 22.5,27,40
(times in --preview are SONG times in seconds)
"""
import math
import multiprocessing as mp
import os
import subprocess
import sys
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 24
REEL0, DUR = 22.0, 60.5
NF = int(round(DUR * FPS))

FONTDIR = None
ROZHA = TIRO = KALAM = KALAM_L = CORM = MONO = None


def hexc(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)


BG = hexc("#0B0A09")
CHARC = hexc("#1A1714")
GOLD = hexc("#A87B34")
CHAMP = hexc("#D9C28E")
IVORY = hexc("#EFE6D2")
BRONZE = hexc("#6E4B2A")
BURG = hexc("#4A1420")
INK = hexc("#1C120C")
DIALINK = hexc("#2B2016")

# ---------------------------------------------------------------- timing ----
FREEZES = [(23.0, 23.6), (30.8, 31.25), (34.13, 34.6), (42.9, 43.25), (46.14, 46.55),
           (70.5, 71.9), (81.0, 1e9)]
BEATS = ONSETS = HBEATS = None
B0 = 0
BAR = 1.486


def ht(ts):
    """held time: stops during freezes (dust keeps real time)"""
    return ts - sum(max(0.0, min(ts, b) - a) for a, b in FREEZES if ts > a)


def sm(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def eio(x):
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def eout(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ein(x):
    x = min(max(x, 0.0), 1.0)
    return x ** 3


def last_beat(ts):
    i = int(np.searchsorted(BEATS, ts, "right")) - 1
    if i < 0:
        return -1, 9.0
    return i, ts - BEATS[i]


def is_down(i):
    return (i - B0) % 4 == 0


def down_pulse(ts, tau=0.12):
    i, dt = last_beat(ts)
    return math.exp(-dt / tau) if i >= 0 and is_down(i) else 0.0


def bsteps(ts):
    """eased beat counter in held time: moves quickly on each beat, then rests"""
    h = ht(ts)
    i = int(np.searchsorted(HBEATS, h, "right")) - 1
    if i < 0:
        return 0.0
    nxt = HBEATS[i + 1] if i + 1 < len(HBEATS) else HBEATS[i] + 0.3715
    f = (h - HBEATS[i]) / max(1e-3, nxt - HBEATS[i])
    return i + sm(min(1.0, f * 2.4))


def onsets_between(a, b):
    return int(((ONSETS >= a) & (ONSETS < b)).sum())


# ----------------------------------------------------------------- text -----
@lru_cache(None)
def F(path, size, var=None):
    f = ImageFont.truetype(path, size)
    if var is not None:
        try:
            f.set_variation_by_axes([var])
        except Exception:
            pass
    return f


@lru_cache(None)
def tsprite(s, path, size, var=None, pad=None):
    f = F(path, size, var)
    l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), s, font=f)
    p = pad if pad is not None else max(8, size // 4)
    im = Image.new("L", (r - l + 2 * p, b - t + 2 * p), 0)
    ImageDraw.Draw(im).text((p - l, p - t), s, font=f, fill=255)
    return np.asarray(im, np.float32) / 255


def is_deva(c):
    return "ऀ" <= c <= "ॿ"


def runs(s):
    out = []
    for c in s:
        d = is_deva(c)
        if c == " " and out:
            d = out[-1][0]
        if out and out[-1][0] == d:
            out[-1][1] += c
        else:
            out.append([d, c])
    return out


@lru_cache(None)
def tag_sprite(s, size=22, track=2):
    lat = F(MONO, size)
    dev = F(TIRO, int(size * 1.2))
    w = 0
    for d, t in runs(s):
        f = dev if d else lat
        w += sum(f.getlength(c) + (0 if d else track) for c in t) if not d else f.getlength(t)
    im = Image.new("L", (int(w) + 24, int(size * 2.4)), 0)
    dr = ImageDraw.Draw(im)
    x, base = 12, int(size * 1.6)
    for d, t in runs(s):
        if d:
            dr.text((x, base), t, font=dev, fill=255, anchor="ls")
            x += dev.getlength(t)
        else:
            for c in t:
                dr.text((x, base), c, font=lat, fill=255, anchor="ls")
                x += lat.getlength(c) + track
    return np.asarray(im, np.float32) / 255


@lru_cache(None)
def tracked(s, path, size, track, var=None):
    f = F(path, size, var)
    w = sum(f.getlength(c) + track for c in s)
    im = Image.new("L", (int(w) + 30, int(size * 1.8)), 0)
    d = ImageDraw.Draw(im)
    x = 15
    for c in s:
        d.text((x, int(size * 1.3)), c, font=f, fill=255, anchor="ls")
        x += f.getlength(c) + track
    return np.asarray(im, np.float32) / 255


# ------------------------------------------------------------ compositing ---
def over(dst, rgb, a, x, y, add=False):
    h, w = a.shape
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sa = a[y0 - y:y1 - y, x0 - x:x1 - x][..., None]
    src = rgb[y0 - y:y1 - y, x0 - x:x1 - x] if (hasattr(rgb, "ndim") and rgb.ndim == 3) else np.asarray(rgb, np.float32)
    if add:
        dst[y0:y1, x0:x1] += src * sa
    else:
        dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - sa) + src * sa


def over_c(dst, rgb, a, cx, cy, add=False):
    over(dst, rgb, a, cx - a.shape[1] / 2, cy - a.shape[0] / 2, add)


def affine(cx, cy, sx, sy, rot, w, h):
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    A = np.array([[sx * c, -sy * s], [sx * s, sy * c]], np.float32)
    t = np.array([cx, cy], np.float32) - A @ np.array([w / 2, h / 2], np.float32)
    return np.hstack([A, t[:, None]]).astype(np.float32)


def draw_sprite(dst, rgb, a, M, add=False, mul=1.0):
    h, w = a.shape
    cs = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float32) @ M.T
    x0, y0 = max(0, int(math.floor(cs[:, 0].min()))), max(0, int(math.floor(cs[:, 1].min())))
    x1, y1 = min(dst.shape[1], int(math.ceil(cs[:, 0].max()))), min(dst.shape[0], int(math.ceil(cs[:, 1].max())))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return
    M2 = M.copy()
    M2[:, 2] -= [x0, y0]
    a2 = cv2.warpAffine(a, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=0) * mul
    if hasattr(rgb, "ndim") and rgb.ndim == 3:
        r2 = cv2.warpAffine(rgb, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        r2 = np.asarray(rgb, np.float32)
    reg = dst[y0:y1, x0:x1]
    if add:
        reg += r2 * a2[..., None]
    else:
        reg[:] = reg * (1 - a2[..., None]) + r2 * a2[..., None]


def blur(img, sigma):
    if sigma < 6:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    f = 4 if sigma >= 16 else 2
    hh, ww = img.shape[:2]
    small = cv2.resize(img, (max(1, ww // f), max(1, hh // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (ww, hh), interpolation=cv2.INTER_LINEAR)


def metal(mask, bevel=6.0, strength=1.0, light=(-0.45, -0.75, 0.55), lo=None, mid=None, hi=None,
          spec=26, rim=0.35):
    lo = BRONZE * 0.28 if lo is None else lo
    mid = GOLD if mid is None else mid
    hi = CHAMP if hi is None else hi
    mask = np.ascontiguousarray(mask, dtype=np.float32)
    hgt = cv2.GaussianBlur(mask, (0, 0), bevel)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3)
    k = strength * bevel * 0.55
    nx, ny = -gx * k, -gy * k
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    nx, ny, nz = nx * inv, ny * inv, inv
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    d = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    Hv = L + np.array([0, 0, 1], np.float32)
    Hv /= np.linalg.norm(Hv)
    s = np.clip(nx * Hv[0] + ny * Hv[1] + nz * Hv[2], 0, 1) ** spec
    rr = np.clip(nx * 0.7 + ny * 0.55, 0, 1) ** 3 * rim
    t = d[..., None]
    col = lo * (1 - t) + mid * t
    col = col + hi * (s[..., None] * 1.25) + np.array([0.7, 0.72, 0.75], np.float32) * rr[..., None] * 0.5
    return col.astype(np.float32)


def crop_metal(mask, **kw):
    """shade only the bounding box of a full-frame mask"""
    ys, xs = np.nonzero(mask > 0.002)
    out = np.zeros(mask.shape + (3,), np.float32)
    if len(ys) == 0:
        return out
    pad = int(kw.get("bevel", 6) * 3) + 4
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    out[y0:y1, x0:x1] = metal(mask[y0:y1, x0:x1], **kw)
    return out


# --------------------------------------------------------------- textures ---
def noise2(h, w, scale, seed):
    r = np.random.default_rng(seed)
    small = r.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.0), (w, h), interpolation=cv2.INTER_CUBIC)


def aged_paper(w, h, seed, tone=None, edge=18):
    tone = IVORY if tone is None else tone
    r = np.random.default_rng(seed)
    low = noise2(h, w, 24, seed)
    mid = noise2(h, w, 6, seed + 1)
    rgb = tone * 0.93 * np.ones((h, w, 3), np.float32)
    stain = np.clip(low - 0.55, 0, 1)[..., None] * np.array([0.0, 0.10, 0.22], np.float32)
    rgb = rgb * (1 - 0.06 * mid[..., None]) - stain * 0.6
    fox = np.zeros((h, w), np.float32)
    for _ in range(int(w * h / 9000)):
        cv2.circle(fox, (int(r.integers(0, w)), int(r.integers(0, h))), int(r.integers(1, 5)), float(r.uniform(0.3, 1)), -1)
    fox = cv2.GaussianBlur(fox, (0, 0), 1.6)
    rgb = rgb - fox[..., None] * np.array([0.10, 0.16, 0.22], np.float32)
    fib = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(fib)
    for _ in range(int(w * h / 1500)):
        x, y = r.integers(0, w), r.integers(0, h)
        a = r.uniform(0, math.pi)
        ln = r.uniform(4, 16)
        d.line([(x, y), (x + ln * math.cos(a), y + ln * math.sin(a))], fill=int(r.integers(20, 60)))
    rgb = rgb - (np.asarray(fib, np.float32) / 255)[..., None] * 0.05
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy])
    rgb = rgb * (1 - 0.32 * np.exp(-dist / edge))[..., None] - np.exp(-dist / 6)[..., None] * np.array([0.02, 0.05, 0.08])
    jag = noise2(h, w, 3, seed + 7)
    alpha = np.clip((dist - 1.5 - 3.0 * jag) / 1.5, 0, 1)
    return np.clip(rgb, 0, 1).astype(np.float32), alpha.astype(np.float32)


def sepia(v):
    v = v[..., None]
    return np.clip(v * np.array([1.0, 0.86, 0.66], np.float32) + (1 - v) * np.array([0.05, 0.035, 0.025], np.float32), 0, 1)


def mem_image(kind, seed, w=400, h=500):
    r = np.random.default_rng(seed)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    glow = Image.new("L", (w, h), 0)
    g = ImageDraw.Draw(glow)

    def vgrad(y0, y1, a, b):
        for y in range(int(y0), int(y1)):
            d.line([(0, y), (w, y)], fill=int(a + (b - a) * (y - y0) / max(1, y1 - y0)))

    def person(x, yh, s=1.0, fill=12):
        d.ellipse([x - 15 * s, yh - 15 * s, x + 15 * s, yh + 15 * s], fill=fill)
        d.polygon([(x - 26 * s, yh + 26 * s), (x + 26 * s, yh + 26 * s), (x + 18 * s, yh + 150 * s), (x - 18 * s, yh + 150 * s)], fill=fill)
        d.rectangle([x - 6 * s, yh + 12 * s, x + 6 * s, yh + 28 * s], fill=fill)

    if kind == "terrace":
        vgrad(0, h * 0.56, 95, 205)
        vgrad(h * 0.56, h, 40, 22)
        for _ in range(70):
            x = r.uniform(0, w)
            y = r.uniform(h * 0.53, h * 0.57)
            g.ellipse([x - 2, y - 2, x + 2, y + 2], fill=255)
        person(w * 0.34, h * 0.40)
        person(w * 0.68, h * 0.41, 0.96)
        d.rectangle([0, h * 0.60, w, h * 0.625], fill=22)
        for x in range(0, w, 20):
            d.rectangle([x, h * 0.625, x + 6, h * 0.82], fill=22)
        d.rectangle([0, h * 0.82, w, h * 0.84], fill=22)
    elif kind == "cups":
        vgrad(0, h, 70, 40)
        d.polygon([(0, h * 0.55), (w, h * 0.55), (w, h), (0, h)], fill=150)
        for cx in (w * 0.34, w * 0.64):
            d.ellipse([cx - 62, h * 0.70 - 14, cx + 62, h * 0.70 + 14], fill=200)
            d.polygon([(cx - 38, h * 0.56), (cx + 38, h * 0.56), (cx + 30, h * 0.69), (cx - 30, h * 0.69)], fill=225)
            d.ellipse([cx - 38, h * 0.56 - 9, cx + 38, h * 0.56 + 9], fill=240)
            d.ellipse([cx - 32, h * 0.56 - 6, cx + 32, h * 0.56 + 6], fill=55)
            d.arc([cx + 26, h * 0.585, cx + 54, h * 0.655], -90, 90, fill=215, width=6)
            for k in range(3):
                pts = [(cx - 12 + k * 12 + 8 * math.sin(y / 14 + k), y) for y in range(int(h * 0.30), int(h * 0.53), 4)]
                g.line(pts, fill=150, width=3)
    elif kind == "chair":
        vgrad(0, h * 0.72, 60, 45)
        vgrad(h * 0.72, h, 70, 30)
        d.rectangle([w * 0.52, h * 0.12, w * 0.9, h * 0.55], fill=240)
        d.rectangle([w * 0.70, h * 0.12, w * 0.72, h * 0.55], fill=60)
        d.rectangle([w * 0.52, h * 0.33, w * 0.9, h * 0.35], fill=60)
        g.polygon([(w * 0.52, h * 0.72), (w * 0.9, h * 0.72), (w * 0.75, h), (w * 0.1, h)], fill=130)
        x = w * 0.32
        d.rectangle([x - 50, h * 0.40, x - 40, h * 0.86], fill=14)
        d.rectangle([x - 50, h * 0.62, x + 50, h * 0.66], fill=14)
        d.rectangle([x + 40, h * 0.62, x + 50, h * 0.86], fill=14)
    elif kind == "door":
        vgrad(0, h, 34, 22)
        d.rectangle([w * 0.30, h * 0.10, w * 0.70, h * 0.80], outline=60, width=8)
        d.polygon([(w * 0.47, h * 0.12), (w * 0.55, h * 0.12), (w * 0.55, h * 0.79), (w * 0.47, h * 0.79)], fill=250)
        g.polygon([(w * 0.47, h * 0.80), (w * 0.55, h * 0.80), (w * 0.85, h), (w * 0.20, h)], fill=150)
    elif kind == "lamp":
        vgrad(0, h, 30, 18)
        g.polygon([(w * 0.60, h * 0.22), (w * 0.35, h * 0.95), (w * 0.95, h * 0.95)], fill=150)
        d.rectangle([w * 0.60 - 4, h * 0.20, w * 0.60 + 4, h * 0.97], fill=8)
        d.rectangle([w * 0.60 - 16, h * 0.18, w * 0.60 + 16, h * 0.22], fill=250)
        for _ in range(160):
            x, y = r.uniform(0, w), r.uniform(0, h)
            g.line([(x, y), (x - 5, y + 18)], fill=int(r.integers(70, 160)), width=1)
        g.rectangle([w * 0.58, h * 0.95, w * 0.62, h], fill=200)
    elif kind == "stairs":
        vgrad(0, h, 80, 50)
        d.rectangle([w * 0.62, h * 0.08, w * 0.88, h * 0.32], fill=240)
        pts = [(0, h)]
        for k in range(9):
            x0, y0 = k * w / 9, h - k * h * 0.07
            pts += [(x0, y0 - h * 0.07), (x0 + w / 9, y0 - h * 0.07)]
        pts += [(w, h)]
        d.polygon(pts, fill=16)
    elif kind == "shadows":
        for x in range(w):
            d.line([(x, 0), (x, h)], fill=int(190 - 90 * x / w))
        d.rectangle([0, h * 0.86, w, h], fill=80)
        person(w * 0.32, h * 0.36, 1.1, fill=70)
        person(w * 0.70, h * 0.37, 1.05, fill=70)
    elif kind == "sea":
        vgrad(0, h * 0.5, 120, 215)
        vgrad(h * 0.5, h, 75, 35)
        for _ in range(60):
            x = w / 2 + r.normal(0, 22)
            y = r.uniform(h * 0.5, h * 0.8)
            g.line([(x - 8, y), (x + 8, y)], fill=230, width=2)
        d.rectangle([w * 0.48, h * 0.60, w * 0.52, h * 0.70], fill=10)
        d.ellipse([w * 0.48, h * 0.575, w * 0.52, h * 0.605], fill=10)
        d.polygon([(0, h * 0.86), (w, h * 0.80), (w, h), (0, h)], fill=26)
    v = np.asarray(im, np.float32) / 255
    gl = cv2.GaussianBlur(np.asarray(glow, np.float32) / 255, (0, 0), 6)
    v = np.clip(cv2.GaussianBlur(v, (0, 0), 1.1) + gl * 0.8, 0, 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vig = 1 - 0.5 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2) * 2
    v = v * vig + np.random.default_rng(seed + 3).normal(0, 0.03, (h, w)).astype(np.float32)
    rgb = sepia(np.clip(v, 0, 1))
    for _ in range(6):
        x = int(r.integers(0, w))
        rgb[:, x:x + 1] *= 0.88
    return rgb


def make_card(kind, seed, back_text=None):
    iw, ih, b = 400, 500, 20
    w, h = iw + 2 * b, ih + b + 52
    paper, alpha = aged_paper(w, h, seed, edge=10)
    front = paper.copy()
    front[b:b + ih, b:b + iw] = mem_image(kind, seed)
    back, _ = aged_paper(w, h, seed + 50, edge=14)
    back = back * 0.95
    pen = tsprite("1:48", KALAM_L, 26)
    over(back, INK * 1.5, pen * 0.45, w - pen.shape[1] - 20, h - pen.shape[0] - 14)
    if back_text:
        t = tsprite(back_text, TIRO, 92)
        over(back, INK, t * 0.88, (w - t.shape[1]) / 2, (h - t.shape[0]) / 2 - 20)
    return dict(front=front, back=back, a=alpha, w=w, h=h)


def rose_sprite(seed=11):
    S = 520
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    r = np.random.default_rng(seed)
    cx, cy = S / 2, S * 0.36
    layers = []
    for k, (rad, n) in enumerate([(120, 6), (95, 5), (70, 5), (46, 4), (26, 3)]):
        lay = Image.new("L", (S, S), 0)
        ld = ImageDraw.Draw(lay)
        for i in range(n):
            a = 2 * math.pi * i / n + k * 0.7 + r.uniform(-0.2, 0.2)
            px, py = cx + math.cos(a) * rad * 0.42, cy + math.sin(a) * rad * 0.32
            ld.ellipse([px - rad * 0.62, py - rad * 0.48, px + rad * 0.62, py + rad * 0.48], fill=255)
        layers.append(np.asarray(lay, np.float32) / 255)
    stem = Image.new("L", (S, S), 0)
    sd = ImageDraw.Draw(stem)
    sd.line([(cx, cy + 60), (cx + 14, S * 0.75), (cx + 4, S - 10)], fill=255, width=9)
    sd.ellipse([cx + 14, S * 0.62, cx + 90, S * 0.68], fill=255)
    stem = cv2.GaussianBlur(np.asarray(stem, np.float32) / 255, (0, 0), 0.8)
    rgb = metal(stem, bevel=3, lo=np.array([0.02, 0.02, 0.01]), mid=np.array([0.10, 0.08, 0.04]), hi=CHAMP * 0.2, spec=10) * stem[..., None]
    alpha = stem.copy()
    for k, lay in enumerate(layers):
        lay = cv2.GaussianBlur(lay, (0, 0), 0.8)
        shade = metal(lay, bevel=9, strength=1.6, lo=BURG * 0.18, mid=BURG * (1.25 + 0.15 * k),
                      hi=np.array([0.55, 0.22, 0.26], np.float32), spec=6, rim=0.15)
        edge = np.clip(lay - cv2.erode(lay, np.ones((5, 5), np.uint8)), 0, 1)
        shade = shade * (1 - 0.5 * edge[..., None])
        rgb = rgb * (1 - lay[..., None]) + shade * lay[..., None]
        alpha = np.maximum(alpha, lay)
    return rgb.astype(np.float32), alpha.astype(np.float32)


# ------------------------------------------------------------------ assets --
A = {}


def gear_mask(r, teeth, spokes, seed=0):
    ss = 2
    S = int(2 * r + 12)
    im = Image.new("L", (S * ss, S * ss), 0)
    d = ImageDraw.Draw(im)
    c = S * ss / 2
    R, Rr = r * ss, r * ss * 0.9
    pts = []
    for i in range(teeth):
        a0 = 2 * math.pi * i / teeth
        p = 2 * math.pi / teeth
        for f, rad in ((0.0, Rr), (0.22, R), (0.52, R), (0.74, Rr)):
            pts.append((c + rad * math.cos(a0 + f * p), c + rad * math.sin(a0 + f * p)))
    d.polygon(pts, fill=255)
    ri = r * ss * 0.74
    d.ellipse([c - ri, c - ri, c + ri, c + ri], fill=0)
    hub = r * ss * 0.2
    for k in range(spokes):
        a = 2 * math.pi * k / spokes
        dx, dy = math.cos(a), math.sin(a)
        wv = r * ss * 0.055
        d.polygon([(c + dx * hub - dy * wv, c + dy * hub + dx * wv), (c + dx * ri * 1.02 - dy * wv * 0.7, c + dy * ri * 1.02 + dx * wv * 0.7),
                   (c + dx * ri * 1.02 + dy * wv * 0.7, c + dy * ri * 1.02 - dx * wv * 0.7), (c + dx * hub + dy * wv, c + dy * hub - dx * wv)], fill=255)
    d.ellipse([c - hub, c - hub, c + hub, c + hub], fill=255)
    ring = r * ss * 0.13
    d.ellipse([c - ring, c - ring, c + ring, c + ring], fill=110)
    hole = r * ss * 0.06
    d.ellipse([c - hole, c - hole, c + hole, c + hole], fill=0)
    m = np.asarray(im.resize((S, S), Image.LANCZOS), np.float32) / 255
    return m


def build_watch():
    R, S = 360, 1000
    cx, cy = 500, 540
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    rr = np.hypot(xx - cx, yy - cy)
    case = np.clip(R - rr, 0, 1) * np.clip(rr - (R - 46), 0, 1)
    rb = np.hypot(xx - cx, yy - (cy - R - 74))
    bow = np.clip(58 - rb, 0, 1) * np.clip(rb - 40, 0, 1)
    stem = ((np.abs(xx - cx) < 20) & (yy > cy - R - 34) & (yy < cy - R + 6)).astype(np.float32)
    crown = ((np.abs(xx - cx) < 36) & (yy > cy - R - 58) & (yy < cy - R - 28)).astype(np.float32)
    crown *= 0.7 + 0.3 * (np.sin(xx * 1.3) > 0)
    mm = np.maximum.reduce([case, bow, stem, crown])
    mm = cv2.GaussianBlur(mm, (0, 0), 0.6)
    mrgb = metal(mm, bevel=8, strength=1.4)
    Rd = R - 46
    dial_m = np.clip(Rd - rr, 0, 1)
    dial = IVORY * (0.98 - 0.30 * (rr / Rd) ** 2)[..., None] * np.array([1.0, 0.97, 0.92], np.float32)
    cr = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(cr)
    r = np.random.default_rng(4)
    for _ in range(170):
        x, y = r.uniform(cx - Rd, cx + Rd), r.uniform(cy - Rd, cy + Rd)
        pts = [(x, y)]
        for _ in range(4):
            x += r.uniform(-30, 30)
            y += r.uniform(-30, 30)
            pts.append((x, y))
        d.line(pts, fill=255, width=1)
    craze = cv2.GaussianBlur(np.asarray(cr, np.float32) / 255, (0, 0), 0.6)
    dial = dial * (1 - 0.10 * craze[..., None])
    mk = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(mk)
    for rad in (R - 74, R - 92):
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=255, width=2)
    for k in range(60):
        a = 2 * math.pi * k / 60
        r0 = R - 92 if k % 5 else R - 100
        d.line([(cx + math.sin(a) * r0, cy - math.cos(a) * r0), (cx + math.sin(a) * (R - 74), cy - math.cos(a) * (R - 74))],
               fill=255, width=4 if k % 5 == 0 else 2)
    lab = tracked("ARCHIVE", MONO, 18, 5)
    mk_np = np.asarray(mk, np.float32) / 255
    lab_canvas = np.zeros((S, S), np.float32)
    lab_canvas[int(cy + R * 0.30):int(cy + R * 0.30) + lab.shape[0], int(cx - lab.shape[1] / 2):int(cx - lab.shape[1] / 2) + lab.shape[1]] = lab
    no = tracked("No 048", MONO, 16, 3)
    lab_canvas[int(cy + R * 0.40):int(cy + R * 0.40) + no.shape[0], int(cx - no.shape[1] / 2):int(cx - no.shape[1] / 2) + no.shape[1]] = no
    marks = np.maximum(cv2.GaussianBlur(mk_np, (0, 0), 0.5), lab_canvas * 0.8)
    dial = dial * (1 - marks[..., None]) + DIALINK * marks[..., None]
    dial = dial * (1 - 0.38 * np.clip(1 - (Rd - rr) / 22, 0, 1))[..., None]
    numerals = []
    roman = ["XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
    dial_num = dial.copy()
    for k, nm in enumerate(roman):
        a = 2 * math.pi * k / 12
        x, y = cx + math.sin(a) * (R - 150), cy - math.cos(a) * (R - 150)
        sp = tsprite(nm, CORM, 64, 650)
        over(dial_num, DIALINK, sp, x - sp.shape[1] / 2, y - sp.shape[0] / 2)
        numerals.append((sp, x, y))
    rgb_base = mrgb * mm[..., None] + dial * dial_m[..., None] * (1 - mm[..., None])
    rgb_num = mrgb * mm[..., None] + dial_num * dial_m[..., None] * (1 - mm[..., None])
    alpha = np.maximum(mm, dial_m)
    hl = np.exp(-((rr - (R - 110)) / 40) ** 2) * np.clip((-(xx - cx) - (yy - cy)) / R, 0, 1) * 0.12 * dial_m
    rgb_num = rgb_num + hl[..., None]
    rgb_base = rgb_base + hl[..., None]
    # seam
    r = np.random.default_rng(9)
    ys = np.linspace(cy - Rd + 20, cy + Rd - 20, 16)
    xs = cx + np.cumsum(r.uniform(-16, 16, 16)) * 0.8
    xs = xs - (xs.mean() - cx)
    seam = list(zip(xs, ys))
    seam = [(cx + 6, cy - R - 4)] + seam + [(cx - 4, cy + R + 4)]
    sx = np.interp(np.arange(S), [p[1] for p in seam], [p[0] for p in seam])
    left = (xx < sx[:, None]).astype(np.float32)
    A["watch"] = dict(R=R, S=S, cx=cx, cy=cy, rgb_num=rgb_num.astype(np.float32), rgb=rgb_base.astype(np.float32),
                      alpha=alpha.astype(np.float32), numerals=numerals, seam=seam, left=left)
    A["numerals_gold"] = [(metal(cv2.GaussianBlur(sp, (0, 0), 0.5), bevel=3, strength=1.6), sp) for sp, _, _ in numerals]


def fit_sprite(txt, path, maxw, size, pad):
    while True:
        s = tsprite(txt, path, size, pad=pad)
        if s.shape[1] - 2 * pad <= maxw or size < 40:
            return s
        size -= 10


def build_title():
    s = fit_sprite("मदिरा", ROZHA, 800, 330, 60)
    m = np.zeros((H, W), np.float32)
    cx, cy = W // 2, int(H * 0.46)
    y0, x0 = cy - s.shape[0] // 2, cx - s.shape[1] // 2
    m[y0:y0 + s.shape[0], x0:x0 + s.shape[1]] = s
    ys, xs = np.nonzero(m > 0.5)
    bb = (xs.min(), ys.min(), xs.max(), ys.max())
    inner = cv2.erode(m, np.ones((7, 7), np.uint8))
    edge = np.clip(m - inner, 0, 1)
    # counter of the first glyph (म): enclosed background holes
    b = (m > 0.5).astype(np.uint8)
    inv = (1 - b).astype(np.uint8)
    ff = inv.copy()
    mask = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 2)
    holes = (ff == 1).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(holes)
    best, bx = None, 1e9
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 300 and cents[i][0] < bx:
            best, bx = cents[i], cents[i][0]
    if best is None:
        best = (bb[0] + (bb[2] - bb[0]) * 0.12, (bb[1] + bb[3]) / 2)
    A["title"] = dict(m=m, inner=inner, edge=edge, bb=bb, counter=(float(best[0]), float(best[1])))
    sub = tracked("MADIRA", CORM, 44, 26, 500)
    A["title_sub"] = sub


def build_word():
    s = fit_sprite("मदिरा", ROZHA, 940, 380, 70)
    m = np.zeros((H, W), np.float32)
    cx, cy = W // 2, int(H * 0.47)
    y0, x0 = cy - s.shape[0] // 2, cx - s.shape[1] // 2
    m[y0:y0 + s.shape[0], x0:x0 + s.shape[1]] = s
    ys, xs = np.nonzero(m > 0.5)
    A["word"] = dict(m=m, bb=(xs.min(), ys.min(), xs.max(), ys.max()),
                     rgb=crop_metal(m, bevel=9, strength=1.3))
    r = np.random.default_rng(21)
    idx = r.choice(len(xs), 2600, replace=False)
    A["word_pts"] = np.stack([xs[idx], ys[idx]], 1).astype(np.float32)


def build_gears():
    # (world x, world y, radius, teeth, depth, dir, spokes)
    spec = [
        (W * 0.16, -520, 520, 72, 0.6, 1, 6),
        (W * 0.30, 260, 430, 60, 1.0, 1, 5),
        (W * 0.86, -190, 300, 42, 1.0, -1, 4),
        (W * 1.02, 620, 380, 54, 1.35, -1, 5),
        (W * 0.02, -1320, 360, 50, 1.3, 1, 5),
        (W * 0.94, -1620, 280, 40, 0.8, -1, 4),
        (W * 0.30, -2520, 400, 56, 0.7, 1, 6),
        (W * 1.00, -2250, 600, 84, 1.4, -1, 6),
        (W * 0.75, -980, 210, 30, 0.75, 1, 3),
    ]
    A["gears"] = [dict(x=x, y=y, r=r, n=n, depth=dp, dir=dr, mask=gear_mask(r, n, sp))
                  for x, y, r, n, dp, dr, sp in spec]
    r = np.random.default_rng(5)
    A["screws"] = [(r.uniform(0, W), r.uniform(-2800, 900), r.uniform(0.6, 1.4), r.uniform(0, 6.28)) for _ in range(30)]


def build_cards():
    kinds = ["terrace", "cups", "chair", "door", "lamp", "stairs", "shadows", "sea"]
    A["cards"] = {k: make_card(k, 100 + i) for i, k in enumerate(kinds)}
    A["card_yaad"] = make_card("cups", 140, back_text="याद")
    A["card_wait"] = make_card("terrace", 100)
    A["rose"] = rose_sprite()
    sph = np.zeros((64, 64), np.float32)
    cv2.circle(sph, (32, 32), 28, 1.0, -1)
    sph = cv2.GaussianBlur(sph, (0, 0), 1.2)
    A["sphere"] = (metal(sph, bevel=7, strength=2.0), sph)
    A["ink_sphere"] = (metal(sph, bevel=7, strength=2.0, lo=INK * 0.5, mid=INK * 1.6, hi=np.array([0.6, 0.55, 0.5]), spec=40), sph)


def build_letter():
    w, h = 700, 960
    paper, alpha = aged_paper(w, h, 77, edge=22)
    back, _ = aged_paper(w, h, 78, edge=22)
    lines = ["तुम्हारे लिए,", "वक़्त वहीं रुका है,", "जहाँ तुम गए थे।"]
    sprites = [tsprite(t, KALAM, 58, pad=10) for t in lines]
    date = tsprite("12:48", KALAM_L, 34, pad=6)
    A["letter"] = dict(w=w, h=h, paper=paper, alpha=alpha, back=back * 0.82, lines=sprites, date=date)
    seal = np.zeros((140, 140), np.float32)
    cv2.circle(seal, (70, 70), 56, 1.0, -1)
    st = tsprite("म", ROZHA, 64, pad=4)
    seal_st = seal.copy()
    y0, x0 = 70 - st.shape[0] // 2, 70 - st.shape[1] // 2
    seal_st[y0:y0 + st.shape[0], x0:x0 + st.shape[1]] -= st * 0.35
    seal_st = cv2.GaussianBlur(np.clip(seal_st, 0, 1), (0, 0), 1.0)
    A["seal"] = (metal(seal_st, bevel=4, strength=1.8, mid=BRONZE * 1.4), cv2.GaussianBlur(seal, (0, 0), 1.0))


def make_cone(cx, spread, top=-120):
    q = 4
    w, h = W // q, H // q
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon([(cx / q - 10, top / q), (cx / q + 10, top / q), (cx / q + spread / q, h), (cx / q - spread / q, h)], fill=255)
    m = cv2.GaussianBlur(np.asarray(im, np.float32) / 255, (0, 0), 16)
    m = m * (1 - 0.6 * np.linspace(0, 1, h)[:, None])
    return m


VIG = GRAIN = CONE = CONEQ = None
DUST = None


def init(audio, fontdir):
    global FONTDIR, ROZHA, TIRO, KALAM, KALAM_L, CORM, MONO, BEATS, ONSETS, HBEATS, B0, VIG, GRAIN, CONE, CONEQ, DUST
    FONTDIR = fontdir
    j = lambda n: os.path.join(fontdir, n)
    ROZHA, TIRO = j("RozhaOne-Regular.ttf"), j("TiroDevanagariHindi-Regular.ttf")
    KALAM, KALAM_L = j("Kalam-Regular.ttf"), j("Kalam-Light.ttf")
    CORM, MONO = j("CormorantGaramond[wght].ttf"), j("IBMPlexMono-Regular.ttf")
    import librosa
    y, sr = librosa.load(audio, sr=22050)
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    _, bf = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
    BEATS = librosa.frames_to_time(bf, sr=sr)
    ONSETS = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, units="time")
    B0 = int(np.argmin(np.abs(BEATS - 23.64)))
    HBEATS = np.array([ht(b) for b in BEATS])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    VIG = np.clip(1 - 0.62 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.47) / (H * 0.62)) ** 2) ** 1.4, 0.2, 1).astype(np.float32)
    GRAIN = [cv2.resize(np.random.default_rng(k).normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H)) for k in range(6)]
    CONEQ = make_cone(W / 2, 560)
    CONE = cv2.resize(CONEQ, (W, H))
    r = np.random.default_rng(3)
    nd = 420
    DUST = dict(x=r.uniform(0, W, nd), y=r.uniform(0, H + 200, nd), z=r.uniform(0.35, 1.7, nd),
                s=r.uniform(6, 20, nd), p=r.uniform(0, 6.28, nd))
    build_watch()
    build_title()
    build_word()
    build_gears()
    build_cards()
    build_letter()


# ------------------------------------------------------------------- post ---
def dust_layer(ts, amt):
    q = 2
    acc = np.zeros((H // q, W // q), np.float32)
    D = DUST
    y = (D["y"] + ts * D["s"] / D["z"]) % (H + 200) - 100
    x = D["x"] + 18 * np.sin(ts * 0.35 + D["p"]) / D["z"]
    rad = np.clip(1.5 / D["z"], 1, 5)
    val = np.clip(0.5 / D["z"], 0.08, 1.0) * (0.75 + 0.25 * np.sin(ts * 2 + D["p"]))
    for xi, yi, ri, vi in zip(x / q, y / q, rad / q, val):
        cv2.circle(acc, (int(xi), int(yi)), max(1, int(round(ri))), float(vi), -1)
    acc = cv2.GaussianBlur(acc, (0, 0), 0.9)
    up = cv2.resize(acc, (W, H))
    return (up * (0.18 + 1.0 * CONE) * amt)[..., None] * CHAMP


def post(img, ts, fi, dust=0.7):
    img = img + dust_layer(ts, dust)
    lum = img.max(-1)
    br = np.clip(lum - 0.62, 0, None)
    b1, b2, b3 = blur(br, 14), blur(br, 48), blur(br, 4)
    img = img + (b1 * 0.5 + b2 * 0.45)[..., None] * np.array([1.0, 0.80, 0.55], np.float32) \
        + b3[..., None] * np.array([0.30, 0.10, 0.04], np.float32)
    img = img * VIG[..., None]
    img = img + BG * 0.7 * np.clip(1 - img.max(-1, keepdims=True) * 4, 0, 1)
    img = np.where(img > 0.82, 0.82 + 0.18 * np.tanh((img - 0.82) / 0.18), img)
    img = img + GRAIN[fi % 6][..., None] * 0.018
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def tag(img, s, cx, cy, alpha=0.6, size=22):
    sp = tag_sprite(s, size)
    over_c(img, IVORY, sp * alpha, cx, cy)


def canvas(spot=1.0):
    img = np.empty((H, W, 3), np.float32)
    img[:] = BG
    if spot:
        img += CONE[..., None] * np.array([0.16, 0.12, 0.07], np.float32) * spot
    return img


# ----------------------------------------------------------------- scenes ---
def hands_layer(sec, minute, hour, S, cx, cy, R):
    ss = 2
    im = Image.new("L", (S * ss, S * ss), 0)
    d = ImageDraw.Draw(im)
    sec_im = Image.new("L", (S * ss, S * ss), 0)
    sd = ImageDraw.Draw(sec_im)

    def hand(dr, ang, length, wid, tail=0.0):
        a = math.radians(ang)
        dx, dy = math.sin(a), -math.cos(a)
        px, py = -dy, dx
        c = (cx * ss, cy * ss)
        tip = (c[0] + dx * length * ss, c[1] + dy * length * ss)
        tl = (c[0] - dx * tail * ss, c[1] - dy * tail * ss)
        dr.polygon([(tl[0] + px * wid * ss * 0.6, tl[1] + py * wid * ss * 0.6), (c[0] + px * wid * ss, c[1] + py * wid * ss),
                    tip, (c[0] - px * wid * ss, c[1] - py * wid * ss), (tl[0] - px * wid * ss * 0.6, tl[1] - py * wid * ss * 0.6)], fill=255)

    hand(d, hour * 30, R * 0.48, 12, 30)
    hand(d, minute * 6, R * 0.76, 8, 40)
    hand(sd, sec * 6, R * 0.84, 2.5, 70)
    a = math.radians(sec * 6)
    sd.ellipse([(cx - math.sin(a) * 52 - 9) * ss, (cy + math.cos(a) * 52 - 9) * ss, (cx - math.sin(a) * 52 + 9) * ss, (cy + math.cos(a) * 52 + 9) * ss], fill=255)
    d.ellipse([(cx - 16) * ss, (cy - 16) * ss, (cx + 16) * ss, (cy + 16) * ss], fill=255)
    hm = np.asarray(im.resize((S, S), Image.LANCZOS), np.float32) / 255
    sm_ = np.asarray(sec_im.resize((S, S), Image.LANCZOS), np.float32) / 255
    return hm, sm_


def scene_watch(ts, fi):
    w = A["watch"]
    S, cx, cy, R = w["S"], w["cx"], w["cy"], w["R"]
    img = canvas(1.0)
    beats = [b for b in BEATS if 22.0 <= b <= 22.95]
    sec = 60 - sum(1 for b in beats if b > ts)
    i, dt = last_beat(ts)
    if ts < 22.95 and dt < 0.08:
        sec -= 0.25 * (1 - dt / 0.08)
    minute = 47 + sec / 60
    hour = 12 + minute / 60
    hm, smk = hands_layer(sec, minute, hour, S, cx, cy, R)
    spr = w["rgb_num"].copy()
    sh = cv2.GaussianBlur(np.maximum(hm, smk), (0, 0), 6)
    spr = spr * (1 - 0.35 * np.roll(np.roll(sh, 10, 0), 6, 1)[..., None] * w["alpha"][..., None])
    hrgb = metal(hm, bevel=2.5, strength=1.3, lo=np.array([0.03, 0.02, 0.015]), mid=np.array([0.17, 0.11, 0.06]), hi=CHAMP * 0.8)
    spr = spr * (1 - hm[..., None]) + hrgb * hm[..., None]
    srgb = metal(smk, bevel=1.5, strength=1.3)
    spr = spr * (1 - smk[..., None]) + srgb * smk[..., None]
    # crack along the seam during the inhale
    if ts >= 23.0:
        p = eout((ts - 23.0) / 0.55)
        crk = Image.new("L", (S, S), 0)
        pts = w["seam"]
        seg = [math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]
        tot = sum(seg) * p
        dr = ImageDraw.Draw(crk)
        for k in range(len(pts) - 1):
            if tot <= 0:
                break
            q = min(1, tot / seg[k])
            tot -= seg[k]
            b = (pts[k][0] + (pts[k + 1][0] - pts[k][0]) * q, pts[k][1] + (pts[k + 1][1] - pts[k][1]) * q)
            dr.line([pts[k], b], fill=255, width=3)
        cm = cv2.GaussianBlur(np.asarray(crk, np.float32) / 255, (0, 0), 0.7)
        spr = spr * (1 - cm[..., None]) + CHAMP * 1.15 * cm[..., None] + blur(cm, 8)[..., None] * GOLD * 0.9
    sc = 1.0 + 0.035 * eio((ht(ts) - 22.0) / 1.0)
    M = affine(W / 2, H * 0.46, sc, sc, 0, S, S)
    draw_sprite(img, spr, w["alpha"], M)
    a = sm((ts - 22.15) / 0.4)
    tag(img, "ARCHIVE № 022 · 12:48 · इंतज़ार", W / 2, H * 0.80, 0.55 * a)
    return img


def door_warp(rgb, a, left, theta, hinge_x, seam_x, y_top, y_bot, cyv):
    k = 1 + 0.35 * math.sin(theta)
    fx = hinge_x + (seam_x - hinge_x) * math.cos(theta)
    src = np.float32([[hinge_x, y_top], [seam_x, y_top], [seam_x, y_bot], [hinge_x, y_bot]])
    dst = np.float32([[hinge_x, y_top], [fx, cyv - (cyv - y_top) * k], [fx, cyv + (y_bot - cyv) * k], [hinge_x, y_bot]])
    P = cv2.getPerspectiveTransform(src, dst)
    shp = (rgb.shape[1], rgb.shape[0])
    return cv2.warpPerspective(rgb, P, shp), cv2.warpPerspective(a * left, P, shp)


def title_layer(img, ts, t, fill):
    T = A["title"]
    m, inner, edge, (x0, y0, x1, y1) = T["m"], T["inner"], T["edge"], T["bb"]
    hgt = y1 - y0
    surf_base = y1 - fill * hgt
    xs = np.arange(W, dtype=np.float32)
    surf = surf_base + 4 * np.sin(xs / 38 + t * 3.2) * (fill > 0.01) + 2 * np.sin(xs / 13 - t * 5)
    yy = np.arange(H, dtype=np.float32)[:, None]
    liq = inner * np.clip(yy - surf[None, :], 0, 1)
    img *= (1 - m[..., None] * 0.55)
    img += m[..., None] * IVORY * 0.035
    img += edge[..., None] * CHAMP * 0.55
    if fill > 0.005:
        lrgb = crop_metal(liq, bevel=7, strength=1.2) + GOLD * 0.18
        img[:] = img * (1 - liq[..., None]) + lrgb * liq[..., None]
        men = inner * np.clip(1.6 - np.abs(yy - surf[None, :]), 0, 1)
        img += men[..., None] * CHAMP * 1.3
        img += blur(liq, 30)[..., None] * GOLD * 0.25
    sweep = np.clip(1 - np.abs((np.arange(W)[None, :] + np.arange(H)[:, None] * 0.6) - (x0 - 400 + (t % 4.5) * 520)) / 60, 0, 1)
    img += (m * sweep * 0.12)[..., None] * IVORY


def scene_dial(ts, fi):
    t = ts - 23.6
    w = A["watch"]
    S, cx, cy, R = w["S"], w["cx"], w["cy"], w["R"]
    img = canvas(0.8)
    fill = 0.66 * eout((t - 0.3) / 2.6)
    title_layer(img, ts, t, fill)
    sub = A["title_sub"]
    T = A["title"]
    aa = sm((t - 1.6) / 0.8)
    over_c(img, CHAMP, sub * 0.85 * aa, W / 2, T["bb"][3] + 90)
    tag(img, "ARCHIVE № 023 · 0:23 · मदिरा", W / 2, T["bb"][3] + 165, 0.55 * aa)
    theta = math.radians(78) * eout(t / 2.97)
    sc = 1.0 + 0.035 + 0.22 * eio(t / 6.2)
    if t < 5.0:
        hx_l, hx_r = cx - R - 6, cx + R + 6
        seam_x = cx
        for side in (0, 1):
            mask = w["left"] if side == 0 else 1 - w["left"]
            hx = hx_l if side == 0 else hx_r
            rgb_w, a_w = door_warp(w["rgb"], w["alpha"], mask, theta, hx, seam_x, cy - R - 140, cy + R + 10, cy)
            shade = 0.25 + 0.75 * math.cos(theta)
            rgb_w = rgb_w * shade + GOLD * 0.22 * math.sin(theta)
            fade = 1 - sm((t - 3.6) / 1.4)
            M = affine(W / 2, H * 0.46, sc, sc, 0, S, S)
            draw_sprite(img, rgb_w.astype(np.float32), a_w.astype(np.float32) * fade, M)
    # numerals burst
    if t < 6.0:
        pul = 1 + 0.03 * down_pulse(ts)
        r = np.random.default_rng(31)
        for k, ((nrgb, nm), (sp, nx, ny)) in enumerate(zip(A["numerals_gold"], w["numerals"])):
            dx, dy = (nx - cx), (ny - cy)
            d = 1 + 2.6 * eout(t / 1.7) + 0.25 * t
            ang = 0.07 * t * (1 if k % 2 else -1)
            ca, sa = math.cos(ang), math.sin(ang)
            px = W / 2 + (dx * ca - dy * sa) * d * sc
            py = H * 0.46 + (dx * sa + dy * ca) * d * sc
            s = (1 + (1.2 + r.uniform(0, 1.4)) * eout(t / 2.5)) * sc * pul
            rot = r.uniform(-40, 40) * eout(t / 3)
            fade = 1 - sm((t - 3.8) / 1.8)
            M = affine(px, py, s, s, rot, nm.shape[1], nm.shape[0])
            draw_sprite(img, nrgb, nm * fade, M)
    if t < 0.12:
        img += (1 - t / 0.12) * 0.35 * CHAMP
    # dive into the counter of म
    if t >= 6.15:
        u = (t - 6.15) / 1.05
        s = math.exp(math.log(46) * ein(u))
        px, py = A["title"]["counter"]
        M = np.float32([[s, 0, W / 2 - s * px + (px - W / 2) * (1 - eio(u))], [0, s, H / 2 - s * py + (py - H / 2) * (1 - eio(u))]])
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(float(BG[0]), float(BG[1]), float(BG[2])))
    return img


def scene_movement(ts, fi):
    h = ht(ts) - ht(30.8)
    tot = ht(48.1) - ht(30.8)
    p = eio(h / tot)
    cam = 2500 * p
    img = canvas(0.55)
    occ = np.zeros((H // 4, W // 4), np.float32)
    bs = bsteps(ts)
    pend_pivot = (W / 2, -3350)
    for g in sorted(A["gears"], key=lambda g: g["depth"]):
        sx, sy = g["x"], g["y"] + cam * g["depth"] + H * 0.5
        r = g["r"]
        if sy + r < -50 or sy - r > H + 50:
            continue
        ang = g["dir"] * bs * 360.0 / g["n"]
        m = g["mask"]
        S = m.shape[0]
        Mrot = cv2.getRotationMatrix2D((S / 2, S / 2), ang, 1.0)
        mr = cv2.warpAffine(m, Mrot, (S, S), flags=cv2.INTER_LINEAR)
        rgb = metal(mr, bevel=max(3, r / 55), strength=1.3)
        tone = 1.0
        if g["depth"] < 0.9:
            tone = 0.42
            rgb = cv2.GaussianBlur(rgb, (0, 0), 2.5)
            mr = cv2.GaussianBlur(mr, (0, 0), 2.5)
        elif g["depth"] > 1.2:
            tone = 0.5
            rgb = cv2.GaussianBlur(rgb, (0, 0), 6)
            mr = cv2.GaussianBlur(mr, (0, 0), 6)
        over(img, rgb * tone, mr, sx - S / 2, sy - S / 2)
        q = cv2.resize(mr, (max(1, S // 4), max(1, S // 4)), interpolation=cv2.INTER_AREA)
        over(occ[..., None], np.ones(1, np.float32), q, (sx - S / 2) / 4, (sy - S / 2) / 4)
    # balance wheel, hairspring, pendulum, screws
    lay = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(lay)
    bwx, bwy = W * 0.5, -1150 + cam + H * 0.5
    hb = ht(ts)
    if -400 < bwy < H + 400:
        ang = 0.75 * math.sin(2 * math.pi * hb / (2 * 0.3715))
        d.ellipse([bwx - 330, bwy - 330, bwx + 330, bwy + 330], outline=255, width=30)
        for k in range(3):
            a = ang + 2 * math.pi * k / 3
            d.line([(bwx, bwy), (bwx + math.cos(a) * 310, bwy + math.sin(a) * 310)], fill=255, width=18)
        pts = []
        for j in range(400):
            u = j / 399
            a = ang * 0.6 + u * 6 * 2 * math.pi
            rad = 40 + u * 170
            pts.append((bwx + math.cos(a) * rad, bwy + math.sin(a) * rad))
        d.line(pts, fill=200, width=3)
        d.ellipse([bwx - 34, bwy - 34, bwx + 34, bwy + 34], fill=255)
    px, py = pend_pivot[0], pend_pivot[1] + cam + H * 0.5
    swing = 0.30 * math.sin(2 * math.pi * hb / (2 * BAR))
    bx, by = px + math.sin(swing) * 1250, py + math.cos(swing) * 1250
    if by + 120 > 0:
        d.line([(px, py), (bx, by)], fill=255, width=16)
        d.ellipse([bx - 110, by - 110, bx + 110, by + 110], fill=255)
    for (x, y, dp, ph) in A["screws"]:
        sy = y + cam * dp + H * 0.5
        if -20 < sy < H + 20:
            a = ph + hb * 0.4
            s = 9 * dp
            d.polygon([(x + math.cos(a) * s * 2, sy + math.sin(a) * s * 2), (x + math.cos(a + 1.6) * s, sy + math.sin(a + 1.6) * s),
                       (x - math.cos(a) * s * 2, sy - math.sin(a) * s * 2), (x - math.cos(a + 1.6) * s, sy - math.sin(a + 1.6) * s)], fill=255)
    lm = cv2.GaussianBlur(np.asarray(lay, np.float32) / 255, (0, 0), 0.8)
    if lm.max() > 0:
        lrgb = crop_metal(lm, bevel=5, strength=1.4)
        img = img * (1 - lm[..., None]) + lrgb * lm[..., None]
        q = cv2.resize(lm, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        occ = np.maximum(occ, q)
    # god rays through the mechanism
    light = CONEQ * (1 - np.clip(occ, 0, 1))
    acc = light.copy()
    sxq, syq = W / 8, -40
    for k in range(1, 8):
        s = 1 + 0.08 * k
        M = np.float32([[s, 0, (1 - s) * sxq], [0, s, (1 - s) * syq]])
        acc += cv2.warpAffine(light, M, (W // 4, H // 4)) * (0.82 ** k)
    rays = cv2.resize(cv2.GaussianBlur(acc, (0, 0), 2), (W, H)) / 4.0
    img += rays[..., None] * np.array([0.35, 0.25, 0.13], np.float32)
    # the gold drop
    if ts >= 46.14:
        u = max(0.0, (ht(ts) - ht(46.55)) / (48.1 - 46.55))
        y = 140 + (H * 0.62 - 140) * u ** 2
        sz = 30 + 110 * u ** 2
        st = 1 + 1.3 * u
        dm = np.zeros((int(sz * 2.2 * st) + 40, int(sz * 1.2) + 40), np.float32)
        hh, ww = dm.shape
        cv2.circle(dm, (ww // 2, int(hh - sz * 0.6 - 20)), int(sz * 0.6), 1.0, -1, cv2.LINE_AA)
        tri = np.array([[ww // 2, 20], [ww // 2 - int(sz * 0.5), int(hh - sz * 0.8 - 20)], [ww // 2 + int(sz * 0.5), int(hh - sz * 0.8 - 20)]], np.int32)
        cv2.fillConvexPoly(dm, tri, 1.0, cv2.LINE_AA)
        dm = cv2.GaussianBlur(dm, (0, 0), 1.2)
        drgb = metal(dm, bevel=max(3, sz / 9), strength=1.6) + GOLD * 0.25
        over(img, drgb, dm, W / 2 - ww / 2, y - hh + sz * 0.6)
        g = np.zeros((H, W), np.float32)
        cv2.circle(g, (W // 2, int(y)), int(sz), 1.0, -1)
        img += blur(g, 60)[..., None] * GOLD * 0.35
    a = sm((h - 0.6) / 0.8) * (1 - sm((ts - 46.0) / 0.6))
    tag(img, "ARCHIVE № 031 · THE MOVEMENT", W / 2, H - 230, 0.5 * a)
    if h < 0.5:
        img = img * sm(h / 0.5)
    return img


SURF_Y = None


def surface_bg():
    img = canvas(0.0)
    hz = int(H * 0.30)
    yy = np.arange(H, dtype=np.float32)[:, None]
    plane = np.clip((yy - hz) / (H - hz), 0, 1)
    img += (plane * 0.05)[..., None] * np.array([0.9, 0.75, 0.5]) * np.ones((1, W, 1))
    xx = np.arange(W, dtype=np.float32)[None, :]
    refl = np.exp(-((xx - W / 2) / 120) ** 2 - ((yy - H * 0.60) / 420) ** 2) * (yy > hz)
    img += refl[..., None] * GOLD * 0.22
    img += CONE[..., None] * np.array([0.10, 0.08, 0.05]) * (yy < hz)[..., None]
    return img


def scene_drop(ts, fi):
    t = ts - 48.1
    P = (W / 2, H * 0.62)
    img = surface_bg()
    hz = int(H * 0.30)
    # ripples
    rip = np.zeros((H, W), np.float32)
    dark = np.zeros((H, W), np.float32)
    emits = [48.1 + 0.3715 * k for k in range(4)] + [48.1 + BAR * k for k in range(2, 8)]
    for k, te in enumerate(emits):
        age = ts - te
        if age <= 0:
            continue
        rad = 260 * age
        amp = 0.75 * math.exp(-age / 3.2) * (1.0 if k < 4 else 0.6)
        if amp < 0.02:
            continue
        ax = (int(rad), int(rad * 0.32))
        cv2.ellipse(rip, (int(P[0]), int(P[1])), ax, 0, 0, 360, amp, 3)
        cv2.ellipse(dark, (int(P[0]), int(P[1])), (max(1, int(rad - 10)), max(1, int((rad - 10) * 0.32))), 0, 0, 360, amp, 5)
    rip = cv2.GaussianBlur(rip, (0, 0), 1.4)
    dark = cv2.GaussianBlur(dark, (0, 0), 3)
    img = img * (1 - 0.5 * dark[..., None]) + rip[..., None] * CHAMP * 0.75
    # crown splash + jet (slow motion)
    srgb, sa = A["sphere"]
    if 0 <= t < 2.4:
        r = np.random.default_rng(48)
        tau = t * 0.5
        for k in range(70):
            a = r.uniform(0, 2 * math.pi)
            v = r.uniform(250, 700)
            vy = r.uniform(500, 1100)
            x = P[0] + math.cos(a) * v * tau
            y = P[1] + math.sin(a) * v * tau * 0.32 - (vy * tau - 0.5 * 2400 * tau * tau)
            if y > P[1] + math.sin(a) * v * tau * 0.32 + 2:
                continue
            s = r.uniform(0.15, 0.42)
            M = affine(x, y, s, s * (1 + 0.6 * max(0, vy - 2400 * tau) / 1100), 0, 64, 64)
            draw_sprite(img, srgb, sa, M)
        if 0.2 < t < 2.2:
            jh = 380 * math.sin(math.pi * (t - 0.2) / 2.0)
            jm = np.zeros((H, W), np.float32)
            cv2.ellipse(jm, (int(P[0]), int(P[1] - jh / 2)), (22, int(max(2, jh / 2))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            cv2.circle(jm, (int(P[0]), int(P[1] - jh - 26)), 20, 1.0, -1, cv2.LINE_AA)
            jm = cv2.GaussianBlur(jm, (0, 0), 1.0)
            img = img * (1 - jm[..., None]) + (crop_metal(jm, bevel=6, strength=1.5) + GOLD * 0.2) * jm[..., None]
    if t < 0.15:
        img += (1 - t / 0.15) * 0.25 * GOLD
    # the photograph rises, back first
    card = A["card_wait"]
    if t >= 1.5:
        u = eout((t - 1.5) / 3.6)
        cy = P[1] + 80 - 820 * u
        s = 0.72 + 0.32 * u
        rot = 4 * math.sin(t * 0.7)
        flip = eio((t - 8.3) / 1.0)
        yaw = math.cos(math.pi * flip)
        exp_ = eio((t - 9.45) / 0.95)
        s = s * (1 + 2.4 * exp_)
        cy = cy + (H * 0.5 - cy) * exp_
        side = card["back"].copy() if yaw > 0 else card["front"].copy()
        if yaw > 0:
            # waiting type: letters arrive one at a time
            parts = ["इं", "इंत", "इंतज़ा", "इंतज़ार"]
            times = [2.6, 3.7, 4.8, 6.3]
            n = sum(1 for x in times if t >= x)
            side = card["back"].copy()
            if n:
                txt = parts[n - 1]
                sp = tsprite(txt, TIRO, 92)
                al = 0.9
                if n == 4:
                    i, dt = last_beat(ts)
                    al = 0.45 if not any(is_down(j) and 6.3 + 48.1 <= BEATS[j] <= ts for j in range(len(BEATS))) else 0.9
                over(side, INK, sp * al, 34, (card["h"] - sp.shape[0]) / 2 - 20)
        a = card["a"] * sm((t - 1.5) / 0.5)
        lit = 0.75 + 0.35 * u
        M = affine(W / 2, cy, s * max(0.02, abs(yaw)), s, rot * (1 - exp_), card["w"], card["h"])
        # reflection
        if exp_ < 0.5:
            Mr = affine(W / 2, P[1] + (P[1] - cy) * 0.35 + 140, s * max(0.02, abs(yaw)), -s * 0.55, -rot, card["w"], card["h"])
            tmp = np.zeros_like(img)
            draw_sprite(tmp, (side * 0.5).astype(np.float32), a, Mr)
            refl = cv2.GaussianBlur(tmp, (0, 0), 3) * (np.arange(H)[:, None, None] > P[1] - 40)
            img += refl * 0.35
        draw_sprite(img, np.clip(side * lit, 0, 1).astype(np.float32), a, M)
        if yaw > 0 and exp_ < 0.1:
            tg = tag_sprite("ARCHIVE № 048 · 0:48", 20)
            over_c(img, IVORY, tg * 0.5 * sm((t - 2.6) / 0.6), W / 2, cy + card["h"] * s / 2 + 40)
    return img


def aged_back(card):
    return card["back"]


def card_proj(z, X, Y):
    s = 1.0 / z
    return W / 2 + X * 520 * s, H / 2 + Y * 520 * s, s


def s06_cards():
    """deterministic layout for the floating photographs"""
    v = 0.42
    downs = [b for i, b in enumerate(BEATS) if is_down(i) and 59.0 <= b <= 69.0]
    tds = [b - 58.5 for b in downs[::2]][:4]
    kinds = ["chair", "door", "lamp", "shadows"]
    xs = [0.32, -0.30, 0.28, -0.22]
    ys = [-0.42, 0.30, 0.36, -0.30]
    cards = [dict(card=A["cards"]["terrace"], z0=0.30, X=0.0, Y=0.0, rot=0, td=None, yaw0=0.0)]
    cards.append(dict(card=A["card_yaad"], z0=1.45, X=-0.34, Y=-0.40, rot=-6, td=None, yaw0=1.0))
    for k, (td, kd) in enumerate(zip(tds, kinds)):
        cards.append(dict(card=A["cards"][kd], z0=0.95 + v * td, X=xs[k], Y=ys[k], rot=(-1) ** k * 7, td=td, yaw0=0.0))
    for k, (z0, X, Y, kd) in enumerate([(2.6, 0.95, -0.2, "sea"), (3.3, -1.0, 0.35, "stairs"), (4.1, 0.9, 0.55, "cups"),
                                       (4.9, -0.85, -0.6, "terrace"), (5.6, 0.2, 0.9, "sea")]):
        cards.append(dict(card=A["cards"][kd], z0=z0, X=X, Y=Y, rot=(-1) ** k * 10, td=None, yaw0=0.0))
    return cards, v


def particles_for(cardinfo, t, v, seed):
    """gold particles that leave a dissolving card"""
    td = cardinfo["td"]
    if td is None or t < td:
        return None
    r = np.random.default_rng(seed)
    n = 520
    u = r.uniform(-0.5, 0.5, n)
    w = r.uniform(-0.5, 0.5, n)
    delay = r.uniform(0, 1.0, n)
    age = t - td - delay
    ok = age > 0
    zt = np.maximum(0.2, cardinfo["z0"] - v * (td + delay))
    s = 1.0 / zt
    cx = W / 2 + cardinfo["X"] * 520 * s
    cy = H / 2 + cardinfo["Y"] * 520 * s
    cw, ch = cardinfo["card"]["w"], cardinfo["card"]["h"]
    x = cx + u * cw * s + np.sin(age * 2 + u * 9) * 30 * age
    y = cy + w * ch * s - 140 * age - 40 * age ** 2
    col = np.clip(age / 1.0, 0, 1)
    alpha = np.clip(1 - age / 3.0, 0, 1) * ok
    return x, y, col, alpha


def scene_photos(ts, fi):
    t = ts - 58.5
    img = canvas(0.7)
    cards, v = s06_cards()
    items = []
    for ci in cards:
        z = ci["z0"] - v * t
        items.append((z, "card", ci))
    rz = 2.5 - 0.08 * t
    items.append((rz, "rose", None))
    items.sort(key=lambda x: -x[0])
    pts = []
    for z, kind, ci in items:
        if kind == "rose":
            rgb, a = A["rose"]
            px, py, s = card_proj(z, 0.16, -0.02)
            sp = np.zeros((H, W), np.float32)
            cv2.circle(sp, (int(px), int(py - 60 * s)), int(240 * s), 1.0, -1)
            img += blur(sp, 40)[..., None] * np.array([0.10, 0.06, 0.04])
            M = affine(px, py, s * 1.2, s * 1.2, -8 + 3 * math.sin(t * 0.5), rgb.shape[1], rgb.shape[0])
            draw_sprite(img, rgb, a, M, mul=1 - sm((t - 10.3) / 0.6))
            continue
        if z < 0.12:
            continue
        px, py, s = card_proj(z, ci["X"], ci["Y"] + 0.03 * math.sin(t * 0.6 + ci["z0"]))
        c = ci["card"]
        yaw = 1.0
        side = c["front"]
        if ci["yaw0"] > 0:
            f = eio((t - 0.2) / 1.4)
            yaw = math.cos(math.pi * (1 - f))
            side = c["back"] if yaw < 0 else c["front"]
        bright = float(np.clip(1.3 / (z ** 0.8), 0.18, 1.05))
        alpha = c["a"].copy()
        mul = 1.0
        if ci["td"] is not None and t >= ci["td"]:
            u = (t - ci["td"]) / 1.0
            nz = noise2(c["h"], c["w"], 8, int(ci["z0"] * 100))
            alpha = alpha * np.clip((nz - u * 1.2) * 6 + 1, 0, 1)
            if u > 1.3:
                continue
        if z < 0.2:
            mul = (z - 0.12) / 0.08
        M = affine(px, py, s * max(0.02, abs(yaw)), s, ci["rot"] * (1 - 0.3 * t / 12), c["w"], c["h"])
        draw_sprite(img, np.clip(side * bright, 0, 1).astype(np.float32), alpha, M, mul=mul)
    pts = [particles_for(ci, t, v, int(ci["z0"] * 1000)) for ci in cards if ci["td"] is not None]
    # particles + convergence into the word
    acc = np.zeros((H // 2, W // 2), np.float32)
    accg = np.zeros((H // 2, W // 2), np.float32)
    conv = eio((t - 10.9) / 1.1)
    for p in pts:
        if p is None:
            continue
        x, y, col, al = p
        ok = al > 0.01
        x, y, col, al = x[ok], y[ok], col[ok], al[ok]
        if conv > 0 and len(x):
            tgt = A["word_pts"][:len(x)]
            x = x * (1 - conv) + tgt[:, 0] * conv
            y = y * (1 - conv) + tgt[:, 1] * conv
            al = np.maximum(al, conv)
            col = np.maximum(col, conv)
        xi = np.clip((x / 2).astype(int), 0, W // 2 - 1)
        yi = np.clip((y / 2).astype(int), 0, H // 2 - 1)
        np.add.at(acc, (yi, xi), al * (1 - col))
        np.add.at(accg, (yi, xi), al * col)
    if conv > 0:
        extra = A["word_pts"]
        r = np.random.default_rng(70)
        sx = r.uniform(0, W, len(extra))
        sy = r.uniform(0, H, len(extra))
        x = sx * (1 - conv) + extra[:, 0] * conv
        y = sy * (1 - conv) + extra[:, 1] * conv
        xi = np.clip((x / 2).astype(int), 0, W // 2 - 1)
        yi = np.clip((y / 2).astype(int), 0, H // 2 - 1)
        np.add.at(accg, (yi, xi), np.full(len(x), 0.8 * conv))
    acc = cv2.resize(cv2.GaussianBlur(acc, (0, 0), 0.8), (W, H))
    accg = cv2.resize(cv2.GaussianBlur(accg, (0, 0), 0.8), (W, H))
    img += acc[..., None] * IVORY * 0.9 + accg[..., None] * CHAMP * 1.1 + blur(accg, 10)[..., None] * GOLD * 0.6
    if conv > 0.85:
        wd = A["word"]
        k = (conv - 0.85) / 0.15
        img[:] = img * (1 - wd["m"][..., None] * k) + (wd["rgb"] + GOLD * 0.15) * wd["m"][..., None] * k
    tag(img, "ARCHIVE № 058 · 0:58 · याद", W / 2, H - 230, 0.5 * sm((t - 1.0) / 0.8) * (1 - sm((t - 10.0) / 0.8)))
    return img


def scene_word(ts, fi):
    t = ts - 70.5
    wd = A["word"]
    m = wd["m"]
    x0, y0, x1, y1 = wd["bb"]
    img = canvas(0.6)
    h = ht(ts) - ht(70.5)
    # frozen pour streaks above the letters (baked motion blur)
    if h < 0.4:
        streak = np.zeros_like(m)
        for k in range(1, 12):
            streak = np.maximum(streak, np.roll(m, -14 * k, 0) * (0.5 * (1 - k / 12)))
        img += streak[..., None] * GOLD * 0.35 * (1 - h / 0.4)
    slump = eio((h - 0.6) / 1.9)
    flat = eio((h - 2.6) / 0.6)
    if slump > 0:
        xs = np.arange(W, dtype=np.float32)
        dr = (np.sin(xs / 37) + np.sin(xs / 13 + 1.3) * 0.5 + np.sin(xs / 71 + 2) * 0.7)
        dr = np.clip(dr, 0, None) * 120 * slump
        yy = np.arange(H, dtype=np.float32)[:, None]
        mid = (y0 + y1) / 2
        k = np.clip((yy - mid) / ((y1 - y0) / 2), 0, 1)
        sag = 1 - 0.10 * slump
        mapy = np.ascontiguousarray(np.broadcast_to(mid + (yy - mid) * (1 / sag) - dr[None, :] * k, (H, W))).astype(np.float32)
        mapx = np.ascontiguousarray(np.repeat(xs[None, :], H, 0)).astype(np.float32)
        mm = cv2.remap(m, mapx, mapy, cv2.INTER_LINEAR, borderValue=0)
    else:
        mm = m
    if flat > 0:
        L = A["letter"]
        sheet = np.zeros((H, W), np.float32)
        lw, lh = L["w"] * 0.86, 240 * 0.86
        cv2.rectangle(sheet, (int(W / 2 - lw / 2), int(H * 0.47 - lh / 2)), (int(W / 2 + lw / 2), int(H * 0.47 + lh / 2)), 1.0, -1)
        mm = mm * (1 - flat) + sheet * flat
    rgb = crop_metal(mm, bevel=9, strength=1.3) + GOLD * 0.12
    if flat > 0:
        rgb = rgb * (1 - flat) + IVORY * 0.85 * flat
    img = img * (1 - mm[..., None]) + rgb * mm[..., None]
    img += blur(mm, 34)[..., None] * GOLD * 0.30 * (1 - flat)
    if 1.4 <= t <= 2.2:
        u = (t - 1.4) / 0.8
        pos = x0 - 300 + u * (x1 - x0 + 900)
        band = np.clip(1 - np.abs(np.arange(W)[None, :] + (np.arange(H)[:, None] - y0) * 0.7 - pos) / 70, 0, 1)
        img += (mm * band)[..., None] * IVORY * 0.55
    if slump > 0.5 and flat < 1:
        srgb, sa = A["sphere"]
        xs = [x0 + (x1 - x0) * f for f in (0.12, 0.33, 0.55, 0.78, 0.92)]
        for k, x in enumerate(xs):
            fall = max(0.0, h - 1.5 - 0.15 * k)
            y = y1 + 60 + 900 * fall ** 2
            if y < H:
                M = affine(x, y, 0.3, 0.42, 0, 64, 64)
                draw_sprite(img, srgb, sa, M)
    if h < 0.02 and t < 0.08:
        img += 0.25 * GOLD
    tag(img, "ARCHIVE № 070 · 1:10 · मदिरा", W / 2, y1 + 200, 0.5 * (1 - flat))
    return img


def ink_reveal(sp, p):
    if p >= 1:
        return sp
    w = sp.shape[1]
    xs = np.arange(w, dtype=np.float32)
    edge = p * (w + 40) - 20
    return sp * np.clip((edge - xs) / 18, 0, 1)[None, :]


def letter_image(ts, h):
    """the letter as a sprite (rgb, alpha) given song time ts and held local time h"""
    L = A["letter"]
    w, ph = L["w"], 240
    panels = 4
    H_ = ph * panels
    rgb = np.zeros((H_ + ph, w, 3), np.float32)
    al = np.zeros((H_ + ph, w), np.float32)
    off = ph  # room above for folded backs
    paper = L["paper"].copy()
    starts = [75.25, 75.25 + 1.6, 75.25 + 3.15]
    for k, (sp, st) in enumerate(zip(L["lines"], starts)):
        if ts < st:
            continue
        n_on = onsets_between(st, ts)
        prog = max(n_on / 7.0, (ts - st) / 1.9)
        prog = min(1.0, prog) if ts < 81.0 else min(1.0, max(n_on / 7.0, (81.0 - st) / 1.9))
        rv = ink_reveal(sp, prog)
        y = 90 + k * ph
        x = 60 if k else 60
        over(paper, INK, cv2.GaussianBlur(rv, (0, 0), 2.2) * 0.18, x, y)
        over(paper, INK, rv * 0.92, x, y)
    over(paper, INK, L["date"] * 0.6, w - L["date"].shape[1] - 40, 30)
    # panel 0 always open; panels 1..3 unfold one per bar
    rgb[off:off + ph] = paper[0:ph]
    al[off:off + ph] = L["alpha"][0:ph]
    bottom = off + ph
    for k in range(1, panels):
        t0 = 75.0 + 0.15 + (k - 1) * BAR
        f = eio((h - (t0 - 75.0)) / 0.95)
        theta = math.pi * (1 - f)
        if f <= 0:
            break
        c = math.cos(theta)
        hgt = int(abs(c) * ph)
        if hgt < 2:
            continue
        if c >= 0:
            src = paper[k * ph:(k + 1) * ph]
            sa = L["alpha"][k * ph:(k + 1) * ph]
            seg = cv2.resize(src, (w, hgt)) * (0.62 + 0.38 * c)
            sga = cv2.resize(sa, (w, hgt))
            y0 = bottom
            y1 = min(rgb.shape[0], y0 + hgt)
            rgb[y0:y1] = seg[:y1 - y0] * sga[:y1 - y0, :, None] + rgb[y0:y1] * (1 - sga[:y1 - y0, :, None])
            al[y0:y1] = np.maximum(al[y0:y1], sga[:y1 - y0])
            bottom = y0 + hgt
        else:
            src = L["back"][k * ph:(k + 1) * ph][::-1]
            sa = L["alpha"][k * ph:(k + 1) * ph][::-1]
            seg = cv2.resize(src, (w, hgt)) * (0.55 + 0.25 * -c)
            sga = cv2.resize(sa, (w, hgt))
            y1 = bottom
            y0 = max(0, y1 - hgt)
            rgb[y0:y1] = rgb[y0:y1] * (1 - sga[-(y1 - y0):, :, None]) + seg[-(y1 - y0):] * sga[-(y1 - y0):, :, None]
            al[y0:y1] = np.maximum(al[y0:y1], sga[-(y1 - y0):])
            break
    return rgb, al, off


def scene_letter(ts, fi):
    t = ts - 75.0
    h = ht(ts) - 75.0
    img = canvas(0.0)
    yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    warm = np.exp(-(((xx - W * 0.1) / (W * 0.9)) ** 2 + ((yy - H * 0.85) / (H * 0.7)) ** 2))
    warm = cv2.resize(warm, (W, H))
    dim = 1.0 - 0.3 * sm((ts - 81.0) / 0.3)
    img += warm[..., None] * np.array([0.20, 0.12, 0.05], np.float32) * dim
    rgb, al, off = letter_image(ts, h)
    rot = -7 + 13 * (h / 6.0) + (0.8 * (ts - 81.0) if ts > 81 else 0)
    sc = 0.92 + (0.012 * (ts - 81.0) if ts > 81 else 0)
    cy = H * 0.50
    shadow = cv2.GaussianBlur(al, (0, 0), 18)
    Ms = affine(W / 2 + 26, cy + 34, sc, sc, rot, al.shape[1], al.shape[0])
    draw_sprite(img, np.zeros(3, np.float32), shadow * 0.75, Ms)
    lit = (0.62 + 0.38 * warm[int(cy), W // 3]) * dim
    M = affine(W / 2, cy, sc, sc, rot, al.shape[1], al.shape[0])
    draw_sprite(img, np.clip(rgb * lit * 1.08, 0, 1).astype(np.float32), al, M)
    # wax seal cracks on the first word
    srgb, sa = A["seal"]
    sx, sy = W / 2, cy - al.shape[0] / 2 * sc + off * sc + 240 * sc
    if h < 1.6:
        u = eout((h - 0.25) / 1.2)
        for side in (-1, 1):
            half = sa.copy()
            if side < 0:
                half[:, 70:] = 0
            else:
                half[:, :70] = 0
            Mx = affine(sx + side * 160 * u, sy + 420 * u * u, 0.9, 0.9, side * 40 * u, 140, 140)
            draw_sprite(img, srgb * dim, half * (1 - sm((h - 0.9) / 0.6)), Mx)
    # pen flick: ink droplets that freeze at 1:21
    if ts >= 80.35:
        isp, isa = A["ink_sphere"]
        r = np.random.default_rng(81)
        tau = ht(ts) - ht(80.35)
        ex, ey = W * 0.70, cy + 210 * sc
        for k in range(16):
            vx, vy = r.uniform(60, 520), r.uniform(380, 900)
            x = ex + vx * tau
            y = ey - vy * tau + 0.5 * 1500 * tau * tau
            s = r.uniform(0.12, 0.34)
            M = affine(x, y, s, s * (1 + 0.5 * (ts < 81.0)), 0, 64, 64)
            draw_sprite(img, isp * dim + 0.03, isa, M)
    tag(img, "ARCHIVE № 075 · 1:15 · ख़त", W / 2, H - 200, 0.5 * sm((t - 0.6) / 0.6) * (1 - sm((ts - 81.4) / 0.3)))
    if ts > 81.75:
        f = sm((ts - 81.75) / 0.35)
        img = img * (1 - f)
        endm = tsprite("मदिरा", ROZHA, 120, pad=24)
        er = metal(endm, bevel=4, strength=1.3)
        over_c(img, er, endm * f, W / 2, H * 0.47)
        cr = tag_sprite("SEEDHE MAUT · PROD. HURRICANE · DL91 FM", 20)
        over_c(img, IVORY, cr * 0.55 * f, W / 2, H * 0.47 + 110)
    return img


def render_frame(fi):
    ts = REEL0 + fi / FPS
    if ts < 23.6:
        img, dust = scene_watch(ts, fi), 0.8
    elif ts < 30.8:
        img, dust = scene_dial(ts, fi), 0.55
    elif ts < 48.1:
        img, dust = scene_movement(ts, fi), 1.0
    elif ts < 58.5:
        img, dust = scene_drop(ts, fi), 0.5
    elif ts < 70.5:
        img, dust = scene_photos(ts, fi), 0.8
    elif ts < 75.0:
        img, dust = scene_word(ts, fi), 0.5
    else:
        img, dust = scene_letter(ts, fi), 0.65
    if ts > 81.75:
        dust *= 1 - sm((ts - 81.75) / 0.35)
    return post(img, ts, fi, dust)


# ------------------------------------------------------------------- main ---
ARGS = None


def worker(job):
    k, f0, f1, out = job
    init(*ARGS)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        enc.stdin.write(render_frame(fi).tobytes())
        if (fi - f0) % 48 == 0:
            print(f"[w{k}] {fi}/{f1}", flush=True)
    enc.stdin.close()
    enc.wait()
    return out


def main():
    global ARGS
    audio, out, work, fontdir = sys.argv[1:5]
    ARGS = (audio, fontdir)
    if "--preview" in sys.argv:
        init(audio, fontdir)
        os.makedirs(out, exist_ok=True)
        for s in sys.argv[sys.argv.index("--preview") + 1].split(","):
            ts = float(s)
            fi = int(round((ts - REEL0) * FPS))
            Image.fromarray(render_frame(fi)).save(os.path.join(out, f"f_{ts:06.2f}.png"))
            print("saved", ts, flush=True)
        return
    os.makedirs(work, exist_ok=True)
    nw = int(os.environ.get("WORKERS", "4"))
    b = np.linspace(0, NF, nw + 1).astype(int)
    jobs = [(k, int(b[k]), int(b[k + 1]), os.path.join(work, f"reel_chunk{k}.mp4")) for k in range(nw)]
    with mp.get_context("fork").Pool(nw) as pool:
        parts = pool.map(worker, jobs)
    lst = os.path.join(work, "reel_chunks.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", str(REEL0), "-t", str(DUR), "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", f"afade=t=in:st=0:d=0.05,afade=t=out:st={DUR - 0.5}:d=0.5", "-c:a", "aac", "-b:a", "256k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
