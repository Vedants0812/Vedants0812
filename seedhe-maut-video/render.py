#!/usr/bin/env python3
"""
claude x seedhe maut -- terminal/HUD remix renderer.

Takes the Seedhe Maut music video (source) and re-renders it in the style of
the "Claude loves Seedhe Maut" reference: terminal boot, glowing Devanagari
glyph, hype-index bar chart, the MV pushed through a dozen render modes synced
to the beat, "स से सीधे / म से मौत" letters filled with footage, neon stick
figures, big type, outro credit.

usage:
  python3 render.py SRC.mp4 OUT.mp4 WORKDIR            # full render
  python3 render.py SRC.mp4 OUT_DIR WORKDIR --preview 3,8,14,20
"""
import math
import os
import subprocess
import sys
from functools import lru_cache
import multiprocessing as mp

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
DUR = 64.0
NF = int(DUR * FPS)
SRC_DUR = 120.86

MONO = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf"
MONO_B = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-ExtraBold.ttf"
DEVA = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf"
DEVA_R = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
BIG = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"

ORANGE = np.array([1.0, 0.42, 0.22], np.float32)
ORANGE_I = (255, 107, 56)
RED_I = (240, 40, 45)
WHITE_I = (240, 236, 232)
GREY_I = (140, 140, 140)
GREEN_I = (90, 220, 120)

SRC = WORK = None


@lru_cache(None)
def font(path, size):
    return ImageFont.truetype(path, size)


# ----------------------------------------------------------------------------
# audio analysis (real numbers from the track)
# ----------------------------------------------------------------------------
def analyse_audio(wav):
    import librosa
    y, sr = librosa.load(wav, sr=22050)
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    ot = librosa.times_like(oenv, sr=sr)
    tempo, bf = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
    beats = librosa.frames_to_time(bf, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, units="time")
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    rt = librosa.times_like(rms, sr=sr, hop_length=256)
    oenv = oenv / np.percentile(oenv, 99)
    rms = rms / rms.max()
    return dict(ot=ot, oenv=oenv, beats=beats, onsets=onsets, rms=rms, rt=rt,
                bpm=float(np.atleast_1d(tempo)[0]))


AU = None
B0 = 0
T = {}


def setup_timeline():
    global B0, T
    bt = AU["beats"]
    B0 = int(np.argmin(np.abs(bt - 12.0)))
    near = lambda x: float(bt[np.argmin(np.abs(bt - x))])
    T = dict(
        drop=float(bt[B0]),
        spell=float(bt[B0 + 96]),
        crowd=near(54.0),
        tera=near(57.6),
        outro=near(60.7),
    )


def oenv_at(t):
    i = min(np.searchsorted(AU["ot"], t), len(AU["oenv"]) - 1)
    return float(AU["oenv"][i])


def last_beat(t):
    i = int(np.searchsorted(AU["beats"], t, "right")) - 1
    if i < 0:
        return -1, 9.0
    return i, t - AU["beats"][i]


def onset_count(t0, t1):
    o = AU["onsets"]
    return int(((o >= t0) & (o < t1)).sum())


# ----------------------------------------------------------------------------
# text helpers (mixed Latin + Devanagari)
# ----------------------------------------------------------------------------
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


def mixed_fonts(size, bold):
    return (font(MONO_B if bold else MONO, size),
            font(DEVA if bold else DEVA_R, int(size * 1.1)))


def text_w(s, size, bold=False):
    lat, dev = mixed_fonts(size, bold)
    return sum((dev if d else lat).getlength(t) for d, t in runs(s))


def draw_mixed(d, x, base, s, size, fill, bold=False):
    lat, dev = mixed_fonts(size, bold)
    for isd, t in runs(s):
        f = dev if isd else lat
        d.text((x, base), t, font=f, fill=fill, anchor="ls")
        x += f.getlength(t)
    return x


def draw_tracked(d, x, base, s, size, fill, track=3, bold=False):
    lat, dev = mixed_fonts(size, bold)
    for c in s:
        f = dev if is_deva(c) else lat
        d.text((x, base), c, font=f, fill=fill, anchor="ls")
        x += f.getlength(c) + track
    return x


def tracked_w(s, size, track=3, bold=False):
    lat, dev = mixed_fonts(size, bold)
    return sum((dev if is_deva(c) else lat).getlength(c) + track for c in s)


def text_mask(s, path, size, cx, cy, anchor="mm"):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).text((cx, cy), s, font=font(path, size), fill=255, anchor=anchor)
    return np.asarray(im, np.float32) / 255


def glow(mask, core=(1, .93, .9), col=ORANGE, k=(5, 14, 34), a=(1.0, .7, .45)):
    g = sum(cv2.GaussianBlur(mask, (0, 0), s) * w for s, w in zip(k, a))
    return mask[..., None] * np.array(core, np.float32) + g[..., None] * col


# ----------------------------------------------------------------------------
# precomputed assets
# ----------------------------------------------------------------------------
def lut(stops):
    x = np.linspace(0, 1, 256)
    p = [s[0] for s in stops]
    return np.stack([np.interp(x, p, [s[1][c] for s in stops]) for c in range(3)], -1).astype(np.float32)


LUT_NEON = lut([(0, (.02, 0, .06)), (.3, (.22, .02, .5)), (.55, (.95, .08, .55)),
                (.8, (1, .45, .2)), (1, (1, .96, .86))])
LUT_TEAL = lut([(0, (0, .03, .05)), (.45, (0, .45, .5)), (.8, (.35, 1, .9)), (1, (1, 1, 1))])

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.55 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.5).clip(0.25, 1)
SCAN = np.where(np.arange(H) % 3 == 0, 0.9, 1.0).astype(np.float32)[:, None] * np.ones((1, W), np.float32)
rng = np.random.default_rng(7)
GRAIN = [rng.normal(0, 0.025, (H, W)).astype(np.float32) for _ in range(6)]
DOTGRID = np.zeros((H, W), np.float32)
DOTGRID[8::16, 8::16] = 1
DOTGRID = cv2.GaussianBlur(DOTGRID, (0, 0), 0.8) * 3
CLAHE = cv2.createCLAHE(2.5, (8, 8))


def glyph_atlas(chars, cell, size):
    out = []
    for c in chars:
        im = Image.new("L", (cell, cell), 0)
        f = font(DEVA if is_deva(c) else MONO_B, size)
        ImageDraw.Draw(im).text((cell / 2, cell / 2), c, font=f, fill=255, anchor="mm")
        out.append(np.asarray(im, np.float32) / 255)
    return np.stack(out)


GCELL = 16
ATLAS_N = glyph_atlas([" ", "·", ".", ":", "+", "न", "न", "न"], GCELL, 15)
ATLAS_M = glyph_atlas([" ", "·", ":", "-", "स", "म", "म", "ौ"], GCELL, 15)


def disc_atlas(cell, levels):
    out = []
    for k in range(levels):
        s = 4
        im = Image.new("L", (cell * s, cell * s), 0)
        r = (k / (levels - 1)) * cell * 0.62 * s
        c = cell * s / 2
        if r > 0:
            ImageDraw.Draw(im).ellipse([c - r, c - r, c + r, c + r], fill=255)
        out.append(np.asarray(im.resize((cell, cell), Image.LANCZOS), np.float32) / 255)
    return np.stack(out)


HCELL = 8
ATLAS_D = disc_atlas(HCELL, 14)


def bayer(n):
    if n == 1:
        return np.zeros((1, 1))
    b = bayer(n // 2)
    return np.block([[4 * b, 4 * b + 2], [4 * b + 3, 4 * b + 1]])


B8 = ((bayer(8) + 0.5) / 64).astype(np.float32)
BTILE = np.tile(B8, (H // 3 // 8 + 1, W // 3 // 8 + 1))[: H // 3, : W // 3]


def tile_atlas(atlas, levels_idx):
    gh, gw = levels_idx.shape
    c = atlas.shape[1]
    img = atlas[levels_idx].transpose(0, 2, 1, 3).reshape(gh * c, gw * c)
    out = np.zeros((H, W), np.float32)
    out[: min(H, gh * c), : min(W, gw * c)] = img[:H, :W]
    return out


# ----------------------------------------------------------------------------
# render modes
# ----------------------------------------------------------------------------
def gray(f):
    return f.mean(-1)


def enh(g):
    return np.clip((g - 0.02) * 1.7, 0, 1) ** 0.85


def clahe(g):
    return CLAHE.apply((np.clip(g, 0, 1) * 255).astype(np.uint8))


def m_raw(f, t, ctx):
    g = enh(gray(f))
    return np.stack([g, g * .94, g * .9], -1)


def m_neon(f, t, ctx):
    return LUT_NEON[clahe(gray(f))]


def m_teal(f, t, ctx):
    return LUT_TEAL[clahe(gray(f))]


def m_blood(f, t, ctx):
    g = enh(gray(f))
    return np.stack([np.clip(g * 1.35, 0, 1), g ** 2.4 * .55, g ** 2.4 * .5], -1)


def m_thermal(f, t, ctx):
    u = clahe(gray(f))
    return cv2.applyColorMap(u, cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32) / 255


def m_sobel(f, t, ctx):
    g = cv2.GaussianBlur(enh(gray(f)), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy)
    m = np.clip(m / (np.percentile(m[::4, ::4], 98) + 1e-4), 0, 1) ** 1.3
    gl = cv2.GaussianBlur(m, (0, 0), 6)
    return (m[..., None] * np.array([1, .62, .35], np.float32)
            + gl[..., None] * np.array([1, .3, .08], np.float32) * 1.1 + g[..., None] * .07)


def _glyph(f, atlas, hot):
    g = clahe(gray(f)).astype(np.float32) / 255
    gh, gw = H // GCELL + 1, W // GCELL + 1
    lum = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA)
    lum = np.clip(lum / (np.percentile(lum, 97) + 1e-3), 0, 1)
    lv = np.clip((lum ** 0.9) * len(atlas), 0, len(atlas) - 1).astype(int)
    mask = tile_atlas(atlas, lv)
    col = np.where((lv >= 5)[..., None], hot, np.array([.75, .75, .78], np.float32) * lum[..., None] * 1.3)
    colimg = cv2.resize(col.astype(np.float32), (gw * GCELL, gh * GCELL), interpolation=cv2.INTER_NEAREST)[:H, :W]
    out = mask[..., None] * colimg
    return out + cv2.GaussianBlur(out, (0, 0), 4) * .6


def m_glyph_n(f, t, ctx):
    return _glyph(f, ATLAS_N, np.array([1, .35, .15], np.float32))


def m_glyph_m(f, t, ctx):
    return _glyph(f, ATLAS_M, np.array([1, .9, .85], np.float32)) * np.array([1, .8, .7], np.float32)


def m_halftone(f, t, ctx):
    g = clahe(gray(f)).astype(np.float32) / 255
    gh, gw = H // HCELL, W // HCELL
    lum = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA)
    lv = np.clip(lum * len(ATLAS_D), 0, len(ATLAS_D) - 1).astype(int)
    m = tile_atlas(ATLAS_D, lv)
    return np.repeat(m[..., None], 3, -1) * np.array([.95, .93, .9], np.float32)


def m_bayer(f, t, ctx):
    g = clahe(gray(f)).astype(np.float32) / 255
    small = cv2.resize(g, (W // 3, H // 3), interpolation=cv2.INTER_AREA)
    bit = (small > BTILE).astype(np.float32)
    up = cv2.resize(bit, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    i, _ = last_beat(t)
    c1 = np.array([.3, .85, 1], np.float32) if i % 2 == 0 else np.array([1, .2, .6], np.float32)
    c0 = np.array([.02, .02, .1], np.float32)
    return c0 + (c1 - c0) * up


def m_mirror(f, t, ctx):
    o = m_teal(f, t, ctx)
    o[:, W // 2:] = o[:, : W - W // 2][:, ::-1]
    i, _ = last_beat(t)
    if i % 2:
        o[H // 2:] = o[: H - H // 2][::-1]
    return o


def m_glitch(f, t, ctx):
    o = m_raw(f, t, ctx) * np.array([1, .95, 1.05], np.float32)
    fi = ctx["fi"]
    r = np.random.default_rng(fi // 2)
    sh = 10 + int(oenv_at(t) * 14)
    o = np.stack([np.roll(o[..., 0], sh, 1), o[..., 1], np.roll(o[..., 2], -sh, 1)], -1)
    for _ in range(r.integers(4, 10)):
        y0 = int(r.integers(0, H - 10))
        hh = int(r.integers(4, 60))
        o[y0:y0 + hh] = np.roll(o[y0:y0 + hh], int(r.integers(-120, 120)), 1)
        if r.random() < .25:
            o[y0:y0 + hh] = 1 - o[y0:y0 + hh]
    return o


def m_echo(f, t, ctx):
    get = ctx["get"]
    g0 = enh(gray(f))
    g1, g2, g3 = (enh(gray(get(t - k / FPS))) for k in (5, 10, 15))
    cols = [np.array(c, np.float32) for c in ((1, 1, 1), (1, .15, .2), (0, .9, 1), (.9, .2, 1))]
    o = g0[..., None] * cols[0]
    for g, c, a in ((g1, cols[1], .85), (g2, cols[2], .65), (g3, cols[3], .45)):
        o = np.maximum(o, g[..., None] * c * a)
    return o


SEGS = [
    ("RAW", m_raw, "render mv --mode raw", None),
    ("GLYPH_न", m_glyph_n, "render mv --mode glyph_न", None),
    ("SOBEL", m_sobel, "render mv --mode sobel   # edges sharper than the bars", "import encore_abj.flow  ✓"),
    ("NEON", m_neon, "render mv --mode neon", None),
    ("HALFTONE", m_halftone, "render mv --mode halftone", "import calm.flow --bars ∞  ✓"),
    ("BAYER_8x8", m_bayer, "render mv --mode bayer_8x8", None),
    ("THERMAL", m_thermal, "render mv --mode thermal   # crowd temp: critical", "sudo rm -rf ~/fake_rappers  ✓"),
    ("MIRROR", m_mirror, "render mv --mode mirror --axis both", None),
    ("GLITCH", m_glitch, "render mv --mode glitch --stutter", "ping seedhe.maut  → 0% loss  ✓"),
    ("ECHO", m_echo, "render mv --mode echo --trail 15", None),
    ("GLYPH_म", m_glyph_m, "render mv --mode glyph_म", "git commit -m 'seedhe maut never miss'  ✓"),
    ("BLOOD", m_blood, "render mv --mode blood   # सीधे मौत", None),
]


def seg_at(t):
    k = int(np.searchsorted([AU["beats"][B0 + 8 * i] for i in range(12)], t, "right")) - 1
    k = max(0, min(11, k))
    return k, float(AU["beats"][B0 + 8 * k])


# ----------------------------------------------------------------------------
# terminal log
# ----------------------------------------------------------------------------
def build_log():
    ev = [(0.5, "claude: open ~/seedhe_maut/"),
          (1.7, "2 members · encore abj × calm · track: namastute"),
          (6.3, "ls ~/seedhe_maut/ --sort hype"),
          (10.9, "play namastute.mp4 --from 0:12 --mode chaos")]
    bt = AU["beats"]
    for k, (_, _, cmd, extra) in enumerate(SEGS):
        s = float(bt[B0 + 8 * k])
        ev.append((s, cmd))
        if extra:
            ev.append((float(bt[B0 + 8 * k + 4]), extra))
    ev += [(T["spell"], "spell 'seedhe maut'"),
           (T["crowd"], "render crowd --mode glyph_म"),
           (T["tera"], "echo 'tera bhai seedhe maut'")]
    return ev


LOG = []


def draw_log(d, t, x=22, y0=26, n=3):
    shown = [(a, s) for a, s in LOG if a <= t][-n:]
    for i, (a, s) in enumerate(shown):
        newest = i == len(shown) - 1
        k = int((t - a) * 45) + 1
        txt = s[:k] if newest else s
        base = y0 + i * 17
        col = WHITE_I if newest else (150, 150, 150)
        d.text((x, base), ">", font=font(MONO_B, 13), fill=ORANGE_I, anchor="ls")
        xe = draw_mixed(d, x + 14, base, txt, 13, col)
        if newest and (int(t * 3) % 2 == 0 or k < len(s)):
            d.rectangle([xe + 2, base - 11, xe + 9, base + 2], fill=ORANGE_I)


# ----------------------------------------------------------------------------
# scenes
# ----------------------------------------------------------------------------
GLYPH_M = None
TBL_ROWS = [("2 Ka Pahada", "2017", 62), ("Bayaan", "2018", 71), ("न", "2021", 86),
            ("Nayaab", "2022", 93), ("Lunch Break", "2023", 97), ("Namastute", "NOW", 999)]


def ease(x):
    x = min(max(x, 0), 1)
    return 1 - (1 - x) ** 3


def scene_intro(t, fi):
    img = np.zeros((H, W, 3), np.float32) + DOTGRID[..., None] * .05
    # red line through the glyph line
    if t > 2.25:
        p = ease((t - 2.25) / .45)
        ly = H // 2 + 14
        half = int(p * W / 2)
        line = np.zeros((H, W), np.float32)
        line[ly - 1:ly + 1, W // 2 - half:W // 2 + half] = 1
        a = 1.0 if t < 6.0 else max(0, 1 - (t - 6.0) / .25)
        img += glow(line, core=(1, .6, .5), col=np.array([1, .2, .1], np.float32), k=(2, 6), a=(1, .6)) * a * .8
    if 2.8 <= t < 6.2:
        r = np.random.default_rng(fi)
        on = 1.0
        if t < 3.1:
            on = 1.0 if r.random() < .6 else .15
        if t > 6.0:
            on = max(0, 1 - (t - 6.0) / .2)
        i, dt = last_beat(t)
        g = GLYPH_M * on * (0.92 + .08 * r.random())
        sh = int(8 * math.exp(-dt / .08))
        if sh:
            g = np.stack([np.roll(g[..., 0], sh, 1), g[..., 1], np.roll(g[..., 2], -sh, 1)], -1)
        img += g
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil, "RGBA")
    draw_log(d, t, n=4)
    if 3.0 <= t < 6.1:
        a = int(255 * ease((t - 3.0) / .4))
        s1 = "[ म ]  SEEDHE MAUT  ·  DELHI"
        draw_tracked(d, W / 2 - tracked_w(s1, 14) / 2, H / 2 - 175, s1, 14, ORANGE_I + (a,))
        s2 = "ENCORE ABJ  ×  CALM  ·  NAMASTUTE"
        draw_tracked(d, W / 2 - tracked_w(s2, 14, 4, True) / 2, H / 2 + 185, s2, 14, WHITE_I + (a,), 4, True)
    if t >= 6.2:
        draw_table(d, t)
    return pil


def draw_table(d, t):
    x0, y0, rh = 120, 210, 58
    hdr = "#  PROJECTS · HYPE INDEX · 10.11.2026"
    draw_tracked(d, x0, y0 - 40, hdr, 13, (170, 170, 170), 2)
    bx0, bx1 = x0 + 420, x0 + 900
    cur = None
    for k, (name, yr, pct) in enumerate(TBL_ROWS):
        ta = 6.4 + k * .3
        if t < ta:
            continue
        a = ease((t - ta) / .25)
        y = y0 + k * rh
        dx = int((1 - a) * -40)
        al = int(255 * a)
        d.text((x0 + dx, y), f"{len(TBL_ROWS) - k:02d}", font=font(MONO, 12), fill=GREY_I + (al,), anchor="ls")
        nx = x0 + 52 + dx
        last = k == len(TBL_ROWS) - 1
        if is_deva(name[0]):
            d.text((nx, y + 6), name, font=font(DEVA, 34), fill=ORANGE_I + (al,), anchor="ls")
        else:
            d.text((nx, y), name[0], font=font(BIG, 30), fill=ORANGE_I + (al,), anchor="ls")
            d.text((nx + font(BIG, 30).getlength(name[0]), y), name[1:], font=font(BIG, 30),
                   fill=(WHITE_I if not last else ORANGE_I) + (al,), anchor="ls")
        d.text((x0 + 300 + dx, y), yr, font=font(MONO, 13), fill=GREY_I + (al,), anchor="ls")
        d.line([(bx0, y - 8), (bx1, y - 8)], fill=(255, 255, 255, int(40 * a)), width=1)
        tf = 8.0 + k * .45
        if t >= tf:
            cur = k
            p = ease((t - tf) / .4)
            if not last:
                ln = (pct / 100) * (bx1 - bx0) * p
                d.rectangle([bx0, y - 11, bx0 + ln, y - 5], fill=WHITE_I)
                d.text((bx1 + 20, y), f"{int(pct * p)}%", font=font(MONO, 13), fill=WHITE_I, anchor="ls")
            else:
                p2 = ease((t - tf) / .65)
                ln = p2 * (W - bx0)
                blink = (int(t * 8) % 2 == 0) or t < tf + .65
                d.rectangle([bx0, y - 13, bx0 + ln, y - 3], fill=ORANGE_I if blink else WHITE_I)
                if t > tf + .5:
                    d.text((bx1 - 10, y - 20), "999%  OVERFLOW", font=font(MONO_B, 13), fill=ORANGE_I, anchor="rs")
    if cur is not None:
        y = y0 + cur * rh
        bx, by = x0 - 14, y - 22
        d.line([(bx, by), (bx - 6, by), (bx - 6, by + 30), (bx, by + 30)], fill=ORANGE_I, width=2)
        ex = bx1 + 80
        d.line([(ex, by), (ex + 6, by), (ex + 6, by + 30), (ex, by + 30)], fill=ORANGE_I, width=2)


def zoom(img, z, dx=0, dy=0):
    M = np.float32([[z, 0, (1 - z) * W / 2 + dx], [0, z, (1 - z) * H / 2 + dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def post(img, t, fi, punch=True):
    if punch:
        i, dt = last_beat(t)
        s = 0.035 * math.exp(-dt / .10) * (1.8 if (i - B0) % 4 == 0 else 1)
        a = oenv_at(t)
        r = np.random.default_rng(fi)
        jx = jy = 0
        if a > 1.0:
            jx, jy = r.integers(-6, 7, 2)
        if s > .002 or jx or jy:
            img = zoom(img, 1 + s, jx, jy)
    a = oenv_at(t)
    sh = int(min(12, max(0, a - .55) * 14))
    if sh:
        img = np.stack([np.roll(img[..., 0], sh, 1), img[..., 1], np.roll(img[..., 2], -sh, 1)], -1)
    img = img * (VIG * SCAN)[..., None] + GRAIN[fi % 6][..., None]
    return np.clip(img, 0, 1)


def fmt(t):
    t = max(0, t)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def draw_hud(d, t, mode):
    m, L = 12, 26
    for x, y, sx, sy in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
        d.line([(x, y), (x + sx * L, y)], fill=(255, 255, 255, 110), width=2)
        d.line([(x, y), (x, y + sy * L)], fill=(255, 255, 255, 110), width=2)
    draw_tracked(d, 22, 86, f"{fmt(t)}  ·  SEEDHE MAUT  ·  NAMASTUTE", 11, (200, 200, 200), 2)
    # render mode
    draw_tracked(d, W - 150, 34, "RENDER MODE", 10, (190, 190, 190), 2)
    lab = "▶ " + mode
    wv = text_w(lab, 15, True)
    d.rectangle([W - 32 - wv - 12, 42, W - 30, 66], fill=(0, 0, 0, 150), outline=ORANGE_I, width=1)
    draw_mixed(d, W - 32 - wv - 6, 60, lab, 15, ORANGE_I, True)
    # vocal onsets box
    bx, by = 22, H - 118
    d.rectangle([bx, by, bx + 196, by + 62], fill=(0, 0, 0, 140), outline=(255, 255, 255, 140), width=1)
    draw_tracked(d, bx + 10, by + 17, "VOCAL ONSETS", 10, (200, 200, 200), 2)
    n = onset_count(T["drop"], t)
    rate = onset_count(t - 1, t)
    d.text((bx + 10, by + 52), f"{n:03d}", font=font(MONO_B, 30), fill=WHITE_I, anchor="ls")
    d.text((bx + 86, by + 50), f"{rate}/s", font=font(MONO, 13), fill=ORANGE_I, anchor="ls")
    _, dt = last_beat(t)
    rr = 3 + 4 * math.exp(-dt / .12)
    d.ellipse([bx + 140 - rr, by + 45 - rr, bx + 140 + rr, by + 45 + rr], fill=ORANGE_I)
    # bpm box
    bx2 = bx + 208
    d.rectangle([bx2, by, bx2 + 92, by + 62], fill=(0, 0, 0, 140), outline=(255, 255, 255, 140), width=1)
    draw_tracked(d, bx2 + 10, by + 17, "BPM", 10, (200, 200, 200), 2)
    d.text((bx2 + 10, by + 52), f"{AU['bpm']:.0f}", font=font(MONO_B, 30), fill=WHITE_I, anchor="ls")
    # scope
    sx0, sx1, sy = W - 440, W - 30, H - 92
    draw_tracked(d, sx1 - 160, sy - 26, "AUDIO · LIVE TAP", 10, (200, 200, 200), 2)
    rt, rms = AU["rt"], AU["rms"]
    i1 = np.searchsorted(rt, t)
    i0 = max(0, i1 - 260)
    seg = rms[i0:i1]
    if len(seg) > 2:
        seg = seg[:: max(1, len(seg) // 130)]
        xs = np.linspace(sx0, sx1, len(seg))
        pts = []
        for j, (x, v) in enumerate(zip(xs, seg)):
            yv = sy - (v ** .8) * 26 * (1 if j % 2 else -.6)
            if pts:
                pts.append((x, pts[-1][1]))
            pts.append((x, yv))
        d.line(pts, fill=ORANGE_I, width=2)
    d.text((W - 30, H - 52), f"{fmt(t)} / {fmt(SRC_DUR)}", font=font(MONO, 12), fill=(220, 220, 220), anchor="rs")
    # progress
    py = H - 34
    d.line([(22, py), (W - 22, py)], fill=(255, 255, 255, 70), width=2)
    px = 22 + (W - 44) * min(1, t / SRC_DUR)
    d.line([(22, py), (px, py)], fill=ORANGE_I, width=2)
    d.rectangle([px - 3, py - 5, px + 3, py + 5], fill=ORANGE_I)
    d.rectangle([24, py + 7, 27, py + 17], fill=(220, 220, 220))
    d.rectangle([31, py + 7, 34, py + 17], fill=(220, 220, 220))


def scene_footage(t, fi, get):
    k, s0 = seg_at(t)
    name, fn, _, _ = SEGS[k]
    ts = t
    if name == "GLITCH":
        ts = math.floor(t * 10) / 10
    f = get(ts)
    img = fn(f, t, dict(get=get, fi=fi))
    # glitch tear between modes on the 2 frames after a cut, then flash
    dt = t - s0
    if dt < 0.12 and k > 0:
        prev = SEGS[k - 1][1](f, t, dict(get=get, fi=fi))
        cut = int(H * (dt / .12))
        img = img.copy()
        img[cut:] = prev[cut:]
        ln = np.zeros((H, W), np.float32)
        ln[max(0, cut - 1):cut + 2] = 1
        img = img + glow(ln, core=(1, .7, .6), col=np.array([1, .25, .1], np.float32), k=(3, 9), a=(1, .6))
    if t - T["drop"] < .25:
        img = img + (1 - (t - T["drop"]) / .25) ** 2
    if name in ("GLITCH", "BLOOD") and oenv_at(t) > 1.6:
        img = 1 - np.clip(img, 0, 1)
    img = post(img, t, fi)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil, "RGBA")
    draw_log(d, t)
    draw_hud(d, t, name)
    return pil


def scene_transition(t, fi, get):
    """11.6 -> drop: tear the table apart, footage bleeding through in glyph mode."""
    base = np.asarray(scene_intro(t, fi), np.float32) / 255
    p = (t - 11.6) / (T["drop"] - 11.6)
    r = np.random.default_rng(fi)
    foot = m_glyph_n(get(t), t, {}) * p
    out = np.maximum(base, foot)
    for _ in range(int(4 + p * 14)):
        y0 = int(r.integers(0, H - 8))
        hh = int(r.integers(3, 40))
        out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], int(r.integers(-200, 200) * p), 1)
    sh = int(4 + p * 20)
    out = np.stack([np.roll(out[..., 0], sh, 1), out[..., 1], np.roll(out[..., 2], -sh, 1)], -1)
    out = post(out, t, fi, punch=False)
    return Image.fromarray((out * 255).astype(np.uint8))


LET = {}


def letter_layers():
    for ch in ("स", "म"):
        m = text_mask(ch, DEVA, 340, W / 2, H / 2 - 30)
        LET[ch] = m
        er = cv2.erode((m > .5).astype(np.uint8), np.ones((7, 7), np.uint8))
        LET[ch + "_o"] = np.clip(m - er.astype(np.float32), 0, 1)


def shift(a, dx):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(a, M, (W, H))


def scene_spell(t, fi, get):
    r = t - T["spell"]
    f = get(t)
    fill = m_neon(f, t, {})
    i, dt = last_beat(t)
    fill = zoom(fill, 1.08 + 0.05 * math.exp(-dt / .1))
    img = DOTGRID[..., None] * .05 + np.zeros((H, W, 3), np.float32)
    slide = ease((r - 2.6) / .5)
    pos = {"स": -slide * W * .2, "म": W * .2}
    caps = []
    for ch, t_out, t_fill in (("स", 0.2, 1.1), ("म", 2.9, 3.7)):
        if r < t_out:
            continue
        dx = pos[ch]
        m = shift(LET[ch], dx)
        o = shift(LET[ch + "_o"], dx)
        wipe = ease((r - t_out) / .45)
        cut = int(H * wipe)
        o[cut:] = 0
        if r >= t_fill:
            fl = min(1, (r - t_fill) / .08)
            flick = 1.0 if (r - t_fill) > .3 or int((r - t_fill) * 30) % 2 == 0 else .3
            img = img * (1 - m[..., None] * fl) + fill * m[..., None] * fl * flick
            img += glow(o * .8, core=(1, .8, .7), col=ORANGE * .7, k=(3, 10), a=(1, .5))
        else:
            img += glow(o, core=(1, .55, .4), col=ORANGE, k=(3, 10, 24), a=(1, .7, .4))
            if cut < H:
                ln = np.zeros((H, W), np.float32)
                ln[max(0, cut - 1):cut + 1, int(W / 2 + dx - 200):int(W / 2 + dx + 200)] = 1
                img += glow(ln, core=(1, .6, .5), col=ORANGE, k=(3, 8), a=(1, .5)) * .8
        caps.append((ch, dx, t_fill + .2))
    img = post(img, t, fi, punch=False)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil, "RGBA")
    draw_log(d, t)
    ttl = "SEEDHE MAUT  =  सीधे मौत"
    lat, dev = mixed_fonts(14, True)
    tw = text_w(ttl, 14, True)
    draw_mixed(d, W / 2 - tw / 2, 40, ttl, 14, WHITE_I, True)
    for ch, dx, tc in caps:
        if r < tc:
            continue
        a = int(255 * ease((r - tc) / .3))
        w1, w2 = (f"{ch} से ", "सीधे") if ch == "स" else (f"{ch} से ", "मौत")
        f1 = font(DEVA, 40)
        tw = f1.getlength(w1 + w2)
        x = W / 2 + dx - tw / 2
        y = H / 2 + 205
        d.text((x, y), w1, font=f1, fill=WHITE_I + (a,), anchor="ls")
        d.text((x + f1.getlength(w1), y), w2, font=f1, fill=ORANGE_I + (a,), anchor="ls")
    if r > 4.5:
        a = int(255 * ease((r - 4.5) / .3))
        s = "S · SEEDHE (STRAIGHT)   —   M · MAUT (DEATH)"
        draw_tracked(d, W / 2 - tracked_w(s, 12) / 2, H / 2 + 250, s, 12, (200, 200, 200, a))
    return pil


def fig_segments():
    """two neon figures (own drawing): one swinging a stick, one sprinting, plus a ball."""
    segs = []

    def circ(cx, cy, r, n=28, start=-90):
        pts = [(cx + r * math.cos(math.radians(start + 360 * i / n)),
                cy + r * math.sin(math.radians(start + 360 * i / n))) for i in range(n + 1)]
        return [(pts[i], pts[i + 1]) for i in range(n)]
    # figure A
    segs += circ(70, 38, 17)
    segs += [((72, 56), (62, 112)), ((62, 112), (38, 168)), ((62, 112), (92, 162)),
             ((70, 70), (40, 52)), ((40, 52), (22, 22)), ((70, 70), (48, 40)),
             ((22, 22), (-10, -12))]
    # figure B
    segs += circ(160, 46, 16)
    segs += [((158, 63), (146, 112)), ((146, 112), (170, 166)), ((146, 112), (118, 160)),
             ((154, 76), (182, 98)), ((154, 76), (126, 92))]
    segs += circ(214, 158, 9, 16)
    return segs


FIG = None


def fig_mask(p, bounce=0.0, thick=5):
    m = np.zeros((H, W), np.uint8)
    s = 1.9
    ox, oy = W / 2 - 105 * s, H / 2 - 85 * s - bounce
    lens = [math.dist(a, b) for a, b in FIG]
    tot = sum(lens) * p
    for (a, b), l in zip(FIG, lens):
        if tot <= 0:
            break
        q = min(1, tot / l)
        tot -= l
        bx_, by_ = a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q
        cv2.line(m, (int(ox + a[0] * s), int(oy + a[1] * s)), (int(ox + bx_ * s), int(oy + by_ * s)), 255, thick, cv2.LINE_AA)
    return m.astype(np.float32) / 255


def scene_crowd(t, fi, get):
    r = t - T["crowd"]
    f = get(t)
    img = m_glyph_n(f, t, {}) * .8
    i, dt = last_beat(t)
    p = ease(r / 1.2)
    if r < 1.5:
        fm = fig_mask(p)
        img = img * (1 - fm[..., None] * .8) + glow(fm, core=(1, .6, .45), col=ORANGE, k=(4, 12, 28), a=(1, .8, .5))
    else:
        b = 14 * math.exp(-dt / .12)
        fm = fig_mask(1, b, 13)
        dark = cv2.GaussianBlur(fm, (0, 0), 30)
        img = img * (1 - np.clip(dark * 1.4, 0, .9))[..., None]
        img = img * (1 - fm[..., None]) + glow(fm, core=(1, .97, .92), col=ORANGE, k=(6, 16, 40), a=(1, .9, .6))
    img = post(img, t, fi)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil, "RGBA")
    draw_log(d, t)
    return pil


def scene_tera(t, fi, get):
    r = t - T["tera"]
    f = get(t)
    img = m_raw(f, t, {}) * 1.1
    i, dt = last_beat(t)
    inv = i % 2 == 1
    tm = text_mask("SEEDHE", BIG, 190, 50, H - 250, "ls")
    tm2 = text_mask("MAUT.", BIG, 190, 50, H - 80, "ls")
    a = ease(r / .25)
    if inv:
        knock = np.clip(tm + tm2, 0, 1)[..., None] * a
        img = img * (1 - knock) + (1 - img) * knock * np.array([1, .45, .25], np.float32)
    else:
        img = img * (1 - tm[..., None] * a) + tm[..., None] * a
        img = img * (1 - tm2[..., None] * a) + tm2[..., None] * a * ORANGE
    img = post(img, t, fi)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil, "RGBA")
    draw_log(d, t)
    draw_tracked(d, 56, H - 470, "NAMASTUTE · DELHI · 10+ SAAL", 14, ORANGE_I, 3, True)
    if r > .4:
        al = int(255 * ease((r - .4) / .3))
        d.text((W - 50, H - 90), "तेरा भाई सीधे मौत", font=font(DEVA, 46), fill=WHITE_I + (al,), anchor="rs")
    return pil


def scene_outro(t, fi):
    r = t - T["outro"]
    img = np.zeros((H, W, 3), np.uint8)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil, "RGBA")
    s = "for @seedhemaut  ·  स × म × claude"
    k = int(max(0, r - .4) * 22)
    txt = s[:k]
    tw = text_w(s, 22)
    x = W / 2 - tw / 2
    y = H / 2 + 8
    fade = 1 - max(0, (t - (DUR - .9)) / .9)
    al = int(255 * fade)
    xe = x
    for part, col in (("for ", WHITE_I), ("@seedhemaut", ORANGE_I), ("  ·  स × म × claude", WHITE_I)):
        if not txt:
            break
        seg = txt[: len(part)]
        txt = txt[len(part):]
        xe = draw_mixed(d, xe, y, seg, 22, col + (al,), True)
    if int(t * 3) % 2 == 0:
        d.rectangle([xe + 3, y - 20, xe + 14, y + 3], fill=ORANGE_I + (al,))
    return pil


# ----------------------------------------------------------------------------
# frame dispatch + decoding
# ----------------------------------------------------------------------------
class Src:
    def __init__(self, t0):
        self.t0 = max(0.0, t0)
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{self.t0:.3f}", "-i", SRC, "-vf",
             f"fps={FPS},scale={W}:{H}:flags=bicubic", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE)
        self.n = -1
        self.cache = {}

    def get(self, t):
        idx = max(0, int(round((t - self.t0) * FPS)))
        while self.n < idx:
            buf = self.p.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            self.n += 1
            self.cache[self.n] = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            self.cache.pop(self.n - 90, None)
        idx = min(idx, self.n)
        while idx not in self.cache:
            idx += 1
        return self.cache[idx].astype(np.float32) / 255

    def close(self):
        self.p.kill()


def render_frame(fi, src):
    t = fi / FPS
    if t < 11.6:
        pil = scene_intro(t, fi)
        if t < 11.6:
            arr = post(np.asarray(pil, np.float32) / 255, t, fi, punch=False)
            pil = Image.fromarray((arr * 255).astype(np.uint8))
    elif t < T["drop"]:
        pil = scene_transition(t, fi, src.get)
    elif t < T["spell"]:
        pil = scene_footage(t, fi, src.get)
    elif t < T["crowd"]:
        pil = scene_spell(t, fi, src.get)
    elif t < T["tera"]:
        pil = scene_crowd(t, fi, src.get)
    elif t < T["outro"]:
        pil = scene_tera(t, fi, src.get)
    else:
        pil = scene_outro(t, fi)
    return np.asarray(pil.convert("RGB"), np.uint8)


def init_globals(src, work):
    global SRC, WORK, AU, LOG, GLYPH_M, FIG
    SRC, WORK = src, work
    AU = analyse_audio(os.path.join(work, "audio.wav"))
    setup_timeline()
    LOG = build_log()
    GLYPH_M = glow(text_mask("म", DEVA, 300, W / 2, H / 2 - 10), core=(1, .92, .9),
                   col=np.array([1, .28, .12], np.float32), k=(6, 18, 44), a=(1.1, .8, .5))
    letter_layers()
    FIG = fig_segments()


def worker(args):
    k, f0, f1, out = args
    src = Src(f0 / FPS - 1.0)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
         "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        enc.stdin.write(render_frame(fi, src).tobytes())
        if (fi - f0) % 60 == 0:
            print(f"[w{k}] {fi}/{f1}", flush=True)
    enc.stdin.close()
    enc.wait()
    src.close()
    return out


def main():
    src, out, work = sys.argv[1:4]
    init_globals(src, work)
    print("timeline", {k: round(v, 2) for k, v in T.items()}, "bpm", round(AU["bpm"], 1), flush=True)
    if "--preview" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]
        os.makedirs(out, exist_ok=True)
        for t in ts:
            s = Src(t - 1.0)
            fi = int(round(t * FPS))
            Image.fromarray(render_frame(fi, s)).save(os.path.join(out, f"p_{t:06.2f}.png"))
            s.close()
        return
    nw = 4
    bounds = np.linspace(0, NF, nw + 1).astype(int)
    jobs = [(k, int(bounds[k]), int(bounds[k + 1]), os.path.join(work, f"chunk{k}.mp4")) for k in range(nw)]
    with mp.get_context("fork").Pool(nw) as pool:
        parts = pool.map(worker, jobs)
    lst = os.path.join(work, "chunks.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
         "-i", os.path.join(work, "audio_st.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
         "-af", f"atrim=0:{T['outro'] + 1.4},afade=t=in:st=0:d=0.2,"
                f"afade=t=out:st={T['outro'] - 0.2}:d=1.6,apad=whole_dur={DUR}",
         "-c:a", "aac", "-b:a", "192k", "-t", str(DUR), "-movflags", "+faststart", out], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
