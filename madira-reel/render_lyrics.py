#!/usr/bin/env python3
"""
MADIRA (Seedhe Maut, prod. Hurricane) - 61 s lyric reel, song 0:59.3 -> 2:00.3
(the hook's second pass, the plea, then Calm's verse).
"Ink of the Night", one scene per lyric line, lyrics as kinetic type.

The lyric text is supplied at run time (--lyrics) and is not part of this repo. Line times follow the song's 12-second / 8-bar blocks found
in the audio; word times snap to measured vocal onsets.

usage:
  python3 render_lyrics.py SONG_AUDIO OUT.mp4 WORKDIR FONTDIR --lyrics LYRICS.txt
  python3 render_lyrics.py SONG_AUDIO OUTDIR  WORKDIR FONTDIR --lyrics LYRICS.txt --preview 49,52,55
LYRICS.txt holds the sung lines, one per line, indexed by LINES[*]["idx"]; --timing gives the
measured start/end and word times for the lines to show (lines missing from it are skipped).
"""
import math
import multiprocessing as mp
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_reel as RR  # noqa: E402
import render_ink as RI   # noqa: E402
from render_reel import over, over_c, affine, draw_sprite, blur, metal, sm, eio, eout, ein, tsprite, tag_sprite  # noqa: E402

W, H, FPS = 1080, 1920, 24
T0, DUR = 59.3, 61.0
NF = int(round(DUR * FPS))
GW, GH, CELL = RI.GW, RI.GH, RI.CELL

hexc = RI.hexc
BG, INDIGO, VIOLET = RI.BG, RI.INDIGO, RI.VIOLET
SILVER, WINE, ROSE, MOON, BLOOD = RI.SILVER, RI.WINE, RI.ROSE, RI.MOON, RI.BLOOD
AMBER = hexc("#E8A25A")
INKS = RI.INKS

F = {}
FREEZES = [(70.5, 71.9), (106.12, 1e9)]
SLOW = [(81.0, 84.0)]  # the beat drops out: slow motion, not a stop


def ht(ts):
    h = ts - sum(max(0.0, min(ts, b) - a) for a, b in FREEZES if ts > a)
    return h - sum(0.7 * max(0.0, min(ts, b) - a) for a, b in SLOW if ts > a)


def frozen(ts):
    return any(a <= ts < b for a, b in FREEZES)


# --------------------------------------------------------------- lyrics -----
LYRICS = []  # loaded at run time from the --lyrics file (one line per entry, in order); not stored in the repo


def L(t0, t1, idx, scene, **kw):
    d = dict(t0=t0, t1=t1, idx=idx, text="", scene=scene)
    d.update(kw)
    return d


V = [96.1, 97.55, 99.0, 100.45, 101.9, 103.35, 104.75, 106.12]
LINES = [
    L(48.1, 51.1, 0, "road"),
    L(51.1, 54.1, 1, "drops", key="थम"),
    L(54.1, 57.1, 2, "hand"),
    L(57.1, 60.1, 3, "rise", key="मदिरा"),
    L(60.1, 62.85, 4, "backs"),
    L(62.85, 65.35, 5, "strike", key="गलती,"),
    L(65.35, 66.1, 6, "echo"),
    L(66.1, 68.6, 7, "wander"),
    L(68.6, 70.5, 8, "lantern", key="बहाना"),
    L(70.5, 72.1, 9, "yuhi"),
    L(72.1, 75.1, 10, "window"),
    L(75.1, 78.1, 11, "ripples", key="थम"),
    L(78.1, 81.1, 12, "envelope"),
    L(81.1, 84.1, 13, "under"),
    L(84.1, 87.1, 14, "rosepath", echo="(तेरी राह)"),
    L(87.1, 90.1, 15, "bloodmoon", echo="(थम चुका)"),
    L(90.1, 93.1, 16, "sky", echo="(तेरे नाम)"),
    L(93.1, 96.1, 17, "pool", key="मदिरा"),
    L(V[0], V[1], 18, "v_song"),
    L(V[1], V[2], 19, "v_lean"),
    L(V[2], V[3], 20, "v_rain"),
    L(V[3], V[4], 21, "v_flame"),
    L(V[4], V[5], 22, "v_cloud"),
    L(V[5], V[6], 23, "v_shake", key="दम"),
    L(V[6], V[7], 24, "v_pen"),
    L(V[7], 108.2, 25, "v_silent"),
    L(108.2, 111.0, 26, "v_busy"),
    L(111.0, 114.0, 27, "v_phone"),
    L(114.0, 117.0, 28, "v_wound"),
    L(117.0, 121.0, 29, "v_snake"),
]


def LI(idx):
    for ln in LINES:
        if ln["idx"] == idx:
            return ln
    return None


def line_at(ts):
    for ln in LINES:
        if ln["t0"] <= ts < ln["t1"]:
            return ln
    return LINES[-1] if ts >= LINES[-1]["t0"] else LINES[0]


def word_times(ln):
    words = ln["text"].split(" ")
    if ln.get("wt") and len(ln["wt"]) == len(words):
        return words, list(ln["wt"])
    span = min(ln["t1"] - ln["t0"], 2.6) * (0.55 if ln["scene"].startswith("v_") else 0.72)
    raw = [ln["t0"] + span * i / max(1, len(words)) for i in range(len(words))]
    on = RI.AU["on"]
    out = []
    for t in raw:
        c = on[(on > t - 0.16) & (on < t + 0.16)]
        out.append(float(c[np.argmin(np.abs(c - t))]) if len(c) else t)
    out[0] = max(out[0], ln["t0"])
    return words, out


# ----------------------------------------------------------------- type -----
LATIN = {"notob": "inter", "tiro": "corm", "kalam": "kalam", "rozha": "rozha", "anton": "anton"}


def tw(s, font, size):
    if not any("\u0900" <= c <= "\u097F" for c in s):
        font = LATIN.get(font, font)
    return tsprite(s, F[font], size, pad=max(10, size // 5))


def glow_text(img, sp, col, cx, cy, a=1.0, g=0.0, gcol=None):
    if a <= 0:
        return
    over_c(img, col, sp * a, cx, cy)
    if g > 0:
        h, w = sp.shape
        gl = np.zeros((h + 80, w + 80), np.float32)
        gl[40:40 + h, 40:40 + w] = sp
        over_c(img, gcol if gcol is not None else col, blur(gl, 14) * g * a, cx, cy, add=True)


def words_line(img, ln, ts, font, size, cy, col=None, key=None, keycol=None, keysize=None, cx=W / 2,
               rise=14, g=0.25, maxw=980, alpha=1.0):
    """word-by-word reveal of a line, centred, wrapped to maxw"""
    col = MOON if col is None else col
    words, times = word_times(ln)
    sps = []
    for wd in words:
        s = keysize if (key and wd.strip(",") == key.strip(",") and keysize) else size
        sps.append((wd, tw(wd, font, s), s))
    gap = size * 0.18
    rows, cur, cw = [], [], 0
    for item in sps:
        w = item[1].shape[1] - 2 * max(10, item[2] // 5)
        if cur and cw + gap + w > maxw:
            rows.append(cur)
            cur, cw = [], 0
        cur.append((item, w))
        cw += (gap if cw else 0) + w
    rows.append(cur)
    k = 0
    rh = [max(it[0][1].shape[0] for it in r) * 0.78 for r in rows]
    y = cy - sum(rh) / 2
    for r, hh in zip(rows, rh):
        tot = sum(w for _, w in r) + gap * (len(r) - 1)
        x = cx - tot / 2
        for (wd, sp, s), w in r:
            a = sm((ts - times[k]) / 0.16) * alpha
            iskey = key and wd.strip(",") == key.strip(",")
            c = (keycol if (iskey and keycol is not None) else col)
            glow_text(img, sp, c, x + w / 2, y + hh / 2 + rise * (1 - a), a, g * (2.2 if iskey else 1))
            x += w + gap
            k += 1
        y += hh


def echo_text(img, s, ts, t, cy, size=60, col=None):
    if ts < t:
        return
    col = SILVER if col is None else col
    a = sm((ts - t) / 0.25)
    sp = tw(s, "tiro", size)
    for k, (dx, al) in enumerate(((0, 0.75), (14, 0.30), (28, 0.14))):
        over_c(img, col, sp * a * al, W / 2 + dx * a, cy + dx * 0.5 * a)


# --------------------------------------------------------------- shapes -----
ASSET = {}


def person_back(h=440):
    S = (int(h * 0.62), int(h * 1.05))
    m = np.zeros((S[1], S[0]), np.float32)
    cx = S[0] / 2
    s = h / 440
    cv2.circle(m, (int(cx), int(52 * s)), int(36 * s), 1.0, -1, cv2.LINE_AA)
    pts = np.array([(cx - 20 * s, 80 * s), (cx + 20 * s, 80 * s), (cx + 26 * s, 100 * s), (cx + 92 * s, 122 * s), (cx + 100 * s, 150 * s),
                    (cx + 82 * s, 300 * s), (cx + 60 * s, 300 * s), (cx + 58 * s, 440 * s), (cx + 10 * s, 440 * s), (cx, 320 * s),
                    (cx - 10 * s, 440 * s), (cx - 58 * s, 440 * s), (cx - 60 * s, 300 * s), (cx - 82 * s, 300 * s), (cx - 100 * s, 150 * s),
                    (cx - 92 * s, 122 * s), (cx - 26 * s, 100 * s)], np.int32)
    cv2.fillPoly(m, [pts], 1.0, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), 0.8)


def walker(phase, h=180):
    S = (int(h * 0.8), int(h * 1.05))
    m = np.zeros((S[1], S[0]), np.uint8)
    cx = S[0] / 2
    s = h / 180
    cv2.circle(m, (int(cx), int(20 * s)), int(15 * s), 255, -1, cv2.LINE_AA)
    hip = (int(cx), int(100 * s))
    sh = (int(cx), int(42 * s))
    cv2.line(m, sh, hip, 255, int(16 * s), cv2.LINE_AA)
    for side in (1, -1):
        a = 0.45 * math.sin(phase) * side
        knee = (int(cx + math.sin(a) * 42 * s), int(100 * s + math.cos(a) * 42 * s))
        foot = (int(knee[0] + math.sin(a - 0.2 * abs(a)) * 44 * s), int(knee[1] + 44 * s))
        cv2.line(m, hip, knee, 255, int(12 * s), cv2.LINE_AA)
        cv2.line(m, knee, foot, 255, int(10 * s), cv2.LINE_AA)
        arm = (int(cx - math.sin(a) * 34 * s), int(42 * s + 56 * s))
        cv2.line(m, sh, arm, 255, int(9 * s), cv2.LINE_AA)
    return m.astype(np.float32) / 255


def rim(mask, col, k=5, strength=0.7):
    e = np.clip(mask - cv2.erode(mask, np.ones((k, k), np.uint8)), 0, 1)
    return e, blur(e, 6) * 0.5 + e


def build_assets():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ASSET["xx"], ASSET["yy"] = xx, yy
    sky = np.zeros((H, W, 3), np.float32) + BG
    sky += np.exp(-((yy - H * 0.30) / (H * 0.55)) ** 2)[..., None] * INDIGO * 0.9
    sky += np.exp(-((xx - W / 2) ** 2 + (yy - H * 1.05) ** 2) / (2 * (W * 0.7) ** 2))[..., None] * VIOLET * 0.22
    ASSET["sky"] = sky
    r = np.random.default_rng(2)
    S = 600
    my, mx = np.mgrid[0:S, 0:S].astype(np.float32)
    d = np.hypot(mx - S / 2, my - S / 2)
    disc = np.clip(220 - d, 0, 1.5) / 1.5
    tex = (0.86 + 0.10 * cv2.resize(r.random((12, 12)).astype(np.float32), (S, S), interpolation=cv2.INTER_CUBIC)
           + 0.05 * cv2.resize(r.random((50, 50)).astype(np.float32), (S, S), interpolation=cv2.INTER_CUBIC))
    limb = np.clip(1 - (d / 220) ** 2, 0, 1) ** 0.25
    ASSET["moon"] = (disc * tex * limb).astype(np.float32)
    ASSET["moon_halo"] = (np.exp(-np.clip(d - 220, 0, None) / 120) * np.clip(1 - d / 300, 0, 1) ** 2).astype(np.float32)
    pb = person_back()
    ASSET["back"] = pb
    ASSET["profiles"] = (RI.profile_mask(W * 0.27, H * 0.50, 330, 1), RI.profile_mask(W * 0.73, H * 0.50, 330, -1))
    ASSET["profiles_away"] = (RI.profile_mask(W * 0.27, H * 0.50, 330, -1), RI.profile_mask(W * 0.73, H * 0.50, 330, 1))
    ASSET["rose"] = RI.rose_sprite()
    ASSET["petals"] = [RI.petal_sprite(k) for k in range(6)]
    dm = np.zeros((40, 26), np.float32)
    cv2.ellipse(dm, (13, 22), (10, 15), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    dm = cv2.GaussianBlur(dm, (0, 0), 0.8)
    ASSET["drop"] = (metal(dm, bevel=4, strength=2.0, lo=INDIGO * 1.2, mid=SILVER * 0.85, hi=MOON * 1.2, spec=20), dm)
    rr = np.random.default_rng(51)
    ASSET["drops"] = np.stack([rr.uniform(-0.6, 1.6, 320), rr.uniform(-0.1, 1.1, 320), rr.uniform(0.4, 2.2, 320)], 1)
    bub = np.zeros((40, 40), np.float32)
    cv2.circle(bub, (20, 20), 15, 1.0, 2, cv2.LINE_AA)
    cv2.circle(bub, (15, 14), 4, 1.0, -1, cv2.LINE_AA)
    ASSET["bubble"] = cv2.GaussianBlur(bub, (0, 0), 0.7)
    # rain window bokeh + glass droplets
    bk = np.zeros((H, W, 3), np.float32) + BG * 1.4
    for k in range(46):
        c = [WINE * 1.3, ROSE, AMBER * 0.9, SILVER * 0.8, MOON * 0.7][k % 5]
        m = np.zeros((H, W), np.float32)
        cv2.circle(m, (int(rr.uniform(0, W)), int(rr.uniform(H * 0.1, H * 0.95))), int(rr.uniform(35, 110)), 1.0, -1, cv2.LINE_AA)
        bk += blur(m, 10)[..., None] * c * rr.uniform(0.15, 0.45)
    ASSET["bokeh"] = bk
    ASSET["bokeh_soft"] = blur(bk, 22)
    gd = np.zeros((H, W), np.float32)
    for _ in range(900):
        x, y, rad = rr.uniform(0, W), rr.uniform(0, H), rr.uniform(2, 9)
        cv2.circle(gd, (int(x), int(y)), int(rad), 1.0, 1, cv2.LINE_AA)
    ASSET["glassdrops"] = cv2.GaussianBlur(gd, (0, 0), 0.6)
    ASSET["streaks"] = np.stack([rr.uniform(0, W, 26), rr.uniform(0, 1, 26), rr.uniform(120, 360, 26)], 1)
    # envelope
    ew, eh = 820, 520
    paper, al = RR.aged_paper(ew, eh, 33, edge=16)
    fl = np.zeros((eh, ew), np.float32)
    cv2.line(fl, (0, 0), (ew // 2, int(eh * 0.55)), 1.0, 3, cv2.LINE_AA)
    cv2.line(fl, (ew, 0), (ew // 2, int(eh * 0.55)), 1.0, 3, cv2.LINE_AA)
    paper = paper * (1 - 0.25 * blur(fl, 2)[..., None])
    sm_ = np.zeros((130, 130), np.float32)
    cv2.circle(sm_, (65, 65), 52, 1.0, -1, cv2.LINE_AA)
    st = tw("म", "rozha", 60)
    y0, x0 = 65 - st.shape[0] // 2, 65 - st.shape[1] // 2
    sm2 = sm_.copy()
    sm2[y0:y0 + st.shape[0], x0:x0 + st.shape[1]] -= st * 0.4
    seal_rgb = metal(cv2.GaussianBlur(np.clip(sm2, 0, 1), (0, 0), 1), bevel=4, strength=1.8, lo=WINE * 0.2, mid=WINE * 1.3,
                     hi=np.array([0.9, 0.5, 0.55], np.float32), spec=14)
    over(paper, seal_rgb, cv2.GaussianBlur(sm_, (0, 0), 1), ew // 2 - 65, int(eh * 0.55) - 65)
    ASSET["envelope"] = (paper.astype(np.float32), al)


def moon(img, cx, cy, r, blood=0.0, a=1.0, halo=0.45):
    col = MOON * (1 - blood) + BLOOD * blood
    hcol = SILVER * 0.55 * (1 - blood) + BLOOD * 0.8 * blood
    s = r / 220
    M = affine(cx, cy, s, s, 0, 600, 600)
    draw_sprite(img, col * 0.85, ASSET["moon"], M, mul=a)
    Mh = affine(cx, cy, s * 2.2, s * 2.2, 0, 600, 600)
    draw_sprite(img, hcol, ASSET["moon_halo"], Mh, add=True, mul=a * halo)


# ----------------------------------------------------------- ink smoke ------
def simulate(path):
    store = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(NF, GH, GW, 3))
    dye = np.zeros((GH, GW, 3), np.float32)
    vx = np.zeros((GH, GW), np.float32)
    vy = np.zeros((GH, GW), np.float32)
    keys = RI.curl_keys(48)
    dt = 1 / FPS
    on, ostr, bass = RI.AU["on"], RI.AU["ostr"], RI.AU["bass"]
    strong = on[ostr > 0.6]
    sky = tw("तेरे नाम", "rozha", 230)
    skym = np.zeros((H, W), np.float32)
    over(skym[..., None], np.ones(1, np.float32), sky, W / 2 - sky.shape[1] / 2, H * 0.40 - sky.shape[0] / 2)
    skyg = cv2.resize(skym, (GW, GH), interpolation=cv2.INTER_AREA)
    xs = np.arange(GW, dtype=np.float32)[None, :]
    starts = [ln["t0"] for ln in LINES]
    for fi in range(NF):
        ts = T0 + fi / FPS
        if not frozen(ts):
            sdt = dt * (0.3 if any(a <= ts < b for a, b in SLOW) else 1.0)
            t_prev = ts - dt
            h = ht(ts)
            kf = (h - T0) / 1.5
            k0 = int(kf) % (len(keys) - 1)
            f = sm(kf - int(kf))
            cx_ = keys[k0][0] * (1 - f) + keys[k0 + 1][0] * f
            cy_ = keys[k0][1] * (1 - f) + keys[k0 + 1][1] * f
            vs = LI(18)["t0"]
            amp = 10 if ts < vs else 22
            vx = vx * 0.9 + cx_ * amp * 0.3
            vy = vy * 0.9 + cy_ * amp * 0.3
            for k, t in enumerate(starts):
                if t_prev < t <= ts or (fi == 0 and k == 0):
                    rr = np.random.default_rng(k + 7)
                    for _ in range(2):
                        px, py = rr.uniform(GW * 0.05, GW * 0.95), rr.choice([rr.uniform(GH * 0.04, GH * 0.2), rr.uniform(GH * 0.8, GH * 0.96)])
                        dye[..., int(rr.integers(0, 3))] += RI.gauss(px, py, 6) * 1.2
                        RI.radial(vx, vy, px, py, 50, 18)
            sk = LI(16)["t0"] if LI(16) else -1
            if sk > 0 and sk <= ts < sk + 2.5:
                u = (ts - sk) / 1.6
                front = np.clip((u * (GW + 20) - 10 - xs) / 6, 0, 1)
                dye[..., 0] += skyg * front * 0.10
                dye[..., 2] += skyg * front * 0.03
            if ts >= vs:
                for o in strong[(strong > t_prev) & (strong <= ts)]:
                    j = int(o * 1000)
                    rr = np.random.default_rng(j)
                    px, py = rr.uniform(GW * 0.08, GW * 0.92), rr.choice([rr.uniform(GH * 0.06, GH * 0.26), rr.uniform(GH * 0.74, GH * 0.94)])
                    s = float(np.interp(o, on, ostr))
                    dye[..., j % 3] += RI.gauss(px, py, 2.6 + 1.2 * s) * (0.9 + 0.6 * s)
                    RI.radial(vx, vy, px, py, 120 + 80 * s, 9)
                    RI.swirl(vx, vy, px + 5, py, (1 if j % 2 else -1) * 150, 11)
            pt = LI(24)["t0"] + 0.4
            if t_prev < pt <= ts:
                dye[..., 1] += RI.gauss(GW / 2, GH * 0.42, 9) * 2.0
                RI.radial(vx, vy, GW / 2, GH * 0.42, 300, 40)
            vx, vy = RI.advect(vx, vx, vy, sdt), RI.advect(vy, vx, vy, sdt)
            dye = RI.advect(dye, vx, vy, sdt)
            dye = cv2.GaussianBlur(dye, (0, 0), 0.32) * (0.988 if ts < vs else 0.984)
            np.clip(dye, 0, 4, out=dye)
        store[fi] = (np.sqrt(np.clip(dye / 4.0, 0, 1)) * 255 + 0.5).astype(np.uint8)
    store.flush()


def ink(img, store, fi, mul=1.0, light=0.8):
    if mul <= 0:
        return img
    dye = (store[fi].astype(np.float32) / 255) ** 2 * 4.0 * mul
    dens = np.clip(cv2.resize(dye, (W, H), interpolation=cv2.INTER_CUBIC), 0, None)
    tot = dens.sum(-1)
    colr = (dens @ INKS) / (tot[..., None] + 1e-4)
    em = (1 - np.exp(-tot * 0.65))[..., None] * colr * light
    return img * np.exp(-tot * 0.55)[..., None] + em * 1.1


# --------------------------------------------------------------- scenes -----
def lt(ln, ts):
    return ts - ln["t0"], (ts - ln["t0"]) / (ln["t1"] - ln["t0"])


def road_layer(img, ts, hz, mx, bright=1.0):
    xx, yy = ASSET["xx"], ASSET["yy"]
    d = yy - hz
    below = (d > 0).astype(np.float32)
    water = np.clip(d / (H - hz), 0, 1)
    img += (below * (0.02 + 0.05 * water))[..., None] * INDIGO * 3
    halfw = 8 + np.clip(d, 0, None) * 0.55
    m = np.clip((halfw - np.abs(xx - mx)) / (10 + np.clip(d, 0, None) * 0.25), 0, 1) * below
    k = 900.0 / (np.clip(d, 0, None) + 25)
    sh = 0.5 + 0.5 * np.sin(k * 6.0 - ht(ts) * 4.0 + np.sin(xx * 0.05 + yy * 0.01) * 1.5)
    b = (0.35 + 0.65 * np.exp(-np.clip(d, 0, None) / 500)) * (0.45 + 0.55 * sh ** 3)
    img += (m * b * bright)[..., None] * SILVER * 0.9


def person_on(img, cx, cy, h=440, lightcol=None):
    pb = ASSET["back"]
    s = h / 440
    lightcol = SILVER if lightcol is None else lightcol
    e, g = rim(pb, lightcol)
    M = affine(cx, cy, s, s, 0, pb.shape[1], pb.shape[0])
    draw_sprite(img, BG * 0.6, pb, M)
    top = np.clip(1 - np.arange(pb.shape[0], dtype=np.float32) / pb.shape[0] * 1.3, 0, 1)[:, None]
    draw_sprite(img, lightcol, np.clip(g * top * 0.8, 0, 1), M, add=True)


def sc_road(img, ts, fi, ln, rose=False):
    t, u = lt(ln, ts)
    hz = H * 0.40
    moon(img, W / 2, hz - 190, 165, blood=0.25 if rose else 0.0)
    road_layer(img, ts, hz, W / 2)
    if rose:
        rgb, al = ASSET["rose"]
        M = affine(W / 2 + 10, H * 0.78, 0.72, 0.30, -8, al.shape[1], al.shape[0])
        draw_sprite(img, rgb, al, M)
        for k in range(6):
            p = ASSET["petals"][k]
            a = (t * 0.5 + k * 0.17) % 1
            x = W / 2 + 260 * math.sin(k * 2.1 + t * 0.6) * (0.4 + a)
            y = H * 0.45 + a * H * 0.5
            M = affine(x, y, 0.5, 0.5 * (0.3 + abs(math.sin(t * 2 + k))), t * 90 + k * 40, 90, 120)
            draw_sprite(img, p[0], p[1], M, mul=sm(a * 4) * (1 - sm((a - 0.85) * 6)))
    else:
        person_on(img, W / 2, H * 0.80, 470)
    words_line(img, ln, ts, "tiro", 72, H * 0.12, maxw=900)
    if ln.get("echo"):
        echo_text(img, ln["echo"], ts, ln.get("te", ln["t0"] + 2.2) - 0.1, H * 0.18)


def sc_drops(img, ts, fi, ln, ripples=False):
    t, u = lt(ln, ts)
    moon(img, W / 2, H * 0.24, 150)
    key = tw(ln["key"], "rozha", 420)
    a = sm(t / 0.3)
    shake = 0
    glow_text(img, key, MOON * 0.92, W / 2, H * 0.47, a, 0.35, SILVER)
    words_line(img, ln, ts, "tiro", 56, H * 0.66, maxw=860)
    if ripples:
        hz = H * 0.74
        img[int(hz):] = img[int(hz):] * 0.6 + INDIGO * 0.2
        rip = np.zeros((H, W), np.float32)
        for k in range(7):
            rad = 60 + k * 85
            cv2.ellipse(rip, (W // 2, int(H * 0.86)), (rad, int(rad * 0.22)), 0, 0, 360, 1.0 - k * 0.12, 2, cv2.LINE_AA)
        img += blur(rip, 1.2)[..., None] * SILVER * 0.55
        p = ASSET["petals"][2]
        M = affine(W / 2, H * 0.80, 0.9, 0.55, 18, 90, 120)
        draw_sprite(img, p[0], p[1], M)
        return
    dr = ASSET["drops"]
    rgb, m = ASSET["drop"]
    drift = ht(ts) * 0.0 + t * 0.02
    for x, y, z in dr:
        px = (x - drift * z) * W
        py = y * H
        if -40 < px < W + 40:
            s = 0.5 + 0.9 / z
            M = affine(px, py, s, s * 1.4, 0, m.shape[1], m.shape[0])
            draw_sprite(img, rgb, m, M, mul=min(1.0, 1.4 / z))


def sc_hand(img, ts, fi, ln, envelope=False):
    t, u = lt(ln, ts)
    words, times = word_times(ln)
    first = " ".join(words[:-2])
    last = " ".join(words[-2:])
    if envelope:
        rgb, al = ASSET["envelope"]
        rgb = rgb.copy()
        sp1 = tw(first, "kalam", 54)
        sp2 = tw(last, "kalam", 70)
        p1 = sm((ts - times[0]) / 1.2)
        p2 = sm((ts - times[-2]) / 0.9)
        over(rgb, RI.INKS[1] * 0.25, RI_reveal(sp1, p1) * 0.9, 90, 250)
        over(rgb, WINE * 0.8, RI_reveal(sp2, p2) * 0.95, 300, 330)
        rot = -6 + 5 * u
        M = affine(W / 2, H * 0.50 + 20 * math.sin(t * 0.8), 1.05, 1.05, rot, al.shape[1], al.shape[0])
        sh = cv2.GaussianBlur(al, (0, 0), 16)
        draw_sprite(img, np.zeros(3, np.float32), sh * 0.7, affine(W / 2 + 24, H * 0.52 + 30, 1.05, 1.05, rot, al.shape[1], al.shape[0]))
        draw_sprite(img, rgb * 0.92, al, M)
        moon(img, W * 0.80, H * 0.16, 90, a=0.9)
        return
    yy = ASSET["yy"]
    for k in range(9):
        y = int(H * 0.36 + k * 74)
        img[y:y + 1] += SILVER * 0.05
    sp1 = tw(first, "kalam", 76)
    sp2 = tw(last, "kalam", 120)
    p1 = sm((ts - times[0]) / 1.3)
    p2 = sm((ts - times[-2]) / 0.9)
    r1 = RI_reveal(sp1, p1)
    r2 = RI_reveal(sp2, p2)
    glow_text(img, r1, SILVER, W / 2, H * 0.42, 1.0, 0.45)
    glow_text(img, r2, ROSE, W / 2, H * 0.53, 1.0, 0.7)
    for sp, p, cy in ((sp1, p1, H * 0.42), (sp2, p2, H * 0.53)):
        if 0.02 < p < 0.98:
            x = W / 2 - sp.shape[1] / 2 + p * sp.shape[1]
            g = np.zeros((H, W), np.float32)
            cv2.circle(g, (int(x), int(cy + 20)), 10, 1.0, -1)
            img += blur(g, 12)[..., None] * MOON * 0.9


def RI_reveal(sp, p):
    w = sp.shape[1]
    xs = np.arange(w, dtype=np.float32)
    return sp * np.clip((p * (w + 40) - 20 - xs) / 18, 0, 1)[None, :]


def underwater(img, level, ts, tint=0.85):
    """wine liquid below y=level: tint, refraction wobble, rays, bubbles, meniscus"""
    lv = int(np.clip(level, -10, H))
    if lv >= H:
        return img
    lv0 = max(0, lv)
    reg = img[lv0:].copy()
    hh = reg.shape[0]
    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(hh, dtype=np.float32)[:, None]
    mx = (xs + 6 * np.sin(ys / 40 + ts * 2.2)).astype(np.float32)
    my = (ys + 4 * np.sin(xs / 55 + ts * 1.7)).astype(np.float32)
    reg = cv2.remap(reg, np.ascontiguousarray(np.broadcast_to(mx, (hh, W))), np.ascontiguousarray(np.broadcast_to(my, (hh, W))), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    depth = np.clip(ys / 900, 0, 1)
    wine = WINE * 0.55
    reg = reg * (1 - tint * 0.7) * np.array([1.0, 0.55, 0.6], np.float32) + wine * (0.25 + 0.5 * depth[..., None]) * tint
    rays = 0.5 + 0.5 * np.sin(xs / 37 + ts * 0.6 + np.sin(xs / 91) * 2)
    reg += ((rays ** 6) * np.exp(-ys / 500) * 0.25)[..., None] * ROSE
    img[lv0:] = reg
    rr = np.random.default_rng(int(lv) // 400 + 3)
    bub = ASSET["bubble"]
    for k in range(40):
        bx = rr.uniform(0, W)
        sp = rr.uniform(60, 160)
        by = H - ((ts * sp + rr.uniform(0, H)) % (H - lv0 + 50))
        if by > lv0 + 10:
            s = rr.uniform(0.3, 0.9)
            M = affine(bx + 10 * math.sin(ts * 2 + k), by, s, s, 0, 40, 40)
            draw_sprite(img, MOON * 0.8, bub, M, add=True, mul=0.5)
    if 0 <= lv < H:
        wav = (lv + 5 * np.sin(np.arange(W) / 33 + ts * 4)).astype(np.float32)
        men = np.zeros((H, W), np.float32)
        pts = np.stack([np.arange(W), wav], 1).astype(np.int32)
        cv2.polylines(men, [pts], False, 1.0, 2, cv2.LINE_AA)
        img += (blur(men, 1) * 0.9 + blur(men, 10) * 0.5)[..., None] * ROSE
    return img


def sc_rise(img, ts, fi, ln):
    t, u = lt(ln, ts)
    sc_road_bg(img, ts)
    level = H - H * 0.92 * eio(u * 1.1)
    underwater(img, level, ts)
    words_line(img, ln, ts, "tiro", 64, H * 0.14, key=ln["key"], keycol=ROSE, keysize=170, maxw=900)


def sc_road_bg(img, ts):
    hz = H * 0.40
    moon(img, W / 2, hz - 190, 165)
    road_layer(img, ts, hz, W / 2)
    person_on(img, W / 2, H * 0.80, 470)


def sc_under(img, ts, fi, ln):
    t, u = lt(ln, ts)
    img[:] = img * 0.4 + WINE * 0.12
    img[:] = underwater(img, -5, ts * 0.4, tint=1.0)
    words, times = word_times(ln)
    y0 = H * 0.30
    for k, (wd, tm) in enumerate(zip(words, times)):
        a = sm((ts - tm) / 0.3)
        if a <= 0:
            continue
        big = wd == "मदिरा"
        sp = tw(wd, "rozha" if big else "tiro", 200 if big else 84)
        sink = (ht(ts) - ht(tm)) * (22 + 9 * k)
        x = W / 2 + 160 * math.sin(k * 1.7) * (0 if big else 1)
        y = y0 + k * 150 + sink
        glow_text(img, sp, ROSE if big else MOON * 0.9, x, y, a * (1 - sm((y - H * 0.95) / 100)), 0.6 if big else 0.25)
    for k in range(5):
        p = ASSET["petals"][k]
        y = (H * 0.1 + k * 330 + ht(ts) * 30) % H
        M = affine(W * (0.15 + 0.17 * k), y, 0.6, 0.45, ts * 25 + k * 60, 90, 120)
        draw_sprite(img, p[0], p[1], M, mul=0.85)


def sc_backs(img, ts, fi, ln):
    t, u = lt(ln, ts)
    f = eio((t - 0.9) / 1.0)
    for k in range(2):
        away = ASSET["profiles_away"][k]
        face = ASSET["profiles"][k]
        m = away * (1 - f) + face * f
        col = SILVER if k == 0 else ROSE
        e, g = rim(m, col)
        img[:] = img * (1 - m[..., None] * 0.85) + BG * 0.5 * m[..., None]
        img += g[..., None] * col * 0.7
    moon(img, W / 2, H * 0.17, 120)
    words, times = word_times(ln)
    left = " ".join(words[:3])
    right = " ".join(words[3:])
    a1 = sm((ts - times[0]) / 0.2)
    a2 = sm((ts - times[3]) / 0.2)
    glow_text(img, tw(left, "tiro", 84), SILVER, W * 0.28, H * 0.84, a1, 0.3)
    glow_text(img, tw(right, "tiro", 84), ROSE, W * 0.72, H * 0.84, a2, 0.3)


def sc_strike(img, ts, fi, ln):
    t, u = lt(ln, ts)
    words, times = word_times(ln)
    top = " ".join(words[:3])
    bot = " ".join(words[3:])
    a1 = sm((ts - times[0]) / 0.2)
    sp = tw(top, "rozha", 120)
    glow_text(img, sp, MOON * 0.92, W / 2, H * 0.42, a1, 0.25)
    gl = tw("गलती,", "rozha", 120)
    p = eout((ts - times[2] - 0.35) / 0.45)
    if p > 0:
        x0 = W / 2 + sp.shape[1] / 2 - gl.shape[1] + 20
        x1 = x0 + (gl.shape[1] - 40) * p
        st = np.zeros((H, W), np.float32)
        pts = np.array([(x, H * 0.43 + 10 * math.sin(x / 30)) for x in np.linspace(x0, x1, 40)], np.int32)
        cv2.polylines(st, [pts], False, 1.0, 16, cv2.LINE_AA)
        st = blur(st, 1.2)
        img[:] = img * (1 - st[..., None]) + WINE * 1.3 * st[..., None]
        img += blur(st, 12)[..., None] * ROSE * 0.4
    a2 = sm((ts - times[3]) / 0.2)
    glow_text(img, tw(bot, "tiro", 76), SILVER, W / 2, H * 0.55, a2, 0.3)


def sc_echo(img, ts, fi, ln):
    t, u = lt(ln, ts)
    sp = tw("तू भी", "rozha", 300)
    a = sm(t / 0.12)
    for k, (dy, al) in enumerate(((0, 1.0), (60, 0.35), (120, 0.15))):
        ak = a if k == 0 else sm((t - 0.35) / 0.15) * al
        glow_text(img, sp, ROSE if k == 0 else SILVER, W / 2, H * 0.45 + dy * ak, ak * (1 if k == 0 else 1), 0.5 if k == 0 else 0)


def sc_wander(img, ts, fi, ln):
    t, u = lt(ln, ts)
    hz = H * 0.70
    img[int(hz):] = img[int(hz):] * 0.55
    img[int(hz):int(hz) + 2] += SILVER * 0.35
    moon(img, W * 0.78, H * 0.20, 110)
    x = W * 0.1 + W * 0.8 * (t / 2.5)
    for k in range(int(t / 0.18)):
        fx = W * 0.1 + W * 0.8 * (k * 0.18 / 2.5)
        a = math.exp(-(t - k * 0.18) / 1.2)
        g = np.zeros((40, 40), np.float32)
        cv2.ellipse(g, (20, 20), (9, 5), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        over_c(img, ROSE, blur(g, 1.5) * 0.8 * a, fx, hz + 8 + (6 if k % 2 else -6))
    wm = walker(t * 7.5)
    over_c(img, BG * 0.4, wm, x, hz - wm.shape[0] / 2 + 6)
    e, gg = rim(wm, SILVER)
    over_c(img, SILVER, np.clip(gg, 0, 1) * 0.7, x, hz - wm.shape[0] / 2 + 6, add=True)
    words, times = word_times(ln)
    a1 = sm((ts - times[0]) / 0.2)
    a2 = sm((ts - times[3]) / 0.2)
    glow_text(img, tw(" ".join(words[:3]), "tiro", 74), MOON * 0.9, W / 2, H * 0.36, a1, 0.25)
    glow_text(img, tw(" ".join(words[3:]), "kalam", 110), ROSE, W / 2 + 30 * math.sin(t * 2.2), H * 0.45 + 12 * math.sin(t * 3.1), a2, 0.5)


def sc_lantern(img, ts, fi, ln, frozen_=False):
    t, u = lt(ln, ts) if not frozen_ else (1.9, 1.0)
    for k in range(2):
        m = ASSET["profiles"][k]
        img[:] = img * (1 - m[..., None] * 0.9) + BG * 0.35 * m[..., None]
        e, g = rim(m, AMBER)
        img += g[..., None] * AMBER * 0.55
    pulse = 1 + 0.12 * math.sin(ht(ts) * 5)
    g = np.zeros((H, W), np.float32)
    cv2.circle(g, (W // 2, int(H * 0.40)), int(26 * pulse), 1.0, -1, cv2.LINE_AA)
    img += (g * 1.0)[..., None] * MOON + (blur(g, 30) * 2.2 + blur(g, 120) * 2.5)[..., None] * AMBER * 0.5
    words, times = word_times(ln)
    if not frozen_:
        a1 = sm((ts - times[0]) / 0.2)
        a2 = sm((ts - times[-1]) / 0.25)
        glow_text(img, tw(" ".join(words[:-1]), "tiro", 54), MOON * 0.9, W / 2, H * 0.24, a1, 0.2)
        glow_text(img, tw("बहाना", "rozha", 190), AMBER, W / 2, H * 0.70, a2, 0.7)


def sc_yuhi(img, ts, fi, ln, store):
    sc_lantern(img, LI(8)["t1"] - 0.01, fi, LI(8), frozen_=True)
    glow_text(img, tw("बहाना", "rozha", 190), AMBER, W / 2, H * 0.70, 1.0, 0.7)
    img *= 0.55
    a = sm((ts - ln["t0"] - 0.02) / 0.25)
    glow_text(img, tw("यू ही", "tiro", 84), MOON, W / 2, H * 0.24, a, 0.3)
    echo_text(img, "(यू ही)", ts, ln["t0"] + 0.5, H * 0.29, 56)


def sc_window(img, ts, fi, ln):
    t, u = lt(ln, ts)
    img[:] = ASSET["bokeh_soft"] * 1.0
    fogA = np.full((H, W), 0.62, np.float32)
    words, times = word_times(ln)
    sp1 = tw(" ".join(words[:-2]), "kalam", 92)
    sp2 = tw(" ".join(words[-2:]), "kalam", 150)
    p1 = sm((ts - times[0]) / 1.2)
    p2 = sm((ts - times[-2]) / 0.8)
    clear = np.zeros((H, W), np.float32)
    over(clear[..., None], np.ones(1, np.float32), cv2.dilate(RI_reveal(sp1, p1), np.ones((5, 5), np.uint8)), W / 2 - sp1.shape[1] / 2, H * 0.38 - sp1.shape[0] / 2)
    over(clear[..., None], np.ones(1, np.float32), cv2.dilate(RI_reveal(sp2, p2), np.ones((7, 7), np.uint8)), W / 2 - sp2.shape[1] / 2, H * 0.50 - sp2.shape[0] / 2)
    fogA = fogA * (1 - 0.9 * blur(clear, 1.5))
    sharp = ASSET["bokeh"]
    fog = np.array([0.42, 0.44, 0.55], np.float32)
    fogA = np.clip(fogA * 1.25, 0, 0.85)
    img[:] = (sharp * 0.55 * (1 - fogA[..., None]) + (ASSET["bokeh_soft"] * 0.5 + fog * 0.6) * fogA[..., None])
    img += ASSET["glassdrops"][..., None] * MOON * 0.18
    st = ASSET["streaks"]
    lay = np.zeros((H, W), np.float32)
    for x, y0, sp in st:
        y = (y0 * H + ts * sp) % (H + 200) - 100
        cv2.line(lay, (int(x), int(y - 120)), (int(x + 3), int(y)), 0.5, 2, cv2.LINE_AA)
        cv2.circle(lay, (int(x + 3), int(y)), 5, 1.0, -1, cv2.LINE_AA)
    img += blur(lay, 1.0)[..., None] * MOON * 0.35
    img *= 0.85


def sc_bloodmoon(img, ts, fi, ln):
    t, u = lt(ln, ts)
    moon(img, W / 2, H * 0.36, 330, blood=0.25 + 0.55 * eio(u), halo=0.7)
    words_line(img, ln, ts, "tiro", 70, H * 0.66, maxw=900)
    echo_text(img, ln["echo"], ts, ln.get("te", ln["t0"] + 2.2) - 0.1, H * 0.74)


def sc_sky(img, ts, fi, ln):
    t, u = lt(ln, ts)
    moon(img, W * 0.75, H * 0.14, 100, blood=0.5)
    words, times = word_times(ln)
    a1 = sm((ts - times[0]) / 0.3)
    glow_text(img, tw(" ".join(words[:-2]), "tiro", 64), MOON * 0.85, W / 2, H * 0.27, a1, 0.2)
    sp = tw("तेरे नाम", "rozha", 230)
    k = sm((t - 0.1) / 1.2) * (1 - sm((t - 1.9) / 1.0))
    glow_text(img, blur(sp, 2.5), SILVER, W / 2, H * 0.40, 0.55 * k, 0.8)
    echo_text(img, ln["echo"], ts, ln.get("te", ln["t0"] + 2.2) - 0.1, H * 0.56, 64, ROSE)


def sc_pool(img, ts, fi, ln):
    t, u = lt(ln, ts)
    hz = H * 0.52
    moon(img, W / 2, H * 0.26, 210, blood=0.85, halo=0.7)
    top = img[:int(hz)].copy()
    refl = top[::-1]
    hh = refl.shape[0]
    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(hh, dtype=np.float32)[:, None]
    mx = np.ascontiguousarray(np.broadcast_to(xs + 10 * np.sin(ys / 18 + ts * 3), (hh, W))).astype(np.float32)
    my = np.ascontiguousarray(np.broadcast_to(ys, (hh, W))).astype(np.float32)
    refl = cv2.remap(refl, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    n = min(hh, H - int(hz))
    img[int(hz):int(hz) + n] = refl[:n] * 0.55 * np.array([1, 0.5, 0.55], np.float32) + WINE * 0.12
    img[int(hz):int(hz) + 2] += ROSE * 0.6
    rgb, al = ASSET["rose"]
    ry = H * 0.62 + 300 * ein(u)
    M = affine(W / 2, ry, 0.7, 0.7, 10 * u, al.shape[1], al.shape[0])
    tmp = img.copy()
    draw_sprite(tmp, rgb, al, M)
    cut = int(hz) + 40
    img[:cut] = tmp[:cut]
    img[cut:] = img[cut:] * 0.6 + tmp[cut:] * 0.4 * np.array([1, 0.6, 0.65], np.float32)
    words_line(img, ln, ts, "tiro", 58, H * 0.43, key=ln["key"], keycol=ROSE, keysize=150, maxw=900)


# ---- verse: fast cards -------------------------------------------------------
def two_halves(img, ln, ts, split, size=104, col1=None, col2=None, y1=H * 0.40, y2=H * 0.52):
    words, times = word_times(ln)
    a = " ".join(words[:split])
    b = " ".join(words[split:])
    t = ts - ln["t0"]
    for s, tm, y, c in ((a, times[0], y1, MOON if col1 is None else col1), (b, times[split] if split < len(times) else times[-1], y2, ROSE if col2 is None else col2)):
        k = (ts - tm)
        if k < 0:
            continue
        pop = 1 + 0.18 * math.exp(-k / 0.08)
        sp = tw(s, "notob", int(size))
        M = affine(W / 2, y, pop, pop, 0, sp.shape[1], sp.shape[0])
        draw_sprite(img, c, sp, M, mul=sm(k / 0.06))
        g = np.zeros_like(sp)
        draw_sprite(img, c, blur(sp, 10), M, add=True, mul=0.35 * sm(k / 0.06))


def stroke(img, pts, col, w=6, a=1.0, g=0.5):
    m = np.zeros((H, W), np.float32)
    cv2.polylines(m, [np.array(pts, np.int32)], False, 1.0, w, cv2.LINE_AA)
    img += (blur(m, 0.8) * a)[..., None] * col + (blur(m, 12) * g * a)[..., None] * col


def sc_verse(img, ts, fi, ln):
    sc = ln["scene"]
    t, u = lt(ln, ts)
    if sc == "v_song":
        f = eio((u - 0.35) / 0.4)
        wave = [(W * 0.15 + i * W * 0.7 / 60, H * 0.68 + 60 * math.sin(i * 0.5 + ts * 10) * (1 - f)) for i in range(61)]
        house = []
        hp = [(0.15, 0.80), (0.15, 0.62), (0.5, 0.52), (0.85, 0.62), (0.85, 0.80), (0.15, 0.80)]
        for i in range(61):
            q = i / 60 * (len(hp) - 1)
            k = min(int(q), len(hp) - 2)
            fr = q - k
            house.append((W * (hp[k][0] + (hp[k + 1][0] - hp[k][0]) * fr), H * (hp[k][1] + (hp[k + 1][1] - hp[k][1]) * fr) - H * 0.02))
        pts = [(a[0] * (1 - f) + b[0] * f, a[1] * (1 - f) + b[1] * f) for a, b in zip(wave, house)]
        stroke(img, pts, SILVER, 6)
        two_halves(img, ln, ts, 3, col1=MOON, col2=ROSE, y1=H * 0.28, y2=H * 0.38)
    elif sc == "v_lean":
        lean = eout((u - 0.4) / 0.4) * 22
        for k, (x, ang, c) in enumerate(((W * 0.42, lean, ROSE), (W * 0.58, 0, SILVER))):
            m = np.zeros((H, W), np.float32)
            cx, cy = int(x), int(H * 0.70)
            M = cv2.getRotationMatrix2D((cx, cy + 180), -ang, 1)
            cv2.circle(m, (cx, cy - 150), 34, 1.0, -1, cv2.LINE_AA)
            cv2.rectangle(m, (cx - 38, cy - 100), (cx + 38, cy + 180), 1.0, -1)
            m = cv2.warpAffine(m, M, (W, H))
            e, g = rim(m, c)
            img[:] = img * (1 - m[..., None] * 0.8) + BG * 0.3 * m[..., None]
            img += g[..., None] * c * 0.8
        two_halves(img, ln, ts, 4, y1=H * 0.24, y2=H * 0.34, size=92)
    elif sc == "v_rain":
        f = eio((u - 0.45) / 0.25)
        lay = np.zeros((H, W), np.float32)
        rr = np.random.default_rng(int(ts * 24))
        for _ in range(140):
            x = rr.uniform(0, W)
            y = rr.uniform(H * 0.4, H)
            if f > 0 and abs(x - W / 2) < 260 * f and y < H * 0.60 + 300:
                continue
            cv2.line(lay, (int(x), int(y)), (int(x - 6), int(y + 38)), 1.0, 2, cv2.LINE_AA)
        img += blur(lay, 0.6)[..., None] * SILVER * 0.55
        if f > 0:
            pts = [(W / 2 + 260 * f * math.cos(a), H * 0.62 - 150 * f * math.sin(a)) for a in np.linspace(0, math.pi, 40)]
            stroke(img, pts + [pts[0]], ROSE, 8)
            stroke(img, [(W / 2, H * 0.62), (W / 2, H * 0.80), (W / 2 - 30, H * 0.83)], ROSE, 7)
        two_halves(img, ln, ts, 3, y1=H * 0.20, y2=H * 0.30)
    elif sc == "v_flame":
        f = eio((u - 0.45) / 0.3)
        g = np.zeros((H, W), np.float32)
        d = 170 * (1 - f)
        for sx in (-1, 1):
            cv2.circle(g, (int(W / 2 + sx * d), int(H * 0.68)), int(46 + 20 * f), 1.0, -1, cv2.LINE_AA)
        if f > 0.6:
            fl = np.zeros((H, W), np.float32)
            k = (f - 0.6) / 0.4
            pts = np.array([(W / 2 + 90 * math.sin(a) * (1 - a / 6.3) * k, H * 0.68 - a * 50 * k - 20 * math.sin(ts * 14 + a)) for a in np.linspace(0, 6.3, 50)], np.int32)
            cv2.fillPoly(fl, [np.vstack([pts, [(int(W / 2 - 90 * k), int(H * 0.68))], [(int(W / 2 + 90 * k), int(H * 0.68))]])], 1.0, cv2.LINE_AA)
            g = np.maximum(g, blur(fl, 6))
        img += (blur(g, 2) * 0.9)[..., None] * AMBER + (blur(g, 40) * 1.4)[..., None] * ROSE * 0.6
        two_halves(img, ln, ts, 4, y1=H * 0.22, y2=H * 0.32, size=96, col2=AMBER)
    elif sc == "v_cloud":
        m = np.zeros((H, W), np.float32)
        for dx, dy, r in ((-120, 0, 90), (0, -40, 120), (120, 0, 95), (-40, 40, 90), (60, 40, 90)):
            cv2.circle(m, (int(W / 2 + dx), int(H * 0.66 + dy)), r, 1.0, -1, cv2.LINE_AA)
        m = blur(m, 4)
        img[:] = img * (1 - m[..., None] * 0.7) + VIOLET * 0.25 * m[..., None]
        e, g = rim(m, SILVER)
        img += g[..., None] * SILVER * 0.4
        dy = (t * 600) % 400
        d = np.zeros((H, W), np.float32)
        cv2.ellipse(d, (W // 2, int(H * 0.76 + dy)), (10, 16), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        img += blur(d, 1)[..., None] * SILVER
        words_line(img, ln, ts, "notob", 84, H * 0.30, maxw=940, rise=0, g=0.2)
    elif sc == "v_shake":
        j = 10 * math.sin(ts * 70) * math.exp(-((ts - ln["t0"]) % 0.37) / 0.12)
        sub = np.zeros_like(img)
        words_line(sub, ln, ts, "notob", 84, H * 0.40, key="दम", keycol=ROSE, keysize=190, maxw=940, rise=0, g=0.2)
        img += np.roll(sub, int(j), 1)
    elif sc == "v_pen":
        rise = eout(u / 0.5)
        tip = (W / 2, H * 0.92 - H * 0.42 * rise)
        nib = np.array([(tip[0], tip[1]), (tip[0] - 70, tip[1] + 200), (tip[0] - 60, tip[1] + 520), (tip[0] + 60, tip[1] + 520), (tip[0] + 70, tip[1] + 200)], np.int32)
        m = np.zeros((H, W), np.float32)
        cv2.fillPoly(m, [nib], 1.0, cv2.LINE_AA)
        cv2.line(m, (int(tip[0]), int(tip[1] + 20)), (int(tip[0]), int(tip[1] + 180)), 0.0, 4, cv2.LINE_AA)
        cv2.circle(m, (int(tip[0]), int(tip[1] + 190)), 12, 0.0, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 0.8)
        rgb = RR.crop_metal(m, bevel=6, strength=1.5, lo=INDIGO * 0.3, mid=SILVER * 0.7, hi=MOON, spec=30)
        img[:] = img * (1 - m[..., None]) + rgb * m[..., None]
        words, times = word_times(ln)
        k = words.index("calm")
        line = " ".join(words[:k])
        a1 = sm((ts - times[0]) / 0.15)
        glow_text(img, tw(line, "notob", 64), MOON, W / 2, H * 0.16, a1, 0.2)
        a2 = sm((ts - times[k]) / 0.1)
        glow_text(img, tw("calm", "anton", 210), ROSE, W / 2, H * 0.27, a2, 0.6)
    elif sc == "v_busy":
        m = np.zeros((H, W), np.float32)
        hx, hy = W / 2, H * 0.70
        pts = np.array([(hx - 210, hy + 150), (hx - 210, hy - 40), (hx, hy - 190), (hx + 210, hy - 40), (hx + 210, hy + 150)], np.int32)
        cv2.polylines(m, [pts], True, 1.0, 6, cv2.LINE_AA)
        img += (blur(m, 0.8) + blur(m, 12) * 0.4)[..., None] * SILVER
        lit = 0.6 + 0.4 * math.sin(ts * 9)
        win = np.zeros((H, W), np.float32)
        cv2.rectangle(win, (int(hx - 60), int(hy - 20)), (int(hx + 60), int(hy + 80)), 1.0, -1)
        img += (win * lit * 0.8 + blur(win, 30) * lit * 0.8)[..., None] * AMBER
        words_line(img, ln, ts, "notob", 76, H * 0.30, key="busy", keycol=ROSE, keysize=150, maxw=960, rise=0, g=0.25)
    elif sc == "v_phone":
        m = np.zeros((H, W), np.float32)
        px, py, pw, ph = W / 2, H * 0.66, 300, 560
        cv2.rectangle(m, (int(px - pw / 2), int(py - ph / 2)), (int(px + pw / 2), int(py + ph / 2)), 1.0, 6, cv2.LINE_AA)
        cv2.rectangle(m, (int(px - pw / 2 + 30), int(py - ph / 2 + 70)), (int(px + pw / 2 - 30), int(py + 60)), 1.0, 3, cv2.LINE_AA)
        img += (blur(m, 0.8) + blur(m, 14) * 0.4)[..., None] * SILVER
        pop = sm((u - 0.5) / 0.15)
        hrt = np.zeros((H, W), np.float32)
        hs = 34 * (1 + 0.25 * math.exp(-max(0, u - 0.5) * 12))
        hc = (int(px - pw / 2 + 60), int(py + 130))
        cv2.circle(hrt, (hc[0] - int(hs * 0.5), hc[1]), int(hs * 0.55), 1.0, -1, cv2.LINE_AA)
        cv2.circle(hrt, (hc[0] + int(hs * 0.5), hc[1]), int(hs * 0.55), 1.0, -1, cv2.LINE_AA)
        cv2.fillConvexPoly(hrt, np.array([(hc[0] - int(hs * 1.02), hc[1] + 6), (hc[0] + int(hs * 1.02), hc[1] + 6), (hc[0], hc[1] + int(hs * 1.3))], np.int32), 1.0, cv2.LINE_AA)
        img += (hrt * pop + blur(hrt, 14) * pop * 0.6)[..., None] * ROSE
        words_line(img, ln, ts, "notob", 76, H * 0.26, maxw=960, rise=0, g=0.25)
    elif sc == "v_wound":
        xs = np.linspace(W * 0.12, W * 0.88, 50)
        pts = [(x, H * 0.66 + 26 * math.sin(x / 23) + 14 * math.sin(x / 7)) for x in xs]
        stroke(img, pts, ROSE, 7, 1.0, 0.7)
        heal = sm((u - 0.35) / 0.5)
        n = int(heal * 9)
        for k in range(n):
            x, y = pts[5 + k * 5]
            stroke(img, [(x - 22, y - 26), (x + 22, y + 26)], SILVER, 4, 0.9, 0.3)
        words_line(img, ln, ts, "notob", 76, H * 0.30, key="heal", keycol=ROSE, keysize=150, maxw=960, rise=0, g=0.25)
    elif sc == "v_snake":
        ph_ = ht(ts) * 3
        xs = np.linspace(-W * 0.1 + (t * 260) % (W * 0.3), W * 1.1, 70)
        pts = [(x, H * 0.66 + 70 * math.sin(x / 70 - ph_)) for x in xs]
        stroke(img, pts, SILVER, 12, 0.9, 0.4)
        hx, hy = pts[-1]
        g = np.zeros((H, W), np.float32)
        cv2.circle(g, (int(min(W - 30, hx)), int(hy)), 16, 1.0, -1, cv2.LINE_AA)
        img += blur(g, 1.5)[..., None] * ROSE
        words_line(img, ln, ts, "notob", 76, H * 0.30, key="ध्यान", keycol=ROSE, keysize=130, maxw=960, rise=0, g=0.25)
    elif sc == "v_silent":
        a = sm(t / 0.08)
        words_line(img, ln, ts, "notob", 64, H * 0.30, maxw=960, rise=0, g=0.2)
        letters = ["C", "A", "L", "M"]
        sps = [tw(c, "anton", 330) for c in letters]
        ws = [s.shape[1] - 2 * 66 for s in sps]
        x = W / 2 - (sum(ws) + 20 * 3) / 2
        fade = sm((ts - 106.0) / 0.4)
        for c, sp, w in zip(letters, sps, ws):
            al = a
            col = MOON
            if c == "L":
                al = a * (1 - fade)
                e, g = rim(sp, ROSE, 5)
                glow_text(img, np.clip(g, 0, 1), ROSE, x + w / 2, H * 0.47, a * fade * (1 - sm((ts - ln["t0"] - 0.9) / 0.4)), 0)
                col = ROSE
            glow_text(img, sp, col, x + w / 2, H * 0.47, al, 0.3)
            x += w + 20
        if False:
            f = 0.0
            img *= 1 - 0.75 * f
            sp = tw("मदिरा", "rozha", 150)
            rgb = metal(sp, bevel=4, strength=1.2, lo=WINE * 0.3, mid=ROSE * 0.95, hi=MOON, spec=18)
            over_c(img, rgb, sp * f, W / 2, H * 0.62)
            cr = tag_sprite("SEEDHE MAUT · PROD. HURRICANE · DL91 FM", 20)
            over_c(img, MOON, cr * 0.6 * f, W / 2, H * 0.62 + 130)


SCENES = {
    "road": sc_road, "drops": sc_drops, "hand": sc_hand, "rise": sc_rise, "backs": sc_backs, "strike": sc_strike,
    "echo": sc_echo, "wander": sc_wander, "lantern": sc_lantern, "window": sc_window, "under": sc_under,
    "bloodmoon": sc_bloodmoon, "sky": sc_sky, "pool": sc_pool,
}
INKMUL = {"road": 0.7, "drops": 0.6, "hand": 0.5, "rise": 0.6, "backs": 0.8, "strike": 0.9, "echo": 0.9, "wander": 0.6,
          "lantern": 0.5, "yuhi": 0.5, "window": 0.0, "ripples": 0.5, "envelope": 0.7, "under": 0.0, "rosepath": 0.6,
          "bloodmoon": 0.9, "sky": 1.4, "pool": 0.5}


def render_frame(fi, store):
    ts = T0 + fi / FPS
    ln = line_at(ts)
    sc = ln["scene"]
    img = ASSET["sky"].copy()
    img = ink(img, store, fi, INKMUL.get(sc, 1.0))
    if sc == "yuhi":
        sc_yuhi(img, ts, fi, ln, store)
    elif sc == "ripples":
        sc_drops(img, ts, fi, ln, ripples=True)
    elif sc == "envelope":
        sc_hand(img, ts, fi, ln, envelope=True)
    elif sc == "rosepath":
        sc_road(img, ts, fi, ln, rose=True)
    elif sc.startswith("v_"):
        sc_verse(img, ts, fi, ln)
    else:
        SCENES[sc](img, ts, fi, ln)
    if sc == "under":
        pass
    if ts > T0 + DUR - 0.8:
        f = sm((ts - (T0 + DUR - 0.8)) / 0.4)
        img *= 1 - 0.8 * f
        sp = tw("मदिरा", "rozha", 150)
        rgb = metal(sp, bevel=4, strength=1.2, lo=WINE * 0.3, mid=ROSE * 0.95, hi=MOON, spec=18)
        over_c(img, rgb, sp * f, W / 2, H * 0.47)
        cr = tag_sprite("SEEDHE MAUT · PROD. HURRICANE · DL91 FM", 20)
        over_c(img, MOON, cr * 0.6 * f, W / 2, H * 0.47 + 130)
    # cut accent: a soft wine flash on each new line
    k = ts - ln["t0"]
    if k < 0.10 and ln is not LINES[0] and sc not in ("yuhi", "v_silent"):
        img += (1 - k / 0.10) * (0.10 if not sc.startswith("v_") else 0.18) * ROSE
    # post
    lum = img.max(-1)
    br = np.clip(lum - 0.55, 0, None)
    img = img + (blur(br, 16) * 0.5 + blur(br, 50) * 0.4)[..., None] * np.array([0.95, 0.85, 1.0], np.float32)
    img = img * POST["vig"][..., None]
    img = np.where(img > 0.85, 0.85 + 0.15 * np.tanh((img - 0.85) / 0.15), img)
    img = img + POST["grain"][fi % 6][..., None] * 0.016
    if fi < 3:
        img = img * (fi / 3)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


POST = {}
CFG = {}


def load_timing(path):
    """measured line/word times (from the isolated vocal track): list of {idx, t0, t1, words}"""
    import json
    global FREEZES, SLOW
    tm = {d["idx"]: d for d in json.load(open(path))}
    LINES[:] = [ln for ln in LINES if ln["idx"] in tm]
    for ln in LINES:
        d = tm[ln["idx"]]
        ln["t0"], ln["te"], ln["wt"] = d["t0"], d["t1"], d["words"]
    LINES[0]["t0"] = T0
    for a, b in zip(LINES, LINES[1:]):
        a["t1"] = b["t0"]
    LINES[-1]["t1"] = T0 + DUR + 1
    FREEZES = [(LI(9)["t0"] + 0.05, LI(9)["t1"])] if LI(9) else []
    FREEZES.append((106.12, 107.45))
    SLOW = [(LI(13)["t0"] + 0.4, LI(13)["t1"])] if LI(13) else []


def load_lyrics(path):
    with open(path, encoding="utf-8") as fh:
        LYRICS[:] = [x.strip() for x in fh if x.strip()]
    for ln in LINES:
        ln["text"] = LYRICS[ln["idx"]]


def setup(audio, fontdir):
    j = lambda n: os.path.join(fontdir, n)
    F.update(tiro=j("TiroDevanagariHindi-Regular.ttf"), kalam=j("Kalam-Regular.ttf"), rozha=j("RozhaOne-Regular.ttf"),
             anton=j("Anton-Regular.ttf"), notob="/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
             inter="/usr/share/fonts/opentype/inter/Inter-Bold.otf", corm=j("CormorantGaramond[wght].ttf"))
    RR.MONO, RR.TIRO = j("IBMPlexMono-Regular.ttf"), F["tiro"]
    RI.analyse(audio)
    build_assets()
    yy, xx = ASSET["yy"], ASSET["xx"]
    POST["vig"] = np.clip(1 - 0.55 * (((xx - W / 2) / (W * 0.65)) ** 2 + ((yy - H * 0.45) / (H * 0.62)) ** 2) ** 1.3, 0.25, 1).astype(np.float32)
    POST["grain"] = [cv2.resize(np.random.default_rng(k).normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H)) for k in range(6)]


def worker(job):
    k, f0, f1, out = job
    store = np.load(CFG["sim"], mmap_mode="r")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        enc.stdin.write(render_frame(fi, store).tobytes())
        if (fi - f0) % 72 == 0:
            print(f"[w{k}] {fi}/{f1}", flush=True)
    enc.stdin.close()
    enc.wait()
    return out


def main():
    audio, out, work, fontdir = sys.argv[1:5]
    os.makedirs(work, exist_ok=True)
    load_lyrics(sys.argv[sys.argv.index("--lyrics") + 1])
    if "--timing" in sys.argv:
        load_timing(sys.argv[sys.argv.index("--timing") + 1])
    setup(audio, fontdir)
    CFG["sim"] = os.path.join(work, "lyric_sim.npy")
    if "--reuse-sim" not in sys.argv or not os.path.exists(CFG["sim"]):
        simulate(CFG["sim"])
    print("simulated", flush=True)
    if "--preview" in sys.argv:
        os.makedirs(out, exist_ok=True)
        store = np.load(CFG["sim"], mmap_mode="r")
        for s in sys.argv[sys.argv.index("--preview") + 1].split(","):
            ts = float(s)
            fi = min(NF - 1, int(round((ts - T0) * FPS)))
            Image.fromarray(render_frame(fi, store)).save(os.path.join(out, f"l_{ts:06.2f}.png"))
        return
    nw = int(os.environ.get("WORKERS", "4"))
    b = np.linspace(0, NF, nw + 1).astype(int)
    jobs = [(k, int(b[k]), int(b[k + 1]), os.path.join(work, f"lyr_chunk{k}.mp4")) for k in range(nw)]
    with mp.get_context("fork").Pool(nw) as pool:
        parts = pool.map(worker, jobs)
    lst = os.path.join(work, "lyr_chunks.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", str(T0), "-t", str(DUR), "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", f"afade=t=in:st=0:d=0.03,afade=t=out:st={DUR - 0.6}:d=0.6", "-c:a", "aac", "-b:a", "256k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
