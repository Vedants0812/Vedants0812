#!/usr/bin/env python3
"""
MADIRA (Seedhe Maut) - 16:9 kinetic lyric video, 1920x1080 @ 24 fps.
Song 0:59.3 -> 1:48 (the hook with echoes, then Calm's verse).

Moon, wine ink and roses with the energy of a kinetic-type video: every lyric
line is a word stack whose words slam in on their syllables; every word fires a
burst of ink from where it lands (a small fluid simulation); each line gets its
own world behind it; CALM loses its L on "silent". Lyric text and timing are read
from local files at run time.

usage:
  python3 render_madira16.py SONG OUT.mp4 WORKDIR FONTDIR --lyrics L.txt --timing T.json
  ... --preview 60,75,82
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_reel as RR  # noqa: E402
import render_ink as RI  # noqa: E402
from render_reel import over, over_c, affine, draw_sprite, blur, metal, sm, eio, tsprite  # noqa: E402

W, H, FPS = 1920, 1080, 24
T0, T1 = 59.3, 108.1
NF = int(round((T1 - T0) * FPS))
GW, GH = 480, 270
CELL = W / GW

hexc = RI.hexc
BG, INDIGO, VIOLET = RI.BG, RI.INDIGO, RI.VIOLET
SILVER, WINE, ROSE, MOON, BLOOD = RI.SILVER, RI.WINE, RI.ROSE, RI.MOON, RI.BLOOD
INKS = RI.INKS
F = {}
AU = {}
LINES = []
A = {}

BGPLAN = ["road", "drops", "rain", "wine",
          "moon", "faces", "rain", "blood", "drops", "moon", "blood", "silent", "rain", "faces", "rose", "drops",
          "moon", "rain", "rose", "faces", "blood", "drops", "blood", "moon", "road", "wine"]
KEYS = {0: "राह", 1: "थम", 2: "नाम", 3: "मदिरा", 4: "ठिकाना", 5: "सहारा", 6: "छाता", 7: "जवाला", 8: "गम",
        9: "ज़ुखाम", 10: "calm", 11: "silent", 12: "busy", 13: "'gram", 14: "heal", 15: "ध्यान", 16: "ख्वाब",
        17: "जवाब", 18: "प्यार", 19: "प्यार", 20: "दिवार", 21: "नाल", 22: "गोली", 23: "पत्थर", 24: "गले", 25: "गुस्से"}
PLACE = ["c", "l", "r", "c"] + ["l", "r", "c", "l", "r", "c", "l", "c", "r", "l", "c", "r", "l", "c", "r", "l", "c", "r", "l", "c", "r", "c"]
SHAKE = {9, 20, 21, 22, 25}


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
    pk = librosa.util.peak_pick(lo, pre_max=6, post_max=6, pre_avg=20, post_avg=20, delta=lo.std() * 1.2, wait=8)
    b0 = int(np.argmin(np.abs(bt - 23.64)))
    AU.update(beats=bt, bass=lt[pk], downs=np.array([b for i, b in enumerate(bt) if (i - b0) % 4 == 0]))


def load(lyrics, timing):
    lyr = [l.strip() for l in open(lyrics, encoding="utf-8") if l.strip()]
    for d in json.load(open(timing)):
        raw = lyr[d["idx"]]
        m = re.match(r"^(.*?)\s*(\(.*\))\s*$", raw)
        txt, echo = (m.group(1), m.group(2)) if m else (raw, None)
        i = d["idx"]
        LINES.append(dict(idx=i, t0=d["t0"], te=d["t1"], wt=d["words"], words=txt.split(), echo=echo,
                          bg=BGPLAN[i], key=KEYS.get(i), place=PLACE[i]))
    LINES.sort(key=lambda x: x["t0"])
    LINES[0]["t0"] = T0
    for a, b in zip(LINES, LINES[1:]):
        a["t1"] = b["t0"]
    LINES[-1]["t1"] = T1 + 1


def line_at(t):
    for ln in LINES:
        if ln["t0"] <= t < ln["t1"]:
            return ln
    return LINES[-1]


def last(arr, t):
    i = int(np.searchsorted(arr, t, "right")) - 1
    return (i, t - arr[i]) if i >= 0 else (-1, 9.0)


def pulse(arr, t, tau):
    i, dt = last(arr, t)
    return math.exp(-dt / tau) if i >= 0 else 0.0


# ------------------------------------------------------------------- type --
def is_deva(s):
    return any("ऀ" <= c <= "ॿ" for c in s)


def wsprite(w, size):
    size = int(max(36, min(300, round(size / 6) * 6)))
    if is_deva(w):
        return tsprite(w, F["deva"], size, pad=int(size * 0.18))
    return tsprite(w.upper(), F["anton"], size, pad=int(size * 0.18))


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
        size = max(50, min(210, base * box_w / max(nat, 1)))
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


def anchor(ln):
    cx = {"c": W / 2, "l": W * 0.33, "r": W * 0.67}[ln["place"]]
    box = 1300 if ln["place"] == "c" else 980
    if ln["bg"] == "silent":
        return W / 2, H * 0.24, 1200
    return cx, H * 0.5, box


def word_pos(ln):
    cx, cy, box = anchor(ln)
    return [(cx + r["x"], cy + r["y"], ln["wt"][min(r["k"], len(ln["wt"]) - 1)]) for r in layout(ln, box)]


def stack(img, ln, t):
    cx, cy, box = anchor(ln)
    for r in layout(ln, box):
        tw = ln["wt"][min(r["k"], len(ln["wt"]) - 1)]
        dt = t - tw
        if dt < -0.02:
            continue
        a = sm((dt + 0.02) / 0.06)
        s = 1 + 0.42 * math.exp(-max(dt, 0) / 0.06)
        sp = wsprite(r["w"], r["size"])
        hot = ln["key"] and ln["key"].lower() in r["w"].lower()
        M = affine(cx + r["x"], cy + r["y"] - 16 * (1 - a), s, s, 0, sp.shape[1], sp.shape[0])
        sh = affine(cx + r["x"] + 6, cy + r["y"] + 8, s, s, 0, sp.shape[1], sp.shape[0])
        draw_sprite(img, BG * 0.3, blur(sp, 5), sh, mul=0.9 * a)
        if hot:
            draw_sprite(img, ROSE, blur(sp, 18), M, add=True, mul=0.7 * a)
        draw_sprite(img, ROSE if hot else MOON, sp, M, mul=a)
    if ln["echo"] and t >= ln["te"] - 0.15:
        a = sm((t - ln["te"] + 0.15) / 0.2)
        sp = tsprite(ln["echo"], F["deva"], 70, pad=14)
        for k, (dx, al) in enumerate(((0, 0.8), (16, 0.35), (32, 0.15))):
            over_c(img, SILVER, sp * a * al, cx + dx, cy + 300 + dx * 0.5)


# ------------------------------------------------------------- ink sim -----
GX, GY = np.meshgrid(np.arange(GW, dtype=np.float32), np.arange(GH, dtype=np.float32))


def gauss(cx, cy, r):
    return np.exp(-((GX - cx) ** 2 + (GY - cy) ** 2) / (2 * r * r)).astype(np.float32)


def radial(vx, vy, cx, cy, s, rad):
    dx, dy = GX - cx, GY - cy
    d = np.sqrt(dx * dx + dy * dy) + 1e-3
    g = np.exp(-(d / rad) ** 2) * s
    vx += dx / d * g
    vy += dy / d * g


def curl_keys(n, seed=7):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        psi = np.zeros((GH, GW), np.float32)
        for (gh, gw), wt in (((4, 6), 0.8), ((7, 12), 0.7), ((14, 24), 0.5), ((27, 48), 0.32)):
            psi += cv2.resize(r.standard_normal((gh, gw)).astype(np.float32), (GW, GH), interpolation=cv2.INTER_CUBIC) * wt
        gy, gx = np.gradient(psi)
        out.append((gy, -gx))
    mag = np.sqrt(out[0][0] ** 2 + out[0][1] ** 2).mean()
    return [(a / mag, b / mag) for a, b in out]


def simulate(path):
    store = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(NF, GH, GW, 3))
    dye = np.zeros((GH, GW, 3), np.float32)
    vx = np.zeros((GH, GW), np.float32)
    vy = np.zeros((GH, GW), np.float32)
    keys = curl_keys(40)
    dt = 1 / FPS
    events = []
    for ln in LINES:
        for k, (x, y, tw) in enumerate(word_pos(ln)):
            events.append((tw, x / CELL, y / CELL, (ln["idx"] + k) % 3))
    events.sort()
    ev = np.array([e[0] for e in events])
    for fi in range(NF):
        t = T0 + fi / FPS
        silent = 106.12 <= t < 107.5
        if not silent:
            kf = (t - T0) / 1.3
            k0 = int(kf) % (len(keys) - 1)
            f = sm(kf - int(kf))
            cx_ = keys[k0][0] * (1 - f) + keys[k0 + 1][0] * f
            cy_ = keys[k0][1] * (1 - f) + keys[k0 + 1][1] * f
            amp = 14 if t < 71.4 else 26
            vx = vx * 0.9 + cx_ * amp * 0.3
            vy = vy * 0.9 + cy_ * amp * 0.3
            for j in np.nonzero((ev > t - dt) & (ev <= t))[0]:
                _, x, y, ch = events[j]
                rr = np.random.default_rng(int(ev[j] * 997))
                ox, oy = rr.uniform(-60, 60), rr.uniform(-30, 30)
                dye[..., ch] += gauss(x + ox, y + oy, 5) * 0.9
                radial(vx, vy, x + ox, y + oy, 130, 16)
            for b in AU["bass"][(AU["bass"] > t - dt) & (AU["bass"] <= t)]:
                radial(vx, vy, GW / 2, GH / 2, 70, 140)
            vx, vy = adv(vx, vx, vy, dt), adv(vy, vx, vy, dt)
            dye = adv(dye, vx, vy, dt)
            dye = cv2.GaussianBlur(dye, (0, 0), 0.32) * 0.982
            np.clip(dye, 0, 4, out=dye)
        store[fi] = (np.sqrt(np.clip(dye / 4.0, 0, 1)) * 255 + 0.5).astype(np.uint8)
    store.flush()


def adv(f, vx, vy, dt):
    return cv2.remap(f, GX - vx * dt, GY - vy * dt, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def ink(img, store, fi, mul=1.0):
    dye = (store[fi].astype(np.float32) / 255) ** 2 * 4.0 * mul
    dens = np.clip(cv2.resize(dye, (W, H), interpolation=cv2.INTER_CUBIC), 0, None)
    tot = dens.sum(-1)
    colr = (dens @ INKS) / (tot[..., None] + 1e-4)
    em = (1 - np.exp(-tot * 0.7))[..., None] * colr
    return img * np.exp(-tot * 0.5)[..., None] + em * 1.05


# ----------------------------------------------------------- backgrounds ---
def build_assets():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    A["xx"], A["yy"] = xx, yy
    sky = np.zeros((H, W, 3), np.float32) + BG
    sky += np.exp(-((yy - H * 0.25) / (H * 0.6)) ** 2)[..., None] * INDIGO * 0.9
    sky += np.exp(-((xx - W / 2) ** 2 + (yy - H * 1.1) ** 2) / (2 * (W * 0.5) ** 2))[..., None] * VIOLET * 0.25
    A["sky"] = sky
    S = 600
    my, mx = np.mgrid[0:S, 0:S].astype(np.float32)
    d = np.hypot(mx - S / 2, my - S / 2)
    r = np.random.default_rng(2)
    tex = 0.86 + 0.10 * cv2.resize(r.random((12, 12)).astype(np.float32), (S, S), interpolation=cv2.INTER_CUBIC)
    A["moon"] = (np.clip(220 - d, 0, 1.5) / 1.5 * tex * np.clip(1 - (d / 220) ** 2, 0, 1) ** 0.25).astype(np.float32)
    A["halo"] = (np.exp(-np.clip(d - 220, 0, None) / 120) * np.clip(1 - d / 300, 0, 1) ** 2).astype(np.float32)
    A["rose"] = RI.rose_sprite()
    A["petals"] = [RI.petal_sprite(k) for k in range(6)]
    dm = np.zeros((40, 26), np.float32)
    cv2.ellipse(dm, (13, 22), (10, 15), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    dm = cv2.GaussianBlur(dm, (0, 0), 0.8)
    A["drop"] = (metal(dm, bevel=4, strength=2.0, lo=INDIGO * 1.2, mid=SILVER * 0.85, hi=MOON * 1.2, spec=20), dm)
    rr = np.random.default_rng(51)
    A["drops"] = np.stack([rr.uniform(-0.2, 1.4, 260), rr.uniform(0, 1, 260), rr.uniform(0.4, 2.2, 260)], 1)
    bk = np.zeros((H, W, 3), np.float32) + BG * 1.4
    for k in range(50):
        c = [WINE * 1.3, ROSE, np.array([0.9, 0.6, 0.35]), SILVER * 0.8, MOON * 0.7][k % 5]
        m = np.zeros((H, W), np.float32)
        cv2.circle(m, (int(rr.uniform(0, W)), int(rr.uniform(0, H))), int(rr.uniform(35, 110)), 1.0, -1, cv2.LINE_AA)
        bk += blur(m, 10)[..., None] * c * rr.uniform(0.15, 0.4)
    A["bokeh"] = blur(bk, 18)
    A["streaks"] = np.stack([rr.uniform(0, W, 40), rr.uniform(0, 1, 40), rr.uniform(160, 420, 40)], 1)
    for k, (cx, f) in enumerate(((W * 0.30, 1), (W * 0.70, -1))):
        pts = [(-0.45, -0.55), (-0.1, -0.78), (0.25, -0.64), (0.33, -0.38), (0.32, -0.30), (0.47, -0.12), (0.33, -0.06),
               (0.36, 0.0), (0.30, 0.04), (0.35, 0.09), (0.30, 0.22), (0.12, 0.31), (0.10, 0.46), (0.22, 0.76), (0.40, 1.05),
               (0.42, 1.6), (-0.65, 1.6), (-0.60, 0.80), (-0.36, 0.42), (-0.50, 0.05)]
        poly = np.array([(cx + f * x * 300, H * 0.48 + y * 300) for x, y in pts], np.int32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [poly], 255, cv2.LINE_AA)
        A[f"face{k}"] = m.astype(np.float32) / 255


def moon(img, cx, cy, r, blood=0.0, halo=0.5):
    col = MOON * (1 - blood) + BLOOD * blood
    hc = SILVER * 0.55 * (1 - blood) + BLOOD * 0.8 * blood
    s = r / 220
    draw_sprite(img, col * 0.85, A["moon"], affine(cx, cy, s, s, 0, 600, 600))
    draw_sprite(img, hc, A["halo"], affine(cx, cy, s * 2.2, s * 2.2, 0, 600, 600), add=True, mul=halo)


def background(img, ln, t, fi):
    kind = ln["bg"]
    u = (t - ln["t0"]) / max(0.5, ln["t1"] - ln["t0"])
    xx, yy = A["xx"], A["yy"]
    if kind == "road":
        hz = H * 0.56
        moon(img, W / 2, hz - 150, 120)
        d = yy - hz
        below = (d > 0).astype(np.float32)
        halfw = 8 + np.clip(d, 0, None) * 1.6
        m = np.clip((halfw - np.abs(xx - W / 2)) / (10 + np.clip(d, 0, None) * 0.5), 0, 1) * below
        k = 600.0 / (np.clip(d, 0, None) + 20)
        sh = 0.5 + 0.5 * np.sin(k * 6 - t * 4 + np.sin(xx * 0.03) * 1.5)
        img += (m * (0.45 + 0.55 * sh ** 3) * 0.8)[..., None] * SILVER
    elif kind == "moon":
        moon(img, W * (0.78 if ln["place"] == "l" else 0.22), H * 0.30, 170)
    elif kind == "blood":
        moon(img, W / 2, H * 0.45, 330, blood=0.5 + 0.4 * u, halo=0.8)
    elif kind == "drops":
        moon(img, W * 0.5, H * 0.28, 130)
        rgb, m = A["drop"]
        for x, y, z in A["drops"]:
            px, py = (x - (t - T0) * 0.01 * z) * W, y * H
            if -40 < px < W + 40:
                s = 0.5 + 0.8 / z
                draw_sprite(img, rgb, m, affine(px, py, s, s * 1.4, 0, 26, 40), mul=min(1.0, 1.3 / z))
    elif kind == "rain":
        img[:] = A["bokeh"]
        lay = np.zeros((H, W), np.float32)
        for x, y0, sp in A["streaks"]:
            y = (y0 * H + t * sp) % (H + 200) - 100
            cv2.line(lay, (int(x), int(y - 140)), (int(x + 3), int(y)), 0.6, 2, cv2.LINE_AA)
        img += blur(lay, 1)[..., None] * MOON * 0.35
        img *= 0.75
    elif kind == "wine":
        moon(img, W / 2, H * 0.25, 150)
        lv = int(H - H * 0.85 * eio(u * 1.2))
        if lv < H:
            reg = img[lv:]
            hh = reg.shape[0]
            ys = np.arange(hh, dtype=np.float32)[:, None]
            reg[:] = reg * np.array([0.6, 0.25, 0.3], np.float32) + WINE * (0.3 + 0.4 * np.clip(ys / 600, 0, 1))[..., None]
            wav = (lv + 6 * np.sin(np.arange(W) / 40 + t * 4)).astype(np.int32)
            men = np.zeros((H, W), np.float32)
            cv2.polylines(men, [np.stack([np.arange(W), wav], 1).astype(np.int32)], False, 1.0, 3, cv2.LINE_AA)
            img += (men + blur(men, 12) * 0.6)[..., None] * ROSE
    elif kind == "faces":
        for k in range(2):
            m = A[f"face{k}"]
            col = SILVER if k == 0 else ROSE
            img[:] = img * (1 - m[..., None] * 0.85) + BG * 0.4 * m[..., None]
            e = np.clip(m - cv2.erode(m, np.ones((5, 5), np.uint8)), 0, 1)
            img += (e + blur(e, 12) * 0.6)[..., None] * col * 0.8
        moon(img, W / 2, H * 0.16, 90)
    elif kind == "rose":
        rgb, al = A["rose"]
        cx = W * (0.80 if ln["place"] == "l" else 0.20)
        draw_sprite(img, rgb, al, affine(cx, H * 0.58, 1.2, 1.2, -6 + 3 * math.sin(t), al.shape[1], al.shape[0]))
        for k in range(8):
            p = A["petals"][k % 6]
            a = ((t - T0) * 0.35 + k * 0.13) % 1
            draw_sprite(img, p[0], p[1], affine(cx + 300 * math.sin(k * 2 + t * 0.7) * a, H * 0.25 + a * H * 0.8,
                                                0.6, 0.6 * (0.3 + abs(math.sin(t * 2 + k))), t * 80 + k * 40, 90, 120),
                        mul=sm(a * 5) * (1 - sm((a - 0.85) * 6)))


def silent_scene(img, ln, t):
    letters = ["C", "A", "L", "M"]
    sps = [tsprite(c, F["anton"], 420, pad=60) for c in letters]
    ws = [s.shape[1] - 120 for s in sps]
    x = W / 2 - (sum(ws) + 30 * 3) / 2
    tw = ln["wt"][3] if len(ln["wt"]) > 3 else ln["t0"] + 0.6
    fade = sm((t - tw) / 0.3)
    for c, sp, w in zip(letters, sps, ws):
        al = 0.6 * (1 - fade) if c == "L" else 0.6
        over_c(img, ROSE if c == "L" else MOON, sp * al, x + w / 2, H * 0.66)
        if c == "L" and fade > 0:
            e = np.clip(sp - cv2.erode(sp, np.ones((5, 5), np.uint8)), 0, 1)
            over_c(img, ROSE, e * 0.7 * (1 - sm((t - tw - 0.8) / 0.4)), x + w / 2, H * 0.66)
        x += w + 30


# ------------------------------------------------------------------ frame --
VIG = None
STORE = {}


def render_frame(fi):
    global VIG
    t = T0 + fi / FPS
    ln = line_at(t)
    img = A["sky"].copy()
    if ln["bg"] == "silent":
        silent_scene(img, ln, t)
    else:
        background(img, ln, t, fi)
    img = ink(img, STORE["s"], fi, 0.0 if ln["bg"] == "rain" else 1.0)
    stack(img, ln, t)
    if t - ln["t0"] < 2 / FPS and ln is not LINES[0]:
        img = img * 0.5 + WINE * 0.6
    if t > T1 - 0.8:
        f = sm((t - (T1 - 0.8)) / 0.4)
        img = img * (1 - 0.985 * f)
        sp = tsprite("मदिरा", F["rozha"], 220, pad=40)
        rgb = metal(sp, bevel=5, strength=1.2, lo=WINE * 0.3, mid=ROSE * 0.95, hi=MOON, spec=18)
        over_c(img, rgb, sp * f, W / 2, H * 0.46)
        cr = RR.tag_sprite("SEEDHE MAUT · PROD. HURRICANE · DL91 FM", 24)
        over_c(img, MOON, cr * 0.6 * f, W / 2, H * 0.46 + 170)
    z = 1 + 0.03 * pulse(AU["beats"], t, 0.08) + 0.03 * pulse(AU["downs"], t, 0.12)
    sh = pulse(AU["bass"], t, 0.07) * 9
    rot = 0.0
    if ln["idx"] in SHAKE:
        sh += 7
        rot = 1.4 * math.sin(fi * 1.7)
    if 106.12 <= t < 107.5:
        z, sh, rot = 1.0, 0.0, 0.0
    if z > 1.002 or sh > 0.5 or rot:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, z)
        M[0, 2] += sh * math.sin(fi * 2.6)
        M[1, 2] += sh * math.cos(fi * 1.9)
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    if VIG is None:
        yy, xx = A["yy"], A["xx"]
        VIG = np.clip(1 - 0.45 * (((xx - W / 2) / (W * 0.7)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2) ** 1.4, 0.3, 1)
    lum = img.max(-1)
    br = np.clip(lum - 0.6, 0, None)
    img = img + (blur(br, 16) * 0.45 + blur(br, 50) * 0.35)[..., None] * np.array([0.95, 0.8, 0.95], np.float32)
    g = np.random.default_rng(fi % 8).normal(0, 0.018, (H // 2, W // 2)).astype(np.float32)
    img = img * VIG[..., None] + cv2.resize(g, (W, H))[..., None]
    if fi < 3:
        img = img * (fi / 3)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------- main --
def setup(audio, fontdir, argv):
    j = lambda n: os.path.join(fontdir, n)
    F.update(anton=j("Anton-Regular.ttf"), rozha=j("RozhaOne-Regular.ttf"),
             deva="/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf")
    RR.MONO, RR.TIRO = j("IBMPlexMono-Regular.ttf"), j("TiroDevanagariHindi-Regular.ttf")
    analyse(audio)
    load(argv[argv.index("--lyrics") + 1], argv[argv.index("--timing") + 1])
    build_assets()


def worker(job):
    k, f0, f1, out = job
    STORE["s"] = np.load(STORE["path"], mmap_mode="r")
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
    STORE["path"] = os.path.join(work, "madira16_sim.npy")
    simulate(STORE["path"])
    print("simulated", flush=True)
    if "--preview" in sys.argv:
        os.makedirs(out, exist_ok=True)
        STORE["s"] = np.load(STORE["path"], mmap_mode="r")
        for s in sys.argv[sys.argv.index("--preview") + 1].split(","):
            t = float(s)
            fi = min(NF - 1, int(round((t - T0) * FPS)))
            Image.fromarray(render_frame(fi)).save(os.path.join(out, f"m_{t:06.2f}.png"))
        return
    nw = int(os.environ.get("WORKERS", "4"))
    b = np.linspace(0, NF, nw + 1).astype(int)
    jobs = [(k, int(b[k]), int(b[k + 1]), os.path.join(work, f"m16_chunk{k}.mp4")) for k in range(nw)]
    with mp.get_context("fork").Pool(nw) as pool:
        parts = pool.map(worker, jobs)
    lst = os.path.join(work, "m16_chunks.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", str(T0), "-t", str(T1 - T0), "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", f"afade=t=in:st=0:d=0.03,afade=t=out:st={T1 - T0 - 0.6}:d=0.6", "-c:a", "aac", "-b:a", "256k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
