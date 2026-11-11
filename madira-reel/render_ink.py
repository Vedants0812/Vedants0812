#!/usr/bin/env python3
"""
MADIRA (Seedhe Maut, prod. Hurricane) - 60 s reel, theme "INK OF THE NIGHT".
Audio: song 0:48.1 (vocals in) -> 1:48.1 (the hand-off silence).

Midnight indigo, wine, crimson rose and moon ivory. Everything is ink smoke
lit by a moon, driven by a small 2D fluid simulation and keyed to the song:
  0:48.1  vocals in: a wine-ink drop blooms, the title forms out of ink
  0:58.5  bass enters: two ink silhouettes (moon-silver, wine) reach across and mix
  1:10.5  pause: everything holds (नशा)
  1:11.9  a vortex builds ...
  1:15.0  ... bursts: Calm's verse, an ink splash on every strong vocal hit,
          shockwaves on bass hits
  1:21.0  3 s break: total freeze (ठहरो)
  1:24.0  a crimson rose sheds a petal per downbeat; each petal bleeds into ink
          while the moon turns blood-red
  1:46.1  hand-off silence: freeze, end mark

usage:
  python3 render_ink.py SONG_AUDIO OUT.mp4 WORKDIR FONTDIR
  python3 render_ink.py SONG_AUDIO OUTDIR  WORKDIR FONTDIR --preview 49,60,77
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
import render_reel as RR  # noqa: E402  (text + compositing helpers)
from render_reel import (over, over_c, affine, draw_sprite, blur, metal, sm, eio, eout, tsprite,  # noqa: E402
                         tracked, tag_sprite)

W, H, FPS = 1080, 1920, 24
T0, DUR = 48.1, 60.0
NF = int(round(DUR * FPS))
GW, GH = 270, 480          # simulation grid (4 px cells)
CELL = W / GW


def hexc(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)


BG = hexc("#05070E")
INDIGO = hexc("#141B3D")
VIOLET = hexc("#3E3170")
SILVER = hexc("#A9B9DE")
WINE = hexc("#8A1236")
ROSE = hexc("#E04A68")
MOON = hexc("#F0E9DA")
BLOOD = hexc("#C2283F")
INKS = np.stack([SILVER, WINE, ROSE])   # dye channels: 0 silver, 1 wine, 2 rose

FREEZES = [(58.12, 58.54), (70.5, 71.9), (80.99, 84.0), (106.12, 1e9)]
MOON_C = (W / 2, H * 0.29)
MOON_R = 215


def ht(ts):
    return ts - sum(max(0.0, min(ts, b) - a) for a, b in FREEZES if ts > a)


def frozen(ts):
    return any(a <= ts < b for a, b in FREEZES)


# ------------------------------------------------------------------ audio ---
AU = {}


def analyse(audio):
    import librosa
    y, sr = librosa.load(audio, sr=22050)
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    ot = librosa.times_like(oenv, sr=sr)
    _, bf = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
    beats = librosa.frames_to_time(bf, sr=sr)
    on = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, units="time")
    ostr = np.interp(on, ot, oenv)
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    lo = librosa.onset.onset_strength(S=librosa.power_to_db(M[:4]), sr=sr)
    lt = librosa.times_like(lo, sr=sr)
    pk = librosa.util.peak_pick(lo, pre_max=6, post_max=6, pre_avg=20, post_avg=20, delta=lo.std() * 1.2, wait=8)
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rt = librosa.times_like(rms, sr=sr, hop_length=512)
    b0 = int(np.argmin(np.abs(beats - 23.64)))
    AU.update(beats=beats, b0=b0, on=on, ostr=ostr / np.percentile(ostr, 95), bass=lt[pk],
              rms=rms / rms.max(), rt=rt)
    AU["downs"] = np.array([b for i, b in enumerate(beats) if (i - b0) % 4 == 0])


def rms_at(ts):
    return float(np.interp(ts, AU["rt"], AU["rms"]))


def last_beat(ts):
    i = int(np.searchsorted(AU["beats"], ts, "right")) - 1
    return (i, ts - AU["beats"][i]) if i >= 0 else (-1, 9.0)


def down_pulse(ts, tau=0.14):
    d = AU["downs"]
    i = int(np.searchsorted(d, ts, "right")) - 1
    return math.exp(-(ts - d[i]) / tau) if i >= 0 else 0.0


def events(arr, a, b):
    return arr[(arr >= a) & (arr < b)]


# ----------------------------------------------------------------- shapes ---
def profile_mask(cx, cy, s, facing):
    pts = [(-0.45, -0.55), (-0.1, -0.78), (0.25, -0.64), (0.33, -0.38), (0.32, -0.30), (0.47, -0.12), (0.33, -0.06),
           (0.36, 0.0), (0.30, 0.04), (0.35, 0.09), (0.30, 0.22), (0.12, 0.31), (0.10, 0.46), (0.22, 0.76), (0.40, 1.05),
           (0.42, 1.6), (-0.65, 1.6), (-0.60, 0.80), (-0.36, 0.42), (-0.50, 0.05)]
    poly = np.array([(cx + facing * x * s, cy + y * s) for x, y in pts], np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly], 255, cv2.LINE_AA)
    return m.astype(np.float32) / 255


def petal_sprite(seed):
    r = np.random.default_rng(seed)
    m = np.zeros((120, 90), np.float32)
    cv2.ellipse(m, (45, 64), (34, 52), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.circle(m, (45, 8), 16, 0.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 1.0)
    rgb = metal(m, bevel=8, strength=1.4, lo=WINE * 0.25, mid=ROSE * (0.75 + 0.2 * r.random()),
                hi=np.array([0.95, 0.6, 0.65], np.float32), spec=8, rim=0.2)
    return rgb, m


def rose_sprite():
    S = 560
    cx, cy = S / 2, S * 0.40
    r = np.random.default_rng(12)
    rgb = np.zeros((S, S, 3), np.float32)
    al = np.zeros((S, S), np.float32)
    stem = np.zeros((S, S), np.float32)
    cv2.line(stem, (int(cx), int(cy + 80)), (int(cx + 10), S - 6), 1.0, 10, cv2.LINE_AA)
    cv2.ellipse(stem, (int(cx + 52), int(S * 0.70)), (46, 16), -25, 0, 360, 1.0, -1, cv2.LINE_AA)
    stem = cv2.GaussianBlur(stem, (0, 0), 0.8)
    rgb += metal(stem, bevel=3, lo=np.array([0.01, 0.02, 0.02]), mid=np.array([0.05, 0.10, 0.09]), hi=SILVER * 0.25, spec=10) * stem[..., None]
    al = np.maximum(al, stem)
    for k, (rad, n) in enumerate([(150, 6), (118, 5), (88, 5), (60, 4), (34, 3)]):
        lay = np.zeros((S, S), np.float32)
        for i in range(n):
            a = 2 * math.pi * i / n + k * 0.7 + r.uniform(-0.2, 0.2)
            px, py = cx + math.cos(a) * rad * 0.40, cy + math.sin(a) * rad * 0.30
            cv2.ellipse(lay, (int(px), int(py)), (int(rad * 0.62), int(rad * 0.47)), math.degrees(a) * 0.3, 0, 360, 1.0, -1, cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (0, 0), 0.9)
        sh = metal(lay, bevel=10, strength=1.7, lo=WINE * 0.18, mid=WINE * (1.1 + 0.25 * k) + ROSE * 0.1 * k,
                   hi=np.array([0.9, 0.45, 0.52], np.float32), spec=7, rim=0.18)
        edge = np.clip(lay - cv2.erode(lay, np.ones((5, 5), np.uint8)), 0, 1)
        sh = sh * (1 - 0.5 * edge[..., None])
        rgb = rgb * (1 - lay[..., None]) + sh * lay[..., None]
        al = np.maximum(al, lay)
    return rgb, al


# ------------------------------------------------------ analytic movers -----
ROSE_POS = (W / 2, H * 0.67)


def rose_petals():
    """petals that leave the rose: one per downbeat 1:24-1:46, analytic"""
    d = [t for t in AU["downs"] if 84.0 <= t < 106.0]
    extra = [b for b in AU["beats"] if 100.0 <= b < 106.0]
    times = sorted(set([round(t, 3) for t in d + extra]))
    r = np.random.default_rng(84)
    out = []
    for k, td in enumerate(times):
        a = r.uniform(-2.6, -0.5) if k % 2 == 0 else r.uniform(-2.6, -0.5) + math.pi * 0.0
        out.append(dict(td=td, a=a, v=r.uniform(260, 520), spin=r.uniform(-260, 260), seed=k,
                        side=1 if k % 2 else -1))
    return out


def petal_state(p, ts):
    """position (px), rotation, alpha, dissolve 0..1 at song time ts (held time)"""
    age = ht(ts) - ht(p["td"])
    if age < 0:
        return None
    vx = math.cos(p["a"]) * p["v"] * 0.6 * p["side"] * -1
    vy = math.sin(p["a"]) * p["v"]
    x = ROSE_POS[0] + vx * age + 40 * math.sin(age * 2.2 + p["seed"])
    y = ROSE_POS[1] - 30 + vy * age + 160 * age * age
    dis = sm((age - 0.9) / 1.4)
    if dis >= 1:
        return None
    return x, y, p["spin"] * age, 1 - dis, dis


def falling_petals():
    r = np.random.default_rng(58)
    return [dict(t0=58.5 + r.uniform(0, 11), x=r.uniform(W * 0.15, W * 0.85), v=r.uniform(160, 260),
                 sw=r.uniform(40, 110), spin=r.uniform(-140, 140), seed=100 + k) for k in range(14)]


def fall_state(p, ts):
    age = ht(ts) - ht(p["t0"])
    if age < 0 or ts >= 71.9:
        return None
    x = p["x"] + p["sw"] * math.sin(age * 1.5 + p["seed"])
    y = -80 + p["v"] * age
    if y > H + 100:
        return None
    return x, y, p["spin"] * age


# ------------------------------------------------------------- simulation ---
TITLE = {}


def build_static(fontdir):
    RR.ROZHA, RR.TIRO = os.path.join(fontdir, "RozhaOne-Regular.ttf"), os.path.join(fontdir, "TiroDevanagariHindi-Regular.ttf")
    RR.MONO = os.path.join(fontdir, "IBMPlexMono-Regular.ttf")
    RR.CORM = os.path.join(fontdir, "CormorantGaramond[wght].ttf")
    s = tsprite("मदिरा", RR.ROZHA, 300, pad=40)
    while s.shape[1] > 900:
        s = tsprite("मदिरा", RR.ROZHA, int(300 * 880 / s.shape[1]), pad=40)
    m = np.zeros((H, W), np.float32)
    cy = int(H * 0.56)
    m[cy - s.shape[0] // 2:cy - s.shape[0] // 2 + s.shape[0], W // 2 - s.shape[1] // 2:W // 2 - s.shape[1] // 2 + s.shape[1]] = s
    TITLE["full"] = m
    TITLE["grid"] = cv2.resize(m, (GW, GH), interpolation=cv2.INTER_AREA)
    L = profile_mask(W * 0.27, H * 0.50, 360, 1)
    R = profile_mask(W * 0.73, H * 0.50, 360, -1)
    TITLE["prof"] = (L, R)
    for k, mm in enumerate((L, R)):
        e = np.clip(cv2.dilate(mm, np.ones((9, 9), np.uint8)) - cv2.erode(mm, np.ones((9, 9), np.uint8)), 0, 1)
        TITLE[f"pg{k}"] = cv2.resize(e, (GW, GH), interpolation=cv2.INTER_AREA)
        TITLE[f"pf{k}"] = cv2.resize(mm, (GW, GH), interpolation=cv2.INTER_AREA)


def curl_keys(n, seed=5):
    r = np.random.default_rng(seed)
    keys = []
    for k in range(n):
        psi = np.zeros((GH, GW), np.float32)
        for (gh, gw), wt in (((6, 4), 0.8), ((12, 7), 0.7), ((24, 14), 0.5), ((48, 27), 0.32)):
            psi += cv2.resize(r.standard_normal((gh, gw)).astype(np.float32), (GW, GH), interpolation=cv2.INTER_CUBIC) * wt
        gy, gx = np.gradient(psi)
        keys.append((gy, -gx))
    mag = np.sqrt(keys[0][0] ** 2 + keys[0][1] ** 2).mean()
    return [(a / mag, b / mag) for a, b in keys]


GX, GY = np.meshgrid(np.arange(GW, dtype=np.float32), np.arange(GH, dtype=np.float32))


def advect(f, vx, vy, dt):
    return cv2.remap(f, GX - vx * dt, GY - vy * dt, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def gauss(cx, cy, rad):
    return np.exp(-((GX - cx) ** 2 + (GY - cy) ** 2) / (2 * rad * rad)).astype(np.float32)


def radial(vx, vy, cx, cy, s, rad):
    dx, dy = GX - cx, GY - cy
    d = np.sqrt(dx * dx + dy * dy) + 1e-3
    g = np.exp(-(d / rad) ** 2) * s
    vx += dx / d * g
    vy += dy / d * g


def swirl(vx, vy, cx, cy, s, rad):
    dx, dy = GX - cx, GY - cy
    g = np.exp(-(dx * dx + dy * dy) / (rad * rad)) * s / rad
    vx += -dy * g
    vy += dx * g


def curl_amp(ts):
    if ts < 58.5:
        return 7 + 4 * sm((ts - 52.5) / 2)
    if ts < 70.5:
        return 9 + 4 * rms_at(ts)
    if ts < 75.0:
        return 14
    if ts < 81.0:
        return 22
    if ts < 106.1:
        return 18 + 14 * sm((ts - 84) / 22)
    return 0


def simulate(path):
    """run the ink simulation for every frame; store dye as uint8 sqrt-encoded memmap"""
    store = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(NF, GH, GW, 3))
    dye = np.zeros((GH, GW, 3), np.float32)
    vx = np.zeros((GH, GW), np.float32)
    vy = np.zeros((GH, GW), np.float32)
    keys = curl_keys(40)
    dt = 1 / FPS
    on, ostr, bass = AU["on"], AU["ostr"], AU["bass"]
    strong = on[ostr > 0.55]
    rp = rose_petals()
    fp = falling_petals()
    mcx, mcy = MOON_C[0] / CELL, MOON_C[1] / CELL
    rng = np.random.default_rng(1)
    for fi in range(NF):
        ts = T0 + fi / FPS
        if not frozen(ts):
            t_prev = ts - dt
            h = ht(ts)
            kf = (h - T0) / 1.6
            k0 = int(kf) % (len(keys) - 1)
            f = kf - int(kf)
            f = f * f * (3 - 2 * f)
            cx_ = keys[k0][0] * (1 - f) + keys[k0 + 1][0] * f
            cy_ = keys[k0][1] * (1 - f) + keys[k0 + 1][1] * f
            amp = curl_amp(ts)
            vx = vx * 0.90 + cx_ * amp * 0.10 * 3
            vy = vy * 0.90 + cy_ * amp * 0.10 * 3
            # --- section forces and sources
            if 48.1 <= ts < 58.5:
                if t_prev < 48.1 + 1e-6 <= ts or fi == 0:
                    dye[..., 1] += gauss(GW / 2, GH * 0.40, 9) * 3.0
                    radial(vx, vy, GW / 2, GH * 0.40, 60, 28)
                if 48.6 <= ts < 52.6:
                    dye[..., 1] += TITLE["grid"] * 0.035
                    dye[..., 2] += TITLE["grid"] * 0.02
                hold = 1 - 0.85 * TITLE["grid"] * (1 - sm((ts - 53.5) / 1.5))
                vx *= hold
                vy *= hold
                vy -= TITLE["grid"] * 6 * sm((ts - 54) / 2)
                for b in events(bass, t_prev, ts):
                    radial(vx, vy, mcx, mcy, 30, 60)
            if 58.54 <= ts < 70.5:
                e = 0.045 * (0.7 + 0.8 * rms_at(ts)) * (1 - sm((ts - 68.0) / 1.5))
                dye[..., 0] += TITLE["pg0"] * e + TITLE["pf0"] * e * 0.02
                dye[..., 1] += TITLE["pg1"] * e + TITLE["pf1"] * e * 0.02
                for d in events(AU["downs"], t_prev, ts):
                    vx += (TITLE["pf0"] * 70 - TITLE["pf1"] * 70) * gauss(GW / 2, GH / 2, 120)
                for b in events(bass, t_prev, ts):
                    radial(vx, vy, mcx, mcy, 26, 50)
                for p in fp:
                    st = fall_state(p, ts)
                    if st and st[1] > H * 0.55:
                        dye[..., 2] += gauss(st[0] / CELL, st[1] / CELL, 2.5) * 0.25
                if ts > 68.0:
                    swirl(vx, vy, GW / 2, GH / 2, 220 * sm((ts - 68) / 2.5), 120)
            if 71.9 <= ts < 75.0:
                u = (ts - 71.9) / 3.1
                swirl(vx, vy, GW / 2, GH / 2, 110 + 260 * u, 110)
                ang = ts * 6
                for k in range(3):
                    a = ang + k * 2.094
                    px, py = GW / 2 + math.cos(a) * 80, GH / 2 + math.sin(a) * 80
                    dye[..., 1 + (k % 2)] += gauss(px, py, 4.5) * (0.8 + 1.2 * u)
                vx *= 0.97
                vy *= 0.97
            if t_prev < 75.0 <= ts:
                radial(vx, vy, GW / 2, GH / 2, 420, 140)
                dye[..., 2] += gauss(GW / 2, GH / 2, 10) * 0.8
            if 75.0 <= ts < 106.1:
                for k, o in enumerate(events(strong, t_prev, ts)):
                    j = int(o * 1000)
                    rr = np.random.default_rng(j)
                    px, py = rr.uniform(GW * 0.12, GW * 0.88), rr.uniform(GH * 0.15, GH * 0.85)
                    if ts >= 84.0:
                        py = rr.uniform(GH * 0.10, GH * 0.55)
                    s = float(np.interp(o, on, ostr))
                    ch = j % 3
                    dye[..., ch] += gauss(px, py, 2.8 + 1.6 * s) * (1.0 + 0.8 * s)
                    radial(vx, vy, px, py, 110 + 90 * s, 9)
                    swirl(vx, vy, px + 6, py, (1 if j % 2 else -1) * 160, 12)
                for b in events(bass, t_prev, ts):
                    radial(vx, vy, mcx, mcy, 60, 90)
            if 84.0 <= ts < 106.1:
                u = (ts - 84.0) / 22.0
                rx, ry = ROSE_POS[0] / CELL, (ROSE_POS[1] - 60) / CELL
                dye[..., 1] += gauss(rx, ry, 5) * (0.18 + 0.35 * u)
                vy -= gauss(rx, ry - 10, 12) * 10
                for p in rp:
                    st = petal_state(p, ts)
                    if st and st[4] > 0:
                        dye[..., 2] += gauss(st[0] / CELL, st[1] / CELL, 4) * 0.35 * st[4]
                        dye[..., 1] += gauss(st[0] / CELL, st[1] / CELL, 6) * 0.15 * st[4]
                dye[..., 1] += gauss(mcx, mcy, 40) * 0.006 * u
            # --- advance
            vx, vy = advect(vx, vx, vy, dt), advect(vy, vx, vy, dt)
            dye = advect(dye, vx, vy, dt)
            dye = cv2.GaussianBlur(dye, (0, 0), 0.32) * (0.990 if ts >= 71.9 else 0.986)
            np.clip(dye, 0, 4, out=dye)
        store[fi] = (np.sqrt(np.clip(dye / 4.0, 0, 1)) * 255 + 0.5).astype(np.uint8)
    store.flush()
    del store


# --------------------------------------------------------------- rendering ---
ST = {}


def build_render_assets():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - MOON_C[0], yy - MOON_C[1])
    ST["moon_d"] = d
    ST["moonglow"] = np.exp(-np.clip(d - MOON_R, 0, None) / 420).astype(np.float32)
    ST["moon_disc"] = np.clip(MOON_R - d, 0, 1.5) / 1.5
    r = np.random.default_rng(2)
    mar = cv2.resize(r.random((14, 14)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    mar2 = cv2.resize(r.random((60, 60)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    ST["moon_tex"] = (0.86 + 0.10 * mar + 0.05 * mar2 - 0.12 * np.clip(mar - 0.6, 0, 1)).astype(np.float32)
    limb = np.clip(1 - (d / MOON_R) ** 2, 0, 1) ** 0.25
    ST["moon_limb"] = limb.astype(np.float32)
    sky = np.zeros((H, W, 3), np.float32) + BG
    sky += np.exp(-((yy - H * 0.25) / (H * 0.55)) ** 2)[..., None] * INDIGO * 0.9
    sky += np.exp(-((xx - W / 2) ** 2 + (yy - H * 1.05) ** 2) / (2 * (W * 0.7) ** 2))[..., None] * VIOLET * 0.25
    ST["sky"] = sky.astype(np.float32)
    ST["vig"] = np.clip(1 - 0.55 * (((xx - W / 2) / (W * 0.65)) ** 2 + ((yy - H * 0.45) / (H * 0.62)) ** 2) ** 1.3, 0.25, 1).astype(np.float32)
    ST["grain"] = [cv2.resize(np.random.default_rng(k).normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H)) for k in range(6)]
    ST["rose"] = rose_sprite()
    ST["petals"] = [petal_sprite(k) for k in range(6)]
    rnd = np.random.default_rng(9)
    n = 260
    ST["motes"] = dict(x=rnd.uniform(0, W, n), y=rnd.uniform(0, H, n), z=rnd.uniform(0.4, 1.6, n), p=rnd.uniform(0, 6.3, n))


def moon_layer(ts):
    blood = sm((ts - 84.0) / 22.0) if ts >= 84 else 0.0
    col = MOON * (1 - blood) + BLOOD * blood
    halo_col = SILVER * 0.55 * (1 - blood) + BLOOD * 0.8 * blood
    pulse = 1 + 0.25 * down_pulse(ts) * (ts >= 58.5)
    disc = ST["moon_disc"][..., None] * ST["moon_tex"][..., None] * ST["moon_limb"][..., None] * col * 0.82
    halo = (ST["moonglow"] * 0.38 * pulse)[..., None] * halo_col
    return disc + halo, blood


def draw_motes(img, ts, amt):
    m = ST["motes"]
    q = 2
    acc = np.zeros((H // q, W // q), np.float32)
    y = (m["y"] - ts * 14 / m["z"]) % H
    x = m["x"] + 16 * np.sin(ts * 0.4 + m["p"]) / m["z"]
    v = np.clip(0.45 / m["z"], 0.08, 1) * (0.6 + 0.4 * np.sin(ts * 1.7 + m["p"]))
    for xi, yi, vi, zi in zip(x / q, y / q, v, m["z"]):
        cv2.circle(acc, (int(xi), int(yi)), 1 if zi > 0.6 else 2, float(vi), -1)
    acc = cv2.resize(cv2.GaussianBlur(acc, (0, 0), 0.8), (W, H))
    img += (acc * (0.25 + 0.9 * ST["moonglow"]) * amt)[..., None] * SILVER


def text_layer(img, ts):
    def word(s, cy, a, size=86, col=MOON):
        if a <= 0:
            return
        sp = tsprite(s, RR.TIRO, size, pad=20)
        over_c(img, col, sp * a, W / 2, cy)

    # title subtitle
    a = sm((ts - 50.5) / 0.8) * (1 - sm((ts - 55.5) / 1.2))
    if a > 0:
        sub = tracked("SEEDHE MAUT", RR.MONO, 24, 14)
        over_c(img, SILVER, sub * 0.7 * a, W / 2, H * 0.56 + 190)
    word("नशा", H * 0.62, sm((ts - 70.55) / 0.35) * (1 - sm((ts - 71.75) / 0.3)), 92)
    word("ठहरो", H * 0.62, sm((ts - 81.05) / 0.4) * (1 - sm((ts - 83.7) / 0.3)), 92)
    if ts >= 106.2:
        f = sm((ts - 106.3) / 0.8)
        sp = tsprite("मदिरा", RR.ROZHA, 128, pad=30)
        rgb = metal(sp, bevel=4, strength=1.2, lo=WINE * 0.3, mid=ROSE * 0.95, hi=MOON, spec=18)
        over_c(img, rgb, sp * f, W / 2, H * 0.60)
        cr = tag_sprite("SEEDHE MAUT · PROD. HURRICANE · DL91 FM", 20)
        over_c(img, MOON, cr * 0.6 * f, W / 2, H * 0.60 + 120)


def render_frame(fi, store):
    ts = T0 + fi / FPS
    dye = (store[fi].astype(np.float32) / 255) ** 2 * 4.0
    dens = cv2.resize(dye, (W, H), interpolation=cv2.INTER_CUBIC)
    np.clip(dens, 0, None, out=dens)
    tot = dens.sum(-1)
    img = ST["sky"].copy()
    moon, blood = moon_layer(ts)
    img += moon * np.exp(-tot * 1.1)[..., None]
    # ink: absorbs the light behind it and glows where the moon lights it
    light = 0.58 + 1.0 * ST["moonglow"]
    colr = (dens @ INKS) / (tot[..., None] + 1e-4)
    em = (1 - np.exp(-tot * 0.65))[..., None] * colr * light[..., None]
    gy, gx = np.gradient(cv2.GaussianBlur(tot, (0, 0), 2.0))
    ux, uy = MOON_C[0] - W / 2, MOON_C[1] - H
    sheen = np.clip(-(gx * 0 + gy) * 6.0, 0, 1) * ST["moonglow"] * 0.6
    img = img * np.exp(-tot * 0.6)[..., None] + em * 1.15 + (sheen[..., None] * (SILVER * (1 - blood) + ROSE * blood))
    # the title shows through the ink while it forms
    if 48.6 <= ts < 56.5:
        a = sm((ts - 49.0) / 1.2) * (1 - sm((ts - 53.0) / 3.0))
        img += (blur(TITLE["full"], 3) * 0.35 * a)[..., None] * ROSE
    # silhouettes: faint rim so the faces read under the smoke
    if 58.54 <= ts < 70.5:
        a = sm((ts - 58.6) / 1.0) * (1 - sm((ts - 68.5) / 1.5))
        L, R = TITLE["prof"]
        eL = np.clip(L - cv2.erode(L, np.ones((5, 5), np.uint8)), 0, 1)
        eR = np.clip(R - cv2.erode(R, np.ones((5, 5), np.uint8)), 0, 1)
        img += (blur(eL, 1.5) * 0.55 * a + blur(eL, 12) * 0.25 * a)[..., None] * SILVER
        img += (blur(eR, 1.5) * 0.55 * a + blur(eR, 12) * 0.25 * a)[..., None] * ROSE
    # falling petals (chorus, after the bass enters)
    for k, p in enumerate(falling_petals()) if 58.5 <= ts < 71.9 else []:
        st = fall_state(p, ts)
        if st:
            rgb, m = ST["petals"][k % 6]
            M = affine(st[0], st[1], 0.55, 0.55 * abs(math.cos(math.radians(st[2]))) + 0.1, st[2], 90, 120)
            draw_sprite(img, rgb, m, M)
    # the rose sheds its petals
    if 84.0 <= ts:
        rgb, al = ST["rose"]
        a = sm((ts - 84.0) / 0.8)
        u = (ts - 84.0) / 22.0
        sc = 1.0 - 0.18 * sm(u)
        M = affine(ROSE_POS[0], ROSE_POS[1], sc, sc, -4 + 2 * math.sin(ht(ts) * 0.8), al.shape[1], al.shape[0])
        draw_sprite(img, rgb * (1 - 0.35 * sm(u)), al, M, mul=a * (1 - sm((ts - 106.2) / 0.6)))
        for k, p in enumerate(rose_petals()):
            st = petal_state(p, ts)
            if st:
                prgb, pm = ST["petals"][k % 6]
                M = affine(st[0], st[1], 0.7, 0.7 * (0.35 + 0.65 * abs(math.cos(math.radians(st[2] * 0.7)))), st[2], 90, 120)
                draw_sprite(img, prgb, pm, M, mul=st[3])
    if 75.0 <= ts < 75.25:
        img += (1 - (ts - 75.0) / 0.25) * 0.35 * ROSE
    draw_motes(img, ts, 0.9)
    if ts >= 106.2:
        img = img * (1 - 0.6 * sm((ts - 106.3) / 0.9))
    text_layer(img, ts)
    # post
    lum = img.max(-1)
    br = np.clip(lum - 0.55, 0, None)
    img = img + (blur(br, 16) * 0.5 + blur(br, 50) * 0.45)[..., None] * np.array([0.95, 0.85, 1.0], np.float32)
    img = img * ST["vig"][..., None]
    img = np.where(img > 0.85, 0.85 + 0.15 * np.tanh((img - 0.85) / 0.15), img)
    img = img + ST["grain"][fi % 6][..., None] * 0.016
    if fi < 3:
        img = img * (fi / 3)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------- main --
CFG = {}


def setup(audio, fontdir):
    analyse(audio)
    build_static(fontdir)
    build_render_assets()


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
    setup(audio, fontdir)
    CFG["sim"] = os.path.join(work, "ink_sim.npy")
    simulate(CFG["sim"])
    print("simulated", flush=True)
    if "--preview" in sys.argv:
        os.makedirs(out, exist_ok=True)
        store = np.load(CFG["sim"], mmap_mode="r")
        for s in sys.argv[sys.argv.index("--preview") + 1].split(","):
            ts = float(s)
            fi = min(NF - 1, int(round((ts - T0) * FPS)))
            Image.fromarray(render_frame(fi, store)).save(os.path.join(out, f"i_{ts:06.2f}.png"))
        return
    nw = int(os.environ.get("WORKERS", "4"))
    b = np.linspace(0, NF, nw + 1).astype(int)
    jobs = [(k, int(b[k]), int(b[k + 1]), os.path.join(work, f"ink_chunk{k}.mp4")) for k in range(nw)]
    with mp.get_context("fork").Pool(nw) as pool:
        parts = pool.map(worker, jobs)
    lst = os.path.join(work, "ink_chunks.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", str(T0), "-t", str(DUR), "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", f"afade=t=in:st=0:d=0.03,afade=t=out:st={DUR - 0.8}:d=0.8", "-c:a", "aac", "-b:a", "256k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
