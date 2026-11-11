#!/usr/bin/env python3
"""
SEEDHE MAUT - RED // HIJACKED BROADCAST  (1920x1080 @ 24 fps)

A pirate TV feed: the RED video plays on a main monitor (true aspect, the
burned-in lines stay visible) and is pushed through render modes that switch
on the bar grid, like the Namastute remix, rebuilt for RED. Around it: a case
file with suspect mugshots frozen on downbeats, a live red spectrum analyser,
a terminal that typed the hijack, a breaking-news ticker, REDACTED bars that
slam on bass hits, SIGNAL LOST drops at the big section changes, a test card
intro and a CRT power-off outro.

usage:
  python3 render_red16.py SOURCE.mp4 OUT.mp4 WORKDIR FONTDIR
  python3 render_red16.py SOURCE.mp4 OUTDIR  WORKDIR FONTDIR --preview 3,12,40
"""
import math
import multiprocessing as mp
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "madira-reel"))
from render_reel import over, over_c, affine, draw_sprite, blur, sm, tsprite  # noqa: E402

W, H, FPS = 1920, 1080, 24
MW, MH = 1018, 720              # monitor (source aspect 1526x1080)
MX, MY = 70, 150
DUR = 70.7
NF = int(DUR * FPS)
SRC = None

RED = np.array([0.88, 0.07, 0.10], np.float32)
DEEP = np.array([0.30, 0.0, 0.03], np.float32)
WHITE = np.array([0.94, 0.94, 0.92], np.float32)
INK = np.array([0.06, 0.06, 0.07], np.float32)
REDI, WHITEI, GREYI = (226, 18, 26), (238, 238, 232), (130, 130, 130)
AU = {}
FN = {}
SEG = []
TAKEOVER = [22.8, 37.8, 58.7]


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
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=9000, hop_length=512), ref=np.max)
    AU.update(beats=bt, bass=lt[pk], downs=bt[::4], spec=np.clip((S + 70) / 70, 0, 1), spec_t=librosa.times_like(S, sr=sr, hop_length=512))


def last(arr, t):
    i = int(np.searchsorted(arr, t, "right")) - 1
    return (i, t - arr[i]) if i >= 0 else (-1, 9.0)


def pulse(arr, t, tau):
    i, dt = last(arr, t)
    return math.exp(-dt / tau) if i >= 0 else 0.0


# --------------------------------------------------------------- schedule --
MODES = {
    "raw": "render red.mp4 --mode raw",
    "glyph": "render red.mp4 --mode glyph_लाल",
    "sobel": "render red.mp4 --mode sobel   # edges only, truth intact",
    "news": "render red.mp4 --mode newsprint",
    "thermal": "render red.mp4 --mode thermal_red",
    "bayer": "render red.mp4 --mode bayer_8x8",
    "mirror": "render red.mp4 --mode mirror --axis both",
    "mosh": "render red.mp4 --mode datamosh",
    "echo": "render red.mp4 --mode echo --trail 12",
}
EXTRA = ["censor --apply   → permission denied", "grep -r 'fear' ~/seedhe_maut   → 0 matches",
         "signal --hijack   ✓", "sudo chmod -R 777 /truth   ✓", "mute --target artist   ✗ failed",
         "ping SEEDHE_MAUT   → 64 bytes, 0% loss"]


def build_schedule():
    d = AU["downs"]

    def bars(a, b, every, modes):
        ds = [x for x in d if a - 0.05 <= x < b]
        if not ds or ds[0] > a + 0.2:
            ds = [a] + ds
        out = []
        for k, i in enumerate(range(0, len(ds), every)):
            s0 = ds[i] if i else a
            s1 = ds[i + every] if i + every < len(ds) else b
            out.append((s0, s1, modes[k % len(modes)]))
        return out

    S = [(0.0, 2.3, "intro"), (2.3, 7.6, "raw"), (7.6, 10.3, "glyph")]
    S += bars(10.3, 22.8, 2, ["sobel", "news", "thermal", "raw"])
    S += bars(22.8, 35.1, 2, ["mirror", "glyph", "mosh", "bayer"])
    S += [(35.1, 37.8, "echo")]
    S += bars(37.8, 58.7, 1, ["sobel", "thermal", "glyph", "news", "mirror", "bayer", "echo", "raw"])
    S += [(58.7, 62.6, "mosh"), (62.6, 65.1, "bayer"), (65.1, 69.0, "raw"), (69.0, DUR + 1, "outro")]
    SEG[:] = S
    log = [(0.9, "claude: open ~/seedhe_maut/red.mp4"), (1.4, "signal --hijack --channel 01"), (1.9, "bypass censor.board   ✓")]
    k = 0
    for a, b, m in S:
        if m in MODES:
            log.append((a, MODES[m]))
            if b - a > 2.5:
                log.append((a + (b - a) * 0.55, EXTRA[k % len(EXTRA)]))
                k += 1
    log.append((69.0, "power --off"))
    AU["log"] = log


def seg_at(t):
    for s in SEG:
        if s[0] <= t < s[1]:
            return s
    return SEG[-1]


# ----------------------------------------------------------------- source --
class Src:
    def __init__(self, f0):
        self.f0 = max(0, f0)
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{self.f0 / FPS:.4f}", "-i", SRC, "-vf",
                                   f"fps={FPS},scale={MW}:{MH}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.n = self.f0 - 1
        self.buf = {}

    def get(self, fi):
        fi = max(0, fi)
        while self.n < fi:
            b = self.p.stdout.read(MW * MH * 3)
            if len(b) < MW * MH * 3:
                break
            self.n += 1
            self.buf[self.n] = np.frombuffer(b, np.uint8).reshape(MH, MW, 3)
            self.buf.pop(self.n - 130, None)
        fi = min(fi, self.n)
        while fi not in self.buf and fi < self.n:
            fi += 1
        return self.buf[fi].astype(np.float32) / 255

    def close(self):
        self.p.kill()


def redmask(f):
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return cv2.GaussianBlur((np.clip((r - np.maximum(g, b) - 0.10) * 5, 0, 1) * (r > 0.22)).astype(np.float32), (0, 0), 1.0)


def bw_red(f):
    m = redmask(f)[..., None]
    g = np.clip((f.mean(-1, keepdims=True) - 0.03) * 1.15, 0, 1)
    return g * (1 - m) + np.clip(f * np.array([1.45, 0.5, 0.55], np.float32), 0, 1) * m, m[..., 0]


# ------------------------------------------------------------------ modes --
ATL = {}


def glyph_atlas(cell):
    if cell in ATL:
        return ATL[cell]
    chars = [" ", "·", ":", "-", "+", "ल", "ा", "ल"]
    out = []
    for c in chars:
        im = Image.new("L", (cell, cell), 0)
        f = ImageFont.truetype(FN["deva"] if "ऀ" <= c <= "ॿ" else FN["mono"], int(cell * 0.95))
        ImageDraw.Draw(im).text((cell / 2, cell / 2), c, font=f, fill=255, anchor="mm")
        out.append(np.asarray(im, np.float32) / 255)
    ATL[cell] = np.stack(out)
    return ATL[cell]


def tile(atlas, idx, h, w):
    gh, gw = idx.shape
    c = atlas.shape[1]
    img = atlas[idx].transpose(0, 2, 1, 3).reshape(gh * c, gw * c)
    out = np.zeros((h, w), np.float32)
    out[:min(h, gh * c), :min(w, gw * c)] = img[:h, :w]
    return out


def mode_frame(mode, src, fi, t):
    f = src.get(fi)
    img, m = bw_red(f)
    g = img.mean(-1)
    if mode == "raw":
        return img + blur(m, 14)[..., None] * RED * 0.4
    if mode == "glyph":
        cell = 14
        gh, gw = MH // cell + 1, MW // cell + 1
        lum = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA)
        lum = np.clip(lum / (np.percentile(lum, 97) + 1e-3), 0, 1)
        mr = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)
        A = glyph_atlas(cell)
        lv = np.clip(lum * len(A), 0, len(A) - 1).astype(int)
        mask = tile(A, lv, MH, MW)
        col = np.where((mr > 0.25)[..., None], RED * 1.1, WHITE * lum[..., None] * 0.9 + 0.1)
        col = cv2.resize(col.astype(np.float32), (gw * cell, gh * cell), interpolation=cv2.INTER_NEAREST)[:MH, :MW]
        o = mask[..., None] * col
        return o + blur(o, 3) * 0.5
    if mode == "sobel":
        gg = cv2.GaussianBlur(g, (0, 0), 1.2)
        e = np.sqrt(cv2.Sobel(gg, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(gg, cv2.CV_32F, 0, 1) ** 2)
        e = np.clip(e / (np.percentile(e[::4, ::4], 98) + 1e-4), 0, 1) ** 1.2
        col = WHITE * (1 - m[..., None]) + RED * 1.2 * m[..., None]
        return e[..., None] * col + blur(e, 5)[..., None] * col * 0.6 + g[..., None] * 0.06
    if mode == "news":
        cell = 9
        gh, gw = MH // cell + 1, MW // cell + 1
        lum = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA)
        mr = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)
        if "dots" not in ATL:
            lv = []
            for k in range(12):
                d = np.zeros((cell * 4, cell * 4), np.float32)
                r = k / 11 * cell * 0.62 * 4
                if r > 0:
                    cv2.circle(d, (cell * 2, cell * 2), int(r), 1.0, -1, cv2.LINE_AA)
                lv.append(cv2.resize(d, (cell, cell), interpolation=cv2.INTER_AREA))
            ATL["dots"] = np.stack(lv)
        dots = tile(ATL["dots"], np.clip((1 - lum) * 11.99, 0, 11).astype(int), MH, MW)
        mr_up = cv2.resize(mr, (gw * cell, gh * cell), interpolation=cv2.INTER_NEAREST)[:MH, :MW]
        paper = np.array([0.92, 0.90, 0.84], np.float32)
        ink = np.where((mr_up > 0.3)[..., None], RED, INK)
        return paper * (1 - dots[..., None]) + ink * dots[..., None]
    if mode == "thermal":
        u = np.clip(g * 1.1, 0, 1)
        stops = [(0, (0.02, 0.0, 0.01)), (0.35, (0.30, 0.0, 0.03)), (0.6, (0.85, 0.06, 0.08)), (0.82, (1.0, 0.5, 0.35)), (1, (1, 0.97, 0.92))]
        x = [s[0] for s in stops]
        return np.stack([np.interp(u, x, [s[1][c] for s in stops]) for c in range(3)], -1).astype(np.float32)
    if mode == "bayer":
        if "bay" not in ATL:
            def bayer(n):
                if n == 1:
                    return np.zeros((1, 1))
                b = bayer(n // 2)
                return np.block([[4 * b, 4 * b + 2], [4 * b + 3, 4 * b + 1]])
            B = (bayer(8) + 0.5) / 64
            ATL["bay"] = np.tile(B, (MH // 3 // 8 + 1, MW // 3 // 8 + 1))[:MH // 3, :MW // 3].astype(np.float32)
        small = cv2.resize(g, (MW // 3, MH // 3), interpolation=cv2.INTER_AREA)
        bit = cv2.resize((small > ATL["bay"]).astype(np.float32), (MW, MH), interpolation=cv2.INTER_NEAREST)[..., None]
        i, _ = last(AU["beats"], t)
        hi = WHITE if i % 2 else RED
        return INK + (hi - INK) * bit
    if mode == "mirror":
        o = img.copy()
        o[:, MW // 2:] = o[:, :MW - MW // 2][:, ::-1]
        if last(AU["beats"], t)[0] % 2:
            o[MH // 2:] = o[:MH - MH // 2][::-1]
        return o
    if mode == "echo":
        o = g[..., None] * WHITE
        for k, (lag, c, a) in enumerate(((4, RED, 0.9), (8, WHITE * 0.6, 0.6), (12, RED * 0.6, 0.45))):
            gk = bw_red(src.get(fi - lag))[0].mean(-1)
            o = np.maximum(o, gk[..., None] * c * a)
        return o
    if mode == "mosh":
        return mosh(src, fi, t)
    return img


MO = {}


def mosh(src, fi, t):
    seg = seg_at(t)
    if MO.get("seg") != seg[0] or MO.get("fi") != fi - 1:
        st = int(round(seg[0] * FPS)) if MO.get("seg") != seg[0] else fi
        st = max(st, fi - 100)
        base = bw_red(src.get(st))[0]
        MO.update(seg=seg[0], acc=base, prev=base, fi=st)
        for k in range(st + 1, fi):
            mstep(src, k)
    mstep(src, fi)
    return MO["acc"]


def mstep(src, k):
    cur, m = bw_red(src.get(k))
    a = cv2.resize((MO["prev"].mean(-1) * 255).astype(np.uint8), (MW // 4, MH // 4))
    b = cv2.resize((cur.mean(-1) * 255).astype(np.uint8), (MW // 4, MH // 4))
    fl = cv2.resize(cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0), (MW, MH)) * 4
    gx, gy = np.meshgrid(np.arange(MW, dtype=np.float32), np.arange(MH, dtype=np.float32))
    acc = cv2.remap(MO["acc"], gx - fl[..., 0] * 1.4, gy - fl[..., 1] * 1.4, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    t = k / FPS
    rf = 0.05 + 0.9 * (pulse(AU["downs"], t, 0.05) > 0.9)
    acc = acc * (1 - rf) + cur * rf
    acc = acc * (1 - m[..., None] * 0.6) + cur * m[..., None] * 0.6
    MO.update(acc=acc, prev=cur, fi=k)


# --------------------------------------------------------------- overlays --
@np.vectorize
def _noop(x):
    return x


def text(d, xy, s, size, fill, font="mono", anchor="la"):
    f = ImageFont.truetype(FN[font], size)
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def mixed(d, x, y, s, size, fill):
    lat = ImageFont.truetype(FN["mono"], size)
    dev = ImageFont.truetype(FN["deva_r"], int(size * 1.1))
    run, isd = "", None
    for c in s + "\0":
        dd = "ऀ" <= c <= "ॿ"
        if c == "\0" or (run and dd != isd and c != " "):
            f = dev if isd else lat
            d.text((x, y), run, font=f, fill=fill, anchor="ls")
            x += f.getlength(run)
            run = ""
        if c != "\0":
            if not run:
                isd = dd
            run += c
    return x


def static_noise(fi, h, w):
    r = np.random.default_rng(fi)
    n = r.random((h // 3, w // 3)).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_NEAREST)


def test_card():
    img = np.zeros((H, W, 3), np.float32)
    cols = [WHITE, np.array([0.7, 0.7, 0.7]), RED, DEEP, INK, RED * 0.6, WHITE * 0.4]
    bw = W // len(cols)
    for k, c in enumerate(cols):
        img[:int(H * 0.68), k * bw:(k + 1) * bw] = c
    for k in range(14):
        img[int(H * 0.68):int(H * 0.78), k * W // 14:(k + 1) * W // 14] = (k / 13)
    img[int(H * 0.78):] = INK
    return img


def compose(t, fi, src):
    seg = seg_at(t)
    mode = seg[2]
    img = np.zeros((H, W, 3), np.float32) + np.array([0.035, 0.035, 0.04], np.float32)
    yy = np.arange(H)[:, None]
    img[::4] *= 0.85
    # --- intro / outro full screens
    if mode == "intro":
        if t < 0.7:
            n = static_noise(fi, H, W)
            img = n[..., None] * WHITE * 0.8 + n[..., None] * RED * 0.2
        elif t < 1.35:
            img = test_card()
        else:
            img = img * 0.3
        return img, mode
    # --- main monitor
    mon = mode_frame(mode if mode != "outro" else "raw", src, fi, t)
    z = 1 + 0.03 * pulse(AU["beats"], t, 0.09) + 0.03 * pulse(AU["downs"], t, 0.12)
    if z > 1.002:
        M = np.float32([[z, 0, (1 - z) * MW / 2], [0, z, (1 - z) * MH / 2]])
        mon = cv2.warpAffine(mon, M, (MW, MH), borderMode=cv2.BORDER_REFLECT)
    # redaction bars on bass hits
    for b in AU["bass"][(AU["bass"] <= t) & (AU["bass"] > t - 0.22)]:
        r = np.random.default_rng(int(b * 1000))
        for _ in range(r.integers(1, 3)):
            y0 = int(r.uniform(0.08, 0.80) * MH)
            x0 = int(r.uniform(0.0, 0.5) * MW)
            w = int(r.uniform(0.25, 0.6) * MW)
            mon[y0:y0 + int(r.uniform(26, 60)), x0:x0 + w] = RED
    # signal lost takeovers
    for tk in TAKEOVER:
        if 0 <= t - tk < 0.45:
            n = static_noise(fi, MH, MW)
            mon = n[..., None] * 0.7 + RED * 0.1
    sh = pulse(AU["bass"], t, 0.06)
    ox = int(6 * sh * math.sin(fi * 2.3))
    img[MY:MY + MH, MX + ox:MX + ox + MW] = mon
    # chromatic fringe on strong hits
    if sh > 0.5:
        img[..., 0] = np.roll(img[..., 0], 4, 1)
        img[..., 2] = np.roll(img[..., 2], -4, 1)
    # case-file mugshots: frozen on the last three downbeats
    d = AU["downs"]
    past = d[d <= t][-3:]
    for k, tb in enumerate(past[::-1]):
        fr = bw_red(src.get(int(round(tb * FPS))))[0]
        thumb = cv2.resize(fr, (300, 212), interpolation=cv2.INTER_AREA)
        thumb = thumb.mean(-1, keepdims=True) * np.ones(3) * (1.0 if k == 0 else 0.7)
        x0, y0 = 1150 + k * 0, 190 + k * 232
        x0 = 1150 + (k % 1) * 0
        img[y0:y0 + 212, x0:x0 + 300] = thumb
    # spectrum
    i = min(len(AU["spec_t"]) - 1, int(np.searchsorted(AU["spec_t"], t)))
    col = AU["spec"][:, i]
    for b in range(40):
        hgt = int(col[b] ** 1.6 * 170)
        x = 1480 + b * 9
        img[880 - hgt:880, x:x + 6] = RED if b % 3 else WHITE * 0.8
    # ticker band
    img[945:1005, :] = INK * 1.4
    img[945:1005, :230] = RED
    # outro: CRT power-off
    if mode == "outro":
        u = (t - seg[0]) / (DUR - seg[0])
        sq = max(0.004, 1 - sm(u / 0.35))
        full = img.copy()
        M = np.float32([[1 + 0.4 * sm((u - 0.3) / 0.2) * 0, 0, 0], [0, sq, (1 - sq) * H / 2]])
        img = cv2.warpAffine(full, M, (W, H))
        if sq < 0.02:
            img = np.zeros_like(img)
            w = int(W * max(0.0, 1 - sm((u - 0.35) / 0.15)))
            img[H // 2 - 2:H // 2 + 2, W // 2 - w // 2:W // 2 + w // 2] = WHITE
    return img, mode


def hud(pil, t, fi, mode):
    d = ImageDraw.Draw(pil, "RGBA")
    if mode == "intro":
        if 0.7 <= t < 1.35:
            text(d, (W / 2, H * 0.34), "TRANSMISSION INTERRUPTED", 64, WHITEI, "bold", "mm")
        if t >= 1.35:
            for k, (ta, s) in enumerate([x for x in AU["log"] if x[0] < 2.3]):
                if t >= ta:
                    n = int((t - ta) * 50) + 1
                    d.text((W * 0.18, H * 0.40 + k * 48), ">", font=ImageFont.truetype(FN["bold"], 34), fill=REDI)
                    mixed(d, W * 0.18 + 34, H * 0.40 + k * 48 + 30, s[:n], 34, WHITEI)
        return
    # top bar
    d.rectangle([0, 0, W, 96], fill=(10, 10, 12, 255))
    text(d, (70, 50), "RED", 54, REDI, "anton", "lm")
    text(d, (190, 52), "//  HIJACKED BROADCAST   ·   SEEDHE MAUT", 24, WHITEI, "mono", "lm")
    if int(t * 2) % 2 == 0:
        d.ellipse([W - 260, 36, W - 232, 64], fill=REDI)
    text(d, (W - 220, 50), "REC", 26, WHITEI, "bold", "lm")
    text(d, (W - 70, 50), f"{int(t // 60):02d}:{int(t % 60):02d}:{int((t % 1) * 24):02d}", 26, WHITEI, "mono", "rm")
    # monitor frame + labels
    d.rectangle([MX - 3, MY - 3, MX + MW + 3, MY + MH + 3], outline=(240, 240, 240, 200), width=2)
    L = 30
    for x, y, sx, sy in ((MX - 14, MY - 14, 1, 1), (MX + MW + 14, MY - 14, -1, 1), (MX - 14, MY + MH + 14, 1, -1), (MX + MW + 14, MY + MH + 14, -1, -1)):
        d.line([(x, y), (x + sx * L, y)], fill=REDI, width=4)
        d.line([(x, y), (x, y + sy * L)], fill=REDI, width=4)
    lab = "MODE ▸ " + mode.upper().replace("GLYPH", "GLYPH_लाल")
    d.rectangle([MX, MY + MH + 18, MX + 360, MY + MH + 58], fill=REDI)
    mixed(d, MX + 12, MY + MH + 48, lab, 24, WHITEI)
    text(d, (MX + MW, MY + MH + 40), f"CH 01   ·   {AU['bpm']:.0f} BPM   ·   FEED: SEEDHE_MAUT/RED", 20, GREYI, "mono", "rm")
    # case file
    text(d, (1150, 150), "CASE FILE  ·  SUBJECTS", 22, WHITEI, "bold", "lm")
    d.line([(1150, 172), (1450, 172)], fill=REDI, width=3)
    for k in range(3):
        y0 = 190 + k * 232
        d.rectangle([1150, y0, 1450, y0 + 212], outline=(230, 230, 230, 160), width=1)
        text(d, (1158, y0 + 200), f"SUBJECT 0{k + 1}", 16, WHITEI, "mono", "ls")
        if k == 0:
            st = "WANTED" if int(t / 1.76) % 2 else "UNCENSORED"
            text(d, (1300, y0 + 106), st, 44, (226, 18, 26, 210), "anton", "mm")
    # terminal log
    text(d, (1480, 150), "TERMINAL", 22, WHITEI, "bold", "lm")
    d.line([(1480, 172), (1850, 172)], fill=REDI, width=3)
    shown = [x for x in AU["log"] if x[0] <= t][-9:]
    for k, (ta, s) in enumerate(shown):
        newest = k == len(shown) - 1
        n = int((t - ta) * 45) + 1 if newest else len(s)
        y = 206 + k * 30
        d.text((1480, y), ">", font=ImageFont.truetype(FN["bold"], 18), fill=REDI, anchor="ls")
        xe = mixed(d, 1500, y, s[:n][:34], 17, WHITEI if newest else (150, 150, 150))
        if newest and int(t * 3) % 2 == 0:
            d.rectangle([xe + 3, y - 15, xe + 12, y + 2], fill=REDI)
    text(d, (1480, 500), "AUDIO  ·  LIVE", 22, WHITEI, "bold", "lm")
    d.line([(1480, 522), (1850, 522)], fill=REDI, width=3)
    # ticker
    text(d, (115, 975), "BREAKING", 30, WHITEI, "anton", "mm")
    msg = ("SEEDHE MAUT  ·  RED  ·  SIGNAL HIJACKED BY CLAUDE  ·  EVERYTHING IS BLACK AND WHITE EXCEPT THE RED  ·  "
           "CENSOR BOARD: PERMISSION DENIED  ·  CHANNEL 01 IS NOW LIVE  ·  ") * 3
    f = ImageFont.truetype(FN["bold"], 28)
    off = (t * 220) % f.getlength(msg[:len(msg) // 3])
    d.text((250 - off, 975), msg, font=f, fill=WHITEI, anchor="lm")
    d.rectangle([0, 945, 230, 1005], fill=REDI)
    text(d, (115, 975), "BREAKING", 30, WHITEI, "anton", "mm")


def render_frame(fi, src):
    t = fi / FPS
    img, mode = compose(t, fi, src)
    if mode == "outro":
        u = (t - seg_at(t)[0]) / (DUR - seg_at(t)[0])
        if u > 0.55:
            a = sm((u - 0.55) / 0.2)
            sp = tsprite("RED", FN["anton"], 420, pad=40)
            over_c(img, RED, sp * a, W / 2, H / 2 - 40)
            sub = tsprite("SEEDHE MAUT", FN["bold"], 44, pad=10)
            over_c(img, WHITE, sub * a, W / 2, H / 2 + 220)
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    else:
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        hud(pil, t, fi, mode)
    out = np.asarray(pil, np.float32) / 255
    seg = seg_at(t)
    if 0 < seg[0] and t - seg[0] < 2 / FPS and mode not in ("outro",):
        out = out * 0.55 + RED * 0.45
    g = np.random.default_rng(fi % 8).normal(0, 0.02, (H // 2, W // 2)).astype(np.float32)
    out = out + cv2.resize(g, (W, H))[..., None]
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------- main --
def setup(src, fontdir):
    global SRC
    SRC = src
    FN.update(mono=os.path.join(fontdir, "IBMPlexMono-Regular.ttf"), bold=os.path.join(fontdir, "IBMPlexMono-Medium.ttf"),
              anton=os.path.join(fontdir, "Anton-Regular.ttf"), deva="/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
              deva_r="/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf")
    analyse(os.environ.get("RED_AUDIO", src))
    import librosa
    AU["bpm"] = 60 / np.median(np.diff(AU["beats"]))
    build_schedule()


def worker(job):
    k, f0, f1, out = job
    src = Src(f0 - 110)
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
    setup(src, fontdir)
    if "--preview" in sys.argv:
        os.makedirs(out, exist_ok=True)
        for s in sys.argv[sys.argv.index("--preview") + 1].split(","):
            t = float(s)
            fi = int(round(t * FPS))
            sr = Src(fi - 110)
            MO.clear()
            Image.fromarray(render_frame(fi, sr)).save(os.path.join(out, f"h_{t:05.2f}.png"))
            sr.close()
        return
    nw = int(os.environ.get("WORKERS", "4"))
    b = np.linspace(0, NF, nw + 1).astype(int)
    jobs = [(k, int(b[k]), int(b[k + 1]), os.path.join(work, f"red16_chunk{k}.mp4")) for k in range(nw)]
    with mp.get_context("fork").Pool(nw) as pool:
        parts = pool.map(worker, jobs)
    lst = os.path.join(work, "red16_chunks.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", src,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest",
                    "-movflags", "+faststart", out], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
