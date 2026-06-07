#!/usr/bin/env python3
"""
Garden of Earthly Delights — Puzzle Piece Finder
=================================================
Point your webcam at a single puzzle piece on a plain (white) background, press
SPACE, and the app shows you exactly where that piece belongs on the painting —
at any rotation, any zoom.

How it works
------------
Bosch's painting is extraordinarily detailed, so even one 1000-piece fragment
carries enough texture to be located by feature matching. The app finds SIFT
keypoints on a high-resolution reference image of the whole painting (once, then
caches them), segments the piece from its background, matches its features to the
reference, and uses a RANSAC homography to pin down its location and orientation.

SETUP
-----
1.  pip install opencv-contrib-python numpy
2.  Get a HIGH-RESOLUTION image of the full triptych and save it next to this file
    as `reference.jpg` (or change REFERENCE_PATH below). The painting is public
    domain; search "Garden of Earthly Delights Prado high resolution" or grab the
    large file from Wikimedia Commons. Bigger = better matching. A few thousand
    pixels wide is the minimum; 6000px+ is great.
    IMPORTANT: use a clean photo of just the painting (the three panels). If your
    image includes frames/walls, crop them off first.
3.  python puzzle_finder.py

CONTROLS (in the window)
------------------------
  SPACE / C ... capture the current frame and locate the piece
  F .......... locate a piece from an image FILE instead of the webcam
  S .......... save the last result to disk (result_XXX.png)
  R .......... rebuild reference features (ignore cache)
  Q / ESC .... quit
"""

import os, sys, time, glob, re, argparse, threading, subprocess, shutil, wave, tempfile

# Make OpenCV's Qt window find system fonts, so its zoom/pan toolbar tooltips
# aren't blank (fixes the "QFontDatabase: Cannot find font directory" warnings).
# Must run BEFORE importing cv2.
if os.name == "posix" and not os.environ.get("QT_QPA_FONTDIR"):
    for _p in ("/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")):
        if os.path.isdir(_p):
            os.environ["QT_QPA_FONTDIR"] = _p
            break

import cv2
import numpy as np

# ----------------------------------------------------------------------------- CONFIG
REFERENCE_PATH = "garden.jpg"      # high-res image of the full triptych
CAMERA_INDEX   = None              # None = list cameras and ask at startup.
                                   # Set a number (0,1,2,...) to skip the prompt,
                                   # or pass --camera N on the command line.
REF_MAX_DIM    = 9000              # reference is downscaled to this max dimension
                                   #   (higher -> better small-piece matching, more RAM/CPU)
DISPLAY_MAX_DIM = 9000             # result viewer keeps the painting at up to this size
                                   #   so zooming stays sharp (display only; not matching).
                                   #   Lower it if the result window feels heavy on RAM.
CACHE_PATH     = "reference_features.npz"   # cached SIFT features (auto-created)
CAMERA_CONFIG  = "camera_choice.txt"        # remembers the last camera (auto-created)
FOCUS_CONFIG   = "camera_focus.txt"         # remembers manual focus (auto-created)
FOCUS_STEP     = 5                          # focus change per keypress ( [ ] or , . )

# Panel boundaries as fractions of the reference WIDTH (left|center|right).
# For a properly-cropped open triptych the wings are ~half the centre panel, so
# the splits fall near 0.25 and 0.75. Adjust to match YOUR reference image.
PANEL_BOUNDS   = (0.25, 0.75)
PANEL_NAMES    = ("LEFT panel (Eden)", "CENTRE panel (Garden)", "RIGHT panel (Hell)")

# Matching parameters (well tested). If you get WRONG locations, raise MIN_INLIERS
# or MIN_RATIO. If it misses pieces it should find, lower them.
SIFT_CONTRAST = 0.012              # lower -> more (weaker) features, better on low-contrast pieces
RATIOS        = (0.75, 0.82, 0.90) # Lowe ratio test, relaxed progressively until enough matches
MIN_GOOD      = 7                  # minimum good matches to attempt a geometric fit
MIN_INLIERS   = 1                  # minimum inliers to TRUST a result
MIN_RATIO     = 0.01               # minimum inliers/good ratio to TRUST a result
# The photographed piece is ~2x higher-res than its region on the reference and is
# covered in print halftone/paper grain (~80% of its SIFT features) that the smooth
# reference lacks. Matching the piece downscaled to these longest-side sizes removes
# that noise and aligns feature scale; the size with the most inliers wins. A piece of
# a 1000-piece triptych spans ~185px on a 7793px reference, so these bracket that.
PIECE_MATCH_SIZES = (250, 185, 140)
# Repetitive painting texture (foliage, dark areas) makes ~85% of a piece's features
# self-similar, so the ratio test rejects correct matches. When the strict ratio test
# is weak, fall back to feeding the top-K reference neighbours per feature into RANSAC
# and letting geometric consistency find the right ones (verified by inlier count).
CONFIDENT_INLIERS      = 8         # a strict-ratio fit this strong is trusted immediately
MULTIMATCH_K           = 3         # candidate reference neighbours per feature in the fallback
MULTIMATCH_MAX_CAND    = 600       # cap candidates fed to RANSAC (bounds runtime)
MULTIMATCH_MIN_INLIERS = 4         # a fallback fit needs at least this many inliers to be trusted
BEEP = True                        # play a soft chime when a capture run finishes
                                   #   (these two together prevent confident-but-wrong answers)
PIECE_DENOISE = True               # set True only if heavy card texture/glare causes
                                   # missed matches; it trades some detail for smoothness

# Piece tracing (segmentation). Background is assumed roughly uniform.
EDGE_CROP_PCT = 3.0                # crop this %% inward from the piece edge so paper/
                                   # shadow is never matched. Raise if any paper sneaks in;
                                   # lower toward 0 to trace closer to the true edge.
TEX_WIN       = 9                  # texture window: smaller = tighter/sharper edges and
                                   # deeper dents, larger = smoother but baggier.
SEG_WORK      = 900                # internal working resolution for tracing (speed).
                                   # Lower = faster live preview, slightly coarser outline.

# Result plausibility / auto-retry. A piece of a 1000-piece triptych is small, so
# a match that projects onto the painting bigger than this fraction of its HEIGHT
# is almost certainly wrong. SPACE keeps re-matching until a plausible result
# (or AUTO_RETRY_TRIES is reached), so you press it far less often.
MAX_PIECE_FRAC   = 0.25
AUTO_RETRY_TRIES = 30

MAX_WINDOW   = (1500, 850)         # max on-screen size for the result image (w, h)
# -----------------------------------------------------------------------------


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def prep(gray):
    """Local-contrast normalization (CLAHE). Applied identically to the reference
    and to each piece so a warm, unevenly-lit photo resembles the clean scan."""
    return _CLAHE.apply(gray)


def prep_piece(piece_gray):
    """Piece-only: gently suppress the woven card texture, then normalize like the
    reference. The texture creates spurious features the flat scan doesn't have."""
    g = cv2.bilateralFilter(piece_gray, 5, 40, 40) if PIECE_DENOISE else piece_gray
    return prep(g)


def rootsift(desc):
    """RootSIFT: L1-normalize then sqrt. Makes SIFT far more robust to the
    appearance changes between a printed/photographed piece and a clean scan."""
    if desc is None:
        return None
    desc = desc / (desc.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(desc).astype(np.float32)


def _fill_interior_holes(mask):
    """Fill holes that are NOT connected to the image border. Concave indents
    (the 'female' parts) ARE connected to the border background, so they stay
    open; only enclosed interior holes (e.g. a dark printed spot misread as
    background) get filled."""
    h, w = mask.shape
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return mask | cv2.bitwise_not(ff)


def _local_std(gray, k=11):
    """Local standard deviation: low on smooth paper, high on printed detail."""
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (k, k))
    mean2 = cv2.boxFilter(g * g, -1, (k, k))
    return np.sqrt(np.clip(mean2 - mean * mean, 0, None))


def segment_piece(bgr, edge_crop_pct=None, tex_k=None, border=10, min_area_frac=0.003):
    """Trace one puzzle piece on a uniform background.

    Two cues make it work whatever the paper colour: local texture (smooth paper
    & soft shadows vs. detailed print) and colour distance from the background.
    The texture mask is de-inflated so the boundary snaps to the real cardboard
    edge; only interior holes are filled so concave indents stay open. Finally the
    silhouette is cropped inward by EDGE_CROP_PCT to guarantee no paper/shadow is
    matched. Runs at a fixed working resolution (SEG_WORK) for speed and so the
    live preview and the captured result trace identically.
    Returns (masked_bgr, mask, (x,y,w,h)) or None.
    """
    if edge_crop_pct is None:
        edge_crop_pct = EDGE_CROP_PCT
    if tex_k is None:
        tex_k = TEX_WIN
    H0, W0 = bgr.shape[:2]
    s = SEG_WORK / float(max(H0, W0))
    img = cv2.resize(bgr, (max(1, int(W0 * s)), max(1, int(H0 * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    eb = np.concatenate([lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3),
                         lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3)])
    bg = np.median(eb, axis=0)
    colordist = np.linalg.norm(lab - bg, axis=2)
    col_thr = max(16.0, float(np.std(np.linalg.norm(eb - bg, axis=1))) * 4 + 12)

    std = _local_std(gray, tex_k)
    bvals = np.concatenate([std[:border].ravel(), std[-border:].ravel(),
                            std[:, :border].ravel(), std[:, -border:].ravel()])
    tex_thr = max(2.5, float(np.median(bvals)) * 2.5 + 1.0)

    # texture mask, de-inflated by the window radius so its boundary is sharp
    tex_mask = (std > tex_thr).astype(np.uint8) * 255
    tex_mask = cv2.erode(tex_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tex_k, tex_k)), 1)
    col_mask = (colordist > col_thr).astype(np.uint8) * 255   # already sharp (per-pixel)
    fg = cv2.bitwise_or(tex_mask, col_mask)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    if area < min_area_frac * h * w or area > 0.95 * h * w:
        return None

    mask = (lbl == idx).astype(np.uint8) * 255
    mask = _fill_interior_holes(mask)
    mask = cv2.medianBlur(mask, 5)            # light smoothing (won't bridge indents)
    mask = _fill_interior_holes(mask)
    if edge_crop_pct > 0:                      # crop inward to exclude paper/shadow
        r = max(1, int(round(edge_crop_pct / 100.0 * (area ** 0.5))))
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)), 1)

    mask = cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_LINEAR)
    mask = ((mask > 127) * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    out = bgr.copy(); out[mask == 0] = 0
    return out[y:y + bh, x:x + bw], mask[y:y + bh, x:x + bw], (x, y, bw, bh)


def piece_outline(frame):
    """Return just the piece's contour (full-frame coords) for the live preview.
    segment_piece already runs at SEG_WORK internally, so the preview and the
    captured result trace identically. None if no piece is found."""
    seg = segment_piece(frame)
    if seg is None:
        return None
    _, mask, (x, y, bw, bh) = seg
    full = np.zeros(frame.shape[:2], np.uint8)
    full[y:y + bh, x:x + bw] = mask
    cnts, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea).astype(np.int32)


# Bump when the feature format changes, so stale caches are rebuilt automatically.
CACHE_VERSION = 2


class Reference:
    """Holds the reference image plus its (RootSIFT) features and a FLANN index."""

    def __init__(self, path, max_dim=REF_MAX_DIM, cache=CACHE_PATH, use_cache=True):
        src = cv2.imread(path)
        if src is None:
            raise FileNotFoundError(
                f"Could not read reference image '{path}'. "
                "Save a high-res photo of the full triptych there (see SETUP in the header).")
        h, w = src.shape[:2]
        s = min(1.0, max_dim / max(h, w))
        self.img = cv2.resize(src, (int(w * s), int(h * s))) if s < 1 else src
        # higher-res copy used only for the result display, so zooming stays crisp
        ds = min(1.0, DISPLAY_MAX_DIM / max(h, w))
        self.disp = cv2.resize(src, (int(w * ds), int(h * ds))) if ds < 1 else src
        self.disp_k = self.disp.shape[1] / self.img.shape[1]   # working -> display scale
        self.sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=SIFT_CONTRAST, edgeThreshold=14)

        sig = (CACHE_VERSION, os.path.getmtime(path), os.path.getsize(path),
               self.img.shape[0], self.img.shape[1])
        if use_cache and os.path.exists(cache):
            try:
                d = np.load(cache, allow_pickle=False)
                if tuple(d["sig"]) == sig:
                    self.desc, self.kp_pts = d["desc"], d["pts"]
                    print(f"loaded cached features: {len(self.kp_pts)} keypoints")
                else:
                    self._compute(cache, sig)
            except Exception:
                self._compute(cache, sig)
        else:
            self._compute(cache, sig)

        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        self.flann.add([self.desc]); self.flann.train()

    def _compute(self, cache, sig):
        print("computing reference features (one-time, please wait)...")
        t = time.time()
        kp, desc = self.sift.detectAndCompute(prep(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)), None)
        self.desc = rootsift(desc)
        self.kp_pts = np.float32([k.pt for k in kp])
        print(f"  {len(kp)} keypoints in {time.time()-t:.1f}s")
        try:
            np.savez(cache, desc=self.desc, pts=self.kp_pts, sig=np.array(sig, np.float64))
            print(f"  cached to {cache}")
        except Exception as e:
            print(f"  (could not write cache: {e})")

    def locate(self, piece_bgr, mask=None):
        """Match the piece to the painting. The piece is tried at several downscales
        (PIECE_MATCH_SIZES): the photo carries far more fine detail + print halftone
        than the reference, so matching it near the reference's own scale gives far
        cleaner correspondences. The scale with the most RANSAC inliers wins."""
        gray_full = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2GRAY)
        H0, W0 = gray_full.shape
        best = None
        tried = []
        for target in PIECE_MATCH_SIZES:
            s = target / float(max(H0, W0))
            if s >= 1.0:
                g, ms, sc = prep_piece(gray_full), mask, 1.0
            else:
                small = cv2.resize(gray_full, (max(1, int(W0 * s)), max(1, int(H0 * s))),
                                   interpolation=cv2.INTER_AREA)
                g = prep_piece(small)
                ms = (cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
                      if mask is not None else None)
                sc = s
            res = self._match_at(g, ms, mask, sc)
            tried.append(res)
            if res.get("ok") and (best is None or res["n_inliers"] > best["n_inliers"]):
                best = res
        if best is not None:
            return best
        # none passed -> report the most informative failure
        for r in tried:
            if "n_inliers" in r:
                return r
        return tried[0] if tried else {"ok": False, "reason": "no features"}

    def _match_at(self, g, mask_small, mask_full, sc):
        """One matching attempt at a given piece scale. g/mask_small are at the
        downscaled size; sc maps downscaled-piece coords back to original-piece
        coords. Tier 1 = strict Lowe ratio (precise). If that is weak, tier 2 feeds
        the top-K reference neighbours per feature into RANSAC so geometric
        consistency can recover matches the ratio test killed on repetitive texture."""
        kp, desc = self.sift.detectAndCompute(g, mask_small)
        if desc is None or len(kp) < MIN_GOOD:
            return {"ok": False, "reason": "too few features on the piece",
                    "n_kp": 0 if kp is None else len(kp)}
        desc = rootsift(desc)
        knn = [p for p in self.flann.knnMatch(desc, k=MULTIMATCH_K) if len(p) >= 2]

        # tier 1: strict ratio test
        good = []
        for r in RATIOS:
            good = [p[0] for p in knn if p[0].distance < r * p[1].distance]
            if len(good) >= max(MIN_GOOD, 20):
                break
        res1 = self._fit(good, kp, g, sc, mask_full, maxIters=5000, min_inliers=MIN_INLIERS)
        if res1.get("ok") and res1["n_inliers"] >= CONFIDENT_INLIERS:
            return res1

        # tier 2: multi-match fallback (interleave neighbour ranks, capped)
        cand = []
        for rank in range(MULTIMATCH_K):
            for p in knn:
                if len(p) > rank:
                    cand.append(p[rank])
            if len(cand) >= MULTIMATCH_MAX_CAND:
                break
        cand = cand[:MULTIMATCH_MAX_CAND]
        res2 = self._fit(cand, kp, g, sc, mask_full, maxIters=15000, min_inliers=MULTIMATCH_MIN_INLIERS)

        # keep whichever passed with more inliers
        oks = [r for r in (res1, res2) if r.get("ok")]
        if oks:
            return max(oks, key=lambda r: r["n_inliers"])
        return res1 if "n_inliers" in res1 else res2

    def _fit(self, matches, kp, g, sc, mask_full, maxIters, min_inliers):
        """Fit a similarity (4-DOF) from a set of candidate matches via RANSAC and
        build the result dict. Returns {ok:False,...} if the fit is too weak."""
        if len(matches) < MIN_GOOD:
            return {"ok": False, "reason": "not enough matches", "n_good": len(matches)}
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([self.kp_pts[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        # similarity (rotation + uniform scale + translation): 4 DOF, not an 8-DOF
        # homography, so spurious matches almost never produce a confident-but-wrong fit.
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=4.0, maxIters=maxIters, confidence=0.999)
        ninl = 0 if inl is None else int(inl.sum())
        ratio = ninl / max(len(matches), 1)
        if M is None or ninl < min_inliers or ratio < MIN_RATIO:
            return {"ok": False, "reason": "weak/ambiguous match",
                    "n_good": len(matches), "n_inliers": ninl, "ratio": ratio}
        scale = float(np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])))
        if not (0.02 < scale < 50):
            return {"ok": False, "reason": "implausible geometry",
                    "n_good": len(matches), "n_inliers": ninl, "ratio": ratio}
        h, w = g.shape
        quad = cv2.transform(
            np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), M).reshape(-1, 2)
        cx, cy = quad.mean(0)
        outline = None
        if mask_full is not None:                       # project the crisp full-res outline
            cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea).astype(np.float32)
                c = cv2.approxPolyDP(c, 1.5, True).astype(np.float32) * sc   # to downscaled coords
                outline = cv2.transform(c.reshape(-1, 1, 2), M).reshape(-1, 2)
        return {"ok": True, "n_good": len(matches), "n_inliers": ninl, "ratio": ratio,
                "center": (float(cx), float(cy)), "quad": quad, "outline": outline,
                "angle": float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
                "ref_size": (self.img.shape[1], self.img.shape[0])}


# ----------------------------------------------------------------------------- rendering
def which_panel(cx, ref_w):
    f = cx / ref_w
    b0, b1 = PANEL_BOUNDS
    return PANEL_NAMES[0] if f < b0 else (PANEL_NAMES[2] if f > b1 else PANEL_NAMES[1])


def render_result(ref, piece_bgr, r, show_overlay=True):
    """Build an annotated image at display resolution (so zooming stays sharp):
    the painting with the traced piece outline + a crosshair, plus an info panel.
    When show_overlay is False the outline/crosshair are omitted (peek beneath)."""
    rw, rh = ref.img.shape[1], ref.img.shape[0]        # working coords (for % / center)
    k = ref.disp_k                                      # working -> display scale
    out = ref.disp.copy()
    DH = out.shape[0]
    f = DH / 850.0                                      # UI scale: keeps look constant when fit-to-window
    def S(x):
        return max(1, int(round(x * f)))

    if r.get("ok") and show_overlay:
        poly = r.get("outline")
        if poly is None:
            poly = r["quad"]                            # fallback if no mask was available
        poly = (poly * k).astype(np.int32)
        cv2.polylines(out, [poly], True, (0, 0, 0), S(6), cv2.LINE_AA)      # dark halo for contrast
        cv2.polylines(out, [poly], True, (0, 255, 0), S(3), cv2.LINE_AA)    # the piece outline
        cx, cy = (np.array(r["center"]) * k).astype(int)
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, S(46), S(4))
        cv2.circle(out, (cx, cy), S(30), (0, 0, 255), S(3))

    H, W = out.shape[:2]
    barw = S(360)
    bar = np.full((H, barw, 3), 30, np.uint8)
    def text(y, s, col=(235, 235, 235), sz=0.6, th=1):
        cv2.putText(bar, s, (S(14), S(y)), cv2.FONT_HERSHEY_SIMPLEX, sz * f, col,
                    max(1, int(round(th * f))), cv2.LINE_AA)
    text(34, "PUZZLE PIECE FINDER", (120, 220, 255), 0.7, 2)
    if r.get("ok"):
        cx, cy = r["center"]; fx, fy = cx / rw, cy / rh
        text(78,  "MATCH FOUND", (90, 230, 90), 0.7, 2)
        text(116, which_panel(cx, rw), (160, 230, 255), 0.55)
        text(150, f"position:  {fx*100:4.1f}% across", sz=0.55)
        text(176, f"           {fy*100:4.1f}% down", sz=0.55)
        ang = r["angle"]
        text(214, f"rotate piece by {(-ang)%360:5.1f}deg", (160, 230, 255), 0.55)
        text(238, "(to match painting orientation)", (150, 150, 150), 0.42)
        text(276, f"confidence: {r['n_inliers']} inliers", sz=0.5)
        text(298, f"match quality: {r.get('ratio',0)*100:.0f}%", (150, 150, 150), 0.45)
    else:
        text(78, "NO CONFIDENT MATCH", (90, 90, 240), 0.7, 2)
        for i, line in enumerate(["Try again:",
                                  "- fill the frame with",
                                  "  the piece (zoom in)",
                                  "- even, bright light",
                                  "- plain background",
                                  "- sharp focus"]):
            text(118 + i * 26, line, (200, 200, 200), 0.5)
        text(300, f"({r.get('reason','?')})", (130, 130, 130), 0.45)

    # piece thumbnail — rotated to how it fits the painting when a match was found
    if piece_bgr is not None and piece_bgr.size:
        show = piece_bgr
        label = "captured piece:"
        if r.get("ok"):
            a = r["angle"]
            ph, pw = piece_bgr.shape[:2]
            Mr = cv2.getRotationMatrix2D((pw / 2, ph / 2), -a, 1.0)  # -angle aligns to painting
            cs, sn = abs(Mr[0, 0]), abs(Mr[0, 1])
            nw, nh = int(ph * sn + pw * cs), int(ph * cs + pw * sn)
            Mr[0, 2] += nw / 2 - pw / 2; Mr[1, 2] += nh / 2 - ph / 2
            show = cv2.warpAffine(piece_bgr, Mr, (nw, nh), borderValue=(0, 0, 0))
            label = "piece, rotated to fit:"
        tw = S(330); th = int(show.shape[0] * tw / show.shape[1])
        th = min(th, H - S(360))
        if th > 10:
            thumb = cv2.resize(show, (tw, th))
            bar[H - th - S(14):H - S(14), S(15):S(15) + tw] = thumb
            text((H - th - S(24)) / f, label, (150, 150, 150), 0.45)

    return np.hstack([out, bar])


# ----------------------------------------------------------------------------- piece I/O
def piece_size_frac(result):
    """Largest projected edge of the matched piece as a fraction of the painting
    height. The quad corners come from the image rectangle, so its two edge
    lengths are the piece's projected width and height (rotation-independent)."""
    if not result.get("ok"):
        return 0.0
    q = result.get("quad")
    if q is None:
        return 0.0
    side1 = float(np.hypot(*(q[1] - q[0])))
    side2 = float(np.hypot(*(q[2] - q[1])))
    return max(side1, side2) / result["ref_size"][1]


def plausible_match(result):
    """Reject matches whose projected piece is implausibly large (a 1000-piece
    triptych piece is small) -- compares to MAX_PIECE_FRAC of the painting height."""
    return result.get("ok") and piece_size_frac(result) <= MAX_PIECE_FRAC


def _result_summary(ref, result):
    if result.get("ok"):
        cx, cy = result["center"]
        return (f"FOUND -> {which_panel(cx, ref.img.shape[1])} at "
                f"{cx/ref.img.shape[1]*100:.1f}% across, {cy/ref.img.shape[0]*100:.1f}% down "
                f"(rotate {(-result['angle'])%360:.0f}deg, {result['n_inliers']} inliers)")
    return f"no confident match: {result.get('reason','?')}"


def analyze(ref, frame, verbose=True):
    """Segment + locate a piece. Returns (piece_bgr, result) and (optionally)
    prints a summary. Display is handled by the caller so results can be re-shown."""
    seg = segment_piece(frame)
    if seg is None:
        piece, result = frame, {"ok": False, "reason": "couldn't find a piece on the background"}
    else:
        piece, mask, _ = seg
        result = ref.locate(piece, mask)
    if verbose:
        print("  " + _result_summary(ref, result))
    return piece, result


def from_file(ref):
    path = input("\nPath to piece image: ").strip().strip('"').strip("'")
    img = cv2.imread(path)
    if img is None:
        print("  could not read that file."); return None
    return analyze(ref, img)


# ----------------------------------------------------------------------------- cameras
def list_video_devices():
    """Return [(index, name), ...] of candidate cameras."""
    nodes = sorted(glob.glob("/dev/video*"),
                   key=lambda p: int(re.findall(r"\d+", p)[0]) if re.findall(r"\d+", p) else 0)
    if nodes:                                   # Linux
        out = []
        for dev in nodes:
            idx = int(re.findall(r"\d+", dev)[0])
            name = ""
            try:
                with open(f"/sys/class/video4linux/video{idx}/name") as f:
                    name = f.read().strip()
            except Exception:
                pass
            out.append((idx, name))
        return out
    return [(i, "") for i in range(6)]           # other OSes: just probe 0..5


def _make_capture(idx):
    """Open a VideoCapture, preferring the V4L2 backend on Linux (avoids the
    flaky obsensor backend probing and is more stable for USB webcams)."""
    if os.name == "posix":
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(idx)


def probe_camera(idx):
    """Open a camera, try to read a frame. Returns (works, (w,h) or None)."""
    cap = _make_capture(idx)
    works, res = False, None
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            works = True
            res = (frame.shape[1], frame.shape[0])
    cap.release()
    return works, res


def find_working_cameras():
    """Probe all candidates; return [(index, name, (w,h)), ...] that actually deliver frames."""
    working = []
    for idx, name in list_video_devices():
        ok, res = probe_camera(idx)
        if ok:
            working.append((idx, name, res))
    return working


def choose_camera(working, default=None):
    """Print the working cameras and let the user pick one (Enter = first/default)."""
    if not working:
        print("  No working cameras found. You can still use 'F' to load piece images from files.")
        return default
    print("\nAvailable cameras:")
    for idx, name, res in working:
        print(f"  [{idx}] {name or 'camera'}  ({res[0]}x{res[1]})")
    first = default if default in [w[0] for w in working] else working[0][0]
    try:
        choice = input(f"Pick a camera index (Enter for {first}): ").strip()
    except EOFError:
        choice = ""
    if choice == "":
        return first
    try:
        return int(choice)
    except ValueError:
        print("  not a number; using", first)
        return first


CAM_HARD_STALL = 5.0   # seconds with no fresh frame before forcing a full reconnect
HISTORY_MAX = 12       # how many recent results to keep for arrow-key browsing
# Arrow key codes from cv2.waitKeyEx (X11/Qt, Windows, macOS variants)
_LEFT_KEYS  = (65361, 2424832, 63234)
_RIGHT_KEYS = (65363, 2555904, 63235)


class CameraStream:
    """Reads frames in a background thread and keeps only the latest one, so a
    stalled/blocking USB camera can never freeze the UI. Auto-reconnects when
    reads start failing."""

    def __init__(self, index, autofocus=True, focus=0):
        self.index = index
        self.autofocus = autofocus
        self.focus = int(focus)
        self.cap = None
        self._frame = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._reopen = True                       # open on the first loop iteration
        self._apply_pending = False               # reapply focus/AF on next loop
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _apply_cam_settings(self):
        if self.cap is None:
            return
        try:
            if self.autofocus:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            else:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self.cap.set(cv2.CAP_PROP_FOCUS, float(self.focus))
        except Exception:
            pass

    def _open(self):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        cap = _make_capture(self.index)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep frames fresh
            except Exception: pass
            self.cap = cap
            self._apply_cam_settings()             # (re)apply focus after every (re)open
            return True
        if cap is not None:
            cap.release()
        self.cap = None
        return False

    def _loop(self):
        fails = 0
        while self._running:
            if self._reopen:
                self._reopen = False
                self._open()
            if self._apply_pending:
                self._apply_pending = False
                self._apply_cam_settings()
            if self.cap is None:
                time.sleep(0.4); self._reopen = True; continue
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if ok and frame is not None:
                fails = 0
                with self._lock:
                    self._frame = frame
                    self._ts = time.time()
            else:
                fails += 1
                time.sleep(0.05)
                if fails >= 5:                    # camera stalled -> reconnect
                    fails = 0
                    self._reopen = True
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass

    def set_focus(self, value):
        self.focus = max(0, min(255, int(value)))
        self.autofocus = False
        self._apply_pending = True

    def nudge_focus(self, delta):
        self.set_focus(self.focus + delta)

    def toggle_autofocus(self):
        self.autofocus = not self.autofocus
        self._apply_pending = True

    def read(self):
        """Return (frame_copy, timestamp). frame is None until the first frame."""
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._ts

    def switch(self, index):
        self.index = index
        self._reopen = True

    def stop(self):
        self._running = False


def wait_for_frame(stream, timeout=2.5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        frame, _ = stream.read()
        if frame is not None:
            return True
        time.sleep(0.05)
    return False


def load_saved_camera():
    try:
        with open(CAMERA_CONFIG) as f:
            return int(f.read().strip())
    except Exception:
        return None


def save_camera(idx):
    try:
        with open(CAMERA_CONFIG, "w") as f:
            f.write(str(int(idx)))
    except Exception:
        pass


def load_focus():
    """Return (autofocus: bool, focus: int). Defaults to autofocus on."""
    try:
        with open(FOCUS_CONFIG) as f:
            af, foc = f.read().split()
            return bool(int(af)), int(foc)
    except Exception:
        return True, 0


def save_focus(autofocus, focus):
    try:
        with open(FOCUS_CONFIG, "w") as f:
            f.write(f"{1 if autofocus else 0} {int(focus)}")
    except Exception:
        pass


def _render_camera_picker(working, default):
    F = cv2.FONT_HERSHEY_SIMPLEX
    W = 860
    H = max(320, 170 + 46 * len(working))
    img = np.full((H, W, 3), 28, np.uint8)
    cv2.putText(img, "SELECT CAMERA", (30, 58), F, 1.0, (120, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "press the number key shown in [ ] to pick a camera",
                (30, 96), F, 0.6, (185, 185, 185), 1, cv2.LINE_AA)
    y = 156
    for idx, name, res in working:
        sel = (idx == default)
        if sel:
            cv2.rectangle(img, (20, y - 32), (W - 20, y + 14), (50, 72, 50), -1)
        col = (90, 230, 90) if sel else (232, 232, 232)
        cv2.putText(img, f"[{idx}]  {name}", (42, y), F, 0.75, col, 2 if sel else 1, cv2.LINE_AA)
        cv2.putText(img, f"{res[0]}x{res[1]}", (W - 230, y), F, 0.6, (170, 170, 170), 1, cv2.LINE_AA)
        y += 46
    cv2.putText(img, "Enter = highlighted     Q = quit", (30, H - 28), F, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
    return img


def pick_camera_in_window(win, working, default=None):
    """Show the camera list inside the OpenCV window and return the chosen index
    (or None to quit). Selection is by number key."""
    idxs = [w[0] for w in working]
    if not idxs:
        return None
    if default not in idxs:
        default = idxs[0]
    cv2.imshow(win, _render_camera_picker(working, default))
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (ord('q'), 27):
            return None
        if k in (13, 10):                 # Enter
            return default
        if 48 <= k <= 57 and (k - 48) in idxs:
            return k - 48


# ----------------------------------------------------------------------------- audio
def _write_tone(path, segments, sr=44100):
    """Write a short mono WAV. segments = list of (freq_hz, dur_s, amp). Each note
    gets a Hann envelope (zero at both ends) so there are no clicks -- a soft chime,
    not a harsh square-wave bleep. A quiet octave harmonic adds a little warmth."""
    parts = []
    for freq, dur, amp in segments:
        n = max(1, int(sr * dur))
        t = np.arange(n) / sr
        tone = np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * 2 * freq * t)
        tone /= np.max(np.abs(tone)) + 1e-9
        parts.append(amp * np.hanning(n) * tone)
    pcm = np.clip(np.concatenate(parts) * 32767, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _find_audio_player():
    for cmd in (["pw-play"], ["paplay"], ["aplay", "-q"], ["play", "-q"],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


class Beeper:
    """Plays a soft chime when a run finishes: a gentle two-note rise on a match,
    a lower two-note fall on no match. Playback is async so the UI never waits."""

    def __init__(self):
        self.player = _find_audio_player() if BEEP else None
        d = tempfile.gettempdir()
        self.ok_path = os.path.join(d, "puzzle_finder_ok.wav")
        self.no_path = os.path.join(d, "puzzle_finder_no.wav")
        if self.player:
            try:
                _write_tone(self.ok_path, [(659.25, 0.10, 0.5), (987.77, 0.16, 0.5)])  # E5 -> B5
                _write_tone(self.no_path, [(523.25, 0.10, 0.4), (392.00, 0.16, 0.4)])  # C5 -> G4
            except Exception:
                self.player = None
        elif BEEP:
            print("  (no audio player found for the chime; install one of: pipewire/pulseaudio, alsa-utils, sox)")

    def play(self, ok):
        if not self.player:
            return
        try:
            subprocess.Popen(self.player + [self.ok_path if ok else self.no_path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


# ----------------------------------------------------------------------------- main loop
def main(cli_camera=None):
    print("Loading reference...")
    ref = Reference(REFERENCE_PATH)
    print(f"Reference ready: {ref.img.shape[1]}x{ref.img.shape[0]} px")

    win = "Puzzle Piece Finder"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, MAX_WINDOW[0], MAX_WINDOW[1])  # initial size; big results fit-to-window, zoom for detail
    saver = {"last": None}
    beeper = Beeper()

    # ----- pick a camera (in-window, remembered across launches) -----
    chosen = cli_camera if cli_camera is not None else CAMERA_INDEX
    working = []
    cam_indices = []
    af, foc = load_focus()                   # remembered manual focus / autofocus
    if chosen is None:
        chosen = load_saved_camera()        # fast path: reuse last choice, skip the scan
    stream = CameraStream(chosen, af, foc) if chosen is not None else None
    if stream is not None and not wait_for_frame(stream):
        stream.stop(); stream = None
    if stream is None:                       # no/failed preset -> scan and let the user pick
        print("Scanning for cameras...")
        working = find_working_cameras()
        cam_indices = [w[0] for w in working]
        while stream is None:
            chosen = pick_camera_in_window(win, working, default=chosen)
            if chosen is None:
                break
            stream = CameraStream(chosen, af, foc)
            if not wait_for_frame(stream):
                stream.stop(); stream = None
                print(f"  camera {chosen} could not be opened; pick another")
    if stream is not None:
        save_camera(chosen)
        print(f"Using camera {chosen}.  Focus: {'auto' if stream.autofocus else stream.focus}")
    else:
        print("No camera selected. Use 'F' to load piece images from files.")

    print("\nControls:  SPACE/C capture   <- -> browse results   H hide outline/overlay   "
          "[ ]/, .  focus   A autofocus   P pick   N next cam   O outline   F file   S save   R rebuild   "
          "Shift/any key = back to live   Q quit")
    showing_result = False
    preview_on = True
    fcount = 0
    last_outline = None
    last_fresh = time.time()
    last_reset = 0.0
    overlay_hidden = False    # toggled by H to hide the result overlay (peek beneath)
    history = []              # recent (piece_bgr, result) for arrow-key browsing
    hist_idx = -1

    def render_hist(i):
        piece, result = history[i]
        canvas = render_result(ref, piece, result, show_overlay=not overlay_hidden)
        if len(history) > 1:
            cv2.putText(canvas, f"result {i+1}/{len(history)}   (<- -> to browse)",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)
        saver["last"] = canvas

    def push_result(piece, result):
        nonlocal hist_idx
        history.append((piece, result))
        if len(history) > HISTORY_MAX:
            history.pop(0)
        hist_idx = len(history) - 1

    # ----- background search worker (keeps the UI/zoom responsive while matching) -----
    searching = False
    search_lock = threading.Lock()
    search_out = {}

    def run_search():
        piece = None
        best = None
        for attempt in range(AUTO_RETRY_TRIES):
            frame, _ = stream.read()
            if frame is None:
                break
            piece, result = analyze(ref, frame, verbose=False)
            if plausible_match(result):
                best = result
                break
            if result.get("ok"):
                print(f"  attempt {attempt+1}: piece spans {piece_size_frac(result)*100:.0f}% "
                      f"of height (implausible), retrying...")
            else:
                print(f"  attempt {attempt+1}: {result.get('reason','no match')}, retrying...")
        if best is not None:
            print("  " + _result_summary(ref, best))
            show = best
        else:
            print(f"  no plausible match after {AUTO_RETRY_TRIES} tries")
            show = {"ok": False, "reason": f"no plausible match in {AUTO_RETRY_TRIES} tries"}
        with search_lock:
            search_out["piece"], search_out["show"], search_out["done"] = piece, show, True

    while True:
        if searching:                            # background match finished?
            with search_lock:
                done = search_out.get("done", False)
            if done:
                searching = False
                piece, show = search_out.get("piece"), search_out.get("show")
                beeper.play(bool(show and show.get("ok")))
                if piece is not None and show is not None:
                    push_result(piece, show)
                    overlay_hidden = False
                    render_hist(hist_idx)
                    showing_result = True

        if stream is not None and not showing_result:
            frame, ts = stream.read()
            now = time.time()
            fresh = frame is not None and (now - ts) < 1.5
            if fresh:
                last_fresh = now
                view = frame.copy()
                if preview_on:
                    fcount += 1
                    if fcount % 4 == 0:          # recompute periodically (segmentation is ~60ms)
                        last_outline = piece_outline(frame)
                    if last_outline is not None:
                        cv2.polylines(view, [last_outline], True, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(view, "Fill frame with ONE piece, then press SPACE",
                            (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(view, f"camera {chosen}   (N switch, P pick)",
                            (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                fstr = "auto" if stream.autofocus else str(stream.focus)
                cv2.putText(view, f"focus: {fstr}   ([ ] or , .   A=auto)",
                            (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(win, view)
            else:
                msg = frame.copy() if frame is not None else np.full((480, 640, 3), 30, np.uint8)
                cv2.putText(msg, "camera reconnecting...", (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.imshow(win, msg)
                if now - last_fresh > CAM_HARD_STALL and now - last_reset > CAM_HARD_STALL:
                    print("  camera stalled; reconnecting...")
                    stream.stop()
                    stream = CameraStream(stream.index, stream.autofocus, stream.focus)  # fresh thread; old one abandoned
                    last_reset = now
                    last_outline = None

        keyfull = cv2.waitKeyEx(30)
        k = 255 if keyfull == -1 else (keyfull & 0xFF)
        prev_showing = showing_result
        if keyfull in _LEFT_KEYS:                 # checked first: arrow codes & 0xFF collide with Q/R/S/T
            if showing_result and history:
                hist_idx = max(0, hist_idx - 1); render_hist(hist_idx)
        elif keyfull in _RIGHT_KEYS:
            if showing_result and history:
                hist_idx = min(len(history) - 1, hist_idx + 1); render_hist(hist_idx)
        elif k in (ord('q'), 27):
            break
        elif k == ord('o'):
            preview_on = not preview_on; last_outline = None
            print(f"  outline preview {'on' if preview_on else 'off'}")
        elif k in (ord(' '), ord('c')) and stream is not None:
            if not searching:                    # run the match off-thread so zoom/pan stay live
                print("Locating...")
                searching = True
                with search_lock:
                    search_out.clear(); search_out["done"] = False
                threading.Thread(target=run_search, daemon=True).start()
        elif k == ord('n'):
            if not cam_indices:                      # lazy scan if we took the fast path
                working = find_working_cameras(); cam_indices = [w[0] for w in working]
            if cam_indices:
                cur = cam_indices.index(chosen) if chosen in cam_indices else -1
                chosen = cam_indices[(cur + 1) % len(cam_indices)]
                if stream is not None:
                    stream.switch(chosen)
                else:
                    stream = CameraStream(chosen, af, foc)
                last_fresh = time.time(); last_outline = None
                save_camera(chosen)
                print(f"  switched to camera {chosen}")
            showing_result = False
        elif k == ord('p'):
            if not working:
                working = find_working_cameras(); cam_indices = [w[0] for w in working]
            pick = pick_camera_in_window(win, working, default=chosen)
            if pick is not None:
                chosen = pick
                if stream is not None:
                    stream.switch(chosen)
                else:
                    stream = CameraStream(chosen, af, foc)
                last_fresh = time.time(); last_outline = None
                save_camera(chosen); print(f"  using camera {chosen}")
            showing_result = False
        elif k in (ord(']'), ord('.')) and stream is not None:
            stream.nudge_focus(FOCUS_STEP)
            save_focus(stream.autofocus, stream.focus); print(f"  focus {stream.focus}")
        elif k in (ord('['), ord(',')) and stream is not None:
            stream.nudge_focus(-FOCUS_STEP)
            save_focus(stream.autofocus, stream.focus); print(f"  focus {stream.focus}")
        elif k == ord('a') and stream is not None:
            stream.toggle_autofocus()
            save_focus(stream.autofocus, stream.focus)
            print(f"  autofocus {'on' if stream.autofocus else 'off'} (focus {stream.focus})")
        elif k == ord('h'):
            if showing_result and history:
                overlay_hidden = not overlay_hidden  # result view: toggle outline+crosshair
                render_hist(hist_idx)
            else:
                preview_on = not preview_on          # live view: toggle the green border
                last_outline = None
        elif k == ord('f'):
            res = from_file(ref)
            if res is not None:
                push_result(*res)
                overlay_hidden = False
                render_hist(hist_idx)
                showing_result = True
        elif k == ord('s') and saver["last"] is not None:
            n = len(glob.glob("result_*.png"))
            fn = f"result_{n:03d}.png"; cv2.imwrite(fn, saver["last"]); print(f"  saved {fn}")
        elif k == ord('r'):
            print("Rebuilding features...")
            ref = Reference(REFERENCE_PATH, use_cache=False)
        elif keyfull != -1:  # any other key (e.g. Shift) returns to the live camera view
            showing_result = False

        if prev_showing and not showing_result:   # returned to the live view
            last_outline = None
            overlay_hidden = False
            last_fresh = time.time()               # don't let result-viewing time trip the watchdog

    if stream is not None:
        stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Locate a puzzle piece on the painting.")
    ap.add_argument("-c", "--camera", type=int, default=None,
                    help="camera index to use (skips the startup prompt)")
    args = ap.parse_args()
    main(cli_camera=args.camera)
