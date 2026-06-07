# Puzzle Piece Finder

Find where a single jigsaw puzzle piece belongs, by holding it under a webcam.

Vibe-coded using Claude Opus 4.8. The rest of the README plus the entire code
was written by Claude. I used this to solve a 1000 piece puzzle. Last time
I have solved a puzzle was maybe 20 years ago, and I have never done a 1000
piece monster. The puzzle was lying around for a month in my room until I decided
enough was enough and I might as well cheat. Using this software, it took me
about 8 hours to complete the puzzle. What I learned is that there is a learning
curve to jigsaw puzzles, and I should have started with a 300 piece puzzle instead.
I made this program MIT-licensed, so have fun with it!

Point a stationary webcam at a white sheet of paper, drop one puzzle piece in
view, and press **SPACE**. The program traces the piece, matches it against a
high‑resolution image of the finished puzzle, and shows you exactly where it
goes — which panel, the position as a percentage across and down, and how much
to rotate the piece to match the painting. It was built and battle‑tested
solving a **1000‑piece reproduction of Hieronymus Bosch's *The Garden of
Earthly Delights*** (the included default reference), but it works with any
puzzle whose finished picture you can supply as an image.

It is not magic — see [Expectations](#expectations) — but it can carry you
through the large, "where on earth does this go?" middle of a big puzzle.

---

## How it works

1. **Trace the piece.** Assuming a roughly uniform background, it segments the
   one piece in frame using local texture + colour distance, traces a tight
   outline (following tabs and dents), and crops slightly inward so paper and
   shadow are never matched.
2. **Match it to the painting.** It extracts SIFT features (RootSIFT), and
   matches them against cached features of your high‑res reference image using
   a similarity transform (rotation + uniform scale + translation) verified by
   RANSAC. To cope with the print halftone on the piece and the painting's
   repetitive texture, it matches the piece at several scales and, when the
   strict ratio test is starved, feeds multiple candidate matches per feature
   into RANSAC and lets geometric consistency find the right ones.
3. **Show the answer.** The painting is displayed with the piece's outline and
   a crosshair drawn where it belongs, plus a sidebar (panel, position %,
   rotation, confidence) and a thumbnail of the piece rotated to fit.

---

## Setup

### Hardware

- A **webcam**, mounted **stationary** above the surface (a phone tripod with a
  webcam clamp, a desk arm, a stack of books — anything that holds it still and
  pointing straight down).
- A **plain, matte, light‑coloured sheet** (white or neutral grey paper) as the
  background. Uniform and non‑glossy is what matters; the tracing assumes the
  background is featureless.
- **Even, diffuse lighting.** Avoid harsh glare/specular highlights on glossy
  pieces — they wipe out the detail the matcher needs. Soft, flat light is best.

### Software

- **Python 3.9+** (developed on 3.14).
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

  That's just NumPy and OpenCV. SIFT is included in `opencv-python` (the patent
  expired; no `-contrib` build is required).
- **Optional (Linux):** for the soft "finished" chime, have one audio player on
  your PATH — `pipewire` (`pw-play`), `pulseaudio` (`paplay`), `alsa-utils`
  (`aplay`), `sox` (`play`), or `ffmpeg` (`ffplay`). Without one it simply runs
  silent (or set `BEEP = False`).

### The reference image

You need a picture of the **finished** puzzle, the higher resolution the better.

- Save it next to the script as **`garden.jpg`**, or change `REFERENCE_PATH` at
  the top of `puzzle_finder.py` to point at your own image.
- **Resolution is the single biggest factor in match quality.** A bigger
  reference means each piece occupies more pixels and carries more distinctive
  features. Use the largest scan you can find. `REF_MAX_DIM` caps the working
  size (default 9000); raise it for accuracy, lower it if you run short on RAM.
- The first run builds a feature cache (`reference_features.npz`). This takes a
  little while on a big image; subsequent runs load it instantly. The cache is
  rebuilt automatically if you change the image or `REF_MAX_DIM` (or press `R`).

> The default subject, Bosch's *Garden of Earthly Delights* (c. 1490–1510), is
> in the public domain. Supply your own reference image for your own puzzle.

---

## Running it

```bash
python puzzle_finder.py            # scans cameras and asks which to use (first run)
python puzzle_finder.py --camera 0 # skip the prompt and use camera index 0
```

On first launch it lists available cameras in the window; press the number key
to pick one. Your choice (and focus settings) are remembered between sessions.

**Basic workflow:** put **one** piece on the sheet so it fills a good part of
the frame, check the green outline is hugging it, press **SPACE**, and read the
result. Press **SPACE** again from the result screen to try another piece (or
the same piece again — see below).

---

## Controls

| Key | Action |
| --- | --- |
| **SPACE** / **C** | Capture and locate the piece (auto‑retries — see below) |
| **← / →** | Browse back/forward through recent results |
| **H** | Hide/show the result outline + crosshair (in live view: hide/show the green border) |
| **O** | Toggle the live green outline preview |
| **[** or **,** | Decrease the manual focus value |
| **]** or **.** | Increase the manual focus value |
| **A** | Toggle autofocus on/off |
| **P** | Pick a camera (in‑window list) |
| **N** | Switch to the next detected camera |
| **F** | Load a piece from an image file instead of the camera |
| **S** | Save the current result as a PNG |
| **R** | Rebuild the reference feature cache |
| **Shift** (or any other key) | Return to the live camera view |
| **Q** / **Esc** | Quit |

Notes:

- **Focus:** the camera defaults to autofocus. Since your camera is stationary,
  set focus once with `[` `]` (or `,` `.`) until the piece is crisp; the value
  is shown on screen and remembered across sessions. Press `A` to return to
  autofocus.
- **Zooming:** the result image is shown at full reference resolution, so you
  can use the OpenCV window's zoom/pan toolbar to inspect the exact spot. The
  match runs on a background thread, so you can keep zooming while it searches.
- **Browsing:** every capture is kept in a short history (default 12). If a good
  result scrolls past because you pressed SPACE again too quickly, just press
  **←** to go back to it. `H` and `S` act on whichever result you're viewing.

---

## Expectations

This is a real, useful aid — not a solver that places every piece. Honest
guidance from building it:

- **Pieces with distinctive content** (faces, figures, sharp colour edges,
  text, hard boundaries) match reliably and quickly.
- **Pieces that are mostly one flat colour, very dark, or pure repetitive
  texture** (plain sky, grass, foliage stippling, shadow) are *genuinely hard* —
  they simply don't contain enough unique structure to localise, and no amount
  of tuning manufactures signal that isn't there. Place those by eye.
- In practice, expect it to confidently place a large majority of pieces and
  leave you the stubborn, low‑information ones — which are usually the ones
  you'd struggle with visually too.
- Match quality scales strongly with **reference resolution** and **piece photo
  quality** (focus, lighting, filling the frame). If results are poor, improve
  those first before touching parameters.

---

## A note on false positives (by design)

The default thresholds are deliberately **loose**. The design choice here is to
**prefer many false positives over a single false negative** — it is better for
the tool to occasionally offer a wrong location you reject at a glance than to
stay silent on a piece it could have placed.

Several mechanisms make that workable:

- A **plausibility filter**: a 1000‑piece puzzle piece is tiny, so any match
  whose projected size is larger than `MAX_PIECE_FRAC` of the painting height is
  obviously wrong and discarded — implausible matches are never even drawn.
- **Auto‑retry**: one press of SPACE re‑matches up to `AUTO_RETRY_TRIES` times
  (the matcher is randomised), keeping the first plausible result, so you rarely
  press it more than once.
- **You, verifying visually.** The outline drawn on the painting and the
  "rotated to fit" thumbnail make a wrong placement obvious in a fraction of a
  second.

If you would rather have fewer wrong suggestions (at the cost of missing some
hard pieces), tighten the gates below. If you want even more aggressive recall,
loosen them.

---

## Tuning guide

All parameters live in a clearly commented `CONFIG` block at the top of
`puzzle_finder.py`. The ones you're most likely to touch:

### Accuracy / recall

| Parameter | Default | What it does |
| --- | --- | --- |
| `REF_MAX_DIM` | `9000` | Working size of the reference. **The biggest lever.** Higher = more features per piece = better matching, at more RAM/CPU and a longer one‑time cache build. |
| `PIECE_MATCH_SIZES` | `(250, 185, 140)` | The piece is matched downscaled to each of these longest‑side sizes; the best wins. Centre them on the piece's true size **on the reference** (≈ `ref_width / sqrt(num_pieces) × aspect`). Re‑centre if you change `REF_MAX_DIM` or piece count. |
| `MAX_PIECE_FRAC` | `0.25` | Plausibility filter. A match projecting larger than this fraction of the painting height is rejected. **Lower = stricter** (fewer wrong placements drawn). |
| `AUTO_RETRY_TRIES` | `30` | How many times one SPACE press re‑matches before giving up. Higher = better odds on hard pieces, longer waits. |

### Precision vs. recall gates

| Parameter | Default | Effect |
| --- | --- | --- |
| `MIN_INLIERS` | `1` | Minimum RANSAC inliers to trust a result. **Raise** for fewer false positives, **lower** for more recall. |
| `MIN_RATIO` | `0.01` | Minimum inliers / good‑matches ratio to trust. Same direction as above. |
| `MIN_GOOD` | `7` | Minimum matches before a fit is even attempted. Lower lets thin pieces try; too low gets noisy. |
| `RATIOS` | `(0.75, 0.82, 0.90)` | Lowe ratio test, relaxed progressively. Higher last value = more (looser) matches. |
| `MULTIMATCH_K` | `3` | Candidate reference neighbours per feature in the repetitive‑texture fallback. |
| `MULTIMATCH_MIN_INLIERS` | `4` | Inliers a fallback match needs to be trusted. **Raise** to suppress fallback false positives. |
| `CONFIDENT_INLIERS` | `8` | A strict‑ratio fit this strong is accepted immediately (skips the fallback). |
| `SIFT_CONTRAST` | `0.012` | Lower detects more (weaker) features — can help very low‑contrast pieces. |

### Tracing (segmentation)

| Parameter | Default | Effect |
| --- | --- | --- |
| `EDGE_CROP_PCT` | `3.0` | Crops this % inward from the traced edge so paper/shadow isn't matched. Raise if background sneaks in; lower to hug the true edge. |
| `TEX_WIN` | `9` | Texture window. Smaller = tighter edges and deeper dents; larger = smoother but baggier. |
| `SEG_WORK` | `900` | Internal resolution for tracing. Lower = faster live preview, slightly coarser outline. |
| `PIECE_DENOISE` | `True` | Bilateral denoise on the piece; helps with heavy card texture/glare at the cost of some fine detail. |

### Your specific puzzle

| Parameter | What to set |
| --- | --- |
| `REFERENCE_PATH` | Path to your finished‑puzzle image. |
| `PANEL_BOUNDS` / `PANEL_NAMES` | Region splits (as fractions of width) and their labels. The defaults split a triptych at 0.25 / 0.75; set whatever makes sense for your image (e.g. a single `(0.5, 0.5)` with one name, or your own zones). |
| `FOCUS_STEP` | Focus change per keypress (depends on your camera's range). |
| `BEEP` | `False` to disable the completion chime. |

---

## Files the program creates

All written in the working directory, all safe to delete (they regenerate):

- `reference_features.npz` — cached SIFT features for your reference image.
- `camera_choice.txt` — the last camera you used.
- `camera_focus.txt` — your manual focus / autofocus setting.

---

## Troubleshooting

- **"Could not find Qt platform plugin 'wayland'" / font warnings (Linux):**
  harmless. OpenCV falls back to XWayland and text still renders.
- **Camera won't open / wrong device:** press `P` to pick another, or pass
  `--camera N`. On Linux the app forces the V4L2 backend for stability.
- **The window seems to hang briefly after a capture:** the match runs in the
  background now, but a very heavy search (large reference × many retries) still
  takes a few seconds; the chime tells you when it's done.
- **Lots of obviously‑wrong matches:** raise `MIN_INLIERS` / `MIN_RATIO` /
  `MULTIMATCH_MIN_INLIERS`, or lower `MAX_PIECE_FRAC`.
- **Misses pieces it should find:** lower those same gates, raise
  `REF_MAX_DIM`, and double‑check focus, lighting, and that the piece fills the
  frame.

---

## License

MIT — see [LICENSE](LICENSE).
