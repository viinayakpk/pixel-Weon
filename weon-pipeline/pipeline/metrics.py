"""Region-scoped metrics. All local — no API, no cost.

A global score cannot see a brand violation: conditions A and B here are 99.91% byte-identical.
So everything measures a declared region, never the picture.

Removed rather than kept for completeness: LPIPS (never installed), FFT energy, OCR edit-distance
(no engine available; it passes a wrong typeface anyway), and `diff_mask` — that one derived the
edit region from the output, which let the model choose which of its own damage was excluded from
its own score.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000


def load_rgb(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img)


def to_gray(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)


def match_size(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape[:2] != b.shape[:2]:
        b = np.asarray(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.LANCZOS))
    return a, b


MIN_EVAL_PIXELS = 64
_SSIM_WIN = 7


def _erode_for_window(m: np.ndarray, win: int = _SSIM_WIN) -> np.ndarray:
    """SSIM uses a local window, so values within win//2 px of a mask edge borrow context
    from outside it. Erode so we only average pixels whose window lies inside the mask."""
    return cv2.erode(m.astype(np.uint8), np.ones((win, win), np.uint8)).astype(bool)


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """SSIM averaged over ONLY the evaluated pixels, taken from the per-pixel SSIM map.

    The previous implementation zeroed pixels outside the mask and then ran global SSIM.
    The identical black regions then dominated the average, so a destroyed ROI could still
    score near 1.0. Never zero-and-average.

    Reproduced and persisted by experiments/eval_receipt.py -> outputs/eval_receipt.json.
    """
    a, b = match_size(a, b)
    _, smap = ssim(a, b, channel_axis=2, full=True)
    if mask is None:
        return float(smap.mean())
    m = _erode_for_window(mask.astype(bool))
    if m.sum() < MIN_EVAL_PIXELS:
        return float("nan")
    return float(smap[m].mean())


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """PSNR from squared error over ONLY the evaluated pixels."""
    a, b = match_size(a, b)
    se = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if mask is None:
        mse = float(se.mean())
    else:
        m = mask.astype(bool)
        if m.sum() < MIN_EVAL_PIXELS:
            return float("nan")
        mse = float(se[m].mean())
    if mse == 0:
        # Perfect match. json.dump would write bare Infinity, which is not strict JSON, so
        # callers serialise this as null. Returning inf here keeps the maths honest.
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def json_safe(obj):
    """Recursively replace inf/-inf/nan so json.dump emits strict JSON (null, not Infinity)."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def bit_exact_pct(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """% of evaluated pixels that are byte-identical across all channels. The honest
    headline for 'we did not touch this': paste-back should be ~100, chaining will not be."""
    a, b = match_size(a, b)
    eq = (a == b).all(axis=2)
    if mask is None:
        return float(100.0 * eq.mean())
    m = mask.astype(bool)
    if m.sum() == 0:
        return float("nan")
    return float(100.0 * eq[m].mean())


def preservation(ref: np.ndarray, test: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Every preservation number for one comparison. mask: 1 = evaluate here."""
    n = int(mask.astype(bool).sum()) if mask is not None else int(ref.shape[0] * ref.shape[1])
    return {
        "ssim": masked_ssim(ref, test, mask),
        "psnr": masked_psnr(ref, test, mask),
        "bit_exact_pct": bit_exact_pct(ref, test, mask),
        "eval_pixels": n,
    }


def template_match_inliers(ref_logo: np.ndarray, test_region: np.ndarray) -> dict:
    """ORB + RANSAC homography inliers. Works on any mark (logo, embroidery, embossing)
    where OCR fails; more inliers = the real mark is present."""
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(to_gray(ref_logo), None)
    k2, d2 = orb.detectAndCompute(to_gray(test_region), None)
    if d1 is None or d2 is None:
        return {"inliers": 0, "matches": 0}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 8:
        return {"inliers": 0, "matches": len(matches)}
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return {"inliers": int(mask.sum()) if mask is not None else 0, "matches": len(matches)}


def color_delta_e(ref: np.ndarray, test: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean CIEDE2000 color difference (lower = truer color)."""
    ref, test = match_size(ref, test)
    de = deltaE_ciede2000(rgb2lab(ref / 255), rgb2lab(test / 255))
    if mask is not None:
        m = mask.astype(bool)
        return float(de[m].mean()) if m.sum() else float("nan")
    return float(de.mean())
