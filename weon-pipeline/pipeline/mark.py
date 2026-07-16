"""The protected mark: extraction, representation and physically-sane compositing.

Central claim being tested in Task 1:

    A printed/foil logo is primarily an ALBEDO change. Tonal embroidery is primarily
    MICRO-GEOMETRY (relief and surface normals). Representing both as a rectangular RGB patch
    is a category error.

Measured support for that claim (experiments/material_probe.py):
    ARIGATO      gold on brown suede   L* std 20.93 -> top-hat gives 2.1% coverage, 7 letters
    BeyondNordic green on green thread L* std 10.43 -> top-hat gives 10.6%, 84 fragments
The same extractor recovers a clean, automatically selectable mark from foil (a clean component
area gap 72 -> 12 leaves exactly the 7 letters) and has no such gap on tonal embroidery (84
components, no principled cut). It does NOT 'disintegrate' there — the alpha stays legible,
because relief itself creates luminance contrast through shadow, which is what a top-hat
responds to. What differs is what the extracted RGB MEANS; see experiments/relief_probe.py.

What this module fixes from the previous version, which the audit caught:
  * it pasted a rectangle of suede rather than an isolated mark  -> alpha + component filtering
  * it located the target by matching the model's own corrupted logo -> targets are DECLARED
  * it warped RGB and alpha separately then blurred alpha -> premultiplied alpha, so colour
    cannot bleed out of the mark and produce a suede halo
  * it multiplied light in sRGB -> compositing and relighting happen in LINEAR light

Photometric model: I = A x S (intrinsic images). We keep the mark's albedo A and swap the
shading S from the source packshot to the target scene, instead of pasting source pixels that
still carry the packshot's lighting.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.color import rgb2lab


# --- colour management -------------------------------------------------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> linear light. Blending and relighting in sRGB is multiplying gamma-encoded
    values, which darkens/haloes edges; do it in linear."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)


# --- extraction --------------------------------------------------------------------------

def extract_mark_alpha(crop_rgb: np.ndarray, stroke: int = 15) -> np.ndarray:
    """Isolate a bright mark from its substrate -> RGBA (alpha = the mark).

    Morphological top-hat on L*: a mark is small bright strokes on a slowly varying background.
    Rejected alternative (measured): thresholding L*/b* directly selects lit substrate and
    returns ~50% coverage — that is how the previous version ended up pasting fabric.
    """
    lab = rgb2lab(crop_rgb / 255.0)
    L8 = (np.clip(lab[..., 0], 0, 100) / 100 * 255).astype(np.uint8)
    top = cv2.morphologyEx(L8, cv2.MORPH_TOPHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke, stroke)))
    _, alpha = cv2.threshold(top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return np.dstack([crop_rgb, alpha]).astype(np.uint8)


def keep_components(rgba: np.ndarray, min_area: int = 20,
                    expect: int | None = None) -> tuple[np.ndarray, dict]:
    """Drop speckle, keep the components that are actually the mark.

    Measured on ARIGATO: 18 components, of which exactly 7 have area >= 20 and correspond to
    the 7 letters (areas 141,139,128,128,123,79,72 then a gap to 12,8,7,6,4...). The gap is
    what makes this safe; `stats` reports it so the caller can see whether a gap existed.
    """
    a = (rgba[..., 3] > 0).astype(np.uint8)
    n, labels, st, _ = cv2.connectedComponentsWithStats(a, 8)
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, ar in enumerate(areas) if ar >= min_area]
    out = rgba.copy()
    mask = np.isin(labels, keep)
    out[..., 3] = np.where(mask, rgba[..., 3], 0)
    info = {
        "components_total": int(n - 1),
        "components_kept": len(keep),
        "kept_areas": sorted((int(x) for x in areas if x >= min_area), reverse=True),
        "dropped_areas": sorted((int(x) for x in areas if x < min_area), reverse=True),
        "expected": expect,
        "matches_expected": (expect is None or len(keep) == expect),
    }
    return out, info


def tight_crop(rgba: np.ndarray, pad: int = 2) -> np.ndarray:
    """Crop to the mark's alpha bounding box so the asset is the mark, not a region."""
    ys, xs = np.where(rgba[..., 3] > 0)
    if len(xs) == 0:
        return rgba
    x0, x1 = max(0, xs.min() - pad), min(rgba.shape[1], xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(rgba.shape[0], ys.max() + 1 + pad)
    return rgba[y0:y1, x0:x1]


def coverage_pct(rgba: np.ndarray) -> float:
    return float(100.0 * (rgba[..., 3] > 0).mean())


# --- placement ---------------------------------------------------------------------------

def box_to_quad(box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def graft(target_rgb: np.ndarray, asset_rgba: np.ndarray, quad: np.ndarray,
          supersample: int = 4, relight: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Warp the mark onto a DECLARED quad and composite it in linear light.

    Pipeline (order matters):
      1. premultiply alpha in linear light  — so interpolation can never drag substrate colour
         out past the mark's edge (the suede-halo bug)
      2. warp premultiplied RGBA at `supersample`x, then downsample ONCE
      3. optionally divide out the source's low-frequency shading and multiply in the target's
         (I = A x S: keep the mark's albedo, adopt the scene's light)
      4. composite over the target, unpremultiply, back to sRGB

    Returns (composited_rgb, alpha_float).
    """
    h, w = asset_rgba.shape[:2]
    rgb_lin = srgb_to_linear(asset_rgba[..., :3].astype(np.float32) / 255.0)
    a = (asset_rgba[..., 3].astype(np.float32) / 255.0)[..., None]

    if relight:
        # remove the packshot's own lighting from the mark: divide by its low-frequency shading
        lum = cv2.cvtColor(asset_rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        s_src = srgb_to_linear(cv2.GaussianBlur(lum, (0, 0), sigmaX=max(2, min(h, w) / 8)))
        rgb_lin = rgb_lin / np.clip(s_src, 0.05, None)[..., None]

    pm = np.dstack([rgb_lin * a, a]).astype(np.float32)      # premultiplied, linear

    th, tw = target_rgb.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    H = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    S = np.diag([supersample, supersample, 1.0]).astype(np.float32)
    warped = cv2.warpPerspective(pm, S @ H, (tw * supersample, th * supersample),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    warped = cv2.resize(warped, (tw, th), interpolation=cv2.INTER_AREA)   # single downsample
    wr, wa = warped[..., :3], np.clip(warped[..., 3:4], 0.0, 1.0)

    tgt_lin = srgb_to_linear(target_rgb.astype(np.float32) / 255.0)
    if relight:
        # apply the TARGET's low-frequency shading to the mark
        tlum = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        s_tgt = srgb_to_linear(cv2.GaussianBlur(tlum, (0, 0), sigmaX=25))[..., None]
        wr = wr * np.clip(s_tgt, 0.02, None)

    out_lin = wr + tgt_lin * (1.0 - wa)          # premultiplied 'over'
    out = (linear_to_srgb(np.clip(out_lin, 0, 1)) * 255).round().astype(np.uint8)
    return out, wa[..., 0]


# --- layer decomposition -----------------------------------------------------------------

def estimate_substrate(crop_rgb: np.ndarray, alpha: np.ndarray, radius: int = 12,
                       dilate: int = 3) -> np.ndarray:
    """Estimate the fabric under the mark by inpainting the mark away.

    The mask must be DILATED first. A threshold alpha stops at the mark's core and leaves its
    anti-aliased rim behind; inpainting only the core then leaves a ghost of the mark in the
    'substrate', which both flatters the recomposition test and is simply wrong. Measured: at
    radius=6 with no dilation, ARIGATO was still legible in the estimated substrate.
    """
    m = cv2.dilate((alpha > 0).astype(np.uint8),
                   np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8))
    return cv2.inpaint(crop_rgb, m, radius, cv2.INPAINT_TELEA)


def substrate_residual(crop_rgb: np.ndarray, substrate: np.ndarray,
                       alpha: np.ndarray) -> float:
    """How much mark structure survives in the estimated substrate?

    Local contrast (top-hat energy) inside the mark region, substrate vs source. ~0 means the
    mark was removed; near 1 means it is still there and the decomposition is fiction.
    """
    m = (alpha > 0)
    if m.sum() == 0:
        return float("nan")
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    def energy(x):
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        return float(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)[m].mean())
    e_src = energy(crop_rgb)
    return float(energy(substrate) / e_src) if e_src > 1e-6 else float("nan")


def recomposition_error(crop_rgb: np.ndarray, rgba: np.ndarray) -> dict:
    """Round-trip test: does substrate + extracted-asset actually reconstruct the source?

    This must push the mark through the REAL representation, otherwise it is a tautology. An
    earlier version blended `crop_rgb` itself over `inpaint(crop_rgb)`; since inpainting only
    alters pixels inside the mask, that returned the input everywhere and scored a perfect
    0.00 error while testing nothing.

    The honest round trip:
        crop -> alpha -> tight-cropped asset -> warp back onto its own quad
             -> composite over the inpainted substrate -> compare with crop

    Residual error is therefore real: alpha quantisation, warp interpolation, premultiplication
    and the linear/sRGB round trip. Reported inside the mark (where fidelity matters) and over
    the whole crop.
    """
    alpha = rgba[..., 3]
    sub = estimate_substrate(crop_rgb, alpha)

    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return {"mean_abs_error": float("nan"), "mark_mean_abs_error": float("nan"),
                "p95_abs_error": float("nan"), "coverage_pct": 0.0,
                "substrate": sub, "recomposed": sub}
    pad = 2
    x0, x1 = max(0, xs.min() - pad), min(rgba.shape[1], xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(rgba.shape[0], ys.max() + 1 + pad)
    asset = rgba[y0:y1, x0:x1]                       # the asset as it would really be stored
    quad = box_to_quad((x0, y0, x1, y1))             # placed back exactly where it came from

    rec, _ = graft(sub, asset, quad, supersample=4, relight=False)

    err = np.abs(rec.astype(np.float64) - crop_rgb.astype(np.float64))
    m = (alpha > 0)
    # normalise the in-mark error by the mark's own contrast, otherwise a high-contrast foil
    # mark is penalised simply for being high-contrast and the number says nothing about the
    # quality of the decomposition (measured: raw in-mark error was higher for ARIGATO (21.3)
    # than for the tonal mark (10.6) purely because gold-on-brown has a bigger dynamic range).
    contrast = float(crop_rgb[m].astype(np.float64).std()) if m.sum() else float("nan")
    mark_err = float(err[m].mean()) if m.sum() else float("nan")
    return {
        "mean_abs_error": float(err.mean()),
        "mark_mean_abs_error": mark_err,
        "mark_contrast_std": contrast,
        "mark_err_over_contrast": mark_err / contrast if contrast > 1e-6 else float("nan"),
        "p95_abs_error": float(np.percentile(err, 95)),
        "coverage_pct": float(100.0 * m.mean()),
        "substrate_residual": substrate_residual(crop_rgb, sub, alpha),
        "substrate": sub,
        "recomposed": rec,
    }
