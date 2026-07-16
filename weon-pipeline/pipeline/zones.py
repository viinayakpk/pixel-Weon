"""Three-zone commit mask.

Why this exists. The ledger proved preservation is structural: pixels outside the declared
mask are copied forward and cannot change. But turn 4 of the Task 4 run produced a visible
rectangular tan artifact, which exposed a second problem: with a hard binary mask, the mask
itself becomes the visible shape of the edit. The commit region acted as an accidental shape
prior.

The spatial contract therefore needs zones, not one binary box:

    protected exterior   pixels stay byte-identical (never written)
    transition collar    controlled, inward-only blend (candidate -> previous)
    semantic core        the candidate's edit, committed fully

Inward-only is the important constraint: the collar is carved OUT of the declared write
support, so no pixel outside the support can change. Preservation stays exact by construction
while the boundary stops being a hard rectangle.

Separately measured caveat (see experiments/three_zone_ablation.py): a collar improves the
BOUNDARY. It cannot fix a write region that does not correspond to the named object. On turn 4
the declared box was much larger than the actual welt pocket, so the artifact was primarily a
grounding failure. Zones and grounding are different problems.
"""
from __future__ import annotations

import cv2
import numpy as np


def polygon_mask(shape: tuple[int, ...], poly: np.ndarray) -> np.ndarray:
    """Binary mask (1 inside) from a declared polygon. Declared from the ORIGINAL image,
    never from a difference against the output."""
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    return m


def three_zone_alpha(support: np.ndarray, collar_px: int) -> np.ndarray:
    """Float alpha in [0,1] over the declared write `support`.

    alpha == 0 outside the support (protected exterior, byte-identical)
    alpha ramps 0 -> 1 across `collar_px` INSIDE the support edge (transition collar)
    alpha == 1 in the eroded interior (semantic core)

    Implemented with a distance transform so the ramp follows the polygon's real shape rather
    than a bounding box.
    """
    s = (support > 0).astype(np.uint8)
    if collar_px <= 0:
        return s.astype(np.float32)
    # distance from each interior pixel to the nearest exterior pixel
    dist = cv2.distanceTransform(s, cv2.DIST_L2, 5)
    a = np.clip(dist / float(collar_px), 0.0, 1.0)
    return (a * s).astype(np.float32)


def commit(prev: np.ndarray, candidate: np.ndarray, support: np.ndarray,
           collar_px: int) -> np.ndarray:
    """Blend candidate into prev through the three-zone alpha. Outside `support`, prev is
    returned byte-for-byte."""
    a = three_zone_alpha(support, collar_px)[..., None]
    out = prev.astype(np.float32) * (1 - a) + candidate.astype(np.float32) * a
    out = np.rint(out).astype(np.uint8)
    outside = (support == 0)
    out[outside] = prev[outside]          # exact, not merely approximately equal
    return out


# --- diagnostics -------------------------------------------------------------------------

def boundary_gradient(img: np.ndarray, support: np.ndarray, band_px: int = 3) -> float:
    """Mean gradient magnitude in a thin ring straddling the support edge.

    This is a SEAM DIAGNOSTIC, not a perceptual metric: a hard paste puts a step edge exactly
    on the boundary, which shows up as a high mean gradient there. Lower is smoother. It says
    nothing about whether the edit is correct or natural-looking overall.
    """
    s = (support > 0).astype(np.uint8)
    k = np.ones((band_px * 2 + 1, band_px * 2 + 1), np.uint8)
    ring = (cv2.dilate(s, k) - cv2.erode(s, k)).astype(bool)
    if ring.sum() == 0:
        return float("nan")
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2)[ring].mean())


def retained_delta_pct(prev: np.ndarray, candidate: np.ndarray, result: np.ndarray,
                       support: np.ndarray) -> float:
    """How much of the candidate's pixel delta survived the blend, inside the support.

    100% = the full candidate delta was retained. This measures changed-pixel magnitude, not
    semantic edit success. A collar necessarily trades some delta away at the rim.
    """
    m = (support > 0)
    if m.sum() == 0:
        return float("nan")
    want = np.abs(candidate[m].astype(np.float64) - prev[m].astype(np.float64)).sum()
    got = np.abs(result[m].astype(np.float64) - prev[m].astype(np.float64)).sum()
    if want == 0:
        return float("nan")
    return float(100.0 * got / want)
