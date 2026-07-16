"""Regression tests for the evaluation layer.

The bug these exist to prevent: the old masked SSIM zeroed pixels outside the mask and
then averaged SSIM over the WHOLE image, so the identical black regions dominated and a
completely destroyed ROI scored ~0.99. Every downstream number inherited that inflation.

Run:  python -m pytest tests/ -q       (from weon-pipeline/)
"""
import numpy as np
import pytest

from pipeline import metrics


def _img(seed=0, h=128, w=128):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _roi(h=128, w=128):
    """A small central ROI — the region we claim to evaluate."""
    m = np.zeros((h, w), np.uint8)
    m[48:80, 48:80] = 1
    return m


def test_destroyed_roi_cannot_score_near_perfect():
    """THE test. Corrupt ONLY the ROI, leave everything else identical.
    A mask-aware metric must report near-zero SSIM for that ROI."""
    a = _img(1)
    m = _roi()
    b = a.copy()
    b[m.astype(bool)] = 0  # obliterate the ROI

    s = metrics.masked_ssim(a, b, m)
    assert s < 0.2, f"destroyed ROI scored {s:.4f} — metric is inflated"


def test_identical_roi_scores_high():
    a = _img(2)
    m = _roi()
    assert metrics.masked_ssim(a, a.copy(), m) > 0.99
    assert metrics.bit_exact_pct(a, a.copy(), m) == 100.0


def test_damage_outside_roi_does_not_affect_roi_score():
    """Preservation of a region must be independent of what happened elsewhere."""
    a = _img(3)
    m = _roi()
    b = a.copy()
    outside = ~m.astype(bool)
    b[outside] = 0  # destroy everything EXCEPT the ROI

    assert metrics.masked_ssim(a, b, m) > 0.99, "ROI score contaminated by outside damage"
    assert metrics.bit_exact_pct(a, b, m) == 100.0


def test_bit_exact_detects_subtle_drift():
    """Chaining degradation is often invisible to SSIM but never bit-exact."""
    a = _img(4)
    b = a.copy()
    b[..., 0] = np.clip(b[..., 0].astype(int) + 1, 0, 255)  # +1 on one channel
    # not exactly 0: pixels already at 255 clip and stay identical
    assert metrics.bit_exact_pct(a, b) < 1.0
    assert metrics.masked_ssim(a, b) > 0.98  # SSIM barely notices — hence bit-exact %


def test_masked_psnr_is_roi_only():
    a = _img(5)
    m = _roi()
    b = a.copy()
    b[~m.astype(bool)] = 0  # destroy outside only
    assert metrics.masked_psnr(a, b, m) == float("inf")  # ROI untouched


def test_too_small_mask_returns_nan_not_a_number():
    """Never invent a score from too few pixels."""
    a = _img(6)
    m = np.zeros((128, 128), np.uint8)
    m[0, 0] = 1
    assert np.isnan(metrics.masked_ssim(a, _img(7), m))


def test_preservation_reports_eval_pixel_count():
    a, m = _img(8), _roi()
    p = metrics.preservation(a, a.copy(), m)
    assert p["eval_pixels"] == int(m.sum())
    assert p["bit_exact_pct"] == 100.0
