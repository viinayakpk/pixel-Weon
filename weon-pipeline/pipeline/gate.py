"""Postmortem Gate v2 prototype: three independent conditions.

    commit = preservation_pass AND instruction_success AND boundary_quality_pass

The live Task 4 run used Gate v1, whose real condition was pixel change in the target, and
treated "a lot of pixels changed" as "the instruction succeeded". Turn 4 is the counter-example
to that inference: a large tan rectangle scored target_change=26.0 and passed, although Gate v1
could not verify the requested shape or material. Its semantic correctness remains unresolved.

Design rule: a check that cannot be performed returns UNKNOWN, never PASS. An unknown
instruction_success blocks an automatic commit and routes to review. Silence is not success.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum

import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

from . import metrics, zones


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"      # no trustworthy specialist exists for this predicate


@dataclass
class Check:
    verdict: Verdict
    detail: str
    value: float | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


# --- 1. preservation ---------------------------------------------------------------------

def check_preservation(prev: np.ndarray, result: np.ndarray, support: np.ndarray) -> Check:
    """Outside the declared write support, nothing may change. This is the one condition the
    architecture can guarantee, so it is checked as an assertion rather than a threshold."""
    pct = metrics.bit_exact_pct(prev, result, 1 - (support > 0).astype(np.uint8))
    if pct == 100.0:
        return Check(Verdict.PASS, "exterior byte-identical", pct)
    return Check(Verdict.FAIL, f"exterior changed ({pct:.4f}% byte-identical)", pct)


# --- 2. instruction success --------------------------------------------------------------

def _mean_lab(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    return rgb2lab(img / 255.0)[m].mean(axis=0)


def check_colour_predicate(prev: np.ndarray, result: np.ndarray, support: np.ndarray,
                           target_rgb: tuple[int, int, int], max_delta_e: float = 25.0) -> Check:
    """Did the target region actually become the requested colour?

    A transparent, checkable predicate for colour instructions ("make it black", "make it tan").
    It verifies the OUTCOME, not merely that pixels moved. It does NOT verify material
    ("canvas" vs "leather") — that is a separate predicate we do not have.
    """
    m = support.astype(bool)
    if m.sum() < 16:
        return Check(Verdict.UNKNOWN, "support too small to measure colour")
    got = _mean_lab(result, support)
    want = rgb2lab(np.array([[list(target_rgb)]], dtype=np.float64) / 255.0)[0, 0]
    de = float(deltaE_ciede2000(want.reshape(1, 1, 3), got.reshape(1, 1, 3))[0, 0])
    before = float(deltaE_ciede2000(
        want.reshape(1, 1, 3), _mean_lab(prev, support).reshape(1, 1, 3))[0, 0])
    if de > max_delta_e:
        return Check(Verdict.FAIL, f"target colour not reached (dE={de:.1f} > {max_delta_e})", de)
    if de >= before:
        return Check(Verdict.FAIL, f"no closer to target than before (dE {before:.1f} -> {de:.1f})", de)
    return Check(Verdict.PASS, f"reached target colour (dE {before:.1f} -> {de:.1f})", de)


def check_material_predicate(material: str) -> Check:
    """Material ("canvas", "leather", "ribbed knit") has no specialist in this pipeline.

    Returning UNKNOWN is the point: nothing in the live gate could verify material. A calibrated
    domain specialist or human review would be required; this prototype does not provide one.
    """
    return Check(Verdict.UNKNOWN, f"no specialist for material predicate '{material}'")


def check_pixels_moved(prev: np.ndarray, result: np.ndarray, support: np.ndarray,
                       min_change: float = 6.0) -> Check:
    """Necessary but NOT sufficient. Kept only as a liveness check: if nothing moved, the edit
    definitely failed. If something moved, that alone says nothing about correctness."""
    m = support.astype(bool)
    if m.sum() == 0:
        return Check(Verdict.UNKNOWN, "empty support")
    d = float(np.abs(result[m].astype(float) - prev[m].astype(float)).mean())
    if d < min_change:
        return Check(Verdict.FAIL, f"nothing changed in target (|d|={d:.1f})", d)
    return Check(Verdict.UNKNOWN, f"pixels moved (|d|={d:.1f}) — liveness only, not success", d)


# --- 3. boundary quality -----------------------------------------------------------------

def check_boundary(result: np.ndarray, prev: np.ndarray, support: np.ndarray,
                   max_ratio: float = 1.6) -> Check:
    """Is the commit boundary a step edge relative to the same ring before the edit?

    Compared against the SAME ring in the previous image, because the absolute gradient of a
    ring is dominated by whatever real structure it happens to cross (measured: a polygon that
    follows a real seam scores 'worse' than a box over plain fabric). A ratio against the prior
    state controls for that; it is still a diagnostic, not a perceptual metric.
    """
    after = zones.boundary_gradient(result, support)
    before = zones.boundary_gradient(prev, support)
    if not np.isfinite(after) or not np.isfinite(before) or before <= 1e-6:
        return Check(Verdict.UNKNOWN, "boundary ring not measurable")
    ratio = after / before
    if ratio > max_ratio:
        return Check(Verdict.FAIL, f"seam introduced (ring gradient x{ratio:.2f})", ratio)
    return Check(Verdict.PASS, f"no step edge introduced (ring gradient x{ratio:.2f})", ratio)


# --- decision ----------------------------------------------------------------------------

@dataclass
class Decision:
    commit: bool
    status: str                      # committed | rejected | review
    checks: dict

    def to_json(self) -> dict:
        return {"commit": self.commit, "status": self.status,
                "checks": {k: v.to_json() for k, v in self.checks.items()}}


def decide(checks: dict[str, Check]) -> Decision:
    """commit requires every condition to PASS.

    Any FAIL -> rejected. No FAIL but any UNKNOWN -> review (a human decides). UNKNOWN never
    becomes an automatic commit, so an unverifiable claim cannot ship as if verified.
    """
    verdicts = [c.verdict for c in checks.values()]
    if Verdict.FAIL in verdicts:
        return Decision(False, "rejected", checks)
    if Verdict.UNKNOWN in verdicts:
        return Decision(False, "review", checks)
    return Decision(True, "committed", checks)
