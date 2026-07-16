"""Regressions for the two specialists' decision logic.

Neither experiment can be rerun cheaply — both cost money and both overwrite canonical evidence —
so the pure functions that turn a raw model reply into a verdict are the only part that can be
pinned. They are also the part that decides every headline number, which makes them exactly the
part worth pinning.

The comparison logic is where an over-eager fold would silently manufacture a pass: if norm() folded
a little too hard, `A DATT MARCH` would equal `A DAY'S MARCH` and the specialist's most important
control would invert without anything visibly breaking.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.grounding_retest import iou_and_overlap, to_px      # noqa: E402
from experiments.spelling_specialist import norm, verdict            # noqa: E402
from pipeline.gate import Verdict                                    # noqa: E402


# --- norm: fold presentation, never fold identity -------------------------------------------

def test_norm_folds_case_apostrophe_and_spacing():
    """These are the same brand string typed three ways, not three strings."""
    for s in ("A DAY'S MARCH", "a days march", "  A Day’s   March ", "ADAYSMARCH"):
        assert norm(s) == "ADAYSMARCH"


def test_norm_does_not_fold_a_real_misspelling():
    """The whole c2 control rests on this staying unequal."""
    assert norm("A DATT MARCH") != norm("A DAY'S MARCH")


def test_norm_does_not_fold_a_transposition():
    assert norm("A DAYS MRACH") != norm("A DAY'S MARCH")


# --- verdict: the three outcomes, and the one that must not collapse ------------------------

def test_verdict_pass_on_exact_match_modulo_presentation():
    v, _ = verdict("a days march", "A DAY'S MARCH")
    assert v is Verdict.PASS


def test_verdict_fail_on_the_real_nano_corruption():
    """The measured turn-5 transcription. If this ever returns PASS the headline is wrong."""
    v, why = verdict("A DATT MARCH", "A DAY'S MARCH")
    assert v is Verdict.FAIL
    assert "DATT" in why


def test_verdict_unknown_on_abstention():
    """Abstention must not be scored as a failure: 'cannot see' and 'is wrong' are different
    facts, and conflating them is what routes a blind check to auto-commit."""
    for raw in ("UNREADABLE", "unreadable", " Unreadable "):
        v, _ = verdict(raw, "A DAY'S MARCH")
        assert v is Verdict.UNKNOWN


def test_verdict_unknown_when_specialist_unavailable():
    """A missing API key must never read as a clean pass."""
    v, _ = verdict(None, "A DAY'S MARCH")
    assert v is Verdict.UNKNOWN


def test_verdict_unknown_on_empty_reply():
    v, _ = verdict("   ", "A DAY'S MARCH")
    assert v is Verdict.UNKNOWN


def test_verdict_pass_is_case_of_string_not_geometry():
    """c4, in one line: the correct string is not the correct mark. This asserts the specialist's
    documented blind spot, so that removing the geometry check breaks a test."""
    v, _ = verdict("ARIGATO", "ARIGATO")
    assert v is Verdict.PASS


# --- grounding: the '0 px overlap' headline depends entirely on this ------------------------

def test_overlap_is_zero_for_disjoint_boxes():
    iou, ov = iou_and_overlap((0, 0, 10, 10), (100, 100, 110, 110))
    assert (iou, ov) == (0.0, 0)


def test_overlap_counts_pixels_for_touching_boxes():
    """Edge-adjacent boxes share a border but no area."""
    _, ov = iou_and_overlap((0, 0, 10, 10), (10, 0, 20, 10))
    assert ov == 0


def test_iou_is_one_for_identical_boxes():
    iou, ov = iou_and_overlap((5, 5, 15, 15), (5, 5, 15, 15))
    assert iou == 1.0 and ov == 100


def test_iou_half_overlap():
    """Two 10x10 boxes sharing exactly half: inter 50, union 150."""
    iou, ov = iou_and_overlap((0, 0, 10, 10), (5, 0, 15, 10))
    assert ov == 50
    assert abs(iou - 50 / 150) < 1e-9


def test_to_px_maps_normalised_ymin_xmin_ymax_xmax():
    """Gemini returns [ymin, xmin, ymax, xmax] on 0-1000. Transposing y/x here would silently
    relocate every returned pocket and invalidate the overlap result."""
    assert to_px([0, 0, 1000, 1000], (1125, 900, 3)) == (0, 0, 900, 1125)
    assert to_px([0, 0, 500, 500], (1000, 1000, 3)) == (0, 0, 500, 500)
    # a box in the lower-left of a portrait frame stays in the lower-left
    x0, y0, x1, y1 = to_px([800, 100, 900, 200], (1125, 900, 3))
    assert (x0, x1) == (90, 180) and (y0, y1) == (900, 1012)
