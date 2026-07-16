"""Gate tests, including the real turn-4 regression.

The old gate passed turn 4 because it treated large pixel change as instruction success, without
verifying the requested shape or material. These tests pin the narrower new behaviour: an
unverifiable predicate must block an automatic commit.
"""
import numpy as np
import pytest

from pipeline import gate, ledger, zones
from pipeline.gate import Verdict


BOX = (60, 60, 100, 100)
SHAPE = (160, 160, 3)


def _img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, SHAPE, dtype=np.uint8)


def _support():
    return ledger.box_mask(SHAPE, BOX)


def test_preservation_passes_when_exterior_untouched():
    prev = _img(1)
    res = prev.copy()
    res[_support().astype(bool)] = 0          # only the support changed
    assert gate.check_preservation(prev, res, _support()).verdict is Verdict.PASS


def test_preservation_fails_when_exterior_moves():
    prev = _img(2)
    res = prev.copy()
    res[0, 0] = (res[0, 0] + 1) % 255         # one pixel outside the support
    assert gate.check_preservation(prev, res, _support()).verdict is Verdict.FAIL


def test_pixels_moved_is_liveness_only_never_success():
    """The core regression: a big pixel change must NOT be reported as instruction success."""
    prev = _img(3)
    res = prev.copy()
    res[_support().astype(bool)] = 255
    c = gate.check_pixels_moved(prev, res, _support())
    assert c.verdict is Verdict.UNKNOWN
    assert "liveness only" in c.detail


def test_pixels_moved_fails_on_no_op():
    prev = _img(4)
    assert gate.check_pixels_moved(prev, prev.copy(), _support()).verdict is Verdict.FAIL


def test_colour_predicate_passes_when_target_colour_reached():
    prev = np.full(SHAPE, 60, np.uint8)              # dark grey
    res = prev.copy()
    res[_support().astype(bool)] = (10, 10, 10)      # -> black
    assert gate.check_colour_predicate(prev, res, _support(), (0, 0, 0)).verdict is Verdict.PASS


def test_colour_predicate_fails_when_wrong_colour():
    prev = np.full(SHAPE, 60, np.uint8)
    res = prev.copy()
    res[_support().astype(bool)] = (200, 30, 30)     # red, asked for black
    assert gate.check_colour_predicate(prev, res, _support(), (0, 0, 0)).verdict is Verdict.FAIL


def test_material_predicate_is_unknown_not_pass():
    """'tan canvas' vs 'tan leather' — we have no specialist, so we must not claim success."""
    assert gate.check_material_predicate("canvas").verdict is Verdict.UNKNOWN


def test_unknown_blocks_automatic_commit():
    d = gate.decide({
        "preservation": gate.Check(Verdict.PASS, ""),
        "instruction": gate.Check(Verdict.UNKNOWN, "no specialist"),
        "boundary": gate.Check(Verdict.PASS, ""),
    })
    assert d.commit is False and d.status == "review"


def test_fail_beats_unknown():
    d = gate.decide({
        "preservation": gate.Check(Verdict.FAIL, ""),
        "instruction": gate.Check(Verdict.UNKNOWN, ""),
    })
    assert d.status == "rejected"


def test_all_pass_commits():
    d = gate.decide({
        "preservation": gate.Check(Verdict.PASS, ""),
        "instruction": gate.Check(Verdict.PASS, ""),
        "boundary": gate.Check(Verdict.PASS, ""),
    })
    assert d.commit is True and d.status == "committed"


# --- the real regression -----------------------------------------------------------------

def test_turn4_tan_rectangle_does_not_auto_commit():
    """Regression on real data: the shipped turn-4 result must not sail through.

    The old gate passed it (target_change=26.0, context_ssim=0.636). The new gate must refuse
    to auto-commit, because 'tan canvas' is unverifiable here.
    """
    import os
    from PIL import Image
    from cases.task4_jacket import EDITS

    out = "outputs/task4"
    if not os.path.exists(f"{out}/ledger_step4.png"):
        pytest.skip("task4 run artifacts not present")

    prev = np.asarray(Image.open(f"{out}/ledger_step3.png").convert("RGB"))
    res = np.asarray(Image.open(f"{out}/ledger_step4.png").convert("RGB"))
    support = ledger.box_mask(prev.shape, EDITS[3][1])

    checks = {
        "preservation": gate.check_preservation(prev, res, support),
        "instruction_colour": gate.check_colour_predicate(prev, res, support, (193, 154, 107)),
        "instruction_material": gate.check_material_predicate("canvas"),
        "boundary": gate.check_boundary(res, prev, support),
    }
    d = gate.decide(checks)
    assert d.commit is False, "turn 4 must not auto-commit — material is unverifiable"
    assert d.status in ("review", "rejected")
    assert checks["preservation"].verdict is Verdict.PASS   # preservation genuinely held
