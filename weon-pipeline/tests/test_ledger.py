"""Ledger invariant tests. No API, no cost.

Every test forces the provider offline and fakes the editor. `clients.py` calls load_dotenv()
at import, so without this a test run would fire real, paid API calls.

The point of the fakes: prove the preservation guarantee is STRUCTURAL (pixels are copied
forward) rather than a property of a well-behaved model.
"""
import numpy as np
import pytest

from pipeline import clients, config, ledger, metrics


BOX = (60, 60, 100, 100)
SHAPE = (160, 160, 3)


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, SHAPE, dtype=np.uint8)


@pytest.fixture
def live(monkeypatch):
    """Force the non-dry-run code path WITHOUT touching a real provider."""
    monkeypatch.setattr(config, "active_provider", lambda: "openrouter")


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(config, "active_provider", lambda: "dry-run")


def _vandal(monkeypatch, fill=0):
    """Worst case: the model corrupts every pixel of the crop it is handed."""
    monkeypatch.setattr(clients, "edit", lambda k, p, refs: np.full_like(refs[0], fill))


def _polite(monkeypatch, fill=255, box=BOX):
    """A well-behaved model: changes only the requested region inside the crop."""
    x0, y0, _, _ = ledger.context_crop(SHAPE, box)
    bx0, by0, bx1, by1 = box

    def fake(k, p, refs):
        out = refs[0].copy()
        out[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = fill
        return out
    monkeypatch.setattr(clients, "edit", fake)


# --- the core guarantee -----------------------------------------------------------------

def test_composite_preserves_outside_mask_even_if_model_returns_garbage(monkeypatch, live):
    """THE ledger claim, isolated from the gate: with the gate forced open, a model that
    destroys its entire crop still cannot alter a single pixel outside the declared mask,
    because those pixels are copied forward, not regenerated."""
    _vandal(monkeypatch)
    orig = _img(1)
    L = ledger.Ledger(orig, editor="gpt-image-2",
                      thresholds=ledger.Thresholds(min_context_ssim=-1.0))
    t = L.apply("obliterate the target", BOX)

    assert t.status == "accepted"          # gate deliberately disabled for this test
    outside = 1 - ledger.box_mask(SHAPE, BOX)
    assert metrics.bit_exact_pct(orig, L.canonical, outside) == 100.0


def test_naive_chain_does_not_hold_that_property(monkeypatch, live):
    """Control: same destructive model, chained normally, damages everything."""
    _vandal(monkeypatch)
    orig = _img(2)
    _, frames = ledger.run_naive_chain(orig, [("obliterate", BOX)], editor="gpt-image-2")
    outside = 1 - ledger.box_mask(SHAPE, BOX)
    assert metrics.bit_exact_pct(orig, frames[-1], outside) < 100.0


def test_accepted_turn_changes_only_the_intended_region(monkeypatch, live):
    _polite(monkeypatch)
    orig = _img(7)
    L = ledger.Ledger(orig, editor="gpt-image-2")
    t = L.apply("fill target white", BOX)

    assert t.status == "accepted"
    m = ledger.box_mask(SHAPE, BOX).astype(bool)
    assert (L.canonical[m] == 255).all()                                # intent applied
    assert np.array_equal(L.canonical[~m], np.asarray(L.original)[~m])  # rest untouched


# --- the gate must change behaviour -----------------------------------------------------

def test_gate_rejects_when_protected_context_is_damaged(monkeypatch, live):
    """A model that wrecks the surrounding protected content must be rejected, not accepted."""
    _vandal(monkeypatch)
    orig = _img(3)
    L = ledger.Ledger(orig, editor="gpt-image-2")   # default thresholds
    t = L.apply("obliterate the target", BOX)
    assert t.status == "rejected"
    assert "damaged protected content" in t.reason


def test_gate_rejects_a_no_op_edit(monkeypatch, live):
    """If the model changed nothing, the edit failed. Silence is not success."""
    monkeypatch.setattr(clients, "edit", lambda k, p, refs: refs[0].copy())
    orig = _img(4)
    L = ledger.Ledger(orig, editor="gpt-image-2")
    t = L.apply("make the mug red", BOX)
    assert t.status == "rejected"
    assert "edit did nothing" in t.reason


def test_rejected_turn_leaves_canonical_untouched(monkeypatch, live):
    _vandal(monkeypatch)
    orig = _img(5)
    L = ledger.Ledger(orig, editor="gpt-image-2")
    before = L.canonical.copy()
    L.apply("obliterate", BOX)
    assert np.array_equal(L.canonical, before)


# --- honesty ----------------------------------------------------------------------------

def test_original_is_immutable(monkeypatch, live):
    _polite(monkeypatch)
    orig = _img(6)
    snapshot = orig.copy()
    L = ledger.Ledger(orig, editor="gpt-image-2")
    L.apply("change it", BOX)
    assert np.array_equal(np.asarray(L.original), snapshot)


def test_dry_run_reports_na_not_fabricated_perfect_scores(offline):
    """A dry run returns the input unchanged. Reporting SSIM 1.000 for that would be a
    fabricated result — the exact failure mode called out in the audit."""
    orig = _img(8)
    L = ledger.Ledger(orig, editor="gpt-image-2")
    t = L.apply("anything", BOX)
    assert t.status == "dry_run"
    assert "outside_mask_vs_prev" not in t.metrics     # no invented preservation number
    assert t.cost_usd == 0.0
