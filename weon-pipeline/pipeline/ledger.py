"""Task 4 — the pixel edit ledger.

Claim under test: constraint-preserving image editing. A generative model can attempt
the requested change but is unreliable at preserving everything else. So we declare what may
change BEFORE generating, let the model change only that, and copy every other pixel forward.

Why previous edits survive: they are stored as *pixels* in a canonical accepted image, not as
text the model must re-interpret on every turn. That is the difference from prompt rebasing.

Three rules:
  1. `original` is immutable.
  2. `intended_mask` is declared before the edit. It is never derived from the output —
     deriving it from the output lets the model decide which of its own damage gets excluded
     from the preservation score, which is circular.
  3. Pixels outside the intended mask are copied from the previous canonical image, so they
     are byte-identical by construction, not by hope.

Not claimed: global relighting, background swaps, or edits whose true extent exceeds the
declared mask. The submitted experiment covers local edits only.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from . import clients, config, metrics


# --- geometry ---------------------------------------------------------------------------

def box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """1 inside the declared box. This is the *intent*, fixed before we generate."""
    m = np.zeros(shape[:2], np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 1
    return m


def context_crop(shape: tuple[int, int], box: tuple[int, int, int, int],
                 pad: float = 0.6) -> tuple[int, int, int, int]:
    """Padded crop around the target. The model needs surrounding context to make a
    coherent edit, but we only ever *accept* pixels from inside the intended mask."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))


# --- ledger -----------------------------------------------------------------------------

@dataclass
class Turn:
    idx: int
    instruction: str
    intended_box: tuple[int, int, int, int]
    crop: tuple[int, int, int, int]
    status: str = "pending"          # accepted | rejected | dry_run | error
    reason: str = ""
    metrics: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    candidate: np.ndarray | None = None   # kept so a REJECTED turn can be audited visually

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("candidate", None)          # image goes to disk, not into the metrics file
        d["intended_box"] = list(self.intended_box)
        d["crop"] = list(self.crop)
        return d


@dataclass
class Thresholds:
    """A gate only counts if it changes behaviour. These cause accept/reject."""
    min_target_change: float = 6.0   # mean |Δ| in ROI; below this the edit did nothing
    min_context_ssim: float = 0.60   # protected content around the ROI must survive


class Ledger:
    def __init__(self, original: np.ndarray, editor: str = "gpt-image-2",
                 thresholds: Thresholds | None = None, candidates: int = 1):
        self.original = original.copy()          # immutable reference
        self.original.flags.writeable = False
        self.canonical = original.copy()         # current accepted composite
        self.editor = editor
        self.th = thresholds or Thresholds()
        self.candidates = candidates
        self.turns: list[Turn] = []
        self.union_mask = np.zeros(original.shape[:2], np.uint8)
        self.frames: list[np.ndarray] = [original.copy()]

    # -- one turn ------------------------------------------------------------------------
    def apply(self, instruction: str, box: tuple[int, int, int, int]) -> Turn:
        mask = box_mask(self.original.shape, box)
        crop = context_crop(self.original.shape, box)
        turn = Turn(len(self.turns), instruction, box, crop)
        prev = self.canonical.copy()
        t0 = time.time()

        dry = config.active_provider() == "dry-run"
        best, best_score, reason = None, -1.0, ""

        for _ in range(max(1, self.candidates)):
            try:
                cand = self._edit_crop(prev, crop, instruction)
            except Exception as e:  # network/API failure is not a preservation result
                turn.status, turn.reason = "error", str(e)[:200]
                turn.latency_s = time.time() - t0
                self.turns.append(turn)
                return turn
            ok, score, why = self._verify(prev, cand, mask, crop, dry)
            if score > best_score:
                best, best_score, reason = cand, score, why
            if ok:
                break

        turn.latency_s = time.time() - t0
        turn.cost_usd = config.MODELS[self.editor].price_usd * max(1, self.candidates) if not dry else 0.0

        if dry:
            # dry-run returns the input unchanged: there is no edit to score. Report N/A
            # rather than SSIM 1.000, which would be a fabricated perfect result.
            turn.status, turn.reason = "dry_run", "no API key — nothing generated, metrics N/A"
            turn.metrics = {"note": "dry-run: not measurable"}
            self.turns.append(turn)
            self.frames.append(self.canonical.copy())
            return turn

        accepted, _, why = self._verify(prev, best, mask, crop, dry)
        turn.reason = why
        turn.candidate = best              # retained either way so rejections are auditable
        if accepted:
            self.canonical = self._composite(prev, best, mask)
            self.union_mask = np.maximum(self.union_mask, mask)
            turn.status = "accepted"
        else:
            turn.status = "rejected"       # canonical is left untouched — a real gate

        turn.metrics = self._measure(prev, self.canonical, mask)
        self.turns.append(turn)
        self.frames.append(self.canonical.copy())
        return turn

    # -- internals -----------------------------------------------------------------------
    def _edit_crop(self, base: np.ndarray, crop, instruction: str) -> np.ndarray:
        """Edit only the context crop, then place it back into a full-size candidate.
        The model never sees the rest of the frame, so it cannot degrade it. The crop is
        also small enough to avoid the client's 1536px downscale."""
        x0, y0, x1, y1 = crop
        patch = np.ascontiguousarray(base[y0:y1, x0:x1])
        out = clients.edit(self.editor, instruction, [patch])
        if out.shape[:2] != patch.shape[:2]:
            out = cv2.resize(out, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        cand = base.copy()
        cand[y0:y1, x0:x1] = out
        return cand

    def _composite(self, prev: np.ndarray, cand: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Accept ONLY the intended pixels. Everything else is copied forward, so it is
        byte-identical by construction. (A feather band would trade that bit-exactness for
        seam smoothing; we keep the hard mask so the claim stays literally true.)"""
        out = prev.copy()
        m = mask.astype(bool)
        out[m] = cand[m]
        return out

    def _verify(self, prev, cand, mask, crop, dry) -> tuple[bool, float, str]:
        """Two questions, both required: did the requested change happen, and did the
        protected content around it survive?"""
        if dry or cand is None:
            return False, 0.0, "dry-run"
        m = mask.astype(bool)
        target_change = float(np.abs(cand[m].astype(float) - prev[m].astype(float)).mean())

        x0, y0, x1, y1 = crop
        ctx = np.zeros(prev.shape[:2], np.uint8)
        ctx[y0:y1, x0:x1] = 1
        ctx[m] = 0                       # context = inside crop, outside the intended mask
        ctx_ssim = metrics.masked_ssim(prev, cand, ctx)

        ok_target = target_change >= self.th.min_target_change
        ok_ctx = np.isnan(ctx_ssim) or ctx_ssim >= self.th.min_context_ssim
        why = (f"target_change={target_change:.1f}"
               f"{'' if ok_target else ' <min (edit did nothing)'}; "
               f"context_ssim={ctx_ssim:.3f}"
               f"{'' if ok_ctx else ' <min (damaged protected content)'}")
        return (ok_target and ok_ctx), target_change, why

    def _measure(self, prev, current, mask) -> dict:
        """Measure in the three directions the brief cares about."""
        outside_now = 1 - mask.astype(np.uint8)
        outside_union = 1 - self.union_mask.astype(np.uint8)
        return {
            "target_roi_change": float(np.abs(
                current[mask.astype(bool)].astype(float) - prev[mask.astype(bool)].astype(float)
            ).mean()),
            "outside_mask_vs_prev": metrics.preservation(prev, current, outside_now),
            "outside_union_vs_original": metrics.preservation(
                np.asarray(self.original), current, outside_union),
        }


# --- baseline for comparison -------------------------------------------------------------

def run_naive_chain(original: np.ndarray, instrs: list[tuple[str, tuple]],
                    editor: str = "gpt-image-2") -> tuple[list[Turn], list[np.ndarray]]:
    """The thing everyone does: feed the whole previous output back in as the next input.
    Measured against the SAME declared masks so the comparison is fair."""
    turns, frames = [], [original.copy()]
    cur = original.copy()
    union = np.zeros(original.shape[:2], np.uint8)
    dry = config.active_provider() == "dry-run"

    for i, (instr, box) in enumerate(instrs):
        mask = box_mask(original.shape, box)
        t = Turn(i, instr, box, (0, 0, original.shape[1], original.shape[0]))
        t0 = time.time()
        prev = cur.copy()
        try:
            cur = clients.edit(editor, instr, [cur])
            if cur.shape[:2] != original.shape[:2]:
                cur = cv2.resize(cur, (original.shape[1], original.shape[0]),
                                 interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            t.status, t.reason = "error", str(e)[:200]
            turns.append(t); frames.append(cur.copy()); continue

        t.latency_s = time.time() - t0
        t.cost_usd = 0.0 if dry else config.MODELS[editor].price_usd
        union = np.maximum(union, mask)
        if dry:
            t.status, t.reason = "dry_run", "no API key — metrics N/A"
            t.metrics = {"note": "dry-run: not measurable"}
        else:
            t.status = "accepted"   # naive chaining accepts unconditionally — that is the point
            m = mask.astype(bool)
            t.metrics = {
                "target_roi_change": float(np.abs(
                    cur[m].astype(float) - prev[m].astype(float)).mean()),
                "outside_mask_vs_prev": metrics.preservation(prev, cur, 1 - mask),
                "outside_union_vs_original": metrics.preservation(original, cur, 1 - union),
            }
        turns.append(t); frames.append(cur.copy())
    return turns, frames
