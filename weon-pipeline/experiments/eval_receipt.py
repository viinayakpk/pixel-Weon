"""Persist the evaluation-correctness numbers the report quotes.

The report claims the old masked-SSIM scored a destroyed ROI at ~0.92 while the corrected one
scores ~0.00. That number was previously produced by an ad-hoc probe and never saved, so
"every number traces to a JSON artifact" was not true of it. This makes it true.

Also persists the hardened gate's verdict on the real turn-4 artifact, which was likewise only
ever printed.

Run:  python -m experiments.eval_receipt      (offline, free)
"""
from __future__ import annotations
import json, os, sys

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cases.task4_jacket import EDITS                       # noqa: E402
from pipeline import config, gate, ledger, metrics         # noqa: E402


def old_masked_ssim(a, b, m):
    """The bug, reproduced exactly: zero outside the mask, then average over the WHOLE image."""
    ao, bo = a.copy(), b.copy()
    ao[~m.astype(bool)] = 0
    bo[~m.astype(bool)] = 0
    return float(ssim(ao, bo, channel_axis=2))


def main() -> None:
    out = {}

    # --- 1. the masked-SSIM pathology -----------------------------------------------------
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    m = np.zeros((256, 256), np.uint8)
    m[96:160, 96:160] = 1
    b = a.copy()
    b[m.astype(bool)] = 0                      # ROI completely destroyed, rest identical

    out["masked_ssim_pathology"] = {
        "fixture": "np.random.default_rng(1), 256x256x3 uint8; ROI [96:160,96:160] zeroed",
        "roi_state": "completely destroyed",
        "old_zero_and_average_ssim": round(old_masked_ssim(a, b, m), 6),
        "new_roi_only_ssim": round(metrics.masked_ssim(a, b, m), 6),
        "new_roi_bit_exact_pct": round(metrics.bit_exact_pct(a, b, m), 4),
        "new_roi_psnr_db": round(metrics.masked_psnr(a, b, m), 3),
        "note": ("The old method averages SSIM over the whole image after zeroing the outside, "
                 "so the identical black regions dominate and a destroyed ROI still scores "
                 "near 1.0. Pinned by tests/test_metrics.py::"
                 "test_destroyed_roi_cannot_score_near_perfect."),
    }
    p = out["masked_ssim_pathology"]
    print(f"masked-SSIM pathology (destroyed ROI):")
    print(f"   old zero-and-average : {p['old_zero_and_average_ssim']}")
    print(f"   new ROI-only         : {p['new_roi_only_ssim']}")

    # --- 2. the hardened gate on the real turn-4 artifact ---------------------------------
    t4 = os.path.join(config.OUTPUTS, "task4")
    if os.path.exists(f"{t4}/ledger_step4.png"):
        prev = np.asarray(Image.open(f"{t4}/ledger_step3.png").convert("RGB"))
        res = np.asarray(Image.open(f"{t4}/ledger_step4.png").convert("RGB"))
        sup = ledger.box_mask(prev.shape, EDITS[3][1])
        checks = {
            "preservation": gate.check_preservation(prev, res, sup),
            "instruction_colour": gate.check_colour_predicate(prev, res, sup, (193, 154, 107)),
            "instruction_material": gate.check_material_predicate("canvas"),
            "boundary": gate.check_boundary(res, prev, sup),
        }
        d = gate.decide(checks)
        out["hardened_gate_postmortem_on_turn4"] = {
            "status": ("POSTMORTEM REPLAY. The live ledger committed turn 4 using gate v1 "
                       "(target pixel movement + context SSIM). This is gate v2 replayed "
                       "against the saved artifact; it is not what ran."),
            "gate_v1_outcome": "accepted (target_change=26.0, context_ssim=0.636)",
            "gate_v2_decision": d.status,
            "gate_v2_checks": {k: v.to_json() for k, v in checks.items()},
        }
        print(f"\nhardened gate (postmortem replay) on turn 4: {d.status}")
        for k, c in checks.items():
            print(f"   {k:22} {c.verdict.value:8} {c.detail}")

    path = os.path.join(config.OUTPUTS, "eval_receipt.json")
    with open(path, "w") as fh:
        json.dump(metrics.json_safe(out), fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
