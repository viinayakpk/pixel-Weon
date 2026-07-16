"""Fix an apples-to-apples error in the Task 4 broad metric. No API calls.

The bug. `task4_compare.py` accumulated the naive arm's union from EVERY attempted support, but
the ledger arm's union from `L.union_mask`, which only grows on ACCEPT. Turn 3 was rejected, so
from turn 3 onward the two arms were scored over DIFFERENT outside regions — the ledger's
"outside" still included edit 3's box while the naive arm's did not. That makes the
`outside_union_*` columns not directly comparable.

The fix. Recompute both arms over one cumulative union of all five PREDECLARED attempted
supports, which is identical for both by construction and does not depend on what either arm
accepted.

Unaffected: the protected-label headline. The label mask is fixed, declared up front, and never
part of any union — so 0.62% / 0.555 stand exactly as reported.

Run:  python -m experiments.recompute_common_union      (offline, free)
"""
from __future__ import annotations
import json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cases.task4_jacket import BASE, EDITS, PROTECTED       # noqa: E402
from pipeline import config, ledger, metrics                # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task4")


def main() -> None:
    original = np.asarray(Image.open(f"{OUT}/step0_original.png").convert("RGB"))
    lmask = ledger.box_mask(original.shape, PROTECTED["brand_label"])

    # one cumulative union of ATTEMPTED supports — identical for both arms at every turn
    attempted = []
    u = np.zeros(original.shape[:2], np.uint8)
    for _, box in EDITS:
        u = np.maximum(u, ledger.box_mask(original.shape, box))
        attempted.append(u.copy())

    rows = []
    for turn in range(1, len(EDITS) + 1):
        outside = 1 - attempted[turn - 1]
        row = {"turn": turn,
               "outside_union_px": int(outside.sum()),
               "union_definition": "cumulative union of all ATTEMPTED supports (same for both arms)"}
        for arm in ("naive", "ledger"):
            f = np.asarray(Image.open(f"{OUT}/{arm}_step{turn}.png").convert("RGB"))
            row[arm] = {
                "outside_bit_exact_pct": round(metrics.bit_exact_pct(original, f, outside), 4),
                "outside_ssim": round(metrics.masked_ssim(original, f, outside), 4),
                "label_bit_exact_pct": round(metrics.bit_exact_pct(original, f, lmask), 4),
                "label_ssim": round(metrics.masked_ssim(original, f, lmask), 4),
            }
        rows.append(row)
        print(f"  turn {turn}: naive outside_exact={row['naive']['outside_bit_exact_pct']:7.3f}%  "
              f"ssim={row['naive']['outside_ssim']:.4f}   |   "
              f"ledger outside_exact={row['ledger']['outside_bit_exact_pct']:7.3f}%  "
              f"ssim={row['ledger']['outside_ssim']:.4f}")

    out = {
        "why": ("Corrects an apples-to-apples error: the original run scored the naive arm over "
                "all attempted supports and the ledger arm over its ACCEPTED union only. After "
                "turn 3 was rejected those are different regions."),
        "rows": rows,
        "note_on_the_headline": ("The protected-label numbers are unaffected — the label mask is "
                                 "declared up front and is never part of any union. 0.62% / 0.555 "
                                 "at turn 1 stand as reported."),
        "note_on_the_ledger_arm": ("Under the ATTEMPTED union the ledger's outside region now "
                                   "includes edit 3's box, which the ledger never wrote because "
                                   "it rejected that turn. Those pixels are therefore still "
                                   "byte-identical, which is why the ledger's figure does not "
                                   "move: refusing to write is preservation."),
        "not_a_single_variable_ablation": ("This is an end-to-end strategy comparison, not a clean "
                                           "ablation: the naive arm receives the full frame, the "
                                           "ledger arm receives a crop. Field of view, scale and "
                                           "context all differ."),
    }
    with open(f"{OUT}/common_union_recompute.json", "w") as fh:
        json.dump(metrics.json_safe(out), fh, indent=2)
    print(f"\nwrote {OUT}/common_union_recompute.json")


if __name__ == "__main__":
    main()
