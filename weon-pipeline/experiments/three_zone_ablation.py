"""Task 4, second direction: does a three-zone commit mask fix the turn-4 rectangle?

Zero additional API calls. Turn 4 was ACCEPTED, so `ledger_step4.png` contains the model's
candidate pixels inside the declared box, and `ledger_step3.png` is the previous canonical
state (turn 3 was rejected, so step2 == step3). Everything below is a re-composite of pixels
we already paid for.

Two hypotheses are tested, because inspecting the original showed they are NOT the same thing:

  H1 (collar): a hard binary mask puts a step edge on the boundary; an inward-only collar
               should reduce the seam while preserving the exterior exactly.
  H2 (grounding): the declared box (580,450,690,570) is much larger than the ACTUAL welt
               pocket (~x578-650, y452-495, traced from the CLAHE-enhanced original). If the
               artifact is mostly a grounding error, committing through the true welt polygon
               should help more than any amount of feathering the wrong region.

Run:  python -m experiments.three_zone_ablation      (offline, free)
"""
from __future__ import annotations
import json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cases.task4_jacket import EDITS                       # noqa: E402
from pipeline import config, ledger, metrics, zones        # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task4")
BOX = EDITS[3][1]                       # (580,450,690,570) — declared by hand before the run
COLLARS = [0, 4, 8, 12]

# The real welt pocket, traced from the CLAHE-enhanced ORIGINAL (outputs/pocket_grid_enhanced.png).
# It is a slanted welt, not a rectangle, and it is far smaller than the box that was declared.
# Clamped to x >= 580 because the box committed nothing left of that, so no candidate pixels
# exist there — a direct consequence of declaring the box by hand.
WELT = np.array([[584, 462], [648, 452], [652, 478], [580, 492]], np.int32)


def main() -> None:
    prev = np.asarray(Image.open(f"{OUT}/ledger_step3.png").convert("RGB"))
    step4 = np.asarray(Image.open(f"{OUT}/ledger_step4.png").convert("RGB"))

    box_mask = ledger.box_mask(prev.shape, BOX)
    welt_mask = zones.polygon_mask(prev.shape, WELT)
    # candidate pixels only exist where turn 4 committed them: inside the box
    welt_mask = (welt_mask & box_mask).astype(np.uint8)
    candidate = step4                    # inside the box these ARE the model's pixels

    print(f"box support  : {int(box_mask.sum()):6d} px")
    print(f"welt support : {int(welt_mask.sum()):6d} px "
          f"({100*welt_mask.sum()/box_mask.sum():.1f}% of the box that was actually committed)")
    print()

    rows = []
    for support_name, support in (("box", box_mask), ("welt_polygon", welt_mask)):
        for collar in COLLARS:
            res = zones.commit(prev, candidate, support, collar)
            outside = 1 - support
            row = {
                "support": support_name,
                "collar_px": collar,
                "exterior_bit_exact_pct": round(metrics.bit_exact_pct(prev, res, outside), 4),
                "retained_delta_pct": round(zones.retained_delta_pct(prev, candidate, res, support), 2),
                "boundary_gradient": round(zones.boundary_gradient(res, support), 3),
                "support_px": int(support.sum()),
            }
            rows.append(row)
            Image.fromarray(res).save(f"{OUT}/zone_{support_name}_collar{collar}.png")
            x0, y0, x1, y1 = ledger.context_crop(prev.shape, BOX)
            Image.fromarray(res[y0:y1, x0:x1]).resize(((x1-x0)*3, (y1-y0)*3), Image.NEAREST) \
                 .save(f"{OUT}/zone_{support_name}_collar{collar}_crop.png")
            print(f"  {support_name:13} collar={collar:2}px | exterior_exact="
                  f"{row['exterior_bit_exact_pct']:7.3f}% | retained_delta="
                  f"{row['retained_delta_pct']:6.2f}% | boundary_grad={row['boundary_gradient']:7.3f}")

    base = next(r for r in rows if r["support"] == "box" and r["collar_px"] == 0)
    out = {
        "note": "Re-composite of already-paid-for pixels from turn 4. No API calls.",
        "declared_box": list(BOX),
        "welt_polygon": WELT.tolist(),
        "baseline_shipped_in_task4_run": "support=box, collar=0 (the tan rectangle)",
        "rows": rows,
        "caveats": [
            "boundary_gradient is a seam DIAGNOSTIC, not a perceptual metric.",
            "retained_delta_pct falls as the collar grows: that is the cost of the tradeoff.",
            "exterior_bit_exact_pct is 100% in every condition BY CONSTRUCTION (inward-only).",
            "The welt polygon was traced by hand from the original; it is not segmentation.",
            "n=1 edit on 1 image.",
        ],
    }
    with open(f"{OUT}/three_zone_ablation.json", "w") as fh:
        json.dump(metrics.json_safe(out), fh, indent=2)
    print(f"\nbaseline actually shipped: box/collar=0 -> boundary_grad={base['boundary_gradient']}")
    print(f"wrote {OUT}/three_zone_ablation.json")


if __name__ == "__main__":
    main()
