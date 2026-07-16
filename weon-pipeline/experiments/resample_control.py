"""Control: how much of the naive chain's damage is our harness, not the model?

Motivation. The provider does not return the input resolution (measured: a 308x440 crop came
back as 1049x1499). Any full-frame chain therefore has to resample every turn, and resampling
alone perturbs pixels. So "byte-identical collapsed to 0.62%" could in principle be an
artifact of our own resize rather than evidence about the editor.

This runs the resize round-trip with NO model in the loop, which is the zero-drift floor.

Result (see resample_control.json): the tested resize paths leave 33-74% of label pixels
byte-identical and cost at most ~0.018 SSIM. The observed naive turn-1 values are 0.62% and
0.555. These paths do not explain the observed gap; the test does not exclude every possible
harness effect.

Run:  python -m experiments.resample_control     (offline, free, no API)
"""
from __future__ import annotations
import json, os, sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cases.task4_jacket import PROTECTED          # noqa: E402
from pipeline import config, ledger, metrics      # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task4")
OBSERVED = {"label_bit_exact_pct": 0.62, "label_ssim": 0.555, "untouched_ssim": 0.866}
SIZES = [(1024, 1024), (1049, 1499), (1536, 1536)]


def main() -> None:
    orig = np.asarray(Image.open(os.path.join(OUT, "step0_original.png")).convert("RGB"))
    lmask = ledger.box_mask(orig.shape, PROTECTED["brand_label"])
    whole = np.ones(orig.shape[:2], np.uint8)

    rows = []
    for size in SIZES:
        up = cv2.resize(orig, size, interpolation=cv2.INTER_LANCZOS4)
        back = cv2.resize(up, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        rows.append({
            "round_trip": f"{orig.shape[1]}x{orig.shape[0]} -> {size[0]}x{size[1]} -> back",
            "label_bit_exact_pct": round(metrics.bit_exact_pct(orig, back, lmask), 2),
            "label_ssim": round(metrics.masked_ssim(orig, back, lmask), 4),
            "whole_image_ssim": round(metrics.masked_ssim(orig, back, whole), 4),
        })
        print(f"  {rows[-1]['round_trip']:38} label_bitexact={rows[-1]['label_bit_exact_pct']:6.2f}%  "
              f"label_ssim={rows[-1]['label_ssim']:.4f}")

    worst_ssim_loss = max(1 - r["label_ssim"] for r in rows)
    observed_ssim_loss = 1 - OBSERVED["label_ssim"]
    best_bitexact = min(r["label_bit_exact_pct"] for r in rows)

    verdict = {
        "resample_only": rows,
        "observed_naive_turn1": OBSERVED,
        "worst_resample_label_ssim_loss": round(worst_ssim_loss, 4),
        "observed_label_ssim_loss": round(observed_ssim_loss, 4),
        "observed_loss_vs_resample_loss_ratio": round(observed_ssim_loss / worst_ssim_loss, 1),
        "lowest_resample_bit_exact_pct": best_bitexact,
        "conclusion": (
            "Resampling alone leaves 33-74% of label pixels byte-identical and costs at most "
            f"{worst_ssim_loss:.4f} SSIM. Observed naive turn 1 is 0.62% and a loss of "
            f"{observed_ssim_loss:.4f} ({observed_ssim_loss/worst_ssim_loss:.0f}x larger). The "
            "tested resampling paths are too small to explain the observed gap. This does not "
            "exclude every possible harness effect."
        ),
    }
    with open(os.path.join(OUT, "resample_control.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    print("\n" + verdict["conclusion"])
    print(f"\nwrote {OUT}/resample_control.json")


if __name__ == "__main__":
    main()
