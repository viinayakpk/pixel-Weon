"""Task 4 experiment: naive sequential chaining vs the pixel edit ledger.

Same base image, same five instructions, same declared intent masks and same editor. This is an
end-to-end workflow comparison, not a single-variable ablation: the naive arm receives a full
frame, while the ledger arm receives a contextual crop before hard compositing.

The probe: the "A DAY'S MARCH" brand label is never targeted by any instruction. Whatever
happens to it is, by definition, off-target degradation. We measure it per turn.

Run:  python -m experiments.task4_compare            (uses OPENROUTER_API_KEY)
      WEON_DRY_RUN=1 python -m experiments.task4_compare   (offline; reports N/A)
"""
from __future__ import annotations
import json, os, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cases.task4_jacket import BASE, EDITS, PROTECTED
from pipeline import config, ledger, metrics

OUT = os.path.join(config.OUTPUTS, "task4" + (
    "_" + os.environ["WEON_OUT_SUFFIX"] if os.getenv("WEON_OUT_SUFFIX") else ""))
os.makedirs(OUT, exist_ok=True)
EDITOR = os.getenv("WEON_EDITOR", "gpt-image-2")


def probe(original: np.ndarray, frame: np.ndarray, union: np.ndarray) -> dict:
    """Everything we can say about off-target damage, measured against the ORIGINAL."""
    lbox = PROTECTED["brand_label"]
    lmask = ledger.box_mask(original.shape, lbox)
    out_union = 1 - union.astype(np.uint8)
    # NOTE: no OCR here. pytesseract is not installed in this environment and returns '' even
    # on the pristine original, so an OCR column would have been empty for every row and
    # measured nothing. Label damage is reported via bit-exact %, SSIM and the visual grid.
    return {
        # the protected logo — never asked to change
        "label_bit_exact_pct": metrics.bit_exact_pct(original, frame, lmask),
        "label_ssim": metrics.masked_ssim(original, frame, lmask),
        # everything nobody asked to change
        "outside_union_bit_exact_pct": metrics.bit_exact_pct(original, frame, out_union),
        "outside_union_ssim": metrics.masked_ssim(original, frame, out_union),
        "outside_union_psnr": metrics.masked_psnr(original, frame, out_union),
    }


def main() -> None:
    original = np.asarray(Image.open(BASE).convert("RGB"))
    dry = config.active_provider() == "dry-run"

    # Canonical evidence is immutable, regardless of provider.
    # A dry rerun would replace it with unedited copies of the input. A KEYED rerun would replace
    # it with a different sample from a non-deterministic model. Either destroys the artifacts
    # every number in the report cites. An earlier version of this guard only fired on dry runs,
    # which meant `python run.py task4` with a key would silently overwrite them.
    if os.path.exists(f"{OUT}/metrics.json") and not os.getenv("WEON_OUT_SUFFIX"):
        print(f"REFUSING TO RUN: {OUT}/metrics.json exists and is canonical evidence.")
        print("  inspect it           : python run.py evidence")
        print("  regenerate elsewhere : WEON_OUT_SUFFIX=run2 python -m experiments.task4_compare")
        print("  promoting a run to canonical is a deliberate, manual copy.")
        sys.exit(2)

    print(f"provider={config.active_provider()} editor={EDITOR} base={original.shape}")
    print(f"{len(EDITS)} edits x 2 strategies = {len(EDITS)*2} calls\n")

    Image.fromarray(original).save(f"{OUT}/step0_original.png")
    lbox = PROTECTED["brand_label"]
    x0, y0, x1, y1 = lbox
    Image.fromarray(original[y0:y1, x0:x1]).save(f"{OUT}/label_reference.png")

    results = {"editor": EDITOR, "provider": config.active_provider(),
               "base": BASE, "protected": {k: list(v) for k, v in PROTECTED.items()},
               "edits": [{"instruction": i, "box": list(b)} for i, b in EDITS],
               "naive": [], "ledger": []}
    t_start = time.time()

    # ---- A. naive chaining -------------------------------------------------------------
    print("[A] naive chaining (previous output -> next input)")
    turns, frames = ledger.run_naive_chain(original, EDITS, editor=EDITOR)
    union = np.zeros(original.shape[:2], np.uint8)
    for i, (t, f) in enumerate(zip(turns, frames[1:])):
        union = np.maximum(union, ledger.box_mask(original.shape, t.intended_box))
        Image.fromarray(f).save(f"{OUT}/naive_step{i+1}.png")
        row = {**t.to_json(), "probe": None if dry else probe(original, f, union)}
        results["naive"].append(row)
        if not dry:
            p = row["probe"]
            print(f"  turn {i+1}: {t.status:9} label_bitexact={p['label_bit_exact_pct']:6.2f}% "
                  f"label_ssim={p['label_ssim']:.3f} outside_bitexact={p['outside_union_bit_exact_pct']:6.2f}%")

    # ---- B. pixel ledger ---------------------------------------------------------------
    print("\n[B] pixel edit ledger (declared mask, crop edit, composite accepted pixels)")
    L = ledger.Ledger(original, editor=EDITOR)
    # Score BOTH arms over the cumulative union of ATTEMPTED supports. Using L.union_mask here
    # would score the ledger over its ACCEPTED union only, so after a rejected turn the two arms
    # would be measured over different outside regions and the columns would not be comparable.
    attempted_union = np.zeros(original.shape[:2], np.uint8)
    for i, (instr, box) in enumerate(EDITS):
        attempted_union = np.maximum(attempted_union, ledger.box_mask(original.shape, box))
        t = L.apply(instr, box)
        Image.fromarray(L.canonical).save(f"{OUT}/ledger_step{i+1}.png")
        if t.status == "rejected" and t.candidate is not None:
            # save what we threw away, so the rollback can be audited visually rather than
            # only inferred from step N-1 and step N being identical
            Image.fromarray(t.candidate).save(f"{OUT}/ledger_step{i+1}_REJECTED_candidate.png")
        row = {**t.to_json(),
               "probe": None if dry else probe(original, L.canonical, attempted_union)}
        results["ledger"].append(row)
        if not dry:
            p = row["probe"]
            print(f"  turn {i+1}: {t.status:9} label_bitexact={p['label_bit_exact_pct']:6.2f}% "
                  f"label_ssim={p['label_ssim']:.3f} outside_bitexact={p['outside_union_bit_exact_pct']:6.2f}%")
            print(f"           gate: {t.reason}")

    results["wall_clock_s"] = round(time.time() - t_start, 1)
    results["cost_usd_total"] = round(
        sum(r.get("cost_usd", 0) for r in results["naive"] + results["ledger"]), 4)
    with open(f"{OUT}/metrics.json", "w") as fh:
        json.dump(metrics.json_safe(results), fh, indent=2)   # inf -> null: strict JSON
    print(f"\nwrote {OUT}/metrics.json  |  {results['wall_clock_s']}s  "
          f"|  ${results['cost_usd_total']}")


if __name__ == "__main__":
    main()
