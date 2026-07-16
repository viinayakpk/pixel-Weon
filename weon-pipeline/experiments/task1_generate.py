"""Task 1, step 1: generate the baselines. Two paid calls, nothing else.

Both calls ask for the same photoshoot from the same packshot. P2 changes two levers together:
the brand mark is described more tightly and supplied as an extra high-resolution reference.
This is an end-to-end conditioning comparison, not a single-variable prompt ablation:

  P1 (plain)       packshot + scene prompt
  P2 (constrained) packshot + tight ARIGATO reference + structured spelling/placement constraints

This tests the cheapest possible lever first — prompting and reference conditioning — before
any deterministic grafting. If P2 already preserves the mark, the graft is unnecessary and we
should say so.

Run:  python -m experiments.task1_generate       (2 API calls, ~$0.16 estimated)
"""
from __future__ import annotations
import json, os, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import clients, config, mark, metrics      # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task1" + (
    "_run_" + os.environ["WEON_RUN_ID"] if os.getenv("WEON_RUN_ID") else ""))
os.makedirs(OUT, exist_ok=True)
EDITOR = os.getenv("WEON_EDITOR", "gpt-image-2")

PACKSHOT = "case1_packshot.jpg"
LOGO_BOX = (793, 1309, 1091, 1448)          # ARIGATO, declared from the packshot

SCENE = ("Editorial product photograph of this exact sneaker on a smooth concrete surface, "
         "soft directional daylight from the left, shallow depth of field, side view.")

CONSTRAINTS = (
    " The gold brand mark on the side of the shoe must read exactly ARIGATO: seven capital "
    "letters A-R-I-G-A-T-O, gold serif type, no other text, correct spelling, same size and "
    "same position relative to the lace holes. Do not restyle, re-letter or re-space the mark. "
    "The second reference image is the exact brand mark."
)


def main() -> None:
    # Canonical paid evidence is immutable in normal operation. p1_plain.png / p2_constrained.png
    # and generate.json are the only record of the two paid Task 1 calls; a rerun (dry OR keyed)
    # would silently replace them. Regeneration must be explicit and go somewhere else.
    if os.path.exists(f"{OUT}/generate.json") and not os.getenv("WEON_RUN_ID"):
        print(f"REFUSING TO RUN: {OUT}/generate.json exists and is canonical paid evidence.")
        print("  regenerate into a fresh run : "
              "WEON_RUN_ID=myrun python -m experiments.task1_generate")
        print("  promoting a run to canonical is a deliberate, manual copy.")
        sys.exit(2)

    pack = np.asarray(Image.open(os.path.join(config.TEST_DATA, PACKSHOT)).convert("RGB"))
    x0, y0, x1, y1 = LOGO_BOX
    logo_ref = np.ascontiguousarray(pack[y0:y1, x0:x1])
    Image.fromarray(logo_ref).save(f"{OUT}/logo_reference.png")

    dry = config.active_provider() == "dry-run"
    print(f"provider={config.active_provider()} editor={EDITOR}")
    print(f"packshot={pack.shape} logo_ref={logo_ref.shape}\n")

    runs = [
        ("p1_plain", SCENE, [pack]),
        ("p2_constrained", SCENE + CONSTRAINTS, [pack, logo_ref]),
    ]
    rec = {"editor": EDITOR, "provider": config.active_provider(),
           "packshot": PACKSHOT, "logo_box": list(LOGO_BOX), "runs": []}

    for name, prompt, refs in runs:
        t0 = time.time()
        try:
            out = clients.edit(EDITOR, prompt, refs)
            dt = time.time() - t0
            Image.fromarray(out).save(f"{OUT}/{name}.png")
            row = {"name": name, "prompt": prompt, "n_refs": len(refs),
                   "status": "dry_run" if dry else "ok",
                   "output_shape": list(out.shape), "latency_s": round(dt, 1),
                   "cost_usd_estimated": 0.0 if dry else config.MODELS[EDITOR].price_usd}
            print(f"  {name:16} {row['status']:8} {out.shape} {dt:5.1f}s")
        except Exception as e:
            row = {"name": name, "status": "error", "error": str(e)[:300]}
            print(f"  {name:16} ERROR {str(e)[:120]}")
        rec["runs"].append(row)

    rec["cost_usd_estimated_total"] = round(
        sum(r.get("cost_usd_estimated", 0) for r in rec["runs"]), 4)
    with open(f"{OUT}/generate.json", "w") as fh:
        json.dump(metrics.json_safe(rec), fh, indent=2)
    print(f"\nwrote {OUT}/generate.json  (estimated ${rec['cost_usd_estimated_total']})")
    print("next: inspect p1/p2, declare the target quad by hand, then run task1_compare")


if __name__ == "__main__":
    main()
