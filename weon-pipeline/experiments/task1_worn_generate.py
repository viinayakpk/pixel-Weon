"""Task 1, worn case: generate the baseline. Hard-capped at 2 attempts.

The product-shot experiment answered an easier question than the brief asks. The brief's product is
model + environment + garment -> the model WEARING the garment. This runs that.

Attempt budget is enforced in code, not by intention. Generating until something looks good is how
a cherry-picked result gets made; the manifest and the scene prompt were both fixed before this ran
(cases/task1_worn.py), and the attempts differ only in conditioning strategy:

  a1_packshot_only   packshot + scene                       (the plain lever)
  a2_with_markref    packshot + tight ARIGATO reference     (the conditioned lever)

Both are paid. If neither yields a usable worn shot with a measurable shoe, that is the reported
result.

Run:  python -m experiments.task1_worn_generate      (2 API calls, ~$0.16 estimated)
"""
from __future__ import annotations
import hashlib, json, os, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cases.task1_worn import MANIFEST, PACKSHOT, SCENE, SUCCESS_CRITERIA, SYNTHETIC_NOTE  # noqa: E402
from pipeline import clients, config, metrics                                             # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task1_worn" + (
    "_" + os.environ["WEON_RUN_ID"] if os.getenv("WEON_RUN_ID") else ""))
os.makedirs(OUT, exist_ok=True)
EDITOR = os.getenv("WEON_EDITOR", "gpt-image-2")
LOGO_BOX = tuple(MANIFEST[0]["packshot_box"])
MAX_ATTEMPTS = 2


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()[:12]


def main() -> None:
    if os.path.exists(f"{OUT}/generate.json") and not os.getenv("WEON_RUN_ID"):
        print(f"REFUSING TO RUN: {OUT}/generate.json exists and is canonical paid evidence.")
        print("  regenerate elsewhere : WEON_RUN_ID=v2 python -m experiments.task1_worn_generate")
        sys.exit(2)

    pack = np.asarray(Image.open(os.path.join(config.TEST_DATA, PACKSHOT)).convert("RGB"))
    x0, y0, x1, y1 = LOGO_BOX
    mark_ref = np.ascontiguousarray(pack[y0:y1, x0:x1])
    Image.fromarray(mark_ref).save(f"{OUT}/mark_reference.png")

    dry = config.active_provider() == "dry-run"
    print(f"provider={config.active_provider()} editor={EDITOR}")
    print(f"packshot={pack.shape} mark_ref={mark_ref.shape}")
    print(f"manifest: {len(MANIFEST)} declared identity attributes "
          f"({sum(1 for m in MANIFEST if m['checkable'].startswith('yes'))} automatically checkable)")
    print(f"attempt budget: {MAX_ATTEMPTS} (enforced)\n")

    attempts = [
        ("a1_packshot_only", SCENE + SYNTHETIC_NOTE, [pack]),
        ("a2_with_markref", SCENE + SYNTHETIC_NOTE +
         " The second reference image is the exact gold ARIGATO wordmark that appears on the "
         "lateral side of the shoe. Reproduce it exactly; do not restyle or re-letter it.",
         [pack, mark_ref]),
    ]

    rec = {"case": "task1_worn", "editor": EDITOR, "provider": config.active_provider(),
           "packshot": PACKSHOT, "packshot_sha12": _sha(pack),
           "manifest_declared_before_generation": True,
           "manifest": MANIFEST, "success_criteria": SUCCESS_CRITERIA,
           "max_attempts": MAX_ATTEMPTS, "attempts": []}

    for name, prompt, refs in attempts[:MAX_ATTEMPTS]:
        t0 = time.time()
        try:
            out = clients.edit(EDITOR, prompt, refs)
            dt = round(time.time() - t0, 1)
            Image.fromarray(out).save(f"{OUT}/{name}.png")
            row = {"name": name, "prompt": prompt, "n_refs": len(refs),
                   "status": "dry_run" if dry else "ok", "output_shape": list(out.shape),
                   "output_sha12": _sha(out), "latency_s": dt,
                   "cost_usd_estimated": 0.0 if dry else config.MODELS[EDITOR].price_usd}
            print(f"  {name:18} {row['status']:8} {out.shape}  {dt:5.1f}s")
        except Exception as e:
            row = {"name": name, "status": "error", "error": str(e)[:300]}
            print(f"  {name:18} ERROR {str(e)[:110]}")
        rec["attempts"].append(row)

    rec["cost_usd_estimated_total"] = round(
        sum(a.get("cost_usd_estimated", 0) for a in rec["attempts"]), 4)
    with open(f"{OUT}/generate.json", "w") as fh:
        json.dump(metrics.json_safe(rec), fh, indent=2)
    print(f"\nwrote {OUT}/generate.json  (est. ${rec['cost_usd_estimated_total']})")
    print("next: LOOK at both. If neither shows a worn shoe with a measurable wordmark,")
    print("      that is the reported result — do not generate a third.")


if __name__ == "__main__":
    main()
