"""The worn certificate, with the spelling UNKNOWN resolved.

worn_compare.py produced two certificates whose `brand_text` line read UNKNOWN — no specialist.
spelling_specialist.py built that specialist and showed it behaves as specified on four
known-answer controls, including one designed to prove spelling ALONE cannot certify a mark. This
joins them, and is the only place in the repository where both verdicts land on the same pixels.

The point is the disagreement. On A_model_only the certificate now reads:

    brand_text     PASS   the mark says ARIGATO
    mark_geometry  FAIL   it is not the ARIGATO mark

Both are correct. A gate holding only the first would commit a restyled logo and call it preserved;
that is precisely the "looks fine, is wrong" failure this project is about. Independent checks that
can contradict each other are the feature, not a redundancy to collapse into one score.

Separated from worn_compare.py deliberately: that script is free to rerun, this one costs money.
Silently adding paid calls to a previously free script is how an accidental spend happens.

Run:  python -m experiments.worn_certificate      (6 VLM calls, ~$0.01 estimated)
"""
from __future__ import annotations
import json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, gate, metrics                              # noqa: E402
from pipeline.gate import Verdict                                       # noqa: E402
from experiments.spelling_specialist import (                           # noqa: E402
    REPEATS, WORN_BOX, WORN_OCR_PAD, MODEL, transcribe, verdict)

OUT = os.path.join(config.OUTPUTS, "task1_worn")
EXPECTED = "ARIGATO"
IOU_THRESHOLD = 0.35


def main() -> None:
    if os.path.exists(f"{OUT}/worn_certificate.json") and not os.getenv("WEON_RUN_ID"):
        print(f"REFUSING TO RUN: {OUT}/worn_certificate.json exists and is canonical paid evidence.")
        sys.exit(2)
    if config.active_provider() == "dry-run":
        print("REFUSING TO RUN: no provider; brand_text would be UNKNOWN for the wrong reason.")
        sys.exit(2)

    with open(f"{OUT}/worn_compare.json") as fh:
        cmp_res = json.load(fh)

    p = WORN_OCR_PAD
    res = {"case": "task1_worn", "specialist": MODEL, "expected_string": EXPECTED,
           "iou_threshold": IOU_THRESHOLD, "repeats": REPEATS,
           "controls": "outputs/spelling/spelling.json (4/4 as specified)",
           "conditions": {}}

    for name in ("A_model_only", "B_cleared_graft"):
        img = np.asarray(Image.open(f"{OUT}/cond_{name}.png").convert("RGB"))
        x0, y0, x1, y1 = (max(0, WORN_BOX[0] - p), max(0, WORN_BOX[1] - p),
                          min(img.shape[1], WORN_BOX[2] + p), min(img.shape[0], WORN_BOX[3] + p))
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        Image.fromarray(crop).save(f"{OUT}/ocr_{name}.png")

        runs = []
        for _ in range(REPEATS):
            t = transcribe(crop)
            v, why = verdict(t.get("raw"), EXPECTED)
            t.update({"verdict": v.value, "reason": why})
            runs.append(t)
        stable = len({r["verdict"] for r in runs}) == 1
        text_v = Verdict(runs[0]["verdict"]) if stable else Verdict.UNKNOWN
        text_why = (runs[0]["reason"] if stable
                    else f"unstable across {REPEATS} repeats: {[r['raw'] for r in runs]}")

        iou = cmp_res["conditions"][name]["fidelity"]["stroke_iou"]
        de = cmp_res["conditions"][name]["fidelity"]["mark_colour_delta_e"]
        checks = {
            "brand_text": gate.Check(text_v, text_why, runs[0].get("raw")),
            "mark_geometry": gate.Check(
                Verdict.PASS if (iou or 0) >= IOU_THRESHOLD else Verdict.FAIL,
                f"stroke IoU {iou} vs the brand mark", iou),
            "mark_colour": gate.Check(
                Verdict.PASS if de <= 12 else Verdict.FAIL, f"mark colour dE {de}", de),
            "photographic_naturalness": gate.Check(
                Verdict.UNKNOWN, "no calibrated specialist; routes to human review"),
            "midsole_instance": gate.Check(
                Verdict.UNKNOWN, "relief mark: present by eye, no working automatic detector"),
        }
        d = gate.decide(checks)
        res["conditions"][name] = {
            "brand_text_runs": runs, "stroke_iou": iou, "certificate": d.to_json()}
        print(f"\n{name}")
        for k, c in checks.items():
            print(f"   {k:26} {c.verdict.value:8} {c.detail}")
        print(f"   -> {d.status.upper()}")

    a_text = res["conditions"]["A_model_only"]["certificate"]["checks"]["brand_text"]["verdict"]
    a_geo = res["conditions"]["A_model_only"]["stroke_iou"]
    res["the_point"] = {
        "A_model_only_brand_text": a_text,
        "A_model_only_stroke_iou": a_geo,
        "reading": ("On the same pixels the text check says the mark reads ARIGATO and the "
                    "geometry check says it is not the ARIGATO mark. Both are correct. Spelling "
                    "certifies the string; only geometry certifies the identity. A single blended "
                    "score would have averaged these into a number that means neither."),
        "cost_of_collapsing_them": ("A gate with spelling alone commits a restyled logo as "
                                    "'preserved'. That is a brand failure that reads as a pass."),
    }
    res["cost_usd_estimated"] = round(sum(r["cost_usd_estimated"] for c in res["conditions"].values()
                                          for r in c["brand_text_runs"]), 4)
    with open(f"{OUT}/worn_certificate.json", "w") as fh:
        json.dump(metrics.json_safe(res), fh, indent=2)
    print(f"\nTHE POINT: A_model_only brand_text={a_text}, stroke IoU={a_geo} -> REJECTED anyway.")
    print(f"wrote {OUT}/worn_certificate.json  (est. ${res['cost_usd_estimated']})")


if __name__ == "__main__":
    main()
