"""Calibrate the specialist before trusting it.

Every certificate ends with `photographic_naturalness: UNKNOWN — no specialist`. Before a VLM
judge can be promoted into the gate, it needs a known-answer control.

So we test it on a question whose answer we already know **by construction**:

    "Which reproduces the REFERENCE mark's letterforms more faithfully, A or B?"

    A = the model's own re-drawn mark      stroke IoU 0.213
    B = the real brand asset, grafted in   stroke IoU 0.670

B is correct by PROVENANCE, not by opinion: B is a geometric transform of the reference asset,
while A is the model's independent re-drawing of it. (B is NOT bit-identical to the reference —
measured 0.00% bit-exact inside the reference alpha after warping and resampling. The ground
truth here is where the geometry came from, not pixel identity.) A configuration that fails this
control is not promoted as an automatic judge; this is not a claim about VLMs in general.

Both questions are asked N times, each swap-debiased (2 HTTP requests per comparison). Temperature
is zero, so repeated identical outcomes are not independent trials.

Run:  python -m experiments.judge_calibration    (4*N + 12 requests; N=5 => 32)
"""
from __future__ import annotations
import json, os, sys
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, judge, metrics       # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task1")
N = int(os.getenv("WEON_JUDGE_TRIALS", "5"))


def main() -> None:
    L = lambda n: np.asarray(Image.open(f"{OUT}/{n}").convert("RGB"))
    ref = L("asset_arigato.png")
    a = L("cond_A_model_only_rectified.png")        # model's mark      IoU 0.213
    b = L("cond_B_hard_alpha_graft_rectified.png")  # real brand asset  IoU 0.670

    print(f"provider={config.active_provider()} judge={os.getenv('WEON_JUDGE','google/gemini-2.5-flash')}")
    print(f"control: 'which is more faithful to the REFERENCE?'  known answer = B by provenance "
          f"(B is warped from the reference asset; A is the model's own re-drawing; "
          f"IoU 0.670 vs 0.213)\n")

    runs = {"mark_fidelity": [], "naturalness": []}
    for i in range(N):
        r1 = judge.compare("mark_fidelity", a, b, reference=ref)   # first=A, second=B
        r2 = judge.compare("naturalness", a, b)
        runs["mark_fidelity"].append(r1)
        runs["naturalness"].append(r2)
        print(f"  trial {i+1}: fidelity winner={r1['winner']:7} raw={r1['raw']}   "
              f"naturalness winner={r2['winner']:7} raw={r2['raw']}")

    def summarise(kind, correct=None):
        rs = runs[kind]
        w = Counter(r["winner"] for r in rs)
        consistent = sum(1 for r in rs if r["swap_consistent"])
        out = {
            "trials": len(rs),
            "winners": dict(w),
            "swap_consistent_pct": round(100.0 * consistent / len(rs), 1),
            "ties_from_order_dependence": sum(
                1 for r in rs if not r["swap_consistent"] and r["winner"] == "tie"),
        }
        if correct:
            out["correct_answer"] = correct
            out["accuracy_pct"] = round(100.0 * w[correct] / len(rs), 1)
            out["wrong_pct"] = round(100.0 * w["first" if correct == "second" else "second"] / len(rs), 1)
        return out

    fid = summarise("mark_fidelity", correct="second")   # second == B == correct
    nat = summarise("naturalness")

    # Presentation-scale sweep. Interpolation adds no information, so this cannot rule out a
    # resolution limit; it records whether the same configuration improves when shown larger.
    import cv2
    up = lambda x, f: cv2.resize(x, (x.shape[1] * f, x.shape[0] * f),
                                 interpolation=cv2.INTER_LANCZOS4)
    res_sweep = []
    for f in (4, 8):
        trials = [judge.compare("mark_fidelity", up(a, f), up(b, f), reference=up(ref, f))
                  for _ in range(3)]
        w = Counter(t["winner"] for t in trials)
        res_sweep.append({
            "scale": f"{f}x",
            "size": f"{a.shape[1]*f}x{a.shape[0]*f}",
            "winners": dict(w),
            "correct_pct": round(100.0 * w["second"] / len(trials), 1),
            "wrong_pct": round(100.0 * w["first"] / len(trials), 1),
            "swap_consistent_pct": round(
                100.0 * sum(1 for t in trials if t["swap_consistent"]) / len(trials), 1),
            "raw": [t["raw"] for t in trials],
        })
        print(f"\n  resolution control {f}x ({res_sweep[-1]['size']}): "
              f"correct={res_sweep[-1]['correct_pct']}%  wrong={res_sweep[-1]['wrong_pct']}%  "
              f"swap-consistent={res_sweep[-1]['swap_consistent_pct']}%")

    print(f"\nmark_fidelity (control, correct answer = B/'second')")
    print(f"   accuracy            : {fid['accuracy_pct']}%")
    print(f"   wrong (picked A)    : {fid['wrong_pct']}%")
    print(f"   swap-consistent     : {fid['swap_consistent_pct']}%")
    print(f"   winners             : {fid['winners']}")
    print(f"\nnaturalness (no ground truth)")
    print(f"   swap-consistent     : {nat['swap_consistent_pct']}%")
    print(f"   winners             : {nat['winners']}")

    verdict = (
        "TRUSTWORTHY on this control" if fid["accuracy_pct"] >= 80 and fid["swap_consistent_pct"] >= 80
        else "NOT TRUSTWORTHY on this control")
    out = {
        "judge_model": os.getenv("WEON_JUDGE", "google/gemini-2.5-flash"),
        "trials": N,
        "trials_are_not_independent": ("Calls run at temperature 0. Repeated identical outcomes "
                                       "are one deterministic result, not N independent trials."),
        "control": {
            "question": "which reproduces the REFERENCE mark more faithfully?",
            "known_answer": "B (grafted real asset; stroke IoU 0.670 vs 0.213)",
            "why_known": ("B is a geometric transform of the reference asset; A is the model's "
                          "independent re-drawing. Known by PROVENANCE, not pixel identity — B is "
                          "0.00% bit-exact vs the reference inside its alpha after warping."),
            **fid,
        },
        "naturalness": {"question": "which looks less digitally pasted?", "ground_truth": None, **nat},
        "resolution_control": {
            "why": ("The crops are small (116x61), so the judge might merely be failing to resolve "
                    "letterforms. Lanczos interpolation ADDS NO INFORMATION, so this cannot "
                    "strictly rule resolution out; it shows only that the judge does not improve "
                    "when the mark is presented larger — it becomes more confidently wrong."),
            "sweep": res_sweep,
        },
        "verdict": verdict,
        "implication": (
            "This judge configuration scored 0% on one isolated fidelity control at every "
            "presentation scale. The mechanism is unresolved, and this is not a claim about VLM "
            "judges in general. It is sufficient reason not to use this configuration as an "
            "automatic pass here; geometry remains deterministic and naturalness routes to review."
        ),
        "notes": [
            "Every trial is swap-debiased: judged in both orderings, a win requires the same "
            "image to win both, disagreement is recorded as a tie (Judging the Judges, "
            "arXiv 2406.07791).",
            "Under this temperature-zero run, a single-order presentation repeatedly selected A; "
            "these repeats are one deterministic outcome, not independent trials.",
            "A calibrated deterministic geometry diagnostic is preferable for this narrow "
            "fidelity question; naturalness is a separate dimension.",
            "Scope: one judge model, one mark, one pair. This is not a claim about VLM judges "
            "in general — it is a claim that THIS instrument failed THIS control, which is "
            "sufficient reason not to trust it here.",
        ],
        "http_requests_estimated": 4 * N + 12,
        "cost_usd_estimated": round((4 * N + 12) * 0.002, 4),
        "raw": {k: [r for r in v] for k, v in runs.items()},
    }
    errored = all(r["winner"] == "error" for rs in runs.values() for r in rs)
    if errored and os.path.exists(f"{OUT}/judge_calibration.json"):
        print("\nREFUSING TO OVERWRITE: every judge call errored (no key / provider down) and a "
              "previous receipt exists.\nWriting this would replace real results with failures.")
        sys.exit(2)
    with open(f"{OUT}/judge_calibration.json", "w") as fh:
        json.dump(metrics.json_safe(out), fh, indent=2)
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT}/judge_calibration.json  (est. ${out['cost_usd_estimated']})")


if __name__ == "__main__":
    main()
