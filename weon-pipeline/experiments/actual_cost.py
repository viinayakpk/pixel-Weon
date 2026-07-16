"""Account-level provider usage compared with retained experiment estimates.

Provider usage covers every call on the key, including calls outside the retained experiments.
It therefore cannot be treated as an experiment-specific invoice or a clean price reconciliation.

Run:  python -m experiments.actual_cost      (1 free metadata call, no generation)
"""
from __future__ import annotations
import json, os, sys

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, metrics       # noqa: E402

load_dotenv()


def estimated_from_our_tables() -> dict:
    """Sum what our own price table claims each experiment cost."""
    out, total = {}, 0.0
    for path, key in (("task4/metrics.json", "cost_usd_total"),
                      ("task4_nano/metrics.json", "cost_usd_total"),
                      ("task4_g31/metrics.json", "cost_usd_total"),
                      ("task4_gpt54/metrics.json", "cost_usd_total"),
                      ("task1/generate.json", "cost_usd_estimated_total"),
                      ("task1/prompt_ceiling.json", "cost_usd_estimated_total"),
                      ("task1/judge_calibration.json", "cost_usd_estimated")):
        p = os.path.join(config.OUTPUTS, path)
        if os.path.exists(p):
            v = json.load(open(p)).get(key) or 0.0
            out[path] = round(float(v), 4)
            total += float(v)
    out["total"] = round(total, 4)
    return out


def main() -> None:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("no OPENROUTER_API_KEY — cannot query real spend")
        return

    r = requests.get("https://openrouter.ai/api/v1/key",
                     headers={"Authorization": f"Bearer {key}"}, timeout=30)
    d = r.json().get("data", {})
    actual = float(d.get("usage") or 0.0)
    limit = d.get("limit")
    est = estimated_from_our_tables()

    out = {
        "actual_usd_billed_by_provider": round(actual, 4),
        "estimated_usd_from_our_price_table": est,
        "drift_pct": round(100.0 * (est["total"] - actual) / actual, 1) if actual else None,
        "budget_limit_usd": limit,
        "remaining_usd": round(float(limit) - actual, 2) if limit else None,
        "drift_direction": ("negative = our table UNDER-estimates what the provider charged"),
        "note": ("`actual` is what the provider charged for everything on this key, including "
                 "smoke tests and failed/aborted activity not cleanly represented in retained "
                 "experiment receipts. So this is NOT a clean per-experiment reconciliation: "
                 "the scopes differ, and the gap cannot be interpreted as a price-table error. "
                 "Per-image prices in config.py remain unverified individually."),
    }
    with open(os.path.join(config.OUTPUTS, "actual_cost.json"), "w") as fh:
        json.dump(metrics.json_safe(out), fh, indent=2)

    direction = "low" if (out["drift_pct"] or 0) < 0 else "high"
    print(f"  actual billed by provider : ${out['actual_usd_billed_by_provider']}")
    print(f"  our tables estimated      : ${est['total']}")
    print(f"  drift                     : {out['drift_pct']}%  (our table runs {direction})")
    print(f"  remaining of ${limit}          : ${out['remaining_usd']}")
    print(f"\nwrote {config.OUTPUTS}/actual_cost.json")


if __name__ == "__main__":
    main()
