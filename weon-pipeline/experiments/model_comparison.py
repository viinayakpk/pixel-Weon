"""Offline cross-editor audit for the four Task 4 naive chains.

This script deliberately uses only the protected-label probe, Gate-v1 decisions, latency and
configured cost. The older paid receipts used different broad-union bookkeeping after rejected
turns, so those broad-region fields are not used for the cross-editor comparison.

Run:  python -m experiments.model_comparison     (offline, free, no API)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, metrics  # noqa: E402


RUNS = {
    "gpt-image-2": {
        "vendor": "OpenAI", "dir": Path(config.OUTPUTS) / "task4",
    },
    "gpt-5.4-image-2": {
        "vendor": "OpenAI", "dir": Path(config.OUTPUTS) / "task4_gpt54",
    },
    "nano-banana-pro (Gemini 3 Pro image)": {
        "vendor": "Google", "dir": Path(config.OUTPUTS) / "task4_nano",
    },
    "gemini-3.1-flash-image": {
        "vendor": "Google", "dir": Path(config.OUTPUTS) / "task4_g31",
    },
}
OUT_JSON = Path(config.OUTPUTS) / "task4" / "model_comparison_4x.json"
OUT_GRID = Path(config.OUTPUTS) / "task4" / "model_comparison_4x.png"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_run(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def build_receipt() -> dict:
    raw = {name: _load_run(meta["dir"]) for name, meta in RUNS.items()}
    first = raw["gpt-image-2"]
    same_metadata = all(
        run["base"] == first["base"]
        and run["edits"] == first["edits"]
        and run["protected"] == first["protected"]
        for run in raw.values()
    )
    step0_hashes = {
        name: _sha256(meta["dir"] / "step0_original.png") for name, meta in RUNS.items()
    }
    same_pixels = len(set(step0_hashes.values())) == 1
    same_case = same_metadata and same_pixels
    if not same_case:
        raise ValueError("cross-editor receipts do not share metadata and byte-identical inputs")

    editors = {}
    hashes = {}
    for name, run in raw.items():
        run_dir = RUNS[name]["dir"]
        hashes[f"{run_dir.name}/metrics.json"] = _sha256(run_dir / "metrics.json")
        hashes[f"{run_dir.name}/step0_original.png"] = step0_hashes[name]
        for idx in range(1, 6):
            p = run_dir / f"naive_step{idx}.png"
            hashes[f"{run_dir.name}/{p.name}"] = _sha256(p)

        statuses = [x["status"] for x in run["ledger"]]
        ledger_complete = all(x in {"accepted", "rejected"} for x in statuses)
        editors[name] = {
            "vendor": RUNS[name]["vendor"],
            "run_editor_key": run["editor"],
            "naive_label_ssim": [round(x["probe"]["label_ssim"], 6) for x in run["naive"]],
            "naive_label_bit_exact_pct": [
                round(x["probe"]["label_bit_exact_pct"], 6) for x in run["naive"]
            ],
            "ledger_complete": ledger_complete,
            "ledger_label_bit_exact_pct": (
                [round(x["probe"]["label_bit_exact_pct"], 6) for x in run["ledger"]]
                if ledger_complete else None
            ),
            "gate_v1_status": statuses,
            "gate_v1_commits": (
                sum(x["status"] == "accepted" for x in run["ledger"])
                if ledger_complete else None
            ),
            "wall_clock_s": run["wall_clock_s"],
            "configured_cost_estimate_usd": run["cost_usd_total"],
        }

    return {
        "scope": "one jacket, one five-edit naive chain per editor, no replicates or usable seeds",
        "same_saved_case_verified": same_case,
        "editors": editors,
        "inputs_sha256": hashes,
        "supported_findings": [
            "Across the four sampled naive runs, one full-frame edit leaves only "
            "0.247748-0.619369% of the protected label byte-identical.",
            "For the three complete ledger arms, the protected label is 100% byte-exact because "
            "pixels outside each declared support are copied forward.",
            "Gate v1 commits 4/5 gpt-image-2, 3/5 nano-banana-pro and 3/5 "
            "gemini-3.1-flash-image candidates. The gpt-5.4 ledger arm is incomplete.",
        ],
        "manual_visual_audit": {
            "artifact": "model_comparison_4x.png",
            "observation": (
                "In this case the two OpenAI runs change label scale/placement substantially, "
                "while the two Google runs stay better aligned but their letterforms visibly "
                "diverge from the original. This is a within-case pattern, not a vendor signature."
            ),
            "metric_implication": (
                "SSIM alone cannot certify text identity. OCR would be a necessary-not-sufficient "
                "check: misspelling can fail, but correct spelling cannot certify typography."
            ),
            "not_claimed": (
                "No exact corrupted-string transcription, OCR score, independent human rating, "
                "vendor-level rule or general model ranking."
            ),
        },
        "limitations": (
            "The editors are non-deterministic and were each sampled once. The gpt-5.4 ledger "
            "arm has three network errors and is excluded from ledger comparisons. Saved "
            "broad-region metrics are excluded because rejected-turn union bookkeeping differed "
            "in historical receipts. Cost values are configured estimates, not per-run invoices."
        ),
    }


def make_grid() -> None:
    # A fixed, generous crop makes displacement visible rather than tracking the label.
    crop = (300, 135, 700, 355)
    cell_w, cell_h = 300, 165
    top, left, gap = 38, 260, 8
    cols = ["original", "turn 1", "turn 2", "turn 3", "turn 4", "turn 5"]
    rows = list(RUNS)
    canvas = Image.new(
        "RGB",
        (left + len(cols) * (cell_w + gap), top + len(rows) * (cell_h + gap)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(cols):
        draw.text((left + col * (cell_w + gap) + 4, 12), label, fill="black")

    for row, (name, meta) in enumerate(RUNS.items()):
        run_dir = meta["dir"]
        draw.text((6, top + row * (cell_h + gap) + 8), name, fill="black")
        paths = [run_dir / "step0_original.png"] + [
            run_dir / f"naive_step{i}.png" for i in range(1, 6)
        ]
        for col, path in enumerate(paths):
            panel = Image.open(path).convert("RGB").crop(crop)
            panel = panel.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = left + col * (cell_w + gap)
            y = top + row * (cell_h + gap)
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline="#888")
    canvas.save(OUT_GRID)


def main() -> None:
    receipt = build_receipt()
    make_grid()
    OUT_JSON.write_text(json.dumps(metrics.json_safe(receipt), indent=2), encoding="utf-8")
    print("verified the same saved base, instructions and supports across four naive chains")
    for name, row in receipt["editors"].items():
        print(
            f"  {name:18} t1 exact={row['naive_label_bit_exact_pct'][0]:.3f}%  "
            f"SSIM {row['naive_label_ssim'][0]:.3f}->{row['naive_label_ssim'][-1]:.3f}  "
            f"gate={row['gate_v1_commits'] if row['ledger_complete'] else 'incomplete'}"
        )
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_GRID}")


if __name__ == "__main__":
    main()
