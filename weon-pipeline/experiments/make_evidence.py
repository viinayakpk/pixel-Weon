"""Turn outputs/task4/metrics.json + frames into the evidence artifacts.

Produces:
  task4_grid.png        full frames, naive (top) vs ledger (bottom), per turn
  task4_label_grid.png  close-up of the PROTECTED label at each turn — the money shot
  task4_curve.png       preservation vs turn count, both strategies
  task4_table.md        the numbers, honestly labelled

Run:  python -m experiments.make_evidence
"""
from __future__ import annotations
import json, os, sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config                                    # noqa: E402
from cases.task4_jacket import PROTECTED                       # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task4")
LBOX = PROTECTED["brand_label"]


def _load(name: str) -> Image.Image | None:
    p = os.path.join(OUT, name)
    return Image.open(p).convert("RGB") if os.path.exists(p) else None


def _label_grid(n: int) -> None:
    """Close-up of the untouched brand label at every turn. If chaining degrades it and the
    ledger doesn't, it is visible here without any metric."""
    ref = _load("step0_original.png")
    if ref is None:
        return
    x0, y0, x1, y1 = LBOX
    w, h = (x1 - x0) * 3, (y1 - y0) * 3
    cols = n + 1
    canvas = Image.new("RGB", (cols * w, 2 * h + 30), "white")
    d = ImageDraw.Draw(canvas)

    for row, strat in enumerate(("naive", "ledger")):
        for i in range(cols):
            im = ref if i == 0 else _load(f"{strat}_step{i}.png")
            if im is None:
                continue
            crop = im.crop(LBOX).resize((w, h), Image.NEAREST)
            canvas.paste(crop, (i * w, 30 + row * h))
        d.text((4, 30 + row * h + 4), strat, fill="red")
    for i in range(cols):
        d.text((i * w + 4, 8), "original" if i == 0 else f"turn {i}", fill="black")
    canvas.save(os.path.join(OUT, "task4_label_grid.png"))
    print("wrote task4_label_grid.png")


def _full_grid(n: int, tw: int = 260) -> None:
    ref = _load("step0_original.png")
    if ref is None:
        return
    th = int(tw * ref.height / ref.width)
    cols = n + 1
    canvas = Image.new("RGB", (cols * tw, 2 * th + 30), "white")
    d = ImageDraw.Draw(canvas)
    for row, strat in enumerate(("naive", "ledger")):
        for i in range(cols):
            im = ref if i == 0 else _load(f"{strat}_step{i}.png")
            if im is None:
                continue
            canvas.paste(im.resize((tw, th), Image.LANCZOS), (i * tw, 30 + row * th))
        d.text((4, 30 + row * th + 4), strat, fill="red")
    for i in range(cols):
        d.text((i * tw + 4, 8), "original" if i == 0 else f"turn {i}", fill="black")
    canvas.save(os.path.join(OUT, "task4_grid.png"))
    print("wrote task4_grid.png")


def _curve(res: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping curve")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    series = [
        ("outside_union_bit_exact_pct", "Untouched pixels byte-identical (%)", (-2, 102)),
        ("label_ssim", "Protected label SSIM vs original", (0, 1.02)),
        ("outside_union_ssim", "Untouched-region SSIM vs original", (0, 1.02)),
    ]
    for ax, (key, title, ylim) in zip(axes, series):
        for strat, style in (("naive", "o--"), ("ledger", "s-")):
            rows = [r for r in res.get(strat, []) if r.get("probe")]
            if not rows:
                continue
            xs = list(range(1, len(rows) + 1))
            ys = [r["probe"][key] for r in rows]
            ax.plot(xs, ys, style, label=strat, linewidth=2, markersize=6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("edit turn")
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Task 4 — off-target degradation across a 5-edit chain "
                 "(measured outside the declared intent masks)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "task4_curve.png"), dpi=130)
    print("wrote task4_curve.png")


def _table(res: dict) -> None:
    lines = ["# Task 4 — naive chaining vs pixel edit ledger", "",
             f"editor: `{res.get('editor')}` · provider: `{res.get('provider')}` · "
             f"wall clock: {res.get('wall_clock_s')}s · "
             f"**estimated** cost: ${res.get('cost_usd_total')} "
             f"(configured per-image price × calls — not provider billing)", "",
             "n = 1 case, 1 editor, no replicates. The editor is non-deterministic "
             "(no true seed), so these are single observations without error bars.", "",
             "Off-target damage measured against the ORIGINAL, outside the union of all "
             "declared intent masks. The brand label was never targeted by any instruction.", "",
             "| turn | strategy | status | label byte-identical | label SSIM | untouched byte-identical | untouched SSIM | untouched PSNR |",
             "|---|---|---|---|---|---|---|---|"]
    for strat in ("naive", "ledger"):
        for i, r in enumerate(res.get(strat, []), 1):
            p = r.get("probe")
            if not p:
                lines.append(f"| {i} | {strat} | {r.get('status')} | N/A | N/A | N/A | N/A | N/A |")
                continue
            # PSNR is null in the JSON when the region is a perfect match (inf is not
            # strict JSON). Render that as the exact match it represents.
            psnr = p.get("outside_union_psnr")
            psnr_s = "perfect (0 error)" if psnr is None else f"{psnr:.1f} dB"
            lines.append(
                f"| {i} | {strat} | {r.get('status')} | {p['label_bit_exact_pct']:.2f}% | "
                f"{p['label_ssim']:.4f} | {p['outside_union_bit_exact_pct']:.2f}% | "
                f"{p['outside_union_ssim']:.4f} | {psnr_s} |")
    lines += ["", "### Reading this table", "",
              "- The ledger's preservation columns are **100% by construction**: pixels outside "
              "the declared mask are copied forward, not regenerated. That is a design "
              "guarantee, not an empirical discovery. The measured claim is the *naive* column, "
              "plus whether target-region pixels changed. Pixel movement is not semantic success.",
              "- `target_roi_change` is mean |Δ| in the declared box. It shows an edit "
              "*happened*; it does not show the edit was *correct*.",
              "- Cost is **estimated** from a configured per-image price, not from provider "
              "billing.",
              "- No OCR column: pytesseract is unavailable here and returns '' even on the "
              "pristine original, so it would have measured nothing.",
              ]
    with open(os.path.join(OUT, "task4_table.md"), "w", encoding="utf8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote task4_table.md")


def main() -> None:
    with open(os.path.join(OUT, "metrics.json")) as fh:
        res = json.load(fh)
    n = len(res.get("edits", []))
    _full_grid(n)
    _label_grid(n)
    _curve(res)
    _table(res)


if __name__ == "__main__":
    main()
