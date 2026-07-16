"""Task 1 exploratory prompt/reference ladder (not headline evidence).

The saved run reused p1/p2 and generated two additional configurations. Its attempted automatic
wordmark locator selected stitching, so no quantitative prompt ranking or prompting ceiling is
reported. The files are retained as a negative grounding result:

  s1 plain            scene only
  s2 attributes       + the mark described (spelling, type style, colour, placement)
  s3 negatives        + explicit constraints against restyling/re-lettering
  s4 reference        + the mark supplied as a tight high-resolution reference image
  s5 vlm_spec         + a structured spec auto-derived from the packshot by a VLM

The script can make paid calls and is intentionally absent from the offline reproduction list.
Its automatic measurements are not used in the report.
"""
from __future__ import annotations
import json, os, sys, time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import clients, config, mark, metrics       # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task1")
os.makedirs(OUT, exist_ok=True)
EDITOR = os.getenv("WEON_EDITOR", "gpt-image-2")
LOGO_BOX = (793, 1309, 1091, 1448)

SCENE = ("Editorial product photograph of this exact sneaker on a smooth concrete surface, "
         "soft directional daylight from the left, shallow depth of field, side view.")

ATTRS = (" The side of the shoe carries the brand mark ARIGATO in gold, seven capital letters, "
         "a fine-stroked serif typeface with thin elegant strokes, small, positioned below the "
         "lace holes and following the curve of the panel.")

NEGATIVES = (" Do not restyle, re-letter, re-space, embolden or reinterpret the mark. Do not "
             "substitute a different typeface. The strokes must stay thin, not heavy or slab.")

REF_NOTE = " The second reference image is the exact brand mark. Reproduce it exactly."

STRATEGIES = {
    "s1_plain":      (SCENE, False),
    "s2_attributes": (SCENE + ATTRS, False),
    "s3_negatives":  (SCENE + ATTRS + NEGATIVES, False),
    "s4_reference":  (SCENE + ATTRS + NEGATIVES + REF_NOTE, True),
    # s5 is built at runtime from a VLM-derived spec
}


def stroke_stats(crop_rgb: np.ndarray) -> dict:
    """Alignment-free description of a mark's letterforms.

    mean_stroke_width_px = 2 * mean(EDT) inside the strokes. A fine serif and a heavy slab
    differ here regardless of where either sits in the frame.
    """
    rgba = mark.extract_mark_alpha(crop_rgb)
    cleaned, comp = mark.keep_components(rgba, min_area=20)
    a = (cleaned[..., 3] > 0).astype(np.uint8)
    if a.sum() < 32:
        return {"ink_pct": 0.0, "mean_stroke_width_px": float("nan"), "components": 0}
    edt = cv2.distanceTransform(a, cv2.DIST_L2, 5)
    return {
        "ink_pct": round(float(100.0 * a.mean()), 3),
        "mean_stroke_width_px": round(float(2.0 * edt[a > 0].mean()), 3),
        "components": int(comp["components_total"]),
    }


def main() -> None:
    pack = np.asarray(Image.open(os.path.join(config.TEST_DATA, "case1_packshot.jpg")).convert("RGB"))
    x0, y0, x1, y1 = LOGO_BOX
    logo_ref = np.ascontiguousarray(pack[y0:y1, x0:x1])
    dry = config.active_provider() == "dry-run"

    # reference letterforms, normalised to roughly the scale marks appear at in a generation
    ref_small = cv2.resize(logo_ref, (116, 54), interpolation=cv2.INTER_AREA)
    ref_stats = stroke_stats(ref_small)
    print(f"reference mark: {ref_stats}\n")

    strategies = dict(STRATEGIES)

    # s5: let a VLM write the spec, rather than me hand-writing it
    spec_text = None
    if not dry:
        try:
            spec_prompt = ("Describe ONLY the gold brand mark on the side of this shoe as a "
                           "compact JSON spec for an image generator: exact text, letter count, "
                           "typeface class, stroke weight, colour, size relative to the panel, "
                           "and placement. No prose.")
            spec_text = clients.describe(spec_prompt, [logo_ref]) if hasattr(clients, "describe") else None
        except Exception as e:
            print(f"  (VLM spec unavailable: {str(e)[:80]})")
    if spec_text:
        strategies["s5_vlm_spec"] = (SCENE + " Brand mark spec: " + spec_text + NEGATIVES, True)
        print(f"VLM spec: {spec_text[:160]}\n")

    rows = []
    for name, (prompt, use_ref) in strategies.items():
        existing = {"s1_plain": "p1_plain.png", "s4_reference": "p2_constrained.png"}.get(name)
        path = f"{OUT}/ceiling_{name}.png"
        if existing and os.path.exists(f"{OUT}/{existing}"):
            # reuse the generations we already paid for
            img = np.asarray(Image.open(f"{OUT}/{existing}").convert("RGB"))
            Image.fromarray(img).save(path)
            lat, cost, note = 0.0, 0.0, f"reused {existing}"
        else:
            refs = [pack, logo_ref] if use_ref else [pack]
            t0 = time.time()
            img = clients.edit(EDITOR, prompt, refs)
            lat = round(time.time() - t0, 1)
            cost = 0.0 if dry else config.MODELS[EDITOR].price_usd
            note = "generated"
            Image.fromarray(img).save(path)
        rows.append({"strategy": name, "note": note, "latency_s": lat,
                     "cost_usd_estimated": cost, "prompt_chars": len(prompt),
                     "used_reference_image": use_ref, "image": os.path.basename(path)})
        print(f"  {name:16} {note:22} {lat:5.1f}s  {img.shape}")

    out = {"editor": EDITOR, "provider": config.active_provider(),
           "reference_mark": ref_stats, "vlm_spec": spec_text, "runs": rows,
           "cost_usd_estimated_total": round(sum(r["cost_usd_estimated"] for r in rows), 4),
           "next": "requires manual or scene-text grounding before a quantitative comparison"}
    with open(f"{OUT}/prompt_ceiling.json", "w") as fh:
        json.dump(metrics.json_safe(out), fh, indent=2)
    print(f"\nwrote {OUT}/prompt_ceiling.json  (est. ${out['cost_usd_estimated_total']})")


if __name__ == "__main__":
    main()
