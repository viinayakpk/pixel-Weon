"""Task 1, worn case: does the deterministic repair transfer from a product shot to a worn shot?

The product-shot experiment showed a repair improving stroke IoU 0.213 -> 0.670. That was an
isolated sneaker on concrete. The brief's product is a model WEARING the garment, which changes
pose, scale, lighting and occlusion all at once. If the mechanism only works on the easy case it
is not a mechanism, it is a demo.

Conditions on the same worn generation, zero additional API calls:
  A  model output only
  B  clear the incumbent mark, graft the reference-derived asset onto a declared quad

The quad is read by hand off a coordinate grid (_grid_a2_upper.png), never located by matching the
model's own re-drawn mark.

Honest framing: B contains reference-DERIVED geometry, not a pixel copy. Warping makes it 0%
bit-exact against the reference.

Run:  python -m experiments.task1_worn_compare      (offline, free)
"""
from __future__ import annotations
import hashlib, json, os, sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cases.task1_worn import MANIFEST, PACKSHOT                       # noqa: E402
from pipeline import config, gate, mark, metrics                       # noqa: E402
from pipeline.gate import Verdict                                      # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task1_worn")
GEN = "a2_with_markref.png"
LOGO_BOX = tuple(MANIFEST[0]["packshot_box"])
TARGET_BOX = (488, 988, 562, 1024)      # declared by hand from _grid_a2_upper.png


def strokes(x: np.ndarray) -> np.ndarray:
    """Identical operator on both sides — comparing a stored alpha against a fresh re-binarisation
    measures the operator, not the mark."""
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)[..., 0]
    t = cv2.morphologyEx(lab, cv2.MORPH_TOPHAT,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    _, b = cv2.threshold(t, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b > 0


def rectify(img: np.ndarray, box, size) -> np.ndarray:
    quad = mark.box_to_quad(box)
    dst = np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), size,
                               flags=cv2.INTER_LANCZOS4)


def fidelity(ref_rgb: np.ndarray, test_rect: np.ndarray) -> dict:
    test = cv2.resize(test_rect, (ref_rgb.shape[1], ref_rgb.shape[0]), interpolation=cv2.INTER_AREA)
    a_ref, a_test = strokes(ref_rgb), strokes(test)
    inter, union = np.logical_and(a_ref, a_test).sum(), np.logical_or(a_ref, a_test).sum()
    de = float("nan")
    if a_ref.sum() > 8 and a_test.sum() > 8:
        de = float(metrics.color_delta_e(
            np.full((1, 1, 3), ref_rgb[a_ref].mean(axis=0).round().astype(np.uint8)),
            np.full((1, 1, 3), test[a_test].mean(axis=0).round().astype(np.uint8))))
    return {"stroke_iou": round(float(inter / union), 4) if union else None,
            "ink_pct_ref": round(float(100 * a_ref.mean()), 2),
            "ink_pct_test": round(float(100 * a_test.mean()), 2),
            "ink_area_ratio": round(float(a_test.mean() / a_ref.mean()), 3) if a_ref.mean() else None,
            "mark_colour_delta_e": round(de, 2)}


def main() -> None:
    pack = np.asarray(Image.open(os.path.join(config.TEST_DATA, PACKSHOT)).convert("RGB"))
    gen = np.asarray(Image.open(f"{OUT}/{GEN}").convert("RGB"))

    crop = pack[LOGO_BOX[1]:LOGO_BOX[3], LOGO_BOX[0]:LOGO_BOX[2]]
    cleaned, comp = mark.keep_components(mark.extract_mark_alpha(crop), 20, expect=7)
    asset = mark.tight_crop(cleaned)
    Image.fromarray(asset).save(f"{OUT}/asset_arigato.png")
    print(f"asset {asset.shape}  components {comp['components_total']} -> kept "
          f"{comp['components_kept']} (expected 7: {comp['matches_expected']})")

    # clear the incumbent first: grafting over it leaves a doubled mark (measured on the product
    # shot: IoU 0.487 for a condition that contains the real asset)
    tx0, ty0, tx1, ty1 = TARGET_BOX
    pad = 6
    r = (max(0, tx0 - pad), max(0, ty0 - pad), min(gen.shape[1], tx1 + pad), min(gen.shape[0], ty1 + pad))
    region = gen[r[1]:r[3], r[0]:r[2]]
    incumbent = mark.extract_mark_alpha(region)
    cleared = gen.copy()
    cleared[r[1]:r[3], r[0]:r[2]] = mark.estimate_substrate(region, incumbent[..., 3], 12, 3)
    print(f"cleared incumbent in {r}: {100*(incumbent[...,3]>0).mean():.1f}% of region was mark")

    quad = mark.box_to_quad(TARGET_BOX)
    conds = {
        "A_model_only": gen,
        "B_cleared_graft": mark.graft(cleared, asset, quad, supersample=4, relight=False)[0],
    }

    size = (asset.shape[1], asset.shape[0])
    res = {"case": "task1_worn", "generation": GEN,
           "generation_sha12": hashlib.sha256(gen.tobytes()).hexdigest()[:12],
           "target_box": list(TARGET_BOX), "logo_box_in_packshot": list(LOGO_BOX),
           "quad_declared_by": "hand, from outputs/task1_worn/_grid_a2_upper.png",
           "conditions": {}}

    for name, img in conds.items():
        Image.fromarray(img).save(f"{OUT}/cond_{name}.png")
        rect = rectify(img, TARGET_BOX, size)
        Image.fromarray(rect).save(f"{OUT}/cond_{name}_rectified.png")
        fid = fidelity(asset[..., :3], rect)
        checks = {
            "mark_geometry": gate.Check(
                Verdict.PASS if (fid["stroke_iou"] or 0) >= 0.35 else Verdict.FAIL,
                f"stroke IoU {fid['stroke_iou']} vs the brand mark", fid["stroke_iou"]),
            "mark_colour": gate.Check(
                Verdict.PASS if fid["mark_colour_delta_e"] <= 12 else Verdict.FAIL,
                f"mark colour dE {fid['mark_colour_delta_e']}", fid["mark_colour_delta_e"]),
            "photographic_naturalness": gate.Check(
                Verdict.UNKNOWN, "no calibrated specialist; routes to human review"),
            "midsole_instance": gate.Check(
                Verdict.UNKNOWN, "relief mark: present by eye, but no working automatic detector"),
        }
        d = gate.decide(checks)
        res["conditions"][name] = {"fidelity": fid, "certificate": d.to_json()}
        print(f"\n{name}")
        for k, c in checks.items():
            print(f"   {k:26} {c.verdict.value:8} {c.detail}")
        print(f"   -> {d.status.upper()}")

    a = res["conditions"]["A_model_only"]["fidelity"]["stroke_iou"]
    b = res["conditions"]["B_cleared_graft"]["fidelity"]["stroke_iou"]
    res["transfer"] = {
        "product_shot_result": {"A": 0.2127, "B": 0.6705},
        "worn_result": {"A": a, "B": b},
        "question": "does the deterministic repair transfer from a product shot to a worn shot?",
        "answer": ("improves" if (b or 0) > (a or 0) else "does NOT improve") +
                  f" on the worn case: {a} -> {b}",
        "caveats": ("One worn generation, one hand-declared quad, no replicates. The worn mark is "
                    "~74x36 px versus ~79x37 in the product shot, so this is not a resolution "
                    "advantage. B is reference-derived, not a pixel copy."),
    }
    with open(f"{OUT}/worn_compare.json", "w") as fh:
        json.dump(metrics.json_safe(res), fh, indent=2)
    print(f"\nTRANSFER: {res['transfer']['answer']}")
    print(f"  (product shot was 0.2127 -> 0.6705)")
    print(f"wrote {OUT}/worn_compare.json")


if __name__ == "__main__":
    main()
