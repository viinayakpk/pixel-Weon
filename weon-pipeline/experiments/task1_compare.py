"""Task 1, step 2: three conditions on the SAME generation. Zero additional API calls.

  A  model output only                      (what the black box produced)
  B  hard-alpha graft                       (real mark pasted, sRGB, no relight)
  C  premultiplied linear-light graft       (real mark, source shading divided out,
                                             target shading applied, 4x supersampled warp)

The target quad is DECLARED by hand, read off a coordinate grid over the original generation
(outputs/task1/p2_target_grid.png). It is never found by matching against the model's own re-drawn
mark, which is the bug the audit caught in the previous version.

Certificate honesty note. B and C use reference-derived geometry by construction, but their
rasterised diagnostics are still measured outcomes after warping and compositing. Provenance does
not certify placement, colour or photographic naturalness.

Run:  python -m experiments.task1_compare       (offline, free)
"""
from __future__ import annotations
import hashlib, json, os, sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, gate, mark, metrics          # noqa: E402
from pipeline.gate import Verdict                         # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task1")
GEN = "p2_constrained.png"
LOGO_BOX = (793, 1309, 1091, 1448)          # ARIGATO in the packshot
TARGET_BOX = (533, 771, 612, 808)           # declared by hand from p2_target_grid.png


def rectify(img: np.ndarray, box, size: tuple[int, int]) -> np.ndarray:
    """Warp a declared region to the asset's canonical frame so marks are comparable."""
    quad = mark.box_to_quad(box)
    dst = np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]])
    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, H, size, flags=cv2.INTER_LANCZOS4)


def mark_fidelity(ref_asset: np.ndarray, test_rect: np.ndarray) -> dict:
    """Compare a rectified candidate mark against the brand's real mark.

    Deliberately not a spelling check. Both generations visibly spell ARIGATO but use the wrong
    typeface; OCR was not tested. A spelling-only verdict would miss geometry and colour.
    """
    ref_rgb, ref_a = ref_asset[..., :3], ref_asset[..., 3]
    test = cv2.resize(test_rect, (ref_rgb.shape[1], ref_rgb.shape[0]),
                      interpolation=cv2.INTER_AREA)

    # geometry: binarise both sides with the IDENTICAL operator.
    # Earlier this compared the reference's stored alpha against a fresh top-hat of the test.
    # Those are different operators, so condition B — which contains the real asset by
    # construction — scored IoU 0.46 instead of ~1.0, and its 'ink ratio 1.45' was just the
    # re-binarisation fattening strokes. That measured my own pipeline, not brand fidelity.
    def strokes(x):
        lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)[..., 0]
        t = cv2.morphologyEx(lab, cv2.MORPH_TOPHAT,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        _, b = cv2.threshold(t, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (b > 0)
    a_ref, a_test = strokes(ref_rgb), strokes(test)
    inter = np.logical_and(a_ref, a_test).sum()
    union = np.logical_or(a_ref, a_test).sum()
    iou = float(inter / union) if union else float("nan")

    # Binarised foreground area. This is not a direct measurement of stroke thickness.
    ink_ref = float(100.0 * a_ref.mean())
    ink_test = float(100.0 * a_test.mean())

    # colour of the mark itself
    de = float("nan")
    if a_ref.sum() > 8 and a_test.sum() > 8:
        de = float(metrics.color_delta_e(
            np.full((1, 1, 3), ref_rgb[a_ref].mean(axis=0).round().astype(np.uint8)),
            np.full((1, 1, 3), test[a_test].mean(axis=0).round().astype(np.uint8))))

    tm = metrics.template_match_inliers(ref_rgb, test)
    return {"stroke_iou": round(iou, 4),
            "ink_pct_ref": round(ink_ref, 2), "ink_pct_test": round(ink_test, 2),
            "ink_ratio": round(ink_test / ink_ref, 3) if ink_ref else None,
            "mark_colour_delta_e": round(de, 2),
            "orb_inliers": tm["inliers"], "orb_matches": tm["matches"]}


def main() -> None:
    pack = np.asarray(Image.open(os.path.join(config.TEST_DATA, "case1_packshot.jpg")).convert("RGB"))
    gen = np.asarray(Image.open(f"{OUT}/{GEN}").convert("RGB"))

    crop = pack[LOGO_BOX[1]:LOGO_BOX[3], LOGO_BOX[0]:LOGO_BOX[2]]
    rgba = mark.extract_mark_alpha(crop)
    cleaned, comp = mark.keep_components(rgba, 20, expect=7)
    asset = mark.tight_crop(cleaned)
    Image.fromarray(asset).save(f"{OUT}/asset_arigato.png")
    print(f"asset {asset.shape} from {comp['components_total']} components "
          f"-> kept {comp['components_kept']} (expected 7: {comp['matches_expected']})")

    quad = mark.box_to_quad(TARGET_BOX)

    # Clear the incumbent mark before grafting.
    # Without this the real asset is painted ON TOP of the model's own re-drawn mark: the graft
    # only covers the letters' alpha, so the model's version stays visible around them and the
    # result is a doubled, ghosted mark (measured: stroke IoU 0.49 for a condition that
    # contains the real asset by construction). Protecting a mark means owning the whole
    # region: substrate first, then mark.
    tx0, ty0, tx1, ty1 = TARGET_BOX
    pad = 6
    rx0, ry0 = max(0, tx0 - pad), max(0, ty0 - pad)
    rx1, ry1 = min(gen.shape[1], tx1 + pad), min(gen.shape[0], ty1 + pad)
    region = gen[ry0:ry1, rx0:rx1]
    incumbent = mark.extract_mark_alpha(region)          # the model's mark, in the target area
    substrate_region = mark.estimate_substrate(region, incumbent[..., 3], radius=12, dilate=3)
    cleared = gen.copy()
    cleared[ry0:ry1, rx0:rx1] = substrate_region
    Image.fromarray(cleared).save(f"{OUT}/cond_B0_cleared_substrate.png")
    print(f"cleared incumbent mark in {(rx0, ry0, rx1, ry1)}: "
          f"{100*(incumbent[..., 3] > 0).mean():.1f}% of the region was mark")

    conditions = {
        "A_model_only": gen,
        "B_hard_alpha_graft": mark.graft(cleared, asset, quad, supersample=1, relight=False)[0],
        "C_linear_relit_graft": mark.graft(cleared, asset, quad, supersample=4, relight=True)[0],
    }

    size = (asset.shape[1], asset.shape[0])
    ref_rect = asset
    results = {"generation": GEN, "target_box": list(TARGET_BOX),
               "logo_box_in_packshot": list(LOGO_BOX),
               "asset": {k: v for k, v in comp.items() if k != "dropped_areas"},
               "conditions": {}, "figure": "rectified_compare.png"}
    rectified_panels = [("REFERENCE (brand asset)", asset[..., :3])]

    for name, img in conditions.items():
        Image.fromarray(img).save(f"{OUT}/cond_{name}.png")
        rect = rectify(img, TARGET_BOX, size)
        Image.fromarray(rect).save(f"{OUT}/cond_{name}_rectified.png")
        rectified_panels.append((name, rect))
        fid = mark_fidelity(ref_rect, rect)

        checks = {
            "mark_geometry": gate.Check(
                Verdict.PASS if fid["stroke_iou"] >= 0.35 else Verdict.FAIL,
                f"stroke IoU {fid['stroke_iou']:.3f} vs brand mark", fid["stroke_iou"]),
            "mark_ink_area": gate.Check(
                Verdict.PASS if 0.7 <= (fid["ink_ratio"] or 0) <= 1.4 else Verdict.FAIL,
                f"foreground-area ratio {fid['ink_ratio']} (1.0 = equal binarised area; not stroke thickness)",
                fid["ink_ratio"]),
            "mark_colour": gate.Check(
                Verdict.PASS if fid["mark_colour_delta_e"] <= 12 else Verdict.FAIL,
                f"mark colour dE {fid['mark_colour_delta_e']}", fid["mark_colour_delta_e"]),
            "photographic_naturalness": gate.Check(
                Verdict.UNKNOWN,
                "not evaluated in this stage; route to a separate human or calibrated specialist"),
        }
        d = gate.decide(checks)
        results["conditions"][name] = {"fidelity": fid, "certificate": d.to_json()}
        print(f"\n{name}")
        for k, c in checks.items():
            print(f"   {k:26} {c.verdict.value:8} {c.detail}")
        print(f"   -> {d.status.upper()}")

    # Compact report figure. All panels use the same rectified coordinate frame and size.
    panel_w = 320
    panel_h = round(panel_w * asset.shape[0] / asset.shape[1])
    label_h, gap = 24, 8
    canvas = Image.new(
        "RGB",
        (len(rectified_panels) * panel_w + (len(rectified_panels) - 1) * gap,
         panel_h + label_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for idx, (label, panel) in enumerate(rectified_panels):
        x = idx * (panel_w + gap)
        draw.text((x + 3, 6), label, fill="black")
        p = Image.fromarray(panel).convert("RGB").resize(
            (panel_w, panel_h), Image.Resampling.LANCZOS
        )
        canvas.paste(p, (x, label_h))
    canvas.save(f"{OUT}/rectified_compare.png")

    pinned = {
        "packshot": os.path.join(config.TEST_DATA, "case1_packshot.jpg"),
        "p2_constrained": f"{OUT}/{GEN}",
        "asset_arigato": f"{OUT}/asset_arigato.png",
        "condition_A": f"{OUT}/cond_A_model_only.png",
        "condition_B": f"{OUT}/cond_B_hard_alpha_graft.png",
        "condition_C": f"{OUT}/cond_C_linear_relit_graft.png",
        "rectified_compare": f"{OUT}/rectified_compare.png",
        "task1_compare_source": __file__,
        "mark_source": mark.__file__,
        "metrics_source": metrics.__file__,
    }
    results["inputs_and_code_sha256"] = {
        name: hashlib.sha256(open(path, "rb").read()).hexdigest()
        for name, path in pinned.items()
    }

    with open(f"{OUT}/task1_compare.json", "w") as fh:
        json.dump(metrics.json_safe(results), fh, indent=2)
    print(f"\nwrote {OUT}/task1_compare.json")


if __name__ == "__main__":
    main()
