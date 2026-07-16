"""Task 1, direction 2: is a tonal mark's signal RELIEF rather than ALBEDO?

This tests the hypothesis the material probe left open. The claim is not that the extractor
fails on tonal embroidery — measured, it does not; the alpha is legibly BEYOND NORDIC, because
relief creates luminance contrast through shadow. The claim is about what the extracted RGB
MEANS, and therefore whether warping it into a new scene is valid at all.

Physical prediction. Under a single directional light:

  ALBEDO mark (gold foil on brown suede)
      the mark reflects differently from its substrate everywhere it exists.
      -> signed residual (mark - local substrate) is UNIPOLAR: almost all positive.

  RELIEF mark (green thread on green fabric)
      thread and fabric share an albedo. The mark is visible only because raised thread catches
      light on the side facing the source and casts shadow on the opposite side.
      -> signed residual is BIPOLAR: a bright lobe and a dark lobe, paired across each stroke,
         and their arrangement encodes the packshot's light direction.

If the tonal mark is bipolar, then its extracted RGB is not albedo — it is baked-in shading.
Warping that asset into a scene lit from elsewhere transplants a shadow pattern pointing the
wrong way, which no amount of colour correction repairs. That is the category error, and it
lives in rendering, not extraction.

Run:  python -m experiments.relief_probe      (offline, free)
"""
from __future__ import annotations
import json, os, sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, mark, metrics       # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task1")
os.makedirs(OUT, exist_ok=True)

MARKS = {
    "arigato": ("case1_packshot.jpg", (793, 1309, 1091, 1448), "foil/print on suede"),
    "beyond_nordic": ("case1_lower.jpg", (578, 1457, 1156, 1763), "tonal embroidery"),
}


def signed_residual(crop: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Luminance of the mark minus the luminance of the fabric it sits on.

    The substrate is estimated by inpainting the (dilated) mark away, so the comparison is
    against what the fabric would look like there, not against a global average.
    """
    sub = mark.estimate_substrate(crop, alpha, radius=12, dilate=3)
    lum = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    return lum(crop) - lum(sub), (alpha > 0)


def polarity(res: np.ndarray, m: np.ndarray) -> dict:
    """Is the mark's signal one-sided (albedo) or two-sided (relief)?"""
    v = res[m]
    if v.size < 32:
        return {}
    pos, neg = v[v > 2.0], v[v < -2.0]
    # a bipolar signal has substantial mass on BOTH sides and a near-zero mean despite
    # large absolute deviations
    return {
        "pct_brighter": round(float(100.0 * len(pos) / len(v)), 2),
        "pct_darker": round(float(100.0 * len(neg) / len(v)), 2),
        "mean_residual": round(float(v.mean()), 3),
        "mean_abs_residual": round(float(np.abs(v).mean()), 3),
        # ~1.0 => purely one-sided (albedo). ~0.0 => balanced light/dark lobes (relief).
        "unipolarity": round(float(abs(v.mean()) / (np.abs(v).mean() + 1e-6)), 3),
        "dark_to_bright_ratio": round(float(len(neg) / max(len(pos), 1)), 3),
    }


def light_direction(res: np.ndarray, m: np.ndarray) -> dict:
    """If the signal is relief, the bright->dark transition across strokes has one dominant
    direction: the packshot's light. Estimated from the residual's gradient inside the mark."""
    gx = cv2.Sobel(res, cv2.CV_32F, 1, 0, ksize=3)[m]
    gy = cv2.Sobel(res, cv2.CV_32F, 0, 1, ksize=3)[m]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    keep = mag > np.percentile(mag, 75)          # strongest transitions only
    if keep.sum() < 16:
        return {}
    ang = np.arctan2(gy[keep], gx[keep])
    # circular mean over the FULL circle: relief has a signed direction (bright->dark)
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    return {
        "dominant_angle_deg": round(float(np.degrees(np.arctan2(s, c))), 1),
        # 1.0 => every transition agrees (one light). 0.0 => no preferred direction.
        "directional_coherence": round(float(np.sqrt(c ** 2 + s ** 2)), 3),
    }


def main() -> None:
    results = {}
    for name, (fn, box, material) in MARKS.items():
        img = np.asarray(Image.open(os.path.join(config.TEST_DATA, fn)).convert("RGB"))
        crop = img[box[1]:box[3], box[0]:box[2]]
        rgba = mark.extract_mark_alpha(crop)
        cleaned, _ = mark.keep_components(rgba, min_area=20)
        res, m = signed_residual(crop, cleaned[..., 3])

        pol = polarity(res, m)
        ld = light_direction(res, m)
        results[name] = {"material": material, "polarity": pol, "light": ld}

        # visualise the signed residual: red = brighter than fabric, blue = darker
        vis = np.zeros((*res.shape, 3), np.uint8)
        s = np.clip(res / 12.0, -1, 1)
        vis[..., 0] = (np.clip(s, 0, 1) * 255).astype(np.uint8)
        vis[..., 2] = (np.clip(-s, 0, 1) * 255).astype(np.uint8)
        vis[~m] = 30
        Image.fromarray(vis).save(f"{OUT}/{name}_residual.png")

        print(f"{name:14} ({material})")
        print(f"   brighter than fabric : {pol['pct_brighter']:5.1f}%")
        print(f"   darker than fabric   : {pol['pct_darker']:5.1f}%")
        print(f"   dark:bright ratio    : {pol['dark_to_bright_ratio']:5.3f}")
        print(f"   unipolarity          : {pol['unipolarity']:5.3f}   (1=albedo, 0=relief)")
        print(f"   light coherence      : {ld.get('directional_coherence', float('nan')):5.3f} "
              f"@ {ld.get('dominant_angle_deg', float('nan')):6.1f} deg")
        print()

    a, b = results["arigato"], results["beyond_nordic"]
    results["verdict"] = {
        "unipolarity_foil": a["polarity"]["unipolarity"],
        "unipolarity_tonal": b["polarity"]["unipolarity"],
        "dark_ratio_foil": a["polarity"]["dark_to_bright_ratio"],
        "dark_ratio_tonal": b["polarity"]["dark_to_bright_ratio"],
        "interpretation": (
            "If the foil mark is unipolar (mostly brighter than its substrate) and the tonal "
            "mark is bipolar (paired bright and dark lobes with a coherent direction), then "
            "the tonal mark's pixels encode the packshot's LIGHT, not the brand's COLOUR. "
            "Warping those pixels into a differently-lit scene transplants a shadow pattern "
            "oriented to the wrong light source."
        ),
    }
    with open(f"{OUT}/relief_probe.json", "w") as fh:
        json.dump(metrics.json_safe(results), fh, indent=2)
    print(f"wrote {OUT}/relief_probe.json")


if __name__ == "__main__":
    main()
