"""The regression that the single-logo metric could not see.

The packshot carries TWO legitimate ARIGATO marks, in two different material modes:

    upper    gold on suede      ALBEDO  — the mark this pipeline extracts and repairs
    midsole  debossed in rubber RELIEF  — same colour as its substrate, legible only by shadow

`task1_compare.py` declares ONE logo box. So the certificate measures the upper mark, improves it
(stroke IoU 0.213 -> 0.670), and is structurally incapable of noticing anything happening to the
other one.

Something did happen to the other one. The "constrained" configuration in task1_generate.py —
written to protect the brand — contains the phrase "no other text" and the saved result lacks the
midsole mark. The plain configuration kept it. Because this is one stochastic sample and the
configuration also changes the reference inputs, the phrase is a plausible contributor, not an
isolated cause.

    packshot            2 marks
    p1_plain            2 marks
    p2_constrained      1 mark   <- the midsole mark is not visibly retained
    A / B / C           1 mark   (all derive from p2)

The local metric improved while product-level completeness remained worse than the plain
generation. Together with Task 4's overly broad support, this demonstrates declaration risk: a
pipeline can only preserve and evaluate constraints it has represented.

Run:  python -m experiments.identity_manifest      (offline, free)
"""
from __future__ import annotations
import hashlib, json, os, sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, metrics       # noqa: E402

OUT = os.path.join(config.OUTPUTS, "task1")

# Post-hoc two-wordmark inventory created after the missing midsole mark was observed. The boxes
# are then declared against the packshot for a reproducible audit. A production manifest should be
# built before generation and cover all identity-bearing attributes, not only these two marks.
MANIFEST = [
    {"id": "upper_arigato", "material_mode": "albedo (gold on suede)", "expected": 1,
     "packshot_box": [793, 1309, 1091, 1448],
     "note": "the mark this pipeline extracts, warps and repairs"},
    {"id": "midsole_arigato", "material_mode": "relief (debossed rubber, same colour)", "expected": 1,
     "packshot_box": [200, 1660, 700, 1810],
     "note": "same colour as its substrate; legible through shadow; absent from the first local certificate"},
]

# Bands to search per image (the shoe sits differently in each generation, so search a wide
# band rather than a guessed box — a guessed box already produced one wrong reading here).
BANDS = {
    "packshot":       ("case1_packshot.jpg", (150, 1650, 900, 1900), (700, 1250, 1150, 1500)),
    "p1_plain":       ("p1_plain.png",       (60, 820, 1000, 1010),  (450, 660, 700, 740)),
    "p2_constrained": ("p2_constrained.png", (60, 820, 1000, 1010),  (500, 760, 660, 820)),
    "cond_B":         ("cond_B_hard_alpha_graft.png", (60, 820, 1000, 1010), (500, 760, 660, 820)),
}

# Tighter boxes used only to make the evidence figure legible. They are declared against the
# source images, not inferred from generated pixels.
DISPLAY_BOXES = {
    "packshot":       {"midsole": (200, 1600, 700, 1830), "upper": (720, 1220, 1160, 1510)},
    "p1_plain":       {"midsole": (120, 800, 350, 980),   "upper": (480, 630, 710, 800)},
    "p2_constrained": {"midsole": (100, 840, 360, 1040),  "upper": (480, 700, 690, 870)},
    "cond_B":         {"midsole": (100, 840, 360, 1040),  "upper": (480, 700, 690, 870)},
}


def _fit(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Aspect-preserving fit on white, used only for the evidence contact sheet."""
    out = Image.new("RGB", size, "white")
    x = im.copy()
    x.thumbnail(size, Image.Resampling.LANCZOS)
    out.paste(x, ((size[0] - x.width) // 2, (size[1] - x.height) // 2))
    return out


def make_grid(images: dict[str, Image.Image]) -> None:
    """Full-product context plus both protected instances at readable scale."""
    names = ["packshot", "p1_plain", "p2_constrained", "cond_B"]
    titles = {
        "packshot": "PACKSHOT (2/2 marks)",
        "p1_plain": "P1 plain (2/2; upper geometry drifts)",
        "p2_constrained": "P2 constrained (1/2; midsole missing)",
        "cond_B": "B local repair (1/2; upper improved)",
    }
    cw, full_h, crop_h = 340, 470, 135
    canvas = Image.new("RGB", (cw * 4, 42 + full_h + 2 * (crop_h + 28)), "white")
    d = ImageDraw.Draw(canvas)
    d.text((6, 5), "GREEN = upper ARIGATO (albedo)    RED = midsole ARIGATO (deboss/relief)",
           fill="black")

    for col, name in enumerate(names):
        src = images[name]
        boxes = DISPLAY_BOXES[name]
        marked = src.copy()
        md = ImageDraw.Draw(marked)
        md.rectangle(boxes["upper"], outline="#00c000", width=max(3, src.width // 350))
        md.rectangle(boxes["midsole"], outline="#ff2020", width=max(3, src.width // 350))
        x0 = col * cw
        canvas.paste(_fit(marked, (cw, full_h)), (x0, 42))
        d.text((x0 + 5, 25), titles[name], fill="#b00000" if "1/2" in titles[name] else "black")

        for row, (kind, colour) in enumerate((("upper", "#008000"), ("midsole", "#c00000"))):
            crop = src.crop(boxes[kind])
            y = 42 + full_h + row * (crop_h + 28)
            canvas.paste(_fit(crop, (cw, crop_h)), (x0, y + 22))
            d.text((x0 + 5, y + 4), f"{kind} instance", fill=colour)

    canvas.save(f"{OUT}/identity_manifest_grid.png")


def mark_energy(crop: np.ndarray) -> float:
    """ATTEMPTED presence indicator. IT DOES NOT WORK — see `detector_failed` in the output.

    Intended: local high-frequency ink energy (top-hat + black-hat, so relief registers too).
    Measured: 18.12 for the p2 midsole band, which is visibly BLANK, versus 16.35 for the p1 band,
    which visibly reads ARIGATO. The band is dominated by tread pattern and the gum-sole edge, so
    this scores substrate texture, not mark presence. Retained only so the failure is auditable.
    """
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[..., 0]
    top = cv2.morphologyEx(lab, cv2.MORPH_TOPHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bot = cv2.morphologyEx(lab, cv2.MORPH_BLACKHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return round(float(np.maximum(top, bot).mean()), 3)


def main() -> None:
    rows = []
    source_images: dict[str, Image.Image] = {}
    for name, (fn, midsole_band, upper_band) in BANDS.items():
        path = os.path.join(config.TEST_DATA, fn) if name == "packshot" else f"{OUT}/{fn}"
        pil = Image.open(path).convert("RGB")
        source_images[name] = pil.copy()
        img = np.asarray(pil)
        rec = {"image": name}
        for label, band in (("midsole_arigato", midsole_band), ("upper_arigato", upper_band)):
            x0, y0, x1, y1 = band
            crop = img[y0:min(y1, img.shape[0]), x0:min(x1, img.shape[1])]
            rec[label] = {"band": list(band), "mark_energy": mark_energy(crop)}
            Image.fromarray(crop).save(f"{OUT}/manifest_{name}_{label}.png")
        rows.append(rec)
        print(f"  {name:16} midsole_energy={rec['midsole_arigato']['mark_energy']:6.2f}   "
              f"upper_energy={rec['upper_arigato']['mark_energy']:6.2f}")

    make_grid(source_images)
    out = {
        "manifest": MANIFEST,
        "timing": ("Post-hoc audit built after observing the missing midsole mark; proposed as a "
                   "pre-generation production step, not presented as a prospective experiment."),
        "inputs_sha256": {
            name: hashlib.sha256(source_images[name].tobytes()).hexdigest()
            for name in source_images
        },
        "why": ("The certificate declared one logo box. The product has two marks. This inventories "
                "both and reports what each image actually contains."),
        "detector_failed": {
            "what": "mark_energy was intended to indicate mark presence/absence per band.",
            "evidence_it_failed": ("p2_constrained midsole scores 18.12 while being visibly blank; "
                                   "p1_plain midsole scores 16.35 while visibly reading ARIGATO. "
                                   "Higher score for the absent mark. The band is dominated by "
                                   "tread pattern and the gum-sole edge."),
            "consequence": ("These numbers are NOT used as evidence. The finding below rests on "
                            "direct visual inspection of the saved bands, which the reader can "
                            "verify: manifest_p1_plain_midsole_arigato.png reads ARIGATO; "
                            "manifest_p2_constrained_midsole_arigato.png is blank across its "
                            "full width."),
            "pattern": ("Multiple attempted mark detectors responded to stitching or tread "
                        "texture instead of the wordmark. In this case, detection and grounding "
                        "are recurring bottlenecks rather than solved components."),
        },
        "measurements_not_evidence": rows,
        "finding": (
            "The packshot and the PLAIN generation both retain two marks. The CONSTRAINED "
            "generation retains one: the debossed midsole ARIGATO is not visibly retained, and conditions A/B/C "
            "inherit that loss because they are all built from it. Verified visually — the "
            "midsole band is blank across its full width in p2 (manifest_p2_constrained_"
            "midsole_arigato.png) and legibly reads ARIGATO in p1."
        ),
        "possible_contributor_not_causal_proof": (
            "The constrained prompt contains 'no other text', added to stop spurious lettering; "
            "the corresponding output lacks a legitimate second mark while the plain output keeps "
            "it. The constrained run also changes its reference inputs, and both are single "
            "stochastic samples, so this comparison does not isolate the phrase as the cause."
        ),
        "consequence": (
            "Local mark fidelity improved (stroke IoU 0.213 -> 0.670 on the upper mark) while "
            "product-level instance completeness remains worse than the plain generation. The "
            "local metric could not detect this because its inventory declared one instance. "
            "Per-region optimisation therefore needs a prior identity-instance inventory."
        ),
        "connection_to_task4": (
            "Both tasks expose declaration risk. Task 4 used an overly broad write support; "
            "Task 1 declared one instance for a product with two marks. A pipeline can only "
            "preserve and evaluate the constraints it declares."
        ),
        "material_note": (
            "The two marks are different material modes on the SAME product: the upper is albedo "
            "(gold on suede), the midsole is relief (debossed rubber, no colour difference). The "
            "albedo/relief distinction measured on Beyond Nordic therefore motivates, but does "
            "not validate, material-aware representations."
        ),
        "limitation": (
            "mark_energy failed as a presence indicator and is retained only as a negative "
            "result. Presence/absence was determined by direct inspection of the saved crops."
        ),
    }
    with open(f"{OUT}/identity_manifest.json", "w") as fh:
        json.dump(metrics.json_safe(out), fh, indent=2)
    print(f"\n{out['finding']}\n")
    print(f"POSSIBLE CONTRIBUTOR: {out['possible_contributor_not_causal_proof']}")
    print(f"\nwrote {OUT}/identity_manifest.json")


if __name__ == "__main__":
    main()
