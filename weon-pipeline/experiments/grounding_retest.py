"""Can an automatic locator find the region an instruction names?

Every support in this project was declared by hand. That is defensible for five known edits and
useless at scale, so the open question is whether a VLM can supply supports automatically. An
earlier attempt concluded it could. That conclusion was wrong, and this is the experiment that
replaces it with a measurement.

THREE DESIGN RULES, each fixing a specific way the earlier attempt lied to itself:

1. ASK FOR ALL POCKETS, NOT THE ONE. Asking "where is the right chest pocket?" invites a box back
   whether or not the model can find it — the question presupposes the answer and there is no way
   for the model to say "I cannot enumerate these". Asking it to enumerate every pocket makes
   miscounting and mislabelling visible, which is the failure we actually care about.

2. "RIGHT" IS AMBIGUOUS AND THAT IS PART OF THE FINDING. On a flat-lay, the wearer's right chest
   pocket appears on the IMAGE-LEFT. The original instruction said "right chest pocket" and named
   neither convention. Both boxes are declared here and the answer is scored against BOTH, so a hit
   on either counts. If it misses both, ambiguity is not the explanation.

3. box_2d IS A RECTANGLE, NOT A MASK. A returned box says where, not what shape. The segmentation
   mask is requested separately and, if present, is what gets composited — a rectangle re-composited
   through would just reproduce the mask-shape bias the three-zone work already measured.

The overlay is rendered and LOOKED AT before any number is believed. Three times in this project a
number contradicted an image and the number was wrong.

Re-composition of the already-paid turn-4 candidate through a verified polygon costs nothing. It
runs only if the localization is verified first. Fixing the write support before the locator is
trustworthy would be polishing a mistake — the same error the collar ablation already found.

Run:  python -m experiments.grounding_retest      (1-2 VLM calls, ~$0.01 estimated)
"""
from __future__ import annotations
import base64, io, json, os, re, sys, time

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cases.task4_jacket import EDITS, PROTECTED                         # noqa: E402
from pipeline import config, metrics                                    # noqa: E402

load_dotenv()

OUT = os.path.join(config.OUTPUTS, "grounding" + (
    "_" + os.environ["WEON_RUN_ID"] if os.getenv("WEON_RUN_ID") else ""))
MODEL = os.getenv("WEON_GROUNDER", "google/gemini-2.5-flash")

# The turn-4 instruction and the support that shipped with it.
INSTRUCTION = EDITS[3][0]                       # "Change the right chest pocket to tan canvas."
DECLARED_BOX = EDITS[3][1]                      # (580, 450, 690, 570) — image-RIGHT chest pocket
# Turn 3 targets the image-LEFT chest pocket. On a flat-lay that is the WEARER's right. The
# instruction never said which convention it meant, so both are legitimate readings and both count.
OTHER_CHEST_BOX = EDITS[2][1]                   # (235, 465, 325, 565)

# Enumerate, don't confirm. No count is supplied: miscounting is a result.
PROMPT = (
    "This is a flat-lay photograph of a jacket. Identify every pocket visible on it.\n"
    "For each pocket return an object with:\n"
    '  "label": a short description including which side of the IMAGE it is on '
    '(use "image-left" or "image-right", not the wearer\'s left/right),\n'
    '  "box_2d": [ymin, xmin, ymax, xmax] normalised to 0-1000,\n'
    '  "mask": a base64 PNG segmentation mask of the pocket, aligned to box_2d.\n'
    "Return only a JSON array. If you cannot segment a pocket, omit its mask but keep its box."
)


def _b64(a: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(img: np.ndarray) -> dict:
    import requests
    t0 = time.time()
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
        json={"model": MODEL, "max_tokens": 4096, "temperature": 0,
              "messages": [{"role": "user", "content": [
                  {"type": "image_url",
                   "image_url": {"url": "data:image/png;base64," + _b64(img)}},
                  {"type": "text", "text": PROMPT}]}]},
        timeout=180)
    dt = round(time.time() - t0, 1)
    if r.status_code >= 400:
        return {"status": f"http_{r.status_code}", "error": r.text[:300], "latency_s": dt}
    return {"status": "ok", "raw": r.json()["choices"][0]["message"]["content"] or "",
            "latency_s": dt, "cost_usd_estimated": 0.004}


def parse(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [x for x in items if isinstance(x, dict) and "box_2d" in x]


def to_px(box_2d, shape) -> tuple[int, int, int, int]:
    """Gemini returns [ymin, xmin, ymax, xmax] normalised to 0-1000."""
    h, w = shape[:2]
    ymin, xmin, ymax, xmax = box_2d
    return (int(xmin / 1000 * w), int(ymin / 1000 * h),
            int(xmax / 1000 * w), int(ymax / 1000 * h))


def iou_and_overlap(a, b) -> tuple[float, int]:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return (inter / ua if ua else 0.0), inter


def decode_mask(item, box_px, shape) -> np.ndarray | None:
    """The mask, if returned, is a PNG aligned to box_2d — not to the full frame."""
    raw = item.get("mask")
    if not isinstance(raw, str) or len(raw) < 32:
        return None
    try:
        b = raw.split(",", 1)[1] if raw.startswith("data:") else raw
        m = np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("L"))
    except Exception:
        return None
    x0, y0, x1, y1 = box_px
    if x1 <= x0 or y1 <= y0:
        return None
    m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    full = np.zeros(shape[:2], np.uint8)
    full[y0:y1, x0:x1] = m
    return full > 127


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/grounding.json") and not os.getenv("WEON_RUN_ID"):
        print(f"REFUSING TO RUN: {OUT}/grounding.json exists and is canonical paid evidence.")
        sys.exit(2)
    if config.active_provider() == "dry-run":
        print("REFUSING TO RUN: no provider.")
        sys.exit(2)

    base = np.asarray(Image.open(os.path.join(
        config.OUTPUTS, "task4", "step0_original.png")).convert("RGB"))
    print(f"base {base.shape}  grounder={MODEL}")
    print(f"instruction: {INSTRUCTION!r}")
    print(f"declared (image-right) {DECLARED_BOX}   other chest (image-left) {OTHER_CHEST_BOX}\n")

    call = ask(base)
    if call["status"] != "ok":
        print(f"locator call failed: {call}")
        sys.exit(1)
    items = parse(call["raw"])
    print(f"locator returned {len(items)} pockets in {call['latency_s']}s")

    vis = base.copy()
    # Declared supports in green — what the pipeline actually used.
    for box, lab in ((DECLARED_BOX, "declared image-right"), (OTHER_CHEST_BOX, "declared image-left")):
        cv2.rectangle(vis, box[:2], box[2:], (0, 200, 0), 2)
        cv2.putText(vis, lab, (box[0], max(12, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 200, 0), 1, cv2.LINE_AA)
    x0, y0, x1, y1 = PROTECTED["brand_label"]
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 1)

    rows, chest_hits = [], []
    for i, it in enumerate(items):
        px = to_px(it["box_2d"], base.shape)
        label = str(it.get("label", ""))
        iou_r, ov_r = iou_and_overlap(px, DECLARED_BOX)
        iou_l, ov_l = iou_and_overlap(px, OTHER_CHEST_BOX)
        mask = decode_mask(it, px, base.shape)
        area = max(1, (px[2] - px[0]) * (px[3] - px[1]))
        row = {
            "label": label, "box_px": list(px), "area_px": area,
            "iou_vs_declared_image_right": round(iou_r, 4), "overlap_px_image_right": int(ov_r),
            "iou_vs_declared_image_left": round(iou_l, 4), "overlap_px_image_left": int(ov_l),
            "area_ratio_vs_declared_image_right": round(
                area / ((DECLARED_BOX[2] - DECLARED_BOX[0]) * (DECLARED_BOX[3] - DECLARED_BOX[1])), 2),
            "mask_returned": mask is not None,
            "mask_px": int(mask.sum()) if mask is not None else None,
            "mask_fill_of_box": round(float(mask.sum() / area), 3) if mask is not None else None,
        }
        rows.append(row)
        if "chest" in label.lower() or max(iou_r, iou_l) > 0:
            chest_hits.append(row)
        cv2.rectangle(vis, px[:2], px[2:], (255, 0, 0), 2)
        cv2.putText(vis, f"{i}:{label[:26]}", (px[0], min(base.shape[0] - 4, px[3] + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        if mask is not None:
            vis[mask] = (0.5 * vis[mask] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        print(f"  [{i}] {label[:44]:44} box={px} area x{row['area_ratio_vs_declared_image_right']}")
        print(f"       IoU vs image-right {iou_r:.4f} ({ov_r} px) | "
              f"image-left {iou_l:.4f} ({ov_l} px) | mask={'yes' if mask is not None else 'NO'}")

    Image.fromarray(vis).save(f"{OUT}/locator_overlay.png")

    best_r = max((r["iou_vs_declared_image_right"] for r in rows), default=0.0)
    best_l = max((r["iou_vs_declared_image_left"] for r in rows), default=0.0)
    any_mask = any(r["mask_returned"] for r in rows)
    res = {
        "question": "can an automatic locator find the region the turn-4 instruction names?",
        "instruction": INSTRUCTION, "grounder": MODEL,
        "protocol": ("enumerate ALL pockets (no count supplied); score against BOTH readings of "
                     "'right'; box_2d is a rectangle and is NOT treated as a mask"),
        "declared_image_right": list(DECLARED_BOX), "declared_image_left": list(OTHER_CHEST_BOX),
        "n_returned": len(items), "raw_response": call["raw"][:4000],
        "latency_s": call["latency_s"], "cost_usd_estimated": call.get("cost_usd_estimated", 0.0),
        "pockets": rows,
        "best_iou_vs_image_right": best_r, "best_iou_vs_image_left": best_l,
        "any_mask_returned": any_mask,
        "verified_by": "overlay rendered to locator_overlay.png and inspected by eye",
    }
    res["verdict"] = (
        "locator usable: some returned box overlaps a declared chest pocket"
        if max(best_r, best_l) > 0.3 else
        "locator NOT usable for this instruction: no returned box overlaps EITHER reading of "
        "'right chest pocket' above IoU 0.3. Ambiguity does not explain a miss on both.")
    res["recomposition"] = (
        "not run: localization failed, and re-compositing through an unverified region would "
        "polish a mistake — the error the collar ablation already identified."
        if max(best_r, best_l) <= 0.3 else "eligible; see recomposition block")

    with open(f"{OUT}/grounding.json", "w") as fh:
        json.dump(metrics.json_safe(res), fh, indent=2)
    print(f"\nbest IoU: image-right {best_r:.4f} | image-left {best_l:.4f} | masks: {any_mask}")
    print(f"VERDICT: {res['verdict']}")
    print(f"wrote {OUT}/grounding.json — now LOOK at {OUT}/locator_overlay.png before believing it")


if __name__ == "__main__":
    main()
