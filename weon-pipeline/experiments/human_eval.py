"""Condition-label-hidden, counterbalanced human-evaluation harness (target: 3 raters).

Supersedes human_ab.py; old files are provenance only. What changed and why:

  * MULTI-RATER. Each rater writes their own file; nothing is overwritten. human_ab.py hardcoded
    one rater and clobbered the previous response.
  * ONE showing per pair per person. Position control now comes from counterbalancing sides
    BETWEEN raters (by rater index), not from repeating each pair to the same person. Halves the
    time and stops raters recognising pairs.
  * A REAL known-answer control. Q1 compares the brand mark against a DELIBERATELY DISTORTED copy
    of itself. The right answer is not a matter of opinion or provenance — it is the undistorted
    one. A rater who misses Q1 tells us their other answers are noise.
  * ABSOLUTE judgements, not just preferences. "B beats A" does not mean B is acceptable. Q3 asks
    accept/reject outright, with the mark enlarged next to its context, because the mark is only
    ~79x37 px in the full frame.
  * TASK 4 SEMANTICS. Q5 asks whether turn 4 satisfies "tan canvas pocket". Live Gate v1 only
    checked movement; postmortem Gate v2 returns UNKNOWN for material.

Run (per person):   python -m experiments.human_eval --rater alice
Aggregate:          python -m experiments.human_eval --report
Images only:        python -m experiments.human_eval --sheets   (to send by message)
"""
from __future__ import annotations
import hashlib, json, os, sys, datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, metrics       # noqa: E402

T1 = os.path.join(config.OUTPUTS, "task1")
T4 = os.path.join(config.OUTPUTS, "task4")
OUT = os.path.join(config.OUTPUTS, "human_eval")
os.makedirs(OUT, exist_ok=True)

MARK_BOX = (533, 771, 612, 808)          # the mark in the Task 1 generations
POCKET_BOX = (580, 450, 690, 570)        # the turn-4 declared support


def _sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


def _distorted_reference() -> np.ndarray:
    """A deliberately wrong copy of the brand mark: heavier strokes, stretched, softened.

    This is the control's wrong answer. It must be plausibly a mark, but plainly not THE mark —
    the same class of error the model makes (right text, wrong letterforms).
    """
    a = np.asarray(Image.open(f"{T1}/asset_arigato.png").convert("RGB"))
    d = cv2.dilate(a, np.ones((2, 2), np.uint8))                       # fatten the strokes
    d = cv2.resize(d, (int(a.shape[1] * 1.18), a.shape[0]))            # stretch horizontally
    d = cv2.resize(d, (a.shape[1], a.shape[0]))                        # squeeze back = distorted
    return cv2.GaussianBlur(d, (3, 3), 0)


def _panel(img: np.ndarray, label: str, h: int = 380) -> Image.Image:
    im = Image.fromarray(img).convert("RGB")
    im = im.resize((max(1, int(im.width * h / im.height)), h), Image.LANCZOS)
    c = Image.new("RGB", (im.width, h + 26), "white")
    ImageDraw.Draw(c).text((4, 6), label, fill="#c00")
    c.paste(im, (0, 26))
    return c


def _context_and_inset(path: str, box, label: str) -> np.ndarray:
    """Full frame plus an enlarged inset of the region under judgement.

    Judging integration from a full frame alone is unfair when the mark is ~79x37 px; judging
    from the inset alone removes the context that makes integration meaningful. Show both.
    """
    im = np.asarray(Image.open(path).convert("RGB")).copy()
    x0, y0, x1, y1 = box
    inset = im[y0:y1, x0:x1]
    inset = cv2.resize(inset, (inset.shape[1] * 6, inset.shape[0] * 6), interpolation=cv2.INTER_LANCZOS4)
    cv2.rectangle(im, (x0, y0), (x1, y1), (0, 220, 0), 3)
    ctx = np.asarray(Image.fromarray(im).resize((520, int(520 * im.shape[0] / im.shape[1])), Image.LANCZOS))
    H = max(ctx.shape[0], inset.shape[0])
    canvas = np.full((H, ctx.shape[1] + inset.shape[1] + 16, 3), 255, np.uint8)
    canvas[:ctx.shape[0], :ctx.shape[1]] = ctx
    canvas[:inset.shape[0], ctx.shape[1] + 16:] = inset
    return canvas


def build_questions() -> list[dict]:
    ref = np.asarray(Image.open(f"{T1}/asset_arigato.png").convert("RGB"))
    dis = _distorted_reference()
    Image.fromarray(dis).save(f"{OUT}/_distorted_reference.png")

    return [
        {"id": "q1_control", "kind": "pair", "options": ["LEFT", "RIGHT"],
         "q": "TOP is the brand's real mark. Which of LEFT / RIGHT matches it?",
         "left": ("reference", ref), "right": ("distorted", dis),
         "ref": ("THE BRAND'S REAL MARK", ref),
         "truth": "reference",
         "why": ("ATTENTION CHECK with a known answer. LEFT is pixel-identical to the reference "
                 "shown above it, so this is easy by design — it catches a rater who is clicking "
                 "randomly. It does NOT establish that they can discriminate at Q2's difficulty; "
                 "failing it discounts a rater, passing it does not validate them.")},

        {"id": "q2_fidelity", "kind": "pair", "options": ["LEFT", "RIGHT"],
         "q": "TOP is the brand's real mark. Which of LEFT / RIGHT reproduces its letterforms better?",
         "left": ("A_model", np.asarray(Image.open(f"{T1}/cond_A_model_only_rectified.png").convert("RGB"))),
         "right": ("B_graft", np.asarray(Image.open(f"{T1}/cond_B_hard_alpha_graft_rectified.png").convert("RGB"))),
         "ref": ("THE BRAND'S REAL MARK", ref),
         "truth": "B_graft",
         "why": "known by provenance: B is warped from the reference, A is the model's re-drawing"},

        {"id": "q3_b_acceptable", "kind": "absolute", "options": ["ACCEPT", "REJECT", "UNSURE"],
         "q": ("Would you ship this to a brand client? The green box (enlarged right) is the "
               "repaired mark. Judge whether it looks naturally part of the photo."),
         "image": _context_and_inset(f"{T1}/cond_B_hard_alpha_graft.png", MARK_BOX, "B"),
         "truth": None,
         "why": "absolute acceptability: 'B beats A' does not mean B is good enough"},

        {"id": "q4_b_vs_c", "kind": "pair", "options": ["LEFT", "RIGHT", "NEITHER"],
         "q": "Which looks more like an unedited photograph (less digitally pasted)?",
         "left": ("B_graft", np.asarray(Image.open(f"{T1}/cond_B_hard_alpha_graft.png").convert("RGB"))),
         "right": ("C_relit", np.asarray(Image.open(f"{T1}/cond_C_linear_relit_graft.png").convert("RGB"))),
         "ref": None, "truth": None,
         "why": "no ground truth; C failed on colour dE but that may not read as 'pasted'"},

        {"id": "q5_task4_semantic", "kind": "absolute", "options": ["YES", "NO", "UNCLEAR"],
         "q": ('The instruction was: "Change the right chest pocket to tan canvas."\n'
               "        Does the green box (enlarged right) satisfy that instruction?"),
         "image": _context_and_inset(f"{T4}/ledger_step4.png", POCKET_BOX, "turn 4"),
         "truth": None,
         "why": ("postmortem Gate v2 returns material:UNKNOWN here; live Gate v1 checked only "
                 "pixel movement and context. "
                 "The pipeline committed this.")},
    ]


def render_sheets(qs: list[dict], flip: bool) -> None:
    for i, q in enumerate(qs, 1):
        if q["kind"] == "absolute":
            img = Image.fromarray(q["image"])
            c = Image.new("RGB", (img.width, img.height + 52), "white")
            d = ImageDraw.Draw(c)
            for j, line in enumerate(q["q"].split("\n")):
                d.text((6, 6 + j * 14), line.strip(), fill="black")
            d.text((6, 36), "Answer: " + " / ".join(q["options"]), fill="#666")
            c.paste(img, (0, 52))
            c.save(f"{OUT}/Q{i}.png")
            continue
        L, R = (q["right"], q["left"]) if flip else (q["left"], q["right"])
        parts = ([_panel(q["ref"][1], q["ref"][0])] if q.get("ref") else []) + \
                [_panel(L[1], "LEFT"), _panel(R[1], "RIGHT")]
        W = max(p.width for p in parts)
        H = sum(p.height for p in parts) + 52
        c = Image.new("RGB", (max(W, 620), H), "white")
        d = ImageDraw.Draw(c)
        d.text((6, 6), q["q"], fill="black")
        d.text((6, 30), "Answer: " + " / ".join(q["options"]), fill="#666")
        y = 52
        for p in parts:
            c.paste(p, (0, y)); y += p.height
        c.save(f"{OUT}/Q{i}.png")


def collect(rater: str) -> None:
    path = f"{OUT}/{rater}.json"
    if os.path.exists(path):
        print(f"REFUSING: {path} exists. Responses are append-only; pick another --rater name.")
        sys.exit(2)

    qs = build_questions()
    # counterbalance sides BETWEEN raters, deterministically from the rater's name
    flip = int(hashlib.sha256(rater.encode()).hexdigest(), 16) % 2 == 1
    render_sheets(qs, flip)

    print(f"\nRater: {rater}   (5 questions, ~5 minutes)")
    print(f"Images are written to {OUT}/Q1.png .. Q5.png and open as you go.\n")
    print("Answer on instinct. There is no trick and no expected answer.")
    print("Use UNSURE / NEITHER / UNCLEAR freely — 'I can't tell' is a real result.\n")

    resp = []
    for i, q in enumerate(qs, 1):
        try:
            Image.open(f"{OUT}/Q{i}.png").show()
        except Exception:
            pass
        print(f"[{i}/5] {q['q']}")
        print(f"       (image: {OUT}/Q{i}.png)")
        opts = q["options"]
        while True:
            a = input(f"       {' / '.join(opts)} > ").strip().upper()
            if a in opts or a in [o[0] for o in opts]:
                break
        a = next(o for o in opts if o == a or o[0] == a)

        rec = {"id": q["id"], "answer": a, "sides_flipped": flip}
        if q["kind"] == "pair":
            L, R = (q["right"], q["left"]) if flip else (q["left"], q["right"])
            rec["chose"] = {"LEFT": L[0], "RIGHT": R[0]}.get(a)
            rec["truth"] = q["truth"]
            if q["truth"]:
                rec["correct"] = (rec["chose"] == q["truth"])
        resp.append(rec)

    evaluated = {
        "task1/asset_arigato.png": f"{T1}/asset_arigato.png",
        "task1/cond_A_model_only_rectified.png": f"{T1}/cond_A_model_only_rectified.png",
        "task1/cond_B_hard_alpha_graft_rectified.png": f"{T1}/cond_B_hard_alpha_graft_rectified.png",
        "task1/cond_B_hard_alpha_graft.png": f"{T1}/cond_B_hard_alpha_graft.png",
        "task1/cond_C_linear_relit_graft.png": f"{T1}/cond_C_linear_relit_graft.png",
        "task4/ledger_step4.png": f"{T4}/ledger_step4.png",
        **{f"human_eval/Q{i}.png": f"{OUT}/Q{i}.png" for i in range(1, 6)},
    }
    out = {"rater": rater, "when": datetime.datetime.now().isoformat(timespec="seconds"),
           "sides_flipped": flip, "responses": resp,
           "evaluated_files_sha12": {name: _sha(path) for name, path in evaluated.items()}}
    json.dump(metrics.json_safe(out), open(path, "w"), indent=2)
    print(f"\nwrote {path}")
    report()


def report() -> None:
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".json") and not f.startswith("_"))
    if not files:
        print("no raters yet — run: python -m experiments.human_eval --rater <name>")
        return
    raters = [json.load(open(f"{OUT}/{f}")) for f in files]
    agg: dict = {"n_raters": len(raters), "raters": [r["rater"] for r in raters], "questions": {}}

    for qid in [r["id"] for r in raters[0]["responses"]]:
        rows = [next(x for x in r["responses"] if x["id"] == qid) for r in raters]
        answers = [x["answer"] for x in rows]
        rec = {"answers": dict(zip(agg["raters"], answers))}
        if "correct" in rows[0]:
            n_ok = sum(1 for x in rows if x.get("correct"))
            rec["correct"] = f"{n_ok}/{len(rows)}"
            rec["chose"] = [x.get("chose") for x in rows]
        rec["unanimous"] = len(set(answers)) == 1
        agg["questions"][qid] = rec
        print(f"  {qid:20} {answers}  {'unanimous' if rec['unanimous'] else 'split'}"
              f"{'  correct ' + rec['correct'] if 'correct' in rec else ''}")

    agg["interpretation"] = {
        "q1_is_an_attention_check": (
            "q1_control shows the reference above two options, one of which IS the reference. It "
            "is easy by design and catches random clicking. Failing it discounts a rater; passing "
            "it does not establish they can discriminate at q2's difficulty."),
        "q2_is_the_real_control": (
            "q2 has an answer known by provenance (B is warped from the reference asset; A is the "
            "model's independent re-drawing). This is the question the VLM judge scored 0% on."),
        "vs_vlm_judge": ("The VLM judge scored 0% on the q2 fidelity question at every "
                         "presentation scale (outputs/task1/judge_calibration.json)."),
        "scope": (f"n={len(raters)} raters, 5 questions. A sanity check, not a study. Sides are "
                  "counterbalanced between raters; each pair is shown once per person."),
    }
    json.dump(metrics.json_safe(agg), open(f"{OUT}/_aggregate.json", "w"), indent=2)
    print(f"\nwrote {OUT}/_aggregate.json   (n={len(raters)} raters)")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif "--sheets" in sys.argv:
        render_sheets(build_questions(), flip=False)
        print(f"wrote {OUT}/Q1.png .. Q5.png  (send these; do not send this file's truth values)")
    elif "--rater" in sys.argv:
        collect(sys.argv[sys.argv.index("--rater") + 1])
    else:
        print(__doc__)
