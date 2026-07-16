"""A spelling specialist, and the four controls that decide whether to believe it.

The gate has been returning `brand_text: UNKNOWN` from the start. That is honest but inert. An
UNKNOWN is a specification: it names the specialist that is missing. This builds that one specialist
and — the actual point — establishes what it can and cannot be trusted to say.

WHY THIS IS NOT AN OCR SYSTEM. It is a probe with a known answer on four inputs. Four inputs is not
a calibration set. Nothing here licenses "our OCR works"; it licenses "on these four inputs, with
these controls, it behaved as specified". A specialist that has only ever been shown cases it passes
is not evidence, and the two controls designed to make it FAIL are worth more than the two designed
to make it pass.

OPEN TRANSCRIPTION. The expected string is never in the prompt. Asking "does this say A DAY'S
MARCH?" measures a VLM's agreeableness, not a label — it can score a perfect run on a blank crop.
The model transcribes blind; the comparison happens in code, below, where it can be read.

THE FOUR CONTROLS, and what each one falsifies:

  c1_pristine     the untouched label            -> expect PASS
                  If this fails, the specialist is broken. Cheapest possible smoke test.

  c2_nano_t5      nano-banana-pro, naive turn 5  -> expect FAIL (a real, unstaged corruption)
                  The known-answer control that matters: a specialist that never fails is a
                  rubber stamp. This input is not synthesised for this test; it is a paid
                  artefact from the Task 4 chain, and the failure it contains is spelling.

  c3_illegible    pristine, destroyed by scale   -> expect UNKNOWN
                  Tests the ABSTENTION path. A specialist that guesses under uncertainty is worse
                  than none: it converts "we cannot see" into a confident verdict, which is
                  exactly the failure this whole pipeline exists to prevent. It must say so.

  c4_typography   worn a2, model's re-lettered   -> expect spelling PASS, identity UNRESOLVED
                  ARIGATO                           The trap. Real model output that spells the
                  brand correctly in the wrong letterforms. If a PASS here were read as "the mark
                  survived", the specialist would be actively harmful — it would greenlight a
                  restyled logo. Its stroke IoU is 0.1268 (worn_compare.json): geometry rejects
                  what spelling accepts. The controls are independent and BOTH are load-bearing.

c4's expected string is ARIGATO rather than the jacket's label because it must be REAL output that
spells right and draws wrong; that artefact exists in the worn sneaker run. A synthetic render of
ARIGATO in a substitute font would test my font choice, not a model.

Every call records prompt, raw response, model id, input SHA, latency and cost, so a reader can
disagree with the interpretation without rerunning anything.

Run:  python -m experiments.spelling_specialist      (12 VLM calls, ~$0.02 estimated)
"""
from __future__ import annotations
import hashlib, json, os, re, sys, time

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cases.task1_worn import MANIFEST                                   # noqa: E402
from cases.task4_jacket import PROTECTED                                # noqa: E402
from pipeline import config, gate, metrics                              # noqa: E402
from pipeline.gate import Verdict                                       # noqa: E402

load_dotenv()   # without this the key is invisible, the provider silently reads as dry-run, and
                # every control returns UNKNOWN — a 0/4 that says nothing about the specialist

OUT = os.path.join(config.OUTPUTS, "spelling" + (
    "_" + os.environ["WEON_RUN_ID"] if os.getenv("WEON_RUN_ID") else ""))
MODEL = os.getenv("WEON_JUDGE", "google/gemini-2.5-flash")
REPEATS = 3          # temperature 0 is not a determinism guarantee; measure it instead of assuming
LABEL_BOX = PROTECTED["brand_label"]
WORN_BOX = (488, 988, 562, 1024)                # same hand-declared quad as worn_compare.py
# The quad above is tight to the strokes because stroke IoU needs it tight. Transcription needs the
# opposite: at zero margin the leading A is clipped by the box edge, and a 'RIGATO' reply would
# score FAIL while measuring my crop rather than the model. Padded here, for OCR only; the IoU in
# worn_compare.py still uses the unpadded quad.
WORN_OCR_PAD = 26

# The expected string is NOT in this prompt. It is compared in code, in `verdict()`.
PROMPT = (
    "This is a close crop of a garment or footwear photograph. Transcribe any brand text you can "
    "read in it, exactly as it appears, letter for letter.\n"
    "If the text is too blurred, too small, or too distorted for you to read the letters with "
    "confidence, reply with exactly UNREADABLE. Do not guess a plausible brand name.\n"
    "Reply with only the transcription, or only UNREADABLE. No explanation."
)


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def norm(s: str) -> str:
    """Fold away everything that is not an identity difference: case, apostrophes, spacing.
    'A DAY'S MARCH', 'a days march' and 'ADAYSMARCH' are the same brand string; 'A DAYS MRACH'
    is not."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def transcribe(img: np.ndarray) -> dict:
    """One open-ended VLM call -> one raw string. No expected answer is supplied to the model."""
    import base64, io, requests

    if config.active_provider() == "dry-run" or not os.getenv("OPENROUTER_API_KEY"):
        return {"raw": None, "status": "unavailable", "latency_s": 0.0, "cost_usd_estimated": 0.0}
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    t0 = time.time()
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
        json={"model": MODEL, "max_tokens": 24, "temperature": 0,
              "messages": [{"role": "user", "content": [
                  {"type": "image_url",
                   "image_url": {"url": "data:image/png;base64," + b64}},
                  {"type": "text", "text": PROMPT}]}]},
        timeout=120)
    dt = round(time.time() - t0, 1)
    if r.status_code >= 400:
        return {"raw": None, "status": f"http_{r.status_code}", "error": r.text[:200],
                "latency_s": dt, "cost_usd_estimated": 0.0}
    return {"raw": (r.json()["choices"][0]["message"]["content"] or "").strip(),
            "status": "ok", "latency_s": dt, "cost_usd_estimated": 0.0015}


def verdict(raw: str | None, expected: str) -> tuple[Verdict, str]:
    """The comparison the model never saw. This is where the expected string finally appears."""
    if raw is None:
        return Verdict.UNKNOWN, "specialist unavailable"
    if norm(raw) in ("UNREADABLE", ""):
        return Verdict.UNKNOWN, "specialist abstained: cannot read the letters"
    if norm(raw) == norm(expected):
        return Verdict.PASS, f"transcribed {raw!r} == expected"
    return Verdict.FAIL, f"transcribed {raw!r} != expected {expected!r}"


def build_controls() -> list[dict]:
    """Four crops. Two must pass, two must fail — in two different ways."""
    t4 = os.path.join(config.OUTPUTS, "task4_nano")
    pristine_src = np.asarray(Image.open(f"{t4}/step0_original.png").convert("RGB"))
    nano_t5_src = np.asarray(Image.open(f"{t4}/naive_step5.png").convert("RGB"))
    worn_src = np.asarray(Image.open(
        os.path.join(config.OUTPUTS, "task1_worn", "a2_with_markref.png")).convert("RGB"))

    x0, y0, x1, y1 = LABEL_BOX
    pristine = np.ascontiguousarray(pristine_src[y0:y1, x0:x1])
    nano_t5 = np.ascontiguousarray(nano_t5_src[y0:y1, x0:x1])
    p = WORN_OCR_PAD
    wx0, wy0, wx1, wy1 = (max(0, WORN_BOX[0] - p), max(0, WORN_BOX[1] - p),
                          min(worn_src.shape[1], WORN_BOX[2] + p),
                          min(worn_src.shape[0], WORN_BOX[3] + p))
    worn = np.ascontiguousarray(worn_src[wy0:wy1, wx0:wx1])

    # Destroy legibility the way real loss of scale destroys it — resample down and back up —
    # rather than by blurring, which leaves letter-shaped ghosts a VLM can still lock onto.
    h, w = pristine.shape[:2]
    small = cv2.resize(pristine, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    illegible = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return [
        {"id": "c1_pristine", "img": pristine, "expected": "A DAY'S MARCH", "box": LABEL_BOX,
         "expect_verdict": "PASS", "source": "task4_nano/step0_original.png",
         "tests": "the specialist reads an undamaged label"},
        {"id": "c2_nano_t5", "img": nano_t5, "expected": "A DAY'S MARCH", "box": LABEL_BOX,
         "expect_verdict": "FAIL", "source": "task4_nano/naive_step5.png (paid artefact)",
         "tests": "the specialist reports a real, unstaged corruption instead of rubber-stamping"},
        {"id": "c3_illegible", "img": illegible, "expected": "A DAY'S MARCH", "box": LABEL_BOX,
         "expect_verdict": "UNKNOWN", "source": "c1, resampled 1/8 and back (legibility destroyed)",
         "tests": "the specialist ABSTAINS instead of guessing a plausible brand"},
        {"id": "c4_typography", "img": worn, "expected": MANIFEST[0]["spelling"],
         "box": (wx0, wy0, wx1, wy1),
         "expect_verdict": "PASS",
         "source": f"task1_worn/a2_with_markref.png (paid artefact), quad padded {WORN_OCR_PAD}px",
         "tests": ("spelling PASSES on re-lettered output whose stroke IoU is 0.1268 — proves "
                   "spelling alone cannot certify mark identity")},
    ]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/spelling.json") and not os.getenv("WEON_RUN_ID"):
        print(f"REFUSING TO RUN: {OUT}/spelling.json exists and is canonical paid evidence.")
        print("  regenerate elsewhere : WEON_RUN_ID=v2 python -m experiments.spelling_specialist")
        sys.exit(2)

    if config.active_provider() == "dry-run":
        print("REFUSING TO RUN: no provider. Every control would return UNKNOWN and the run would")
        print("  report 0/4 — a result about a missing API key, not about the specialist.")
        sys.exit(2)

    controls = build_controls()
    print(f"provider={config.active_provider()} specialist={MODEL} repeats={REPEATS}")
    print(f"expected string is NOT in the prompt; comparison happens in verdict()\n")

    res = {"specialist": MODEL, "provider": config.active_provider(), "prompt": PROMPT,
           "repeats": REPEATS, "protocol": "open transcription, compared in code",
           "not_a_calibration": ("4 known-answer inputs. This establishes behaviour on these four, "
                                 "not general OCR accuracy."),
           "controls": {}}

    for c in controls:
        Image.fromarray(c["img"]).save(f"{OUT}/{c['id']}.png")
        runs = []
        for _ in range(REPEATS):
            t = transcribe(c["img"])
            v, why = verdict(t.get("raw"), c["expected"])
            t.update({"verdict": v.value, "reason": why})
            runs.append(t)

        seen = {r["verdict"] for r in runs}
        stable = len(seen) == 1
        got = runs[0]["verdict"].upper() if stable else "UNSTABLE"
        agrees = stable and got == c["expect_verdict"].upper()

        res["controls"][c["id"]] = {
            "expected_string": c["expected"], "expect_verdict": c["expect_verdict"],
            "source": c["source"], "tests": c["tests"],
            "crop_box": list(c["box"]),
            "input_sha12": _sha(c["img"]), "input_shape": list(c["img"].shape),
            "observed_verdict": got, "stable_across_repeats": stable,
            "as_specified": bool(agrees), "runs": runs,
        }
        flag = "as specified" if agrees else "*** NOT AS SPECIFIED ***"
        print(f"{c['id']:15} expect {c['expect_verdict']:8} got {got:9} {flag}")
        for r in runs:
            print(f"                  raw={r.get('raw')!r:30} {r['reason']}")
        print()

    # c4 is only meaningful next to the geometry check it contradicts.
    res["c4_identity_resolution"] = {
        "spelling_verdict": res["controls"]["c4_typography"]["observed_verdict"],
        "stroke_iou_same_crop": 0.1268,
        "geometry_verdict": "FAIL (threshold 0.35)",
        "conclusion": ("Spelling and geometry disagree on the same pixels, and both are right. "
                       "The mark says ARIGATO and is not the ARIGATO mark. A gate with only a "
                       "spelling specialist would commit a restyled logo; identity needs both, "
                       "which is why the certificate keeps them as separate checks."),
    }
    ok = sum(1 for v in res["controls"].values() if v["as_specified"])
    res["summary"] = {
        "controls_as_specified": f"{ok}/4",
        "cost_usd_estimated": round(sum(r["cost_usd_estimated"] for v in res["controls"].values()
                                        for r in v["runs"]), 4),
        "claim_licensed": ("On these four inputs the specialist passed a clean label, caught a real "
                           "corruption, abstained when legibility was destroyed, and passed a "
                           "correctly-spelled restyled mark that geometry rejects. It may now be "
                           "wired into the gate for the brand_text check on THIS case."),
        "claim_NOT_licensed": ("General OCR accuracy. Any threshold. Any other brand, font, "
                               "language or resolution. 4 inputs x 3 repeats is a probe."),
    }

    with open(f"{OUT}/spelling.json", "w") as fh:
        json.dump(metrics.json_safe(res), fh, indent=2)
    print(f"controls as specified: {ok}/4   est. ${res['summary']['cost_usd_estimated']}")
    print(f"wrote {OUT}/spelling.json")
    if ok < 4:
        print("\nA control that did not behave as specified is a RESULT, not a bug to tune away.")
        print("Report it. Do not adjust the prompt until it passes.")


if __name__ == "__main__":
    main()
