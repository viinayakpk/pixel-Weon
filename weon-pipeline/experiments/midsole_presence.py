"""Can the spelling specialist score the RELIEF mark that every pixel metric failed on?

`midsole_instance: UNKNOWN` is the last silent row in the worn certificate. The manifest declares a
second ARIGATO — debossed into the rubber, same colour as its substrate — and says honestly that
this pipeline cannot check it: the earlier automatic presence detector scored a BLANK BAND higher
than one that visibly reads ARIGATO.

The reason that detector failed is informative. It keyed on colour/luminance structure, and a relief
mark has almost none: it is carried by shading, not albedo. So the question here is whether a VLM,
which reads shape rather than colour contrast, can do what the pixel metric could not.

Controls first, same discipline as spelling_specialist.py:

  k1_packshot_midsole   the brand's own deboss     -> expect PASS
  k2_blank_band         blank midsole rubber       -> expect UNKNOWN, and above all NOT 'ARIGATO'
                        This is the input class that broke the previous detector. If the specialist
                        hallucinates the brand onto blank rubber it is worthless, and that failure
                        must be able to surface BEFORE the real queries are believed.

Then the real queries: the two paid worn generations. Open transcription — the expected string is
never in the prompt; the comparison happens in code.

Either outcome completes the manifest row. Transcribes -> the row is scored. Abstains -> the row is
a receipted UNKNOWN with a named reason. What it must never be again is silently unscored.

Run:  python -m experiments.midsole_presence      (12 VLM calls, ~$0.02 estimated)
"""
from __future__ import annotations
import hashlib, json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, metrics                                   # noqa: E402
from pipeline.gate import Verdict                                      # noqa: E402
from experiments.spelling_specialist import MODEL, REPEATS, PROMPT, norm, transcribe  # noqa: E402

OUT = os.path.join(config.OUTPUTS, "midsole" + (
    "_" + os.environ["WEON_RUN_ID"] if os.getenv("WEON_RUN_ID") else ""))
EXPECTED = "ARIGATO"

# Boxes read by hand off a coordinate grid, then LOOKED AT before spending (see midsole_crops.png).
# Generous by design: transcription needs the glyphs in frame, unlike stroke IoU which needs a tight
# quad. No stroke metric is computed here.
CROPS = [
    {"id": "k1_packshot_midsole", "src": "packshot", "box": (200, 1660, 700, 1810),
     "kind": "control", "expect": "PASS",
     "tests": "the specialist reads the brand's own relief mark"},
    {"id": "k2_blank_band", "src": "a2", "box": (420, 1125, 580, 1175),
     "kind": "control", "expect": "UNKNOWN",
     "tests": ("blank rubber: the input class that broke the previous detector. Must not "
               "hallucinate the brand onto an empty substrate")},
    {"id": "q1_a1_midsole", "src": "a1", "box": (80, 1030, 200, 1090),
     "kind": "query", "expect": None,
     "tests": "does a1 retain the second, debossed instance?"},
    {"id": "q2_a2_midsole", "src": "a2", "box": (120, 1110, 280, 1180),
     "kind": "query", "expect": None,
     "tests": "does a2 retain the second, debossed instance?"},
]


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def presence(raw: str | None) -> tuple[Verdict, str]:
    """Presence, not spelling. Absence is NOT provable from a transcription: a specialist that
    reads nothing has failed to find a mark, which is not the same as the mark being gone. So the
    only honest verdicts are PRESENT and UNKNOWN."""
    if raw is None:
        return Verdict.UNKNOWN, "specialist unavailable"
    n = norm(raw)
    if n in ("UNREADABLE", ""):
        return Verdict.UNKNOWN, "specialist abstained: no legible mark found (not proof of absence)"
    if n == norm(EXPECTED):
        return Verdict.PASS, f"transcribed {raw!r}: relief instance present"
    return Verdict.FAIL, f"transcribed {raw!r}: text present but not {EXPECTED!r}"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/midsole.json") and not os.getenv("WEON_RUN_ID"):
        print(f"REFUSING TO RUN: {OUT}/midsole.json exists and is canonical paid evidence.")
        sys.exit(2)
    if config.active_provider() == "dry-run":
        print("REFUSING TO RUN: no provider; every row would be UNKNOWN for the wrong reason.")
        sys.exit(2)

    src = {
        "packshot": np.asarray(Image.open(
            os.path.join(config.TEST_DATA, "case1_packshot.jpg")).convert("RGB")),
        "a1": np.asarray(Image.open(f"{config.OUTPUTS}/task1_worn/a1_packshot_only.png").convert("RGB")),
        "a2": np.asarray(Image.open(f"{config.OUTPUTS}/task1_worn/a2_with_markref.png").convert("RGB")),
    }
    print(f"provider={config.active_provider()} specialist={MODEL} repeats={REPEATS}")
    print("controls run first; a hallucinated brand on k2 invalidates the queries\n")

    res = {"question": "can a VLM score a relief mark that pixel metrics could not?",
           "specialist": MODEL, "prompt": PROMPT, "repeats": REPEATS,
           "protocol": "open transcription, compared in code; presence only, never absence",
           "boxes_declared_by": "hand, off a coordinate grid, inspected before spending",
           "crops": {}}

    for c in CROPS:
        img = np.ascontiguousarray(
            src[c["src"]][c["box"][1]:c["box"][3], c["box"][0]:c["box"][2]])
        Image.fromarray(img).save(f"{OUT}/{c['id']}.png")
        runs = []
        for _ in range(REPEATS):
            t = transcribe(img)
            v, why = presence(t.get("raw"))
            t.update({"verdict": v.value, "reason": why})
            runs.append(t)
        stable = len({r["verdict"] for r in runs}) == 1
        got = runs[0]["verdict"].upper() if stable else "UNSTABLE"
        row = {"kind": c["kind"], "expect": c["expect"], "tests": c["tests"],
               "box": list(c["box"]), "source": c["src"], "input_sha12": _sha(img),
               "observed": got, "stable_across_repeats": stable, "runs": runs}
        if c["kind"] == "control":
            row["as_specified"] = bool(stable and got == c["expect"])
        res["crops"][c["id"]] = row
        flag = ""
        if c["kind"] == "control":
            flag = "as specified" if row["as_specified"] else "*** NOT AS SPECIFIED ***"
        print(f"{c['id']:22} {got:9} {flag}")
        for r in runs:
            print(f"                       raw={r.get('raw')!r}")

    # The control that decides whether any of this is believable.
    k2 = res["crops"]["k2_blank_band"]
    k2_clean = all(norm(r.get("raw") or "") != norm(EXPECTED) for r in k2["runs"])
    res["k2_did_not_hallucinate"] = k2_clean
    controls_ok = all(v.get("as_specified") for v in res["crops"].values()
                      if v["kind"] == "control") and k2_clean

    res["manifest_row"] = {
        "attribute": "midsole_instance",
        "was": "UNKNOWN — declared so its loss is visible, but no working detector",
        "now": (f"a1={res['crops']['q1_a1_midsole']['observed']}, "
                f"a2={res['crops']['q2_a2_midsole']['observed']}") if controls_ok
               else "still UNKNOWN — the controls did not behave as specified",
        "controls_passed": controls_ok,
    }
    res["what_this_licenses"] = (
        "Presence of the relief instance on these two paid generations, with a blank-substrate "
        "control that did not hallucinate the brand. It does NOT license absence detection: a "
        "transcription that reads nothing has failed to find a mark, which is not proof there is "
        "none. It does not license any claim about relief marks in general — 4 crops is a probe."
    ) if controls_ok else "Nothing: the controls failed. Reported as a negative result."
    res["cost_usd_estimated"] = round(sum(r["cost_usd_estimated"] for v in res["crops"].values()
                                          for r in v["runs"]), 4)

    with open(f"{OUT}/midsole.json", "w") as fh:
        json.dump(metrics.json_safe(res), fh, indent=2)
    print(f"\ncontrols passed: {controls_ok}   k2 stayed clean: {k2_clean}")
    print(f"manifest row now: {res['manifest_row']['now']}")
    print(f"wrote {OUT}/midsole.json  (est. ${res['cost_usd_estimated']})")


if __name__ == "__main__":
    main()
