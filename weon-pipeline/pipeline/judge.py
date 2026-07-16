"""The specialist the gate keeps asking for.

Every certificate so far ends with `photographic_naturalness: UNKNOWN — no specialist`. That is
correct behaviour, not a placeholder: an unverifiable property must not ship as verified. But an
UNKNOWN is also a *specification*. It names exactly which specialist is missing. This is that
specialist.

Design follows the evidence, not intuition:

  * PAIRWISE, not pointwise. VLM judges rank far better than they score; switching paradigm
    alone lifted image-editing judge accuracy ~58% -> ~84%
    (VLM Judges Can Rank but Cannot Score, arXiv 2604.25235).
  * SWAP-DEBIASED. Position bias is systematic. Every pair is judged twice with the order
    swapped and a win counts only if the same image wins BOTH orderings; disagreement is
    recorded as a tie. That single mitigation removes most position bias
    (Judging the Judges, arXiv 2406.07791).
  * A tie is an honest outcome. An order-dependent verdict is evidence of bias, not a result.
  * The judge answers ONE question at a time. Asking for naturalness and fidelity together
    invites the model to average them.

Cost: 2 calls per pair (both orderings). Used only where a deterministic metric cannot reach.
"""
from __future__ import annotations

import base64
import io
import os
import re

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from . import config

load_dotenv()   # judge.py can be imported without clients.py; without this the key is invisible
                # and every verdict silently degrades to ERROR


PROMPTS = {
    "naturalness": (
        "Two crops of a product photograph, labelled A and B.\n"
        "Question: which looks more like an unedited photograph, and less like something was "
        "digitally pasted or retouched onto it?\n"
        "Ignore which one you find prettier. Ignore brand accuracy. Judge only whether the "
        "surface, edges and lighting look physically consistent with the rest of the photo.\n"
        "Answer with exactly one token: A, B, or TIE."
    ),
    "mark_fidelity": (
        "Three crops. REFERENCE is a brand's real mark. Then A and B.\n"
        "Question: which of A or B reproduces the REFERENCE mark's letterforms more faithfully "
        "— stroke weight, typeface and proportions?\n"
        "Ignore photographic quality and lighting. Judge only fidelity to the REFERENCE.\n"
        "Answer with exactly one token: A, B, or TIE."
    ),
}


def _b64(a: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _ask(question: str, images: list[np.ndarray], labels: list[str]) -> str:
    """One VLM call -> one token. Returns 'A' | 'B' | 'TIE' | 'ERROR'."""
    import requests

    # Respect the repository-wide offline switch even when .env contains a live key.
    if config.active_provider() == "dry-run":
        return "ERROR"
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return "ERROR"
    content: list = []
    for lab, im in zip(labels, images):
        content.append({"type": "text", "text": f"{lab}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + _b64(im)}})
    content.append({"type": "text", "text": question})

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": os.getenv("WEON_JUDGE", "google/gemini-2.5-flash"),
              "messages": [{"role": "user", "content": content}],
              "max_tokens": 8, "temperature": 0},
        timeout=120)
    if r.status_code >= 400:
        return "ERROR"
    txt = (r.json()["choices"][0]["message"]["content"] or "").strip().upper()
    m = re.search(r"\b(A|B|TIE)\b", txt)
    return m.group(1) if m else "ERROR"


def compare(kind: str, first: np.ndarray, second: np.ndarray,
            reference: np.ndarray | None = None) -> dict:
    """Swap-debiased pairwise verdict.

    Judged twice: (first, second) and (second, first). A win requires the SAME image to win
    both. Anything else is a tie — including a disagreement, which is a positive finding about
    the judge rather than a fact about the images.
    """
    q = PROMPTS[kind]
    ref = [reference] if reference is not None else []
    ref_lab = ["REFERENCE"] if reference is not None else []

    v1 = _ask(q, ref + [first, second], ref_lab + ["A", "B"])
    v2 = _ask(q, ref + [second, first], ref_lab + ["A", "B"])   # order swapped

    # translate each verdict into which ORIGINAL image it names
    w1 = {"A": "first", "B": "second"}.get(v1, v1)
    w2 = {"A": "second", "B": "first"}.get(v2, v2)

    if "ERROR" in (v1, v2):
        winner, why = "error", f"judge unavailable ({v1}/{v2})"
    elif w1 == w2 and w1 in ("first", "second"):
        winner, why = w1, f"consistent across both orderings ({v1}/{v2})"
    elif w1 == w2 == "TIE":
        winner, why = "tie", "judge called a tie in both orderings"
    else:
        winner, why = "tie", f"order-dependent verdict ({v1}/{v2}) — recorded as a tie, not a win"

    return {"kind": kind, "winner": winner, "reason": why,
            "raw": {"order_1": v1, "order_2": v2},
            "swap_consistent": w1 == w2,
            # One comparison makes two HTTP requests (one per ordering).
            "cost_usd_estimated": 0.0 if config.active_provider() == "dry-run" else 0.004}
