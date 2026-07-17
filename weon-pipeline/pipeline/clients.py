"""OpenRouter image-edit client.

No key → dry-run, which returns the base image unchanged so the pipeline stays exercisable
offline. Dry-run output is reported as N/A and never scored as a successful edit.
"""
from __future__ import annotations
import os, io, base64
import numpy as np
from PIL import Image
import requests
from dotenv import load_dotenv

from . import config

load_dotenv()


def _data_uri(a: np.ndarray, max_side: int = 1536) -> str:
    h, w = a.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        a = np.asarray(Image.fromarray(a).resize((int(w * s), int(h * s)), Image.LANCZOS))
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _fetch(url: str) -> np.ndarray:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))


def edit(model_key: str, prompt: str, refs: list[np.ndarray]) -> np.ndarray:
    """Instruction edit. refs[0] is the base; extra refs are conditioning (e.g. a packshot).

    Region preservation is enforced downstream by our own paste-back/graft, never by a mask
    handed to the model — that is the whole point of the ledger.
    """
    model = config.MODELS[model_key]
    if config.active_provider() == "dry-run":
        return refs[0]

    payload = {"model": model.openrouter_id, "prompt": prompt, "n": 1, "output_format": "png",
               "input_references": [{"type": "image_url", "image_url": {"url": _data_uri(r)}}
                                    for r in refs[: max(model.max_refs, 1)]]}
    r = requests.post("https://openrouter.ai/api/v1/images", json=payload, timeout=180,
                      headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"})
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:300]}")
    item = (r.json().get("data") or [{}])[0]
    if item.get("b64_json"):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(item["b64_json"]))).convert("RGB"))
    return _fetch(item["url"]) if item.get("url") else refs[0]
