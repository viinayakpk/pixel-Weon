"""Model registry. OpenRouter ids re-checked against the live /models endpoint; fal entries
are UNVERIFIED (no fal key was available) and unusable here.

Prices are per image unless noted. Reference-image caps and seed behaviour matter for
the pipeline design, so they are recorded here rather than buried in code.
"""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    key: str
    endpoint: str            # fal endpoint id
    price_usd: float         # approx per image
    max_refs: int
    true_seed: bool          # does a fixed seed reproduce output?
    note: str = ""
    openrouter_id: str = ""  # OpenRouter model id, if served there


# 2026 roster, verified against live fal.ai / OpenRouter on 2026-07-16.
MODELS: dict[str, Model] = {
    "gpt-image-2": Model(
        "gpt-image-2", "openai/gpt-image-2/edit", 0.08, 16, False,
        "mask-based edit. Listed range $0.01 (1K) .. $0.41 (4K); per-run billing unverified.",
        openrouter_id="openai/gpt-image-2"),
    "nano-banana-pro": Model(
        "nano-banana-pro", "fal-ai/nano-banana-pro/edit", 0.15, 14, False,
        "listed $0.15/1K, $0.30/4K; no usable deterministic seed in this workflow.",
        openrouter_id="google/gemini-3-pro-image-preview"),
    "seedream-4.5": Model(
        "seedream-4.5", "fal-ai/bytedance/seedream/v4.5/edit", 0.04, 10, True,
        "multi-reference editor; not exercised in the retained submission."),
    "qwen-edit-2511": Model(
        "qwen-edit-2511", "fal-ai/qwen-image-edit-2511", 0.03, 3, True,
        "Apache-2.0, seed-deterministic; reproducible ablation control."),
    "fashn-tryon": Model(
        "fashn-tryon", "fal-ai/fashn/tryon/v1.6", 0.075, 2, True,
        "dedicated VTON backend; not exercised in the retained submission."),
    # Added after querying https://openrouter.ai/api/v1/models live. Note: `gpt-image-2` above is
    # NOT in that list at all — it is an unlisted alias that still routes and bills. Anyone
    # re-running this may find it gone; these two are listed and current.
    "gpt-5.4-image-2": Model(
        "gpt-5.4-image-2", "", 0.03, 16, False,
        "listed OpenAI image editor; image_output $0.00003/unit in provider metadata.",
        openrouter_id="openai/gpt-5.4-image-2"),
    "gemini-3.1-flash-image": Model(
        "gemini-3.1-flash-image", "", 0.06, 14, False,
        "listed Gemini flash image editor; image_output $0.00006/unit in provider metadata.",
        openrouter_id="google/gemini-3.1-flash-image"),
}

EDITOR = os.getenv("WEON_EDITOR", "gpt-image-2")


def active_provider() -> str:
    """Backend chosen by which key is set. WEON_DRY_RUN=1 forces offline."""
    if os.getenv("WEON_DRY_RUN", "").lower() in ("1", "true", "yes"):
        return "dry-run"
    if os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY"):
        return "fal"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "dry-run"

# Paths ------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA = os.path.join(ROOT, "test_data")
OUTPUTS = os.path.abspath(os.getenv("WEON_OUTPUT_ROOT", os.path.join(ROOT, "outputs")))
os.makedirs(OUTPUTS, exist_ok=True)
