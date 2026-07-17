"""Model registry: the four editors this submission actually ran, plus paths.

Every model listed here was exercised on the retained evidence, over OpenRouter. Ids were
re-checked against the live /models endpoint on 2026-07-16.

Prices are the provider's *listed* per-image rates, not a measured invoice. See
outputs/actual_cost.json, where this table runs 9.6% low against the real account charge.
Reference-image caps matter for the pipeline design, so they are recorded here rather than
buried in code.
"""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    key: str
    openrouter_id: str       # OpenRouter model id
    price_usd: float         # approx per image, as listed by the provider
    max_refs: int            # reference-image cap
    note: str = ""


MODELS: dict[str, Model] = {
    "gpt-image-2": Model(
        "gpt-image-2", "openai/gpt-image-2", 0.08, 16,
        "NOT in the live /models list at all — an unlisted alias that still routes and bills, so "
        "anyone re-running this may find it gone. Listed $0.01 (1K) .. $0.41 (4K); per-run "
        "billing unverified."),
    "nano-banana-pro": Model(
        "nano-banana-pro", "google/gemini-3-pro-image-preview", 0.15, 14,
        "listed $0.15/1K, $0.30/4K; no usable deterministic seed in this workflow."),
    "gpt-5.4-image-2": Model(
        "gpt-5.4-image-2", "openai/gpt-5.4-image-2", 0.03, 16,
        "listed OpenAI image editor; image_output $0.00003/unit in provider metadata."),
    "gemini-3.1-flash-image": Model(
        "gemini-3.1-flash-image", "google/gemini-3.1-flash-image", 0.06, 14,
        "listed Gemini flash image editor; image_output $0.00006/unit in provider metadata."),
}

EDITOR = os.getenv("WEON_EDITOR", "gpt-image-2")


def active_provider() -> str:
    """OpenRouter when a key is set, otherwise offline. WEON_DRY_RUN=1 forces offline."""
    if os.getenv("WEON_DRY_RUN", "").lower() in ("1", "true", "yes"):
        return "dry-run"
    return "openrouter" if os.getenv("OPENROUTER_API_KEY") else "dry-run"

# Paths ------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA = os.path.join(ROOT, "test_data")
OUTPUTS = os.path.abspath(os.getenv("WEON_OUTPUT_ROOT", os.path.join(ROOT, "outputs")))
os.makedirs(OUTPUTS, exist_ok=True)
