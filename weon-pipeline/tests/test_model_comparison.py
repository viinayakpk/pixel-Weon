from experiments.model_comparison import build_receipt


def test_cross_editor_receipt_uses_only_supported_label_metrics():
    receipt = build_receipt()
    assert receipt["same_saved_case_verified"] is True
    step0 = [v for k, v in receipt["inputs_sha256"].items() if k.endswith("step0_original.png")]
    assert len(step0) == 4 and len(set(step0)) == 1
    assert set(receipt["editors"]) == {
        "gpt-image-2",
        "gpt-5.4-image-2",
        "nano-banana-pro (Gemini 3 Pro image)",
        "gemini-3.1-flash-image",
    }
    assert receipt["editors"]["gpt-image-2"]["gate_v1_commits"] == 4
    assert receipt["editors"]["nano-banana-pro (Gemini 3 Pro image)"]["gate_v1_commits"] == 3
    assert receipt["editors"]["gemini-3.1-flash-image"]["gate_v1_commits"] == 3
    assert receipt["editors"]["gpt-5.4-image-2"]["gate_v1_commits"] is None
    assert receipt["editors"]["gpt-image-2"]["naive_label_bit_exact_pct"][0] == 0.619369
    assert receipt["editors"]["nano-banana-pro (Gemini 3 Pro image)"]["naive_label_bit_exact_pct"][0] == 0.281532
    assert "outside_union" not in str(receipt)
