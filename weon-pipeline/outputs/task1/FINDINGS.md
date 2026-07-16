# Task 1 supplementary findings

The authoritative interpretation is in `../../../REPORT.md`. This file provides compact artifact
notes for one controlled footwear product-shot case; it is not evidence of complete garment
consistency.

## Tested directions

1. **Prompt/reference conditioning.** A plain generation and a constrained generation with a
   tight wordmark reference both spell ARIGATO, but both restyle the upper mark's fine serif
   geometry. This is a visual comparison in `logo_zoom.png`; no IoU was persisted for p1.
2. **Deterministic local repair.** On p2, clear the incumbent upper mark and apply a
   reference-derived graft to an operator-declared quad. The local symmetric diagnostic changes:

| condition | stroke IoU | ink-area ratio | mark ΔE | prototype decision |
|---|---:|---:|---:|---|
| A model output | 0.2127 | 1.056 | 6.03 | reject geometry |
| B cleared + graft | **0.6705** | 1.008 | 8.15 | review |
| C supersampled + relit | 0.5112 | 1.391 | 43.17 | reject colour |

The thresholds are exploratory. `ink-area ratio` is binarised area, not stroke weight. B and C
are not a clean ablation: B uses supersample 1 without relighting; C uses supersample 4 plus
relighting. B is derived from the reference asset but is 0% bit-exact against it after warping.

Artifacts: `task1_compare.json`, `iou_reference_sensitivity.json`, `rectified_compare.png`.

## Whole-product correction

The input has two clearly legible ARIGATO instances: a gold upper wordmark and a debossed midsole
wordmark. The plain generation retains both; the constrained configuration and A/B/C retain only
the upper instance. The constrained prompt contains `no other text`, which may have contributed,
but one stochastic comparison with changed reference inputs cannot isolate causality.

The constrained generation lacks the midsole mark; the subsequent repair inherits rather than
causes that omission. Its local upper-mark metric improves while product-level instance
completeness remains worse than the plain generation. This motivates an identity manifest with
expected instance count, placement and material mode before per-instance fidelity is measured.

Artifacts: `identity_manifest.json`, `identity_manifest_grid.png`. Reproduce with
`python -m experiments.identity_manifest`.

## Material and human checks

The foil mark's residual is more unipolar (`0.971`) than the tonal mark's (`0.527`). This is
consistent with albedo- versus relief-dominated appearance, but one photograph plus an inpainted
substrate cannot prove that decomposition. A relief renderer is not implemented.

The current human evidence is an author pilot (`n=1`), with condition names hidden but the rater
aware of the hypothesis. The author accepted B and preferred C over B for naturalness. This shows
that prototype metrics and perception can disagree; it does not validate photographic
naturalness. One VLM configuration failed the isolated known-answer fidelity control, so it was
not used as an automatic pass.

## Limits

- One product, one editor, no replicates; not an on-person garment case.
- Manual target placement; planar repair only; no folds or occlusion.
- Colour, texture, seams, closures and overall structure are not comprehensively evaluated.
- Automatic mark-presence attempts confused lettering with stitching/tread texture; the missing
  midsole instance is supported by the saved visual crops, not those failed detectors.

Offline reproduction: `task1_compare`, `identity_manifest`, `material_probe`, `relief_probe`, and
`human_eval --report`. Paid p1/p2 regeneration requires a unique `WEON_RUN_ID`.
