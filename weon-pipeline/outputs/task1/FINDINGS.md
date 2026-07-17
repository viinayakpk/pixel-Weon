# Task 1 supplementary findings

The authoritative interpretation is in `../../../REPORT.md`. This file gives compact artifact
notes for one controlled footwear product-shot case. It is not evidence of complete garment
consistency.

## Tested directions

1. **Prompt/reference conditioning.** A plain generation and a constrained generation with a
   tight wordmark reference both spell ARIGATO, and both restyle the upper mark's fine serif
   geometry. This is a visual comparison in `logo_zoom.png`; p1 has no persisted IoU.
2. **Deterministic local repair.** On p2, clear the incumbent upper mark and apply a
   reference-derived graft to an operator-declared quad. The local symmetric diagnostic changes:

| condition | stroke IoU | ink-area ratio | mark ΔE | prototype decision |
|---|---:|---:|---:|---|
| A model output | 0.2127 | 1.056 | 6.03 | reject geometry |
| B cleared + graft | **0.6705** | 1.008 | 8.15 | review |
| C supersampled + relit | 0.5112 | 1.391 | 43.17 | reject colour |

The thresholds are exploratory. `ink-area ratio` measures binarised area rather than stroke weight.
B and C do not form a clean ablation: B uses supersample 1 without relighting; C uses supersample 4
plus relighting. B derives from the reference asset and holds 0% bit-exact against it after warping.

Artifacts: `task1_compare.json`, `iou_reference_sensitivity.json`, `rectified_compare.png`.

## Whole-product correction

The input has two legible ARIGATO instances: a gold upper wordmark and a debossed midsole
wordmark. The plain generation retains both; the constrained configuration and A/B/C retain the
upper instance alone. The constrained prompt contains `no other text`, which may have contributed,
but one stochastic comparison with changed reference inputs cannot isolate causality.

The constrained generation lacks the midsole mark; the subsequent repair inherits that omission
rather than causing it. Its local upper-mark metric improves while product-level instance
completeness stays worse than the plain generation. This motivates an identity manifest with
expected instance count, placement and material mode, declared before per-instance fidelity
measurement.

Artifacts: `identity_manifest.json`, `identity_manifest_grid.png`. Reproduce with
`python -m experiments.identity_manifest`.

## Material and human checks

The foil mark's residual is more unipolar (`0.971`) than the tonal mark's (`0.527`). This is
consistent with albedo- versus relief-dominated appearance, but one photograph plus an inpainted
substrate cannot prove that decomposition. I implemented no relief renderer.

The current human evidence is an author pilot (`n=1`), with condition names hidden and the rater
aware of the hypothesis. The author accepted B and preferred C over B for naturalness. This shows
that prototype metrics and perception can disagree; it does not validate photographic
naturalness. One VLM configuration failed the isolated known-answer fidelity control, so I did not
use it as an automatic pass.

## Limits

- One product, one editor, no replicates; not an on-person garment case.
- Manual target placement; planar repair alone; no folds or occlusion.
- I did not evaluate colour, texture, seams, closures and overall structure in full.
- Automatic mark-presence attempts confused lettering with stitching/tread texture. The saved
  visual crops support the missing midsole instance; those failed detectors do not.

Offline reproduction: `task1_compare`, `identity_manifest`, `material_probe`, `relief_probe`, and
`human_eval --report`. Paid p1/p2 regeneration requires a unique `WEON_RUN_ID`.
