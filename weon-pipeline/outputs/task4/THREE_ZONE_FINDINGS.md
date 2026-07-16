# Task 4 appendix — three-zone support diagnostic

This is an offline recomposition of the saved turn-4 pixels, not another model call. The aim is to
separate boundary smoothing from support selection.

| support | inward collar | exterior byte-identical | retained edit delta | seam diagnostic |
|---|---:|---:|---:|---:|
| box | 0 px | 100% | 100.00% | 34.101 |
| box | 8 px | 100% | 94.14% | 21.669 |
| welt polygon | 0 px | 100% | 100.00% | 50.817 |
| welt polygon | 8 px | 100% | 75.34% | 26.429 |

Within the fixed box, an 8 px inward collar reduces this seam diagnostic while preserving the
exterior exactly. The polygon result is a **post-hoc oracle ablation**: it was hand-drawn after the
failure, occupies only 15.2% of the original box and is not a grounding system or ground truth.
Changing the support changes the footprint more than changing collar width.

The seam values are not perceptual scores and are not comparable across differently shaped
supports: a support following a real seam includes genuine image gradients in its measurement
ring. The useful result is the within-support trend, not the absolute cross-support ranking.

This experiment supports a bounded conclusion: an inward collar can soften a hard commit edge, but
it cannot repair an overly broad or semantically wrong support. See `three_zone_ablation.json` and
`zone_compare.png`. Reproduce with `python -m experiments.three_zone_ablation`.
