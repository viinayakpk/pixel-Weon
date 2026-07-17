# Task 4 appendix — three-zone support diagnostic

This is an offline recomposition of the saved turn-4 pixels rather than another model call. The aim
is to separate boundary smoothing from support selection.

| support | inward collar | exterior byte-identical | retained edit delta | seam diagnostic |
|---|---:|---:|---:|---:|
| box | 0 px | 100% | 100.00% | 34.101 |
| box | 8 px | 100% | 94.14% | 21.669 |
| welt polygon | 0 px | 100% | 100.00% | 50.817 |
| welt polygon | 8 px | 100% | 75.34% | 26.429 |

Within the fixed box, an 8 px inward collar reduces this seam diagnostic and keeps the exterior
exact. The polygon result is a **post-hoc oracle ablation**: I drew it by hand after the failure, it
occupies 15.2% of the original box, and it is neither a grounding system nor ground truth. Changing
the support changes the footprint more than changing collar width.

The seam values are not perceptual scores, and they do not compare across differently shaped
supports: a support following a real seam includes genuine image gradients in its measurement
ring. The useful result is the within-support trend rather than the absolute cross-support ranking.

This experiment supports a bounded conclusion: an inward collar can soften a hard commit edge, and
it cannot repair an overly broad or semantically wrong support. See `three_zone_ablation.json` and
`zone_compare.png`. Reproduce with `python -m experiments.three_zone_ablation`.
