# Task 1, worn case — findings

The earlier Task 1 experiment used an isolated sneaker on concrete. The brief's product is a model
**wearing** the garment, which changes pose, scale, lighting and occlusion at once. A mechanism that
only works on the easy case is a demo. This is the real case.

Manifest declared before generation: [`cases/task1_worn.py`](../../cases/task1_worn.py).
Attempt budget: 2, enforced in code. Both attempts were usable; no third was generated.

## 1. The repair transfers, and the effect is larger on the harder case

| | model only | cleared + graft | gain |
|---|---|---|---|
| product shot | 0.2127 | 0.6705 | +0.458 |
| **worn shot** | **0.1268** | **0.7485** | **+0.622** |

Stroke IoU vs the packshot-extracted asset. The model does *worse* when the shoe is worn
(0.213 → 0.127) — pose, occlusion and scale, as expected. The deterministic repair does not degrade
with it; it lands higher.

The worn mark is ~74×36 px vs ~79×37 in the product shot, so this is not a resolution advantage.

Verified visually before being believed ([`worn_rectified_compare.png`](worn_rectified_compare.png)):
the graft carries the reference's fine serif letterforms; the model's own version is a heavier
re-drawing. The number and the image agree.

## 2. Both marks survived — the regression that motivated the manifest did not recur

The manifest declares two ARIGATO instances in two material modes (albedo upper, relief midsole).
Both attempts retained both. The earlier product-shot run lost the midsole instance to a prompt
containing "no other text", and the certificate could not notice, because it declared one logo on a
two-mark product. Declaring the inventory first is what makes that loss *visible*; it does not
prevent it.

## 3. Spelling passes on a mark that is not the mark

`brand_text` had been UNKNOWN in every certificate. An UNKNOWN is a specification: it names the
missing specialist. [`experiments/spelling_specialist.py`](../../experiments/spelling_specialist.py)
builds it, under open transcription — the expected string is never in the prompt; the model
transcribes blind and the comparison happens in code.

Four known-answer controls, 3 repeats each, **4/4 as specified, stable across all repeats**
([`outputs/spelling/spelling.json`](../spelling/spelling.json)):

| control | input | expected | observed |
|---|---|---|---|
| c1 pristine | untouched label | PASS | PASS — `A DAY'S MARCH` |
| c2 nano turn 5 | paid artefact, real corruption | FAIL | FAIL — `A DATT MARCH` |
| c3 illegible | c1 resampled 1/8 and back | UNKNOWN | UNKNOWN — `UNREADABLE` |
| c4 typography | paid artefact, re-lettered mark | PASS | PASS — `ARIGATO` |

c2 and c3 are the ones worth having. A specialist that never fails is a rubber stamp; a specialist
that guesses under uncertainty is worse than none, because it converts "we cannot see" into a
confident verdict. c3 abstained on a label it had just read correctly twice.

**This is a probe, not a calibration.** Four inputs establish behaviour on four inputs. It licenses
wiring the specialist into `brand_text` for this case. It licenses no claim about general OCR
accuracy, any threshold, or any other brand, font, language or resolution.

## 4. The point: two correct checks that contradict each other

The final certificate ([`worn_certificate.json`](worn_certificate.json)) for the model's own output:

```
A_model_only
   brand_text                 pass     transcribed 'ARIGATO' == expected
   mark_colour                pass     mark colour dE 6.44
   mark_geometry              fail     stroke IoU 0.1268 vs the brand mark
   photographic_naturalness   unknown  no calibrated specialist; routes to human review
   midsole_instance           unknown  relief mark: present by eye, no working automatic detector
   -> REJECTED
```

Three checks pass. It is rejected anyway.

The mark says ARIGATO and is not the ARIGATO mark. Both verdicts are correct on the same pixels.
Spelling certifies the string; only geometry certifies the identity. A single blended score would
have averaged them into a number meaning neither — and a gate holding spelling alone would commit a
restyled logo as "preserved", which is a brand failure that reads as a pass.

`B_cleared_graft` returns **REVIEW**, not commit: naturalness has no calibrated specialist, so it
stays UNKNOWN and blocks auto-commit. That is the intended behaviour. Silence is not success.

## Limits

- One worn generation, one hand-declared quad, no replicates.
- B is reference-**derived**, not a pixel copy: the warp makes it 0% bit-exact against the reference.
- The midsole relief mark is declared but not scored — this pipeline cannot represent or repair a
  relief mark, and the earlier automatic detector scored a blank band above one that visibly reads
  ARIGATO. It is in the manifest so its loss is visible, not because it can be checked.
- The quad was declared by hand off a coordinate grid, never located by matching the model's own
  re-drawn mark. Locating it from the output would make the measurement circular.
