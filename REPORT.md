# Constraint-preserving black-box image workflows
### Tasks 1 (Garment Consistency) and 4 (Edit Degradation)

**Models:** four black-box image editors from OpenAI and Google via OpenRouter; no fine-tuning.  
**Scope:** Task 4 has one five-edit naive chain per editor; Task 1 has two footwear cases — a product
shot and a worn shot; no replicates.  
**Cost:** the provider account charged **$3.8512** across all activity on the key, including smoke
tests, four paired Task 4 runs and aborted/failed calls, so no experiment-specific invoice is claimed.
Our own price table runs **9.3% low** against it ([receipt](weon-pipeline/outputs/actual_cost.json)).  
**Code/evidence:** `weon-pipeline/`; 45 offline tests; headline values have saved JSON and images.

## 1. Decision summary

The experiments suggest a bounded operational rule:

> **Keep protected state outside the model's mutable representation. Automated preservation and
> scoring are only as complete as the declared constraint inventory.**

```text
Task 4: predeclared support -> crop candidate -> Gate v1 -> hard commit / rollback
Task 1: generated shot -> identity inventory -> declared repair placement -> review
```

Headline results:

| Finding | Result | Meaning |
|---|---:|---|
| Destroyed-ROI evaluation control | old SSIM **0.921626** -> corrected **0.000004** | measurement had to be fixed before optimisation |
| Task 4 protected label after one naive edit | **0.25–0.62%** byte-identical across four editors | full-frame rerendering rewrote almost every label pixel |
| Task 1 upper wordmark repair, product shot | stroke IoU **0.213 -> 0.670** | local geometry improved, but full-product consistency did not |
| Same repair, **worn** shot (§5) | stroke IoU **0.127 -> 0.749** | the model does worse when worn; the repair does not |
| Spelling vs geometry on the same pixels (§5) | text **PASS**, IoU **0.1268 FAIL** -> **REJECTED** | one score would have averaged these into a meaningless number |
| Automatic locator (§6) | **2 of 4** pockets, **0 px** overlap with either chest reading | it omits silently; supports stay hand-declared |

Observed declaration failures were concrete: one Task 4 support was broader than the named pocket,
and Task 1's automated check covered one of two wordmarks. Neither check could score what its
declared scope omitted.

## 2. Evaluation and failure analysis

The original masked SSIM zeroed pixels outside the mask, so identical black regions hid a destroyed
target. The correction evaluates only the ROI and adds PSNR and byte-exact percentage; dry runs
report N/A. See the [regression](weon-pipeline/tests/test_metrics.py) and
[receipt](weon-pipeline/outputs/eval_receipt.json).

Instruction success, preservation, boundary quality and naturalness remain separate, matching the
decoupled, human-checked direction of
[I2I-Bench](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_I2I-Bench_A_Comprehensive_Benchmark_Suite_for_Image-to-Image_Editing_Models_CVPR_2026_paper.html)
and the [OpenVTON-Bench preprint](https://arxiv.org/abs/2601.22725).

Observed failure taxonomy:

| Task | Failure | Evidence |
|---|---|---|
| 4 | global rerender drift | untouched label and texture change after one edit |
| 4 | grounding / mask-shape bias | a rectangular support produces a rectangular-looking edit |
| 4 | silent locator omission | 2 of 4 pockets returned, confidently, with no signal of incompleteness |
| 1 | correct text, wrong brand geometry | both generations spell ARIGATO but use heavier letterforms |
| 1 | a passing check on a failing mark | spelling PASS at stroke IoU 0.1268 — correct string, wrong mark |
| 1 | incomplete identity inventory | constrained generation lacks a second, debossed ARIGATO mark |
| 1 | representation risk (hypothesis) | residual probes suggest albedo and relief need different renderers |

## 3. Task 4 — edit ledger and cross-editor stress test

Five instructions were chained on one jacket. The naive arm rerenders the whole previous frame;
the ledger edits a crop and commits inside a predeclared support. This end-to-end comparison also
changes field of view, effective scale and context, so it is not a single-variable ablation.

| editor | naive label SSIM t1 → t5 | exact t1 | Gate-v1 commits | wall / configured estimate |
|---|---:|---:|---:|---:|
| `gpt-image-2` | 0.555 → 0.131 | 0.62% | 4/5 | 662 s / $0.80 |
| `gpt-5.4-image-2` | 0.403 → 0.165 | 0.38% | incomplete* | 411 s / $0.21 |
| `nano-banana-pro` (Gemini 3 Pro image) | 0.886 → 0.763 | 0.28% | 3/5 | 211 s / $1.50 |
| `gemini-3.1-flash-image` | 0.885 → 0.757 | 0.25% | 3/5 | 129 s / $0.60 |

![Main five-edit comparison: naive chain above, ledger below](weon-pipeline/outputs/task4/task4_grid.png)

For the main run, Gate v1 rejected turn 3 at context SSIM `0.545 < 0.60`; rollback left the
canonical image unchanged. The ledger's 100% exterior equality is architectural, not empirical:
pixels outside the declared support are copied forward. The empirical outcome is 4/5 commits.

The main run's broad metric initially used inconsistent unions after rejection; the corrected
[receipt](weon-pipeline/outputs/task4/common_union_recompute.json) uses all attempted supports and
does not affect the label result. Its [resize control](weon-pipeline/outputs/task4/resample_control.json)
lost `0.018` SSIM at worst versus `0.445` observed, excluding those resize paths—not every harness
effect. See the full [curve](weon-pipeline/outputs/task4/task4_curve.png).

![Same protected label across four sampled editors](weon-pipeline/outputs/task4/model_comparison_4x.png)

All four sampled naive runs changed almost every protected-label pixel. In this case the OpenAI
outputs changed scale/placement, while the Google outputs stayed aligned but visibly changed
letterforms; this is not a vendor rule or ranking. SSIM therefore cannot certify text identity: at
nano turn 5 it reads `0.7634` at `0.0113%` byte-equality, which detects the drift but cannot say the
failure is *spelling*. A specialist later transcribed that same crop as `A DATT MARCH` (§5). The
[receipt and hashes](weon-pipeline/outputs/task4/model_comparison_4x.json)
make the saved comparison auditable. The `gpt-5.4-image-2` naive chain completed, but three ledger
turns had network errors, so that ledger arm is excluded.

Direction 2 was an offline three-zone commit: an 8 px inward collar reduced the within-box seam
diagnostic `34.101 → 21.669`, retained 94.14% of candidate delta and kept the exterior exact. A
post-hoc polygon changed the footprint more, consistent with [mask-shape bias](https://arxiv.org/abs/2605.07846),
but is not ground truth. One hypothesis-aware author answered **YES** to turn 4; without a calibrated
shape/material check, semantics remain unresolved. Gate v2 is postmortem only.

With one candidate the ledger adds no generation call; retries or semantic review can add cost.

## 4. Task 1 — local repair versus product completeness

Direction 1 compared a plain prompt with tight reference conditioning. Both outputs spell ARIGATO
but restyle its serif geometry; p1 has no persisted IoU and OCR was not tested. A spelling-only
judgment would accept both: **a brand mark is geometry, not just text.** Direction 2 clears the
generated upper mark and applies a reference-derived repair.

| condition | stroke IoU ↑ | ink-area ratio | mark ΔE ↓ | prototype decision |
|---|---:|---:|---:|---|
| A: model output | 0.213 | 1.056 | 6.03 | reject geometry |
| B: cleared + graft | **0.670** | 1.008 | 8.15 | review |
| C: supersampled + relit | 0.511 | 1.391 | **43.17** | reject colour |

![Reference and local A/B/C comparison](weon-pipeline/outputs/task1/rectified_compare.png)

These are uncalibrated diagnostics. B is reference-derived, not a post-warp pixel copy. B versus C
changes supersampling and relighting together, so no causal claim about relighting is supported. The
[receipt](weon-pipeline/outputs/task1/task1_compare.json) records all checks.

The two-instance wordmark audit changed the conclusion. The packshot and plain generation contain two
ARIGATO instances: a gold upper wordmark and debossed midsole wordmark. The constrained generation
lacks the midsole instance; A/B/C inherit that input. The repair improved the remaining upper mark—it
did **not** cause the omission. The phrase `no other text` may have contributed, but changed
references and one stochastic sample prevent causal attribution.

![Identity inventory: local improvement with a missing second mark](weon-pipeline/outputs/task1/identity_manifest_grid.png)

Local IoU improved while the result still had one of two expected wordmarks. A future full
**identity manifest** should record counts, placement and material mode for logos, prints, closures
and structural details before per-instance checks.

An offline probe compared the ARIGATO foil crop with a separate Beyond Nordic tonal-embroidery
crop. It motivates material-specific routing but does not measure the missing midsole deboss, and
no relief renderer was implemented.

One hypothesis-aware author accepted B in a condition-label-hidden presentation; this is a sanity
check, not validation. A VLM configuration scored 0% on the known-answer fidelity control and was not
promoted to a gate. The two Task 1 calls took 99.9 s and cost $0.16 by the configured table; repair
and evaluation were offline.

## 5. Task 1 — the worn case, and a specialist that disagrees with itself

The product-shot case above answers an easier question than the brief asks. The brief's product is a
model *wearing* the garment, which changes pose, scale, lighting and occlusion at once. The
[manifest](weon-pipeline/cases/task1_worn.py) — six identity attributes, both ARIGATO instances,
each marked with whether this pipeline can actually check it — was declared before generating, and
the attempt budget of **2 was enforced in code**. Both attempts were usable; both retained both marks.

| | model only | cleared + graft | gain |
|---|---:|---:|---:|
| product shot | 0.2127 | 0.6705 | +0.458 |
| **worn shot** | **0.1268** | **0.7485** | **+0.622** |

The repair transfers, and the effect is larger on the harder case: the model does worse when the
shoe is worn, the deterministic repair does not. The worn mark is ~74×36 px against ~79×37 in the
product shot, so this is not a resolution advantage.

![Worn case: reference, model output, and cleared+graft](weon-pipeline/outputs/task1_worn/worn_rectified_compare.png)

`brand_text` had been UNKNOWN in every certificate. An UNKNOWN is a specification — it names the
missing specialist — so we built it, under **open transcription**: the expected string never enters
the prompt, and the comparison happens in code. Four known-answer controls, 3 repeats,
[4/4 as specified](weon-pipeline/outputs/spelling/spelling.json): it reads a clean label; it
transcribes the real nano corruption as `A DATT MARCH`; it answers `UNREADABLE` when legibility is
destroyed rather than guessing a brand it had just read; and it passes a correctly-spelled,
re-lettered mark. Four inputs is a probe, not a calibration — it licenses wiring this check into
this case, and no claim about general OCR accuracy.

The last control is the point. On the model's own worn output the
[certificate](weon-pipeline/outputs/task1_worn/worn_certificate.json) reads:

```text
A_model_only
   brand_text                 pass     transcribed 'ARIGATO' == expected
   mark_colour                pass     mark colour dE 6.44
   mark_geometry              fail     stroke IoU 0.1268 vs the brand mark
   photographic_naturalness   unknown  no calibrated specialist; routes to human review
   -> REJECTED
```

Three checks pass; it is rejected anyway. The mark says ARIGATO and is not the ARIGATO mark — both
verdicts correct on the same pixels. Spelling certifies the string, only geometry certifies the
identity, and a blended score would average them into a number meaning neither. `B_cleared_graft`
returns **review**, not commit: naturalness stays UNKNOWN and blocks auto-commit.

## 6. Grounding: the locator omits, silently

Every support here is hand-declared, which is defensible for five known edits and useless at scale —
so whether a VLM can supply them is the load-bearing question. Asking *"where is the right chest
pocket?"* returns a box whether or not the model found one; that earlier form put the box near the
**centre placket**, `0 px` against *both* declared chest boxes at ~6.2× their area. So we asked it to
**enumerate every pocket** instead, with no count supplied, and scored against both readings of
"right" (on a flat-lay the wearer's right chest pocket appears image-left).

The jacket has four pockets. It returned **two** — both lower flaps, with accurate masks — and
omitted both chest welts entirely. Not mislocated: absent. Both declared boxes contain a
[real, visible welt opening](weon-pipeline/outputs/grounding/missed_pockets_verification.png), so the
declarations were not the fiction.

![Locator overlay: declared supports in green, returned pockets in red/magenta](weon-pipeline/outputs/grounding/locator_overlay.png)

The response is confident, well-formed JSON with correct labels. **Nothing in it signals
incompleteness** — the failure mode is silent omission, which is the class this project exists to
catch. It also explains the placket answer: forced to name a pocket it cannot see, the model invents;
allowed to enumerate, it omits.

Consequently the planned re-composition of the already-paid turn-4 candidate **was not run**. Its
precondition — verified localization — failed, and re-compositing through an unverified region would
polish a mistake: exactly the error the collar ablation already identified. One image, one grounder,
one prompt; this blocks the automation rather than benchmarking Gemini.

## 7. Limits and next work

- `n=1` chain per editor; two Task 1 cases (product shot, worn), one generation each, no replicates.
- Supports and placement are hand-declared, and §6 is the measurement of why they still are.
- The worn quad was declared by hand off a coordinate grid — never located from the model's own
  re-drawn mark, which would make the measurement circular. B is reference-*derived*, so the warp
  leaves it 0% bit-exact against the reference.
- The spelling specialist is a 4-input probe. No threshold, brand, font or resolution generalises.
- Thresholds (IoU 0.35, SSIM 0.60, dE 25) are hand-set and uncalibrated.
- The midsole relief mark is declared but unscored: no working detector, and the earlier attempt
  ranked a blank band above one that visibly reads ARIGATO. It is in the manifest so its loss is
  *visible*, not because it can be checked.
- Exact exterior preservation applies to local edits with correct supports, not relighting or
  background replacement; drift inside the support remains.
- Human evidence is an author pilot, not independent validation.

Next experiments:

1. **Grounding, now the largest known dependency rather than an assumption** (§6): stronger
   segmentation, and an enumeration check that can report its own incompleteness.
2. Extend the specialist pattern to the two remaining UNKNOWNs — naturalness and relief presence —
   each with failure controls before use, as in §5.
3. Add material-aware rendering for print/albedo versus embroidery/deboss/relief.
4. Evaluate 5–15 cases spanning plain, printed, structured and layered garments, reporting
   preservation, instruction success, integration, coverage, latency and cost separately.

Reproduction commands and artifact locations are in the
[README](weon-pipeline/README.md).
