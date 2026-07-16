# Constraint-preserving black-box image workflows
### Tasks 1 (Garment Consistency) and 4 (Edit Degradation)

**Models:** four black-box image editors from OpenAI and Google via OpenRouter; no fine-tuning.  
**Scope:** Task 4 has one five-edit naive chain per editor; Task 1 has one footwear case; no replicates.  
**Cost:** the provider account charged **$3.7326** across all activity on the key, including smoke
tests, four paired Task 4 runs and aborted/failed calls, so no experiment-specific invoice is claimed.  
**Code/evidence:** `weon-pipeline/`; 31 offline tests; headline values have saved JSON and images.

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
| Task 1 upper wordmark repair | stroke IoU **0.213 -> 0.670** | local geometry improved, but full-product consistency did not |

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
| 1 | correct text, wrong brand geometry | both generations spell ARIGATO but use heavier letterforms |
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
letterforms; this is not a vendor rule or ranking. SSIM therefore cannot certify text identity.
OCR could fail misspellings but not certify typeface. The [receipt and hashes](weon-pipeline/outputs/task4/model_comparison_4x.json)
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

## 5. Limits and next work

- `n=1` chain per editor and one Task 1 case; no usable seed or error bars.
- Task 1 is a footwear product-shot case, not a model wearing garments in an environment.
- Supports and placement are hand-declared; thresholds are uncalibrated.
- Exact exterior preservation applies to local edits with correct supports, not relighting or
  background replacement; drift inside the support remains.
- Human evidence is an author pilot, not independent validation.

Next experiments:

1. Segment supports and predeclare a complete identity manifest with expected instance counts.
2. Calibrate semantic specialists on known-answer pass/fail pairs; route UNKNOWN to people.
3. Add material-aware rendering for print/albedo versus embroidery/deboss/relief.
4. Evaluate 5–15 cases spanning plain, printed, structured and layered garments, reporting
   preservation, instruction success, integration, coverage, latency and cost separately.

Reproduction commands and artifact locations are in the
[README](weon-pipeline/README.md).
