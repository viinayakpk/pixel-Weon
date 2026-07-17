# Task 4 supplementary findings

The authoritative interpretation is in `../../../REPORT.md`. This is one five-turn local-edit case,
four non-deterministic editors and one chain per editor; there are no replicates.

## Comparison

The naive workflow rerenders the full previous frame. The ledger edits a contextual crop and hard
commits inside a predeclared support alone. It is an **end-to-end strategy comparison** rather than
a single-variable ablation: field of view, scale and context differ.

| turn | naive label byte-identical | naive label SSIM | ledger label |
|---|---:|---:|---:|
| 1 | 0.62% | 0.555 | 100% / 1.000 |
| 3 | 0.01% | 0.172 | 100% / 1.000 (rejected) |
| 5 | 0.03% | 0.131 | 100% / 1.000 |

No instruction named the protected label. Ledger preservation is exact by construction, because the
ledger copies outside-support pixels forward. Gate v1 committed four of five candidates and rejected
turn 3 at context SSIM `0.545 < 0.60`; I set that threshold by hand and did not calibrate it.

I recomputed historical broad-region values over one cumulative union of all attempted supports
(`common_union_recompute.json`). The conclusion holds, and future runs use this definition in
`task4_compare.py`.

## Cross-editor stress test

I verified the same saved base, instructions and supports across four naive chains. After one
full-frame edit, protected-label byte exactness ran `0.25–0.62%`. SSIM ranked the Google runs
higher, while direct inspection showed different failure modes: better alignment with changed
letterforms, against larger scale/placement drift in the OpenAI runs. This is one within-case
pattern. It is not a vendor rule or model ranking. `model_comparison_4x.json` excludes historical
broad-region metrics and records input hashes. The gpt-5.4 naive arm completed; its ledger arm had
network errors.

## Controls and semantic limit

Resize-only controls produced label SSIM `0.982–0.999`; the worst tested loss (`0.018`) is far
smaller than the first naive turn's loss (`0.445`). This rules out the tested resize paths as an
explanation. It does not rule out other harness effects.

Turn 4 shows the limit: Gate v1 accepted a rectangular tan edit, because pixel movement measures
liveness rather than semantic success. One hypothesis-aware author answered YES when asked whether
it satisfied the instruction; that single response does not establish shippability. Semantic
correctness remains unresolved. Gate v2 is a postmortem prototype and did not run in the ledger.

## Cost and scope

The main comparison used 10 calls, 662.2 seconds and $0.80 estimated. With one candidate, the
ledger itself adds no generation call; retries or semantic specialists can. Exact exterior equality
suits local edits with accurate supports. It does not extend to relighting or background
replacement, and it does not address drift inside the support.

Primary artifacts: `metrics.json`, `common_union_recompute.json`, `resample_control.json`,
`model_comparison_4x.json`, `model_comparison_4x.png`, `task4_grid.png`, `task4_label_grid.png`,
`task4_curve.png`, and `intent_check.png`.

Offline reproduction: `eval_receipt`, `recompute_common_union`, `model_comparison`,
`three_zone_ablation`, `run.py control`, and `run.py evidence`. Paid regeneration requires a
unique `WEON_OUT_SUFFIX`.
