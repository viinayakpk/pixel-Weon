# Constraint-preserving image workflows — Tasks 1 and 4

Read [`../REPORT.md`](../REPORT.md) first. This repository contains the black-box generation
adapters, offline evaluation, saved paid evidence and reproducible figures used in that report.

The submission makes two bounded claims:

- **Task 4:** hard compositing guarantees byte-exact pixels outside a correctly declared local
  write support. It does not guarantee grounding or semantic success inside that support.
- **Task 1:** deterministic repair improves one upper wordmark in one product-shot case, while a
  full identity inventory reveals that another legitimate wordmark is missing.

## Setup

Tested with Python 3.12.

```bash
python -m venv .venv
# activate .venv for your shell
pip install -r requirements.txt
copy .env.example .env     # Windows; optional, only for paid regeneration
```

`requirements.pinned.txt` records the exact offline QA environment. The optional fal.ai adapter
uses `requirements-paid.txt`; OpenRouter needs only the core `requests` dependency.

`WEON_DRY_RUN=1` forces offline mode even if `.env` contains a key. Offline generation returns the
input and reports N/A; it is never scored as a successful edit.

## Reproduce the submitted evidence — offline and free

Run from `weon-pipeline/`:

```bash
python run.py test
python -m experiments.eval_receipt
python run.py control
python -m experiments.recompute_common_union
python -m experiments.model_comparison
python -m experiments.three_zone_ablation
python -m experiments.identity_manifest
python -m experiments.material_probe
python -m experiments.relief_probe
python -m experiments.task1_compare
python -m experiments.human_eval --report   # recompute aggregate from the saved response
python run.py evidence
```

The shipped Task 4 metrics are historical paid evidence. `recompute_common_union` corrects the
main run's accepted-union/attempted-union comparison without making new calls. Cross-editor
comparison deliberately uses only the protected-label probe, Gate-v1 status, latency and configured
cost; it excludes historical broad-region fields. Future runs use cumulative attempted supports.

## Paid regeneration

Canonical paid source frames and metrics are protected from accidental reruns. A paid rerun must
use a new output name:

```powershell
$env:WEON_OUT_SUFFIX="replica"
python -m experiments.task4_compare     # 10 calls: baseline + ledger

$env:WEON_RUN_ID="replica"
python -m experiments.task1_generate    # 2 calls
```

Do not expect a deterministic reproduction: the editors expose no usable seed. Promote a new run
to canonical evidence only as a deliberate manual review step. Per-run costs are configured
estimates. `outputs/actual_cost.json` records a $3.7326 whole-key provider charge, but its scope also
includes smoke and failed/aborted activity, so it is not a per-experiment invoice.

## What ran versus what is a prototype

- **Gate v1 ran live:** target-region pixel movement plus crop-context SSIM. It committed 4/5
  candidates in the main run, 3/5 in each complete Google-editor ledger arm, and the gpt-5.4
  ledger arm was incomplete because of network errors.
- **Gate v2 is postmortem only:** preservation, instruction, material and boundary checks with
  UNKNOWN blocking automatic commit. `ledger.py` does not import it.
- **Task 1 placement is manual:** the target quad is declared after inspecting the generated
  product shot.
- **Human evidence is an author pilot (`n=1`):** the harness supports independent raters and
  append-only responses, but the submission does not claim independent validation.

## Layout

```text
pipeline/       metrics, ledger, compositing, gates and provider adapters
cases/          Task 4 instructions, protected label and predeclared supports
experiments/    one reproducible script per retained result
tests/          offline regression and evidence-immutability tests
test_data/      source images needed by the retained experiments
outputs/        saved paid frames, receipts and report figures
```

Important receipts:

- `outputs/eval_receipt.json` — destroyed-ROI evaluation control and Gate-v2 replay
- `outputs/task4/metrics.json` — original five-turn paid comparison
- `outputs/task4/common_union_recompute.json` — apples-to-apples historical correction
- `outputs/task4/model_comparison_4x.json` — four-editor protected-label audit and input hashes
- `outputs/task1/task1_compare.json` — local A/B/C repair diagnostics
- `outputs/task1/identity_manifest.json` — two-wordmark product audit
- `outputs/human_eval/alice.json` — raw author-pilot response
- `outputs/actual_cost.json` — whole-key provider usage, not experiment attribution

The report is authoritative. Exploratory thresholds are uncalibrated, and negative/superseded
experiments are not headline evidence.
