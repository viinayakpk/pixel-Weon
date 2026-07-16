# Constraint-preserving black-box image workflows

**weon.ai take-home — Task 1 (Garment Consistency) and Task 4 (Edit Degradation).**

Black-box models only, no fine-tuning. Four image editors from OpenAI and Google via OpenRouter.

### 📄 [**Read the report → REPORT.md**](REPORT.md)

---

## The one-line thesis

> Generative models are good at proposing semantic change and unreliable at preserving constraints.
> So don't ask them to preserve. **Declare what may change before generating, let the model change
> only that, copy every other pixel forward, and verify both — abstaining when you can't verify.**

That gives exact preservation, provably. It does not give correctness: it *relocates* the ceiling
onto **grounding** (does the declared region correspond to the named object?) and **inventory** (did
you declare everything that matters?). Both failures in this project were declaration failures, not
model failures — and a wrong declaration costs twice, corrupting what you *write* and what you
*measure*.

## What the evidence says

| Finding | Result |
|---|---|
| The evaluation was broken, and got fixed **first** | destroyed-ROI masked SSIM **0.921626 → 0.000004** |
| An untouched brand label after **one** naive edit, across 4 editors | **0.25–0.62%** byte-identical — nobody preserves it |
| Deterministic mark repair, product shot → **worn** shot | stroke IoU **0.213 → 0.670**, then **0.127 → 0.749** |
| Spelling vs geometry on the **same pixels** | text **PASS**, IoU **0.1268 FAIL** → **REJECTED** |
| Can a VLM supply the write support automatically? | **no** — 2 of 4 pockets, omitted silently |

The fourth row is the one to look at. The mark says `ARIGATO` and is not the `ARIGATO` mark: both
checks are correct, three of five checks pass, and the gate rejects it anyway. A single blended
score would have averaged them into a number meaning neither.

## Where things are

| Path | What |
|---|---|
| [`REPORT.md`](REPORT.md) | **The submission.** Start here. |
| [`weon-pipeline/README.md`](weon-pipeline/README.md) | How to reproduce it — offline and free |
| [`weon-pipeline/pipeline/`](weon-pipeline/pipeline/) | The library: ledger, gate, metrics, mark repair, zones |
| [`weon-pipeline/cases/`](weon-pipeline/cases/) | Manifests, **declared before generation** — the point of the whole exercise |
| [`weon-pipeline/experiments/`](weon-pipeline/experiments/) | One file per question asked |
| [`weon-pipeline/outputs/`](weon-pipeline/outputs/) | Saved evidence: JSON receipts, hashes, images |
| [`weon-pipeline/tests/`](weon-pipeline/tests/) | 45 offline tests, no API |

Findings live next to the evidence that supports them:
[Task 4](weon-pipeline/outputs/task4/FINDINGS.md) ·
[Task 1 worn case](weon-pipeline/outputs/task1_worn/FINDINGS.md) ·
[grounding](weon-pipeline/outputs/grounding/FINDINGS.md)

## Reproduce

```bash
cd weon-pipeline
pip install -r requirements.pinned.txt
python -m pytest -q          # 45 tests, offline
python run.py test           # exercises the pipeline with no API key (dry-run)
```

Every headline number recomputes from saved artifacts **without spending anything**. Paid
regeneration is deliberately awkward: canonical evidence is write-protected and a rerun must name a
new output directory. See [`weon-pipeline/README.md`](weon-pipeline/README.md).

## Honest scope

- `n=1` chain per editor; two Task 1 cases; **no replicates** — the editors expose no usable seed.
- Supports and placement are **hand-declared**, and the grounding retest is the measurement of why
  they still are.
- Thresholds are hand-set and uncalibrated.
- Human evidence is an author pilot, not independent validation.
- The provider billed **$3.8512** across all activity on the key, including smoke tests and aborted
  calls — that is not a per-experiment invoice, and our own price table runs 9.3% low against it.

No real person's photograph was used: the worn case renders a clearly synthetic figure, per the
brief. Training, fine-tuning, UI and deployment were out of scope and were not attempted.
