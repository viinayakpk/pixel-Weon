# Constraint-preserving black-box image workflows

**weon.ai take-home. Task 1 (Garment Consistency) and Task 4 (Edit Degradation).**

Black-box models only, no fine-tuning. Four image editors from OpenAI and Google via OpenRouter.

### 📄 [**Read the report → REPORT.md**](REPORT.md)

---

## The thesis

> Generative models are good at proposing semantic change and unreliable at preserving constraints.
> **Declare what may change before generating, let the model change only that region, copy the other
> pixels forward, then verify the edit and the protected content. Abstain when you cannot verify.**

Exterior preservation then holds by construction. Correctness does not. Two harder problems set the
new ceiling: **grounding** (does the declared region correspond to the named object?) and
**inventory** (did you declare the things that matter?). Wrong declarations caused both failures in
this project. A wrong declaration corrupts the pixels you write and the score you measure.

## What the evidence says

| Finding | Result |
|---|---|
| I fixed the evaluation before optimising against it | destroyed-ROI masked SSIM **0.921626 → 0.000004** |
| An untouched brand label after **one** naive edit, across 4 editors | **0.25–0.62%** byte-identical, in all four |
| Deterministic mark repair, product shot → **worn** shot | stroke IoU **0.213 → 0.670**, then **0.127 → 0.749** |
| Spelling vs geometry on the **same pixels** | text **PASS**, IoU **0.1268 FAIL** → **REJECTED** |
| Can a VLM supply the write support? | **no**: 2 of 4 pockets, and no signal of the omission |

The fourth row carries the method argument. On the worn output the spelling check transcribes
`ARIGATO` and passes; the geometry check measures stroke IoU `0.1268` against the brand mark and
fails. Both verdicts hold on the same pixels. Three of five checks pass and the gate rejects the
image anyway. A blended score would average the two into a number that carries neither meaning.

## Where things are

| Path | What |
|---|---|
| [`REPORT.md`](REPORT.md) | **The submission.** Start here. |
| [`weon-pipeline/README.md`](weon-pipeline/README.md) | How to reproduce it, offline and free |
| [`weon-pipeline/pipeline/`](weon-pipeline/pipeline/) | The library: ledger, gate, metrics, mark repair, zones |
| [`weon-pipeline/cases/`](weon-pipeline/cases/) | Manifests, **declared before generation** |
| [`weon-pipeline/experiments/`](weon-pipeline/experiments/) | One file per question asked |
| [`weon-pipeline/outputs/`](weon-pipeline/outputs/) | Saved evidence: JSON receipts, hashes, images |
| [`weon-pipeline/tests/`](weon-pipeline/tests/) | 45 offline tests, no API |

Findings sit next to the evidence that supports them:
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

Each headline number recomputes from saved artifacts at no cost. Paid regeneration takes extra steps
by design: the repo write-protects canonical evidence, and a rerun must name a new output directory.
See [`weon-pipeline/README.md`](weon-pipeline/README.md).

## Honest scope

- `n=1` chain per editor, two Task 1 cases, **no replicates**. The editors expose no usable seed.
- I declared supports and placement by hand. The grounding retest measures why they stay that way.
- I set the thresholds by hand and did not calibrate them.
- Human evidence is an author pilot. No independent rater took part.
- The provider billed **$3.8646** across all activity on the key, including smoke tests and aborted
  calls. That figure covers the whole account rather than one experiment, and our own price table
  runs 9.6% low against it.

I used no photograph of a real person: the worn case renders a synthetic figure, per the brief. The
brief put training, fine-tuning, UI and deployment out of scope, and I attempted none of them.
