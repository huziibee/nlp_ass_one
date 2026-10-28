# Final matched-budget study: Base + Extended at 300,000 episodes

## Decision and scope

We retain the scientifically completed original analyses without altering their measurements, and conduct the **missing five-seed Base experiment at 300,000 episodes**. Original Extended already has a five-seed 300k G0–G10 study. The separate one-seed Base 300k pilot (clean 76.95%, recursive 49.41% at G10) remains an exploratory sensitivity result and does **not** replace the new five-seed Base sample.

**The three 1M-training workflows were cancelled at user request.** GitHub cancellation was accepted and all three run statuses were confirmed `completed/cancelled` on 25 October 2026:
- Original fresh-full study: [40194253117](https://github.com/huziibee/nlp_ass_one/actions/runs/40194253117)
- Matched 1M review-branch duplicate: [40195817836](https://github.com/huziibee/nlp_ass_one/actions/runs/40195817836)
- Matched 1M `main` run: [40196072746](https://github.com/huziibee/nlp_ass_one/actions/runs/40196072746)

Completed artifacts remain historical records; do not call any of these partial runs completed full retraining.

## New 300k experimental design

| Variable | New Base | Existing Extended |
| --- | --- | --- |
| Episodes at G0 | 300,000 | 300,000 |
| Episodes at each G1–G10 descendant | 300,000 | 300,000 |
| Optimizer steps per generation (batch 32) | 9,375 | 9,375 |
| Model seeds | **Five newly trained, independent initializations** | Five previously completed seeds |
| Conditions | `clean_only`, `recursive_replacement` | `clean_only`, `recursive_replacement` |
| G0 + descendant model count | 5 + 100 = 105 | Previously trained, recorded |
| Model features | Original Omniglot/ResNet18 frozen features | Same original features |
| Evaluation | Validation roles; no G10 final test | Validation roles; no G10 final test |

Base initialization seeds are **2105, 2205, 2305, 2405, 2505**; independently defined data-order seeds are **44001, 45001, 46001, 47001, 48001**. Both Base arms use the same data ordering and model initialization per generation/seed. All five seeds remain in the analysis, even if their clean baselines are weak. This is **not a selection of the best seeds**.

Original Extended five-seed data: `docs/experiments/g10_five_seed/g10_seed_summary.json`.

## Run and verify

Read the five-seed predeclared plan:

```bash
python run.py base300k-plan
```

Training source: `scripts/base_300k_replication.py`. Evidence aggregator: `scripts/aggregate_base_300k.py`. Historical Base 300k loader and separate one-seed pilot: `scripts/base_fully_matched_300k.py`.

To run all 5 Base G0 models and 10 clean/recursive ten-generation chains, use [the GitHub Actions workflow](.github/workflows/matched-base-300k-five-seeds.yml). It restores **only new 300k Base G0 checkpoints**, not existing Base 1M weights or the one-seed pilot. It validates trained checkpoint files and per-generation metrics, outputs SHA-256 hashes, and preserves result artifacts for review.

The aggregate accepts only 105 newly trained Base checkpoints, five complete paired clean/recursive seed records, and the original five-seed Extended 300k JSON. It generates:

```text
outputs/base_300k_results/
  matched_300k_results.md
  matched_300k_results.json
  base_300k_seed_outcomes.csv
  base_300k_checkpoint_manifest.csv
```

Manual verification after downloading the workflow's artifacts into the documented input directories:

```bash
python run.py base300k-aggregate
```

The final aggregation runs only when all training jobs complete. **The workflow and unit tests establish a runnable protocol, not successful retraining until the training jobs actually pass.**

## What must appear in the manuscript

Until these five Base 300k runs finish, **continue reporting the existing original results honestly**. Once the new records pass verification, use the matched 300k experiment as the common-exposure comparison and retain earlier 1M Base work as corroborating sensitivity evidence.

The concise methods statement is:

> We additionally compared Base and Extended at a matched training budget of 300,000 episodes per generation, with five independent model seeds per task and clean/recursive conditions evaluated on validation data.

Do not write that statement as a completed-result claim until all five new Base G0 checkpoints and 100 descendants are actually verified.

The discussion still needs one scientifically honest sentence:

> Matching episode counts removes an exposure-count difference but does not equate the tasks' output spaces, learning difficulty or computational costs; cross-task effect sizes are therefore interpreted cautiously.

**Important:** The original one-seed Base 300k clean G10 score was 76.95%. Equal example counts cannot guarantee a strongly learned Base baseline. The aggregator reports all new Base clean accuracies and explicitly flags seeds under 60%, without discarding them or claiming that matching episodes by itself resolved undertraining.

The original final-test split was opened at G3 and cannot retroactively become untouched. G10 remains validation-only.
