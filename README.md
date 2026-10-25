# Where Does Recursive Feedback Fail?
**Wits NLP Project 1 (2026)** — Contextual Symbol Generation and Attention-Head Dependence

## What this project does

The Base task learns to predict a numerical label for a query handwritten character from the current Omniglot prompt. The Extended task predicts the numerical label, a further character, and that character's numerical label. We study model collapse when generations G1–G10 train on predecessor-generated **answer targets**; the input character features remain real.

The paper compares clean targets with recursive targets (five model seeds, **300,000 training episodes per generation** for each task), audits invalid next characters, corrects malformed generated targets, and tests attention-head contributions and within-model activation patching.

At G10 the Base clean-minus-recursive contextual-label gap is **12.19 percentage points** (descriptive paired 95% interval **3.29–21.09**); the Extended sampled next-character validity gap is **19.73 points** (interval **14.24–25.22**). Target correction raises valid-character probability mass by **21.17 points**. All G10 findings use validation data; no new independent G10 test was performed.

## Install (Ubuntu or WSL2, Python 3.10)

```bash
git clone --recurse-submodules https://github.com/huziibee/nlp_ass_one.git
cd nlp_ass_one
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The submodule obtains the original Singh et al. model implementation and Omniglot/ResNet-18 feature file from the authors' repository. If you already cloned without it: `git submodule update --init --recursive`. Dependencies are pinned to the original JAX environment. A standard **CPU** is sufficient for validation and tests; full multi-generation retraining is considerably more expensive.

## Run the submission checks

```bash
python run.py check
python run.py data-check
python run.py tests
python run.py results
```

| Command | What it verifies or produces | How it relates to the paper |
| --- | --- | --- |
| `check` | Presence of code, submodule and evidence inputs | Reproduction prerequisites |
| `data-check` | Shape/split checks on original Omniglot features | Section 4: data and validation |
| `tests` | Unit and regression test suite | Code correctness, repair and aggregation contracts |
| `results` | `outputs/reports/RESULTS_SUMMARY.md`, `.csv`, plus evidence manifest | Table 1: **matched 300k Base** and Extended five-seed G10 effects; Section 7: repair and patching |
| `methodology` | `outputs/reports/METHOD_CONTROLS.*` | Historical supplementary single-seed budget/corpus sensitivity checks |

`results` **reaggregates stored summaries into a readable report**. It does not retrain models, rerun neural inference, or access the final test. Source data are in `docs/experiments/`; the primary matched-Base figures are under `docs/experiments/matched_300k/`.

## Run the model code (requires original G0 checkpoints)

Install the [GitHub CLI](https://cli.github.com/), authenticate with access to the repository, then run:

```bash
gh auth login
python run.py restore-checkpoints
python run.py verify-checkpoints
python run.py smoke-base
python run.py smoke-extended
```

`restore-checkpoints` downloads original checkpoint artifacts; `verify-checkpoints` checks metadata and local SHA-256 manifests. The two smoke commands **execute four training steps for one generation** in both clean and recursive arms; outputs appear in `outputs/smoke_base/` and `outputs/smoke_extended/`. These are **integration tests**, not five-seed retraining.

For full Base 300k G0–G10 training, the recorded protocol and executable code are in `scripts/base_300k_replication.py` and `.github/workflows/matched-base-300k-five-seeds.yml`; the final aggregate is produced by `scripts/aggregate_base_300k.py`. Extended generation and intervention code is under `scripts/extended_recursive.py`, `scripts/factorial_invalid_symbol_repair.py`, `scripts/extended_joint_head_ablations.py`, and `scripts/q2_context_activation_patch_pilot.py`. Matched repair evidence appears in `docs/experiments/q2_repair_mediation/`.

**Scope:** the previous final-test split was accessed at G3, so G10 is validation-only. Character identity is limited to the 50 training classes, and attention ablations do not prove a unique circuit. External assets and distribution boundaries: [THIRD_PARTY_ASSETS.md](THIRD_PARTY_ASSETS.md).
