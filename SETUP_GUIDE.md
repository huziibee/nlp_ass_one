# Setup guide: from clone to results and scientific reruns

This guide supplements the self-contained lecturer-required [README.txt](README.txt). It distinguishes **reading original evidence** from **actually running training**.

## 1. Exact environment

The historical computation used Ubuntu, CPU JAX, Python 3.10 and a pinned `icl-dynamics` submodule. The repository root `requirements.txt` imports `requirements-model-smoke.txt`, which specifies JAX/JAXLIB 0.4.26, Equinox 0.11.4, Optax 0.2.2 and compatible supporting libraries.

```bash
git clone --recurse-submodules https://github.com/huziibee/nlp_ass_one.git
cd nlp_ass_one
git switch main
git submodule update --init --recursive
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows with Python 3.10 and PowerShell:

```powershell
git clone --recurse-submodules https://github.com/huziibee/nlp_ass_one.git
cd nlp_ass_one
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Recommended for Windows:** WSL2 Ubuntu for maximum agreement with historical CPU-JAX runs. If dependency installation fails, report the exact system/Python error instead of changing library versions while calling it an identical reproduction.

## 2. What you can run without model checkpoints

```bash
python run.py check
python run.py data-check
python run.py tests
python run.py results
```

- `check`: verifies necessary repository paths and the original HDF5 location.
- `data-check`: checks the pinned author dataset; its expected shape is `(1623, 5, 512)`.
- `tests`: runs `unittest discover -s tests -v`, including episode invariants, repair contracts and aggregation logic.
- `results`: **reads** three committed original JSON summaries, verifies five-seed/test-isolation markers and generates formatted Markdown/CSV. It does not create new research metrics or checkpoints.

Generated presentation files: `outputs/reports/RESULTS_SUMMARY.md`, `outputs/reports/RESULTS_SUMMARY.csv`, `outputs/reports/RESULTS_INDEX.md`, and `outputs/reports/RESULTS_FILE_MANIFEST.csv` (every original committed evidence file, exact size, SHA-256). They can be regenerated safely and should not replace any original `docs/experiments/` evidence.

## 3. Required original feature dataset

Expected file:

```text
external/icl-dynamics/omniglot_resnet18_randomized_order_s0.h5
```

The feature file comes from the pinned original author repository, **not** a newly generated mock dataset or PNG images. The split is 50 training, 1,473 reserved validation and 100 final-test character classes, with training exemplars and held-out validation exemplars. The Extended identity classifier predicts among the 50 training IDs only; other class groups can still be used for the numerical-label task. Do not treat out-of-vocabulary next-symbol identity as a defined test metric.

## 4. Get original trained checkpoints

Recorded GitHub Actions sources:

| Asset | Original workflow run | Artifact name |
| --- | ---: | --- |
| Base G0, 1,000,000 episodes | `38612082174` | `base-paper-1m-validation-only` |
| Extended G0, 300,000 episodes | `38630071147` | `extended-omniglot-lr3e4-300k` |

For example, with the [GitHub CLI](https://cli.github.com/) installed and authenticated with repository access:

```bash
gh auth login
mkdir -p outputs/base_full_1m outputs/extended_g0_full
gh run download 38612082174 -n base-paper-1m-validation-only -D outputs/base_full_1m
gh run download 38630071147 -n extended-omniglot-lr3e4-300k -D outputs/extended_g0_full
```

Artifact download and extraction layouts **must be inspected**, not assumed to match the loader. If the artifact has expired, use the preserved original tagged releases, where available:

- [Original Omniglot checkpoints](https://github.com/huziibee/nlp_ass_one/releases/tag/nlp-assignment1-original-omniglot-checkpoints-20261015)
- [Mechanistic follow-up checkpoints](https://github.com/huziibee/nlp_ass_one/releases/tag/nlp-assignment1-mechanistic-followups-20261015)

**Exact Base checkpoint paths expected by `scripts/base_mechanistic.py`:**

```text
outputs/base_full_1m/
└── safe_train_val_smoke/
    ├── config.json
    └── checkpoints/
        └── 00001000000.eqx
```

The Base loader verifies the 1,000,000-episode iteration count, class/exemplar/label splits, and original checkpoint format. The Extended G0 loader expects `extended_g0_model.eqx` and `metrics.json` under `outputs/extended_g0_full/` (or an explicitly supplied alternate path). Original per-seed generation checkpoints also require matching historical source and metadata: never use an arbitrary G9 from another seed as the previous-generation source.

## 5. Short end-to-end training integrations

After inspecting and validating the checkpoint files:

```bash
python run.py smoke-base
python run.py smoke-extended
```

Both run one generation for only four training steps and compare correct-target vs recursive-target conditions; both write to new `outputs/smoke_*` directories. These are smoke checks **only**. Scientific figure reproduction requires original full-length seed-specific histories.

## 6. Full-scale training parameters

Original G0 Base: `python scripts/paper_base_full.py` (1,000,000 episodes; original source runtime).

Extended G0 example:

```bash
python scripts/extended_g0.py --artifact-root outputs/base_full_1m --steps 9375 --batch-size 32 --eval-every 937 --eval-count 512 --learning-rate 0.0003 --out outputs/extended_g0_full
```

Base G1–G10: `scripts/base_recursive.py` with 31,250 updates × batch 32 per generation. Extended G1–G10: `scripts/extended_recursive.py` with 9,375 updates × batch 32 per generation and learning rate 0.0003. The exact commands are in [README.txt](README.txt) section 5, including the four experimental arms.

For multiple independent seeds and mechanism interventions, read the original workflow YAML and `docs/experiments/` original summaries. A single fresh run is **not** the same as the existing five-seed study. The five-seed analyses involve independent initialisations and carefully matched repair targets and evaluations.

## 7. Figures, analyses, provenance

Original plots and trajectories are under `docs/experiments/archived_original_runs/g10_dashboard/` and `docs/experiments/g10_five_seed/`. The dashboard script can regenerate its SVG visualizations from a **complete** directory of source trajectories (not from the five-seed aggregate alone):

```bash
python scripts/render_g10_dashboard.py --input docs/experiments/archived_original_runs/g10_trajectories --out outputs/reproduced_dashboard
```

This reads original trajectory JSON/CSV. The root `run.py results` always displays the four primary paper-aligned statistics rather than claiming to rerun all plots. Other analyses have distinct scripts for induction-head ablation, counterfactual softmax, correction, and activation patching. Each recorded experiment's original JSON/CSV and matching workflow form its provenance.

## 8. Common failures

| Symptom | Likely cause | Required correction |
| --- | --- | --- |
| `ModuleNotFoundError: equinox` | Wrong virtual environment | Activate the Python 3.10 venv and reinstall `requirements.txt` |
| JAX/Equinox compatibility error | New unsupported library versions | Use Python 3.10 + pins from repository, not a floating upgrade |
| Missing `.h5` file | Submodule omitted | `git submodule update --init --recursive` |
| Missing Base config or `00001000000.eqx` | Artifact not downloaded/extracted into expected layout | Check the actual archive contents and paths shown above |
| Missing Extended `metrics.json` | Only the model weight file was restored | Restore the matching complete Extended G0 artifact |
| Unusual accuracy / impossible q2 class | Mixed seed/role, wrong label split or q2 semantics | Recheck experiment JSON and frozen split, never patch a result |
| Missing historical Actions archive | Artifact retention expired | Check tagged releases; mark unrecoverable files as unavailable rather than substituting results |

## 9. Existing budget-matched and accumulating-corpus experiments

Two additional experiments are already recorded. Their results are
reproducible from their **original JSON** without retraining:

```bash
python run.py methodology
python run.py results
```

The Base 300k sensitivity control equalises G0 and descendant exposure
counts with Extended (300k), but not task difficulty, architecture or
seeds. The accumulating corpus permanently retains all prior generation
target samples and an original real anchor; its input features are
always genuine original Omniglot features. Both controls have one model
seed, so they cannot establish a five-seed statistical comparison.

`scripts/audit_methodology.py` independently checks the number and
order of all ten generations, saved training exposure counts, corpus
growth, absence of final-test access and seed-count provenance. It
outputs `METHOD_CONTROLS.md`, `.csv` and `.json`.

For checkpoints, use an authenticated `gh` CLI:

```bash
gh auth login
python run.py restore-checkpoints
python run.py verify-checkpoints
```

The restore utility refuses to overwrite checkpoint folders; run the
verify command directly if you restored the original files manually.
It calculates local SHA-256 digests and checks the expected metadata;
the results are **not** compared against an external independently
published checkpoint digest. Full five-seed G10 result reaggregation
from the original 20 checkpoint-backed artifacts is automated by
`.github/workflows/verify-five-seed-reaggregation.yml`.
This reconstructs summaries, **not** a new multi-generation training run.

## 10. Fresh G0–G10 five-seed training from new initializations

A dedicated full-budget [workflow](.github/workflows/fresh-five-seed-full.yml) and deterministic `scripts/fresh_five_seed.py` verifier are now included. They are **distinct from the historical five-seed reaggregation**, which only reads existing model metadata.

```bash
python run.py fresh-plan
```

The full GitHub Actions workflow is partitioned into five independent Base G0 training runs, five independent Extended G0 training runs, twenty clean/recursive descendant chains and final aggregation. Each chain saves ten new G1–G10 checkpoints. No original held-out test is reopened. The workflow uses fresh training weights but the same fixed original Omniglot input features.

See [FRESH_FIVE_SEED_REPRODUCTION.md](FRESH_FIVE_SEED_REPRODUCTION.md) for execution, extraction layout, costs, failure recovery and validation criteria. A checked-in workflow is **not** proof that training finished. Only a successful completed run with the expected G0 and descendant artifacts and a passing aggregate verifies completion.

## 11. Matched 1M Base and Extended generation budgets

An additional [matched-budget study](MATCHED_ONE_MILLION_STUDY.md) equalizes training examples at **one million per generation from G0 onward**. It reuses the five *freshly trained* Base 1M model chains from the explicitly recorded Base run 40194253117 and trains five independently initialized Extended G0 models followed by ten new Extended clean/recursive G1–G10 chains at the same exposure budget.

Check the frozen design with `python run.py matched-plan`. Run the full workflow through `.github/workflows/matched-million-extended.yml` and independently audit the restored matched-result artifacts with `python run.py matched-aggregate`.

Do not merge original Extended 300K results with new Extended 1M results, and do not claim the matched study completed until all training and aggregate jobs pass. It does not remove differences in output task, computational cost or the historically opened final test.

## 12. Final-test and submission boundaries

The final original test was accessed at G3. **Do not rerun `scripts/evaluate_sealed.py`** as a routine smoke test or tune models on that result. G10 statistics are validation-only. The course also requires the official NeurIPS 2024 ethics questionnaire after references and a Faculty of Science AI Ethics declaration. These must be finalized from the official Moodle templates; an explanatory note in a draft PDF does not replace them.
