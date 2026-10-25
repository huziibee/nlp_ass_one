NLP PROJECT 1 — INSTALLATION AND RUN GUIDE

PROJECT
Where Does Recursive Feedback Fail? Contextual Symbol Generation and
Attention-Head Dependence (Wits NLP Project 1, 2026).

Base: predict the query character's context-assigned numeric label.
Extended: predict the first numeric label, a next character from the prompt,
and that character's numeric label. G1–G10 study target-only synthetic
feedback with real Omniglot input features.

REPORT RESULTS
Five model seeds; Base and Extended each use 300,000 episodes/generation.
Base G10 clean-minus-recursive label-accuracy gap: 12.19 percentage points
(descriptive paired 95% interval 3.29–21.09).
Extended G10 sampled next-character validity gap: 19.73 pp (14.24–25.22).
Matched target repair: +21.17 pp valid-symbol probability mass.
G10 uses validation data; the final test was previously opened at G3.

INSTALL — Ubuntu or WSL2, Python 3.10
  git clone --recurse-submodules https://github.com/huziibee/nlp_ass_one.git
  cd nlp_ass_one
  python3.10 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt

Without the original submodule, run:
  git submodule update --init --recursive

RUN — NO TRAINED CHECKPOINTS NECESSARY
  python run.py check
  python run.py data-check
  python run.py tests
  python run.py results
  python run.py methodology

check: verifies the required source, data and evidence files.
data-check: validates original Omniglot feature arrays and split assumptions.
tests: runs Python unit/regression tests for the model interfaces, aggregation,
       data protocol and target-repair contract.
results: generates outputs/reports/RESULTS_SUMMARY.md and .csv for the
         matched 300k Base and Extended Table 1 gaps, five-seed repair and
         attention response; also generates a SHA-256 file inventory.
methodology: creates outputs/reports/METHOD_CONTROLS.md/.json/.csv for
             separate exploratory budget/corpus controls.

The results command reads saved experimental metrics; it does not retrain
models, run inference, or open the protected final test.

RUN — CHECKPOINT-BACKED MODEL INTEGRATION
Requires GitHub CLI (gh) and access to the original model artifacts:
  gh auth login
  python run.py restore-checkpoints
  python run.py verify-checkpoints
  python run.py smoke-base
  python run.py smoke-extended

restore-checkpoints: downloads the original Base and Extended G0 artifacts.
verify-checkpoints: checks G0 metadata and writes local SHA-256 records.
smoke-base / smoke-extended: execute short, four-step, single-generation
       clean/recursive training, writing outputs/smoke_base/ and
       outputs/smoke_extended/. These are NOT full research replications.

FULL RESEARCH REPRODUCTION
Use scripts/base_300k_replication.py and the matching GitHub workflow to
run the five-seed 300k Base experiment, then scripts/aggregate_base_300k.py.
Use scripts/extended_recursive.py for the Extended generation loops. More
experiment sources and saved audit results are in scripts/ and
docs/experiments/. Checkpoints for full runs are separate GitHub artifacts,
not source-controlled files.

LICENSES AND ATTRIBUTION
The original implementation and derived feature file are hosted by Singh
et al. in the pinned external/icl-dynamics Git submodule. Do not redistribute
them as our files; see THIRD_PARTY_ASSETS.md.
