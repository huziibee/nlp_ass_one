NLP PROJECT 1: INTERPRETABILITY OF MODEL COLLAPSE
================================================
Repository: https://github.com/huziibee/nlp_ass_one
Python: 3.10 (pinned legacy upstream JAX/Equinox runtime)
Source: Singh et al. (2024) author's code in external/icl-dynamics
Deadline: 27 October 2026, 17:00 South Africa time

STATUS AND IMPORTANT LIMITATIONS
--------------------------------
As of 14 October, this repository includes a verified 1,000,000-sequence
original Base generation-zero training result and validation-only attention
head interventions. Base/Extended recursively trained generations and
the final test/extended abstract are NOT yet provided. These must not be
presented as completed by the current repository.

1. SETUP
--------
git clone --recurse-submodules https://github.com/huziibee/nlp_ass_one.git
cd nlp_ass_one

Linux/macOS:
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

Windows PowerShell (with Python 3.10 installed):
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

If clone was made without submodules:
git submodule update --init --recursive

If legacy JAX pins fail on a platform, use Python 3.10 on Ubuntu or the
published GitHub Actions Ubuntu environment; do not silently upgrade
JAX and claim unmodified source reproducibility.

2. VERIFY DATA, SPLIT, TESTS
----------------------------
python scripts/check_data.py
python scripts/reference_split.py --out docs/data-audit/reference_split_manifest.json
python -m unittest discover -s tests -v

The one canonical split is original Singh et al.:
  class indices: train 0-49, validation 50-1522, sealed test 1523-1622
  exemplar indices: train 0; validation 1-4
  label pairs: eight training, two validation
See docs/SPLIT_DECISION.md and docs/TRAIN_VAL_TEST_PROTOCOL.md.
DO NOT evaluate reserved final test classes while developing or tuning.

3. ORIGINAL BASE MODEL TRAINING
-------------------------------
Small smoke (not an induction-head result):
python scripts/paper_base_smoke.py --steps 4 --batch-size 8

Reference-scale original Base G0 (1,000,000 samples; CPU-heavy):
python scripts/paper_base_full.py

Training output: outputs/base_full_1m/ (untracked; see saved logs/config).
Pre-existing measured results: docs/model-audit/README.md.
The repository GitHub Actions workflow base-full-reproduction.yml provides
the published environment and uploads the full outputs as an artifact.

4. MECHANISTIC BASE ANALYSIS
----------------------------
Requires the complete Base checkpoint files downloaded/extracted into
outputs/base_full_1m/ from the matching GitHub Actions run.
python scripts/base_mechanistic.py --num-eval 256 --seed 3201

Summary and figures: docs/mechanistic-audit/README.md.
The workflow base-mechanistic-validation.yml runs this on Ubuntu with
the pinned checkpoint artifact. All reported analyses are development
or validation only, not the protected final test.

5. DATA EXPLORATION
-------------------
python scripts/inspect_data.py --out docs/data-audit
jupyter lab notebooks/01_explore_omniglot.ipynb

These explore frozen ResNet18 feature geometry and do NOT train models.

6. SUBMISSION REQUIREMENTS AND UNSOLVED COMPONENTS
--------------------------------------------------
Read docs/ASSIGNMENT_REQUIREMENTS.md for every requirement and the rubric.
Before final submission, add a correct Extended three-output model, both
Base/Extended recursive pipelines, multiple-generation evaluations,
appropriately controlled mechanistic analysis, and independent final test.
Submit complete code, requirements.txt, README.txt, two-page PDF extended
abstract (including figures/equations/tables), separate references,
contribution statement, the exact NeurIPS 2024 ethics questionnaire
(after references, before any supplementary material) and Faculty of
Science AI Ethics statement. Obtain official Moodle templates.

The original paper's periodic test evaluator has been replaced with a
validation-only class evaluator; sealed test classes are never touched
by the supplied training commands. See docs/PROVENANCE.md for attribution.


7. BASE RECURSIVE GENERATIONS (NEW)
-----------------------------------
Requires completed original Base G0 checkpoint from the same repository:
outputs/base_full_1m/safe_train_val_smoke/checkpoints/00001000000.eqx

Small integration run:
python scripts/base_recursive.py --generations 2 --steps 8 --batch-size 8 --eval-every 8 --eval-count 64 --out outputs/base_recursive_smoke

Reference-length 3-generation experiment for each arm:
python scripts/base_recursive.py --generations 3 --steps 31250 --batch-size 32 --eval-every 3125 --eval-count 512 --arms recursive_replacement --out outputs/base_recursive_full

Other documented arms:
  clean_only, real_anchor_half, frozen_g0.
All conditions use original HDF5 data and original class/label splits.
The generation-to-generation substitution is of the *target label only*;
the full source input prefix is not generated.

Summarize training and validation curves:
python scripts/summarize_base_recursive.py --root outputs/base_recursive_full --generations 3 --arms recursive_replacement --out outputs/base_recursive_full/recursive_curves.csv

8. EXTENDED THREE-TOKEN G0 (NEW)
--------------------------------
The Extended output is:
  query label (5 logits), next symbol (unmasked 50 TRAINING class logits),
  next label (5 logits).
Input uses original frozen ResNet18 feature vectors and original 2-layer
transformer architecture. Target next symbol in clean data is sampled
uniformly from the two contextual symbols.
The next-symbol head CANNOT output never-trained final-test class IDs.

Small integration:
python scripts/extended_g0.py --steps 8 --batch-size 8 --eval-every 8 --eval-count 64 --out outputs/extended_g0_smoke

Reference-length:
python scripts/extended_g0.py --steps 31250 --batch-size 32 --eval-every 3125 --eval-count 512 --out outputs/extended_g0_full

Check causal teacher ordering:
python scripts/verify_extended_causality.py --trained outputs/extended_g0_full

9. EXTENDED RECURSION (NEW)
--------------------------
Requires a completed, validation-gated Extended G0 artifact and the
original Base configuration artifact under outputs/base_full_1m/.
Do NOT bypass the G0 quality gate except for a miniature integration smoke.

python scripts/extended_recursive.py --generations 2 --steps 31250 --batch-size 32 --eval-every 3125 --eval-count 512 --arms recursive_replacement --out outputs/extended_recursive

Other matched arms: clean_only, real_anchor_half, frozen_g0.
Models are reinitialized each generation and trained on new targets
sampled AUTOREGRESSIVELY from the previous model (or frozen G0).
Feature input prefixes remain clean; no historical dataset grows.

10. INDEPENDENT FINAL TEST -- KEEP SEALED UNTIL FROZEN
-----------------------------------------------------
During development, all training scripts use train/validation ONLY.
Once the complete architecture, split, loss, generation protocol,
hyperparameters and evaluated endpoints are final, use
scripts/evaluate_sealed.py with --confirm-final-frozen. It refuses
existing output folders and scores only the 100 protected classes.
DO NOT RUN before protocol freeze or reuse results to tune models.

The Extended classifier has no valid zero-shot global symbol-class-ID
target on unseen test classes, so final Extended evaluation computes
first-label and teacher-forced second-label accuracy/loss only.
Training/validation evaluation additionally reports generated next-symbol
context validity, valid probability mass and continuation consistency.

These code and integration tests DO NOT establish scientific collapse
or replace trained multi-generation results.
