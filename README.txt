NLP PROJECT 1 - HOW TO RUN THIS REPOSITORY
==========================================
University of the Witwatersrand - COMS4054 / COMS7062 / COMS7066
Project: Interpretability of Model Collapse
Paper: Where Does Recursive Feedback Fail?
       Contextual Symbol Generation and Attention-Head Dependence

READ THIS FIRST
---------------
This is the required executable setup description for the lecturer. The
README.md explains the scientific task to a student; SETUP_GUIDE.md gives
troubleshooting and checkpoint details. Work from the MAIN branch.

The training uses the original Singh et al. 2024 two-layer/eight-head JAX
transformer and precomputed Omniglot/ResNet18 features. The first (Base)
task predicts the contextual numeric label. The Extended task predicts
the first numeric label, a next character, then its numeric label.
The next character MUST belong to one of the prompt's two characters.
The model's next-character output vocabulary is 50 TRAINING identities.
Generated training targets are synthetic; input prompts stay real.

System required for the documented research commands:
- Ubuntu Linux 22.04/24.04 or WSL2 Ubuntu recommended
- Python 3.10; Git; enough free disk for multi-generation checkpoints
- Network access for cloning submodules and downloading GitHub artifacts
- GitHub CLI (gh) authenticated to the private repository if downloading
  historical workflow outputs
- requirements.txt; pinned legacy JAX 0.4.26 runtime

1. GET SOURCE AND INSTALL
-------------------------
git clone --recurse-submodules https://github.com/huziibee/nlp_ass_one.git
cd nlp_ass_one
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

If submodule was not installed:
git submodule update --init --recursive

Windows PowerShell alternative (Python 3.10):
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Use WSL2 rather than upgrading old JAX silently if Windows wheels fail.

2. VERIFY INSTALLATION AND ORIGINAL DATA
----------------------------------------
python run.py check
python run.py data-check
python run.py tests

The data comes from the pinned upstream submodule:
external/icl-dynamics/omniglot_resnet18_randomized_order_s0.h5

Expected shape: 1,623 character classes x 5 exemplars x 512 features.
Train/validation/test classes: 50 / 1,473 / 100.
First 100-class final test has ALREADY been accessed historically at G3.
Never re-open it for G10 model selection or retuning.

3. DISPLAY THE ORIGINAL RESEARCH RESULTS (NO TRAINING REQUIRED)
---------------------------------------------------------------
python run.py results

Creates:
outputs/reports/RESULTS_SUMMARY.md
outputs/reports/RESULTS_SUMMARY.csv
outputs/reports/RESULTS_INDEX.md
outputs/reports/RESULTS_FILE_MANIFEST.csv

Results are read from the ORIGINAL committed experimental evidence:
docs/experiments/g10_five_seed/g10_seed_summary.json
docs/experiments/q2_repair_mediation/summary.json
docs/experiments/q2_activation_patching/five_seed_summary.json

Expected mean paired five-seed effects:
  Base G10 first-label accuracy clean - recursive: +9.22 pp
  Extended G10 valid sampled next character:       +19.73 pp
  Repair valid character probability mass:         +21.17 pp
  Repair natural context preference:               +5.94 logits

These are VALIDATION effects; they are not a new independent final test.
Paper figures and original per-generation CSV/JSON/SVG material remain
under docs/experiments/. No historical artifact is overwritten by results.

4. SMALL MODEL INTEGRATION TESTS (CHECKPOINTS REQUIRED)
------------------------------------------------------
Historical Base and Extended G0 model checkpoints are not checked into
Git source. Download them from the original GitHub Actions runs or tagged
checkpoint releases and restore exactly the required structure.

Base 1M original run:   GitHub Actions run 38612082174
Extended G0 300k run:  GitHub Actions run 38630071147
Representative names:
  base-paper-1m-validation-only
  extended-omniglot-lr3e4-300k

After authentication, GitHub CLI supports downloading with:
gh run download 38612082174 -n base-paper-1m-validation-only -D outputs/base_full_1m
gh run download 38630071147 -n extended-omniglot-lr3e4-300k -D outputs/extended_g0_full

IMPORTANT: Artifact ZIP roots may differ. Inspect extracted files before
running; the Base loader requires original model checkpoint and config,
and Extended G0 needs extended_g0_model.eqx AND metrics.json.
Do not rename checkpoints or invent metadata to satisfy the loader.
Artifact retention may expire; see the GitHub release links in the
SETUP_GUIDE.md and original Actions workflow YAML.

python run.py smoke-base
python run.py smoke-extended

These execute a one-generation, four-step integration check only. A
passing smoke test is NOT equivalent to reproducing the science.

5. REPRODUCE FULL BASE AND EXTENDED TRAINING (VERY EXPENSIVE)
-------------------------------------------------------------
First train/restore the source G0 checkpoints. Start with a clean outputs/
directory or unique new output names, NEVER overwrite preserved results.

G0 Base:
python scripts/paper_base_full.py

G0 Extended, 300,000 episodes (9,375 steps x 32):
python scripts/extended_g0.py --artifact-root outputs/base_full_1m --steps 9375 --batch-size 32 --eval-every 937 --eval-count 512 --learning-rate 0.0003 --out outputs/extended_g0_full

Base G1-G10:
python scripts/base_recursive.py --artifact-root outputs/base_full_1m --generations 10 --steps 31250 --batch-size 32 --eval-every 3125 --eval-count 512 --arms clean_only recursive_replacement real_anchor_half frozen_g0 --out outputs/new_base

Extended G1-G10:
python scripts/extended_recursive.py --base-artifact outputs/base_full_1m --extended-g0-artifact outputs/extended_g0_full --generations 10 --steps 9375 --batch-size 32 --eval-every 937 --eval-count 512 --learning-rate 0.0003 --arms clean_only recursive_replacement real_anchor_half frozen_g0 --out outputs/new_extended

Running these once does not reproduce ALL historical five-seed runs.
For seeds, interventions, post-hoc analyses, generator relationships and
original artifacts, consult docs/experiments/ and .github/workflows/.
Historical original results are already committed as small evidence files.

6. RESEARCH INTERPRETATION / SCIENTIFIC SAFETY
-----------------------------------------------
- Base and Extended use different descendants' compute budgets.
- Five paired model seeds give descriptive intervals, not independent
  confirmation on a fresh final-test set.
- Synthetic feedback replaces training targets ONLY; never claim full
  recursively generated inputs.
- The Extended symbol head is not open-vocabulary on unseen classes.
- Head interventions are causal responses to activation changes; they
  cannot alone identify the unique corruption-to-head training pathway.
- Never call scripts/evaluate_sealed.py to repeat the already-opened
  protected test.

7. WHAT MUST BE SUBMITTED TO THE LECTURER
------------------------------------------
The 27 October 2026 17:00 SAST brief requires:
- Complete source (src/, scripts/, pinned submodule reference as applicable)
- This literal README.txt and literal requirements.txt
- PDF extended abstract: TWO main-text pages incl. figures/equations/tables
- References and truthful author contributions (additional pages allowed)
- Completed official NeurIPS 2024 questionnaire AFTER references and BEFORE
  supplementary material (the brief says missing it means no marking)
- Faculty of Science AI Ethics Statement with truthful AI disclosure

The supplied PDF contains a note that the official Faculty text was not
provided. Obtain official Moodle wording and complete the sign-off.
There is no proof in this repository that the submission was uploaded.

8. TROUBLESHOOTING / HISTORICAL MATERIAL
----------------------------------------
Read SETUP_GUIDE.md for path validation and troubleshooting.
Read PROVENANCE_AND_BRANCHES.md for active vs archive content.
Do not treat old status TODOs in historical docs as current state.
Full historical research snapshot: branch research-archive-20261022.
