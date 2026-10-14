# NLP Assignment 1 — Data and Research Workspace

**Project:** Interpretability of Model Collapse (Wits NLP, 2026)  
**Current phase (14 October 2026):** the original Singh **Base G0 has completed 1,000,000 training sequences** and validation-only causal head interventions. **Recursive Base and Extended training, independent final testing and the submission PDF remain incomplete.**

## Verified dataset exploration — completed 14 October 2026

The **real original HDF5** was loaded in GitHub Actions from the pinned `icl-dynamics` submodule; file identity and HDF5 shape were verified. Results, plots and numerical summaries were committed to [the data audit](docs/data-audit/README.md). This is an actual data analysis, **not** the earlier synthetic test-fixture smoke test.

- Shape: **1,623 classes × 5 exemplars × 512 ResNet18 features** (float32).
- Mean cosine similarity: **0.889** for same-class exemplar pairs, **0.771** for different-class pairs.
- Simple **non-neural** cosine baseline: **92.91%** accuracy on two-way one-shot matching with a different query exemplar, versus 50% random choice.
- More difficult all-class retrieval: **16.45% top-1** among 1,623 classes, versus approximately 0.062% random chance.
- The file also contains a `class_order` array with **1,623 unique entries**, a nonidentity permutation. Its mapping to source raw-image indices is **not yet independently validated**.
- View [the five actual-data figures](docs/data-audit/README.md) or use the editable [feature notebook](notebooks/01_explore_omniglot.ipynb).

Only [Notebook 1](notebooks/01_explore_omniglot.ipynb) remains active; the redundant raw-image gallery notebook was removed. The original Omniglot-feature baseline is required for faithful task reproduction.

## Train / validation / final test (now codified)

The original authors' published run uses **50 training classes, 1,473 validation-reserved classes, and 100 test classes**, plus **1 training exemplar and 4 validation exemplars**, plus **8 training label pairs and 2 validation label pairs**. The original named evaluators assess training classes with held-out exemplars/relabelings and also periodically assess test classes. Our course project will make the final 100 classes a **protected once-only model evaluation**, never a hyperparameter-tuning monitor. The split is now **frozen**; reject 70/15/15 and 80/10/10 alternatives for the primary experiment. See [split rationale](docs/SPLIT_DECISION.md), [protocol](docs/TRAIN_VAL_TEST_PROTOCOL.md) and [manifest](docs/data-audit/reference_split_manifest.json).

Figure 2 asks whether feature pairs from the same character are more similar than pairs from different characters. Figure 5 tests how the nearest-cosine rule suffers as candidate distractors increase, **not whether a transformer has learned contextual label copying**. Our [new split-aware analysis](docs/data-audit/SPLIT_AWARE_ANALYSIS.md) explicitly excludes the 100 final-test classes from its evaluation. The original five plots remain historical, descriptive data exploration.

## Code readiness — measured vs not-yet-trained

- `scripts/reference_split.py`, `scripts/episode_probe.py`, and `tests/test_reference_split.py`: verifies the paper's exact class/exemplar/label-pair splits and valid two-symbol episodes.
- `scripts/split_diagnostics.py` + `scripts/make_split_report.py`: class-split aware geometric measurements and a teaching report for Figures 2 and 5.
- `src/target_repairs.py` + `tests/test_target_repairs.py`: deterministic raw/label/structure-joint repair operators with oracle invariants. **These are contracts, not trained-model evidence.**
- `src/recursive_dataset.py` + `tests/test_recursive_dataset.py`: original split-enforced clean Base/Extended targets, previous-generation replacement, half-real anchoring and fixed-G0 controls. **Dataset preparation only; no neural generation or retraining has occurred.**
- `tests/test_split_diagnostics.py`: verifies candidate-set probability calculations against known toy cases.
- GitHub Actions runs tests and publishes measured descriptive results on the **original author features**.
- **Original Base G0 has trained for 1,000,000 sequences** and has validated label-pair and held-out-class accuracy; [saved plots and evidence](docs/model-audit/README.md). [Actual causal head-intervention evidence](docs/mechanistic-audit/README.md) is available from the same checkpoint. Recursive Base/Extended pipelines, replication and sealed final evaluation are still missing.

## Up-to-date status (14 October 2026)

**[Read the verified current project status](docs/PROJECT_STATUS.md)**. A completed Base G0 and validation causal head analysis now exist; we next need an Extended three-token generator and real generation-to-generation retraining. See the [Extended output contract](docs/EXTENDED_OUTPUT_CONTRACT.md) for the open next-symbol representation issue.

The actual [16,000-sequence Base pilot results](docs/BASE_PILOT_16K.md) show training accuracy increasing to 50.39% but held-out-label-pair accuracy reaching only 35.55%. They do not establish learned induction heads. The [source-based full Base run](https://github.com/huziibee/nlp_ass_one/actions/runs/38612082174) **completed**; the [validation intervention run](https://github.com/huziibee/nlp_ass_one/actions/runs/38613565782) also completed.

## What is included

- **Original Singh et al. (2024) resources** in a pinned Git submodule at `external/icl-dynamics/`: the researchers' Omniglot ResNet18 feature file (`omniglot_resnet18_randomized_order_s0.h5`), JAX/Equinox transformer, samplers, training scripts, optogenetics/intervention framework, and plotting notebooks.
- **Dataset explorer**: `notebooks/01_explore_omniglot.ipynb` and `scripts/inspect_data.py` (no model training).
- **Research documentation**: data provenance, existing-asset inventory, current project status, assignment compliance, research notes, and the proposed Option B scientific question.

The original HDF5 binary is **referenced by submodule**, not copied into this repository. The submodule is pinned to an identified upstream commit for reproducibility. This repository is currently **public**; university lecture PDFs and a private historical experiment archive are intentionally **not redistributed** here.

## First setup — Windows PowerShell

For the complete lecturer-required reproducibility instructions use [README.txt](README.txt) and install [requirements.txt](requirements.txt).

```powershell
git clone --recurse-submodules https://github.com/huziibee/nlp_ass_one.git
cd nlp_ass_one
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/inspect_data.py
jupyter lab notebooks/01_explore_omniglot.ipynb
```


If you already cloned without submodules, run `git submodule update --init --recursive`.

The expected dataset is at:

`external/icl-dynamics/omniglot_resnet18_randomized_order_s0.h5`

**File integrity:** 16,638,016 bytes; Git blob SHA-1 `9eb588c516b641f5f25ac22227865a7ce5ab0b2f`. The inspector independently checks both before reading the file.

## What the data is

The HDF5 stores numerical features from Omniglot handwritten-character examples, extracted using a frozen pretrained ResNet18. It is **not** a directory of image pixels; a feature vector cannot simply be rendered as the original handwritten character. The official loader uses HDF5 dataset `resnet18/224/feat`. See [data card](docs/DATA_CARD.md) and the notebook for interactive inspection. See [Upstream reuse inventory](docs/UPSTREAM_REUSE.md) for what the original researchers already implemented.

## What work already exists?

**Do not start from zero.** The original Singh repository already contains the baseline training pipeline and mechanistic intervention framework. Separately, the **private** `huziibee/nlp_ass_1` repository contains a *completed, post-test* symbolic Base/Extended model-collapse study, including multi-seed results. Read [Current status](docs/PROJECT_STATUS.md) before planning fresh experiments.

## Next research direction — Option B

We propose to investigate an unresolved causal link from the previous symbolic study:

> Does corruption of **context-conditioned symbol/label structure** during recursive generation *cause* the later **induction-matching failures**, or are these parallel consequences of recursive training?

That is a **proposed new experiment**, not a conclusion or a claim that we have already collected new evidence. The prior final held-out evaluation is already used; future confirmatory tests require new, protected data/evaluation decisions.

See the [next-stage plan](docs/NEXT_STAGE_PLAN.md), [complete assignment requirements](docs/ASSIGNMENT_REQUIREMENTS.md) and [Option B protocol](docs/OPTION_B_PLAN.md). The canonical split is fixed; avoid changing data partitions while expanding the model.

## Why not run ablations directly on the HDF5?

The precomputed dataset has no transformer weights or attention heads. A true ablation needs a **trained checkpoint** and an intervention that disables or overwrites part of its computation. Notebook 1 can later become the *interactive dashboard* for these experiments, but the intervention itself should live in tested Python functions rather than being hard-coded as notebook cells.

## Scientific guardrails

1. Preserve the original upstream code and the previous completed study without rewriting their history.
2. Separate original paper reproduction, prior symbolic results, and proposed new experiments.
3. Declare training/data-generation conditions and outcome measures *before* looking at any new confirmatory test.
4. Do not interpret attention maps alone as causal evidence.
5. Use validation for model selection; keep the new final test protected.
6. Explicitly acknowledge that the completed symbolic task does not literally use Omniglot embeddings.
7. Submit Base **and** Extended recursive models, full code, train/validation/final-test analysis, and a two-page abstract with required separate contribution and ethics declarations. See [all explicit requirements and grading](docs/ASSIGNMENT_REQUIREMENTS.md).

## Attribution and availability

Upstream: [aadityasingh/icl-dynamics](https://github.com/aadityasingh/icl-dynamics); primary paper: [Singh et al., 2024](https://arxiv.org/abs/2404.07129). This repository does not claim ownership of the upstream work or grant a licence to its contents. See [Provenance](docs/PROVENANCE.md).
