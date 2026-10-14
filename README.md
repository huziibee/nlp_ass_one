# NLP Assignment 1 — Data and Research Workspace

**Project:** Interpretability of Model Collapse (Wits NLP, 2026)  
**Current phase:** Stage 0 — inspect existing data, code, and previously completed research. **No new models trained in this repository.**

## What is included

- **Original Singh et al. (2024) resources** in a pinned Git submodule at `external/icl-dynamics/`: the researchers' Omniglot ResNet18 feature file (`omniglot_resnet18_randomized_order_s0.h5`), JAX/Equinox transformer, samplers, training scripts, optogenetics/intervention framework, and plotting notebooks.
- **Dataset explorer**: `notebooks/01_explore_omniglot.ipynb` and `scripts/inspect_data.py`, with no model training.
- **Research documentation**: data provenance, existing-asset inventory, current project status, assignment compliance, research notes, and the proposed Option B scientific question.

The original HDF5 binary is **referenced by submodule**, not copied into this repository. The submodule is pinned to an identified upstream commit for reproducibility. This repository is currently **public**; university lecture PDFs and a private historical experiment archive are intentionally **not redistributed** here.

## First setup — Windows PowerShell

```powershell
git clone --recurse-submodules https://github.com/huziibee/nlp_ass_one.git
cd nlp_ass_one
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-data.txt
python scripts/inspect_data.py
jupyter lab notebooks/01_explore_omniglot.ipynb
```

If you already cloned without submodules, run `git submodule update --init --recursive`.

The expected dataset is at:

`external/icl-dynamics/omniglot_resnet18_randomized_order_s0.h5`

**File integrity:** 16,638,016 bytes; Git blob SHA-1 `9eb588c516b641f5f25ac22227865a7ce5ab0b2f`. The inspector independently checks both before reading the file.

## What the data is

The HDF5 stores numerical features from Omniglot handwritten-character examples, extracted using a frozen pretrained ResNet18. It is **not** a directory of image pixels; a feature vector cannot simply be rendered as the original handwritten character. The official loader uses HDF5 dataset `resnet18/224/feat`. See [data card](docs/DATA_CARD.md) and the notebook for interactive inspection.

## What work already exists?

**Do not start from zero.** The original Singh repository already contains the baseline training pipeline and mechanistic intervention framework. Separately, the **private** `huziibee/nlp_ass_1` repository contains a *completed, post-test* symbolic Base/Extended model-collapse study, including multi-seed results. Read [Current status](docs/PROJECT_STATUS.md) before planning fresh experiments.

## Next research direction — Option B

We propose to investigate an unresolved causal link from the previous symbolic study:

> Does corruption of **context-conditioned symbol/label structure** during recursive generation *cause* the later **induction-matching failures**, or are these parallel consequences of recursive training?

That is a **proposed new experiment**, not a conclusion or a claim that we have already collected new evidence. The prior final held-out evaluation is already used; future confirmatory tests require new, protected data/evaluation decisions.

See [Experimental plan](docs/OPTION_B_PLAN.md). We will agree on the workstreams **before writing a prompt for the coding model**.

## Scientific guardrails

1. Preserve the original upstream code and the previous completed study without rewriting their history.
2. Separate original paper reproduction, prior symbolic results, and proposed new experiments.
3. Declare training/data-generation conditions and outcome measures *before* looking at any new confirmatory test.
4. Do not interpret attention maps alone as causal evidence.
5. Use validation for model selection; keep the new final test protected.
6. Explicitly acknowledge that the completed symbolic task does not literally use Omniglot embeddings.
7. Submit the required Base and Extended models, full code, train/validation/test assessment, mechanistic analysis, two-page abstract, contribution statement, and ethics declarations.

## Attribution and availability

Upstream: [aadityasingh/icl-dynamics](https://github.com/aadityasingh/icl-dynamics); primary paper: [Singh et al., 2024](https://arxiv.org/abs/2404.07129). This repository does not claim ownership of the upstream work or grant a licence to its contents. See [Provenance](docs/PROVENANCE.md).
