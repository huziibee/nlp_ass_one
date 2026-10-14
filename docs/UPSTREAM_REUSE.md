# Reuse inventory — which original research components already exist?

The source files below are provided **unmodified by the pinned upstream Git submodule**, at `external/icl-dynamics/` after a recursive clone.

| Upstream file | Original functionality | Our likely use |
| --- | --- | --- |
| `omniglot_resnet18_randomized_order_s0.h5` | Precomputed ResNet18 exemplar embeddings | Actual Singh task input |
| `omni_features_extract.py`, `omniglot_dataset.py` | Download/preprocess original Omniglot | Only if we must reconstruct image-to-feature mapping |
| `samplers.py` | Episodes, class/exemplar sampling, random contextual relabelling | No reason to rewrite for literal reproduction |
| `models.py` | Attention-only causal model with cached activations | Mechanistic model baseline |
| `main.py`, `main_utils.py` | Training, validation, checkpoints, CLI options | Original published training configuration |
| `opto.py`, `artificial_optogenetics_guide.md` | Artificial optogenetics, activation clamps/ablation | Within-model matching/copying interventions |
| `ih_paper_runs.sh` | Published figures' run definitions | Source of baseline hyperparameters, evaluation splits |
| `ih_paper_plots.ipynb`, `ih_paper_plot_utils.py` | Published results/visuals | Verify reproduction without rebuilding plotting |
| `ih_paper_appendix_runs.sh`, `ih_paper_appendix_plots.ipynb` | Additional original experiments | Optional robustness/interpretability |
| `simple_model_solver.py`, `coopetition_model_solver.py` | Toy mathematical learning-model tooling | Optional alternative-mechanism baselines |

### Critical distinction

The private archival `nlp_ass_1` project already has its *own* symbolic transformer and G0–G10 Base/Extended experiments. Its model inputs are **learned discrete symbols**, unlike the original upstream fixed Omniglot features. Reproduction, comparison and extension of the private study are separate work from a strict published-task reproduction.

### Why we are not rewriting these modules immediately

A complete new transformer and optogenetics toolkit would create extensive implementation and validation work while contributing little new science. Our potential contribution is **which source of synthetic training-data corruption creates a measurable matching error**. Reuse the proven basic mechanics with attribution, and implement only missing experimental controls after the hypothesis and data contracts have been approved.

### Suggested reading order for implementation
1. `README.md` in the upstream project: architecture and data setup.
2. `ih_paper_runs.sh`: actual published experimental arguments.
3. `samplers.py`: how contextual mappings are generated.
4. `models.py` + optogenetics guide: how to measure and intervene on circuit activations.
5. Local `docs/ASSIGNMENT_COMPLIANCE.md` and `docs/OPTION_B_PLAN.md`: our new scientific constraints.

**Not implemented in this workspace yet:** new recursive-model training pipeline, new controlled dataset-repair experiments, new protected confirmatory test, mechanism-causality comparison, two-page submission abstract or ethics statements. The historical *private* repository contains earlier versions of many of those components but is not imported into this public workspace.
