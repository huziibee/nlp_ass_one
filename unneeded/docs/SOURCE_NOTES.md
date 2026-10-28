# Research reading notes — use and limits of each source

These notes distinguish published findings, prior archived observations and **future experimental proposals**. Sources are identified by their arXiv/PMLR/proceedings links and the local PDFs previously supplied; no PDF is redistributed in this public repo.

## Assignment (Wits 2026)
**Core obligations:** two-way episodically relabelled Singh-style transformer; both Base and Extended recursive training; train/validation/test; interpret attention and circuits; two-page abstract; full code, requirements and README.txt; ethics and contribution statements.
**Important ambiguity:** "exact task" versus symbolic-only approximation, and what precisely counts as **adding synthetic data**. See `ASSIGNMENT_COMPLIANCE.md`.

## Singh et al. 2024 — *What Needs to Go Right for an Induction Head?*
https://arxiv.org/abs/2404.07129
**What it establishes:** a compact two-layer attention-only model can learn match-and-copy operations from randomly relabelled Omniglot exemplars. First-layer previous-token pathways, second-layer Q/K match and value copying are interacting subcircuits.
**Already implemented for us:** `external/icl-dynamics/models.py`, `samplers.py`, `main.py`, `opto.py`, `ih_paper_runs.sh`, `ih_paper_plots.ipynb`, original HDF5 feature data.
**Use:** reference implementation, separate held-out matching versus copying tests, head-necessity and retain-only ablations.
**Avoid:** assuming head attention scores alone establish causal necessity.

## Olsson et al. 2022 — *In-context Learning and Induction Heads*
https://arxiv.org/abs/2209.11895
**Finding:** induction heads can implement a pattern analogous to A B ... A -> B through attention composition.
**Use:** mechanistic vocabulary and causal head interventions.
**Avoid:** claiming that every form of ICL uses the same induction circuit.

## Conmy et al. 2023 — *Towards Automated Circuit Discovery for Mechanistic Interpretability*
https://arxiv.org/abs/2304.14997
**Finding:** ACDC uses targeted activation interventions to identify task-relevant computational subgraphs.
**Use:** controlled clean/corrupted inputs, patching, explicit causal effect metrics, random component controls.
**Avoid:** equating circuit plots with proof that training-distribution changes caused circuit changes.

## Chen, Luo and Pan 2026 — *Mechanistic Data Attribution*
https://arxiv.org/abs/2601.21996
**Finding:** training examples can be associated with influence on particular interpretable model components; intervention/augmentation provides further evidence.
**Use:** motivation for dataset-level perturbations tied to a particular head or mechanism.
**Avoid:** implementing expensive influence-function attribution unless it answers a unique question.

## Shumailov et al. 2023 — *The Curse of Recursion*
https://arxiv.org/abs/2305.17493
**Finding:** recursive learning from generated examples can progressively lose information about rare events and drift from the original distribution.
**Use:** recursive generation definitions, distribution shift, finite-sampling/learning approximation concerns.
**Avoid:** treating this preprint as independent confirmation of its related 2024 article.

## Shumailov et al. 2024 — *AI Models Collapse When Trained on Recursively Generated Data*
https://www.nature.com/articles/s41586-024-07566-y
**Finding:** recursive synthetic training can accumulate distributional errors, especially if model samples replace authentic distributional support.
**Use:** background motivation for repeated generation and rare-tail metrics.
**Avoid:** asserting inevitability for all algorithms and every real/synthetic accumulation regime.

## Borji 2024 — *A Note on Shumailov et al. (2024)*
https://arxiv.org/abs/2410.12954
**Finding:** independent critique/illustration of recursive finite-sample distributional behaviour using simple density estimates.
**Use:** caution in interpreting KL divergence, variance and sampling effects.
**Avoid:** relying on separately chosen histogram bins for a claimed KL comparison; use shared support/bins.

## Alemohammad et al. 2024 — *Self-Consuming Generative Models Go MAD*
https://arxiv.org/abs/2307.01850
**Finding:** recursively trained generative models can show reduced output diversity and quality.
**Use:** separate diversity measures from performance measures.
**Avoid:** equating all diversity reduction with an unusable model.

## Gerstgrasser et al. 2024 — *Is Model Collapse Inevitable?*
https://arxiv.org/abs/2404.01413
**Finding:** accumulation of original alongside synthetic data changes the model-collapse analysis and may prevent collapse in specific conditions.
**Use:** explicitly compare real-data preservation, mixing and full replacement.
**Avoid:** treating real-data accumulation as a universal guarantee.

## Chan et al. 2022 — *Data Distributional Properties Drive Emergent In-Context Learning in Transformers*
https://arxiv.org/abs/2205.05055
**Finding:** distributional properties, including burstiness and frequency, affect whether ICL rather than in-weights learning emerges.
**Use:** conceptual alternative explanations and test design with remapping/shifted class statistics.
**Avoid:** assuming the same mechanism of competition under randomly remapped episodic labels.

## Singh et al. 2023 — *The Transient Nature of Emergent In-Context Learning in Transformers*
https://arxiv.org/abs/2311.08360
**Finding:** ICL can disappear while ordinary training loss continues to fall in settings where other strategies can solve training data.
**Use:** distinguish within-generation training transience from between-generation recursive feedback.
**Avoid:** assuming all ICL loss is caused by synthetic data.

## Singh et al. 2025 — *Strategy Coopetition Explains the Emergence and Transience of ICL*
https://arxiv.org/abs/2503.05631
**Finding:** alternative in-context/weight-based pathways can compete or cooperate under some training regimes.
**Use:** search for strategy switching as an alternative to complete circuit destruction.
**Avoid:** directly transferring observations from a different dataset and label-randomisation regime.

## Barzilai and Shamir 2025 — *When Models Don't Collapse: On the Consistency of Iterative MLE*
https://proceedings.neurips.cc/
**Finding:** iterative model training can be consistent without collapse under particular formal conditions.
**Use:** specify regime and assumptions before making universal claims.
**Avoid:** claiming every form of synthetic-data accumulation has a guarantee.

## Responsible paper usage
Scientific evidence flows from correctly identified source -> stated hypothesis -> measurable prediction -> matched contrast -> uncertainty -> appropriately bounded inference. Neither a large bibliography nor reproduced plotting code substitutes for new independent evidence.
