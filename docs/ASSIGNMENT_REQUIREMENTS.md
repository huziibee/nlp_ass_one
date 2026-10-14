# Exhaustive checklist: NLP Project 1 (source: seven-page lecturer PDF)

**Brief:** Wits COMS4054A / COMS7062A / COMS7066A, *Project 1: Interpretability of Model Collapse*.
**Deadline:** **27 October 2026 at 17:00**. This checklist distinguishes explicit requirements from illustrative research suggestions. Source: `_NLP_2026__Mechanistic_Interp_Collapse_Project.pdf`, pp. 1-7 (available in the course context; do not redistribute it in this public repository).

## Explicit course requirements

- [ ] Group of **1 to 3** people. Submit a preliminary contribution statement **two weeks before project submission**; group composition cannot change after the brief's stated freeze period. If not already submitted, contact teaching staff promptly rather than pretending it was submitted.
- [x] Use the **exact Singh et al. (2024) two-way few-shot symbol/exemplar–label task** as the primary baseline; original authors' GitHub code is explicitly permitted. Current Base G0 uses their pinned original Omniglot/ResNet18 dataset and implementation.
- [x] Implement a transformer capable of learning the induction-head form of in-context learning. Original Base G0 and causal head probes run successfully, but repeated independent seeds remain outstanding.
- [ ] Implement the **Base recursive model-collapse pipeline**: the one-number query-label output, across model generations, with clearly described synthetic-data training regime.
- [ ] Implement the **Extended recursive model-collapse pipeline**: first query's numeric label, **next symbol**, then **that next symbol's numeric label** (three generated targets). Verify how a generated symbol can be valid and what its output space means.
- [ ] Analyse performance **over training** and **across recursive generations** with meaningful metrics; explicitly measure data-collapse and in-context-learning/circuit behaviour rather than conflating them.
- [x] Use genuine **train/validation/test partitions**: fitting on train, hyperparameter/model decisions on validation, protected final evaluation on test. Canonical split locked in `docs/SPLIT_DECISION.md`; final test not yet opened.
- [ ] **Optimise/tune hyperparameters** meaningfully and explain choices. The brief explicitly warns that obvious, omitted tuning can lose method marks even if raw model accuracy is not the primary grading objective.
- [ ] Apply **interpretability techniques**, with sound evidence about relevant internal computation; investigate the model's induction/previous-token heads and their changes. Base causal head ablations already exist; generational and Extended mechanistic comparisons remain outstanding.
- [ ] Record scientific choices, limitations, unsuccessful but informative experiments, training challenges, controls and working. Accuracy alone does not earn marks; conclusions must follow measurements.
- [ ] Submit complete **code implementation** of every claimed method/experiment.
- [ ] Submit a file literally named **`requirements.txt`** listing required packages.
- [ ] Submit a file literally named **`README.txt`** explaining the process to run the code.
- [ ] Submit an **extended abstract as a PDF** using the Moodle template, **strictly two main-text pages including figures, equations and tables**. References and contribution statement may occupy unlimited *additional* pages.
- [ ] Format the abstract as an **academic paper**: introduction, useful background/related work, clearly justified methodology, results and interpretations/discussion relevant to the precise question.
- [ ] Include **contribution statement** as required, including accurate division of group work.
- [ ] Complete and append the **NeurIPS 2024 ethics statement/questionnaire** from Moodle's `.tex` file **after the references and before any supplementary material**. **Missing this means the project will not be marked.**
- [ ] Include the **Faculty of Science AI Ethics Statement**. Disclose generative-AI assistance appropriately and take full responsibility for code and writing.
- [ ] Audit academic integrity, citation/attribution, figure captions, accuracy of scientific statements and originality. Generative AI is permitted for writing/code, but other plagiarism rules remain; poor writing/formatting is subject to negative marking and factual errors/hallucinations are heavily penalised.
- [ ] Final submission package contains `requirements.txt`, `README.txt`, complete code and **two-page PDF extended abstract** (with contribution and ethics appendices placed as required).

## Illustrative techniques/metrics suggested by the brief (not individually mandated)

- Train loss and accuracy; validation loss and accuracy; final test accuracy; perplexity; other measures chosen to answer a specific research question.
- Symbol embeddings, attention maps and ideally model circuits. Conmy et al. (2023) and Chen et al. (2026) are suggested further methods, not compulsory individual algorithms.
- Additional reasonable task/data/collapse adjustments and experiment tracking/plotting tools are allowed.
- A smaller sufficient dataset or clever efficient experiment is encouraged; maximum performance and huge datasets are not the objective.
- Poor/null outcomes can receive strong marks if the experimental reasoning and interpretation are rigorous.

## Rubric from page 7 (total 100%)

| Criterion | Weight | Top-band expectation |
| --- | ---: | --- |
| Write-up structure + introduction | **10%** | Excellent concise academic writing, structure, readable labelled figures, captions and legends. **Negative marking** |
| Background + related work | **10%** | Sufficiently detailed but concise, self-contained context; makes project's significance clear. **Negative marking** |
| Method | **30%** | Reproducible architecture/data/training/experiments with detailed rationale linking each decision to the research question |
| Results | **30%** | Thorough, appropriate metrics and clear visualisations that answer the question and establish correctness |
| Discussion | **20%** | Evidence-grounded interpretation that answers the topic, demonstrates insight about model mechanisms or original thought |

## Current blockers, in priority order

1. Obtain and submit/verify preliminary contribution statement; the **two-weeks-before-deadline date is 13 October 2026**, which has already passed as of this audit. Ask staff for remedy if missed.
2. Freeze Extended `q2` symbol-head semantics (see `docs/EXTENDED_OUTPUT_CONTRACT.md`).
3. Implement and validate Base **and** Extended recursively trained generations under at least a prespecified original+synthetic mixture (because the brief says 'adding synthetic data'), plus a clearly labelled replacement condition if useful.
4. Replicate seeds and mechanistic interventions across generations; explain error bars, controls, model selection and limitations.
5. Obtain actual Moodle report/NeurIPS ethics templates. Do **not** fabricate an official completed form.
6. Write, compile, inspect and submit the final two-page PDF and ethics/contribution appendices. Run from `README.txt` in a clean environment.

**Source nuance:** the PDF describes model-collapse behaviour broadly but intentionally leaves many design decisions to students. It does not specify a 70/15/15 or 80/10/10 ratio, number of seeds, generation count, mixing proportion or particular causal discovery algorithm. Those are research decisions, not lecturer mandates.
