# NLP Project 1: assignment requirements and submission audit

Source: Wits COMS4054A / COMS7062A / COMS7066A, *Project 1: Interpretability of Model Collapse*, pp. 1–7. This checklist is **current and evidence-based**, replacing the superseded October 14–15 development TODOs (retained on `research-archive-complete-20261025`).

**Due date:** 27 October 2026 at **17:00 South Africa time**.

## Required research implementation

| Brief requirement | Repository evidence | Honest status |
| --- | --- | --- |
| Original Singh induction-head task | Pinned upstream `external/icl-dynamics`, HDF5, Base training and validation checks | Implemented; historical runs recorded |
| Base recursive training | `scripts/base_recursive.py`, G1–G10 checkpoints/results and arms | Historical runs recorded |
| Extended three-output model and recursion | `src/extended_learner.py`, `src/extended_omniglot.py`, `scripts/extended_recursive.py` | Historical runs recorded |
| Train/validation/final-test split | 50 / 1,473 / 100 classes; role-specific evaluations | Implemented; original final test already accessed at G3 |
| Train and validation metrics | Per-generation original JSON/CSV plus five-seed analysis | Original evidence committed |
| Mechanistic interpretation | Head ablation, swapping and attention activation patching scripts and summaries | Original evidence committed |
| Appropriate controls and tuning | Clean, frozen-G0, half-real, recursive; recorded G0 learning-rate exploration | Original evidence committed; budget caveats apply |
| Complete source and reproducibility | `src/`, `scripts/`, `run.py`, `requirements.txt`, `README.txt`, `SETUP_GUIDE.md` | Source available; full clean-room replay not yet independently verified |

**Important:** The assignment emphasizes methodology and insight rather than high accuracy alone. The text must explain why each metric/control answers the research question and distinguish failure of numerical in-context retrieval from failures of contextual next-symbol generation. Attention heatmaps alone are not enough to claim a circuit.

## Required final deliverables

| Lecturer requirement | Current evidence | Remaining action |
| --- | --- | --- |
| Literal `README.txt` and `requirements.txt` | Both on `main` | Run clean-room setup and verify |
| Complete implementation | Executable source on `main` | Verify dependencies, checkpoints and CI |
| PDF extended abstract | Supplied 11-page PDF: first two pages are main text | Verify official Moodle template and final PDF version |
| References | Supplied PDF includes references after main text | Confirm citation and source attribution |
| **Separate individual contributions** | Names on title page; **no standalone contribution statement located** in uploaded PDF | **Add truthful, author-approved individual contributions** |
| **NeurIPS 2024 ethics questionnaire** | Uploaded PDF includes a lengthy checklist after references | Compare exact format/answers with official Moodle .tex; ensure complete |
| **Faculty of Science AI Ethics Statement** | Supplied PDF page 11 says official statement text is pending | **Insert the official Faculty statement and appropriate AI disclosure** |
| Preliminary contribution statement two weeks before due date | Not established by available source | Verify whether submitted by 13 October or contact lecturer |
| Moodle submission | No submission confirmation available | Package and actually submit prior to deadline |

The brief explicitly says that **missing the NeurIPS questionnaire means the project will not be marked**. Do not substitute a homemade summary for the lecturer-supplied template.

## Grading rubric

| Category | Weight | Highest-band emphasis |
| --- | ---: | --- |
| Structure/introduction | 10% | Clearly linked sections, concise academic writing, legible figures/captions |
| Background/related work | 10% | Self-contained, accurate and relevant prior work |
| Method | 30% | Specific architecture, training/data methodology, motivated experiment design |
| Results | 30% | Correctness demonstrated with carefully chosen metrics and readable graphics |
| Discussion | 20% | Evidence-grounded interpretation, mechanism insights and acknowledged limitations |

The written extended abstract is a **strict two-main-page limit including charts, equations and tables**. References and author contributions can use additional pages. The NeurIPS ethics questions must follow references **before** supplementary material. The Faculty declaration is also compulsory.

## Scientific claims requiring explicit qualification

- Synthetic generated *targets*, not entire generated Omniglot prompts.
- Extended q2 head is a 50-way training-character identity classifier and has no defined identity target for 1,473 held-out validation or 100 test classes.
- Base uses 1M training examples per descendant; Extended uses 300k.
- Report G10 as validation-only, not a newly sealed test.
- Model-seed paired t intervals with n=5 are descriptive and exploratory.
- Repair changes all learned model weights and improves context-sensitive outputs; patching shows a contribution by attention representations, not an isolated training-data-to-head mediation proof.
- A few control seeds have unexpectedly weak first-label accuracy; do not suppress these.

The research is available for lecturer review, but **the source record does not demonstrate completion of all submission and clean-room verification obligations**.
