# Assignment compliance audit — Option B symbolic research extension

## Relevant brief
The Wits NLP Project 1 brief explicitly says to **use the exact task from Singh et al. (2024)**, implement a two-layer transformer learning an induction-head circuit, run **both Base and Extended recursive versions**, evaluate with a **train/validation/test split**, interpret model circuitry and report methodology, results and limitations. Deliverables include **complete code, requirements.txt, README.txt, a two-page extended abstract PDF, a contribution statement, a completed NeurIPS 2024 ethics questionnaire and Faculty of Science AI Ethics Statement**. Due 27 Oct 2026 at 17:00.

## Is our preferred Option B allowed?
**Investigating an unanswered causal mechanism is within the scientific scope.** The brief welcomes additional adjustments and stresses rigorous reasoning over maximising accuracy.

**But two details require explicit justification, ideally lecturer confirmation:**

1. **Original task fidelity:** Singh et al. use pretrained **Omniglot ResNet18 exemplar features**, whereas the archived `nlp_ass_1` experiments use **50 learned symbolic embeddings**. Their sequence/mapping *logic* is similar, but a discrete-symbol analogue is **not literal identical data or architecture input**. To minimize marking risk, include a literal original-task reproduction or obtain lecturer agreement that the analogue counts.
2. **Meaning of 'adding synthetic data':** The archive's frozen protocol replaces targets with recursively generated synthetics. The brief describes repeatedly **adding synthetic data**. Full replacement and real-data accumulation can produce different outcomes and should not be silently conflated. Report replacement clearly; consider a fixed real-data mixture or explicit accumulation condition if required.

The upstream codebase is expressly allowed by the brief. Reusing code still requires attribution and compliance with local plagiarism requirements and redistribution rights.

## Existing work versus fresh evidence
The previous symbolic archive is **post-test**: its final test has already been evaluated and its hypotheses informed our plan. We can reproduce or critically audit it, but cannot present a reused final test as fresh confirmation. A new confirmatory experiment should freeze changes first, select meaningful independent held-out conditions, and label any post-hoc analysis exploratory.

## Short clarification to lecturer before finalising the scientific approach
"May we use a faithful two-way, one-shot, dynamically relabelled **symbolic analogue** of Singh et al. (2024), with mechanistic previous-token/induction-head experiments, rather than literal Omniglot ResNet18 embeddings? Our extension would test the causal role of recursively corrupted generated contexts. Also, is **full synthetic replacement** acceptable as the recursive pipeline, or do you specifically require accumulating synthetic data alongside original data?"

## Grading risks independent of research quality
- Missing required Base/Extended code or dataset split.
- Training/validation/test leakage.
- Describing correlational attention as proof of a causal mediator.
- Missing ethics statements (brief says the project will not be marked).
- Misrepresenting the symbolic analogue as a literal Omniglot reproduction.
- Reporting new experiments without source/provenance or claiming published source code as original.
