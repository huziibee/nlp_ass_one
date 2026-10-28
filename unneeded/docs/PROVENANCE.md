# Provenance, attribution and reuse boundaries

**Original researchers:** Aaditya K. Singh and collaborators, [icl-dynamics](https://github.com/aadityasingh/icl-dynamics), associated with *What Needs to Go Right for an Induction Head?* (ICML 2024).

This workspace **references** the original codebase via a version-pinned Git submodule. It does not claim authorship of upstream source code, dataset feature extraction, or academic results. The upstream repo currently has no top-level LICENSE file in the inspected root; do **not** infer permission to republish or relicense copied files merely because the repository is public. Submodule use retains authorship and source context.

**Historical student archive:** `huziibee/nlp_ass_1` (private). It contains a completed, post-final-test symbolic adaptation, including research notes, code, model checkpoints, plots, and audits. No sealed test data, private course PDFs, or archived material is copied into the present public repository.

**Why no literal original HDF5 binary here?** It is supplied *through the pinned submodule*. A `git clone --recurse-submodules` operation will fetch it from upstream. The working tree sees the data in `external/icl-dynamics/` automatically, without duplicated source ownership.

**Study boundaries:** Published article claims, earlier archived findings, and future Option B experiments must be labelled distinctly. A successful old experiment is not independent evidence of a novel result in the new work. New confirmatory work needs a fresh protocol and a protected test, and should acknowledge prior exposure to old results.

Assignment includes attribution, contribution statement, NeurIPS ethics questionnaire, faculty AI ethics statement, full code, reproducibility instructions and a two-page extended abstract.
