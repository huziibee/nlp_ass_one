# Proposed parallel workstreams — awaiting agreement

Use this file to agree scope **before** writing prompts to any coding agent. No work in the table below has been delegated.

| Role | Can start now? | Target | Non-overlapping ownership |
| --- | --- | --- | --- |
| User: data investigator | YES | Explore official feature data, surprising examples, similarities, source-paper visuals | Notebook only, no model training |
| Assistant: scientific auditor | YES | Map historical claims to actual files, verify whether published circuit interventions isolate matching vs copying | Notes/review only; don't edit archived results |
| Coding model: environment/reproduction | AFTER AGREEMENT | Port and run existing symbolic tests, document expected vs measured outputs | Reproduction scripts and compatibility fixes only |
| Coding model: data repair experiment | AFTER SCIENTIFIC PROTOCOL | Controlled A/B/C/D training-set construction with unit tests | New data-intervention module |
| Coding model: mechanistic evaluator | AFTER SCIENTIFIC PROTOCOL | Causal match metrics/ablations on fixed test configurations | Evaluation/intervention module |
| Assistant + user: synthesis | AFTER RESULT INSPECTION | Hypothesis register, limitations and two-page abstract | Research write-up |

## Coordination rules
- Work on separate branches/PRs. No parallel edits to the same implementation module.
- Every PR states what question it addresses, its tests, seed/config, changed files, and evidence.
- Review correctness and scientific validity before merging.
- Keep archived `nlp_ass_1` untouched.
- Do not claim new results from inspecting the old project's already-evaluated final test.
- No prompts to coding models until user approves assignments.
