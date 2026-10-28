# Original model-collapse results (five paired seeds)

This report reads existing original experiment JSON. It does not rerun research.

| Study | Outcome | Paired effect | Descriptive 95% interval |
|:--|:--|--:|--:|
| Base G10 | Label accuracy (control minus recursive) | +12.19 pp | [3.29, 21.09] pp |
| Extended G10 | Sampled q2 validity (control minus recursive) | +19.73 pp | [14.24, 25.22] pp |
| Target repair | Valid-symbol mass (repaired minus raw) | +21.17 pp | [16.76, 25.58] pp |
| Target repair | Context-switch preference (repaired minus raw) | +5.94 logits | [5.39, 6.48] logits |

## Limitations

- G10 findings are validation-only: the original 100-class final test was already opened at G3.
- The synthetic loop replaces training targets, not the genuine Omniglot input context.
- The reported Base and Extended G10 primary comparisons each use 300,000 episodes per generation and five seed pairs; their metrics, tasks, and difficulty still differ.
- Base clean G10 accuracy is 61.56% in the matched-budget sample; one of five clean seeds is below 60%.
- The q2 vocabulary is the 50 training identities and is not defined for unseen class IDs.
- The five-seed intervals describe variation across model seeds; they are not a new independent test result.
- Repair changes all trainable weights; attention activation patching does not prove an isolated unique head mechanism.

## Provenance

- docs/experiments/g10_five_seed/g10_seed_summary.json
- docs/experiments/matched_300k/summary.json
- docs/experiments/q2_repair_mediation/summary.json
