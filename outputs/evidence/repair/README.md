# Repairing generated targets

Prerequisite: Run python scripts/run_paper.py repair after the clean and recursive G0–G10 experiments.

Status: all expected inputs are present.

Only completed, observed seed runs appear below. Source-error percentages use the generated 300,000-answer training stream and their categories may overlap. Joint repair changes both invalid symbols and their associated numbers, so it does not isolate a symbol-only mediated effect. A partial seed set is descriptive evidence, not the paper’s five-seed estimate.

## First numerical-label accuracy

| Condition | Seed | Generation | Validation | Value | Change (pp) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Recursive | 6001 | 10 | Held-out label assignments | 99.41 % |  |
| Joint target repair | 6001 | 10 | Held-out label assignments | 99.41 % | 0.00 |
| Recursive | 5054 | 10 | Held-out label assignments | 78.91 % |  |
| Joint target repair | 5054 | 10 | Held-out label assignments | 91.02 % | 12.11 |
| Recursive | 6054 | 10 | Held-out label assignments | 98.83 % |  |
| Joint target repair | 6054 | 10 | Held-out label assignments | 98.83 % | 0.00 |
| Recursive | 7054 | 10 | Held-out label assignments | 99.61 % |  |
| Joint target repair | 7054 | 10 | Held-out label assignments | 99.41 % | -0.20 |
| Recursive | 8054 | 10 | Held-out label assignments | 70.12 % |  |
| Joint target repair | 8054 | 10 | Held-out label assignments | 42.77 % | -27.34 |

## Teacher-forced second-label accuracy

| Condition | Seed | Generation | Validation | Value | Change (pp) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Recursive | 6001 | 10 | Held-out label assignments | 99.22 % |  |
| Joint target repair | 6001 | 10 | Held-out label assignments | 99.22 % | 0.00 |
| Recursive | 5054 | 10 | Held-out label assignments | 67.97 % |  |
| Joint target repair | 5054 | 10 | Held-out label assignments | 81.64 % | 13.67 |
| Recursive | 6054 | 10 | Held-out label assignments | 98.05 % |  |
| Joint target repair | 6054 | 10 | Held-out label assignments | 99.41 % | 1.37 |
| Recursive | 7054 | 10 | Held-out label assignments | 99.02 % |  |
| Joint target repair | 7054 | 10 | Held-out label assignments | 99.02 % | 0.00 |
| Recursive | 8054 | 10 | Held-out label assignments | 60.55 % |  |
| Joint target repair | 8054 | 10 | Held-out label assignments | 36.72 % | -23.83 |

## Context-valid symbol probability mass

| Condition | Seed | Generation | Validation | Value | Change (pp) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Recursive | 6001 | 10 | Held-out label assignments | 78.40 % |  |
| Joint target repair | 6001 | 10 | Held-out label assignments | 97.47 % | 19.07 |
| Recursive | 5054 | 10 | Held-out label assignments | 72.92 % |  |
| Joint target repair | 5054 | 10 | Held-out label assignments | 96.26 % | 23.34 |
| Recursive | 6054 | 10 | Held-out label assignments | 77.10 % |  |
| Joint target repair | 6054 | 10 | Held-out label assignments | 96.89 % | 19.79 |
| Recursive | 7054 | 10 | Held-out label assignments | 77.99 % |  |
| Joint target repair | 7054 | 10 | Held-out label assignments | 97.36 % | 19.37 |
| Recursive | 8054 | 10 | Held-out label assignments | 73.00 % |  |
| Joint target repair | 8054 | 10 | Held-out label assignments | 96.90 % | 23.90 |

## Sampled symbol validity

| Condition | Seed | Generation | Validation | Value | Change (pp) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Recursive | 6001 | 10 | Held-out label assignments | 79.49 % |  |
| Joint target repair | 6001 | 10 | Held-out label assignments | 97.46 % | 17.97 |
| Recursive | 5054 | 10 | Held-out label assignments | 72.66 % |  |
| Joint target repair | 5054 | 10 | Held-out label assignments | 95.31 % | 22.66 |
| Recursive | 6054 | 10 | Held-out label assignments | 78.91 % |  |
| Joint target repair | 6054 | 10 | Held-out label assignments | 97.85 % | 18.95 |
| Recursive | 7054 | 10 | Held-out label assignments | 78.12 % |  |
| Joint target repair | 7054 | 10 | Held-out label assignments | 96.88 % | 18.75 |
| Recursive | 8054 | 10 | Held-out label assignments | 72.07 % |  |
| Joint target repair | 8054 | 10 | Held-out label assignments | 97.46 % | 25.39 |

## Valid symbol and correct second number

| Condition | Seed | Generation | Validation | Value | Change (pp) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Recursive | 6001 | 10 | Held-out label assignments | 76.56 % |  |
| Joint target repair | 6001 | 10 | Held-out label assignments | 94.34 % | 17.77 |
| Recursive | 5054 | 10 | Held-out label assignments | 43.55 % |  |
| Joint target repair | 5054 | 10 | Held-out label assignments | 66.21 % | 22.66 |
| Recursive | 6054 | 10 | Held-out label assignments | 72.46 % |  |
| Joint target repair | 6054 | 10 | Held-out label assignments | 90.82 % | 18.36 |
| Recursive | 7054 | 10 | Held-out label assignments | 70.90 % |  |
| Joint target repair | 7054 | 10 | Held-out label assignments | 90.62 % | 19.73 |
| Recursive | 8054 | 10 | Held-out label assignments | 47.66 % |  |
| Joint target repair | 8054 | 10 | Held-out label assignments | 38.67 % | -8.98 |

## Source target: off-context character

| Condition | Seed | Generation | Value |
| :-- | :-- | :-- | :-- |
| Recursive | 6001 | 9 | 17.76 % |
| Recursive | 5054 | 9 | 21.21 % |
| Recursive | 6054 | 9 | 21.68 % |
| Recursive | 7054 | 9 | 19.78 % |
| Recursive | 8054 | 9 | 25.30 % |

## Source target: valid character but wrong second number

| Condition | Seed | Generation | Value |
| :-- | :-- | :-- | :-- |
| Recursive | 6001 | 9 | 1.92 % |
| Recursive | 5054 | 9 | 3.92 % |
| Recursive | 6054 | 9 | 1.77 % |
| Recursive | 7054 | 9 | 2.01 % |
| Recursive | 8054 | 9 | 5.14 % |

## Source target: wrong first number

| Condition | Seed | Generation | Value |
| :-- | :-- | :-- | :-- |
| Recursive | 6001 | 9 | 1.91 % |
| Recursive | 5054 | 9 | 3.36 % |
| Recursive | 6054 | 9 | 2.02 % |
| Recursive | 7054 | 9 | 1.55 % |
| Recursive | 8054 | 9 | 5.40 % |

## Five-seed matched comparison: Context-valid symbol probability mass

| Condition | Generation | Validation | Value | 95% paired interval |
| :-- | :-- | :-- | :-- | :-- |
| Recursive mean | 10 | Held-out label assignments | 76.02 % |  |
| Repaired mean | 10 | Held-out label assignments | 96.95 % |  |
| Paired repair effect | 10 | Held-out label assignments | 20.93 pp | [19.07, 22.79] |

## Five-seed matched comparison: Context-switch logit response

| Condition | Generation | Validation | Value | 95% paired interval |
| :-- | :-- | :-- | :-- | :-- |
| Recursive mean | 10 | Held-out label assignments | 9.46 logits |  |
| Repaired mean | 10 | Held-out label assignments | 15.21 logits |  |
| Paired repair effect | 10 | Held-out label assignments | 5.75 logits | [5.14, 6.36] |

## Five-seed matched comparison: Fraction transferred by layer 2

| Condition | Generation | Validation | Value | 95% paired interval |
| :-- | :-- | :-- | :-- | :-- |
| Recursive mean | 10 | Held-out label assignments | 81.10 % |  |
| Repaired mean | 10 | Held-out label assignments | 80.32 % |  |
| Paired repair effect | 10 | Held-out label assignments | -0.77 pp | [-4.11, 2.57] |

## Five-seed matched comparison: Layer 2 raw patch margin shift

| Condition | Generation | Validation | Value | 95% paired interval |
| :-- | :-- | :-- | :-- | :-- |
| Recursive mean | 10 | Held-out label assignments | -7.76 logits |  |
| Repaired mean | 10 | Held-out label assignments | -12.26 logits |  |
| Paired repair effect | 10 | Held-out label assignments | -4.50 logits | [-5.96, -3.04] |

## Source files

- [outputs/extended_5054_recursive_replacement/recursive_replacement/g10/metrics.json](../../extended_5054_recursive_replacement/recursive_replacement/g10/metrics.json)
- [outputs/extended_6054_recursive_replacement/recursive_replacement/g10/metrics.json](../../extended_6054_recursive_replacement/recursive_replacement/g10/metrics.json)
- [outputs/extended_7054_recursive_replacement/recursive_replacement/g10/metrics.json](../../extended_7054_recursive_replacement/recursive_replacement/g10/metrics.json)
- [outputs/extended_8054_recursive_replacement/recursive_replacement/g10/metrics.json](../../extended_8054_recursive_replacement/recursive_replacement/g10/metrics.json)
- [outputs/extended_primary/recursive_replacement/g10/metrics.json](../../extended_primary/recursive_replacement/g10/metrics.json)
- [outputs/q2_study/mediation/summary.json](../../q2_study/mediation/summary.json)
- [outputs/q2_study/repairs/5054/fix_invalid_q2/g10/metrics.json](../../q2_study/repairs/5054/fix_invalid_q2/g10/metrics.json)
- [outputs/q2_study/repairs/6001/fix_invalid_q2/g10/metrics.json](../../q2_study/repairs/6001/fix_invalid_q2/g10/metrics.json)
- [outputs/q2_study/repairs/6054/fix_invalid_q2/g10/metrics.json](../../q2_study/repairs/6054/fix_invalid_q2/g10/metrics.json)
- [outputs/q2_study/repairs/7054/fix_invalid_q2/g10/metrics.json](../../q2_study/repairs/7054/fix_invalid_q2/g10/metrics.json)
- [outputs/q2_study/repairs/8054/fix_invalid_q2/g10/metrics.json](../../q2_study/repairs/8054/fix_invalid_q2/g10/metrics.json)
