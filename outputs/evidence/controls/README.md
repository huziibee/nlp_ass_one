# Training controls and learning-rate pilot

Prerequisite: Run python scripts/run_paper.py controls after the primary Extended G0 checkpoint exists.

Status: all expected inputs are present.

Fixed-generator and mixed-target controls use the primary 300,000-episode protocol. The accumulating-corpus pilot uses 100,000 episodes per generation and is not directly comparable. The learning-rate comparison uses one seed.

## First numerical-label accuracy

| Condition | Seed | Generation | Validation | Value |
| :-- | :-- | :-- | :-- | :-- |
| Learning rate 0.0001 | 4054 | 0 | Held-out label assignments | 99.80 % |
| Learning rate 0.0003 | 4054 | 0 | Held-out label assignments | 100.00 % |
| Fixed G0 generator | 6001 | 10 | Held-out label assignments | 99.61 % |
| Half real targets | 6001 | 10 | Held-out label assignments | 99.61 % |
| Real-target history | 7712 | 10 | Held-out label assignments | 97.27 % |
| Accumulating recursive corpus | 7712 | 10 | Held-out label assignments | 83.20 % |

## Teacher-forced second-label accuracy

| Condition | Seed | Generation | Validation | Value |
| :-- | :-- | :-- | :-- | :-- |
| Learning rate 0.0001 | 4054 | 0 | Held-out label assignments | 99.22 % |
| Learning rate 0.0003 | 4054 | 0 | Held-out label assignments | 100.00 % |
| Fixed G0 generator | 6001 | 10 | Held-out label assignments | 99.61 % |
| Half real targets | 6001 | 10 | Held-out label assignments | 99.61 % |
| Real-target history | 7712 | 10 | Held-out label assignments | 98.83 % |
| Accumulating recursive corpus | 7712 | 10 | Held-out label assignments | 87.11 % |

## Context-valid symbol probability mass

| Condition | Seed | Generation | Validation | Value |
| :-- | :-- | :-- | :-- | :-- |
| Learning rate 0.0001 | 4054 | 0 | Held-out label assignments | 94.25 % |
| Learning rate 0.0003 | 4054 | 0 | Held-out label assignments | 96.96 % |
| Fixed G0 generator | 6001 | 10 | Held-out label assignments | 93.87 % |
| Half real targets | 6001 | 10 | Held-out label assignments | 94.85 % |
| Real-target history | 7712 | 10 | Held-out label assignments | 91.16 % |
| Accumulating recursive corpus | 7712 | 10 | Held-out label assignments | 83.99 % |

## Sampled symbol validity

| Condition | Seed | Generation | Validation | Value |
| :-- | :-- | :-- | :-- | :-- |
| Learning rate 0.0001 | 4054 | 0 | Held-out label assignments | 93.36 % |
| Learning rate 0.0003 | 4054 | 0 | Held-out label assignments | 96.48 % |
| Fixed G0 generator | 6001 | 10 | Held-out label assignments | 94.92 % |
| Half real targets | 6001 | 10 | Held-out label assignments | 94.53 % |
| Real-target history | 7712 | 10 | Held-out label assignments | 87.11 % |
| Accumulating recursive corpus | 7712 | 10 | Held-out label assignments | 83.20 % |

## Valid symbol and correct second number

| Condition | Seed | Generation | Validation | Value |
| :-- | :-- | :-- | :-- | :-- |
| Learning rate 0.0001 | 4054 | 0 | Held-out label assignments | 91.02 % |
| Learning rate 0.0003 | 4054 | 0 | Held-out label assignments | 96.09 % |
| Fixed G0 generator | 6001 | 10 | Held-out label assignments | 93.16 % |
| Half real targets | 6001 | 10 | Held-out label assignments | 93.95 % |
| Real-target history | 7712 | 10 | Held-out label assignments | 82.03 % |
| Accumulating recursive corpus | 7712 | 10 | Held-out label assignments | 64.06 % |

## Source files

- [outputs/cumulative/clean_history/g10/metrics.json](../../cumulative/clean_history/g10/metrics.json)
- [outputs/cumulative/cumulative_recursive/g10/metrics.json](../../cumulative/cumulative_recursive/g10/metrics.json)
- [outputs/extended_controls/frozen_g0/frozen_g0/g10/metrics.json](../../extended_controls/frozen_g0/frozen_g0/g10/metrics.json)
- [outputs/extended_controls/real_anchor_half/real_anchor_half/g10/metrics.json](../../extended_controls/real_anchor_half/real_anchor_half/g10/metrics.json)
- [outputs/lr_pilot/0.0001/metrics.json](../../lr_pilot/0.0001/metrics.json)
- [outputs/lr_pilot/0.0003/metrics.json](../../lr_pilot/0.0003/metrics.json)
