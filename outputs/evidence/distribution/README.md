# Where next-character probability goes

Prerequisite: Run python scripts/run_paper.py distribution after the primary G0–G10 experiment.

Status: all expected inputs are present.

Probability mass and greedy validity are different measurements from freely sampled symbol validity. This diagnostic uses one model seed and shared validation prompts.

## Context-valid symbol probability mass

| Condition | Generation | Validation | Value |
| :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | 97.22 % |
| Real targets | 0 | Held-out handwriting | 38.17 % |
| Real targets | 10 | Held-out label assignments | 96.96 % |
| Real targets | 10 | Held-out handwriting | 35.46 % |
| Recursive | 0 | Held-out label assignments | 97.22 % |
| Recursive | 0 | Held-out handwriting | 38.17 % |
| Recursive | 10 | Held-out label assignments | 80.26 % |
| Recursive | 10 | Held-out handwriting | 25.71 % |

## Off-context symbol probability mass

| Condition | Generation | Validation | Value |
| :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | 2.78 % |
| Real targets | 0 | Held-out handwriting | 61.83 % |
| Real targets | 10 | Held-out label assignments | 3.04 % |
| Real targets | 10 | Held-out handwriting | 64.54 % |
| Recursive | 0 | Held-out label assignments | 2.78 % |
| Recursive | 0 | Held-out handwriting | 61.83 % |
| Recursive | 10 | Held-out label assignments | 19.74 % |
| Recursive | 10 | Held-out handwriting | 74.29 % |

## Greedy choice outside context

| Condition | Generation | Validation | Value |
| :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | 0.00 % |
| Real targets | 0 | Held-out handwriting | 50.00 % |
| Real targets | 10 | Held-out label assignments | 0.00 % |
| Real targets | 10 | Held-out handwriting | 50.78 % |
| Recursive | 0 | Held-out label assignments | 0.00 % |
| Recursive | 0 | Held-out handwriting | 50.00 % |
| Recursive | 10 | Held-out label assignments | 0.78 % |
| Recursive | 10 | Held-out handwriting | 60.16 % |

## Valid symbol mass with correct first label

| Condition | Generation | Validation | Value |
| :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | 97.22 % |
| Real targets | 0 | Held-out handwriting | 38.24 % |
| Real targets | 10 | Held-out label assignments | 96.96 % |
| Real targets | 10 | Held-out handwriting | 35.56 % |
| Recursive | 0 | Held-out label assignments | 97.22 % |
| Recursive | 0 | Held-out handwriting | 38.24 % |
| Recursive | 10 | Held-out label assignments | 80.27 % |
| Recursive | 10 | Held-out handwriting | 25.78 % |

## Mean per-prompt symbol entropy

| Condition | Generation | Validation | Value |
| :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | 0.799 nats |
| Real targets | 0 | Held-out handwriting | 1.793 nats |
| Real targets | 10 | Held-out label assignments | 0.833 nats |
| Real targets | 10 | Held-out handwriting | 1.885 nats |
| Recursive | 0 | Held-out label assignments | 0.799 nats |
| Recursive | 0 | Held-out handwriting | 1.793 nats |
| Recursive | 10 | Held-out label assignments | 1.304 nats |
| Recursive | 10 | Held-out handwriting | 2.358 nats |

## Global symbol entropy

| Condition | Generation | Validation | Value |
| :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | 3.779 nats |
| Real targets | 0 | Held-out handwriting | 3.583 nats |
| Real targets | 10 | Held-out label assignments | 3.774 nats |
| Real targets | 10 | Held-out handwriting | 3.632 nats |
| Recursive | 0 | Held-out label assignments | 3.779 nats |
| Recursive | 0 | Held-out handwriting | 3.583 nats |
| Recursive | 10 | Held-out label assignments | 3.684 nats |
| Recursive | 10 | Held-out handwriting | 3.543 nats |

## Source files

- [outputs/probes/distribution/clean_only/summary.json](../../probes/distribution/clean_only/summary.json)
- [outputs/probes/distribution/recursive_replacement/summary.json](../../probes/distribution/recursive_replacement/summary.json)
