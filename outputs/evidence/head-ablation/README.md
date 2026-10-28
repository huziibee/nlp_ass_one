# Single-head ablation evidence

Prerequisite: Run python scripts/run_paper.py head-ablation after the primary G0–G10 experiment.

Status: all expected inputs are present.

Change (pp) is intervention minus the natural model on the same prompts. A negative value means the intervention reduced the metric. Candidate heads are selected on G0 training prompts, not validation outcomes. Head indices are zero-based. Ablations are one-seed component tests; they do not prove a unique circuit.

## Previous-token attention

| Condition | Generation | Validation | Intervention | Value |
| :-- | :-- | :-- | :-- | :-- |
| Real targets | 0 | Training prompts used for head selection | Layer 1, head 4 | 0.927 score |
| Recursive | 0 | Training prompts used for head selection | Layer 1, head 4 | 0.927 score |

## First-label matching margin

| Condition | Generation | Validation | Intervention | Value |
| :-- | :-- | :-- | :-- | :-- |
| Real targets | 0 | Training prompts used for head selection | Layer 2, head 1 | 0.873 score |
| Real targets | 0 | Held-out label assignments | Layer 2, head 1 | 0.900 score |
| Real targets | 0 | Held-out handwriting | Layer 2, head 1 | 0.672 score |
| Real targets | 0 | Held-out character classes | Layer 2, head 1 | 0.771 score |
| Real targets | 10 | Held-out label assignments | Layer 2, head 1 | 0.886 score |
| Real targets | 10 | Held-out handwriting | Layer 2, head 1 | 0.604 score |
| Real targets | 10 | Held-out character classes | Layer 2, head 1 | 0.753 score |
| Recursive | 0 | Training prompts used for head selection | Layer 2, head 1 | 0.873 score |
| Recursive | 0 | Held-out label assignments | Layer 2, head 1 | 0.900 score |
| Recursive | 0 | Held-out handwriting | Layer 2, head 1 | 0.672 score |
| Recursive | 0 | Held-out character classes | Layer 2, head 1 | 0.771 score |
| Recursive | 10 | Held-out label assignments | Layer 2, head 1 | 0.931 score |
| Recursive | 10 | Held-out handwriting | Layer 2, head 1 | 0.669 score |
| Recursive | 10 | Held-out character classes | Layer 2, head 1 | 0.768 score |

## Second-label matching margin

| Condition | Generation | Validation | Intervention | Value |
| :-- | :-- | :-- | :-- | :-- |
| Real targets | 0 | Training prompts used for head selection | Layer 2, head 1 | 0.862 score |
| Real targets | 0 | Held-out label assignments | Layer 2, head 1 | 0.894 score |
| Real targets | 0 | Held-out handwriting | Layer 2, head 1 | 0.780 score |
| Real targets | 0 | Held-out character classes | Layer 2, head 1 | 0.758 score |
| Real targets | 10 | Held-out label assignments | Layer 2, head 1 | 0.850 score |
| Real targets | 10 | Held-out handwriting | Layer 2, head 1 | 0.741 score |
| Real targets | 10 | Held-out character classes | Layer 2, head 1 | 0.709 score |
| Recursive | 0 | Training prompts used for head selection | Layer 2, head 1 | 0.862 score |
| Recursive | 0 | Held-out label assignments | Layer 2, head 1 | 0.894 score |
| Recursive | 0 | Held-out handwriting | Layer 2, head 1 | 0.780 score |
| Recursive | 0 | Held-out character classes | Layer 2, head 1 | 0.758 score |
| Recursive | 10 | Held-out label assignments | Layer 2, head 1 | 0.841 score |
| Recursive | 10 | Held-out handwriting | Layer 2, head 1 | 0.739 score |
| Recursive | 10 | Held-out character classes | Layer 2, head 1 | 0.687 score |

## First numerical-label accuracy

| Condition | Generation | Validation | Intervention | Value | Change (pp) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | Unmodified | 99.22 % | 0.00 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 73.44 % | -25.78 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 73.44 % | -25.78 |
| Real targets | 0 | Held-out handwriting | Unmodified | 91.41 % | 0.00 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 76.56 % | -14.84 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 76.56 % | -14.84 |
| Real targets | 0 | Held-out character classes | Unmodified | 96.09 % | 0.00 |
| Real targets | 0 | Held-out character classes | Remove layer 2 head 1 (first-label candidate) | 82.03 % | -14.06 |
| Real targets | 0 | Held-out character classes | Remove layer 2 head 1 (second-label candidate) | 82.03 % | -14.06 |
| Real targets | 10 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 93.75 % | -6.25 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 93.75 % | -6.25 |
| Real targets | 10 | Held-out handwriting | Unmodified | 91.41 % | 0.00 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 85.16 % | -6.25 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 85.16 % | -6.25 |
| Real targets | 10 | Held-out character classes | Unmodified | 96.09 % | 0.00 |
| Real targets | 10 | Held-out character classes | Remove layer 2 head 1 (first-label candidate) | 92.19 % | -3.91 |
| Real targets | 10 | Held-out character classes | Remove layer 2 head 1 (second-label candidate) | 92.19 % | -3.91 |
| Recursive | 0 | Held-out label assignments | Unmodified | 99.22 % | 0.00 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 73.44 % | -25.78 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 73.44 % | -25.78 |
| Recursive | 0 | Held-out handwriting | Unmodified | 91.41 % | 0.00 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 76.56 % | -14.84 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 76.56 % | -14.84 |
| Recursive | 0 | Held-out character classes | Unmodified | 96.09 % | 0.00 |
| Recursive | 0 | Held-out character classes | Remove layer 2 head 1 (first-label candidate) | 82.03 % | -14.06 |
| Recursive | 0 | Held-out character classes | Remove layer 2 head 1 (second-label candidate) | 82.03 % | -14.06 |
| Recursive | 10 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 92.19 % | -7.81 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 92.19 % | -7.81 |
| Recursive | 10 | Held-out handwriting | Unmodified | 93.75 % | 0.00 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 85.94 % | -7.81 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 85.94 % | -7.81 |
| Recursive | 10 | Held-out character classes | Unmodified | 99.22 % | 0.00 |
| Recursive | 10 | Held-out character classes | Remove layer 2 head 1 (first-label candidate) | 94.53 % | -4.69 |
| Recursive | 10 | Held-out character classes | Remove layer 2 head 1 (second-label candidate) | 94.53 % | -4.69 |

## Teacher-forced second-label accuracy

| Condition | Generation | Validation | Intervention | Value | Change (pp) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 68.75 % | -31.25 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 68.75 % | -31.25 |
| Real targets | 0 | Held-out handwriting | Unmodified | 97.66 % | 0.00 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 78.91 % | -18.75 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 78.91 % | -18.75 |
| Real targets | 0 | Held-out character classes | Unmodified | 94.53 % | 0.00 |
| Real targets | 0 | Held-out character classes | Remove layer 2 head 1 (first-label candidate) | 71.88 % | -22.66 |
| Real targets | 0 | Held-out character classes | Remove layer 2 head 1 (second-label candidate) | 71.88 % | -22.66 |
| Real targets | 10 | Held-out label assignments | Unmodified | 99.22 % | 0.00 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 92.19 % | -7.03 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 92.19 % | -7.03 |
| Real targets | 10 | Held-out handwriting | Unmodified | 96.09 % | 0.00 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 91.41 % | -4.69 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 91.41 % | -4.69 |
| Real targets | 10 | Held-out character classes | Unmodified | 97.66 % | 0.00 |
| Real targets | 10 | Held-out character classes | Remove layer 2 head 1 (first-label candidate) | 90.62 % | -7.03 |
| Real targets | 10 | Held-out character classes | Remove layer 2 head 1 (second-label candidate) | 90.62 % | -7.03 |
| Recursive | 0 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 68.75 % | -31.25 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 68.75 % | -31.25 |
| Recursive | 0 | Held-out handwriting | Unmodified | 97.66 % | 0.00 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 78.91 % | -18.75 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 78.91 % | -18.75 |
| Recursive | 0 | Held-out character classes | Unmodified | 94.53 % | 0.00 |
| Recursive | 0 | Held-out character classes | Remove layer 2 head 1 (first-label candidate) | 71.88 % | -22.66 |
| Recursive | 0 | Held-out character classes | Remove layer 2 head 1 (second-label candidate) | 71.88 % | -22.66 |
| Recursive | 10 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 96.88 % | -3.12 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 96.88 % | -3.12 |
| Recursive | 10 | Held-out handwriting | Unmodified | 98.44 % | 0.00 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 94.53 % | -3.91 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 94.53 % | -3.91 |
| Recursive | 10 | Held-out character classes | Unmodified | 96.09 % | 0.00 |
| Recursive | 10 | Held-out character classes | Remove layer 2 head 1 (first-label candidate) | 94.53 % | -1.56 |
| Recursive | 10 | Held-out character classes | Remove layer 2 head 1 (second-label candidate) | 94.53 % | -1.56 |

## Context-valid symbol probability mass

| Condition | Generation | Validation | Intervention | Value | Change (pp) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | Unmodified | 96.91 % | 0.00 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 95.21 % | -1.71 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 95.21 % | -1.71 |
| Real targets | 0 | Held-out handwriting | Unmodified | 38.68 % | 0.00 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 32.92 % | -5.76 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 32.92 % | -5.76 |
| Real targets | 10 | Held-out label assignments | Unmodified | 96.72 % | 0.00 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 95.78 % | -0.94 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 95.78 % | -0.94 |
| Real targets | 10 | Held-out handwriting | Unmodified | 38.50 % | 0.00 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 34.43 % | -4.07 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 34.43 % | -4.07 |
| Recursive | 0 | Held-out label assignments | Unmodified | 96.91 % | 0.00 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 95.21 % | -1.71 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 95.21 % | -1.71 |
| Recursive | 0 | Held-out handwriting | Unmodified | 38.68 % | 0.00 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 32.92 % | -5.76 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 32.92 % | -5.76 |
| Recursive | 10 | Held-out label assignments | Unmodified | 79.07 % | 0.00 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 head 1 (first-label candidate) | 75.05 % | -4.02 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 head 1 (second-label candidate) | 75.05 % | -4.02 |
| Recursive | 10 | Held-out handwriting | Unmodified | 24.17 % | 0.00 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 head 1 (first-label candidate) | 21.01 % | -3.16 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 head 1 (second-label candidate) | 21.01 % | -3.16 |

## Source files

- [outputs/probes/head-selection/clean_only.json](../../probes/head-selection/clean_only.json)
- [outputs/probes/head-selection/recursive_replacement.json](../../probes/head-selection/recursive_replacement.json)
