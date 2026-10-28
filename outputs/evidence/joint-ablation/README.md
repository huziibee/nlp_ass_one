# Joint-head ablation evidence

Prerequisite: Run python scripts/run_paper.py joint-ablation after the primary G0–G10 experiment.

Status: all expected inputs are present.

Change (pp) is intervention minus the natural model on the same prompts. A negative value means the intervention reduced the metric. Candidate heads are selected on G0 training prompts, not validation outcomes. Head indices are zero-based. Ablations are one-seed component tests; they do not prove a unique circuit.

## First numerical-label accuracy

| Condition | Generation | Validation | Intervention | Value | Change (pp) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1 | 71.88 % | -28.12 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 0 | 96.88 % | -3.12 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1, 0 | 32.03 % | -67.97 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 20.31 % | -79.69 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1, 7 | 52.34 % | -47.66 |
| Real targets | 10 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1 | 94.53 % | -5.47 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 0 | 100.00 % | 0.00 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1, 0 | 92.97 % | -7.03 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 46.09 % | -53.91 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1, 7 | 57.03 % | -42.97 |
| Real targets | 0 | Held-out handwriting | Unmodified | 86.72 % | 0.00 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1 | 71.88 % | -14.84 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 0 | 85.94 % | -0.78 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1, 0 | 44.53 % | -42.19 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 28.91 % | -57.81 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1, 7 | 67.19 % | -19.53 |
| Real targets | 10 | Held-out handwriting | Unmodified | 87.50 % | 0.00 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1 | 80.47 % | -7.03 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 0 | 89.06 % | 1.56 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1, 0 | 80.47 % | -7.03 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 55.47 % | -32.03 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1, 7 | 66.41 % | -21.09 |
| Real targets | 0 | Held-out character classes | Unmodified | 98.44 % | 0.00 |
| Real targets | 0 | Held-out character classes | Remove layer 2 heads 1 | 78.91 % | -19.53 |
| Real targets | 0 | Held-out character classes | Remove layer 2 heads 0 | 97.66 % | -0.78 |
| Real targets | 0 | Held-out character classes | Remove layer 2 heads 1, 0 | 42.97 % | -55.47 |
| Real targets | 0 | Held-out character classes | Remove layer 2 heads 1, 0, 7 | 32.03 % | -66.41 |
| Real targets | 0 | Held-out character classes | Remove layer 2 heads 1, 7 | 77.34 % | -21.09 |
| Real targets | 10 | Held-out character classes | Unmodified | 99.22 % | 0.00 |
| Real targets | 10 | Held-out character classes | Remove layer 2 heads 1 | 92.97 % | -6.25 |
| Real targets | 10 | Held-out character classes | Remove layer 2 heads 0 | 98.44 % | -0.78 |
| Real targets | 10 | Held-out character classes | Remove layer 2 heads 1, 0 | 90.62 % | -8.59 |
| Real targets | 10 | Held-out character classes | Remove layer 2 heads 1, 0, 7 | 63.28 % | -35.94 |
| Real targets | 10 | Held-out character classes | Remove layer 2 heads 1, 7 | 67.97 % | -31.25 |
| Recursive | 0 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1 | 71.88 % | -28.12 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 0 | 96.88 % | -3.12 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1, 0 | 32.03 % | -67.97 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 20.31 % | -79.69 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1, 7 | 52.34 % | -47.66 |
| Recursive | 10 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1 | 97.66 % | -2.34 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 0 | 100.00 % | 0.00 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1, 0 | 92.97 % | -7.03 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 82.81 % | -17.19 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1, 7 | 91.41 % | -8.59 |
| Recursive | 0 | Held-out handwriting | Unmodified | 86.72 % | 0.00 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1 | 71.88 % | -14.84 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 0 | 85.94 % | -0.78 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1, 0 | 44.53 % | -42.19 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 28.91 % | -57.81 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1, 7 | 67.19 % | -19.53 |
| Recursive | 10 | Held-out handwriting | Unmodified | 87.50 % | 0.00 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1 | 88.28 % | 0.78 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 0 | 88.28 % | 0.78 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1, 0 | 82.81 % | -4.69 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 73.44 % | -14.06 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1, 7 | 83.59 % | -3.91 |
| Recursive | 0 | Held-out character classes | Unmodified | 98.44 % | 0.00 |
| Recursive | 0 | Held-out character classes | Remove layer 2 heads 1 | 78.91 % | -19.53 |
| Recursive | 0 | Held-out character classes | Remove layer 2 heads 0 | 97.66 % | -0.78 |
| Recursive | 0 | Held-out character classes | Remove layer 2 heads 1, 0 | 42.97 % | -55.47 |
| Recursive | 0 | Held-out character classes | Remove layer 2 heads 1, 0, 7 | 32.03 % | -66.41 |
| Recursive | 0 | Held-out character classes | Remove layer 2 heads 1, 7 | 77.34 % | -21.09 |
| Recursive | 10 | Held-out character classes | Unmodified | 98.44 % | 0.00 |
| Recursive | 10 | Held-out character classes | Remove layer 2 heads 1 | 96.09 % | -2.34 |
| Recursive | 10 | Held-out character classes | Remove layer 2 heads 0 | 98.44 % | 0.00 |
| Recursive | 10 | Held-out character classes | Remove layer 2 heads 1, 0 | 93.75 % | -4.69 |
| Recursive | 10 | Held-out character classes | Remove layer 2 heads 1, 0, 7 | 78.12 % | -20.31 |
| Recursive | 10 | Held-out character classes | Remove layer 2 heads 1, 7 | 89.84 % | -8.59 |

## Teacher-forced second-label accuracy

| Condition | Generation | Validation | Intervention | Value | Change (pp) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1 | 67.97 % | -32.03 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 0 | 95.31 % | -4.69 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1, 0 | 28.12 % | -71.88 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 17.97 % | -82.03 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1, 7 | 52.34 % | -47.66 |
| Real targets | 10 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1 | 95.31 % | -4.69 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 0 | 100.00 % | 0.00 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1, 0 | 93.75 % | -6.25 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 46.88 % | -53.12 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1, 7 | 49.22 % | -50.78 |
| Real targets | 0 | Held-out handwriting | Unmodified | 100.00 % | 0.00 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1 | 78.12 % | -21.88 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 0 | 97.66 % | -2.34 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1, 0 | 39.06 % | -60.94 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 25.78 % | -74.22 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1, 7 | 65.62 % | -34.38 |
| Real targets | 10 | Held-out handwriting | Unmodified | 100.00 % | 0.00 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1 | 89.06 % | -10.94 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 0 | 100.00 % | 0.00 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1, 0 | 92.19 % | -7.81 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 59.38 % | -40.62 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1, 7 | 65.62 % | -34.38 |
| Real targets | 0 | Held-out character classes | Unmodified | 96.09 % | 0.00 |
| Real targets | 0 | Held-out character classes | Remove layer 2 heads 1 | 81.25 % | -14.84 |
| Real targets | 0 | Held-out character classes | Remove layer 2 heads 0 | 96.09 % | 0.00 |
| Real targets | 0 | Held-out character classes | Remove layer 2 heads 1, 0 | 41.41 % | -54.69 |
| Real targets | 0 | Held-out character classes | Remove layer 2 heads 1, 0, 7 | 31.25 % | -64.84 |
| Real targets | 0 | Held-out character classes | Remove layer 2 heads 1, 7 | 78.91 % | -17.19 |
| Real targets | 10 | Held-out character classes | Unmodified | 97.66 % | 0.00 |
| Real targets | 10 | Held-out character classes | Remove layer 2 heads 1 | 89.84 % | -7.81 |
| Real targets | 10 | Held-out character classes | Remove layer 2 heads 0 | 97.66 % | 0.00 |
| Real targets | 10 | Held-out character classes | Remove layer 2 heads 1, 0 | 88.28 % | -9.38 |
| Real targets | 10 | Held-out character classes | Remove layer 2 heads 1, 0, 7 | 60.16 % | -37.50 |
| Real targets | 10 | Held-out character classes | Remove layer 2 heads 1, 7 | 65.62 % | -32.03 |
| Recursive | 0 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1 | 67.97 % | -32.03 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 0 | 95.31 % | -4.69 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1, 0 | 28.12 % | -71.88 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 17.97 % | -82.03 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1, 7 | 52.34 % | -47.66 |
| Recursive | 10 | Held-out label assignments | Unmodified | 100.00 % | 0.00 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1 | 94.53 % | -5.47 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 0 | 100.00 % | 0.00 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1, 0 | 87.50 % | -12.50 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 68.75 % | -31.25 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1, 7 | 84.38 % | -15.62 |
| Recursive | 0 | Held-out handwriting | Unmodified | 100.00 % | 0.00 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1 | 78.12 % | -21.88 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 0 | 97.66 % | -2.34 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1, 0 | 39.06 % | -60.94 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 25.78 % | -74.22 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1, 7 | 65.62 % | -34.38 |
| Recursive | 10 | Held-out handwriting | Unmodified | 99.22 % | 0.00 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1 | 94.53 % | -4.69 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 0 | 99.22 % | 0.00 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1, 0 | 92.19 % | -7.03 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 82.81 % | -16.41 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1, 7 | 92.19 % | -7.03 |
| Recursive | 0 | Held-out character classes | Unmodified | 96.09 % | 0.00 |
| Recursive | 0 | Held-out character classes | Remove layer 2 heads 1 | 81.25 % | -14.84 |
| Recursive | 0 | Held-out character classes | Remove layer 2 heads 0 | 96.09 % | 0.00 |
| Recursive | 0 | Held-out character classes | Remove layer 2 heads 1, 0 | 41.41 % | -54.69 |
| Recursive | 0 | Held-out character classes | Remove layer 2 heads 1, 0, 7 | 31.25 % | -64.84 |
| Recursive | 0 | Held-out character classes | Remove layer 2 heads 1, 7 | 78.91 % | -17.19 |
| Recursive | 10 | Held-out character classes | Unmodified | 97.66 % | 0.00 |
| Recursive | 10 | Held-out character classes | Remove layer 2 heads 1 | 93.75 % | -3.91 |
| Recursive | 10 | Held-out character classes | Remove layer 2 heads 0 | 98.44 % | 0.78 |
| Recursive | 10 | Held-out character classes | Remove layer 2 heads 1, 0 | 86.72 % | -10.94 |
| Recursive | 10 | Held-out character classes | Remove layer 2 heads 1, 0, 7 | 79.69 % | -17.97 |
| Recursive | 10 | Held-out character classes | Remove layer 2 heads 1, 7 | 94.53 % | -3.12 |

## Context-valid symbol probability mass

| Condition | Generation | Validation | Intervention | Value | Change (pp) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Real targets | 0 | Held-out label assignments | Unmodified | 97.10 % | 0.00 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1 | 94.33 % | -2.77 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 0 | 96.19 % | -0.90 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1, 0 | 91.12 % | -5.97 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 84.96 % | -12.14 |
| Real targets | 0 | Held-out label assignments | Remove layer 2 heads 1, 7 | 91.25 % | -5.84 |
| Real targets | 10 | Held-out label assignments | Unmodified | 96.75 % | 0.00 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1 | 95.27 % | -1.48 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 0 | 95.35 % | -1.40 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1, 0 | 91.60 % | -5.15 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 86.77 % | -9.97 |
| Real targets | 10 | Held-out label assignments | Remove layer 2 heads 1, 7 | 92.28 % | -4.47 |
| Real targets | 0 | Held-out handwriting | Unmodified | 38.65 % | 0.00 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1 | 32.05 % | -6.60 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 0 | 37.21 % | -1.44 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1, 0 | 30.59 % | -8.06 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 26.58 % | -12.07 |
| Real targets | 0 | Held-out handwriting | Remove layer 2 heads 1, 7 | 29.25 % | -9.40 |
| Real targets | 10 | Held-out handwriting | Unmodified | 38.64 % | 0.00 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1 | 33.28 % | -5.36 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 0 | 37.39 % | -1.25 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1, 0 | 31.47 % | -7.17 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 27.81 % | -10.83 |
| Real targets | 10 | Held-out handwriting | Remove layer 2 heads 1, 7 | 30.68 % | -7.96 |
| Recursive | 0 | Held-out label assignments | Unmodified | 97.10 % | 0.00 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1 | 94.33 % | -2.77 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 0 | 96.19 % | -0.90 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1, 0 | 91.12 % | -5.97 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 84.96 % | -12.14 |
| Recursive | 0 | Held-out label assignments | Remove layer 2 heads 1, 7 | 91.25 % | -5.84 |
| Recursive | 10 | Held-out label assignments | Unmodified | 78.83 % | 0.00 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1 | 73.13 % | -5.70 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 0 | 75.37 % | -3.46 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1, 0 | 68.55 % | -10.28 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1, 0, 7 | 62.47 % | -16.36 |
| Recursive | 10 | Held-out label assignments | Remove layer 2 heads 1, 7 | 68.22 % | -10.61 |
| Recursive | 0 | Held-out handwriting | Unmodified | 38.65 % | 0.00 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1 | 32.05 % | -6.60 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 0 | 37.21 % | -1.44 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1, 0 | 30.59 % | -8.06 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 26.58 % | -12.07 |
| Recursive | 0 | Held-out handwriting | Remove layer 2 heads 1, 7 | 29.25 % | -9.40 |
| Recursive | 10 | Held-out handwriting | Unmodified | 23.92 % | 0.00 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1 | 22.71 % | -1.20 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 0 | 22.32 % | -1.60 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1, 0 | 20.37 % | -3.54 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1, 0, 7 | 20.14 % | -3.78 |
| Recursive | 10 | Held-out handwriting | Remove layer 2 heads 1, 7 | 22.86 % | -1.05 |

## Source files

- [outputs/probes/joint-heads/clean_only.json](../../probes/joint-heads/clean_only.json)
- [outputs/probes/joint-heads/recursive_replacement.json](../../probes/joint-heads/recursive_replacement.json)
