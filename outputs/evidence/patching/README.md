# Attention activation-patching evidence

Prerequisite: Run python scripts/run_paper.py patching after the clean, recursive and repaired G10 checkpoints exist.

Status: all expected inputs are present.

Patches transfer activations between edited contexts within each trained model. Logit response and probability mass have different units. The sign of the transferred response is reversed from the raw recipient-margin shift so a positive value indicates transfer in the intended context direction. Missing seeds are listed explicitly; no five-seed average is inferred from a partial set.

## Context-switch logit response

| Condition | Seed | Generation | Validation | Intervention | Value |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1 | 15.89 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2 | 15.89 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1 | 10.47 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2 | 10.47 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1 | 15.88 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2 | 15.88 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1 | 15.76 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2 | 15.76 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1 | 9.02 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2 | 9.02 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1 | 14.56 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2 | 14.56 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1 | 15.63 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2 | 15.63 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1 | 10.03 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2 | 10.03 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1 | 15.30 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2 | 15.30 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1 | 15.71 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2 | 15.71 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1 | 9.12 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2 | 9.12 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1 | 15.26 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2 | 15.26 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1 | 15.61 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2 | 15.61 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1 | 8.66 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2 | 8.66 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1 | 15.06 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2 | 15.06 logits |

## Transferred logit response

| Condition | Seed | Generation | Validation | Intervention | Value |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1 | 0.21 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2 | 15.62 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1 | 0.09 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2 | 10.37 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1 | 0.18 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2 | 15.65 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1 | 6.72 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2 | 10.04 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1 | 2.31 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2 | 6.91 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1 | 3.05 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2 | 11.59 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1 | 3.40 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2 | 12.44 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1 | 1.53 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2 | 8.66 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1 | 2.44 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2 | 13.03 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1 | 1.08 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2 | 14.50 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1 | 0.65 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2 | 8.46 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1 | 1.19 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2 | 14.04 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1 | 10.31 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2 | 7.58 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1 | 5.06 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2 | 4.40 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1 | 11.19 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2 | 6.98 logits |

## Fraction of context response transferred

| Condition | Seed | Generation | Validation | Intervention | Value |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1 | 1.30 % |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2 | 98.29 % |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1 | 0.87 % |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2 | 99.03 % |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1 | 1.12 % |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2 | 98.55 % |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1 | 42.62 % |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2 | 63.71 % |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1 | 25.55 % |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2 | 76.55 % |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1 | 20.93 % |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2 | 79.59 % |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1 | 21.79 % |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2 | 79.63 % |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1 | 15.27 % |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2 | 86.31 % |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1 | 15.92 % |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2 | 85.13 % |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1 | 6.88 % |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2 | 92.33 % |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1 | 7.08 % |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2 | 92.73 % |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1 | 7.78 % |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2 | 92.03 % |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1 | 66.05 % |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2 | 48.54 % |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1 | 58.43 % |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2 | 50.87 % |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1 | 74.27 % |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2 | 46.33 % |

## Recipient context-valid symbol mass

| Condition | Seed | Generation | Validation | Intervention | Value |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1 | 96.32 % |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2 | 96.32 % |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1 | 76.53 % |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2 | 76.53 % |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1 | 97.00 % |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2 | 97.00 % |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1 | 96.59 % |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2 | 96.59 % |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1 | 73.18 % |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2 | 73.18 % |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1 | 96.45 % |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2 | 96.45 % |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1 | 96.08 % |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2 | 96.08 % |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1 | 77.11 % |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2 | 77.11 % |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1 | 96.86 % |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2 | 96.86 % |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1 | 96.50 % |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2 | 96.50 % |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1 | 77.59 % |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2 | 77.59 % |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1 | 97.27 % |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2 | 97.27 % |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1 | 96.02 % |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2 | 96.02 % |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1 | 75.69 % |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2 | 75.69 % |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1 | 97.19 % |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2 | 97.19 % |

## Patched head logit-margin shift

| Condition | Seed | Generation | Validation | Intervention | Value |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1, head 0 | -0.04 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1, head 1 | -0.02 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1, head 2 | -0.02 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1, head 3 | -0.00 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1, head 4 | -0.00 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1, head 5 | -0.00 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1, head 6 | -0.11 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 1, head 7 | -0.00 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2, head 0 | -1.89 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2, head 1 | -2.23 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2, head 2 | -1.65 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2, head 3 | -1.66 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2, head 4 | -1.88 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2, head 5 | -1.94 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2, head 6 | -1.72 logits |
| Real targets | 6001 | 10 | Held-out label assignments | Layer 2, head 7 | -2.16 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1, head 0 | -0.01 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1, head 1 | -0.01 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1, head 2 | -0.02 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1, head 3 | -0.00 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1, head 4 | -0.01 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1, head 5 | -0.00 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1, head 6 | -0.03 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 1, head 7 | -0.00 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2, head 0 | -1.22 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2, head 1 | -1.51 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2, head 2 | -0.96 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2, head 3 | -1.33 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2, head 4 | -1.44 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2, head 5 | -1.12 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2, head 6 | -1.10 logits |
| Recursive | 6001 | 10 | Held-out label assignments | Layer 2, head 7 | -1.39 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1, head 0 | -0.03 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1, head 1 | -0.02 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1, head 2 | -0.02 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1, head 3 | -0.01 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1, head 4 | -0.01 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1, head 5 | -0.00 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1, head 6 | -0.08 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 1, head 7 | -0.00 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2, head 0 | -2.01 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2, head 1 | -2.18 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2, head 2 | -1.45 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2, head 3 | -1.83 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2, head 4 | -1.97 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2, head 5 | -1.76 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2, head 6 | -1.88 logits |
| Joint target repair | 6001 | 10 | Held-out label assignments | Layer 2, head 7 | -2.13 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1, head 0 | -0.78 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1, head 1 | -0.96 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1, head 2 | -1.53 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1, head 3 | -0.97 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1, head 4 | -1.18 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1, head 5 | -0.36 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1, head 6 | -0.01 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 1, head 7 | -0.53 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2, head 0 | -1.38 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2, head 1 | -1.19 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2, head 2 | -0.88 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2, head 3 | -0.99 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2, head 4 | -1.05 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2, head 5 | -0.58 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2, head 6 | -1.59 logits |
| Real targets | 5054 | 10 | Held-out label assignments | Layer 2, head 7 | -1.70 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1, head 0 | -0.50 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1, head 1 | -0.51 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1, head 2 | -0.34 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1, head 3 | -0.44 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1, head 4 | -0.11 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1, head 5 | -0.20 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1, head 6 | -0.01 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 1, head 7 | -0.10 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2, head 0 | -0.97 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2, head 1 | -0.87 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2, head 2 | -0.76 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2, head 3 | -0.63 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2, head 4 | -0.68 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2, head 5 | -0.41 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2, head 6 | -1.20 logits |
| Recursive | 5054 | 10 | Held-out label assignments | Layer 2, head 7 | -1.08 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1, head 0 | -0.50 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1, head 1 | -0.58 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1, head 2 | -0.39 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1, head 3 | -0.80 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1, head 4 | -0.17 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1, head 5 | -0.25 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1, head 6 | -0.06 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 1, head 7 | -0.16 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2, head 0 | -1.52 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2, head 1 | -1.42 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2, head 2 | -1.36 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2, head 3 | -0.94 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2, head 4 | -1.08 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2, head 5 | -0.90 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2, head 6 | -1.93 logits |
| Joint target repair | 5054 | 10 | Held-out label assignments | Layer 2, head 7 | -1.80 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1, head 0 | -0.00 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1, head 1 | -0.44 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1, head 2 | -0.00 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1, head 3 | -0.03 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1, head 4 | -0.85 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1, head 5 | -1.03 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1, head 6 | -0.37 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 1, head 7 | -0.51 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2, head 0 | -1.65 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2, head 1 | -1.52 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2, head 2 | -1.31 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2, head 3 | -1.12 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2, head 4 | -0.96 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2, head 5 | -1.69 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2, head 6 | -1.99 logits |
| Real targets | 6054 | 10 | Held-out label assignments | Layer 2, head 7 | -1.45 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1, head 0 | -0.02 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1, head 1 | -0.16 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1, head 2 | -0.00 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1, head 3 | -0.01 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1, head 4 | -0.37 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1, head 5 | -0.48 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1, head 6 | -0.24 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 1, head 7 | -0.21 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2, head 0 | -1.08 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2, head 1 | -1.20 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2, head 2 | -1.29 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2, head 3 | -0.58 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2, head 4 | -0.50 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2, head 5 | -1.19 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2, head 6 | -1.31 logits |
| Recursive | 6054 | 10 | Held-out label assignments | Layer 2, head 7 | -1.21 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1, head 0 | -0.01 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1, head 1 | -0.40 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1, head 2 | -0.00 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1, head 3 | -0.04 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1, head 4 | -0.62 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1, head 5 | -0.77 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1, head 6 | -0.28 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 1, head 7 | -0.27 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2, head 0 | -1.48 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2, head 1 | -1.84 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2, head 2 | -1.54 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2, head 3 | -1.12 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2, head 4 | -0.97 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2, head 5 | -1.86 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2, head 6 | -2.05 logits |
| Joint target repair | 6054 | 10 | Held-out label assignments | Layer 2, head 7 | -1.67 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1, head 0 | -0.22 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1, head 1 | -0.01 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1, head 2 | -0.15 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1, head 3 | -0.37 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1, head 4 | -0.00 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1, head 5 | -0.00 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1, head 6 | -0.28 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 1, head 7 | -0.04 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2, head 0 | -2.03 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2, head 1 | -2.18 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2, head 2 | -1.62 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2, head 3 | -1.68 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2, head 4 | -1.19 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2, head 5 | -2.05 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2, head 6 | -1.58 logits |
| Real targets | 7054 | 10 | Held-out label assignments | Layer 2, head 7 | -1.53 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1, head 0 | -0.10 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1, head 1 | -0.00 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1, head 2 | -0.03 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1, head 3 | -0.12 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1, head 4 | -0.00 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1, head 5 | -0.00 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1, head 6 | -0.20 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 1, head 7 | -0.17 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2, head 0 | -1.07 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2, head 1 | -1.21 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2, head 2 | -1.05 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2, head 3 | -1.01 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2, head 4 | -0.84 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2, head 5 | -1.30 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2, head 6 | -0.94 logits |
| Recursive | 7054 | 10 | Held-out label assignments | Layer 2, head 7 | -0.67 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1, head 0 | -0.18 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1, head 1 | -0.00 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1, head 2 | -0.06 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1, head 3 | -0.26 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1, head 4 | -0.00 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1, head 5 | -0.00 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1, head 6 | -0.38 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 1, head 7 | -0.26 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2, head 0 | -1.75 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2, head 1 | -1.84 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2, head 2 | -1.84 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2, head 3 | -1.45 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2, head 4 | -1.47 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2, head 5 | -2.21 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2, head 6 | -1.52 logits |
| Joint target repair | 7054 | 10 | Held-out label assignments | Layer 2, head 7 | -1.40 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1, head 0 | -2.19 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1, head 1 | -1.39 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1, head 2 | -1.33 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1, head 3 | -1.68 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1, head 4 | -0.33 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1, head 5 | -0.80 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1, head 6 | -1.66 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 1, head 7 | -0.40 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2, head 0 | -0.95 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2, head 1 | -0.95 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2, head 2 | -0.85 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2, head 3 | -0.84 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2, head 4 | -1.04 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2, head 5 | -0.91 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2, head 6 | -0.57 logits |
| Real targets | 8054 | 10 | Held-out label assignments | Layer 2, head 7 | -1.10 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1, head 0 | -1.63 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1, head 1 | -0.78 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1, head 2 | -0.00 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1, head 3 | -1.04 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1, head 4 | -0.36 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1, head 5 | -0.12 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1, head 6 | -0.13 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 1, head 7 | -0.70 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2, head 0 | -0.66 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2, head 1 | -0.53 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2, head 2 | -0.58 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2, head 3 | -0.49 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2, head 4 | -0.43 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2, head 5 | -0.53 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2, head 6 | -0.40 logits |
| Recursive | 8054 | 10 | Held-out label assignments | Layer 2, head 7 | -0.51 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1, head 0 | -2.27 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1, head 1 | -1.47 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1, head 2 | -0.98 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1, head 3 | -1.48 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1, head 4 | -0.68 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1, head 5 | -0.72 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1, head 6 | -1.44 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 1, head 7 | -1.47 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2, head 0 | -1.11 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2, head 1 | -0.84 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2, head 2 | -0.81 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2, head 3 | -0.79 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2, head 4 | -0.85 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2, head 5 | -0.98 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2, head 6 | -0.52 logits |
| Joint target repair | 8054 | 10 | Held-out label assignments | Layer 2, head 7 | -0.91 logits |

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

- [outputs/q2_study/mediation/summary.json](../../q2_study/mediation/summary.json)
- [outputs/q2_study/raw_patches/5054-clean_only.json](../../q2_study/raw_patches/5054-clean_only.json)
- [outputs/q2_study/raw_patches/5054-recursive_replacement.json](../../q2_study/raw_patches/5054-recursive_replacement.json)
- [outputs/q2_study/raw_patches/6001-clean_only.json](../../q2_study/raw_patches/6001-clean_only.json)
- [outputs/q2_study/raw_patches/6001-recursive_replacement.json](../../q2_study/raw_patches/6001-recursive_replacement.json)
- [outputs/q2_study/raw_patches/6054-clean_only.json](../../q2_study/raw_patches/6054-clean_only.json)
- [outputs/q2_study/raw_patches/6054-recursive_replacement.json](../../q2_study/raw_patches/6054-recursive_replacement.json)
- [outputs/q2_study/raw_patches/7054-clean_only.json](../../q2_study/raw_patches/7054-clean_only.json)
- [outputs/q2_study/raw_patches/7054-recursive_replacement.json](../../q2_study/raw_patches/7054-recursive_replacement.json)
- [outputs/q2_study/raw_patches/8054-clean_only.json](../../q2_study/raw_patches/8054-clean_only.json)
- [outputs/q2_study/raw_patches/8054-recursive_replacement.json](../../q2_study/raw_patches/8054-recursive_replacement.json)
- [outputs/q2_study/repaired_patches/5054-fix_invalid_q2.json](../../q2_study/repaired_patches/5054-fix_invalid_q2.json)
- [outputs/q2_study/repaired_patches/6001-fix_invalid_q2.json](../../q2_study/repaired_patches/6001-fix_invalid_q2.json)
- [outputs/q2_study/repaired_patches/6054-fix_invalid_q2.json](../../q2_study/repaired_patches/6054-fix_invalid_q2.json)
- [outputs/q2_study/repaired_patches/7054-fix_invalid_q2.json](../../q2_study/repaired_patches/7054-fix_invalid_q2.json)
- [outputs/q2_study/repaired_patches/8054-fix_invalid_q2.json](../../q2_study/repaired_patches/8054-fix_invalid_q2.json)
