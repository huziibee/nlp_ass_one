# Does retrieval follow the prompt?

Prerequisite: Run python scripts/run_paper.py counterfactual after the primary G0–G10 experiment.

Status: all expected inputs are present.

Label swaps and support-order edits use the same validation prompts. Replacing a query with the other support exemplar is an easier task than a new handwriting query. These are one-seed retrieval probes.

## Accuracy against edited prompt

| Condition | Generation | Validation | Intervention | Value |
| :-- | :-- | :-- | :-- | :-- |
| Real targets | g0 | Held-out label assignments | swap support labels | 100.00 % |
| Real targets | g0 | Held-out label assignments | swap support order | 100.00 % |
| Real targets | g0 | Held-out label assignments | flip query to other support | 99.22 % |
| Real targets | g10 | Held-out label assignments | swap support labels | 99.22 % |
| Real targets | g10 | Held-out label assignments | swap support order | 100.00 % |
| Real targets | g10 | Held-out label assignments | flip query to other support | 100.00 % |
| Real targets | g0 | Held-out handwriting | swap support labels | 89.84 % |
| Real targets | g0 | Held-out handwriting | swap support order | 88.28 % |
| Real targets | g0 | Held-out handwriting | flip query to other support | 98.44 % |
| Real targets | g10 | Held-out handwriting | swap support labels | 90.62 % |
| Real targets | g10 | Held-out handwriting | swap support order | 88.28 % |
| Real targets | g10 | Held-out handwriting | flip query to other support | 98.44 % |
| Real targets | g0 | Held-out character classes | swap support labels | 99.22 % |
| Real targets | g0 | Held-out character classes | swap support order | 98.44 % |
| Real targets | g0 | Held-out character classes | flip query to other support | 99.22 % |
| Real targets | g10 | Held-out character classes | swap support labels | 99.22 % |
| Real targets | g10 | Held-out character classes | swap support order | 98.44 % |
| Real targets | g10 | Held-out character classes | flip query to other support | 100.00 % |
| Recursive | g0 | Held-out label assignments | swap support labels | 100.00 % |
| Recursive | g0 | Held-out label assignments | swap support order | 100.00 % |
| Recursive | g0 | Held-out label assignments | flip query to other support | 99.22 % |
| Recursive | g10 | Held-out label assignments | swap support labels | 100.00 % |
| Recursive | g10 | Held-out label assignments | swap support order | 100.00 % |
| Recursive | g10 | Held-out label assignments | flip query to other support | 100.00 % |
| Recursive | g0 | Held-out handwriting | swap support labels | 89.84 % |
| Recursive | g0 | Held-out handwriting | swap support order | 88.28 % |
| Recursive | g0 | Held-out handwriting | flip query to other support | 98.44 % |
| Recursive | g10 | Held-out handwriting | swap support labels | 89.84 % |
| Recursive | g10 | Held-out handwriting | swap support order | 92.19 % |
| Recursive | g10 | Held-out handwriting | flip query to other support | 98.44 % |
| Recursive | g0 | Held-out character classes | swap support labels | 99.22 % |
| Recursive | g0 | Held-out character classes | swap support order | 98.44 % |
| Recursive | g0 | Held-out character classes | flip query to other support | 99.22 % |
| Recursive | g10 | Held-out character classes | swap support labels | 98.44 % |
| Recursive | g10 | Held-out character classes | swap support order | 96.88 % |
| Recursive | g10 | Held-out character classes | flip query to other support | 95.31 % |

## Predictions changed

| Condition | Generation | Validation | Intervention | Value |
| :-- | :-- | :-- | :-- | :-- |
| Real targets | g0 | Held-out label assignments | swap support labels | 99.22 % |
| Real targets | g0 | Held-out label assignments | swap support order | 0.78 % |
| Real targets | g0 | Held-out label assignments | flip query to other support | 98.44 % |
| Real targets | g10 | Held-out label assignments | swap support labels | 99.22 % |
| Real targets | g10 | Held-out label assignments | swap support order | 0.00 % |
| Real targets | g10 | Held-out label assignments | flip query to other support | 100.00 % |
| Real targets | g0 | Held-out handwriting | swap support labels | 95.31 % |
| Real targets | g0 | Held-out handwriting | swap support order | 3.12 % |
| Real targets | g0 | Held-out handwriting | flip query to other support | 86.72 % |
| Real targets | g10 | Held-out handwriting | swap support labels | 94.53 % |
| Real targets | g10 | Held-out handwriting | swap support order | 6.25 % |
| Real targets | g10 | Held-out handwriting | flip query to other support | 86.72 % |
| Real targets | g0 | Held-out character classes | swap support labels | 99.22 % |
| Real targets | g0 | Held-out character classes | swap support order | 0.00 % |
| Real targets | g0 | Held-out character classes | flip query to other support | 97.66 % |
| Real targets | g10 | Held-out character classes | swap support labels | 97.66 % |
| Real targets | g10 | Held-out character classes | swap support order | 3.12 % |
| Real targets | g10 | Held-out character classes | flip query to other support | 98.44 % |
| Recursive | g0 | Held-out label assignments | swap support labels | 99.22 % |
| Recursive | g0 | Held-out label assignments | swap support order | 0.78 % |
| Recursive | g0 | Held-out label assignments | flip query to other support | 98.44 % |
| Recursive | g10 | Held-out label assignments | swap support labels | 98.44 % |
| Recursive | g10 | Held-out label assignments | swap support order | 1.56 % |
| Recursive | g10 | Held-out label assignments | flip query to other support | 98.44 % |
| Recursive | g0 | Held-out handwriting | swap support labels | 95.31 % |
| Recursive | g0 | Held-out handwriting | swap support order | 3.12 % |
| Recursive | g0 | Held-out handwriting | flip query to other support | 86.72 % |
| Recursive | g10 | Held-out handwriting | swap support labels | 92.97 % |
| Recursive | g10 | Held-out handwriting | swap support order | 1.56 % |
| Recursive | g10 | Held-out handwriting | flip query to other support | 89.06 % |
| Recursive | g0 | Held-out character classes | swap support labels | 99.22 % |
| Recursive | g0 | Held-out character classes | swap support order | 0.00 % |
| Recursive | g0 | Held-out character classes | flip query to other support | 97.66 % |
| Recursive | g10 | Held-out character classes | swap support labels | 97.66 % |
| Recursive | g10 | Held-out character classes | swap support order | 0.78 % |
| Recursive | g10 | Held-out character classes | flip query to other support | 92.97 % |

## Edited-prompt accuracy among originally correct predictions

| Condition | Generation | Validation | Intervention | Value |
| :-- | :-- | :-- | :-- | :-- |
| Real targets | g0 | Held-out label assignments | swap support labels | 100.00 % |
| Real targets | g0 | Held-out label assignments | swap support order | 100.00 % |
| Real targets | g0 | Held-out label assignments | flip query to other support | 99.21 % |
| Real targets | g10 | Held-out label assignments | swap support labels | 99.22 % |
| Real targets | g10 | Held-out label assignments | swap support order | 100.00 % |
| Real targets | g10 | Held-out label assignments | flip query to other support | 100.00 % |
| Real targets | g0 | Held-out handwriting | swap support labels | 98.23 % |
| Real targets | g0 | Held-out handwriting | swap support order | 98.23 % |
| Real targets | g0 | Held-out handwriting | flip query to other support | 98.23 % |
| Real targets | g10 | Held-out handwriting | swap support labels | 98.23 % |
| Real targets | g10 | Held-out handwriting | swap support order | 96.46 % |
| Real targets | g10 | Held-out handwriting | flip query to other support | 98.23 % |
| Real targets | g0 | Held-out character classes | swap support labels | 100.00 % |
| Real targets | g0 | Held-out character classes | swap support order | 100.00 % |
| Real targets | g0 | Held-out character classes | flip query to other support | 99.21 % |
| Real targets | g10 | Held-out character classes | swap support labels | 99.21 % |
| Real targets | g10 | Held-out character classes | swap support order | 98.41 % |
| Real targets | g10 | Held-out character classes | flip query to other support | 100.00 % |
| Recursive | g0 | Held-out label assignments | swap support labels | 100.00 % |
| Recursive | g0 | Held-out label assignments | swap support order | 100.00 % |
| Recursive | g0 | Held-out label assignments | flip query to other support | 99.21 % |
| Recursive | g10 | Held-out label assignments | swap support labels | 100.00 % |
| Recursive | g10 | Held-out label assignments | swap support order | 100.00 % |
| Recursive | g10 | Held-out label assignments | flip query to other support | 100.00 % |
| Recursive | g0 | Held-out handwriting | swap support labels | 98.23 % |
| Recursive | g0 | Held-out handwriting | swap support order | 98.23 % |
| Recursive | g0 | Held-out handwriting | flip query to other support | 98.23 % |
| Recursive | g10 | Held-out handwriting | swap support labels | 95.69 % |
| Recursive | g10 | Held-out handwriting | swap support order | 100.00 % |
| Recursive | g10 | Held-out handwriting | flip query to other support | 98.28 % |
| Recursive | g0 | Held-out character classes | swap support labels | 100.00 % |
| Recursive | g0 | Held-out character classes | swap support order | 100.00 % |
| Recursive | g0 | Held-out character classes | flip query to other support | 99.21 % |
| Recursive | g10 | Held-out character classes | swap support labels | 99.20 % |
| Recursive | g10 | Held-out character classes | swap support order | 99.20 % |
| Recursive | g10 | Held-out character classes | flip query to other support | 95.20 % |

## Source files

- [outputs/probes/counterfactual/clean_only.json](../../probes/counterfactual/clean_only.json)
- [outputs/probes/counterfactual/recursive_replacement.json](../../probes/counterfactual/recursive_replacement.json)
