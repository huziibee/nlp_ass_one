# Frozen Base and Extended test evaluation
| Model | First-label acc. | First-label CE | Second-label acc. (clean Q2) | Second-label CE |
|---|---:|---:|---:|---:|
| `base_g0` | 94.19% | 0.1568 | n/a | n/a |
| `base_clean_g3` | 93.60% | 0.1591 | n/a | n/a |
| `base_recursive_g3` | 92.70% | 0.1734 | n/a | n/a |
| `base_halfreal_g3` | 93.51% | 0.1604 | n/a | n/a |
| `base_frozen_g3` | 93.38% | 0.1632 | n/a | n/a |
| `extended_g0` | 97.39% | 0.0870 | 97.19% | 0.0774 |
| `extended_clean_g3` | 95.92% | 0.1236 | 96.70% | 0.1038 |
| `extended_recursive_g3` | 98.32% | 0.0402 | 98.46% | 0.0396 |
| `extended_halfreal_g3` | 98.34% | 0.0466 | 97.95% | 0.0525 |
| `extended_frozen_g3` | 98.07% | 0.0522 | 97.80% | 0.0563 |

The Extended next-symbol class-ID head cannot output the 100 unseen final-test class IDs. Second-label scores are explicitly *teacher forced* with a valid clean Q2. This is not autonomous three-token sequence accuracy. Results are single training seeds and should not be presented as broad significance estimates.
