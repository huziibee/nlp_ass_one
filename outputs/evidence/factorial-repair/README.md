# Table 2: which target needs repair?

Prerequisite: Run python scripts/run_paper.py factorial-repair after the primary recursive G9 and G10 checkpoints and joint repair exist.

Status: all expected inputs are present.

This is the one-seed factorial repair comparison. Symbol-only and associated-number-only repairs are distinct from jointly repairing the invalid symbol and its label. The joint-validity metric checks q2 and y2 and does not score y1.

## Table 2

| G10 target correction | Valid symbol (%) | Valid symbol and correct second number (%) |
|:--|--:|--:|
| No repair | 79.49 | 76.56 |
| Symbol-only repair | 97.46 | 79.30 |
| Number-only repair | 79.10 | 70.90 |
| Joint repair | 97.46 | 94.34 |

## Source files

- [outputs/factorial/summary.json](../../factorial/summary.json)
