# Methodology control audit

This table is calculated from existing original experimental JSON; it is not new training.

| Control | Seeds | G10 clean | G10 synthetic | Clean minus synthetic |
|:--|--:|--:|--:|--:|
| Base 300k G0 + descendants, validation label accuracy | 1 | 76.95% | 49.41% | +27.54 pp |
| Accumulated training examples, next-symbol context validity | 1 | 87.11% | 83.20% | +3.91 pp |

The 300k Base budget matches Extended's exposure count, **not** its output task or architecture.
The corpus-control experiment retains past examples and appends new generated targets.
Neither single-seed control can replace the original five-seed uncertainty analysis.

## Sources

- `docs/experiments/base_full_budget_300k/summary.json`
- `docs/experiments/cumulative_corpus/summary.json`
- `docs/experiments/g10_five_seed/g10_seed_summary.json`

## Limits

- The brief mandates Base and Extended continuation outputs, not full prompt generation.
- Budget-matched Base 300k evidence is a single-seed sensitivity control, not a matched-task effect.
- Accumulating corpus uses real input prefixes plus retained generated targets, not generated prompts.
- The primary five-seed study used unequal Base/Extended budgets; its cross-task causal comparison remains limited.
- The original final test was opened at G3. All these G10 records use validation only.
