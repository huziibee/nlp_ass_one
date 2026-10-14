# Development Base training pilot — 16,000 sequences

**Source:** Original Singh et al. JAX transformer, unchanged pinned `external/icl-dynamics/main.py` and official Omniglot ResNet18 HDF5, through `scripts/paper_base_smoke.py`.  
**Verification:** [Successful GitHub Actions run](https://github.com/huziibee/nlp_ass_one/actions/runs/38611683875); preserved `log.h5`, checkpoints and config as an Actions artifact.  
**Setup:** 500 optimizer updates × batch size 32 = **16,000 training sequences**. Model: two layers, eight heads, d_model64, learning rate 1e-5, seed 5. Exactly one run, no independent seeds. Evaluation: 256 sampled episodes in each of four development conditions at each checkpoint. **No final-test class model score was computed.**

## What happened

| Metric | 0 sequences | 16,000 sequences | Direction |
| --- | ---: | ---: | --- |
| Original training classes and label pairs — accuracy | 20.31% | **50.39%** | Higher |
| Held-out label pairs — accuracy | 21.48% | **35.55%** | Higher, still weak |
| Held-out exemplars — accuracy | 17.97% | **52.73%** | Higher |
| Reserved validation classes — accuracy | 17.19% | **48.44%** | Higher |
| Training-class loss | 2.015 | **1.098** | Lower |
| Held-out-label-pair loss | 2.082 | **1.498** | Lower |

Other stored measurement positions are 3,200; 6,400; 9,600; 12,800 sequences.

The model predicts one of **five numeric labels**. Unconditional uniform random guessing therefore yields 20% accuracy. But given the two distinct context labels, an algorithm that already knows to choose between them *randomly* would achieve **50%**. Thus, reaching approximately 50% is **not sufficient evidence of reliable contextual matching**. In particular, the held-out-label-pair validation remains below that two-choice baseline.

**Scientific conclusion limited to this pilot:** the source model's JAX training and evaluation functions work, and optimisation improves short-run development metrics. The model has **not established near-correct ICL, a functional induction head, model collapse, or any response to synthetic-data recursion**.

## What to do next

A one-million-sequence Base run with exact published optimiser and source task configuration is queued/running in [the full Base reproduction workflow](https://github.com/huziibee/nlp_ass_one/actions/runs/38612082174). Its periodic evaluator for held-out **test** classes is deliberately replaced by a **validation**-class evaluator. It saves checkpoint snapshots about every 100,000 training sequences to permit later circuit analyses.

`scripts/plot_base_training.py` is a unit-tested reader/plotter for the original authors' saved `log.h5`. Run it locally after downloading either artifact:

```bash
python scripts/plot_base_training.py --log path/to/safe_train_val_smoke/log.h5 --out outputs/base_pilot_plots
```

The pilot outcome must not determine which final-test examples to expose or which heads to select on the final test.
