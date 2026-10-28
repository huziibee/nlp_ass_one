# Conditional probability audit — original Omniglot Extended model

**Executed successfully** in [GitHub Actions run 38720720707](https://github.com/huziibee/nlp_ass_one/actions/runs/38720720707). Both clean-only and recursive G0/G3/G6/G10 jobs succeeded, using 128 matched development episodes per role and all original checkpoints, with original protected final test excluded. Files `docs/experiments/conditional_symbol_distribution/detailed-conditional-softmax-{arm}/{arm}/` contain aggregate JSON and all per-episode 50-way probability vectors.

## What is new beyond prior probability histograms?

The previous experiment measured **global marginal sampled frequencies**, entropy, and fraction of sampled q2 tokens occurring in the context. That does not determine whether the model's internal predictive distribution is decisively wrong or merely leaks softmax mass into irrelevant classes.

This experiment computes the five possible first-numeric-label output probabilities and the conditional q2 distribution for each first-label choice, forming an **exact five-way predictive mixture** (conditional on fixed forward-pass stochastic seeds). The same episodes and conditions are used across models. Thus uncertainty from a sampled first label is analytically marginalized, not subject to Monte Carlo noise. We also compute first-label probability/confidence, free-symbol top-1 context validity, between-episode support-mass variation, average per-episode entropy, global marginal entropy, an episode-symbol mutual information score, and separate teacher-forced first-label distributions.

## First main result — contextual leakage without greedy failure

| Validation held-out **label pairs**, n=128 | Clean G10 | Recursive G10 |
|---|---:|---:|
| Softmax probability mass on two context symbols | 96.93% | **78.70%** |
| Softmax probability mass off context | 3.07% | **21.30%** |
| Fraction of episodes whose argmax next symbol is invalid | 0/128 | **0/128** |
| Mean per-episode 50-way q2 entropy (nats) | 0.834 | **1.380** |
| Entropy of mean 50-way softmax (nats) | 3.773 | 3.730 |
| Episode–symbol mutual information H(marginal) − E H(conditional), nats | 2.939 | **2.350** |
| Mean first-label probability assigned to correct number | 99.76% | 97.94% |
| Extra context-valid q2 mass after teacher forcing first number | <0.001 pp | +0.006 pp |

**Interpretation.** Recursive feedback broadens and damages each episode's conditional next-symbol distribution, decreasing information about which symbol belongs to the particular current context. Yet in this selected 128-episode validation set, the correct symbol still wins the **argmax**. As a result, **stochastic sampling** draws invalid symbols much more frequently even without a greedy top-1 error. The generated marginal still covers a broad vocabulary, so this is not a simplistic all-output mode collapse. Switching to greedy would suppress this source of error on this probe but would change the generative policy rather than repair the learned distribution; do not claim it is a general cure.

The exact G0 clean and recursive distributions are equal: about 97.22% valid mass, 0.799 nats per-episode entropy. By G3 recursive valid mass is 90.78%, G6 84.02%, G10 78.70%; corresponding clean values 96.96%, 96.62%, 96.93%. This is a **primary-seed descriptive trajectory**.

**First-label teacher forcing does not recover the q2 distribution.** The G10 recursive valid mass under oracle first-number conditioning rises by only ~0.006 pp over exact first-number-label marginalization. This indicates the main q2 leakage is not a trivial cascade from errors in predicted first numeric labels. This does not rule out other interactions or head-level causal pathways.

## Held-out exemplars reveal a different baseline problem

On **val_exemplar** (new exemplar features from training character IDs), both variants already allocate low context-valid q2 mass, even at G0: 38.17%. At G10 clean this becomes 35.41% and recursive 28.36%, with roughly 51.6% of the latter's greedy q2 argmax predictions invalid. This demonstrates an important **generalization limitation of the symbol-head feature representation**. We should not quietly transfer val_relabel conclusions to held-out exemplars. Some of the absolute exemplar failure exists in the **clean control** and predates recursive training.

## Mechanism implications

- **A (matching-circuit destroyed):** First-label matching stays accurate, while q2's distribution becomes diffuse and off-context. This favors a specialized output-distribution failure over a universal collapse of numerical retrieval.
- **B (redistribution):** Probability leakage and head-ablation interactions can coexist. Distribution metrics alone do NOT localize a new retrieval head. Contrast q2 softmax before/after specific head-output knockouts to link confidence changes to a path.
- **C (shortcut):** Previous legal counterfactual relabeling tests refute simple fixed-label shortcuts. The new q2 entropy result does not determine whether a different context-sensitive retrieval algorithm is used.

## Scientific presentation priority

**HIGH:** two numbers and a contrast can fit in the main two-page paper: `G10 context-valid q2 mass clean 96.93% vs recursive 78.70%; greedy q2 remains valid on 128/128 while sampled invalidity is ~21%, revealing conditional uncertainty and leakage rather than universal argmax errors.` Give the 128-episode, single-seed caveat. Prefer this mechanism-aware difference over a large global symbol histogram. Supporting cross-generation curves, entropy, mutual information, and held-out-exemplar limitations belong in the full paper/appendix.

## Necessary cautions

1. The per-episode softmax and mixture are computed with **fixed JAX keys**. Because the architecture calls dropout in forward execution, this is an algorithmic fixed-key evaluation, not a guarantee of literal fully integrated distribution over dropout randomness. Do not state an exact marginal over all stochastic model randomness.
2. Primary model seed only, 128 evaluation episodes per role; statistical confidence across independently trained models not established.
3. `val_classes` 50-way symbolic class ID probability is not defined for the 1,473 reserved unseen classes. This audit intentionally used only label-pair and exemplar role, not protected final classes.
4. Direct comparison to previously saved 1,024 **sampled** outputs is conceptually useful but different sampling/evaluation seeds and estimands; their percentage point values should not be artificially equated.
5. A trivial entropy comparison between global marginal distributions would hide the key per-episode conditional failure; **always distinguish H(q2) from E[H(q2|context)]**.
