# Evidence review: Does attention mediate the recursive q2 context-validity failure?

Status: **Completed five-seed exploratory validation**, 16 October 2026.
Workflow: https://github.com/huziibee/nlp_ass_one/actions/runs/38816067421
Pilot: https://github.com/huziibee/nlp_ass_one/actions/runs/38815824528
Data: `docs/experiments/q2_activation_patching/five_seeds/`
Summary: `docs/experiments/q2_activation_patching/five_seed_summary.json`

## Hypothesis and outcome

The previously published paper's primary behavioral result concerns the *next symbol q2*
rather than the *first label y1*. Prior G10 head ablations focused on y1 accuracy. We
therefore tested whether contextual character information reaches the q2 output through
attention heads and whether the output is less context-sensitive under recursive
synthetic target training.

**Finding:** Across five independent pretrained Extended model pairs, changing a
nonquery contextual character from B to C changes the logit preference
`logit(C)-logit(B)` by 15.72 (control) versus 9.13 (recursive) on average.
The paired control-minus-recursive difference is **6.59 logit units** (descriptive
95% paired-t interval 5.72–7.46, `n=5` model-seed pairs). Five of five paired
comparisons have smaller *raw-logit* context sensitivity in the recursive model.

This is **consistent with** weakened contextual influence on the next-symbol logits,
but NOT proof of the precise cause: training conditions can change logit magnitude,
head/output calibration, and other model features.

## Intervention and safeguards

- **Model and data:** Original Omniglot features, same 2-layer 8-head Extended architecture;
  G10 real-target control and recursively generated-target model for each of 5
  independently trained model seeds. All five corresponding G10 checkpoints
  were separately verified by the original five-seed G10 aggregation.
- **Validation:** 128 `val_relabel` episodes per trained model, training-vocabulary
  symbol IDs and held-out numerical-label assignment pairs; same episode constructor
  and random seed across all models. Never uses 100-class final test.
- **Paired contexts:** Original prompt has support identities A and B and query A.
  Donor input is `A->lA, B->lB, query A`. Target input substitutes unseen-in-episode
  training-character C for B, keeping `lB` and all other symbols/numbers fixed;
  support is then `A->lA, C->lB, query A`. Labels remain valid and the query
  numerical answer is unchanged. The new C uses the original train prototype
  representation (exemplar 0), rather than an independent held-out exemplar.
- **Correct initial y1 supplied:** The next-symbol q2 distribution is measured
  conditioned on correct y1 to isolate q2; this is NOT free-running sampling.
- **Intervention:** With model weights fixed, copy the source attention-head
  `values_pre_proj` activation at the q2-prediction token position 5 into
  the target run. Patch each individual head or all eight in each of layers 1/2.
  Each patch is within the **same model**; weights/neurons are never aligned
  across distinct trained models.
- **Controls:** Identity (target-to-target) patching and noncausal future-token
  patching both reproduce the natural target output exactly (maximum numerical
  error 0 in saved checkpoints). This tests implementation and causal isolation.

## Quantitative summary (mean across 5 independent model pairs)

| Outcome (not classification accuracy) | Control | Recursive |
|---|---:|---:|
| Absolute change in `logit(C)-logit(B)` from donor to new context | 15.72 | 9.13 |
| Donor->recipient layer-1 all-head patch shift in logit margin | -4.34 | -1.68 |
| Donor->recipient layer-2 all-head patch shift in logit margin | -12.04 | -7.63 |
| Natural next-symbol probability on recipient's 2 valid identities | 96.32% | 75.80% |
| After full layer-2 donor patch: recipient valid-support probability | 74.51% | 54.56% |

Raw logits, probabilities, and these interventions use a *specific controlled
A,B -> A,C context perturbation*, not the full G10 free-sampling validation
metric; their values must not be substituted for Table 1.

All eight second-layer heads contribute appreciably to the contextual preference
shift; **no single unique head** was identified. The proportion of information
transferred in layer 1 versus layer 2 varies considerably by model seed.
Single-head effects are not guaranteed additive.

## Conclusion supported by evidence

1. Within each model, changing a support character causally changes the q2 logits.
2. Within-model attention activation replacement causally transfers a substantial
   part of this contextual-symbol preference signal, particularly through layer 2.
3. The raw-logit sensitivity to the changed character is smaller under recursive
   training across all five independent model pairs.
4. This does NOT identify an exact circuit or demonstrate that the specific
   training-target corruptions *caused* the attention-layer changes. Such a claim
   requires matched repaired/unrepaired training-source interventions, calibrated
   metrics, and further path-level mediation evidence.

## Suggested paper use

**Worth adding** as the mechanistic evidence directly targeting the paper's
main failure mode. Prefer 1–2 sentences in the mechanistic-discussion section
over another graph or standalone paragraph in the tightly constrained
two-page extended abstract. Consider replacing a less-relevant y1-only head
ablation detail.

Possible calibrated wording:

> Across five Extended model pairs, changing a contextual support character
> shifted the relative next-symbol logits less after recursive training
> (mean 9.13 versus 15.72 in real-target controls). Within-model activation
> patching of second-layer attention heads transferred much of this contextual
> preference signal, indicating continued attention-mediated computation rather
> than a vanished q2 circuit. Differences in logit scaling and neural alignment
> prevent causal attribution of the change to a particular head.

If included, explain that these analyses are **validation-only,
teacher-forced y1, 128 episodes per model**, with the additional caveat
that logit scales may differ across models. Existing Conmy et al. (2023)
circuit-methods citation remains relevant; cite Zhang & Nanda (ICLR 2024)
if discussing methodological best practices in detail.
