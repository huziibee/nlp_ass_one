# Where Does Recursive Feedback Fail?
## Contextual Symbol Generation and Attention-Head Dependence

**Priyanka Gohil (2586288), Arno Strauss (2613224), and Mohammed Bangie (2610990)**  
School of Computer Science and Applied Mathematics  
University of the Witwatersrand, Johannesburg, South Africa

## Abstract

Repeatedly training models on their own predictions can reinforce errors. We investigate this feedback using a small transformer that predicts numerical labels from examples of handwritten characters. Our extension also generates another character and its numerical label. After ten recursive generations under matched 300,000-episode budgets, five-seed comparisons find losses of 12.19 percentage points in Base numerical-label accuracy and 19.73 points in Extended context-valid character sampling. In five matched repair experiments, correcting invalid generated character–label pairs restores both context-valid outputs and the model’s attention-mediated response to changes in the prompt. Thus recursive feedback produces a repairable failure of context-appropriate generation, while contextual numerical-label retrieval can survive.

## 1. Introduction and related work

In-context learning (ICL) uses labelled examples in a prompt to predict without updating weights at inference. Previous-token heads carry nearby information; induction heads use repeated inputs to retrieve contextual labels [4,6]. Recursive training can amplify errors when models learn from predecessors’ predictions [5]; retaining real data can mitigate this in other settings [2].

We ask whether recursive feedback disrupts contextual numerical retrieval or instead degrades the distribution of generated continuations. Five-seed comparisons, repairs of malformed targets, and attention-head interventions distinguish degraded contextual generation from partly preserved numerical retrieval; they do not establish a unique neural pathway.

## 2. Task: one output versus three

An Omniglot episode assigns two characters different labels from {0, 1, 2, 3, 4}. A later query repeats one character. Because labels change between episodes, success requires reading the current prompt.

Illustration: A → 3, B → 1; the query is A.

Base: output y1 = 3.

Extended: output (y1, q2, y2) = (3, B, 1) or (3, A, 3).

Here q2 must be one of the two context characters, and y2 must equal that character’s assigned label. An out-of-context q2 is invalid even if its number looks plausible.

Base predicts a number; Extended also generates a character q2 and its assigned number y2. Clean q2 targets are sampled uniformly from the two context characters. Unlike a two-way pointer, an unmasked 50-class character head can generate off-context symbols, making contextual errors measurable. This is our extension of Singh et al. [6], using 512-dimensional features extracted by an 18-layer residual network (ResNet-18) from Omniglot [3].

## 3. Recursive feedback and controls

Generation 0 (G0) learns from correct targets. In generations 1–10 (G1–G10), fresh models learn from their predecessors’ answers. If only A and B appear but the predecessor predicts q2 = C, the next model learns that invalid target. Input contexts remain real; only sampled answer continuations are recycled, isolating errors in supervision rather than effects of entirely generated input contexts. Real-target controls instead train fresh models on correct answers (`clean_only` in code). Additional controls use a fixed G0 generator or a 50/50 mixture of correct and generated targets. A separate accumulating-corpus pilot retains earlier examples.

**Figure 1.** Five-seed validation at 300k episodes/generation. Base: paired G10 accuracy (ticks: means); Extended: G1–G10 valid-symbol rate (mean ± SD). Blue: clean; red: recursive.

*Figure labels in source:* Real targets; Recursive; Validation accuracy (%); Five matched seeds; Base: G10 label accuracy; Generation (1, 3, 5, 7, 10); Extended: valid next symbol; Real-target control; Recursive.

## 4. Methods and measurement

**Model and splits.** Both tasks use Singh et al.’s two-layer, eight-head, width-64 causal attention-only transformer [6]. Base predicts one of five contextual numbers; Extended adds a 50-class character head and a second five-class number output. Omniglot is divided into 50 training, 1,473 validation, and 100 final-test classes. Held-out label assignments evaluate contextual generalisation; new handwriting exemplars probe input sensitivity. Character identity is scored on representable training-class symbols. A historical G3 evaluation accessed the final-test classes, so G10 is validation-only.

**Training.** Both Base and Extended use 300,000 episodes per generation, batch 32 and five independent initialisations for the matched G10 comparison. Base uses Adam at learning rate 10−5 [6]; Extended uses Adam at 3 × 10−4, selected in a one-seed validation pilot against 10−4 (valid sampled q2: 96.48% versus 93.36%). Extended trains y1, q2, y2 with equally weighted cross-entropy losses. Successor models start from fresh initialisations and sample predecessor target continuations at temperature 1.0; input image features stay real. Controlled repair pairs share their training conditions.

**Metrics.** Base accuracy requires the correct contextual number; Extended valid-q2 rate requires a sampled next character to occur among the two prompt characters. Valid q2+y2 additionally requires the correct second number, but does not score y1. Context-valid softmax mass sums the predicted probabilities of the two permitted characters. Teacher-forced y2 evaluation supplies correct preceding outputs rather than generating them freely.

**Output boundary.** The unmasked next-character head covers only 50 training identities; it can emit out-of-context symbols. A next-character identity score is undefined for the 1,473 unseen validation classes or 100 reserved test classes, although contextual numerical retrieval is measurable there.

**Causal probes.** We audit G9-generated targets and retrain G10 after jointly repairing invalid q2 symbols and associated y2 values. Candidate heads are selected using G0 training-condition attention scores and evaluated with single/joint ablations on validation episodes. Counterfactual support edits and within-model activation patching transfer head activations at the q2 position [1]. Head ablations use one primary model seed; the matched repair and patching study uses five seed pairs and 128 fixed validation episodes per model.

## 5. Main result: sampled validity degrades

At 300,000 training episodes per generation, five matched Base seeds have mean G10 held-out-label accuracy 61.56% with clean targets and 49.38% with recursive targets. The Base clean baseline is modest (one seed below 60%), but its paired accuracy gap is +12.19 percentage points (pp). Extended G10 sampled q2 validity is 96.76% clean versus 77.03% recursive, a +19.73 pp gap. Four of five Base seed pairs and all five Extended pairs favour clean training (Table 1; Fig. 1). The tasks use different output metrics and are not directly comparable in intrinsic susceptibility.

Two Extended clean controls achieve only 53.13% and 48.63% on held-out label assignments (recursive: 88.67% and 67.38%), but reach 95.70% and 97.46% on held-out character classes. These role-specific failures make the clean numerical-label baseline uneven; we therefore do not claim that recursive feedback uniformly destroys or improves retrieval.

**Table 1.** G10 clean-minus-recursive validation gaps (pp; five paired seeds; descriptive 95% t intervals).

| Outcome | Gap (pp) | 95% interval |
|---|---:|---|
| Base contextual-label accuracy | 12.19 | [3.29, 21.09] |
| Extended sampled q2 validity | 19.73 | [14.24, 25.22] |

In the primary Extended seed, frozen-G0 and 50/50 real–generated training controls retain 94.53% and 94.14% valid-q2 sampling, respectively.

## 6. Diagnosing and repairing generated errors

**Error audit.** Of 300,000 G9-generated training answers, 20.38% select a q2 absent from the prompt, whereas 1.79% give a wrong y2 when q2 is valid, and 1.77% give a wrong y1 (categories may overlap). Invalid next characters are the dominant observed error.

On 128 further validation prompts, the control puts 96.93% of its probability on the two allowed characters, compared with 78.70% under recursion. Yet both models’ top-ranked choices are valid in all 128 prompts. Per-prompt uncertainty rises (entropy 0.834 to 1.380 nats), while diversity across the whole collection stays high (global entropy 3.773 versus 3.730). Thus probability leaks to off-context characters despite high aggregate diversity: this is conditional distribution degradation, not wholesale vocabulary collapse. Greedy q2 remains valid on all 128 audited prompts. This diagnostic covers one model pair, not five.

**Matched target repair.** We correct selected G9-generated training answers, then retrain G10 under the same optimisation and data budget. Table 2 tests whether repairing q2, its associated number y2, or both helps.

**Table 2.** G10 sampled-symbol validity and q2–y2 consistency after target repair (one seed).

| G10 target correction | Valid q2 | Valid q2 + correct y2 |
|---|---:|---:|
| None | 82.81% | 81.45% |
| Invalid symbol only | 97.27% | 82.03% |
| Associated number only | 82.03% | 74.02% |
| Both symbol and number | 97.66% | 95.70% |

In this one-seed factorial test, repairing q2 alone raises validity to 97.27%, but correct q2–y2 pairs remain at 82.03%; joint repair raises correct pairs to 95.70%. Across five matched G10 retrainings, jointly repairing invalid q2 and its label raises valid-symbol probability mass by 21.17 pp (descriptive 95% paired interval: 16.76–25.58 pp). This does not isolate a purely symbol-only mediated effect.

## 7. Attention-mediated context recovery

**Candidate induction pathway.** In Extended G0, first-layer head 4 puts 0.926 attention on previous-token positions and second-layer head 1 has +0.901 contextual-label matching margin. At G10, removing the G0-selected second-layer head reduces valid-q2 probability mass by 1.52 pp (clean) and 4.95 pp (recursive). Jointly removing heads 1/0/7 reduces mass by 10.12 and 17.43 pp, respectively (one seed, 128 validation prompts). This implicates attention components but does not prove a connected or unique inter-layer pathway; a measured schematic appears in the supplement.

**Direct next-character intervention.** Changing a support symbol from B to C alters the logit preference for C over B. Across five matched seed pairs, repairing invalid G9-generated q2–y2 targets strengthens this contextual response from 9.13 to 15.06 logit units (gain 5.94; descriptive 95% interval 5.39–6.48), near the clean-control 15.72. Valid-q2 mass increases from 75.80% to 96.97%. Second-layer activation patching transfers 82.72% of contextual preference before repair and 82.62% afterward, although its absolute transferred logit response rises from 7.63 to 12.50 units. Patching occurs within each trained model, teacher-forcing y1 on 128 fixed validation prompts; similar fractions do not establish identical circuits.

**Contextual retrieval persists.** In one earlier model pair, the recursive model follows swapped support labels in 127/128 held-out-label and 125/128 held-out-class episodes. Thus valid numeric context use can coexist with degraded sampled character generation.

## 8. Discussion and limitations

Matched retraining supports a causal contribution of malformed synthetic targets to reduced context-valid character generation. Ablations and patching show sensitivity of predictions to attention-mediated information, but separately initialised models need not place equivalent computations at the same head indices. Joint knockouts may perturb hidden states, and repair modifies all learned weights. We therefore identify candidate functional components, not a proven corruption-to-induction-head causal path.

Matching Base and Extended at 300,000 episodes removes an exposure-count discrepancy, not differences in task difficulty, outputs or optimisation cost. Base clean controls remain weak at this budget, limiting strong cross-task comparisons. A separate one-seed accumulating-corpus study yields G10 valid-q2 rates of 87.11% (real-target history) versus 83.20% (cumulative recursive) under 100,000 episodes per generation; it is not directly comparable. Across held-out handwriting exemplars, even the primary clean model samples context-valid symbols only 35.94% of the time (recursive: 27.93%). G10 evidence remains validation-only; large language model behaviour is not established.

## 9. Conclusion

Five-seed matched-budget results show weaker contextual numerical accuracy and substantially worse context-valid character sampling after recursive target-only feedback. Joint target correction restores valid-symbol probability and strengthens an attention-mediated response to context changes. The study identifies a recoverable failure of conditional generation, with candidate induction-related heads but no unique causal circuit.

## References

[1] Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. 2023. Towards Automated Circuit Discovery for Mechanistic Interpretability. In Advances in Neural Information Processing Systems, Vol. 36, 16318–16352. https://arxiv.org/abs/2304.14997.

[2] Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, et al. 2024. Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data. arXiv:2404.01413. https://arxiv.org/abs/2404.01413.

[3] Brenden M. Lake, Ruslan Salakhutdinov, and Joshua B. Tenenbaum. 2015. Human-level Concept Learning through Probabilistic Program Induction. Science 350, 6266 (2015), 1332–1338. https://doi.org/10.1126/science.aab3050.

[4] Catherine Olsson, Nelson Elhage, Neel Nanda, et al. 2022. In-context Learning and Induction Heads. Transformer Circuits Thread. arXiv:2209.11895. https://arxiv.org/abs/2209.11895.

[5] Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. 2024. AI Models Collapse When Trained on Recursively Generated Data. Nature 631 (2024), 755–759. https://doi.org/10.1038/s41586-024-07566-y.

[6] Aaditya K. Singh, Ted Moskovitz, Felix Hill, Stephanie C. Y. Chan, and Andrew M. Saxe. 2024. What Needs to Go Right for an Induction Head? A Mechanistic Study of In-context Learning Circuits and Their Formation. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024), Proceedings of Machine Learning Research, Vol. 235, 45637–45662. https://proceedings.mlr.press/v235/singh24c.html.
