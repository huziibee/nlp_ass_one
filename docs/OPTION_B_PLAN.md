# Option B — Structural corruption and induction-matching failure

**Status:** proposed investigation only. No new results or coding-model instructions authorised yet.

## Why this investigation matters
The prior completed symbolic study found (according to its saved reports):
1. Recursive Extended models increasingly generate symbols and values inconsistent with the current context.
2. Extended models lose held-out first/second-value ICL accuracy relative to Fixed-G0 synthetic controls.
3. Artificially forcing correct Layer-2 induction matching repairs a large fraction of errors.

**Missing causal link:** does the growing *invalidity of generated context data* actually **cause** matching failures in the next trained models, rather than both being independent effects of recursive feedback?

### Causal diagram (hypotheses, not proven arrows)
```text
Recursive sampling ─┬─> Invalid/mismatched generated contexts ──?──> Induction matching
                   ├─> Label corruption                        └──> ICL performance
                   ├─> Distribution skew / rare events
                   └─> Other optimization changes
```

## Proposed experiment — fixed synthetic data, controlled repairs
Start from one saved generated dataset at a **preselected generation** and train fresh models with **matched seed, architecture, sample count, optimizer, exposure and test set** under:

| Data condition | Symbol/structural distribution | Assigned labels |
| --- | --- | --- |
| A. Raw synthetic | Keep generated symbols, including invalid if applicable | Generated |
| B. Label-only repair | Keep generated symbols | Oracle-correct labels where a contextual mapping exists |
| C. Structure-only repair | Restore legal symbol selection from original context | Preserve or recompute labels according to a prespecified rule |
| D. Joint repair | Restore legal symbol selection | Oracle-correct labels |

**Important:** if `Q2` is not present in a context, a context-defined oracle label for `Q2` does not exist. Do not silently manufacture one. Define *repair semantics* and counterfactual targets before implementation. Since forcing a valid symbol changes the conditional input distribution, interpret comparisons as interventions on **constructed datasets**, not pure manipulations of a single independent factor unless the factorial is engineered carefully.

## Controls and measurements
- Compare A–D to frozen-G0 synthetic, clean-generator and (if feasible) sampling-only controls.
- Main outcomes: held-out first-value and second-value accuracy/log loss; induction matching attention margin; error-conditioned matching tail; head ablation/retain-only and forced-match repair gain.
- Data outcomes: conditional valid-symbol rate, context-consistent second-value rate, conditional next-symbol entropy, class coverage, label inconsistency. **Global marginal entropy alone can hide contextual invalidity**.
- Train each arm **from scratch** with paired seed/initialisation, equal budgets, logged interventions and unchanged model architecture.
- Separate **training-data intervention** from **inference-time activation intervention**. The former tests which data damage drives trained behaviour; the latter localises a within-model bottleneck.
- Run multiple independent replicates and report seed-level paired effect sizes, not example-level pseudo-replications.

## Two crucial limitations
1. This is a *post-hoc inspired* experiment; a fresh evaluation must not use the previous already-seen held-out label pairs as though they were sealed. Merely changing random seeds on the same previously examined held-out distribution does **not** automatically provide independent confirmation. Consider a withheld input domain, newly registered compositional stress tests or an Omniglot-domain test with held-out character classes, and explicitly distinguish exploratory versus confirmatory inference.
2. The result remains specific to a small task and architecture. Data repair may partially change the target distribution and model learning difficulty. Causal claims must match exactly what was intervened on.

## Decision gate before assigning implementation
We should agree on **(a)** scientific goal/primary contrast, **(b)** task fidelity (original Omniglot vs symbolic), **(c)** synthetic-data replacement versus accumulation, **(d)** repair semantics, **(e)** genuinely new test construction, **(f)** owner/workstream boundaries and (g) compute budget.

Once frozen, parallel coding can split into independent tasks: model/data auditor; repair-data constructor; intervention/evaluator; plots/statistical summaries. **No coding-model prompt before agreement.**
