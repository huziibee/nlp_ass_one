from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/model-audit"))
    args = p.parse_args()
    d = json.loads((args.out / "base_training_summary.json").read_text())
    ev = d["evaluation_tracks"]
    allowed = ("fsl_train", "fsl_val_rl", "fsl_train_valex", "fsl_val_class")
    assert set(ev) == set(allowed)
    assert not d["model_final_test_evaluated"]

    def fmt(x):
        return f"{100*float(x):.2f}%"

    def f4(x):
        return f"{float(x):.4f}"

    rows = []
    for k in allowed:
        a = ev[k]["accuracy"]
        loss = ev[k]["loss"]
        s = d["evaluation_timeline"]
        above90 = next((step for step, v in zip(s, a) if v >= 0.90), None)
        above95 = next((step for step, v in zip(s, a) if v >= 0.95), None)
        rows.append(
            f'| {ev[k]["description"]} | {fmt(a[0])} | **{fmt(a[-1])}** | '
            f'{f4(loss[-1])} | {above90 or "not reached"} | {above95 or "not reached"} |'
        )
    m = f"""# Original Singh Base model: one-million-sequence validation-only reproduction

**Evidence:** [GitHub Actions training run](https://github.com/huziibee/nlp_ass_one/actions/runs/38612082174) and downloadable checkpoint artifact (\`base-paper-1m-validation-only\`). 
The report and plots were generated from the original authors' **saved \`log.h5\`**, not invented predictions.

## Reproduction protocol

- Original reference code and ResNet18 feature HDF5 from pinned \`external/icl-dynamics\` submodule (original author commit \`85b895c720844e795734b9391b32d2619065f9a1\`).
- Training: **1,000,000 training sequences** with batch 32 (=31,250 gradient updates). Original author's main-run config: 2 layers, 8 heads, model width 64, Adam LR \(10^{-5}\), 50 training classes, 8 contextual label pairs, 1 exemplar/class; initialisation seed 5.
- Evaluation on **512 fixed examples per condition** at 0,20k,...,1M; all original training/evaluation mathematics and architecture are reused. To comply with course final testing, the original script's periodic **test-class** evaluator was replaced with a **validation-class** evaluator. Evaluation and checkpoint frequency was reduced.
- A separate 100-class final-test pool remains sealed. No final-test neural accuracy was obtained. This is **one model seed**, not a robustness replication.

## What we measured

| Development condition | Initial accuracy | Final accuracy | Final cross-entropy | First ≥90% | First ≥95% |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The unconditional five-label chance baseline is **20%**. A random choice between the two context labels yields **50%** if the learner already restricts itself to those two labels.

### Accuracy over training

![Original Base 1M: development accuracies](02_base_development_accuracy.png)

### Development loss

![Original Base 1M: development cross-entropy](03_base_development_losses.png)

### Training loss

![Original Base 1M: smoothed training loss](01_base_training_loss.png)

## What is and is not established

**Evidence:**
1. The unmodified original Singh Base JAX implementation and HDF5 features are operational and train to near-perfect development accuracy on familiar class/label pairs.
2. Generalisation to **unseen contextual relabelings** is very strong at the end of training; this supports learned dependence on the current context rather than memorised symbol→number mappings.
3. Generalisation to 1,473 reserved validation classes is also strong under the original exemplar-0 distribution.
4. Generalisation to **different exemplars of the same characters** is worse than the other development conditions, despite trained-label near-ceiling results. Different drawings change the similarity structure, so this is a plausible input-level sensitivity to investigate.

**Not established:** a specific causal induction-head circuit, per-head ablation necessity, recursive synthetic-data model collapse, Extended q2 generation, multi-seed reproducibility, or any final-test result. Head/circuit interventions require loaded checkpoints and controlled evaluation.

## What next?

- Analyze the saved checkpoints with authors' *optogenetics* intervention framework on development/validation episodes: previous-token attention, induction matching, head ablation and retain-only tests. Head indices selected on training-only sources; all results exploratory until replicated.
- Decide the Extended symbol-output representation. A 50-way global class-ID head cannot directly generate brand-new test class identities; a 2-way pointer rules out invalid symbols by design. See [Extended output contract](../EXTENDED_OUTPUT_CONTRACT.md).
- Implement Extended Base G0 and both recursive Base/Extended, with matched fixed-source, clean and data-repair controls. Protect final test until all training and selection are frozen.

**Method warning:** High contextual label accuracy cannot alone prove an induction-head circuit, and head ablation cannot alone establish a training-distribution mechanism.
"""
    (args.out / "README.md").write_text(m, encoding="utf8")
    print(f"MODEL_REPORT_GENERATED={args.out/'README.md'}")


if __name__ == "__main__":
    main()
