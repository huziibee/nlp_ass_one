"""Export measured Extended validation trajectories without opening final test.

Inputs are actual trained model JSON logs with saved checkpoints. No test
values, made-up numbers, or model-science interpretations are introduced.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

ARMS=("clean_only","recursive_replacement","real_anchor_half","frozen_g0")
METRICS=("first_label_accuracy","first_label_loss",
         "teacher_forced_second_label_accuracy",
         "teacher_forced_second_label_loss",
         "next_symbol_valid_probability_mass",
         "free_next_symbol_context_validity",
         "free_second_label_context_consistency_all")
FIELDS=("arm","generation","training_exposures","train_loss",
        "real_fraction","validation_scope") + METRICS


def extract_trajectories(root,arms,gens):
    if gens<1 or len(set(arms))!=len(arms) or not set(arms)<=set(ARMS):
        raise ValueError("Invalid generation/arm specification")
    results=[]
    for arm in arms:
        for g in range(1,gens+1):
            folder=Path(root)/arm/f"g{g:02d}"
            data=json.loads((folder/"metrics.json").read_text())
            if data["condition"]!=arm or data["generation"]!=g:
                raise AssertionError("Wrong model provenance")
            if data["final_test_evaluated"] or data["recursively_generated_input_prefix"]:
                raise AssertionError("Not a protected train/validation experiment")
            if data["training_class_split"]!=[50,1473,100] or data["label_pair_split"]!=[8,2,0]:
                raise AssertionError("Canonical Singh split changed")
            if not (folder/"model.eqx").is_file():
                raise FileNotFoundError(folder/"model.eqx")
            if data["num_real_targets"]+data["num_synthetic_targets"]!=data["training_exposures"]:
                raise AssertionError("Training target provenance count mismatch")
            for step in data["history"]:
                for scope,vals in step["validation"].items():
                    if scope not in ("val_relabel","val_exemplar","val_classes"):
                        raise AssertionError("Non-development evaluator used")
                    if vals.get("final_test_evaluated") is not False:
                        raise AssertionError("Final test was accessed")
                    row={
                        "arm":arm, "generation":g,
                        "training_exposures":step["sequence_exposures"],
                        "train_loss":step["mean_three_token_train_loss"],
                        "real_fraction":step["real_target_fraction"],
                        "validation_scope":scope,
                    }
                    for metric in METRICS:
                        row[metric]=vals.get(metric,"")
                    results.append(row)
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--generations",type=int,required=True)
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    a=p.parse_args()
    rows=extract_trajectories(a.root,a.arms,a.generations)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"EXTENDED_TRAJECTORY_CSV={a.out} records={len(rows)}")


if __name__=="__main__":
    main()
