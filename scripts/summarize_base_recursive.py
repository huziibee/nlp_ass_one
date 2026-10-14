"""Summarize independently run original-Base recursive arms from verified JSON logs.

No final-test data and no head-based mechanism claims. Reads individual
training curves and emits CSV for downstream analysis. A generated summary
is evidence only after the linked GitHub Actions workflows succeed.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

ARMS=("clean_only","recursive_replacement","real_anchor_half","frozen_g0")
COLUMNS=("arm","generation","train_sequences","train_loss",
         "real_fraction","val_label_pair_accuracy","val_label_pair_loss",
         "val_class_accuracy","val_class_loss")


def read_arm(root: Path, arm: str, *, expected_generations: int) -> list[dict]:
    result=[]
    if arm not in ARMS:
        raise ValueError(f"Unknown Base condition {arm}")
    for generation in range(1,expected_generations+1):
        path=root/arm/f"g{generation:02d}"/"metrics.json"
        data=json.loads(path.read_text())
        if data["arm"]!=arm or data["generation"]!=generation:
            raise AssertionError(f"Wrong source/provenance for {path}")
        if data["final_test_evaluated"] or data["whole_prefix_generated"]:
            raise AssertionError("Unsafe source or mislabelled experiment")
        if not (root/arm/f"g{generation:02d}"/"model.eqx").is_file():
            raise FileNotFoundError(f"Missing model checkpoint for {path}")
        for h in data["history"]:
            metrics=h["validation"]
            if set(metrics)!={"val_relabel","val_classes"}:
                raise AssertionError("Expected validation-only tracks")
            result.append({
                "arm":arm,
                "generation":generation,
                "train_sequences":h["seen_training_sequences"],
                "train_loss":h["last_train_loss"],
                "real_fraction":h["real_training_target_fraction"],
                "val_label_pair_accuracy":metrics["val_relabel"]["accuracy"],
                "val_label_pair_loss":metrics["val_relabel"]["loss"],
                "val_class_accuracy":metrics["val_classes"]["accuracy"],
                "val_class_loss":metrics["val_classes"]["loss"],
            })
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--generations",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    a=p.parse_args()
    if a.generations<1: raise ValueError("Need >=1 generations")
    rows=[]
    for arm in a.arms:
        rows.extend(read_arm(a.root,arm,expected_generations=a.generations))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print("BASE_RECURSION_CSV="+str(a.out))
    for arm in a.arms:
        final=[r for r in rows if r["arm"]==arm and
               r["generation"]==a.generations][-1]
        print("BASE_LAST="+json.dumps(final))


if __name__=="__main__":
    main()
