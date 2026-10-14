"""Collect frozen, protected Base/Extended test scores into one auditable table.

Only consumes verified output JSON from `scripts/evaluate_sealed.py`.
All ten endpoints must share identical seed and independent test size.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

EXPECTED=(
    "base_g0","base_clean_g3","base_recursive_g3",
    "base_halfreal_g3","base_frozen_g3",
    "extended_g0","extended_clean_g3","extended_recursive_g3",
    "extended_halfreal_g3","extended_frozen_g3",
)


def combine(root):
    found={}
    for path in Path(root).rglob("sealed_final_scores.json"):
        name=path.parent.name
        if name.startswith("sealed-final-"):
            name=name[len("sealed-final-"):]
        if name not in EXPECTED or name in found:
            raise ValueError(f"Unexpected or duplicate result for {name}: {path}")
        data=json.loads(path.read_text())
        if data["final_n"]!=4096 or data["fixed_evaluation_seed"]!=20261027:
            raise AssertionError("Mismatched independent test episode protocol")
        if not data["once_only_final_protocol"]:
            raise AssertionError("Not a frozen final-test result")
        if not (1523<=data["sealed_class_ids_min"]<=
                data["sealed_class_ids_max"]<=1622):
            raise AssertionError("Incorrect test class partition")
        if ("extended_" in name)!=(data["model_kind"].startswith("extended")):
            raise AssertionError("Mislabelled model type")
        for metric in data["scores"].values():
            if not isinstance(metric,dict):
                continue
            if metric["n"]!=4096 or not(0<=metric["accuracy"]<=1):
                raise AssertionError("Invalid metric count or accuracy")
            if not math.isfinite(metric["cross_entropy"]):
                raise AssertionError("Nonfinite test cross entropy")
        found[name]=data
    missing=set(EXPECTED)-set(found)
    if missing:
        raise FileNotFoundError("Missing frozen endpoints: "+", ".join(sorted(missing)))
    if len({v["reference_split_manifest_sha256"] for v in found.values()})!=1:
        raise AssertionError("Different data split manifest across endpoints")
    rows=[]
    for name in EXPECTED:
        x=found[name]
        first=x["scores"]["first_label"]
        second=x["scores"].get("second_label_given_clean_q2")
        rows.append({
            "model":name,
            "task":"extended" if name.startswith("extended") else "base",
            "first_label_accuracy":first["accuracy"],
            "first_label_cross_entropy":first["cross_entropy"],
            "second_label_given_clean_q2_accuracy":
                second["accuracy"] if second else None,
            "second_label_given_clean_q2_cross_entropy":
                second["cross_entropy"] if second else None,
            "n":4096,"test_seed":20261027,
        })
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    rows=combine(args.root)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"summary.json").write_text(json.dumps({
        "frozen_test_models":len(rows),
        "results":rows,
        "final_test_seed":20261027,
        "final_test_examples_per_model":4096,
        "class_split":[50,1473,100],
        "next_symbol_accuracy_test":"undefined on unseen class IDs"
    },indent=2))
    header="| Model | First-label acc. | First-label CE | Second-label acc. (clean Q2) | Second-label CE |"
    lines=["# Frozen Base and Extended test evaluation",header,
           "|---|---:|---:|---:|---:|"]
    for x in rows:
        a=x["second_label_given_clean_q2_accuracy"]
        b=x["second_label_given_clean_q2_cross_entropy"]
        lines.append(
            f"| `{x['model']}` | {100*x['first_label_accuracy']:.2f}% "
            f"| {x['first_label_cross_entropy']:.4f} | "
            f"{f'{100*a:.2f}%' if a is not None else 'n/a'} | "
            f"{f'{b:.4f}' if b is not None else 'n/a'} |")
    lines.append("")
    lines.append("The Extended next-symbol class-ID head cannot output the 100 unseen final-test class IDs. Second-label scores are explicitly *teacher forced* with a valid clean Q2. This is not autonomous three-token sequence accuracy. Results are single training seeds and should not be presented as broad significance estimates.")
    (args.out/"README.md").write_text("\n".join(lines)+"\n")
    print("FROZEN_EVALUATION_AGGREGATED="+str(args.out))


if __name__=="__main__":
    main()
