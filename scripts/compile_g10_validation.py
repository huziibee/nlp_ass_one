"""Combine G1-G10 model-validation histories from original G3 and continued G10.

No final-test data are loaded. Original G0 is reported separately because
the initial Base/Extended G0 record used a different validation episode seed.

Use separately for each task and arm:
  python scripts/compile_g10_validation.py --task base --arm recursive_replacement \
    --g3-root outputs/base_original_g3 --g10-root outputs/base_g10 \
    --out outputs/compiled/base_recursive_replacement.csv

This script deliberately reports uncertainty as unknown for single-seed data.
It does NOT assume that teacher-forced Extended accuracy measures quality of
free-running generated next symbols.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

ARMS=("clean_only","recursive_replacement","real_anchor_half","frozen_g0")
TASKS=("base","extended")

BASE_KEYS=(
    "val_relabel_accuracy","val_relabel_loss",
    "val_classes_accuracy","val_classes_loss",
)
EXTENDED_KEYS=(
    "val_relabel_first_label_accuracy",
    "val_relabel_second_teacher_forced_accuracy",
    "val_relabel_generated_symbol_context_validity",
    "val_relabel_generated_second_context_consistency",
    "val_relabel_next_symbol_valid_probability_mass",
    "val_exemplar_first_label_accuracy",
    "val_exemplar_second_teacher_forced_accuracy",
    "val_classes_first_label_accuracy",
    "val_classes_second_teacher_forced_accuracy",
)
KEYS={
    "base":BASE_KEYS,
    "extended":EXTENDED_KEYS,
}

def extract(task,validation):
    if task=="base":
        if set(validation)!={"val_relabel","val_classes"}:
            raise AssertionError("Expected only original Base validation evaluators")
        return {
            "val_relabel_accuracy":validation["val_relabel"]["accuracy"],
            "val_relabel_loss":validation["val_relabel"]["loss"],
            "val_classes_accuracy":validation["val_classes"]["accuracy"],
            "val_classes_loss":validation["val_classes"]["loss"],
        }
    if set(validation)!={"val_relabel","val_exemplar","val_classes"}:
        raise AssertionError("Expected original-feature Extended validation evaluators")
    r=validation["val_relabel"]
    e=validation["val_exemplar"]
    c=validation["val_classes"]
    return {
        "val_relabel_first_label_accuracy":r["first_label_accuracy"],
        "val_relabel_second_teacher_forced_accuracy":r["teacher_forced_second_label_accuracy"],
        "val_relabel_generated_symbol_context_validity":r["free_next_symbol_context_validity"],
        "val_relabel_generated_second_context_consistency":r["free_second_label_context_consistency_all"],
        "val_relabel_next_symbol_valid_probability_mass":r["next_symbol_valid_probability_mass"],
        "val_exemplar_first_label_accuracy":e["first_label_accuracy"],
        "val_exemplar_second_teacher_forced_accuracy":e["teacher_forced_second_label_accuracy"],
        "val_classes_first_label_accuracy":c["first_label_accuracy"],
        "val_classes_second_teacher_forced_accuracy":c["teacher_forced_second_label_accuracy"],
    }

def extract_validation(task,meta):
    if meta.get("final_test_evaluated") is not False:
        raise AssertionError("Final-test evidence cannot enter G10 validation aggregation")
    if task=="base":
        if meta.get("whole_prefix_generated") is not False:
            raise AssertionError("Mislabelled Base generation data")
        if meta.get("class_split")!=[50,1473,100]:
            raise AssertionError("Wrong Base original split")
        if meta.get("optimizer_exposures")!=1_000_000:
            raise AssertionError("Wrong Base training exposures")
        return meta["history"][-1]["validation"]
    if meta.get("recursively_generated_input_prefix") is not False:
        raise AssertionError("Mislabelled Extended generation data")
    if meta.get("training_class_split")!=[50,1473,100]:
        raise AssertionError("Wrong Extended original split")
    if meta.get("training_exposures")!=300_000:
        raise AssertionError("Wrong Extended training exposures")
    return meta["history"][-1]["validation"]

def compile_arm(task,arm,g3_root,g10_root):
    if task not in TASKS or arm not in ARMS:
        raise ValueError("Invalid task/arm")
    records=[]
    digests={}
    for gen in range(1,11):
        root=g3_root if gen<=3 else g10_root
        folder=root/arm/f"g{gen:02d}"
        file=folder/"metrics.json"
        checkpoint=folder/"model.eqx"
        if not file.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"Expected trained checkpoint and complete metrics: {folder}")
        b=file.read_bytes()
        m=json.loads(b)
        if m.get("generation")!=gen:
            raise AssertionError(f"Generation mismatch for {file}")
        if (m.get("arm") if task=="base" else m.get("condition"))!=arm:
            raise AssertionError(f"Condition mismatch for {file}")
        digest=hashlib.sha256(b).hexdigest()
        digests[str(gen)]=digest
        val=extract_validation(task,m)
        row={"generation":gen,"task":task,"arm":arm,
             "validation_only":True,
             "independent_model_seeds":1,
             "single_seed_descriptive":True}
        row.update(extract(task,val))
        records.append(row)
    # Verify G10 continuation includes the exact original G1-G3 metadata
    s=json.loads((g10_root/"summary.json").read_text())
    if s.get("final_test_evaluated") is not False:
        raise AssertionError("G10 experiment contains final-test evaluation")
    if len(s["arms"][arm])!=10 or [x["generation"] for x in s["arms"][arm]]!=list(range(1,11)):
        raise AssertionError("Incomplete G1-G10 pedigree")
    for rec in records:
        g=rec["generation"]
        last=s["arms"][arm][g-1]["last_validation"]
        for k,v in extract(task,last).items():
            if abs(float(rec[k])-float(v))>1e-6:
                raise AssertionError(f"History/metadata disagree for {arm} G{g} {k}")
    return records,digests

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=TASKS,required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--g3-root",type=Path,required=True)
    p.add_argument("--g10-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rows,digests=compile_arm(a.task,a.arm,a.g3_root,a.g10_root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta={
        "task":a.task,"arm":a.arm,
        "validated_generations":list(range(1,11)),
        "final_test_evaluated":False,
        "single_seed":True,
        "g0_omitted_due_to_different_validation_episode_seed":True,
        "checkpoint_metrics_sha256_by_generation":digests,
        "g10":rows[-1],
    }
    (a.out.with_suffix(".json")).write_text(json.dumps(meta,indent=2))
    print("G10_VALIDATION_TRAJECTORY="+json.dumps(meta),flush=True)

if __name__=="__main__":
    main()
