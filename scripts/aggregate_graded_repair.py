"""Merge 0/25/50/75/100% genuinely matched primary-seed repair evidence."""
import json,csv,argparse
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--root",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args();records=[]
for percentage in (0,25,50,75,100):
    folder=(a.root/"raw" if percentage==0 else
            a.root/"full" if percentage==100 else a.root/str(percentage))
    ms=list(folder.rglob("metrics.json"))
    if len(ms)!=1:raise AssertionError(f"Expected one metrics JSON at {folder}: {ms}")
    d=json.loads(ms[0].read_text())
    if d.get("final_test_evaluated",d.get("final_test_access")) is not False:
        raise AssertionError("Final test accessed")
    if percentage in (0,100):
        episodes=d.get("training_exposures",d.get("train_exposures"))
        assert episodes==300000
        if percentage==0:assert d["condition"]=="recursive_replacement"
        else:assert d["repair"]=="fix_invalid_q2"
    else:
        assert d["correction_fraction_of_invalid_generated_examples"]==percentage/100
        assert d["training_exposures"]==300000
    v=d["history"][-1]["validation"]["val_relabel"]
    records.append({"repair_pct":percentage,
       "validity":v["free_next_symbol_context_validity"],
       "joint_valid_symbol_label_correct":v["free_second_label_context_consistency_all"],
       "first_label_accuracy":v["first_label_accuracy"],
       "second_teacher_forced_accuracy":v["teacher_forced_second_label_accuracy"],
       "n_training_episodes":300000,
       "n_seeds":1})
a.out.mkdir(parents=True,exist_ok=True)
with (a.out/"graded_repair_effects.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]))
    w.writeheader();w.writerows(records)
(a.out/"summary.json").write_text(json.dumps({
 "results":records,
 "intervention":"off-context q2 plus corresponding second label, selected Bernoulli fraction",
 "note":"one primary seed; 0 and 100 are existing full-budget matched reference checkpoints",
 "descriptive_only":True,"protected_final_test_accessed":False
},indent=2))
print("GRADED_REPAIR_COMPLETE="+json.dumps(records))
