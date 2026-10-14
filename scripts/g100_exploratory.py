"""LOW-VALUE exploratory Omniglot Extended G100: 100 generations of small-budget target feedback.
Original class split; never access protected final test. Both clean and recursive arms
receive identical SMALL per-generation exposure; do not compare absolute effects
against the original 300,000-exposure-per-generation G10 paper experiment.
"""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"external"/"icl-dynamics"))
import jax
from scripts.extended_g0 import load_config,feature_prototypes
from scripts.extended_recursive import load_verified_g0,train_generation
from src.extended_omniglot import make_episodes

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--base-artifact",type=Path,required=True)
 p.add_argument("--g0-artifact",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 p.add_argument("--steps",type=int,default=512)
 p.add_argument("--seed",type=int,default=7100)
 p.add_argument("--source-commit",default="unknown")
 a=p.parse_args()
 if not (32<=a.steps<=1024):raise ValueError("bounded exploratory step budget")
 c=load_config(a.base_artifact)
 if c["class_split"]!=[50,1473,100] or c["fs_relabel_split"]!=[8,2,0]:
  raise AssertionError("Original split changed")
 g0,g0meta=load_verified_g0(c,a.g0_artifact,require_gate=True)
 prototypes=feature_prototypes(c)
 sampler=make_episodes(c,role="train",batch_size=32)
 validation={role:make_episodes(c,role=role,batch_size=128)(
  jax.random.PRNGKey(a.seed+911+i)) for i,role in enumerate(
  ("val_relabel","val_exemplar","val_classes"))}
 out=a.out;out.mkdir(parents=True,exist_ok=True)
 arms=("clean_only","recursive_replacement")
 summary={"study":"LOW VALUE exploratory G100 original Omniglot Extended",
 "source_commit":a.source_commit,"max_generations":100,"steps_per_generation":a.steps,
 "training_exposures_per_generation":a.steps*32,"primary_extended_G10_exposures":300000,
 "not_training_budget_comparable_to_primary_G10":True,"data_seed":a.seed,
 "model_initialization_seed":int(g0meta["seed"]),"class_split":[50,1473,100],
 "real_prefixes_fixed":True,"full_historical_corpus_accumulation":False,
 "final_test_evaluated":False,"arms":{}}
 for arm in arms:
  summary["arms"][arm]=[]
  predecessor=g0
  for generation in range(1,101):
   t=time.monotonic()
   predecessor,m=train_generation(
    arm=arm,generation=generation,predecessor=predecessor,g0=g0,
    config=c,train_sampler=sampler,val=validation,prototypes=prototypes,
    steps=a.steps,batch_size=32,eval_every=a.steps,seed=a.seed,
    lr=0.0003,temperature=1.0,out=out,init_seed=int(g0meta["seed"]))
   if m["final_test_evaluated"] is not False or m["training_exposures"]!=a.steps*32:
    raise AssertionError("Invalid G100 training provenance")
   v=m["history"][-1]["validation"]["val_relabel"]
   d={"generation":generation,"arm":arm,
    "first_label_accuracy":v["first_label_accuracy"],
    "second_label_teacher_forced_accuracy":v["teacher_forced_second_label_accuracy"],
    "free_next_symbol_context_validity":v["free_next_symbol_context_validity"],
    "free_second_label_context_consistency_all":v["free_second_label_context_consistency_all"],
    "next_symbol_valid_probability_mass":v["next_symbol_valid_probability_mass"],
    "runtime_seconds":round(time.monotonic()-t,2)}
   summary["arms"][arm].append(d)
   (out/"summary.json").write_text(json.dumps(summary,indent=2))
   with (out/"generation_metrics.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(d));writer.writeheader()
    for x in arms:
     writer.writerows(summary["arms"].get(x,[]))
   print("G100_PROGRESS="+json.dumps(d),flush=True)
 assert all(len(summary["arms"][arm])==100 for arm in arms)
 (out/"COMPLETE.txt").write_text("100 clean and recursive generations completed; final test not opened\n")
 print("G100_COMPLETED="+json.dumps({arm:summary["arms"][arm][-1] for arm in arms}),flush=True)
if __name__=="__main__":main()
