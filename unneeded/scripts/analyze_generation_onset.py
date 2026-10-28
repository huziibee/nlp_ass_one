"""Exploratory onset comparison on original author-feature G1-G10 source trajectories.
No hypothesis-test onset, no reuse of original protected test; 5 percentage points
is a post-hoc descriptive threshold, not a preregistered changepoint.
"""
from pathlib import Path
import argparse,csv,json
TASKS={
 "base":("val_relabel_accuracy","Base contextual numeric-label accuracy"),
 "extended":("val_relabel_generated_symbol_context_validity","Extended freely generated contextual symbol validity"),
}
def load(root,task,arm,metric):
 p=root/f"g10-trajectory-{task}-{arm}"/f"{task}_{arm}.csv"
 rows=list(csv.DictReader(p.open()))
 assert len(rows)==10 and [int(r["generation"]) for r in rows]==list(range(1,11))
 assert all(r["validation_only"]=="True" and r["single_seed_descriptive"]=="True" for r in rows)
 return [float(r[metric]) for r in rows],str(p)
def main():
 q=argparse.ArgumentParser()
 q.add_argument("--root",type=Path,default=Path("docs/experiments/archived_original_runs/g10_trajectories"))
 q.add_argument("--out",type=Path,default=Path("docs/experiments/behavioral_onset"))
 args=q.parse_args()
 records=[];summary={"source":"Original source-feature matched primary-seed validation curves","n_seeds":1,
 "not_a_statistical_changepoint":True,"exploratory_descriptive_threshold_pp":5.0,
 "final_test_accessed":False,"tasks":{}}
 for task,(metric,label) in TASKS.items():
  clean,p1=load(args.root,task,"clean_only",metric)
  recur,p2=load(args.root,task,"recursive_replacement",metric)
  gaps=[100*(a-b) for a,b in zip(clean,recur)]
  first_nonzero=next((i+1 for i,gap in enumerate(gaps) if gap>0),None)
  first_gt5=next((i+1 for i,gap in enumerate(gaps) if gap>=5),None)
  summary["tasks"][task]={"label":label,"metric":metric,
     "clean_source":p1,"recursive_source":p2,"first_positive_difference_generation":first_nonzero,
     "first_at_least_5pp_generation":first_gt5,"g1_gap_pp":gaps[0],
     "g10_gap_pp":gaps[-1],"trajectory_gaps_pp":gaps}
  for gen,(a,b,gap) in enumerate(zip(clean,recur,gaps),1):
   records.append({"task":task,"generation":gen,"clean":a,"recursive":b,
    "clean_minus_recursive_pp":gap,"validation_only":True})
 args.out.mkdir(parents=True,exist_ok=True)
 with (args.out/"per_generation.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
 rows=summary["tasks"]
 (args.out/"README.md").write_text(
   "# Exploratory generation-of-degradation audit\n\n"
   "Source: actual primary-seed Omniglot feature trajectories in repo. "
   "**No protected final test accessed.** This is descriptive rather than a preregistered change-point test.\n\n"
   "| Task | G1 gap (pp) | First >=5pp generation (post hoc) | G10 gap (pp) |\n"
   "|---|---:|---:|---:|\n"+
   "".join(f"| {k} | {v['g1_gap_pp']:.2f} | G{v['first_at_least_5pp_generation']} | {v['g10_gap_pp']:.2f} |\n" for k,v in rows.items())+
   "\nThe first positive sample difference is not necessarily evidence of statistically detected onset. "
   "Base label copying and Extended free symbol generation are different outcome definitions and training budgets. "
   "For meaningful independent model-seed G10 inference refer to docs/experiments/g10_three_seed/.\n")
 print("GENERATION_ONSET_EXPLORATORY="+json.dumps({k:v["first_at_least_5pp_generation"] for k,v in rows.items()}))
if __name__=="__main__":main()
