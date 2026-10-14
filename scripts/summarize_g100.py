"""Persist small scientific summaries for G100 without committing checkpoint binaries."""
from pathlib import Path
import argparse,csv,json
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
 a=p.parse_args()
 d=json.loads((a.root/"summary.json").read_text())
 if d["final_test_evaluated"] is not False:raise AssertionError("Final-test use not allowed")
 arms=d["arms"]
 n=min(len(arms["clean_only"]),len(arms["recursive_replacement"]))
 a.out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for i in range(n):
  c=arms["clean_only"][i];r=arms["recursive_replacement"][i]
  if c["generation"]!=i+1 or r["generation"]!=i+1:raise AssertionError("Nonsequential generations")
  rows.append({"generation":i+1,
   "clean_context_validity":c["free_next_symbol_context_validity"],
   "recursive_context_validity":r["free_next_symbol_context_validity"],
   "clean_minus_recursive_validity":c["free_next_symbol_context_validity"]-r["free_next_symbol_context_validity"],
   "clean_first_accuracy":c["first_label_accuracy"],
   "recursive_first_accuracy":r["first_label_accuracy"],
   "clean_second_teacher_forced":c["second_label_teacher_forced_accuracy"],
   "recursive_second_teacher_forced":r["second_label_teacher_forced_accuracy"]})
 with (a.out/"generation_comparison.csv").open("w",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["generation"])
  writer.writeheader();writer.writerows(rows)
 manifest={k:v for k,v in d.items() if k!="arms"}
 manifest["paired_completed_generations"]=n
 manifest["G100_complete"]=bool(n==100 and (a.root/"COMPLETE.txt").is_file())
 manifest["endpoint"]=rows[-1] if rows else None
 (a.out/"manifest.json").write_text(json.dumps(manifest,indent=2))
 (a.out/"README.md").write_text("# Exploratory G100 — LOW assignment value\n\n"
  "Actual two-layer Omniglot Extended model; same small per-generation budget in both arms.\n"
  f"Paired completed generations: {n}/100; full completion: {manifest['G100_complete']}.\n"
  f"Per-generation exposures: {d['training_exposures_per_generation']:,}, versus 300,000 in the primary G10 study.\n"
  "Never use as a direct full-budget G10 to G100 comparison. No protected test used.\n"
  "The source, numeric CSV and manifest are committed here. Full per-generation model checkpoints "
  "remain in GitHub Actions artifact storage with finite retention.\n")
 print("G100_SUMMARY="+json.dumps({"n":n,"complete":manifest["G100_complete"]}),flush=True)
if __name__=="__main__":main()
