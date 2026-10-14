"""Archive nonbinary primary experiment evidence from downloaded GitHub Actions artifacts.
Preserve per-source run/filename provenance and SHA-256; exclude trained checkpoint
binaries and large binary data. Historic G3 final scores remain explicitly spent.
"""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ALLOWED={".json",".csv",".svg",".md",".txt",".png"}
RUNS={
 "g10_trajectories":38640835888,
 "g10_dashboard":38640835888,
 "symbol_distribution":38640996978,
 "base_circuits":38641237829,
 "extended_circuits":38641720122,
 "primary_repairs":38641963903,
 "extended_independent":38641472801,
 "historical_frozen_g3_test":38633134512,
}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--input",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 records=[];used=0
 for category,runid in RUNS.items():
  src=a.input/category
  if not src.is_dir():raise FileNotFoundError("Missing source download "+str(src))
  for f in sorted(src.rglob("*")):
   if not f.is_file() or f.suffix.lower() not in ALLOWED:continue
   if f.stat().st_size>2_000_000:continue
   rel=f.relative_to(src)
   if any(".."==part for part in rel.parts):raise ValueError("Unsafe filename")
   target=a.out/category/rel
   target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copyfile(f,target)
   raw=target.read_bytes();used+=len(raw)
   records.append({"github_run_id":runid,"source":category,
     "path":str(target.relative_to(a.out)),
     "sha256":hashlib.sha256(raw).hexdigest(),"size_bytes":len(raw)})
 if len(records)<20 or used>30_000_000:raise AssertionError(
  "Unexpected archive coverage or size "+str((len(records),used)))
 metadata={"description":"Permanent textual and figure evidence from full G10, head, distribution, repair and earlier G3 result workflows",
 "created_utc":datetime.now(timezone.utc).isoformat(),
 "files":records,"saved_file_count":len(records),"saved_bytes":used,
 "source_github_repo":"huziibee/nlp_ass_one",
 "all_source_runs":RUNS,
 "contains_full_checkpoints":False,
 "source_figure_and_metadata_only":True,
 "original_G3_final_test_already_evaluated":True,
 "G10_and_G100_final_test_never_opened":True,
 "cautions":["Script copies real workflow outputs, not synthetic estimates",
 "G3 scores are historical and not a fresh test confirmation",
 "G10 primary repair is one seed; cross-seed replication is separate",
 "GitHub Actions source checkpoint artifacts have finite retention"]}
 (a.out/"EVIDENCE_MANIFEST.json").write_text(json.dumps(metadata,indent=2))
 (a.out/"README.md").write_text("# Original Omniglot experiments — permanent source-linked archive\n\n"
 "This folder contains actual downloaded nonbinary GitHub Actions results from "
 "the original Base/Extended G10, eight-arm curves, free-symbol distribution, "
 "causal head interventions, six repair conditions, Extended independent replicates "
 "and historical G3 frozen test. **Nothing in this folder should be interpreted "
 "as a newly performed independent final test.**\n\n"
 "See `EVIDENCE_MANIFEST.json` for source run ID, relative filename, size and SHA-256. "
 "Binary model checkpoints stay in the original Actions artifacts and may expire. "
 "The original G3 protected test has been used and must not be used for new G10 selection.\n")
 print("EVIDENCE_ARCHIVE_COMPLETE="+json.dumps({"files":len(records),"bytes":used}))
if __name__=="__main__":main()
