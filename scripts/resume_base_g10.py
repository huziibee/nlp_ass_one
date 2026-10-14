"""Resume the verified original-Omniglot Base study from G3 to G10.

Does NOT restart G1..G3 or open the sealed test. Requires the original
artifact with a completed G3 checkpoint, metrics and summary for ONE arm.

The output summary combines immutable G0-G3 evidence with new G4-G10
observations. Each fresh descendant has the same sampled task, init seed
policy, and train/evaluation episode schedule as scripts/base_recursive.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import equinox as eqx
from base_recursive import (
    ARMS, create_fresh_model, get_validation_data,
    load_original_checkpoint, make_original_train_sampler,
    train_arm_generation, validation_metrics,
)

def load_g3(base_config, folder:Path, arm:str, *,seed:int, steps:int,
            batch_size:int,temperature:float):
    checkpoint=folder/arm/"g03"/"model.eqx"
    metrics_path=folder/arm/"g03"/"metrics.json"
    prior_summary=folder/"summary.json"
    if not checkpoint.is_file() or not metrics_path.is_file() or not prior_summary.is_file():
        raise FileNotFoundError("Expected completed G3 model.eqx, metrics.json, summary.json")
    meta=json.loads(metrics_path.read_text())
    previous=json.loads(prior_summary.read_text())
    if meta["arm"]!=arm or meta["generation"]!=3 or meta["final_test_evaluated"]:
        raise AssertionError("Invalid G3 arm/checkpoint origin")
    if meta["optimizer_steps"]!=steps or meta["batch_size"]!=batch_size or meta["data_seed"]!=seed:
        raise AssertionError("Cannot change G3 sequence exposure or seed")
    if meta["temperature"]!=temperature:
        raise AssertionError("Cannot change original G3 sampling temperature")
    if meta["initialisation_seed"]!=seed+3*17:
        raise AssertionError("G3 expected initialisation seed mismatch")
    if meta["class_split"]!=[50,1473,100] or meta["label_pair_split"]!=[8,2,0]:
        raise AssertionError("G3 does not use original paper data partition")
    if previous["final_test_evaluated"] or previous["training_prefix_generated"]:
        raise AssertionError("Previous G3 evidence unsafe")
    if len(previous["arms"])!=1 or len(previous["arms"].get(arm,[]))!=3:
        raise AssertionError("Expected G1-G3 history for requested condition")
    template,_,_=create_fresh_model(base_config,seed+3*17)
    model=eqx.tree_deserialise_leaves(str(checkpoint),template)
    return model,previous

def run(*,original_artifact:Path,resume_root:Path,out:Path,
        arm:str, start:int,end:int,steps:int,batch_size:int,
        eval_every:int,eval_count:int,seed:int,temperature:float):
    if arm not in ARMS:
        raise ValueError("Unknown Base condition")
    if start!=4 or not 4<=end<=20:
        raise ValueError("Resume source is G3; start must be G4")
    if steps<1 or batch_size<2 or batch_size>32 or batch_size%2 or eval_every<1 or eval_every>steps:
        raise ValueError("Bad training parameters")
    if eval_count<32 or temperature<=0:
        raise ValueError("Bad validation parameters")
    if out.resolve()==resume_root.resolve():
        raise ValueError("Never overwrite original G1-G3 artifact")
    g0,config=load_original_checkpoint(original_artifact)
    predecessor,previous=load_g3(config,resume_root,arm,seed=seed,
                                 steps=steps,batch_size=batch_size,
                                 temperature=temperature)
    sampler=make_original_train_sampler(config,batch_size)
    validation=get_validation_data(config,eval_count,seed+501)
    out.mkdir(parents=True,exist_ok=True)
    evidence={
        "g0_validation":previous["g0_validation"],
        "arms":{arm:list(previous["arms"][arm])},
        "resumed_from_generation":3,
        "new_trained_generations":list(range(start,end+1)),
        "resume_checkpoint":str(resume_root/arm/"g03"/"model.eqx"),
        "no_retraining_of_g1_to_g3":True,
        "original_split":[50,1473,100],
        "training_prefix_generated":False,
        "final_test_evaluated":False,
        "validation_evaluation_tracks":["val_relabel","val_classes"],
    }
    for gen in range(start,end+1):
        predecessor,meta=train_arm_generation(
            arm=arm,generation=gen,predecessor=predecessor,g0=g0,
            config=config,sampler=sampler,val_data=validation,
            steps=steps,batch_size=batch_size,eval_every=eval_every,
            seed=seed,out=out,temperature=temperature)
        evidence["arms"][arm].append({
            "generation":gen,
            "last_validation":meta["history"][-1]["validation"],
            "real_target_count":meta["real_target_count"],
            "synthetic_target_count":meta["synthetic_target_count"],
        })
        (out/"summary.json").write_text(json.dumps(evidence,indent=2))
        print("BASE_RESUMED_GENERATION="+json.dumps({
            "arm":arm,"generation":gen,"validation":meta["history"][-1]["validation"],
            "final_test_evaluated":False}),flush=True)
    assert len(evidence["arms"][arm])==end
    print("BASE_G10_EXTENSION_COMPLETE="+json.dumps({
        "arm":arm,"final_generation":end,"real_context_prefixes":True,
        "final_test_evaluated":False}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-artifact",type=Path,
                   default=ROOT/"outputs/base_full_1m")
    p.add_argument("--resume-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--start",type=int,default=4)
    p.add_argument("--end",type=int,default=10)
    p.add_argument("--steps",type=int,default=31250)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--eval-every",type=int,default=3125)
    p.add_argument("--eval-count",type=int,default=512)
    p.add_argument("--seed",type=int,default=34001)
    p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--preflight-only",action="store_true",
                   help="Validate source G3 identity and reproduce its recorded validation metrics; do not train")
    a=p.parse_args()
    if a.preflight_only:
        if a.start!=4:
            raise ValueError("Only G3 to G4 continuation supported")
        g0,cfg=load_original_checkpoint(a.original_artifact)
        previous_model,prior=load_g3(cfg,a.resume_root,a.arm,
                                     seed=a.seed,steps=a.steps,
                                     batch_size=a.batch_size,
                                     temperature=a.temperature)
        original=prior["arms"][a.arm][-1]["last_validation"]
        validation=get_validation_data(cfg,a.eval_count,a.seed+501)
        recovered=validation_metrics(previous_model,cfg,validation,
                                     seed=a.seed+100_000+a.steps-1)
        for track in ("val_relabel","val_classes"):
            for metric in ("accuracy","loss"):
                if abs(original[track][metric]-recovered[track][metric])>0.005:
                    raise AssertionError(
                        "Loaded G3 checkpoint does not reproduce its logged "+
                        track+"/"+metric)
        print("BASE_G3_RESUME_PREFLIGHT_PASSED="+json.dumps({
            "arm":a.arm,"original":original,"recovered":recovered,
            "opened_final_test":False}),flush=True)
        return
    run(original_artifact=a.original_artifact,resume_root=a.resume_root,
        out=a.out,arm=a.arm,start=a.start,end=a.end,steps=a.steps,
        batch_size=a.batch_size,eval_every=a.eval_every,
        eval_count=a.eval_count,seed=a.seed,temperature=a.temperature)
if __name__=="__main__":
    main()
