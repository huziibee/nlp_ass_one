"""Continue genuine Omniglot Extended autoregressive G3 to G10.

Input source is an existing, successful, fixed-initialization G3 artifact.
This performs validation-only resumption; it never accesses the original
100 final-test classes, never synthesizes the input-prefix tokens, and
never silently tunes the previously selected 300k/G0/3e-4 hyperparameters.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"external"/"icl-dynamics"))

import equinox as eqx
import jax

from scripts.extended_g0 import load_config,feature_prototypes
from scripts.extended_recursive import (
    ARMS, load_verified_g0,train_generation
)
from src.extended_omniglot import ExtendedOmniglot,make_episodes
from src.extended_learner import evaluate_extended

def frozen_validation(config,*,n,seed):
    return {
        name:make_episodes(config,role=name,batch_size=n)(
            jax.random.PRNGKey(seed+911+i))
        for i,name in enumerate(("val_relabel","val_exemplar","val_classes"))
    }

def verified_previous_model(
    config,resume_root:Path,*,arm:str,
    steps:int,batch_size:int,seed:int,init_seed:int,
    lr:float,temperature:float,
):
    path=resume_root/arm/"g03"
    ckpt=path/"model.eqx"
    metrics_path=path/"metrics.json"
    history_path=resume_root/"summary.json"
    if any(not p.is_file() for p in (ckpt,metrics_path,history_path)):
        raise FileNotFoundError("Missing complete G3 Extended checkpoint, metrics or summary")
    m=json.loads(metrics_path.read_text())
    previous=json.loads(history_path.read_text())
    if m.get("generation")!=3 or m.get("condition")!=arm:
        raise AssertionError("G3 checkpoint provenance/arm mismatch")
    if m.get("final_test_evaluated") or previous.get("final_test_evaluated"):
        raise AssertionError("This study requires an unopened final test")
    if m["training_exposures"]!=steps*batch_size or m["batch_size"]!=batch_size:
        raise AssertionError("G3 training exposure mismatch")
    if m["data_seed"]!=seed or m["init_seed"]!=init_seed:
        raise AssertionError("G3 training or model seed mismatch")
    if abs(m["lr"]-lr)>1e-12 or m["temperature"]!=temperature:
        raise AssertionError("G3 hyperparameters mismatch")
    if m["training_class_split"]!=[50,1473,100] or m["label_pair_split"]!=[8,2,0]:
        raise AssertionError("Original class/label-pair split mismatch")
    if m["recursively_generated_input_prefix"] or m["growing_history_corpus_accumulation"]:
        raise AssertionError("Mislabelled full-context or accumulation condition")
    if len(previous.get("arms",{}))!=1 or len(previous["arms"].get(arm,[]))!=3:
        raise AssertionError("Expected exact G1-G3 study for one arm")
    if any(x["generation"]!=i for i,x in enumerate(previous["arms"][arm],1)):
        raise AssertionError("Previous G1-G3 generations missing or unordered")
    source=(
        None if arm=="clean_only" else 0 if arm=="frozen_g0" else 2
    )
    if m["synthetic_source_generation"]!=source:
        raise AssertionError("Wrong predecessor model at G3")
    template=ExtendedOmniglot(config=config,seed=init_seed)
    model=eqx.tree_deserialise_leaves(str(ckpt),template)
    return model,previous

def preflight(*,config,model,previous,arm,val,prototypes,seed,temperature):
    recovered=evaluate_extended(
        model,val,feature_lookup=prototypes,
        seed=seed+170_000,temperature=temperature)
    recorded=previous["arms"][arm][-1]["last_validation"]
    checks=[
        "first_label_accuracy","teacher_forced_second_label_accuracy",
        "first_label_loss","teacher_forced_second_label_loss",
    ]
    for role in ("val_relabel","val_exemplar","val_classes"):
        for metric in checks:
            if abs(recovered[role][metric]-recorded[role][metric])>0.005:
                raise AssertionError(
                    "Resumed G3 does not reproduce validation "+
                    role+"/"+metric)
        if role!="val_classes":
            for metric in ("next_symbol_valid_probability_mass",
                           "free_next_symbol_context_validity"):
                if abs(recovered[role][metric]-recorded[role][metric])>0.005:
                    raise AssertionError(
                        "Resumed G3 next-symbol behavior mismatch "+role+"/"+metric)
    print("EXTENDED_G3_RESUME_PREFLIGHT_PASSED="+json.dumps({
        "arm":arm,"recorded_relabel":recorded["val_relabel"],
        "recovered_relabel":recovered["val_relabel"],
        "final_test_evaluated":False}),flush=True)
    return recovered

def run(*,base_artifact:Path,g0_artifact:Path,resume_root:Path,out:Path,
        arm:str,start:int,end:int,steps:int,batch_size:int,
        eval_every:int,eval_count:int,seed:int,init_seed:int,
        lr:float,temperature:float,preflight_only:bool):
    if arm not in ARMS or start!=4 or not 4<=end<=20:
        raise ValueError("Exactly G3->G4 to at most G20 continuation")
    if not (steps>=1 and 2<=batch_size<=32 and batch_size%2==0):
        raise ValueError("Bad exposure configuration")
    if not (1<=eval_every<=steps and 32<=eval_count<=1024 and temperature>0):
        raise ValueError("Bad validation configuration")
    if out.resolve()==resume_root.resolve():
        raise ValueError("Cannot overwrite original G1-G3 evidence")
    cfg=load_config(base_artifact)
    g0,g0_meta=load_verified_g0(cfg,g0_artifact,require_gate=True)
    if init_seed!=int(g0_meta["seed"]):
        raise AssertionError("G3 controlled initialization must equal selected G0 seed")
    previous_model,previous=verified_previous_model(
        cfg,resume_root,arm=arm,steps=steps,batch_size=batch_size,
        seed=seed,init_seed=init_seed,lr=lr,temperature=temperature)
    prototypes=feature_prototypes(cfg)
    validation=frozen_validation(cfg,n=eval_count,seed=seed)
    preflight(config=cfg,model=previous_model,previous=previous,arm=arm,
              val=validation,prototypes=prototypes,seed=seed,
              temperature=temperature)
    if preflight_only:
        return
    train_sampler=make_episodes(cfg,role="train",batch_size=batch_size)
    out.mkdir(parents=True,exist_ok=True)
    lineage={
        "g0_validation":previous["g0_validation"],
        "arms":{arm:list(previous["arms"][arm])},
        "original_feature_dataset":"pinned Singh Omniglot ResNet18",
        "class_split":[50,1473,100],
        "label_pair_split":[8,2,0],
        "resumed_from_generation":3,
        "no_retraining_of_g1_g2_g3":True,
        "source_artifact":str(resume_root/arm/"g03"/"model.eqx"),
        "new_generations":list(range(start,end+1)),
        "final_test_evaluated":False,
        "recursively_generated_input_prefix":False,
        "growing_history_corpus_accumulation":False,
    }
    parent=previous_model
    for generation in range(start,end+1):
        parent,meta=train_generation(
            arm=arm,generation=generation,predecessor=parent,g0=g0,
            config=cfg,train_sampler=train_sampler,val=validation,
            prototypes=prototypes,steps=steps,batch_size=batch_size,
            eval_every=eval_every,seed=seed,lr=lr,
            temperature=temperature,out=out,init_seed=init_seed)
        lineage["arms"][arm].append({
            "generation":generation,
            "last_validation":meta["history"][-1]["validation"],
            "num_real_targets":meta["num_real_targets"],
            "num_synthetic_targets":meta["num_synthetic_targets"],
        })
        (out/"summary.json").write_text(json.dumps(lineage,indent=2))
        print("EXTENDED_G10_CONTINUED="+json.dumps({
            "arm":arm,"generation":generation,
            "validation":meta["history"][-1]["validation"],
            "final_test_evaluated":False
        }),flush=True)
    assert len(lineage["arms"][arm])==end
    print("EXTENDED_G10_COMPLETE="+json.dumps({
        "arm":arm,"generations":end,"sealed_test_scored":False}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-artifact",type=Path,default=ROOT/"outputs/base_full_1m")
    p.add_argument("--extended-g0-artifact",type=Path,default=ROOT/"outputs/extended_g0_full")
    p.add_argument("--resume-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--start",type=int,default=4)
    p.add_argument("--end",type=int,default=10)
    p.add_argument("--steps",type=int,default=9375)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--eval-every",type=int,default=3125)
    p.add_argument("--eval-count",type=int,default=512)
    p.add_argument("--seed",type=int,default=6001)
    p.add_argument("--initialization-seed",type=int,default=4054)
    p.add_argument("--learning-rate",type=float,default=0.0003)
    p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--preflight-only",action="store_true")
    a=p.parse_args()
    run(base_artifact=a.base_artifact,g0_artifact=a.extended_g0_artifact,
        resume_root=a.resume_root,out=a.out,arm=a.arm,
        start=a.start,end=a.end,steps=a.steps,batch_size=a.batch_size,
        eval_every=a.eval_every,eval_count=a.eval_count,
        seed=a.seed,init_seed=a.initialization_seed,lr=a.learning_rate,
        temperature=a.temperature,preflight_only=a.preflight_only)
if __name__=="__main__":
    main()
