"""G0 preselected L2 joint-head ablations at G0/G3/G6/G10.

Tests if the G0 induction-head effect survives via other heads in clean or
recursive descendants. Same validation episodes/seeds/intervention width
within each condition; no model is fine-tuned and no final test is used.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'external'/'icl-dynamics'))
import jax
import numpy as np
from scripts.extended_g0 import load_config
from scripts.compare_extended_circuits_g10 import (
    load_model,batch_aux,accuracy_scores,attention_metrics)
from src.extended_omniglot import make_episodes,clean_continuation
import equinox as eqx

def remove_heads(model,heads):
    weight=model.base.transformer.blocks[1].attn.proj.weight
    width=weight.shape[1]//8
    for h in heads:
        weight=weight.at[:,h*width:(h+1)*width].set(0)
    return eqx.tree_at(lambda m:m.base.transformer.blocks[1].attn.proj.weight,model,weight)

def perform(args):
    cfg=load_config(args.base_root)
    g0,_=load_model(cfg,args.g0_root,gen=0)
    b=make_episodes(cfg,role="train",batch_size=256)(jax.random.PRNGKey(4854))
    oracle=clean_continuation(b,key=jax.random.PRNGKey(4855))
    trace=batch_aux(g0,b,oracle,key=4856)
    margins=attention_metrics(b,oracle,trace[3],trace[4])["l2_query_first_number_matching_margin_by_head"]
    ranked=np.argsort(-np.asarray(margins)).tolist()
    first,second,third=ranked[:3]
    if first!=1:
        raise AssertionError(f"Unexpected G0 top head {first}; refuse post-hoc selection")
    interventions={"natural":[],"head1":[first],"runner_up":[second],
                   "pair_top2":[first,second],"pair_top3":[first,second,third],
                   "head1_plus_third":[first,third]}
    result={"experiment":"original Extended causal L2 multi-head intervention",
      "selected_using":"train-only G0 matching margins, common across clean/recursive",
      "top_heads":ranked,"train_g0_margins":margins,"arms":[args.arm],
      "retrained_after_ablation":False,"final_test_evaluated":False,
      "intervention_on_model_only":True,"generations":[0,3,6,10],
      "validation":{"val_relabel":{},"val_exemplar":{},"val_classes":{}}}
    for gen in (0,3,6,10):
        model=(g0 if gen==0 else load_model(cfg,
          args.g3_root/args.arm/"g03" if gen==3 else args.g10_root/args.arm/f"g{gen:02d}",
          gen=gen)[0])
        for i,role in enumerate(result["validation"]):
            batch=make_episodes(cfg,role=role,batch_size=args.n)(
                  jax.random.PRNGKey(60200+i*101))
            target=clean_continuation(batch,key=jax.random.PRNGKey(60300+i*101))
            source_key=60400+i*101
            scores={}
            for name,heads in interventions.items():
                changed=remove_heads(model,heads) if heads else model
                t=batch_aux(changed,batch,target,key=source_key)
                scores[name]=accuracy_scores(batch,target,t[:3])
            result["validation"][role][str(gen)]=scores
            print("JOINT_ABLATION="+json.dumps({"arm":args.arm,"role":role,
                 "gen":gen,"scores":scores}),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--base-root",type=Path,required=True)
    p.add_argument("--g0-root",type=Path,required=True)
    p.add_argument("--g3-root",type=Path,required=True)
    p.add_argument("--g10-root",type=Path,required=True)
    p.add_argument("--arm",choices=("clean_only","recursive_replacement"),required=True)
    p.add_argument("--n",type=int,default=128)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if not 32<=a.n<=256:raise ValueError("Batch exceeds bounded resources")
    perform(a)
