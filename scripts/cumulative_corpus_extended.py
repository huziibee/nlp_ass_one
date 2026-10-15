"""Genuine cumulative *example* corpus with fixed real anchor and historic samples.

At each G1-G10: permanently retain a fixed original clean-example pool plus
every previous generation's full sampled training pair (real input context,
model-generated 3-token target). Newly generated samples are ADDED rather
than replacing earlier samples; a same-size clean-history arm controls for
growing corpus size. Model gradient exposure is matched per generation.
This does NOT recursively synthesize new input context prefixes, and the
100k/g training budget is NOT equivalent to primary 300k/g results.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'external'/'icl-dynamics'))
import numpy as np
import jax,jax.numpy as jnp
import equinox as eqx
from scripts.extended_g0 import load_config,feature_prototypes
from scripts.extended_recursive import load_verified_g0
from src.extended_omniglot import ExtendedOmniglot,make_episodes,clean_continuation
from src.extended_learner import (build_optimizer,extended_train_step,
                                  evaluate_extended,autoregressive_sample)
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--base-root",type=Path,required=True)
 p.add_argument("--g0-root",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 p.add_argument("--steps",type=int,default=3125)
 p.add_argument("--pool-block",type=int,default=2048)
 p.add_argument("--seed",type=int,default=7712)
 a=p.parse_args()
 if a.steps!=3125 or a.pool_block!=2048:raise ValueError("Frozen 100k/g budget, 2048 newly sampled episodes")
 cfg=load_config(a.base_root)
 g0,meta=load_verified_g0(cfg,a.g0_root,require_gate=True)
 model_seed=int(meta["seed"])
 prototypes=feature_prototypes(cfg)
 samples=make_episodes(cfg,role="train",batch_size=a.pool_block)
 val={role:make_episodes(cfg,role=role,batch_size=256)(
      jax.random.PRNGKey(a.seed+900+i)) for i,role in enumerate(
      ("val_relabel","val_exemplar","val_classes"))}
 original_batch=samples(jax.random.PRNGKey(a.seed+11))
 original_tokens=clean_continuation(original_batch,key=jax.random.PRNGKey(a.seed+12))["tokens"]
 # Batch blocks are actual fixed source Omniglot feature episodes, never test.
 history_batches=[original_batch]
 clean_target_blocks=[original_tokens]
 recursive_target_blocks=[original_tokens]
 predecessor=g0
 summary={"study":"explicit growing historical original+synthetic training examples",
  "actual_pool_increases_per_generation":True,
  "historical_examples_replaced":False,
  "real_context_prefixes_always_original":True,
  "generated_input_prefixes":False,
  "source":"pinned original Singh Omniglot features",
  "source_g0_selected_validation_only":True,
  "clean_history_control_matched_pool_sizes":True,
  "initialization_seed":model_seed,"data_seed":a.seed,
  "class_split":[50,1473,100],
  "fixed_anchor_real_examples":a.pool_block,
  "new_block_per_generation":a.pool_block,
  "per_generation_gradient_exposures":a.steps*32,
  "not_comparable_with_primary_extended_300k":True,
  "final_test_evaluated":False,"arms":{"clean_history":[],"cumulative_recursive":[]}}
 a.out.mkdir(parents=True,exist_ok=True)
 for g in range(1,11):
  new_batch=samples(jax.random.PRNGKey(a.seed+g*90007))
  new_clean=clean_continuation(new_batch,key=jax.random.PRNGKey(a.seed+g*90007+1))["tokens"]
  new_synth=autoregressive_sample(
     predecessor,new_batch,jax.random.PRNGKey(a.seed+g*90007+2),1.,prototypes)
  invalid=(~jnp.any(new_synth[:,1,None]==new_batch["class_ids"][:,:2],axis=-1))
  new_invalid=float(jnp.mean(invalid))
  history_batches.append(new_batch)
  clean_target_blocks.append(new_clean)
  recursive_target_blocks.append(new_synth)
  common={k:jnp.concatenate([item[k] for item in history_batches]) for k in original_batch}
  pools={"clean_history":jnp.concatenate(clean_target_blocks),
         "cumulative_recursive":jnp.concatenate(recursive_target_blocks)}
  n=int(common["labels"].shape[0]);assert n==(g+1)*a.pool_block
  for arm in ("clean_history","cumulative_recursive"):
   model=ExtendedOmniglot(config=cfg,seed=model_seed)
   optimizer,opt_state=build_optimizer(model,.0003)
   targets=pools[arm]
   assert targets.shape==(n,3)
   start=time.monotonic()
   for step in range(a.steps):
    k=jax.random.PRNGKey(a.seed+g*103111+step)
    k_idx,k_train=jax.random.split(k)
    indices=jax.random.randint(k_idx,(32,),0,n)
    minibatch={k:v[indices] for k,v in common.items()}
    y=targets[indices]
    q2feat=prototypes[y[:,1]]
    loss,model,opt_state=extended_train_step(
      model,optimizer,opt_state,minibatch,y,q2feat,k_train)
   metrics=evaluate_extended(
       model,val,feature_lookup=prototypes,seed=a.seed+70000,temperature=1.)
   folder=a.out/arm/f"g{g:02d}";folder.mkdir(parents=True,exist_ok=True)
   eqx.tree_serialise_leaves(str(folder/"model.eqx"),model)
   row={"generation":g,"arm":arm,"historical_pool_size":n,
        "real_anchor_fraction":a.pool_block/n,"stored_original_examples":a.pool_block,
        "stored_prior_synthetic_blocks":g if arm=="cumulative_recursive" else 0,
        "new_source_generator_generation":g-1 if arm=="cumulative_recursive" else None,
        "new_synthetic_off_context_fraction":new_invalid if arm=="cumulative_recursive" else None,
        "training_exposures":a.steps*32,
        "last_batch_train_loss":float(loss),"validation":metrics,
        "elapsed_seconds":time.monotonic()-start,
        "final_test_evaluated":False}
   (folder/"metrics.json").write_text(json.dumps(row,indent=2))
   summary["arms"][arm].append(row)
   (a.out/"summary.json").write_text(json.dumps(summary,indent=2))
   print("CUMULATIVE_CORPUS_GENERATION="+json.dumps({
       "arm":arm,"g":g,"pool":n,
       "val":metrics["val_relabel"]}),flush=True)
   if arm=="cumulative_recursive":predecessor=model
 assert all(len(v)==10 for v in summary["arms"].values())
 print("CUMULATIVE_REAL_PLUS_SYNTHETIC_COMPLETE="+json.dumps({
   "n_generations":10,"g10":{
    arm:records[-1]["validation"]["val_relabel"] for arm,records in summary["arms"].items()},
   "final_test_evaluated":False}),flush=True)
if __name__=="__main__":main()
