"""Prespecified graded off-context (q2, corresponding-label) target repair.

Reuses original fixed Omniglot Extended G9 teacher and train/validation episodes.
Fraction is applied only to OFF-CONTEXT q2 rows, so fraction 0 is exactly
the raw source and fraction 1 is the existing joint invalid-symbol repair.
We run intermediate 25/50/75% with the same G10 model initialization,
optimization count, and episode ordering. Never opens the final test.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"external"/"icl-dynamics"))
import jax,jax.numpy as jnp
import equinox as eqx
import numpy as np
from scripts.extended_g0 import load_config,feature_prototypes
from scripts.extended_recursive import load_verified_g0,training_targets
from scripts.extended_g10_repair import load_previous_g9
from src.extended_omniglot import ExtendedOmniglot,make_episodes,clean_continuation
from src.extended_learner import build_optimizer,extended_train_step,evaluate_extended

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--base-root",type=Path,required=True)
 p.add_argument("--g0-root",type=Path,required=True)
 p.add_argument("--g10-root",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 p.add_argument("--fraction",type=float,required=True)
 a=p.parse_args()
 if a.fraction not in (.25,.5,.75):raise ValueError("Intermediate, prespecified fractions only")
 seed,init_seed=6001,4054;steps=9375;temperature=1.;lr=.0003
 cfg=load_config(a.base_root)
 g0,g0meta=load_verified_g0(cfg,a.g0_root,require_gate=True)
 assert int(g0meta["seed"])==init_seed
 source=load_previous_g9(cfg,a.g10_root,init_seed=init_seed,data_seed=seed)
 reference=json.loads((a.g10_root/"recursive_replacement"/"g10"/"metrics.json").read_text())
 assert reference["final_test_evaluated"] is False and reference["training_exposures"]==300000
 sampler=make_episodes(cfg,role="train",batch_size=32)
 validation={name:make_episodes(cfg,role=name,batch_size=512)(
     jax.random.PRNGKey(seed+911+i)) for i,name in enumerate(
     ("val_relabel","val_exemplar","val_classes"))}
 prototypes=feature_prototypes(cfg)
 model=ExtendedOmniglot(config=cfg,seed=init_seed)
 optimizer,state=build_optimizer(model,lr)
 totals={"invalid_q2_seen":0,"joint_pair_repaired":0,"target_rows_changed":0}
 history=[]
 t=time.monotonic()
 for step in range(steps):
  key=jax.random.PRNGKey(seed+10*103_111+step)
  k_episode,k_clean,k_synth,k_train=jax.random.split(key,4)
  batch=sampler(k_episode)
  clean=clean_continuation(batch,key=k_clean)
  raw,_=training_targets("recursive_replacement",batch,clean,source,prototypes,
                         data_key=k_synth,temperature=temperature)
  invalid=~jnp.any(raw[:,1,None]==batch["class_ids"][:,:2],axis=-1)
  sample=(jax.random.uniform(jax.random.fold_in(k_synth,91813),(32,))<a.fraction)
  chosen=invalid & sample
  target=jnp.stack((
    raw[:,0],
    jnp.where(chosen,clean["tokens"][:,1],raw[:,1]),
    jnp.where(chosen,clean["tokens"][:,2],raw[:,2])
  ),axis=-1)
  totals["invalid_q2_seen"]+=int(np.asarray(invalid).sum())
  totals["joint_pair_repaired"]+=int(np.asarray(chosen).sum())
  totals["target_rows_changed"]+=int(np.asarray(jnp.any(target!=raw,axis=-1)).sum())
  q2_features=prototypes[target[:,1]]
  loss,model,state=extended_train_step(
     model,optimizer,state,batch,target,q2_features,k_train)
  if (step+1)%3125==0 or step+1==steps:
   metrics=evaluate_extended(
     model,validation,feature_lookup=prototypes,
     seed=seed+170_000,temperature=temperature)
   history.append({"step":step+1,"train_loss":float(loss),"exposures":(step+1)*32,
                   "validation":metrics})
   print("FRACTION_REPAIR="+json.dumps({"fraction":a.fraction,
         "step":step+1,"validity":metrics["val_relabel"]["free_next_symbol_context_validity"]}),flush=True)
 a.out.mkdir(parents=True,exist_ok=True)
 eqx.tree_serialise_leaves(str(a.out/"model.eqx"),model)
 metadata={"type":"graded G9 synthetic invalid-(q2,label) correction",
     "correction_fraction_of_invalid_generated_examples":a.fraction,
     "fraction_applied":"bernoulli independently on off-context generated q2 only",
     "model_seed":init_seed,"data_seed":seed,"generation":10,
     "training_exposures":300000,"optimizer_steps":steps,
     "source_g9_original":True,"real_prefix":True,"final_test_evaluated":False,
     "intervention_modifies_two_target_tokens_together":True,
     "count_data":totals,"history":history,"elapsed_seconds":time.monotonic()-t}
 (a.out/"metrics.json").write_text(json.dumps(metadata,indent=2))
 print("FRACTION_REPAIR_FINISHED="+json.dumps({"fraction":a.fraction,
     "val_relabel":history[-1]["validation"]["val_relabel"],
     "counts":totals}),flush=True)

if __name__=="__main__":main()
