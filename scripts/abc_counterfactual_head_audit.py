"""Distinguish attention-head persistence, redistribution and contextual shortcuts.

Source-aligned Omniglot Extended G0/G10. Prespecified original G0 heads
1/0/7. G10 alternative heads are selected on TRAIN episodes only, then
tested on identical held-out validation episodes for clean and recursion.

Counterfactuals:
- Swap contextual labels with identical symbol features and unchanged query.
- Swap support order while preserving each support's label.
- Flip query to the opposite in-context support symbol.
All counterfactual oracles are recomputed. The final test is NEVER accessed.

Interpretation: sensitivity supports contextual algorithmic behavior but does
not uniquely identify attention pathways or rule out all alternative shortcuts.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"external"/"icl-dynamics"))
import jax
import jax.numpy as jnp
import numpy as np
from scripts.extended_g0 import load_config
from scripts.compare_extended_circuits_g10 import (
    load_model,batch_aux,accuracy_scores,attention_metrics)
from scripts.extended_joint_head_ablations import remove_heads
from src.extended_omniglot import make_episodes,clean_continuation

G0_SELECTED=(1,0,7)
DEVELOPMENT_ROLES=("val_relabel","val_exemplar","val_classes")
BASE_EVAL_SEED=82401

def safe_metrics(model,batch,targets,seed):
    trace=batch_aux(model,batch,targets,key=seed)
    scores=accuracy_scores(batch,targets,trace[:3])
    attention=attention_metrics(batch,targets,trace[3],trace[4])
    return trace,scores,attention

def label_swap(batch):
    labs=batch["labels"]
    new=jnp.stack((labs[:,1],labs[:,0],
        jnp.where(batch["class_ids"][:,2]==batch["class_ids"][:,0],
                  labs[:,1],labs[:,0])),axis=1)
    assert bool(jnp.all(new[:,2]!=labs[:,2]))
    return {**batch,"labels":new}

def support_swap(batch):
    out={}
    for k,v in batch.items():
        out[k]=v[:,jnp.array([1,0,2]),...] if v.ndim>=2 else v
    assert bool(jnp.all(out["labels"][:,2]==batch["labels"][:,2]))
    return out

def query_flip(batch):
    ids=batch["class_ids"]
    labs=batch["labels"]
    ex=batch["examples"]
    indices=jnp.where(ids[:,2]==ids[:,0],1,0)
    rr=jnp.arange(len(indices))
    new_example=ex[rr,indices,:]
    new_id=ids[rr,indices]
    new_lab=labs[rr,indices]
    changed={**batch,
             "examples":ex.at[:,2,:].set(new_example),
             "class_ids":ids.at[:,2].set(new_id),
             "labels":labs.at[:,2].set(new_lab),
             "exemplar_ids":batch["exemplar_ids"].at[:,2].set(
                  batch["exemplar_ids"][rr,indices])}
    assert bool(jnp.all(changed["labels"][:,2]!=batch["labels"][:,2]))
    return changed

def predict_first(trace):
    return np.asarray(trace[0]).argmax(axis=-1)

def run(args):
    cfg=load_config(args.base_root)
    g0,g0_meta=load_model(cfg,args.g0_root,gen=0)
    g10,g10_meta=load_model(cfg,args.g10_root/args.arm/"g10",gen=10)
    assert g0_meta["final_test_evaluated"] is False
    assert g10_meta["final_test_evaluated"] is False
    models={"g0":g0,"g10":g10}
    train=make_episodes(cfg,role="train",batch_size=128)(jax.random.PRNGKey(71333))
    train_c=clean_continuation(train,key=jax.random.PRNGKey(71334))
    selection={}
    for generation,model in models.items():
        natural,baseline,mech=safe_metrics(model,train,train_c,71335)
        margins=mech["l2_query_first_number_matching_margin_by_head"]
        head_deltas=[]
        for h in range(8):
            _,v,_=safe_metrics(remove_heads(model,[h]),train,train_c,71335)
            head_deltas.append(float(baseline["first_label_accuracy"]-
                                     v["first_label_accuracy"]))
        outside=[h for h in range(8) if h not in G0_SELECTED]
        best_new_by_causal=sorted(outside,key=lambda h:(-head_deltas[h],h))[0]
        best_new_by_margin=sorted(outside,key=lambda h:(-margins[h],h))[0]
        selection[generation]={
           "training_g0_original_heads":list(G0_SELECTED),
           "first_label_head_ablation_effect_train_pp":[100*x for x in head_deltas],
           "matching_margins_train":margins,
           "best_outside_g0_heads_by_train_causal":best_new_by_causal,
           "best_outside_g0_heads_by_train_margin":best_new_by_margin}
    original_heads=list(G0_SELECTED)
    alternate=selection["g10"]["best_outside_g0_heads_by_train_causal"]
    margin_alt=selection["g10"]["best_outside_g0_heads_by_train_margin"]
    remaining=sorted(set(range(8))-set(original_heads))
    interventions={
       "natural":[],
       "g0_head1":[1],
       "g0_top3":original_heads,
       "remaining_five":remaining,
       "all_eight":list(range(8)),
       "new_train_causal_head":[alternate],
       "new_train_matching_head":[margin_alt],
       "original_plus_new_causal":sorted(set(original_heads+[alternate])),
    }
    for h in range(8):interventions[f"single_head_{h}"]=[h]
    output={"hypotheses":["A_original_matching_stops","B_compensation_or_redistribution",
                          "C_alternative_behavioral_shortcut"],
            "arm":args.arm,"source_run_type":"real original Omniglot",
            "protected_final_test_evaluated":False,
            "selection_train_examples":128,
            "validation_episodes_per_role":args.n,
            "selection":selection,
            "g0_selected_heads":original_heads,
            "g10_alternative_head_according_to_own_training_data":alternate,
            "g10_other_attention_margin_candidate":margin_alt,
            "roles":{},"scientific_guardrails":[
             "Original G0 head selection is kept fixed for all models",
             "New G10 candidate is selected only on train and evaluated independently",
             "All comparisons within an arm use identical validation episodes",
             "Label-swap and support-order counterfactuals remain legal contexts",
             "Query-flip uses an identical support exemplar and is easier than ordinary held-out query",
             "Head knockout tests component necessity, not cross-model neural alignment",
             "Covers just G0 and primary G10: one model-seed pairing",
             "Context-following cannot disprove every non-attention algorithmic shortcut"]}
    for i,role in enumerate(DEVELOPMENT_ROLES):
        batch=make_episodes(cfg,role=role,batch_size=args.n)(
                 jax.random.PRNGKey(BASE_EVAL_SEED+i*101))
        target=clean_continuation(batch,key=jax.random.PRNGKey(BASE_EVAL_SEED+i*101+1))
        result={}
        for gen,model in models.items():
            t0=time.monotonic()
            unablated,base_scores,base_mech=safe_metrics(model,batch,target,
                       BASE_EVAL_SEED+i*101+2)
            base_preds=predict_first(unablated)
            perturbations={}
            for name,modify in [("swap_support_labels",label_swap),
                                ("swap_support_order",support_swap),
                                ("flip_query_to_other_support",query_flip)]:
                changed=modify(batch)
                newtarget=clean_continuation(changed,
                           key=jax.random.PRNGKey(BASE_EVAL_SEED+i*101+1))
                trace,scores,_=safe_metrics(model,changed,newtarget,
                           BASE_EVAL_SEED+i*101+2)
                pred=predict_first(trace)
                ref=np.asarray(changed["labels"])[:,-1]
                old=np.asarray(batch["labels"])[:,-1]
                correct_orig=base_preds==old
                perturbations[name]={
                     "accuracy_against_new_oracle":float(np.mean(pred==ref)),
                     "fraction_prediction_changed":float(np.mean(pred!=base_preds)),
                     "on_orig_correct_fraction_follows_changed_oracle":
                         float(np.mean((pred==ref)[correct_orig])) if correct_orig.any() else None,
                     "counterfactual_changes_correct_target":bool(np.any(ref!=old)),
                     "n":len(ref),
                     "notes":("Query replaced with exact opposite support exemplar; simpler query"
                      if name=="flip_query_to_other_support" else
                      "Same examples and same query; only assigned support-label mapping changed"
                      if name=="swap_support_labels" else
                      "Same support pairs, opposite order; query unchanged")}
            ablations={}
            for name,heads in interventions.items():
                if not heads:
                    scores=base_scores
                else:
                    _,scores,_=safe_metrics(remove_heads(model,heads),
                                     batch,target,BASE_EVAL_SEED+i*101+2)
                diff={}
                for k in ("first_label_accuracy","second_label_teacher_forced_accuracy",
                          "next_symbol_valid_probability_mass"):
                    if k in base_scores and k in scores:
                        diff[k+"_drop_pp"]=100*(base_scores[k]-scores[k])
                ablations[name]={"ablated_heads":heads,"scores":scores,"drops":diff}
            result[gen]={
                "natural":base_scores,
                "head_matching_margins":base_mech["l2_query_first_number_matching_margin_by_head"],
                "head_matching_second":base_mech["l2_query_second_number_matching_margin_by_head"],
                "counterfactual_tests":perturbations,
                "ablations":ablations,
                "time_seconds":time.monotonic()-t0}
            print("ABC_DISCRIMINATION_ROLE="+json.dumps({
                    "arm":args.arm,"gen":gen,"role":role,
                    "counterfactuals":{k:v["accuracy_against_new_oracle"] for k,v in perturbations.items()},
                    "g0_top3_drop_first_pp":ablations["g0_top3"]["drops"]["first_label_accuracy_drop_pp"],
                    "g10_alt_head":alternate}),flush=True)
        output["roles"][role]=result
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(output,indent=2))
    print("ABC_DISCRIMINATION_COMPLETE="+json.dumps({
       "arm":args.arm,"n":args.n,"candidate":alternate,"final_test":False}),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--arm",choices=("clean_only","recursive_replacement"),required=True)
 p.add_argument("--base-root",type=Path,required=True)
 p.add_argument("--g0-root",type=Path,required=True)
 p.add_argument("--g10-root",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 p.add_argument("--n",type=int,default=128)
 a=p.parse_args()
 if a.n not in (64,128,256):raise ValueError("Invalid bounded validation budget")
 run(a)
