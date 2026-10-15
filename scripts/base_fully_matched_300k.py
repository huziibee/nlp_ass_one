"""Original Base Omniglot fully matched 300k G0 and descendant 300k budgets.

Uses pinned Singh JAX implementation, same feature file/class roles and no
protected test access. Contrasts with Extended 300k G0+300k descendants.
Even with matched exposure, architectures/output tasks still differ.
"""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"external"/"icl-dynamics"))
from types import SimpleNamespace
import jax
import equinox as eqx
import main_utils
from scripts.paper_base_smoke import source_checked_cmd
import scripts.base_recursive as source_recursive
from scripts.base_mechanistic import DATA

def training(root):
 out=root;out.mkdir(parents=True,exist_ok=True)
 cmd=source_checked_cmd(steps=9375,out=out,batch_size=32,
   eval_every_updates=9375,eval_iters=512,ckpt_every_updates=9375,
   max_updates=100000)
 assert 'test' not in cmd[cmd.index('--pe_classes')+1:cmd.index('--pe_classes')+5]
 subprocess.run(cmd,cwd=ROOT,check=True)
 assert (out/'safe_train_val_smoke'/'checkpoints'/'00000300000.eqx').is_file()
 print("BASE_G0_300K_COMPLETED",flush=True)

def load_g0(root):
 folder=root/'safe_train_val_smoke'
 config=json.loads((folder/'config.json').read_text())
 assert config['train_iters']==300000
 assert config['class_split']==[50,1473,100]
 assert config['fs_relabel_split']==[8,2,0]
 config['data_file']=str(DATA)
 opts=SimpleNamespace(**config)
 model=main_utils.get_model_from_opts(opts)
 optimizer=main_utils.get_optimizer_from_opts(opts)
 template={
  "iter":-1,
  "seeds":{"eval_model_seed":jax.random.PRNGKey(0),
          "train_data_seed":jax.random.PRNGKey(0),
          "train_model_seed":jax.random.PRNGKey(0)},
  "opt_state":optimizer.init(eqx.filter(model,eqx.is_array)),
  "model":model}
 checkpoint=folder/'checkpoints'/'00000300000.eqx'
 restored=eqx.tree_deserialise_leaves(str(checkpoint),template)
 if int(restored['iter'])!=300000:raise AssertionError("Wrong G0 training budget")
 return restored['model'],config

def descendants(base_root,out,arm):
 source_recursive.load_original_checkpoint=load_g0
 source_recursive.experiment(base_root,out,generations=10,steps=9375,
     batch_size=32,eval_every=9375,eval_count=512,seed=34001,
     temperature=1.,arms=[arm])
 r=json.loads((out/'summary.json').read_text())
 if r.get('final_test_evaluated') is not False:raise AssertionError("Final-test access")
 assert len(r['arms'][arm])==10
 for g in range(1,11):
    meta=json.loads((out/arm/f'g{g:02d}'/'metrics.json').read_text())
    assert meta['optimizer_exposures']==300000
 print("FULLY_MATCHED_BASE_300K="+json.dumps({
  "arm":arm,"g0_sequences":300000,"descendant_sequences_per_gen":300000,
  "g10":r['arms'][arm][-1]['last_validation'],"no_final_test":True}),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--task",choices=("g0","descendants"),required=True)
 p.add_argument("--g0-root",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 p.add_argument("--arm",choices=("clean_only","recursive_replacement"))
 a=p.parse_args()
 if a.task=="g0":training(a.g0_root)
 else:
  if a.arm is None:raise ValueError("descendant arm required")
  descendants(a.g0_root,a.out,a.arm)
