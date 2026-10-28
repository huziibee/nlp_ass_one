"""Data-only matching baselines on published Omniglot ResNet18 features.

All numbers are descriptive, not transformer accuracy.
- 2-way 1-shot: query is a different exemplar of one of 2 support classes.
- All-class 1-shot: nearest cosine similarity among 1623 support classes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from check_data import DATA, KEY, inspect

ROOT=Path(__file__).resolve().parents[1]

def norm_rows(x):
    x=x.astype(np.float32,copy=False)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def run(path:Path, out:Path, seed:int=42, episodes:int=50000):
    provenance=inspect(path)
    with h5py.File(path,'r') as h:
        features=np.asarray(h[KEY],dtype=np.float32)
    c,e,d=features.shape
    if e<2:
        raise RuntimeError("Need at least two exemplars per class")
    f=norm_rows(features)
    supports=f[:,0,:]
    rng=np.random.default_rng(seed)
                                                                           
                                                                      
                                                                  
    actual=rng.integers(0,c,size=episodes)
    distractor=rng.integers(0,c-1,size=episodes)
    distractor += (distractor >= actual)
    exemplars=rng.integers(1,e,size=episodes)
    queries=f[actual,exemplars]
    correct_sim=np.sum(queries*supports[actual],axis=1)
    distract_sim=np.sum(queries*supports[distractor],axis=1)
    wins=correct_sim>distract_sim
    ties=np.isclose(correct_sim,distract_sim,atol=1e-10)
    two_way=float(np.mean(wins.astype(float)+0.5*ties.astype(float)))

                                                                     
                                                                    
                                                             
    q=f[:,1:,:].reshape(-1,d)
    labels=np.repeat(np.arange(c),e-1)
    predicted=[]
    first5=[]
    correct_vals=[]
    best_wrong_vals=[]
    for lo in range(0,len(q),256):
        hi=min(len(q),lo+256)
        sims=q[lo:hi]@supports.T
        lab=labels[lo:hi]
        best=np.argmax(sims,axis=1)
                                                              
        correct=sims[np.arange(hi-lo),lab].copy()
        copy=sims.copy()
        copy[np.arange(hi-lo),lab]=-np.inf
        strongest_wrong=np.max(copy,axis=1)
        predicted.extend(best.tolist())
        first5.extend((np.sum(sims>correct[:,None],axis=1)<5).tolist())
        correct_vals.extend(correct.tolist())
        best_wrong_vals.extend(strongest_wrong.tolist())
    pred=np.asarray(predicted)
    margins=np.asarray(correct_vals)-np.asarray(best_wrong_vals)
    k1=float(np.mean(pred==labels))
    k5=float(np.mean(first5))
    out.mkdir(parents=True,exist_ok=True)

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].hist(correct_sim,density=True,bins=45,alpha=0.65,label='Correct class')
    axes[0].hist(distract_sim,density=True,bins=45,alpha=0.65,label='Random distractor')
    axes[0].set(xlabel='Cosine similarity',ylabel='Density',
                title='Two-way matching: held-out query exemplar')
    axes[0].legend(fontsize=8)
    axes[1].hist(margins,bins=65,density=True)
    axes[1].axvline(0,linestyle='--',color='black',linewidth=1)
    axes[1].set(xlabel='Correct - strongest incorrect similarity',
                ylabel='Density',title='Full-class nearest-neighbour margin')
    fig.tight_layout()
    fig.savefig(out/'05_cosine_matching_baselines.png',dpi=160)
    plt.close(fig)
    report={
        'source_sha':provenance['git_blob_sha1'],
        'seed':seed,
        'two_way_n':episodes,
        'two_way_heldout_exemplar_accuracy':two_way,
        'two_way_random_chance':0.5,
        'all_class_n':len(q),
        'all_class_classes':c,
        'all_class_top1_accuracy':k1,
        'all_class_top5_accuracy':k5,
        'all_class_random_top1_chance':1/c,
        'mean_correct_cosine':float(np.mean(correct_vals)),
        'mean_best_wrong_cosine':float(np.mean(best_wrong_vals)),
        'median_margin_vs_best_distractor':float(np.median(margins)),
        'notes':[
            'Uses raw precomputed vectors; no transformer and no training.',
            'Queries use exemplars 1..4, supports use exemplar 0.',
            '2-way distractor sampled uniformly from other classes per episode.',
            'Accuracy is descriptive on correlated examples, not independent trial CI.',
            'One-shot cosine matching is not the original learned induction-head algorithm.'
        ]
    }
    (out/'matching_baselines.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('FEATURE_MATCHING_RESULTS='+json.dumps(report))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file',type=Path,default=DATA)
    p.add_argument('--out',type=Path,default=ROOT/'docs'/'data-audit')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--episodes',type=int,default=50000)
    a=p.parse_args()
    run(a.file,a.out,a.seed,a.episodes)
