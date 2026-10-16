"""Aggregate matched raw-versus-G9-target-corrected G10 q2 causal activation interventions."""
import argparse,csv,json,math,statistics
from pathlib import Path

SEEDS=(6001,5054,6054,7054,8054)
ARMS=("recursive_replacement","fix_invalid_q2")
T95_DF4=2.7764451051977987

def get(root,seed,arm):
    files=list(root.rglob(f"{seed}-{arm}.json"))
    if len(files)!=1:raise AssertionError((root,seed,arm,files))
    d=json.loads(files[0].read_text())
    assert d['arm']==arm and d['n']==128 and d['final_test_accessed'] is False
    assert d['patched_position']==5 and set(d['layers'])=={'1','2'}
    for layer in d['layers'].values():
        assert layer['identity_patch_max_error']<3e-5
        assert layer['future_token_patch_max_error']<3e-5
        assert len(layer['head_level'])==8
    return d

def get_metrics(d):
    l1,l2=d['layers']['1'],d['layers']['2']
    assert abs(l1['recipient_margin_mean']-l2['recipient_margin_mean'])<1e-5
    sensitivity=float(l2['recipient_margin_mean']-l2['donor_margin_mean'])
    assert sensitivity>0
    return {
        'valid_mass':float(l2['recipient_valid_mass']),
        'context_preference_logit_units':sensitivity,
        'layer1_patch_shift_logit_units':float(l1['whole_layer_patch_margin_shift']),
        'layer2_patch_shift_logit_units':float(l2['whole_layer_patch_margin_shift']),
        'layer2_patch_fraction_of_context_preference': -float(l2['whole_layer_patch_margin_shift'])/sensitivity,
        'layer2_head_effect_sum_logit_units':sum(float(x['q2_old_B_vs_new_C_margin_shift']) for x in l2['head_level'].values())
    }

def summarize(ds):
    mu=statistics.mean(ds)
    sd=statistics.stdev(ds)
    m=T95_DF4*sd/math.sqrt(5)
    return {'mean':mu,'sd':sd,'paired_t95_interval':[mu-m,mu+m],
            'per_model_seed':{str(k):v for k,v in zip(SEEDS,ds)},
            'n_seed_pairs':5}

def run(args):
    rows=[]
    for seed in SEEDS:
        raw=get(args.raw_root,seed,'recursive_replacement')
        fixed=get(args.repaired_root,seed,'fix_invalid_q2')
        r=get_metrics(raw);f=get_metrics(fixed)
        line={'seed':seed}
        for k in r:
            line['raw_'+k]=r[k]
            line['repaired_'+k]=f[k]
            line['repaired_minus_raw_'+k]=f[k]-r[k]
        rows.append(line)
    assert len(rows)==5
    metrics=list(get_metrics(raw))
    summary={'study':'Matched G9-generated synthetic-target correction G10, then within-model activation patching',
             'comparison':'Raw recursive G10 versus G10 retrained with invalid q2 and its number repaired',
             'n_independent_paired_training_seed_experiments':5,
             'n_fixed_validation_episodes_each_model':128,
             'evaluation':'Omniglot training-class symbols and held-out numeric label pairs (val_relabel)',
             'final_test_accessed':False,
             'repair_affects':'generated q2 and linked number only for invalid-context symbol continuations; same G9 generator and training procedure',
             'activation_patching':'within each fixed trained model, source B versus target C at q2 position 5',
             'metrics':{},
             'caveats':['Logit margin differs from normalized probability; raw scale can differ by training condition',
                       'Head patching identifies sensitivity to a replaceable internal activation, not a unique head circuit',
                       'The intervention changes the full learned model, so post-training activation mediation is not isolated as the only cause',
                       'Evaluation reuses development validation tasks, not fresh final test',
                       'One shared set of 128 validation contexts per model is not 128 independent models']}
    for name in metrics:
        summary['metrics'][name]=summarize([x['repaired_minus_raw_'+name] for x in rows])
        summary['metrics'][name]['raw_mean']=statistics.mean(x['raw_'+name] for x in rows)
        summary['metrics'][name]['repaired_mean']=statistics.mean(x['repaired_'+name] for x in rows)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'paired_model_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Q2_TARGET_CORRECTION_MEDIATION_FIVE_SEED='+json.dumps({
        k:{'raw':v['raw_mean'],'repaired':v['repaired_mean'],
           'effect':v['mean'],'interval':v['paired_t95_interval']}
        for k,v in summary['metrics'].items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--repaired-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
