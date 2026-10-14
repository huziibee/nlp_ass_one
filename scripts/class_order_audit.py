"""Inspect the HDF5's class_order metadata without assuming raw-image alignment."""
from __future__ import annotations
import json
from pathlib import Path
import h5py
import numpy as np
from check_data import DATA, inspect

def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--file',type=Path,default=DATA)
    ap.add_argument('--out',type=Path,default=Path('docs/data-audit'))
    a=ap.parse_args()
    verified=inspect(a.file)
    result={'feature_data_sha':verified['git_blob_sha1']}
    with h5py.File(a.file,'r') as f:
        if 'class_order' not in f:
            result['class_order_present']=False
        else:
            v=np.asarray(f['class_order'])
            expected=np.arange(len(v),dtype=np.int64)
            result.update({
                'class_order_present':True,
                'dtype':str(v.dtype),
                'length':len(v),
                'first_25':v[:25].astype(int).tolist(),
                'last_10':v[-10:].astype(int).tolist(),
                'min':int(np.min(v)),
                'max':int(np.max(v)),
                'all_unique':len(np.unique(v))==len(v),
                'is_permutation_0_to_n_minus_1':bool(np.array_equal(np.sort(v),expected)),
                'is_identity_order':bool(np.array_equal(v,expected)),
                'interpretation':'Order metadata may encode the permutation used to reorder classes. Aligning it with torchvision raw-image labels requires an explicit validation against the original preprocessing pipeline; this audit does not demonstrate that alignment.'
            })
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'class_order.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('CLASS_ORDER_AUDIT='+json.dumps(result))
if __name__=='__main__':
    main()
