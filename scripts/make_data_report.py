"""Build concise, browsable data-audit README from measured JSON files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone

def f(x,d=3):
    return f'{float(x):.{d}f}'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('docs/data-audit'))
    args=p.parse_args()
    path=args.out
    x=json.loads((path/'summary.json').read_text())
    y=json.loads((path/'matching_baselines.json').read_text())
    sh=x['shape']
    text=f"""# Verified original Omniglot feature exploration

**Source:** [Singh et al. original code](https://github.com/aadityasingh/icl-dynamics), pinned submodule `external/icl-dynamics`.  
**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by the repository data-audit workflow.  
**Git blob SHA-1:** `{x['git_blob_sha1']}`. No neural network was trained.

## What is in the file?

| Property | Direct measurement |
| --- | --- |
| HDF5 array | `{x['hdf5_key']}` |
| Shape (classes × exemplars × features) | **{sh[0]} × {sh[1]} × {sh[2]}** |
| Data type | `{x['dtype']}` |
| Feature coordinate minimum / maximum | {f(x['feature_min'])} / {f(x['feature_max'])} |
| Feature-coordinate standard deviation | {f(x['feature_std'])} |
| Average L2 norm of example features | {f(x['mean_feature_norm'])} |
| Mean within-class cosine similarity | **{f(x['mean_within_class_cosine'])}** |
| Mean between-class cosine similarity | **{f(x['mean_between_class_cosine'])}** |

The features are numerical ResNet18 representations, **not** raw handwritten-character images. The class order was randomised. These descriptive numbers are evidence about the *data*, not evidence of transformer induction heads or recursive model collapse.

## Visuals — actual published feature data

![ResNet18 feature norms](01_feature_norms.png)

*Figure 1: variation in feature-vector magnitudes.*

![Same-class and different-class similarities](02_similarity_histograms.png)

*Figure 2: within-class and between-class cosine similarity distributions.*

![Exemplar similarity heatmap](03_exemplar_heatmap.png)

*Figure 3: five exemplars from the selected class compared via cosine similarity.*

![PCA projection](04_feature_pca.png)

*Figure 4: unsupervised two-component PCA projection of randomly sampled classes. This is illustrative, not proof of discriminability across all features.*

## Simple data-only matching baseline

We can ask: if we choose **one reference exemplar per class** and use **another exemplar as the query**, how informative are ResNet18 features for matching? This is **not** transformer training. It is a cosine-similarity control to understand what information is already present in the input representation.

| Task | Measured data-only accuracy | Random choice |
| --- | --- | --- |
| 2-way, one-shot choice (random distractor) | **{f(100*y['two_way_heldout_exemplar_accuracy'],2)}%** over {y['two_way_n']:,} sampled episodes | 50.00% |
| 1-vs-{y['all_class_classes']:,} nearest-neighbour matching | **{f(100*y['all_class_top1_accuracy'],2)}%** over {y['all_class_n']:,} queries | {f(100*y['all_class_random_top1_chance'],3)}% |
| 1-vs-{y['all_class_classes']:,} top-5 inclusion | **{f(100*y['all_class_top5_accuracy'],2)}%** | {f(500/y['all_class_classes'],2)}% |

![Data-only matching baselines](05_cosine_matching_baselines.png)

*Figure 5: distribution of cosine similarities to true versus distractor classes; and retrieval margin against the strongest incorrect class. These checks use **held-out exemplars 1–4** against **first-exemplar support**. They do not prove the original induction-head algorithm is learned.*

## Scientific interpretation and limits

1. **Feature informativeness:** within-class versus between-class similarities show whether related character instances are statistically distinguishable in the frozen ResNet18 embedding space.
2. **Representation versus algorithm:** even excellent cosine retrieval would not demonstrate contextual label copying. In the paper, numerical labels are assigned **dynamically per episode**. The transformer must read the labels *from context* rather than memorise a symbol-to-number lookup.
3. **Geometry versus collapse:** nothing in this audit tests successive synthetic model generations. We will study recursive data feedback only after deciding how the archived symbolic study is being extended.
4. **No hidden image identities:** the feature file does not provide a verified mapping back to raw image-class numbers. The optional second notebook downloads/display Omniglot pictures for intuition but is **not required** for the induction-head mechanism or the proposed symbolic Option B experiment.
5. **Statistical note:** many sampled episodes reuse classes/features, so the episode count must not be read as independent training-run replications.

## Inspect independently

```bash
git clone --recurse-submodules https://github.com/huziibee/nlp_ass_one.git
cd nlp_ass_one
python -m pip install -r requirements-data.txt
python scripts/check_data.py
python scripts/inspect_data.py
python scripts/feature_matching_baseline.py
jupyter lab notebooks/01_explore_omniglot.ipynb
```

**Measured raw summaries:** [summary.json](summary.json) and [matching_baselines.json](matching_baselines.json). The reproducible scripts remain under `scripts/`.
"""
    (path/'README.md').write_text(text,encoding='utf8')
    print('REPORT_GENERATED: '+str(path/'README.md'))

if __name__=='__main__':
    main()
