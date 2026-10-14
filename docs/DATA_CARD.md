# Data card — Original Omniglot ResNet18 features

## Identity and provenance
- Official project: https://github.com/aadityasingh/icl-dynamics
- Binary: `omniglot_resnet18_randomized_order_s0.h5`
- Location: `external/icl-dynamics/omniglot_resnet18_randomized_order_s0.h5` (Git submodule)
- Upstream pinned commit: `85b895c720844e795734b9391b32d2619065f9a1`
- Size: **16,638,016 bytes**
- Git blob SHA-1: `9eb588c516b641f5f25ac22227865a7ce5ab0b2f`
- HDF5 feature key read by original code: `resnet18/224/feat`
- Upstream preprocessing script: `external/icl-dynamics/omni_features_extract.py`

## Observed HDF5 metadata (real file, run 14 October 2026)
- `resnet18/224/feat`: shape `[1623, 5, 512]`, dtype `float32`.
- `class_order`: shape `[1623]`, dtype `int64`, all unique values 0 through 1622, nonidentity permutation.
- Measured mean cosine similarity: 0.889 for same-class exemplar comparisons versus 0.771 for different-class comparisons.
- Data-only two-way matching among two contextual options: 92.91% using a **different exemplar** as query; this should **not** be described as transformer ICL accuracy.
- `class_order` plausibly records the shuffle but **has not been validated** as a raw-image-ID correspondence.
- Measurements and plots: [verified data audit](data-audit/README.md).

## Meaning

The original Omniglot dataset contains handwritten character pictures. The upstream researchers passed example images through a frozen, ImageNet-pretrained ResNet18 and saved numerical feature vectors. Their readme states five examples per class were retained; the official paper uses one per class in its default training experiment.

The numeric array uses `[class, exemplar, feature coordinate]`. Its expected first two dimensions are 1,623 classes and five exemplar vectors per class. **Do not assume the feature dimension until `scripts/check_data.py` has printed the real array shape.**

The feature file contains **not the source image pixels**. A line chart of its 512-ish feature coordinates is not an actual handwritten character. IDs have been randomly reordered and should not be described as named writing systems.

## Why this matters for the assignment

The original task randomly maps two character classes to two labels *within an episode*. Predicting a held-out query label requires context-dependent matching and copying. The original repository can generate these episodes from feature vectors and supports held-out examples, classes and relabelings.

The private, historical symbolic experiment `huziibee/nlp_ass_1` used **learned discrete-symbol embeddings**, not this HDF5, although it stored an identical copy of the binary. Treat any connection between them as a *new hypothesis*, not an established replication.

## What we can investigate before training
- Data dimensions, array keys, basic numeric integrity and identity hash.
- Distribution and magnitudes of embeddings.
- Within-class versus between-class similarity.
- Two-dimensional PCA projections (exploratory only).
- Oracle symbol/label episodes and the fact that mappings vary by context.

These exploratory plots **cannot establish model collapse or induction-head causality**. Those require trained models, recursive synthetic-data experiments, matched controls, and interventions.

## Privacy, redistribution and source integrity
The original source is public, but this repository links it as a submodule instead of embedding another copy or asserting redistribution rights. Keep the university teaching PDFs and the completed private research archive out of this public repository unless permissions are checked. The dataset identity is checked by byte count and Git blob SHA-1; do not accidentally commit another modified copy.
