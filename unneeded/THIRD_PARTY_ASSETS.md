# Third-party assets, attribution, and release boundary

This file describes the **assets actually used** in the Wits NLP Project 1 research and the terms that can be verified from their original sources. It does not assign a license to somebody else's work.

## Asset inventory

| Asset | Source / authors | Verified license or permission statement | Handling in this repository |
| --- | --- | --- | --- |
| Original Omniglot dataset | Brenden Lake et al.; https://github.com/brendenlake/omniglot | The original repository includes an MIT LICENSE, copyright (c) 2015 Brenden Lake. Read the license at https://github.com/brendenlake/omniglot/blob/master/LICENSE. | Cite Lake, Salakhutdinov and Tenenbaum (2015). No copy of the original image/stroke corpus is vendored here. If redistributing original MIT-covered material elsewhere, retain the required license/copyright notice. |
| Singh et al. research code, including model/sampler/opto implementation | Aaditya K. Singh et al.; https://github.com/aadityasingh/icl-dynamics; Singh et al. (ICML 2024) | **No top-level LICENSE/COPYING/NOTICE or explicit general-purpose redistribution license was found in the pinned upstream revision** (checked 25 October 2026). The upstream README makes the code publicly available for research and invites inquiries, but that is not equivalent to an express open-source license. | Referenced only as a pinned Git submodule (`external/icl-dynamics`, original gitlink revision `85b895c720844e795734b9391b32d2619065f9a1`); it is **not vendored as ordinary source files** and is not offered under a student license. Users who initialize the submodule fetch it directly from the original authors' repository. Do not republish this source tree, mirror it into another archive, or assert MIT/Apache/CC rights without an explicit author grant. |
| Precomputed Omniglot / ResNet-18 feature file `omniglot_resnet18_randomized_order_s0.h5` | Distributed by Singh et al. in their `icl-dynamics` repository; derived using Omniglot source data | The original Omniglot repository has an MIT notice, but no separate explicit redistribution license for **Singh et al.'s transformed feature file** was located in the pinned `icl-dynamics` revision. The original MIT notice alone does not settle every right in independently produced preprocessing or packaging. | Do **not** commit, upload, mirror, or bundle this `.h5` in the student repository, downloadable submission archives, or releases. A recursive submodule clone may fetch it directly from upstream: that is an upstream-hosted dependency, not an assertion that the student group may sublicense it. |
| Student-authored research scripts, reports, plots, metadata and tests | Wits NLP Project 1 student authors | Ownership/license for student work is a **separate** question; none is granted to upstream files merely by publishing this repository. | Retained here with attribution to the authors. No license on this repository should claim rights to third-party submodule code or the upstream feature file. |

## Provenance and practical reproduction

The submitted research reuses the original Singh et al. task by reference and uses student-authored training/evaluation/analysis scripts. The assignment brief explicitly welcomes using the original paper's GitHub repository for the project; this course instruction **does not substitute for an upstream author's general public redistribution license**.

The root Git tree records `external/icl-dynamics` as a **gitlink (mode 160000)**, not ordinary vendored files. The root Git tree contains no tracked `.h5`, `.hdf5`, `.eqx`, or model-checkpoint binary. The documented `git clone --recurse-submodules` command obtains the third-party files from their original host, and normal GitHub Actions checkpoints remain separate from ordinary source control.

For attribution, cite:

- Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). *Human-level concept learning through probabilistic program induction*. Science 350(6266), 1332-1338.
- Singh, A. K., Moskovitz, T., Hill, F., Chan, S. C. Y., and Saxe, A. M. (2024). *What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation*. ICML, PMLR 235, 45637-45662. https://proceedings.mlr.press/v235/singh24c.html

## Release checklist

- [x] Identify the authors and official URLs of all principal third-party research assets.
- [x] Verify the original Omniglot MIT notice and the lack of a declared license in the pinned Singh code repository.
- [x] Keep upstream code as a gitlink instead of copying the upstream source tree into student files.
- [x] Do not bundle or rehost the Singh feature HDF5 in this repository.
- [x] Distinguish academic acknowledgement, reproducibility instructions and legal permission.
- [ ] **Optional for broader redistribution:** request written clarification from the Singh repository maintainers if you want to copy, mirror, package, modify or sublicense upstream code/assets beyond what the existing host and applicable law permit.

**Interpretation:** This is a *non-redistributing release arrangement*, not a claim that the license of the original Singh code has been obtained or that every downstream use is automatically authorized. If an independent review requires an express license for use rather than non-redistribution, obtain permission from the authors.
