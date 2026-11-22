# Reference files from ProteinGym

Copied from the ProteinGym repository (https://github.com/OATML-Markslab/ProteinGym, MIT licence, see
`LICENSE_ProteinGym`) on 22 November 2026:

* `DMS_substitutions.csv`: one row per substitution assay (target sequence, function, taxon, MSA depth, ...),
  from `reference_files/`.
* `DMS_substitutions_Spearman_DMS_level.csv`: every model's published Spearman correlation on every assay,
  from `benchmarks/DMS_zero_shot/substitutions/Spearman/`.
* `Summary_performance_DMS_substitutions_Spearman.csv`: the published aggregate scores, from the same folder.

The assay data themselves (`DMS_ProteinGym_substitutions.zip`, `cv_folds_singles_substitutions.zip`, ProteinGym
v1.3) are not stored here; see the main README for how to download them.

Notin, P. et al. (2023). ProteinGym: Large-scale benchmarks for protein fitness prediction and design.
*Advances in Neural Information Processing Systems* 36.
