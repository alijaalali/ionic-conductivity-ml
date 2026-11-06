# Machine-learning prediction of lithium solid-electrolyte conductivity (OBELiX)

Code, derived data, and result files for "Generalization and Uncertainty in Machine-Learning Predictions of Lithium Solid-Electrolyte Conductivity". Gradient-boosted trees are trained on the OBELiX dataset with three feature tiers (A: composition; B: plus lattice parameters, space group, formula units per cell; C: plus Voronoi/coordination descriptors) and evaluated under four protocols (OBELiX fixed split, leave-one-family-out, leave-one-space-group-out, leave-one-duplicate-group-out).

## Repository layout

| Path | Content |
|---|---|
| `config.py` | Relative paths shared by all scripts |
| `data/master_table.parquet` | Derived feature table (599 entries; Magpie, lattice, Voronoi/CrystalNN features; target `log_IC` = log10 of the conductivity in S/cm) |
| `data/obelix/` | OBELiX files used here (`all.csv`, `train_idx.csv`, `test_idx.csv`, `all_randomized_cifs/`), copied unchanged from the public OBELiX repository, CC BY 4.0 (license text included) |
| `results/` | JSON/CSV outputs of the scripts below, as used for the numbers in the manuscript (fitted model files are not stored; `train_xgboost_baseline.py` regenerates them) |
| `figures/` | Figures written by the plotting scripts |

## Scripts (run from the repository root, in this order)

1. `build_master_table.py`: Magpie composition features and Voronoi/CrystalNN structure features from the OBELiX files; writes `data/master_table.parquet`. Optional, since the table is included (about 40 minutes; needs matminer, pymatgen and the `obelix-data` package).
2. `train_xgboost_baseline.py`: XGBoost point estimates, two feature tiers, four protocols.
3. `train_uncertainty.py`: quantile-regression uncertainty, noise-floor comparison.
4. `tier_comparison_and_robustness_checks.py`: three-tier comparison with paired bootstrap, noise-floor reanalysis, censored-measurement check, composition-grouped CV, quantile-model selection and calibration, LGPS feature-space distance, lithium-site-imputation ablation.
5. `feature_selection_bootstrap_and_screening_metrics.py`: reduced-descriptor experiment, Tier C hyperparameter sensitivity, grouped-CV distribution check, ranking metrics, cosine/city-block distances, pinball-loss sweep.
6. `leakage_free_bootstrap_and_group_cv_checks.py`: feature inventory, composition-overlap audit, fold-specific feature selection, group-level bootstrap, extended screening metrics, missingness-indicator ablation, per-fold hyperparameter tuning, fixed-split versus grouped-CV error overlap.
7. `supplementary_checks.py`: grouped bootstrap of the screening enrichment factors, numeric versus one-hot space-group encoding, nearest-neighbour distances for LGPS and comparison families.
8. `make_figures.py`, `make_figures_robustness_checks.py`, `make_graphical_abstract.py`: figures of the manuscript and the Supporting Information.

`run_all.sh` runs steps 2 to 8 (the data table is included). Tree counts of the point-estimate models: 300 by default, 500 for the full-dataset paper split, 200 for leave-one-duplicate-group-out in the tier comparison (see the `n_estimators` arguments in the scripts); the quantile models use 30 trees, depth 2, lambda 20. Random seeds are fixed (42); small differences in the last digit can occur across library versions.

## Requirements

Python 3 with the packages in `requirements.txt` (the versions listed are those used for the manuscript: XGBoost 3.2.0, scikit-learn 1.9.1, matminer 0.10.1).

## Data and license

The OBELiX dataset is by Therrien et al. (Digital Discovery 2026, 5, 910; arXiv:2502.14234) and is available at https://github.com/NRC-Mila/OBELiX under CC BY 4.0; the files in `data/obelix/` are unchanged copies. The code in this repository is released under the MIT license (`LICENSE`).
