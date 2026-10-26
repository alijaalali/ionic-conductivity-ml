# ML for room-temperature ionic conductivity of lithium solid-state electrolytes (OBELiX)

Scripts used to build features, train models, run the robustness/uncertainty analyses, and generate figures/tables for the manuscript.

## Pipeline order

1. `build_master_table.py` — composition-tier (Magpie) and structure-tier (Voronoi/CrystalNN) featurization of the OBELiX dataset
2. `train_xgboost_baseline.py` — XGBoost point-estimate baseline, composition-only and composition+structure tiers, four evaluation protocols
3. `train_uncertainty.py` — quantile-regression uncertainty for the composition-only model, checked against the dataset's own empirical measurement noise floor
4. `tier_comparison_and_robustness_checks.py` — three-tier feature comparison with paired bootstrap, noise-floor reanalysis, censored-measurement sensitivity, composition-grouped CV, quantile-calibration, LGPS feature-space distance, lithium-site-imputation ablation
5. `feature_selection_bootstrap_and_screening_metrics.py` — feature-selection/dimensionality experiment, hyperparameter sensitivity, composition-grouped-CV distributional check, ranking-usefulness metrics, LGPS distance (cosine) + PCA projection, pinball-loss quantile-model selection
6. `leakage_free_bootstrap_and_group_cv_checks.py` — feature-inventory check, composition-overlap audit, fold-specific (leakage-free) feature selection, group-level bootstrap, extended screening metrics, missingness-indicator ablation, nested hyperparameter selection, fixed-split-vs-grouped-CV error overlap

## Figures

- `make_figures.py` — main-text and SI figures: dataset overview, parity plot, leave-one-family-out MAE, uncertainty calibration
- `make_figures_robustness_checks.py` — tier-comparison, duplicate-group noise-floor, and LGPS feature-space-distance figures

## Requirements

Python 3, pandas, numpy, scikit-learn, scipy, xgboost, matminer, pymatgen, matplotlib.

## Data

The OBELiX dataset (Therrien et al.) is publicly available at `github.com/NRC-Mila/OBELiX`. Scripts expect it, plus the derived feature table, in a local `data/` directory. See the manuscript's Data Availability statement.
