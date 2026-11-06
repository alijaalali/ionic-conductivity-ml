#!/bin/bash
# Reproduces every number and figure of the manuscript from data/master_table.parquet.
# Run from the repository root (about 30-60 minutes on 8 cores).
set -e
python train_xgboost_baseline.py
python train_uncertainty.py
python tier_comparison_and_robustness_checks.py
python feature_selection_bootstrap_and_screening_metrics.py
python leakage_free_bootstrap_and_group_cv_checks.py
python supplementary_checks.py
python make_figures.py
python make_figures_robustness_checks.py
python make_graphical_abstract.py
