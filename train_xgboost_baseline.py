# XGBoost baseline for OBELiX ionic conductivity (composition-only and composition+structure
# feature tiers) under four evaluation protocols: the paper's fixed split, leave-one-family-
# out, leave-one-space-group-out, and leave-one-duplicate-measurement-group-out (compared
# against the dataset's own reported empirical noise floor).
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_DIR = Path("/glade/work/ajalali/Ionic-Conductivity-ML-project/data")
OBELIX_DATA = Path("/glade/derecho/scratch/ajalali/obelix-data/OBELiX/data")
OUT_DIR = Path("/glade/work/ajalali/Ionic-Conductivity-ML-project/models/xgb_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ROWS_PER_GROUP = 8
PAPER_NOISE_RMSD = 0.63  # log10(S/cm), from the OBELiX paper's duplicate-measurement analysis
PAPER_NOISE_MAD = 0.41


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    df = pd.read_parquet(DATA_DIR / "master_table.parquet")
    magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
    lattice_cols = ["a", "b", "c", "alpha", "beta", "gamma", "Space_group_number", "Z"]
    struct_cols = [
        c for c in df.columns
        if c.startswith("all_voro_") or c.startswith("li_voro_")
        or c in ("all_coord_num_mean", "all_coord_num_std", "li_coord_num_mean", "li_coord_num_std",
                  "n_sites", "n_li_sites", "cell_volume", "density")
    ]
    return df, magpie_cols, lattice_cols, struct_cols


def fit_and_eval(X, y, train_mask, test_mask, n_estimators=300):
    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        n_jobs=8, random_state=42,
    )
    model.fit(X[train_mask], y[train_mask], verbose=False)
    pred = model.predict(X[test_mask])
    metrics = {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "mae": float(mean_absolute_error(y[test_mask], pred)),
        "rmse": float(np.sqrt(mean_squared_error(y[test_mask], pred))),
        "r2": float(r2_score(y[test_mask], pred)) if test_mask.sum() > 1 else None,
    }
    return model, metrics, pred


def run_tier(df, feature_cols, tier_name):
    log(f"=== Tier: {tier_name} ({len(df)} rows, {len(feature_cols)} features) ===")
    cols = feature_cols
    X = df[cols].to_numpy(dtype=float)
    y = df["log_IC"].to_numpy(dtype=float)
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)

    results = {}

    # --- 1. paper's fixed split ---
    train_ids = set(pd.read_csv(OBELIX_DATA / "train_idx.csv")["ID"])
    test_ids = set(pd.read_csv(OBELIX_DATA / "test_idx.csv")["ID"])
    train_mask = df["ID"].isin(train_ids).to_numpy()
    test_mask = df["ID"].isin(test_ids).to_numpy()
    if train_mask.sum() > 0 and test_mask.sum() > 0:
        model, m, pred = fit_and_eval(X, y, train_mask, test_mask, n_estimators=500)
        results["paper_split"] = m
        log(f"  paper split: {m}")
        joblib_path = OUT_DIR / f"model_{tier_name}.joblib"
        import joblib
        joblib.dump(model, joblib_path)
        with open(OUT_DIR / f"feature_names_{tier_name}.json", "w") as f:
            json.dump(cols, f)
        pred_df = df.loc[test_mask, ["ID", "Reduced_Composition", "Family"]].copy()
        pred_df["log_IC_true"] = y[test_mask]
        pred_df["log_IC_pred"] = pred
        pred_df.to_csv(OUT_DIR / f"predictions_{tier_name}.csv", index=False)

    # --- 2. leave-one-family-out ---
    log("  Leave-one-family-out...")
    family_results = {}
    for fam, count in df["Family"].value_counts().items():
        if count < MIN_ROWS_PER_GROUP:
            continue
        test_m = (df["Family"] == fam).to_numpy()
        train_m = df["Family"].notna().to_numpy() & ~test_m
        _, m, _ = fit_and_eval(X, y, train_m, test_m)
        family_results[fam] = m
        log(f"    leave-{fam}-out: n={m['n_test']} MAE={m['mae']:.3f} R2={m['r2']}")
    results["leave_one_family_out"] = family_results

    # --- 3. leave-one-space-group-out ---
    log("  Leave-one-space-group-out...")
    sg_results = {}
    for sg, count in df["Space_group_number"].value_counts().items():
        if count < MIN_ROWS_PER_GROUP:
            continue
        test_m = (df["Space_group_number"] == sg).to_numpy()
        train_m = ~test_m
        _, m, _ = fit_and_eval(X, y, train_m, test_m)
        sg_results[str(sg)] = m
        log(f"    leave-sg{sg}-out: n={m['n_test']} MAE={m['mae']:.3f}")
    results["leave_one_space_group_out"] = sg_results

    # --- 4. leave-one-duplicate-group-out ---
    log("  Leave-one-duplicate-group-out...")
    dup_results = {}
    dup_groups = df[df["duplicate_group_size"] > 1]["duplicate_group_id"].unique()
    all_group_maes = []
    for gid in dup_groups:
        test_m = (df["duplicate_group_id"] == gid).to_numpy()
        if test_m.sum() < 2:
            continue
        train_m = ~test_m
        _, m, pred = fit_and_eval(X, y, train_m, test_m, n_estimators=200)
        within_group_true_std = float(np.std(y[test_m]))
        dup_results[gid] = {**m, "within_group_true_std": within_group_true_std}
        all_group_maes.append(m["mae"])
    if all_group_maes:
        results["duplicate_group_summary"] = {
            "n_groups_tested": len(all_group_maes),
            "mean_mae_across_groups": float(np.mean(all_group_maes)),
            "median_mae_across_groups": float(np.median(all_group_maes)),
            "paper_reported_noise_rmsd": PAPER_NOISE_RMSD,
            "paper_reported_noise_mad": PAPER_NOISE_MAD,
        }
        log(f"    mean MAE across {len(all_group_maes)} duplicate groups: "
            f"{np.mean(all_group_maes):.3f} (paper's noise floor MAD: {PAPER_NOISE_MAD})")
    results["duplicate_group_detail"] = dup_results

    return results


def main():
    t0 = time.time()
    df, magpie_cols, lattice_cols, struct_cols = load_data()
    comp_feature_cols = magpie_cols + lattice_cols
    struct_feature_cols = comp_feature_cols + struct_cols

    log(f"Composition features: {len(comp_feature_cols)}, structure tier adds {len(struct_cols)} more")

    df_comp = df.copy()
    results_comp = run_tier(df_comp, comp_feature_cols, "composition_only")

    df_struct = df[df["has_structure"]].reset_index(drop=True)
    results_struct = run_tier(df_struct, struct_feature_cols, "composition_plus_structure")

    all_results = {
        "composition_only": results_comp,
        "composition_plus_structure": results_struct,
        "runtime_seconds": time.time() - t0,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
