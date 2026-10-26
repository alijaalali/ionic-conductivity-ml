# Feature-inventory check, composition-overlap audit for leave-one-duplicate-group-out,
# fold-specific (leakage-free) feature selection, group-level/hierarchical bootstrap for
# the Tier B vs Tier C comparison, extended screening metrics, a missingness-indicator
# ablation, nested hyperparameter selection, and a fixed-split-vs-grouped-CV error-overlap
# check. Builds on build_master_table.py's featurized table.
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path("/glade/work/ajalali/11 - Ionic-Conductivity-ML-project")
DATA_DIR = ROOT / "data"
OBELIX_DATA = Path("/glade/derecho/scratch/ajalali/obelix-data/OBELiX/data")
OUT_DIR = ROOT / "models" / "leakage_free_bootstrap_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MIN_ROWS_PER_GROUP = 8
RNG = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load():
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


def fit_xgb(X, y, train_mask, test_mask, n_estimators=300, max_depth=5, seed=RNG, reg_lambda=1.0):
    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=reg_lambda,
        n_jobs=8, random_state=seed,
    )
    model.fit(X[train_mask], y[train_mask], verbose=False)
    pred = model.predict(X[test_mask])
    return model, pred


def nan_fill(X, fill=None):
    if fill is None:
        fill = np.nanmean(X, axis=0)
    return np.where(np.isnan(X), fill, X), fill


def protocol_masks_struct(dfx, train_ids_paper, test_ids_paper):
    protos = {}
    tm = dfx["ID"].isin(train_ids_paper).to_numpy()
    te = dfx["ID"].isin(test_ids_paper).to_numpy()
    protos["paper_split"] = [(tm, te, "paper_split")] if tm.sum() and te.sum() else []
    fam_list = []
    for fam, count in dfx["Family"].value_counts().items():
        if count < MIN_ROWS_PER_GROUP:
            continue
        te_m = (dfx["Family"] == fam).to_numpy()
        tr_m = dfx["Family"].notna().to_numpy() & ~te_m
        fam_list.append((tr_m, te_m, fam))
    protos["leave_one_family_out"] = fam_list
    sg_list = []
    for sg, count in dfx["Space_group_number"].value_counts().items():
        if count < MIN_ROWS_PER_GROUP:
            continue
        te_m = (dfx["Space_group_number"] == sg).to_numpy()
        tr_m = ~te_m
        sg_list.append((tr_m, te_m, str(sg)))
    protos["leave_one_space_group_out"] = sg_list
    dup_list = []
    dup_groups = dfx[dfx["duplicate_group_size"] > 1]["duplicate_group_id"].unique()
    for gid in dup_groups:
        te_m = (dfx["duplicate_group_id"] == gid).to_numpy()
        if te_m.sum() < 2:
            continue
        tr_m = ~te_m
        dup_list.append((tr_m, te_m, gid))
    protos["leave_one_duplicate_group_out"] = dup_list
    return protos


log("Loading data")
df, magpie_cols, lattice_cols, struct_cols = load()
df_struct = df[df["has_structure"]].reset_index(drop=True)
train_ids_paper = set(pd.read_csv(OBELIX_DATA / "train_idx.csv")["ID"])
test_ids_paper = set(pd.read_csv(OBELIX_DATA / "test_idx.csv")["ID"])

TIERS = {
    "A_composition_only": magpie_cols,
    "B_composition_plus_crystal": magpie_cols + lattice_cols,
    "C_composition_crystal_voronoi": magpie_cols + lattice_cols + struct_cols,
}
X_by_tier = {}
y_struct = df_struct["log_IC"].to_numpy(dtype=float)
for tname, cols in TIERS.items():
    Xraw = df_struct[cols].to_numpy(dtype=float)
    Xf, _ = nan_fill(Xraw)
    X_by_tier[tname] = Xf

protos = protocol_masks_struct(df_struct, train_ids_paper, test_ids_paper)
results = {}

# ============================================================
# PART 0: feature inventory (dimensionality bug check)
# ============================================================
log("=" * 70); log("PART 0: feature inventory")
part0 = {
    "tier_A_composition_only": {"magpie": len(magpie_cols), "total": len(magpie_cols)},
    "tier_B_composition_plus_crystal": {
        "magpie": len(magpie_cols), "lattice_crystallographic": len(lattice_cols),
        "lattice_cols_list": lattice_cols, "total": len(magpie_cols) + len(lattice_cols)},
    "tier_C_full": {
        "magpie": len(magpie_cols), "lattice_crystallographic": len(lattice_cols),
        "structural_voronoi_coord": len(struct_cols),
        "total": len(magpie_cols) + len(lattice_cols) + len(struct_cols)},
    "missing_value_treatment": "per-column mean imputation, fit on training fold only in every protocol",
}
log(f"  Tier A: {part0['tier_A_composition_only']['total']}, "
    f"Tier B: {part0['tier_B_composition_plus_crystal']['total']}, "
    f"Tier C: {part0['tier_C_full']['total']}")
results["part0_feature_inventory"] = part0

# ============================================================
# PART 1: composition overlap check, leave-one-duplicate-group-out folds
# ============================================================
log("=" * 70); log("PART 1: composition overlap, duplicate-group folds")
part1 = []
for tr_m, te_m, gid in protos["leave_one_duplicate_group_out"]:
    held_out_comps = set(df_struct.loc[te_m, "True_Composition"])
    train_comps = set(df_struct.loc[tr_m, "True_Composition"])
    overlap = held_out_comps & train_comps
    part1.append({"group_id": str(gid), "n_held_out": int(te_m.sum()),
                   "held_out_compositions": sorted(held_out_comps),
                   "n_overlapping_compositions_in_train": len(overlap),
                   "overlapping_compositions": sorted(overlap)})
n_groups_with_overlap = sum(1 for g in part1 if g["n_overlapping_compositions_in_train"] > 0)
log(f"  {n_groups_with_overlap} of {len(part1)} duplicate-group folds have the SAME "
    f"True_Composition appearing elsewhere in that fold's training set")
results["part1_composition_overlap_dupgroups"] = {
    "n_folds": len(part1), "n_folds_with_composition_overlap": n_groups_with_overlap,
    "detail": part1,
}

# ============================================================
# PART 2: fold-specific feature selection (fixes the leakage concern)
# ============================================================
log("=" * 70); log("PART 2: fold-specific (no-leakage) feature selection")
cols_C = TIERS["C_composition_crystal_voronoi"]
struct_col_set = set(struct_cols)
col_idx_C = {c: i for i, c in enumerate(cols_C)}


def topn_struct_cols_from_fold(tr_m, n_top, n_estimators=200):
    """Rank structural descriptors by importance using ONLY this fold's training rows."""
    model = XGBRegressor(n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                          n_jobs=8, random_state=RNG)
    model.fit(X_by_tier["C_composition_crystal_voronoi"][tr_m], y_struct[tr_m], verbose=False)
    imp = model.feature_importances_
    struct_imp = [(c, float(imp[col_idx_C[c]])) for c in struct_cols]
    struct_imp.sort(key=lambda x: -x[1])
    return [c for c, _ in struct_imp[:n_top]]


part2 = {}
for topn in (10, 20):
    proto_res = {}
    for proto_name, fold_list in protos.items():
        all_errs = []
        for tr_m, te_m, label in fold_list:
            if tr_m.sum() == 0 or te_m.sum() == 0:
                continue
            # feature selection uses ONLY this fold's training rows -- no leakage
            # from the held-out family/space-group/duplicate-group into the ranking.
            top_cols = topn_struct_cols_from_fold(tr_m, topn)
            cols_bplus = TIERS["B_composition_plus_crystal"] + top_cols
            Xbp = df_struct[cols_bplus].to_numpy(dtype=float)
            Xbp, _ = nan_fill(Xbp)
            n_est = 200 if proto_name == "leave_one_duplicate_group_out" else 300
            _, pred = fit_xgb(Xbp, y_struct, tr_m, te_m, n_estimators=n_est)
            all_errs.extend(np.abs(y_struct[te_m] - pred).tolist())
        proto_res[proto_name] = float(np.mean(all_errs))
    part2[f"B_plus_top{topn}_struct_foldspecific"] = proto_res
    log(f"  Tier B + top{topn} struct (fold-specific selection): "
        + ", ".join(f"{p}={v:.3f}" for p, v in proto_res.items()))
results["part2_foldspecific_feature_selection"] = part2

# ============================================================
# PART 3: group-level / hierarchical bootstrap for Table 2 (Tier B vs Tier C)
# ============================================================
log("=" * 70); log("PART 3: group-level bootstrap, Tier B vs Tier C paired difference")
part3 = {}
rng_master = np.random.RandomState(RNG)
for proto_name, fold_list in protos.items():
    # Collect per-fold (= per held-out group) paired errors
    fold_records = []  # list of dict: label, true, predB, predC
    for tr_m, te_m, label in fold_list:
        if tr_m.sum() == 0 or te_m.sum() == 0:
            continue
        n_est = 200 if proto_name == "leave_one_duplicate_group_out" else 300
        _, pB = fit_xgb(X_by_tier["B_composition_plus_crystal"], y_struct, tr_m, te_m, n_estimators=n_est)
        _, pC = fit_xgb(X_by_tier["C_composition_crystal_voronoi"], y_struct, tr_m, te_m, n_estimators=n_est)
        true_v = y_struct[te_m]
        errB = np.abs(true_v - pB)
        errC = np.abs(true_v - pC)
        fold_records.append({"label": str(label), "diff": (errC - errB).tolist(), "n": int(te_m.sum())})

    if proto_name == "paper_split":
        # Only one fold -- cluster by Family within the test set instead, since
        # individual test observations are not independent at the family level.
        tr_m, te_m, _ = fold_list[0]
        fam_test = df_struct.loc[te_m, "Family"].fillna("__unknown__").to_numpy()
        _, pB = fit_xgb(X_by_tier["B_composition_plus_crystal"], y_struct, tr_m, te_m, n_estimators=300)
        _, pC = fit_xgb(X_by_tier["C_composition_crystal_voronoi"], y_struct, tr_m, te_m, n_estimators=300)
        true_v = y_struct[te_m]
        diff = np.abs(true_v - pC) - np.abs(true_v - pB)
        clusters = [diff[fam_test == f] for f in np.unique(fam_test)]
        clusters = [c for c in clusters if len(c) > 0]
        boots = []
        for _ in range(5000):
            chosen = [clusters[i] for i in rng_master.randint(0, len(clusters), size=len(clusters))]
            boots.append(np.concatenate(chosen).mean())
        boots = np.array(boots)
        mean_diff = diff.mean()
    else:
        # Resample whole held-out folds (families / space groups / duplicate groups)
        # with replacement, preserving each fold's internal observations intact.
        clusters = [np.array(r["diff"]) for r in fold_records]
        all_diff = np.concatenate([r["diff"] for r in fold_records])
        mean_diff = all_diff.mean()
        boots = []
        for _ in range(5000):
            chosen = [clusters[i] for i in rng_master.randint(0, len(clusters), size=len(clusters))]
            boots.append(np.concatenate(chosen).mean())
        boots = np.array(boots)

    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    significant = (ci[0] > 0) or (ci[1] < 0)
    part3[proto_name] = {
        "n_clusters": len(fold_records) if proto_name != "paper_split" else int(len(np.unique(fam_test))),
        "mean_diff_C_minus_B": float(mean_diff),
        "group_bootstrap_ci95": ci,
        "significant_at_group_level": bool(significant),
    }
    log(f"  {proto_name}: mean diff={mean_diff:+.3f}, group-bootstrap CI95={ci}, "
        f"significant={significant}")
results["part3_group_level_bootstrap"] = part3

# ============================================================
# PART 4: extended screening metrics (precision/recall/enrichment @ 10/20/30%)
# ============================================================
log("=" * 70); log("PART 4: extended screening metrics, Tier B paper split")
Xb_full = df[TIERS["B_composition_plus_crystal"]].to_numpy(dtype=float)
Xb_full, _ = nan_fill(Xb_full)
y_full = df["log_IC"].to_numpy(dtype=float)
tm_full = df["ID"].isin(train_ids_paper).to_numpy()
te_full = df["ID"].isin(test_ids_paper).to_numpy()
_, predB_paper = fit_xgb(Xb_full, y_full, tm_full, te_full, n_estimators=500)
y_test = y_full[te_full]
rho, rho_p = spearmanr(y_test, predB_paper)
n_test = len(y_test)

part4 = {"n_test": int(n_test), "spearman_rho": float(rho), "spearman_p": float(rho_p), "by_threshold": {}}
for pct in (10, 20, 30):
    k = max(1, int(round(pct / 100 * n_test)))
    true_top = set(np.argsort(-y_test)[:k].tolist())
    pred_top = set(np.argsort(-predB_paper)[:k].tolist())
    overlap = len(true_top & pred_top)
    precision = overlap / k  # = recall here since |pred_top| == |true_top| == k
    recall = overlap / k
    chance = k / n_test
    enrichment = (overlap / k) / chance if chance > 0 else None
    part4["by_threshold"][f"top_{pct}pct"] = {
        "k": int(k), "overlap": int(overlap), "precision": float(precision),
        "recall": float(recall), "enrichment_factor_vs_chance": float(enrichment),
    }
    log(f"  top-{pct}%: k={k}, overlap={overlap}, precision=recall={precision:.3f}, "
        f"enrichment={enrichment:.2f}x chance")
results["part4_screening_metrics"] = part4

# ============================================================
# PART 5: missingness-indicator ablation (92 imputed Li-site entries)
# ============================================================
log("=" * 70); log("PART 5: missingness-indicator ablation")
li_missing = (df_struct["n_li_sites"] == 0).to_numpy(dtype=float)
X_C_plus_flag = np.column_stack([X_by_tier["C_composition_crystal_voronoi"], li_missing])
part5 = {}
for proto_name, fold_list in protos.items():
    all_errs_base, all_errs_flag = [], []
    for tr_m, te_m, label in fold_list:
        if tr_m.sum() == 0 or te_m.sum() == 0:
            continue
        n_est = 200 if proto_name == "leave_one_duplicate_group_out" else 300
        _, p0 = fit_xgb(X_by_tier["C_composition_crystal_voronoi"], y_struct, tr_m, te_m, n_estimators=n_est)
        _, p1 = fit_xgb(X_C_plus_flag, y_struct, tr_m, te_m, n_estimators=n_est)
        all_errs_base.extend(np.abs(y_struct[te_m] - p0).tolist())
        all_errs_flag.extend(np.abs(y_struct[te_m] - p1).tolist())
    part5[proto_name] = {"tierC_mae": float(np.mean(all_errs_base)),
                          "tierC_plus_missingflag_mae": float(np.mean(all_errs_flag))}
    log(f"  {proto_name}: Tier C={np.mean(all_errs_base):.3f}, "
        f"Tier C + missingness flag={np.mean(all_errs_flag):.3f}")
results["part5_missingness_indicator"] = part5

# ============================================================
# PART 6: training-fold-only nested hyperparameter selection, Tier B and Tier C
# ============================================================
log("=" * 70); log("PART 6: training-fold-only nested hyperparameter selection")
grid = [
    {"n_estimators": 300, "max_depth": 5, "reg_lambda": 1.0},
    {"n_estimators": 300, "max_depth": 3, "reg_lambda": 1.0},
    {"n_estimators": 150, "max_depth": 3, "reg_lambda": 5.0},
    {"n_estimators": 150, "max_depth": 2, "reg_lambda": 10.0},
    {"n_estimators": 500, "max_depth": 3, "reg_lambda": 3.0},
]


def inner_cv_select(X, y, train_mask, grid, n_splits=3, seed=RNG):
    """Pick the grid config with lowest 3-fold CV MAE using ONLY the training rows."""
    idx = np.where(train_mask)[0]
    kf = KFold(n_splits=min(n_splits, len(idx)), shuffle=True, random_state=seed)
    best_cfg, best_mae = None, np.inf
    for cfg in grid:
        errs = []
        for in_idx, val_idx in kf.split(idx):
            tr_i, va_i = idx[in_idx], idx[val_idx]
            m = XGBRegressor(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                              learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                              reg_lambda=cfg["reg_lambda"], n_jobs=8, random_state=seed)
            m.fit(X[tr_i], y[tr_i], verbose=False)
            errs.extend(np.abs(y[va_i] - m.predict(X[va_i])).tolist())
        mae = float(np.mean(errs))
        if mae < best_mae:
            best_mae, best_cfg = mae, cfg
    return best_cfg, best_mae


part6 = {}
for tier_name in ("B_composition_plus_crystal", "C_composition_crystal_voronoi"):
    proto_res = {}
    for proto_name, fold_list in protos.items():
        all_errs_fixed, all_errs_tuned = [], []
        chosen_cfgs = []
        for tr_m, te_m, label in fold_list:
            if tr_m.sum() < 15 or te_m.sum() == 0:
                # too few training rows for a meaningful inner split; skip tuning,
                # use the fixed default for this fold only
                n_est = 200 if proto_name == "leave_one_duplicate_group_out" else 300
                _, pf = fit_xgb(X_by_tier[tier_name], y_struct, tr_m, te_m, n_estimators=n_est)
                all_errs_fixed.extend(np.abs(y_struct[te_m] - pf).tolist())
                all_errs_tuned.extend(np.abs(y_struct[te_m] - pf).tolist())
                continue
            n_est = 200 if proto_name == "leave_one_duplicate_group_out" else 300
            _, pf = fit_xgb(X_by_tier[tier_name], y_struct, tr_m, te_m, n_estimators=n_est)
            all_errs_fixed.extend(np.abs(y_struct[te_m] - pf).tolist())
            best_cfg, _ = inner_cv_select(X_by_tier[tier_name], y_struct, tr_m, grid)
            chosen_cfgs.append(best_cfg)
            _, pt = fit_xgb(X_by_tier[tier_name], y_struct, tr_m, te_m,
                             n_estimators=best_cfg["n_estimators"], max_depth=best_cfg["max_depth"],
                             reg_lambda=best_cfg["reg_lambda"])
            all_errs_tuned.extend(np.abs(y_struct[te_m] - pt).tolist())
        proto_res[proto_name] = {
            "mae_fixed_default": float(np.mean(all_errs_fixed)),
            "mae_training_fold_tuned": float(np.mean(all_errs_tuned)),
            "n_folds_tuned": len(chosen_cfgs),
        }
    part6[tier_name] = proto_res
    for p, v in proto_res.items():
        log(f"  {tier_name} / {p}: fixed={v['mae_fixed_default']:.3f} "
            f"tuned={v['mae_training_fold_tuned']:.3f} (n_folds_tuned={v['n_folds_tuned']})")
results["part6_nested_hyperparam_selection"] = part6

# recompute Tier B vs Tier C diff using the TUNED numbers, per protocol
part6_diff = {}
for proto_name in protos:
    bB = part6["B_composition_plus_crystal"][proto_name]["mae_training_fold_tuned"]
    bC = part6["C_composition_crystal_voronoi"][proto_name]["mae_training_fold_tuned"]
    part6_diff[proto_name] = {"tuned_mae_B": bB, "tuned_mae_C": bC, "tuned_diff_C_minus_B": bC - bB}
    log(f"  tuned diff {proto_name}: B={bB:.3f} C={bC:.3f} diff={bC-bB:+.3f}")
results["part6b_tuned_tierB_vs_tierC"] = part6_diff

# ============================================================
# PART 7: fixed-split vs grouped-CV -- high-error material overlap
# ============================================================
log("=" * 70); log("PART 7: high-error material comparison, fixed split vs grouped CV")
groups = df["True_Composition"].astype("category").cat.codes.to_numpy()
gkf = GroupKFold(n_splits=5)
all_grouped_errs = np.full(len(df), np.nan)
for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(Xb_full, y_full, groups)):
    tr_m = np.zeros(len(df), dtype=bool); tr_m[tr_idx] = True
    te_m = np.zeros(len(df), dtype=bool); te_m[te_idx] = True
    _, pred = fit_xgb(Xb_full, y_full, tr_m, te_m, n_estimators=500)
    all_grouped_errs[te_m] = np.abs(y_full[te_m] - pred)
grouped_err_series = pd.Series(all_grouped_errs, index=df.index)
fixed_err = np.full(len(df), np.nan)
fixed_err[te_full] = np.abs(y_full[te_full] - predB_paper)
fixed_err_series = pd.Series(fixed_err, index=df.index)

top_err_grouped = df.loc[grouped_err_series.nlargest(15).index, ["ID", "Family"]].assign(
    err=grouped_err_series.nlargest(15).values)
top_err_fixed = df.loc[fixed_err_series.dropna().nlargest(15).index, ["ID", "Family"]].assign(
    err=fixed_err_series.dropna().nlargest(15).values)
part7 = {
    "grouped_cv_mean_mae": float(np.nanmean(all_grouped_errs)),
    "fixed_split_mean_mae": float(np.nanmean(fixed_err)),
    "grouped_cv_family_share_of_top15_err": top_err_grouped["Family"].value_counts(normalize=True).round(3).to_dict(),
    "fixed_split_family_share_of_top15_err": top_err_fixed["Family"].value_counts(normalize=True).round(3).to_dict(),
}
log(f"  grouped-CV mean MAE={part7['grouped_cv_mean_mae']:.3f}, "
    f"fixed-split mean MAE={part7['fixed_split_mean_mae']:.3f}")
log(f"  grouped-CV top-15-error families: {part7['grouped_cv_family_share_of_top15_err']}")
log(f"  fixed-split top-15-error families: {part7['fixed_split_family_share_of_top15_err']}")
results["part7_high_error_material_comparison"] = part7

# ============================================================
with open(OUT_DIR / "all_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
log("ALL SECTIONS DONE.")
