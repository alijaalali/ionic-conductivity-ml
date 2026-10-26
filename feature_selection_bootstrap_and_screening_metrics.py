# Tier B vs Tier C paired bootstrap per protocol, feature-selection/dimensionality
# experiment, Tier C hyperparameter sensitivity, composition-grouped-CV distributional
# check, ranking-usefulness metrics, LGPS distance (cosine) + PCA projection, and pinball
# loss for quantile-model selection. Builds on build_master_table.py's featurized table.
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path("/glade/work/ajalali/11 - Ionic-Conductivity-ML-project")
DATA_DIR = ROOT / "data"
OBELIX_DATA = Path("/glade/derecho/scratch/ajalali/obelix-data/OBELiX/data")
OUT_DIR = ROOT / "models" / "feature_selection_and_screening"
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


def metrics_dict(y_true, y_pred):
    return {"n": int(len(y_true)), "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None}


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
# PART 1: Tier B vs Tier C paired bootstrap, per protocol, with per-sample preds
# ============================================================
log("=" * 70); log("PART 1: Tier B vs Tier C paired bootstrap, all 4 protocols")
part1 = {}
for proto_name, fold_list in protos.items():
    true_all, predB_all, predC_all, id_all = [], [], [], []
    for tr_m, te_m, label in fold_list:
        if tr_m.sum() == 0 or te_m.sum() == 0:
            continue
        n_est = 200 if proto_name == "leave_one_duplicate_group_out" else 300
        _, pB = fit_xgb(X_by_tier["B_composition_plus_crystal"], y_struct, tr_m, te_m, n_estimators=n_est)
        _, pC = fit_xgb(X_by_tier["C_composition_crystal_voronoi"], y_struct, tr_m, te_m, n_estimators=n_est)
        true_all.extend(y_struct[te_m].tolist())
        predB_all.extend(pB.tolist())
        predC_all.extend(pC.tolist())
        id_all.extend(df_struct.loc[te_m, "ID"].tolist())
    true_all = np.array(true_all); predB_all = np.array(predB_all); predC_all = np.array(predC_all)
    errB = np.abs(true_all - predB_all)
    errC = np.abs(true_all - predC_all)
    diff = errC - errB  # positive = C worse than B
    rng = np.random.RandomState(RNG)
    boots = np.array([np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(5000)])
    part1[proto_name] = {
        "n": int(len(true_all)),
        "mae_B": float(errB.mean()), "mae_C": float(errC.mean()),
        "mean_diff_C_minus_B": float(diff.mean()),
        "diff_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }
    log(f"  {proto_name}: MAE_B={errB.mean():.3f} MAE_C={errC.mean():.3f} "
        f"diff={diff.mean():+.3f} CI95={part1[proto_name]['diff_ci95']}")
results["part1_tierB_vs_tierC_paired"] = part1

# ============================================================
# PART 2: feature-selection / dimensionality-reduction experiment
# ============================================================
log("=" * 70); log("PART 2: feature importance + Tier B + top-N Tier C descriptors")
tm = df_struct["ID"].isin(train_ids_paper).to_numpy()
te = df_struct["ID"].isin(test_ids_paper).to_numpy()
modelC, predC_full = fit_xgb(X_by_tier["C_composition_crystal_voronoi"], y_struct, tm, te, n_estimators=300)
cols_C = TIERS["C_composition_crystal_voronoi"]
importances = modelC.feature_importances_
struct_col_set = set(struct_cols)
struct_importance = [(c, float(imp)) for c, imp in zip(cols_C, importances) if c in struct_col_set]
struct_importance.sort(key=lambda x: -x[1])
top20_struct = [c for c, _ in struct_importance[:20]]
top10_struct = [c for c, _ in struct_importance[:10]]

part2 = {"top20_struct_features": struct_importance[:20]}
for topn, cols_extra in [("top10", top10_struct), ("top20", top20_struct)]:
    cols_bplus = TIERS["B_composition_plus_crystal"] + cols_extra
    Xbp = df_struct[cols_bplus].to_numpy(dtype=float)
    Xbp, _ = nan_fill(Xbp)
    proto_res = {}
    for proto_name, fold_list in protos.items():
        all_errs = []
        for tr_m, te_m, label in fold_list:
            if tr_m.sum() == 0 or te_m.sum() == 0:
                continue
            n_est = 200 if proto_name == "leave_one_duplicate_group_out" else 300
            _, pred = fit_xgb(Xbp, y_struct, tr_m, te_m, n_estimators=n_est)
            all_errs.extend(np.abs(y_struct[te_m] - pred).tolist())
        proto_res[proto_name] = float(np.mean(all_errs))
    part2[f"B_plus_{topn}_struct"] = proto_res
    log(f"  Tier B + {topn} struct features: " + ", ".join(f"{p}={v:.3f}" for p, v in proto_res.items()))
results["part2_feature_selection"] = part2

# ============================================================
# PART 3: Tier C hyperparameter sensitivity
# ============================================================
log("=" * 70); log("PART 3: Tier C hyperparameter sensitivity")
grid = [
    {"n_estimators": 300, "max_depth": 5, "reg_lambda": 1.0},   # current default
    {"n_estimators": 300, "max_depth": 3, "reg_lambda": 1.0},   # shallower
    {"n_estimators": 150, "max_depth": 3, "reg_lambda": 5.0},   # shallower + more reg
    {"n_estimators": 150, "max_depth": 2, "reg_lambda": 10.0},  # much more conservative
    {"n_estimators": 500, "max_depth": 3, "reg_lambda": 3.0},
]
part3 = {}
for cfg in grid:
    cfg_key = f"n{cfg['n_estimators']}_d{cfg['max_depth']}_l{cfg['reg_lambda']}"
    proto_res = {}
    for proto_name, fold_list in protos.items():
        all_errs = []
        for tr_m, te_m, label in fold_list:
            if tr_m.sum() == 0 or te_m.sum() == 0:
                continue
            n_est = cfg["n_estimators"]
            _, pred = fit_xgb(X_by_tier["C_composition_crystal_voronoi"], y_struct, tr_m, te_m,
                               n_estimators=n_est, max_depth=cfg["max_depth"], reg_lambda=cfg["reg_lambda"])
            all_errs.extend(np.abs(y_struct[te_m] - pred).tolist())
        proto_res[proto_name] = float(np.mean(all_errs))
    part3[cfg_key] = {"config": cfg, "mae_by_protocol": proto_res}
    log(f"  {cfg_key}: " + ", ".join(f"{p}={v:.3f}" for p, v in proto_res.items()))
results["part3_tierC_hyperparam_sensitivity"] = part3

# ============================================================
# PART 4: composition-grouped CV vs fixed split -- distributional comparison
# ============================================================
log("=" * 70); log("PART 4: composition-grouped CV vs fixed-split distributional comparison")
Xb_full = df[TIERS["B_composition_plus_crystal"]].to_numpy(dtype=float)
Xb_full, _ = nan_fill(Xb_full)
y_full = df["log_IC"].to_numpy(dtype=float)
tm_full = df["ID"].isin(train_ids_paper).to_numpy()
te_full = df["ID"].isin(test_ids_paper).to_numpy()
censored_mask_full = (df["log_IC"] == -15.0).to_numpy()

groups = df["True_Composition"].astype("category").cat.codes.to_numpy()
gkf = GroupKFold(n_splits=5)
fold_summaries = []
for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(Xb_full, y_full, groups)):
    te_m = np.zeros(len(df), dtype=bool); te_m[te_idx] = True
    fam_counts = df.loc[te_m, "Family"].value_counts(normalize=True).round(3).to_dict()
    fold_summaries.append({
        "fold": fold_i, "n_test": int(te_m.sum()),
        "target_mean": float(y_full[te_m].mean()), "target_std": float(y_full[te_m].std()),
        "target_min": float(y_full[te_m].min()), "target_max": float(y_full[te_m].max()),
        "n_censored": int((te_m & censored_mask_full).sum()),
        "family_share": fam_counts,
    })
part4 = {
    "fixed_split_test": {
        "n_test": int(te_full.sum()),
        "target_mean": float(y_full[te_full].mean()), "target_std": float(y_full[te_full].std()),
        "target_min": float(y_full[te_full].min()), "target_max": float(y_full[te_full].max()),
        "n_censored": int((te_full & censored_mask_full).sum()),
        "family_share": df.loc[te_full, "Family"].value_counts(normalize=True).round(3).to_dict(),
    },
    "grouped_cv_fold_summaries": fold_summaries,
}
log(f"  Fixed-split test: n={part4['fixed_split_test']['n_test']}, "
    f"target std={part4['fixed_split_test']['target_std']:.2f}, "
    f"n_censored={part4['fixed_split_test']['n_censored']}")
for fs in fold_summaries:
    log(f"  CV fold {fs['fold']}: n={fs['n_test']}, target std={fs['target_std']:.2f}, "
        f"n_censored={fs['n_censored']}")
results["part4_cv_vs_fixed_distribution"] = part4

# ============================================================
# PART 5: ranking-usefulness metrics (Spearman + top-10% enrichment), Tier B paper split
# ============================================================
log("=" * 70); log("PART 5: ranking metrics, Tier B paper split")
_, predB_paper = fit_xgb(Xb_full, y_full, tm_full, te_full, n_estimators=500)
y_test = y_full[te_full]
rho, rho_p = spearmanr(y_test, predB_paper)
n_test = len(y_test)
top_k = max(1, int(round(0.10 * n_test)))
true_top_idx = set(np.argsort(-y_test)[:top_k].tolist())
pred_top_idx = set(np.argsort(-predB_paper)[:top_k].tolist())
enrichment = len(true_top_idx & pred_top_idx) / top_k
part5 = {
    "n_test": int(n_test), "spearman_rho": float(rho), "spearman_p": float(rho_p),
    "top_k": int(top_k), "top10pct_enrichment_fraction": float(enrichment),
}
log(f"  Spearman rho={rho:.3f} (p={rho_p:.2e}); top-10% enrichment: "
    f"{enrichment:.2f} of true top-{top_k} captured in predicted top-{top_k}")
results["part5_ranking_metrics"] = part5

# ============================================================
# PART 6: LGPS alternate distance metric (cosine) + PCA 2D projection coords
# ============================================================
log("=" * 70); log("PART 6: LGPS alternate distance metrics")
Xb_std_fit = StandardScaler().fit(Xb_full[~(df["Family"] == "LGPS").to_numpy()])
Xb_std = Xb_std_fit.transform(Xb_full)
fam_arr = df["Family"].to_numpy()
part6 = {}
for metric in ["euclidean", "cosine", "cityblock"]:
    row = {}
    for fam in ["LGPS", "thio-LISICON"]:
        mask = fam_arr == fam
        d = cdist(Xb_std[mask], Xb_std[~mask], metric=metric)
        nn = d.min(axis=1)
        row[fam] = {"median_nn_distance": float(np.median(nn)), "n": int(mask.sum())}
    part6[metric] = row
    log(f"  metric={metric}: LGPS median NN={row['LGPS']['median_nn_distance']:.3f}, "
        f"thio-LISICON median NN={row['thio-LISICON']['median_nn_distance']:.3f}")

# PCA 2D for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=RNG).fit(Xb_std)
coords = pca.transform(Xb_std)
part6["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
np.save(OUT_DIR / "pca_coords.npy", coords)
pd.DataFrame({"family": fam_arr, "pc1": coords[:, 0], "pc2": coords[:, 1]}).to_csv(
    OUT_DIR / "pca_coords.csv", index=False)
log(f"  PCA explained variance ratio (2D): {part6['pca_explained_variance_ratio']}")
results["part6_lgps_alt_distance"] = part6

# ============================================================
# PART 7: pinball loss for quantile model selection (Table S3 ask)
# ============================================================
log("=" * 70); log("PART 7: pinball loss for the quantile-regression sweep config")
df_unc = df[df.log_IC != -15.0].reset_index(drop=True)
Xu = df_unc[TIERS["B_composition_plus_crystal"]].to_numpy(dtype=float)
Xu, _ = nan_fill(Xu)
yu = df_unc["log_IC"].to_numpy(dtype=float)
tm_u = df_unc["ID"].isin(train_ids_paper).to_numpy()
te_u = df_unc["ID"].isin(test_ids_paper).to_numpy()

rng = np.random.RandomState(RNG)
train_idx_all = np.where(tm_u)[0]
rng.shuffle(train_idx_all)
n_val = int(0.2 * len(train_idx_all))
val_idx = train_idx_all[:n_val]
subtrain_idx = train_idx_all[n_val:]
tr2 = np.zeros(len(df_unc), dtype=bool); tr2[subtrain_idx] = True
va2 = np.zeros(len(df_unc), dtype=bool); va2[val_idx] = True


def make_q(alpha, n_estimators, max_depth, reg_lambda, seed=RNG):
    return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                         n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.7, reg_lambda=reg_lambda,
                         n_jobs=8, random_state=seed)


def pinball(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


grid_q = [
    {"n_estimators": 30, "max_depth": 2, "reg_lambda": 20.0},
    {"n_estimators": 30, "max_depth": 2, "reg_lambda": 10.0},
    {"n_estimators": 50, "max_depth": 2, "reg_lambda": 20.0},
    {"n_estimators": 30, "max_depth": 3, "reg_lambda": 20.0},
    {"n_estimators": 100, "max_depth": 2, "reg_lambda": 30.0},
]
part7 = []
for cfg in grid_q:
    m10 = make_q(0.1, **cfg); m10.fit(Xu[tr2], yu[tr2])
    m90 = make_q(0.9, **cfg); m90.fit(Xu[tr2], yu[tr2])
    p10v = m10.predict(Xu[va2]); p90v = m90.predict(Xu[va2])
    cov = float(np.mean((yu[va2] >= p10v) & (yu[va2] <= p90v)))
    width = float((p90v - p10v).mean())
    pb10 = pinball(yu[va2], p10v, 0.1)
    pb90 = pinball(yu[va2], p90v, 0.9)
    part7.append({**cfg, "val_coverage": cov, "val_abs_dev_from_80": abs(cov - 0.8),
                  "val_mean_interval_width": width, "val_pinball_q10": pb10, "val_pinball_q90": pb90,
                  "val_pinball_mean": (pb10 + pb90) / 2})
    log(f"  cfg={cfg} coverage={cov:.3f} width={width:.2f} pinball_mean={(pb10+pb90)/2:.3f}")
results["part7_quantile_pinball_sweep"] = part7

# ============================================================
with open(OUT_DIR / "all_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
log("ALL SECTIONS DONE.")
