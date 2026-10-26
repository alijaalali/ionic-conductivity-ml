# Three-tier feature comparison (composition-only / +crystallography / +Voronoi-coordination)
# across four evaluation protocols with paired bootstrap, noise-floor reanalysis, censored-
# measurement sensitivity, train/test leakage check with composition-grouped CV, honest
# quantile-regression calibration, LGPS feature-space-distance diagnostic, and a lithium-site
# imputation ablation. Re-analysis of the existing featurized table; no new DFT calculation.
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path("/glade/work/ajalali/11 - Ionic-Conductivity-ML-project")
DATA_DIR = ROOT / "data"
OBELIX_DATA = Path("/glade/derecho/scratch/ajalali/obelix-data/OBELiX/data")
OUT_DIR = ROOT / "models" / "tier_comparison_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ROWS_PER_GROUP = 8
PAPER_NOISE_RMSD = 0.63
PAPER_NOISE_MAD = 0.41
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


def fit_xgb(X, y, train_mask, test_mask, n_estimators=300, max_depth=5, seed=RNG):
    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        n_jobs=8, random_state=seed,
    )
    model.fit(X[train_mask], y[train_mask], verbose=False)
    pred = model.predict(X[test_mask])
    return model, pred


def metrics_dict(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }


def bootstrap_mae_ci(errs, n_boot=2000, seed=RNG):
    errs = np.asarray(errs)
    rng = np.random.RandomState(seed)
    boots = np.array([np.mean(rng.choice(errs, size=len(errs), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def nan_fill(X, fill=None):
    if fill is None:
        fill = np.nanmean(X, axis=0)
    return np.where(np.isnan(X), fill, X), fill


results = {}

# ============================================================
log("Loading data")
df, magpie_cols, lattice_cols, struct_cols = load()
df_struct = df[df["has_structure"]].reset_index(drop=True)
log(f"Full n={len(df)}, structure-resolved n={len(df_struct)}")

train_ids_paper = set(pd.read_csv(OBELIX_DATA / "train_idx.csv")["ID"])
test_ids_paper = set(pd.read_csv(OBELIX_DATA / "test_idx.csv")["ID"])

TIERS = {
    "A_composition_only": magpie_cols,
    "B_composition_plus_crystal": magpie_cols + lattice_cols,
    "C_composition_crystal_voronoi": magpie_cols + lattice_cols + struct_cols,
}

# ============================================================
# SECTION 1: three-tier comparison on the matched 321-subset
# ============================================================
log("=" * 70); log("SECTION 1: three-tier matched comparison (321-entry subset)")
section1 = {}


def protocol_masks_struct(dfx):
    """Return dict protocol_name -> list of (train_mask, test_mask, label)."""
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


X_by_tier = {}
y_struct = df_struct["log_IC"].to_numpy(dtype=float)
for tname, cols in TIERS.items():
    Xraw = df_struct[cols].to_numpy(dtype=float)
    Xf, _ = nan_fill(Xraw)
    X_by_tier[tname] = Xf

protos = protocol_masks_struct(df_struct)

for tname in TIERS:
    X = X_by_tier[tname]
    tier_res = {}
    for proto_name, fold_list in protos.items():
        fold_res = {}
        all_errs = []
        all_true, all_pred = [], []
        for tr_m, te_m, label in fold_list:
            if tr_m.sum() == 0 or te_m.sum() == 0:
                continue
            n_est = 200 if proto_name == "leave_one_duplicate_group_out" else 300
            _, pred = fit_xgb(X, y_struct, tr_m, te_m, n_estimators=n_est)
            m = metrics_dict(y_struct[te_m], pred)
            fold_res[str(label)] = m
            all_errs.extend(np.abs(y_struct[te_m] - pred).tolist())
            all_true.extend(y_struct[te_m].tolist())
            all_pred.extend(pred.tolist())
        if all_errs:
            lo, hi = bootstrap_mae_ci(all_errs)
            tier_res[proto_name] = {
                "per_fold": fold_res,
                "pooled_mae": float(np.mean(all_errs)),
                "pooled_mae_ci95": [lo, hi],
                "pooled_rmse": float(np.sqrt(np.mean(np.square(np.array(all_true) - np.array(all_pred))))),
                "pooled_r2": float(r2_score(all_true, all_pred)) if len(all_true) > 1 else None,
                "n_folds": len(fold_res),
            }
    section1[tname] = tier_res
    log(f"  {tname}: " + ", ".join(
        f"{p}={section1[tname][p]['pooled_mae']:.3f}" for p in section1[tname]))

results["section1_three_tier_matched"] = section1

# Paired bootstrap difference: tier A vs tier C on paper_split (matched test set)
log("Paired bootstrap: tier A vs tier C, paper split (matched 321-subset)")
tm = df_struct["ID"].isin(train_ids_paper).to_numpy()
te = df_struct["ID"].isin(test_ids_paper).to_numpy()
if tm.sum() and te.sum():
    _, predA = fit_xgb(X_by_tier["A_composition_only"], y_struct, tm, te, n_estimators=300)
    _, predC = fit_xgb(X_by_tier["C_composition_crystal_voronoi"], y_struct, tm, te, n_estimators=300)
    errA = np.abs(y_struct[te] - predA)
    errC = np.abs(y_struct[te] - predC)
    diff = errC - errA  # positive = structure tier worse
    rng = np.random.RandomState(RNG)
    boots = np.array([np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(5000)])
    results["section1_paired_AvC_paper_split"] = {
        "n_test": int(te.sum()),
        "mae_A": float(errA.mean()), "mae_C": float(errC.mean()),
        "mean_diff_C_minus_A": float(diff.mean()),
        "diff_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }
    log(f"  MAE A={errA.mean():.3f} C={errC.mean():.3f} diff={diff.mean():.3f} "
        f"CI95={results['section1_paired_AvC_paper_split']['diff_ci95']}")

with open(OUT_DIR / "section1_results.json", "w") as f:
    json.dump(results["section1_three_tier_matched"], f, indent=2, default=str)

log("Section 1 done, saved intermediate.")

# ============================================================
# SECTION 2: noise-floor reanalysis (full 599-entry composition-only model)
# ============================================================
log("=" * 70); log("SECTION 2: noise-floor reanalysis")
Xa_full = df[TIERS["A_composition_only"]].to_numpy(dtype=float)
Xa_full, _ = nan_fill(Xa_full)
Xb_full = df[TIERS["B_composition_plus_crystal"]].to_numpy(dtype=float)
Xb_full, _ = nan_fill(Xb_full)
y_full = df["log_IC"].to_numpy(dtype=float)

dup_groups_full = df[df["duplicate_group_size"] > 1]["duplicate_group_id"].unique()
group_maes, group_rmses, group_ns = [], [], []
obs_abs_errs, obs_sq_errs = [], []
for gid in dup_groups_full:
    te_m = (df["duplicate_group_id"] == gid).to_numpy()
    if te_m.sum() < 2:
        continue
    tr_m = ~te_m
    _, pred = fit_xgb(Xb_full, y_full, tr_m, te_m, n_estimators=200)  # tier B = current paper's "composition-only" (includes lattice)
    err = np.abs(y_full[te_m] - pred)
    sqerr = np.square(y_full[te_m] - pred)
    group_maes.append(float(err.mean()))
    group_rmses.append(float(np.sqrt(sqerr.mean())))
    group_ns.append(int(te_m.sum()))
    obs_abs_errs.extend(err.tolist())
    obs_sq_errs.extend(sqerr.tolist())

group_mae_arr = np.array(group_maes)
rng = np.random.RandomState(RNG)
boot_group_means = np.array([np.mean(rng.choice(group_mae_arr, size=len(group_mae_arr), replace=True))
                              for _ in range(5000)])

noise_floor = {
    "n_groups": len(group_maes),
    "group_weighted_mean_mae": float(np.mean(group_maes)),
    "group_weighted_mean_mae_ci95": [float(np.percentile(boot_group_means, 2.5)),
                                      float(np.percentile(boot_group_means, 97.5))],
    "group_weighted_median_mae": float(np.median(group_maes)),
    "group_weighted_mean_rmse": float(np.mean(group_rmses)),
    "observation_weighted_mae": float(np.mean(obs_abs_errs)),
    "observation_weighted_rmse": float(np.sqrt(np.mean(obs_sq_errs))),
    "paper_reported_noise_rmsd": PAPER_NOISE_RMSD,
    "paper_reported_noise_mad": PAPER_NOISE_MAD,
    "correct_comparison_MAE_vs_MAD": {
        "model_group_weighted_MAE": float(np.mean(group_maes)),
        "experimental_MAD": PAPER_NOISE_MAD,
    },
    "correct_comparison_RMSE_vs_RMSD": {
        "model_group_weighted_RMSE": float(np.mean(group_rmses)),
        "experimental_RMSD": PAPER_NOISE_RMSD,
    },
}
log(f"  Group-weighted MAE={noise_floor['group_weighted_mean_mae']:.3f} "
    f"(CI {noise_floor['group_weighted_mean_mae_ci95']}) vs experimental MAD={PAPER_NOISE_MAD}")
log(f"  Group-weighted RMSE={noise_floor['group_weighted_mean_rmse']:.3f} vs experimental RMSD={PAPER_NOISE_RMSD}")
log(f"  Observation-weighted MAE={noise_floor['observation_weighted_mae']:.3f}")

# Is the duplicate-group subset systematically easier? Compare family mix.
dup_member_mask = (df["duplicate_group_size"] > 1).to_numpy()
fam_dup = df.loc[dup_member_mask, "Family"].value_counts(normalize=True)
fam_all = df["Family"].value_counts(normalize=True)
noise_floor["duplicate_subset_family_share"] = fam_dup.to_dict()
noise_floor["full_dataset_family_share"] = fam_all.to_dict()

# direct difficulty check: restrict paper-split test set to dup-group members vs not
tm = df["ID"].isin(train_ids_paper).to_numpy()
te = df["ID"].isin(test_ids_paper).to_numpy()
_, pred_full = fit_xgb(Xb_full, y_full, tm, te, n_estimators=500)
te_dup = te & dup_member_mask
te_nondup = te & ~dup_member_mask
if te_dup.sum() > 1 and te_nondup.sum() > 1:
    idx_te = np.where(te)[0]
    pred_map = dict(zip(idx_te, pred_full))
    mae_dup = float(np.mean(np.abs(y_full[te_dup] - np.array([pred_map[i] for i in np.where(te_dup)[0]]))))
    mae_nondup = float(np.mean(np.abs(y_full[te_nondup] - np.array([pred_map[i] for i in np.where(te_nondup)[0]]))))
    noise_floor["paper_split_test_MAE_dupgroup_members"] = mae_dup
    noise_floor["paper_split_test_MAE_non_dupgroup_members"] = mae_nondup
    log(f"  Paper-split test MAE: dup-group members={mae_dup:.3f} (n={te_dup.sum()}), "
        f"non-members={mae_nondup:.3f} (n={te_nondup.sum()})")

results["section2_noise_floor"] = noise_floor
with open(OUT_DIR / "section2_results.json", "w") as f:
    json.dump(noise_floor, f, indent=2, default=str)
log("Section 2 done.")

# ============================================================
# SECTION 3: censored-measurement sensitivity
# ============================================================
log("=" * 70); log("SECTION 3: censored (-15 placeholder) sensitivity")
censored_mask = (df["log_IC"] == -15.0).to_numpy()
log(f"  {censored_mask.sum()} censored entries of {len(df)}")

tm = df["ID"].isin(train_ids_paper).to_numpy()
te = df["ID"].isin(test_ids_paper).to_numpy()
_, pred_with = fit_xgb(Xb_full, y_full, tm, te, n_estimators=500)
m_with = metrics_dict(y_full[te], pred_with)

keep = ~censored_mask
Xb_f = Xb_full[keep]
y_f = y_full[keep]
tm_f = tm[keep]
te_f = te[keep]
_, pred_without = fit_xgb(Xb_f, y_f, tm_f, te_f, n_estimators=500)
m_without = metrics_dict(y_f[te_f], pred_without)

censored_sensitivity = {
    "with_censored": m_with,
    "without_censored": m_without,
    "n_censored_in_test": int((te & censored_mask).sum()),
    "n_censored_in_train": int((tm & censored_mask).sum()),
}
log(f"  With censored: {m_with}")
log(f"  Without censored: {m_without}")
results["section3_censored_sensitivity"] = censored_sensitivity
with open(OUT_DIR / "section3_results.json", "w") as f:
    json.dump(censored_sensitivity, f, indent=2, default=str)
log("Section 3 done.")

# ============================================================
# SECTION 4: leakage check + composition-grouped CV
# ============================================================
log("=" * 70); log("SECTION 4: leakage check + composition-grouped CV")
train_comps = set(df.loc[tm, "True_Composition"])
test_comps = set(df.loc[te, "True_Composition"])
shared_comps = train_comps & test_comps
train_groups = set(df.loc[tm, "duplicate_group_id"])
test_groups = set(df.loc[te, "duplicate_group_id"])
shared_groups = train_groups & test_groups

leakage = {
    "paper_split_shared_compositions": len(shared_comps),
    "paper_split_shared_duplicate_groups": len(shared_groups),
    "paper_split_n_train": int(tm.sum()), "paper_split_n_test": int(te.sum()),
}
log(f"  Paper split: {len(shared_comps)} compositions and {len(shared_groups)} duplicate-groups "
    f"appear on both sides of train/test.")

# composition-grouped 5-fold CV (all rows of a composition forced into same fold)
groups = df["True_Composition"].astype("category").cat.codes.to_numpy()
gkf = GroupKFold(n_splits=5)
fold_metrics = []
for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(Xb_full, y_full, groups)):
    tr_m = np.zeros(len(df), dtype=bool); tr_m[tr_idx] = True
    te_m = np.zeros(len(df), dtype=bool); te_m[te_idx] = True
    _, pred = fit_xgb(Xb_full, y_full, tr_m, te_m, n_estimators=300)
    fold_metrics.append(metrics_dict(y_full[te_m], pred))
    log(f"    composition-grouped fold {fold_i}: {fold_metrics[-1]}")

leakage["composition_grouped_cv_folds"] = fold_metrics
leakage["composition_grouped_cv_mean_mae"] = float(np.mean([m["mae"] for m in fold_metrics]))
leakage["composition_grouped_cv_mean_rmse"] = float(np.mean([m["rmse"] for m in fold_metrics]))
log(f"  Composition-grouped CV mean MAE: {leakage['composition_grouped_cv_mean_mae']:.3f}")

results["section4_leakage"] = leakage
with open(OUT_DIR / "section4_results.json", "w") as f:
    json.dump(leakage, f, indent=2, default=str)
log("Section 4 done.")

# ============================================================
# SECTION 5: honest (non-leaked) quantile-regression hyperparameters + UQ
# ============================================================
log("=" * 70); log("SECTION 5: honest quantile hyperparameter selection + UQ validation")

df_unc = df[df.log_IC != -15.0].reset_index(drop=True)
Xu = df_unc[TIERS["B_composition_plus_crystal"]].to_numpy(dtype=float)
Xu, _ = nan_fill(Xu)
yu = df_unc["log_IC"].to_numpy(dtype=float)
tm_u = df_unc["ID"].isin(train_ids_paper).to_numpy()
te_u = df_unc["ID"].isin(test_ids_paper).to_numpy()

# carve an internal validation set out of the TRAINING data only (80/20),
# sweep hyperparameters there, then refit on full training and evaluate
# ONLY ONCE on the untouched paper-split test set.
rng = np.random.RandomState(RNG)
train_idx_all = np.where(tm_u)[0]
rng.shuffle(train_idx_all)
n_val = int(0.2 * len(train_idx_all))
val_idx = train_idx_all[:n_val]
subtrain_idx = train_idx_all[n_val:]
tr2 = np.zeros(len(df_unc), dtype=bool); tr2[subtrain_idx] = True
va2 = np.zeros(len(df_unc), dtype=bool); va2[val_idx] = True


def make_q(alpha, n_estimators, max_depth, reg_lambda, seed=RNG):
    return XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=alpha,
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=reg_lambda,
        n_jobs=8, random_state=seed,
    )


grid = [
    {"n_estimators": 30, "max_depth": 2, "reg_lambda": 20.0},
    {"n_estimators": 30, "max_depth": 2, "reg_lambda": 10.0},
    {"n_estimators": 50, "max_depth": 2, "reg_lambda": 20.0},
    {"n_estimators": 30, "max_depth": 3, "reg_lambda": 20.0},
    {"n_estimators": 100, "max_depth": 2, "reg_lambda": 30.0},
]
sweep_results = []
for cfg in grid:
    m10 = make_q(0.1, **cfg); m10.fit(Xu[tr2], yu[tr2])
    m90 = make_q(0.9, **cfg); m90.fit(Xu[tr2], yu[tr2])
    p10v = m10.predict(Xu[va2]); p90v = m90.predict(Xu[va2])
    cov = float(np.mean((yu[va2] >= p10v) & (yu[va2] <= p90v)))
    sweep_results.append({**cfg, "val_coverage": cov, "val_abs_dev_from_80": abs(cov - 0.8)})
    log(f"  sweep cfg={cfg} val_coverage={cov:.3f}")

best_cfg = min(sweep_results, key=lambda r: r["val_abs_dev_from_80"])
log(f"  Best config selected on validation only: {best_cfg}")

final_cfg = {k: best_cfg[k] for k in ("n_estimators", "max_depth", "reg_lambda")}
m10f = make_q(0.1, **final_cfg); m10f.fit(Xu[tm_u], yu[tm_u])
m90f = make_q(0.9, **final_cfg); m90f.fit(Xu[tm_u], yu[tm_u])
p10t = m10f.predict(Xu[te_u]); p90t = m90f.predict(Xu[te_u])
honest_coverage = float(np.mean((yu[te_u] >= p10t) & (yu[te_u] <= p90t)))
n_crossed = int(np.sum(p10t > p90t))
log(f"  HONEST (validation-selected, never-touched-test) paper-split coverage: {honest_coverage:.3f} "
    f"(previous leaked number was 0.823); quantile crossings on test: {n_crossed}")

# coverage by conductivity range (deciles of true value) + bootstrap CI
order = np.argsort(yu[te_u])
true_sorted = yu[te_u][order]
covered = ((yu[te_u] >= p10t) & (yu[te_u] <= p90t))
n_bins = 4
bin_edges = np.quantile(yu[te_u], np.linspace(0, 1, n_bins + 1))
range_cov = {}
for i in range(n_bins):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    sel = (yu[te_u] >= lo) & (yu[te_u] <= hi)
    if sel.sum() == 0:
        continue
    c = covered[sel]
    rng2 = np.random.RandomState(RNG)
    boots = np.array([np.mean(rng2.choice(c, size=len(c), replace=True)) for _ in range(2000)]) if len(c) > 1 else np.array([c.mean()])
    range_cov[f"bin_{i}_[{lo:.2f},{hi:.2f}]"] = {
        "n": int(sel.sum()), "coverage": float(c.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }

# interval score (Gneiting-Raftery), alpha=0.2 for 80% interval
alpha_is = 0.2
width = p90t - p10t
below = yu[te_u] < p10t
above = yu[te_u] > p90t
interval_score = width + (2 / alpha_is) * (p10t - yu[te_u]) * below + (2 / alpha_is) * (yu[te_u] - p90t) * above

uq_results = {
    "sweep": sweep_results,
    "selected_config": final_cfg,
    "honest_paper_split_coverage": honest_coverage,
    "previous_leaked_coverage_for_comparison": 0.823,
    "n_quantile_crossings_on_test": n_crossed,
    "mean_interval_width": float(width.mean()),
    "mean_interval_score": float(interval_score.mean()),
    "coverage_by_conductivity_range": range_cov,
}

# coverage by family (leave-one-family-out, using the honestly-selected config)
fam_cov = {}
for fam, count in df_unc["Family"].value_counts().items():
    if count < MIN_ROWS_PER_GROUP:
        continue
    te_m = (df_unc["Family"] == fam).to_numpy()
    tr_m = df_unc["Family"].notna().to_numpy() & ~te_m
    m10g = make_q(0.1, **final_cfg); m10g.fit(Xu[tr_m], yu[tr_m])
    m90g = make_q(0.9, **final_cfg); m90g.fit(Xu[tr_m], yu[tr_m])
    p10g = m10g.predict(Xu[te_m]); p90g = m90g.predict(Xu[te_m])
    cov = float(np.mean((yu[te_m] >= p10g) & (yu[te_m] <= p90g)))
    w = float((p90g - p10g).mean())
    crossed = int(np.sum(p10g > p90g))
    fam_cov[fam] = {"n": int(te_m.sum()), "coverage": cov, "mean_width": w, "n_crossed": crossed}
    log(f"    LOFO UQ {fam}: coverage={cov:.3f} width={w:.2f}")

uq_results["coverage_by_family_honest_config"] = fam_cov
results["section5_uq_honest"] = uq_results
with open(OUT_DIR / "section5_results.json", "w") as f:
    json.dump(uq_results, f, indent=2, default=str)
log("Section 5 done.")

# ============================================================
# SECTION 6: LGPS vs thio-LISICON feature-space distance
# ============================================================
log("=" * 70); log("SECTION 6: LGPS feature-space distance diagnostic")
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

Xb_std_fit = StandardScaler().fit(Xb_full[~(df["Family"] == "LGPS").to_numpy()])
Xb_std = Xb_std_fit.transform(Xb_full)

fam_arr = df["Family"].to_numpy()
distance_diag = {}
for fam in ["LGPS", "thio-LISICON"]:
    mask = fam_arr == fam
    train_mask_f = ~mask
    d = cdist(Xb_std[mask], Xb_std[train_mask_f], metric="euclidean")
    nn_dist = d.min(axis=1)
    distance_diag[fam] = {
        "n": int(mask.sum()),
        "median_nn_distance_to_rest": float(np.median(nn_dist)),
        "mean_nn_distance_to_rest": float(np.mean(nn_dist)),
    }
    log(f"  {fam}: median NN distance to rest of dataset = {np.median(nn_dist):.2f} (n={mask.sum()})")

results["section6_lgps_distance"] = distance_diag
with open(OUT_DIR / "section6_results.json", "w") as f:
    json.dump(distance_diag, f, indent=2, default=str)
log("Section 6 done.")

# ============================================================
# SECTION 7: lithium-site-imputation ablation
# ============================================================
log("=" * 70); log("SECTION 7: Li-site imputation ablation (drop 92 zero-Li-site entries)")
keep_mask = (df_struct["n_li_sites"] > 0).to_numpy()
log(f"  Keeping {keep_mask.sum()} of {len(df_struct)} structure-resolved entries "
    f"(dropping {(~keep_mask).sum()} zero-Li-site imputed entries)")

df_abl = df_struct[keep_mask].reset_index(drop=True)
y_abl = df_abl["log_IC"].to_numpy(dtype=float)
Xc_abl = df_abl[TIERS["C_composition_crystal_voronoi"]].to_numpy(dtype=float)
Xc_abl, _ = nan_fill(Xc_abl)
Xa_abl = df_abl[TIERS["A_composition_only"]].to_numpy(dtype=float)
Xa_abl, _ = nan_fill(Xa_abl)

protos_abl = protocol_masks_struct(df_abl)
ablation = {}
for tname, X in [("A_composition_only", Xa_abl), ("C_composition_crystal_voronoi", Xc_abl)]:
    tier_res = {}
    for proto_name, fold_list in protos_abl.items():
        all_errs = []
        for tr_m, te_m, label in fold_list:
            if tr_m.sum() == 0 or te_m.sum() == 0:
                continue
            n_est = 200 if proto_name == "leave_one_duplicate_group_out" else 300
            _, pred = fit_xgb(X, y_abl, tr_m, te_m, n_estimators=n_est)
            all_errs.extend(np.abs(y_abl[te_m] - pred).tolist())
        if all_errs:
            tier_res[proto_name] = {"pooled_mae": float(np.mean(all_errs)), "n": len(all_errs)}
    ablation[tname] = tier_res
    log(f"  {tname} (Li-site-present subset only): " +
        ", ".join(f"{p}={ablation[tname][p]['pooled_mae']:.3f}" for p in ablation[tname]))

results["section7_imputation_ablation"] = ablation
with open(OUT_DIR / "section7_results.json", "w") as f:
    json.dump(ablation, f, indent=2, default=str)
log("Section 7 done.")

# ============================================================
with open(OUT_DIR / "all_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
log("ALL SECTIONS DONE.")
