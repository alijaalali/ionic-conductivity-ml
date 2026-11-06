# Supplementary checks reported in the manuscript and Supporting Information:
#   1. composition-grouped bootstrap of the screening enrichment factors (Tier B, paper split),
#   2. numeric versus one-hot space-group encoding (Tier B, 321-entry subset),
#   3. median nearest-neighbour distances (Euclidean, cosine, city-block) in standardized
#      Tier B feature space for LGPS and several comparison families.
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import DATA_DIR, OBELIX_DATA, RESULTS

warnings.filterwarnings("ignore")
OUT_DIR = RESULTS / "supplementary_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RNG = 42
MIN_ROWS_PER_GROUP = 8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


df = pd.read_parquet(DATA_DIR / "master_table.parquet")
magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
lattice_cols = ["a", "b", "c", "alpha", "beta", "gamma", "Space_group_number", "Z"]
train_ids = set(pd.read_csv(OBELIX_DATA / "train_idx.csv")["ID"])
test_ids = set(pd.read_csv(OBELIX_DATA / "test_idx.csv")["ID"])


def nan_fill(X):
    fill = np.nanmean(X, axis=0)
    return np.where(np.isnan(X), fill, X)


def fit_predict(X, y, tr, te, n_estimators=300):
    m = XGBRegressor(n_estimators=n_estimators, max_depth=5, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.7, reg_lambda=1.0, n_jobs=4, random_state=RNG)
    m.fit(X[tr], y[tr], verbose=False)
    return m.predict(X[te])


results = {}

# ---- 1. grouped bootstrap of enrichment --------------------------------------------
log("Part 1: enrichment bootstrap")
XB = nan_fill(df[magpie_cols + lattice_cols].to_numpy(dtype=float))
y = df["log_IC"].to_numpy(dtype=float)
tr = df["ID"].isin(train_ids).to_numpy()
te = df["ID"].isin(test_ids).to_numpy()
pred = fit_predict(XB, y, tr, te, n_estimators=500)
y_te = y[te]
groups = df.loc[te, "True_Composition"].to_numpy() if "True_Composition" in df.columns else np.arange(te.sum())
uniq = pd.unique(groups)
idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
rng = np.random.RandomState(RNG)
B = 5000
res = {p: [] for p in (10, 20, 30)}
for _ in range(B):
    pick = rng.choice(len(uniq), size=len(uniq), replace=True)
    idx = np.concatenate([idx_by_group[uniq[i]] for i in pick])
    yt, yp = y_te[idx], pred[idx]
    n = len(idx)
    for p in res:
        k = max(1, int(round(p / 100 * n)))
        overlap = len(set(np.argsort(-yt, kind="stable")[:k]) & set(np.argsort(-yp, kind="stable")[:k]))
        res[p].append((overlap / k) / (k / n))
part1 = {"n_groups": int(len(uniq))}
for p, v in res.items():
    v = np.array(v)
    part1[f"top_{p}pct"] = {"bootstrap_median": float(np.median(v)),
                            "bootstrap_ci95_grouped": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    log(f"  top-{p}%: median={np.median(v):.3f}, CI95=[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]")
results["enrichment_bootstrap"] = part1

# ---- 2. space-group encoding -------------------------------------------------------
log("Part 2: space-group encoding")
ds = df[df["has_structure"]].reset_index(drop=True)
ys = ds["log_IC"].to_numpy(dtype=float)
Xnum = nan_fill(ds[magpie_cols + lattice_cols].to_numpy(dtype=float))
sg_onehot = pd.get_dummies(ds["Space_group_number"].astype(int), prefix="sg").to_numpy(dtype=float)
lat_nosg = [c for c in lattice_cols if c != "Space_group_number"]
Xoh = np.column_stack([nan_fill(ds[magpie_cols + lat_nosg].to_numpy(dtype=float)), sg_onehot])
trs = ds["ID"].isin(train_ids).to_numpy()
tes = ds["ID"].isin(test_ids).to_numpy()
part2 = {"paper_split": {"n_test": int(tes.sum())}, "leave_one_duplicate_group_out": {}}
for name, X in (("numeric_sg", Xnum), ("onehot_sg", Xoh)):
    p = fit_predict(X, ys, trs, tes, n_estimators=300)
    part2["paper_split"][f"mae_{name}"] = float(np.mean(np.abs(p - ys[tes])))
errs = {"numeric_sg": [], "onehot_sg": []}
gids = ds.loc[ds["duplicate_group_size"] > 1, "duplicate_group_id"].unique()
n_groups = 0
for gid in gids:
    tm = (ds["duplicate_group_id"] == gid).to_numpy()
    if tm.sum() < 2:
        continue
    n_groups += 1
    for name, X in (("numeric_sg", Xnum), ("onehot_sg", Xoh)):
        p = fit_predict(X, ys, ~tm, tm, n_estimators=300)
        errs[name].extend(np.abs(p - ys[tm]).tolist())
part2["leave_one_duplicate_group_out"]["n_groups"] = n_groups
for name in errs:
    part2["leave_one_duplicate_group_out"][f"mae_{name}"] = float(np.mean(errs[name]))
log(f"  {json.dumps(part2)}")
results["space_group_encoding"] = part2

# ---- 3. nearest-neighbour distances -------------------------------------------------
log("Part 3: nearest-neighbour distances")
dfa = df.reset_index(drop=True)
Xb = nan_fill(dfa[magpie_cols + lattice_cols].to_numpy(dtype=float))
fam = dfa["Family"].to_numpy()
Xs = StandardScaler().fit(Xb[fam != "LGPS"]).transform(Xb)
part3 = {}
for metric in ("euclidean", "cosine", "cityblock"):
    part3[metric] = {}
    for f in ["LGPS", "thio-LISICON", "argyrodites", "oxides", "garnet"]:
        m = fam == f
        d = cdist(Xs[m], Xs[~m], metric=metric).min(axis=1)
        part3[metric][f] = {"n": int(m.sum()), "median_nn_distance": float(np.median(d))}
    log(f"  {metric}: " + ", ".join(f"{f}={v['median_nn_distance']:.3f}" for f, v in part3[metric].items()))
results["nn_distances"] = part3

with open(OUT_DIR / "all_results.json", "w") as fh:
    json.dump(results, fh, indent=2)
log("done")
