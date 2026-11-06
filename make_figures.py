# Main-text and SI figures for the OBELiX ionic-conductivity manuscript: dataset overview,
# parity plot, leave-one-family-out MAE, uncertainty calibration, duplicate-group and space-group figures.
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR, FIG_DIR, RESULTS
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

LGPS_COLOR = "#d62728"
DEFAULT_COLOR = "#4c72b0"
STRUCT_COLOR = "#dd8452"

def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {name}")

df = pd.read_parquet(DATA_DIR / "master_table.parquet")
xgb_metrics = json.load(open(RESULTS / "xgb_baseline" / "metrics.json"))
unc_metrics = json.load(open(RESULTS / "uncertainty" / "metrics.json"))
pred_comp = pd.read_csv(RESULTS / "xgb_baseline" / "predictions_composition_only.csv")
pred_struct = pd.read_csv(RESULTS / "xgb_baseline" / "predictions_composition_plus_structure.csv")
pred_unc = pd.read_csv(RESULTS / "uncertainty" / "predictions_paper_split.csv")

MAIN_FAMILIES = ["NASICON", "garnet", "perovskites", "argyrodites", "LGPS",
                  "oxides", "thio-LISICON", "LISICON", "sulfides", "phosphates", "halides"]

fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7))

fam_counts = df["Family"].value_counts()
top_fam = fam_counts.head(11)
colors = [LGPS_COLOR if f == "LGPS" else DEFAULT_COLOR for f in top_fam.index]
axes[0].barh(top_fam.index[::-1], top_fam.values[::-1], color=colors[::-1])
axes[0].set_xlabel(r"$N$ [entries]")
axes[0].set_title("(a) Crystal families of OBELiX")

axes[1].hist(df["log_IC"], bins=40, color=DEFAULT_COLOR, edgecolor="white", linewidth=0.4)
lgps_vals = df.loc[df["Family"] == "LGPS", "log_IC"]
axes[1].hist(lgps_vals, bins=15, color=LGPS_COLOR, edgecolor="white", linewidth=0.4,
             label="LGPS family")
axes[1].axvline(-15.0, color="gray", linestyle="--", linewidth=1, label="censored placeholder")
axes[1].set_xlabel(r"$\log_{10}\sigma$ [S cm$^{-1}$]")
axes[1].set_ylabel(r"$N$ [entries]")
axes[1].set_title("(b) Target distribution (n=599)")
axes[1].legend(frameon=False, fontsize=8)

fig.tight_layout()
save(fig, "fig1_dataset_overview")

fig, ax = plt.subplots(figsize=(3.9, 3.9))
is_lgps = pred_comp["Family"] == "LGPS"
ax.scatter(pred_comp.loc[~is_lgps, "log_IC_true"], pred_comp.loc[~is_lgps, "log_IC_pred"],
           s=22, alpha=0.65, color=DEFAULT_COLOR, label="other families")
ax.scatter(pred_comp.loc[is_lgps, "log_IC_true"], pred_comp.loc[is_lgps, "log_IC_pred"],
           s=32, alpha=0.9, color=LGPS_COLOR, marker="^", label="LGPS")
lims = [min(pred_comp["log_IC_true"].min(), pred_comp["log_IC_pred"].min()) - 0.5,
        max(pred_comp["log_IC_true"].max(), pred_comp["log_IC_pred"].max()) + 0.5]
ax.plot(lims, lims, color="black", linewidth=1, linestyle="--")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(r"True $\log_{10}\sigma$ [S cm$^{-1}$]")
ax.set_ylabel(r"Predicted $\log_{10}\sigma$ [S cm$^{-1}$]")
mae = xgb_metrics["composition_only"]["paper_split"]["mae"]
r2 = xgb_metrics["composition_only"]["paper_split"]["r2"]
ax.set_title(f"Tier B, paper's fixed split\n$\\mathit{{MAE}}$ = {mae:.2f}, $R^2$ = {r2:.2f}")
ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
save(fig, "fig2_parity_paper_split")

fam_res = xgb_metrics["composition_only"]["leave_one_family_out"]
fams = list(fam_res.keys())
maes = [fam_res[f]["mae"] for f in fams]
order = np.argsort(maes)
fams_sorted = [fams[i] for i in order]
maes_sorted = [maes[i] for i in order]
colors = [LGPS_COLOR if f == "LGPS" else DEFAULT_COLOR for f in fams_sorted]

fig, ax = plt.subplots(figsize=(4.9, 3.0))
bars = ax.barh(fams_sorted, maes_sorted, color=colors)
ax.axvline(xgb_metrics["composition_only"]["paper_split"]["mae"], color="black",
           linestyle="--", linewidth=1, label="random-split MAE (in-distribution)")
ax.set_xlabel(r"$\mathit{MAE}$ [log$_{10}$(S cm$^{-1}$)]")
ax.set_title("Leave-one-family-out (Tier B)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
save(fig, "fig3_loo_family_mae")

fam_unc = unc_metrics["leave_one_family_out"]
fams_u = list(fam_unc.keys())
cov = [fam_unc[f]["empirical_coverage"] for f in fams_u]
width = [fam_unc[f]["mean_width"] for f in fams_u]
UNDERCOVERED_COLOR = "#dd8452"
colors_u = [LGPS_COLOR if f == "LGPS" else (UNDERCOVERED_COLOR if c < 0.80 else DEFAULT_COLOR)
            for f, c in zip(fams_u, cov)]

fig, ax = plt.subplots(figsize=(4.2, 3.4))
sizes = [70 if f == "LGPS" else 45 for f in fams_u]
ax.scatter(width, cov, c=colors_u, s=sizes, edgecolor="black", linewidth=0.5, zorder=3)
offsets = {"LGPS": (6, 4), "argyrodites": (8, 8), "oxides": (-55, -12),
           "LISICON": (7, -3), "halides": (6, 6), "perovskites": (6, 4)}
for f, wv, cv in zip(fams_u, width, cov):
    if f == "LGPS" or cv < 0.80:
        color = LGPS_COLOR if f == "LGPS" else UNDERCOVERED_COLOR
        xytext = offsets.get(f, (6, 4))
        ax.annotate(f, (wv, cv), textcoords="offset points", xytext=xytext, fontsize=8,
                    color=color, fontweight="bold")
ax.axhline(0.80, color="black", linestyle="--", linewidth=1, label="target coverage (80%)")
ax.set_xlabel(r"$\bar{w}_{80}$ [log$_{10}$(S cm$^{-1}$)]")
ax.set_ylabel(r"Empirical coverage $c$ [$-$]")
ax.set_ylim(0, 1.05)
ax.set_title("Calibration under family extrapolation")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
save(fig, "fig5_uncertainty_family")

sub = pred_unc.sort_values("log_IC_true").reset_index(drop=True)
n_show = min(50, len(sub))
sub = sub.iloc[:: max(1, len(sub) // n_show)].reset_index(drop=True)
yerr_lower = sub["log_IC_true"] - sub["q10"]
yerr_upper = sub["q90"] - sub["log_IC_true"]
covered = (sub["log_IC_true"] >= sub["q10"]) & (sub["log_IC_true"] <= sub["q90"])

fig, ax = plt.subplots(figsize=(5.5, 3.0))
idx = np.arange(len(sub))
ax.vlines(idx, sub["q10"], sub["q90"], color="lightgray", linewidth=2, zorder=1)
ax.scatter(idx[covered], sub.loc[covered, "log_IC_true"], color=DEFAULT_COLOR, s=18,
           label="covered", zorder=2)
ax.scatter(idx[~covered], sub.loc[~covered, "log_IC_true"], color=LGPS_COLOR, s=22,
           marker="x", label="not covered", zorder=2)
ax.set_xlabel(r"Test-set entry $i$ [$-$] (sorted by true value, subsampled)")
ax.set_ylabel(r"$\log_{10}\sigma$ [S cm$^{-1}$]")
cov_val = unc_metrics["paper_split"]["empirical_coverage"]
ax.set_title(f"10th-90th percentile intervals, paper split\n(full-set coverage = {cov_val:.1%})")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
save(fig, "figs2_calibration_paperset")

sg_res = xgb_metrics["composition_only"]["leave_one_space_group_out"]
sgs = list(sg_res.keys())
sg_maes = [sg_res[s]["mae"] for s in sgs]
order = np.argsort(sg_maes)
sgs_sorted = [f"sg {sgs[i]}" for i in order]
sg_maes_sorted = [sg_maes[i] for i in order]

fig, ax = plt.subplots(figsize=(4.5, 3.4))
ax.barh(sgs_sorted, sg_maes_sorted, color=DEFAULT_COLOR)
ax.axvline(xgb_metrics["composition_only"]["paper_split"]["mae"], color="black",
           linestyle="--", linewidth=1, label="random-split MAE")
ax.set_xlabel(r"$\mathit{MAE}$ [log$_{10}$(S cm$^{-1}$)]")
ax.set_title("Leave-one-space-group-out (Tier B)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
save(fig, "figs3_space_group_mae")

fam_res_s = xgb_metrics["composition_plus_structure"]["leave_one_family_out"]
fams_s = list(fam_res_s.keys())
maes_s = [fam_res_s[f]["mae"] for f in fams_s]
order = np.argsort(maes_s)
fams_s_sorted = [fams_s[i] for i in order]
maes_s_sorted = [maes_s[i] for i in order]
colors_s = [LGPS_COLOR if f == "LGPS" else STRUCT_COLOR for f in fams_s_sorted]

fig, ax = plt.subplots(figsize=(4.5, 2.7))
ax.barh(fams_s_sorted, maes_s_sorted, color=colors_s)
ax.axvline(xgb_metrics["composition_plus_structure"]["paper_split"]["mae"], color="black",
           linestyle="--", linewidth=1, label="random-split MAE (composition+structure)")
ax.set_xlabel(r"$\mathit{MAE}$ [log$_{10}$(S cm$^{-1}$)]")
ax.set_title("Leave-one-family-out (Tier C)")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
fig.tight_layout()
save(fig, "figs4_struct_loo_family")

print("\nAll figures generated successfully.")
print(f"Files in {FIG_DIR}:")
for p in sorted(FIG_DIR.glob("*.pdf")):
    print(" ", p.name)
