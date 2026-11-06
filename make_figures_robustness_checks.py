# Additional figures: three-tier feature comparison, noise-floor reanalysis with
# uncertainty, and LGPS feature-space-distance diagnostic.
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIG_DIR, RESULTS

REV_DIR = RESULTS / "tier_comparison"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
})

TIER_A_COLOR = "#4c72b0"
TIER_B_COLOR = "#8172b2"
TIER_C_COLOR = "#dd8452"
LGPS_COLOR = "#d62728"

def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {name}")

res = json.load(open(REV_DIR / "all_results.json"))

s1 = res["section1_three_tier_matched"]
protocols = ["paper_split", "leave_one_family_out", "leave_one_space_group_out", "leave_one_duplicate_group_out"]
proto_labels = ["Paper split\n(interpolation)", "Leave-one-\nfamily-out",
                "Leave-one-\nspace-group-out", "Leave-one-\nduplicate-group-out"]
tiers = ["A_composition_only", "B_composition_plus_crystal", "C_composition_crystal_voronoi"]
tier_labels = ["A: composition only\n(132 Magpie, no crystal info)",
               "B: + basic crystallography\n(lattice, space group, Z)",
               "C: + Voronoi/coordination\n(full resolved structure)"]
colors = [TIER_A_COLOR, TIER_B_COLOR, TIER_C_COLOR]

x = np.arange(len(protocols))
w = 0.26
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for i, (tname, label, color) in enumerate(zip(tiers, tier_labels, colors)):
    vals = [s1[tname][p]["pooled_mae"] for p in protocols]
    err_lo = [s1[tname][p]["pooled_mae"] - s1[tname][p].get("pooled_mae_ci95", [vals[j], vals[j]])[0]
              for j, p in enumerate(protocols)]
    err_hi = [s1[tname][p].get("pooled_mae_ci95", [vals[j], vals[j]])[1] - s1[tname][p]["pooled_mae"]
              for j, p in enumerate(protocols)]
    ax.bar(x + (i - 1) * w, vals, width=w, color=color, label=label,
           yerr=[err_lo, err_hi], capsize=3, error_kw={"linewidth": 0.8})
ax.set_xticks(x)
ax.set_xticklabels(proto_labels, fontsize=8)
ax.set_ylabel(r"Pooled $\mathit{MAE}$ [log$_{10}$(S cm$^{-1}$)]")
ax.set_title("Matched 321-entry subset, identical splits across tiers")
ax.set_ylim(0, 7.2)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
save(fig, "fig4_tier_comparison")

s2 = res["section2_noise_floor"]
fig, ax = plt.subplots(figsize=(4.9, 3.4))
cats = ["Model MAE\n(group-weighted)", "Experimental\nMAD", "Model RMSE\n(group-weighted)", "Experimental\nRMSD"]
vals = [s2["group_weighted_mean_mae"], s2["paper_reported_noise_mad"],
        s2["group_weighted_mean_rmse"], s2["paper_reported_noise_rmsd"]]
errs = [[s2["group_weighted_mean_mae"] - s2["group_weighted_mean_mae_ci95"][0]], [0], [0], [0]]
bars_colors = ["#4c72b0", "#999999", "#dd8452", "#999999"]
bar_pos = np.arange(4)
ax.bar(bar_pos, vals, color=bars_colors)
ax.errorbar([0], [s2["group_weighted_mean_mae"]],
            yerr=[[s2["group_weighted_mean_mae"] - s2["group_weighted_mean_mae_ci95"][0]],
                  [s2["group_weighted_mean_mae_ci95"][1] - s2["group_weighted_mean_mae"]]],
            fmt="none", ecolor="black", capsize=4, linewidth=1.3)
ax.set_xticks(bar_pos)
ax.set_xticklabels(cats, fontsize=8)
ax.set_ylabel(r"Error [log$_{10}$(S cm$^{-1}$)]")
ax.set_title("Like-for-like noise-floor comparison\n"
             f"95% CI on model MAE: [{s2['group_weighted_mean_mae_ci95'][0]:.2f}, "
             f"{s2['group_weighted_mean_mae_ci95'][1]:.2f}]")
fig.tight_layout()
save(fig, "figs1_duplicate_noise")

s6 = res["section6_lgps_distance"]
fig, ax = plt.subplots(figsize=(3.6, 3.4))
fams = list(s6.keys())
dists = [s6[f]["median_nn_distance_to_rest"] for f in fams]
cols = [LGPS_COLOR if f == "LGPS" else TIER_A_COLOR for f in fams]
ax.bar(fams, dists, color=cols)
for i, (f, d) in enumerate(zip(fams, dists)):
    ax.text(i, d + 0.15, f"n={s6[f]['n']}", ha="center", fontsize=8)
ax.set_ylabel(r"Median $d_{\mathrm{NN}}$ [standardized units]")
ax.set_title("LGPS is not a feature-space outlier")
fig.tight_layout()
save(fig, "fig6_lgps_distance")

print("DONE")
