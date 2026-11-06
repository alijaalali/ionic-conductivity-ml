"""Graphical abstract drawn at printed size (5.5 x 3.7 in); all text >= 8 pt."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from config import FIG_DIR
W, H = 5.5, 3.7
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": False})
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.add_patch(Rectangle((0.03, 0.03), W - 0.06, H - 0.06, lw=1.2, ec="#333333", fc="none"))

def box(cx, cy, w, h, text, fc, ec, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                                lw=1.0, ec=ec, fc=fc))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="#1a1a1a",
            fontweight="bold" if bold else "normal", linespacing=1.25)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#2b5a8c"))

ax.text(W/2, H - 0.17, "Composition versus structure in ML models of ionic conductivity",
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#15314d")
yc = 2.55
box(0.78, yc, 1.3, 0.85, "OBELiX dataset:\n599 measured Li\nconductors", "#f3e9d8", "#8a6a2f")
ty = [3.07, yc, 2.03]
tiers = ["Tier A: composition", "Tier B: + lattice,\nspace group", "Tier C: + Voronoi,\ncoordination"]
for y, t in zip(ty, tiers):
    box(2.45, y, 1.45, 0.46, t, "#eef3fb", "#2b5a8c")
    arrow(1.43, yc, 1.72, y); arrow(3.18, y, 3.42, yc)
box(4.45, yc, 1.9, 1.35, "Gradient-boosted trees,\n4 protocols: random split,\nleave-family-out,\nleave-space-group-out,\nleave-duplicate-group-out",
    "#eaf2ea", "#3f7a4a")
ax.plot([0.2, W - 0.2], [1.62, 1.62], color="#888888", lw=0.7, ls="--")
ax.text(W/2, 1.46, "Key findings", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#15314d")
bw, by = 1.66, 0.7
box(0.2 + bw/2, by, bw, 1.0, "Composition alone:\nMAE 1.86, R$^2$ 0.29;\nbelow published\nbaseline (1.59)", "#fdeeee", "#b23b3b")
box(W/2, by, bw, 1.0, "+ lattice, space group:\nMAE 1.62, matches\nbaseline; Voronoi adds\nnothing on 3 of 4 protocols", "#eaf2ea", "#3f7a4a")
box(W - 0.2 - bw/2, by, bw, 1.0, "Screening: top-30%\nenrichment about 1.9;\ntop-10/20% equals\nchance", "#eef3fb", "#2b5a8c")
fig.savefig(FIG_DIR / "graphical_abstract.pdf")
fig.savefig(FIG_DIR / "graphical_abstract.png", dpi=300)
