"""
plot_mc_predictions.py
======================
Scatter plot of net9009 Mach number predictions vs observational ground truth
for all molecular clouds. Each MC gets a unique color; tiled MCs have multiple
points of the same color.

Fill in the PREDICTIONS dict below with your values.
Ground truth values are from Arce et al. (2010) Table 5, using
  M_rms = sigma_v / c_s  with sigma_v = Delta_v / 2.355
  and c_s = 0.188 * sqrt(T_ex / 10 K) km/s
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

GROUND_TRUTH = {
    # Perseus sub-regions
    "L1448":    (4.3, 0.77),
    "NGC1333":  (4.4, 0.79),
    "B1-Ridge": (3.8, 0.68),
    "B1":       (4.2, 0.76),
    "IC348":    (3.3, 0.60),
    "B5":       (3.1, 0.56),
    # Add other MCs here as you get their literature values
    # "Taurus":   (M_rms, delta_M),
    # "Ophiuchus":(M_rms, delta_M),
    # "Orion":    (M_rms, delta_M),
    # "Serpens":  (M_rms, delta_M),
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PREDICTIONS  ← FILL THESE IN
# Format: mc_name -> list of (tile_label, pred_M, pred_sigma, gt_key)
#   tile_label : string shown in tooltip/legend (e.g. "Tile 1", or cloud name)
#   pred_M     : net9009 predicted Mach number (after exp() conversion)
#   pred_sigma : predicted uncertainty (pred_M * sigma_logspace)
#   gt_key     : key into GROUND_TRUTH dict to use as x-axis value
#
# Clouds with no tiling have one entry; tiled clouds have one per tile.
# All tiles for the same cloud share the same color automatically.
# ─────────────────────────────────────────────────────────────────────────────
PREDICTIONS = {
    "Perseus": [
        # (tile_label, pred_M, pred_sigma, gt_key)
        ("Tile 1 (L1448)",    4.56, 0.87, "L1448"),
        ("Tile 2 (NGC1333)",  4.59, 0.81, "NGC1333"),
        ("Tile 3 (B1-Ridge)", 4.04, 0.55, "B1-Ridge"),
        ("Tile 4 (B1)",       4.26, 0.52, "B1"),
        ("Tile 5 (IC348)",    4.07, 0.58, "IC348"),
        ("Tile 6 (B5)",       3.58, 0.39, "B5"),
    ],
    # "Taurus": [
    #     ("Tile 1", pred_M, pred_sigma, "Taurus"),
    #     ("Tile 2", pred_M, pred_sigma, "Taurus"),
    # ],
    # "Ophiuchus": [
    #     ("Tile 1", pred_M, pred_sigma, "Ophiuchus"),
    # ],
    # "Orion": [
    #     ("Tile 1", pred_M, pred_sigma, "Orion"),
    # ],
    # "Serpens": [
    #     ("Single", pred_M, pred_sigma, "Serpens"),
    # ],
}

# ─────────────────────────────────────────────────────────────────────────────
# PLOT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = [
    "#e41a1c",  # red      — Perseus
    "#377eb8",  # blue     — Taurus
    "#4daf4a",  # green    — Ophiuchus
    "#ff7f00",  # orange   — Orion
    "#984ea3",  # purple   — Serpens
    "#a65628",  # brown    — spare
    "#f781bf",  # pink     — spare
]

OUTPUT_PATH = "/home/x-nbisht1/plots/mc_mach_predictions.png"

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PLOT
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

all_x, all_y = [], []
legend_handles = []

for mc_idx, (mc_name, tiles) in enumerate(PREDICTIONS.items()):
    color = COLORS[mc_idx % len(COLORS)]

    xs, ys, xerrs, yerrs = [], [], [], []
    for (label, pred_M, pred_sigma, gt_key) in tiles:
        if gt_key not in GROUND_TRUTH:
            print(f"Warning: ground truth key '{gt_key}' not found, skipping {label}")
            continue
        gt_M, gt_err = GROUND_TRUTH[gt_key]
        xs.append(gt_M)
        ys.append(pred_M)
        xerrs.append(gt_err)
        yerrs.append(pred_sigma)
        all_x.append(gt_M)
        all_y.append(pred_M)

    if not xs:
        continue

    ax.errorbar(
        xs, ys,
        xerr=xerrs, yerr=yerrs,
        fmt="o", color=color,
        ecolor=color, elinewidth=1.2, capsize=4, capthick=1.2,
        markersize=7, markeredgewidth=0.8, markeredgecolor="white",
        alpha=0.85, zorder=3,
    )

    legend_handles.append(
        mpatches.Patch(color=color, label=mc_name)
    )

# 45-degree reference line
if all_x and all_y:
    lo = min(min(all_x), min(all_y)) * 0.55
    hi = max(max(all_x), max(all_y)) * 1.25
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, alpha=0.6,
            label="1:1 reference", zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

ax.set_xlabel("Observed Mach Number $\\mathcal{M}_{rms}$\n"
              "(Arce et al. 2010, from linewidth)", fontsize=12)
ax.set_ylabel("Predicted Mach Number (net9009)", fontsize=12)
ax.set_title("ML Mach Number Predictions vs Observations\n"
             "Horizontal bars: observational uncertainty  "
             "| Vertical bars: model $\\sigma$",
             fontsize=11)

legend_handles.append(
    plt.Line2D([0], [0], ls="--", color="k", alpha=0.6, label="1:1 reference")
)
ax.legend(handles=legend_handles, fontsize=10, loc="upper left",
          framealpha=0.85)
ax.grid(True, alpha=0.25)
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT_PATH}")