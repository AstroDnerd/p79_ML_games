"""
make_paper_figures.py
=====================
Draft figures for the net9010 paper, built from the stored datasets and the
pooled out-of-fold predictions (<model_dir>/net9010[_<tag>]_allmom_fold<k>_test.npz).
Re-run whenever the results change; every figure is overwritten.

  python make_paper_figures.py                      # all figures
  python make_paper_figures.py --only ms_scatter coverage
Output: <paper>/figures/fig_<name>.{pdf,png}
"""

import argparse, glob, json, os, shutil
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import norm

DATA   = "/home/x-nbisht1/scratch/projects/p79d_dataset"
MODELS = "/home/x-nbisht1/projects/p79d_dataset/models"
PLOTS  = "/home/x-nbisht1/plots"
PAPER  = "/home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_paper/6a9880a1fb2736b154e43f78"
OUT    = f"{PAPER}/figures"

COL1, COL2 = 3.4, 7.1       # AASTeX column / page widths (inches)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "serif", "mathtext.fontset": "dejavuserif",
})
XI_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
XI_COLORS = dict(zip(XI_VALUES, plt.cm.viridis(np.linspace(0.0, 0.9, 5))))
MS_BINS   = [(0, 0.5), (0.5, 1), (1, 3), (3, 7), (7, 20.01)]

# model tag -> (label, folds) for the comparison figures
MODELS_ORDER = [
    ("",         "projected maps"),
    ("13co",     r"$^{13}$CO run 1, 0.1 K (v1)"),
    ("13cov2",   r"$^{13}$CO run 1, 0.2 K + masking (v2)"),
    ("co2_13",   r"run 2, $^{13}$CO"),
    ("co2_12",   r"run 2, $^{12}$CO"),
    ("co2_both", r"run 2, $^{13}$CO + $^{12}$CO"),
]


def save(fig, name, png=False):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/fig_{name}.{'png' if png else 'pdf'}"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def pooled(tag):
    """Concatenate the out-of-fold test predictions of every fold of one model."""
    t = f"_{tag}" if tag else ""
    files = sorted(glob.glob(f"{MODELS}/net9010{t}_allmom_fold[0-9]_test.npz"))
    if not files:
        return None
    out = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in d.files:
            if d[k].ndim == 1 and k != "sim_names":
                out.setdefault(k, []).append(d[k])
        out["sim_names"] = d["sim_names"]
    out = {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in out.items()}
    out["n_folds"] = len(files)
    return out


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def binned(x, y, edges, stat=np.median):
    c, v = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() > 20:
            c.append(np.sqrt(lo * hi) if lo > 0 else hi / 2); v.append(stat(y[m]))
    return np.array(c), np.array(v)


# ─── 1. Overview: moment maps, solenoidal vs compressive ─────────────────────

def fig_overview():
    proj = h5py.File(f"{DATA}/p79d_mach_grid_256_v1.h5", "r")
    rt   = h5py.File(f"{DATA}/p79d_mach_grid_256_13co_v1_fcrao500_v2.h5", "r")
    names = proj["sims/name"][:].astype(str)
    sid, frame, los = proj["samples/sim_id"][:], proj["samples/frame"][:], proj["samples/los"][:]
    ms, chi = proj["labels/Ms"][:], proj["labels/chi_v"][:]
    rows = []
    for name, label in [("xi_1_mach21.78", r"solenoidal ($\xi=1$)"), ("xi_0_mach35.3", r"compressive ($\xi=0$)")]:
        g = np.where(names == name)[0][0]
        m = np.where((sid == g) & (los == 2))[0]
        i = m[np.argmin(np.abs(ms[m] - 3.3))]        # similar Ms for both rows
        rows.append((i, label))
    fig, axes = plt.subplots(2, 4, figsize=(COL2, 3.75))
    for r, (i, label) in enumerate(rows):
        img = proj["images"][i]
        w   = rt["images_obs"][i, 0]; mask = rt["mask_obs"][i]
        panels = [
            (np.log10(img[0]), "viridis", r"$\log_{10}\Sigma/\langle\Sigma\rangle$"),
            (img[1] - img[1].mean(), "RdBu_r", r"$v_c - \langle v_c\rangle$ [$c_s$]"),
            (img[2], "magma", r"$\sigma_v$ [$c_s$]"),
            (np.where(mask, w, np.nan), "cividis", r"$^{13}$CO $W$ [K km s$^{-1}$]"),
        ]
        for c, (a, cmap, lab) in enumerate(panels):
            ax = axes[r, c]
            if cmap == "RdBu_r":
                v = np.nanpercentile(np.abs(a), 99); kw = dict(vmin=-v, vmax=v)
            else:
                kw = dict(vmin=np.nanpercentile(a, 1), vmax=np.nanpercentile(a, 99))
            cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.8")
            im = ax.imshow(a, origin="lower", cmap=cm, **kw)
            ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(lab)
        axes[r, 0].set_ylabel(f"{label}\n$\\mathcal{{M}}_s={ms[i]:.1f}$, $\\chi_v={chi[i]:.2f}$")
    fig.subplots_adjust(wspace=0.25, hspace=0.08)
    save(fig, "overview", png=True)


# ─── 2. Realised chi vs Ms by forcing ────────────────────────────────────────

def fig_chi_xi():
    f = h5py.File(f"{DATA}/p79d_mach_grid_256_v1.h5", "r")
    ms, chi = f["frames/Ms"][:], f["frames/chi_v"][:]
    xi = f["sims/xi"][:][f["frames/sim_id"][:]]
    keep = ms <= 20
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    for x in XI_VALUES:
        m = keep & np.isclose(xi, x)
        ax.scatter(ms[m], chi[m], s=1.5, color=XI_COLORS[x], alpha=0.5, lw=0, rasterized=True)
        fc = (1 - x) ** 2 / (1 - 2 * x + 3 * x ** 2)
        ax.plot([], [], "o", ms=3, color=XI_COLORS[x], label=fr"$\xi={x:g}$ (forcing {100*fc:.0f}% compr.)")
    ax.axhline(1 / 3, color="0.4", ls=":", lw=0.8)
    ax.text(0.04, 1 / 3 + 0.02, "random field", color="0.4", fontsize=6)
    ax.set_xscale("log"); ax.set_xlabel(r"$\mathcal{M}_s$"); ax.set_ylabel(r"$\chi_v = E_c/(E_c+E_s)$")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=5.5,
              handletextpad=0.2, columnspacing=0.8)
    save(fig, "chi_xi")


# ─── 3. Architecture schematic ───────────────────────────────────────────────

def fig_architecture():
    """The architecture figure is a draw.io diagram (~/plots/ML_Model_arch/p79d_paper_net9010/make_diagram.py)."""
    src = "/home/x-nbisht1/plots/ML_Model_arch/p79d_paper_net9010/p79d_paper_net9010.png"
    shutil.copy(src, f"{OUT}/fig_architecture.png"); print("copied", src)


def fig_ms_scatter():
    sets = [("", "projected maps"), ("13cov2", r"synthetic $^{13}$CO (run 1)"),
            ("co2_both", r"run 2: $^{13}$CO + $^{12}$CO")]
    sets = [(t, l, pooled(t)) for t, l in sets]
    sets = [s for s in sets if s[2] is not None]
    fig, axes = plt.subplots(2, len(sets), figsize=(COL2, 3.9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05, "wspace": 0.08})
    lim = (0.02, 25)
    for c, (tag, label, d) in enumerate(sets):
        ax, axr = axes[0, c], axes[1, c]
        y, p = d["ms_true"], d["ms_mean"]
        for x in XI_VALUES:
            m = np.isclose(d["xi_forcing"], x)
            ax.scatter(y[m], p[m], s=0.6, color=XI_COLORS[x], alpha=0.35, lw=0, rasterized=True)
        ax.plot(lim, lim, "k-", lw=0.6)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(label)
        ax.text(0.04, 0.95, f"$R^2(\\ln\\mathcal{{M}}_s)={r2(np.log(y), np.log(p)):.3f}$\n"
                f"median error {100*np.median(np.abs(p-y)/y):.1f}%\n{d['n_folds']} folds, {len(y)} images",
                transform=ax.transAxes, va="top", fontsize=6)
        fr = p / y - 1
        edges = np.logspace(np.log10(0.03), np.log10(20), 16)
        cx, med = binned(y, fr, edges)
        _, lo = binned(y, fr, edges, lambda v: np.percentile(v, 16))
        _, hi = binned(y, fr, edges, lambda v: np.percentile(v, 84))
        axr.fill_between(cx, lo, hi, color="0.75", lw=0); axr.plot(cx, med, "k-", lw=0.9)
        axr.axhline(0, color="k", lw=0.4, ls=":"); axr.set_ylim(-0.8, 1.2); axr.set_xscale("log")
        axr.set_xlabel(r"true $\mathcal{M}_s$")
        if c == 0:
            ax.set_ylabel(r"predicted $\mathcal{M}_s$"); axr.set_ylabel(r"$\Delta\mathcal{M}_s/\mathcal{M}_s$")
        else:
            ax.set_yticklabels([]); axr.set_yticklabels([])
    for x in XI_VALUES:
        axes[0, -1].plot([], [], "o", ms=3, color=XI_COLORS[x], label=fr"$\xi={x:g}$")
    axes[0, -1].legend(loc="lower right", frameon=False, handletextpad=0.1)
    save(fig, "ms_scatter")


# ─── 5. chi: predicted vs true ───────────────────────────────────────────────

def fig_chi_scatter():
    d = pooled("")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    sc = ax.scatter(d["chi_true"], d["chi_mean"], c=np.log10(d["ms_true"]), s=0.6, cmap="plasma",
                    alpha=0.4, lw=0, rasterized=True, vmin=-1, vmax=1.3)
    ax.plot([0, 1], [0, 1], "k-", lw=0.6); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"true $\chi_v$"); ax.set_ylabel(r"predicted $\chi_v$")
    hi = d["ms_true"] >= 1
    ax.text(0.04, 0.95, f"$R^2$ = {r2(d['chi_true'][hi], d['chi_mean'][hi]):.2f} ($\\mathcal{{M}}_s\\geq1$)\n"
            f"$R^2$ = {r2(d['chi_true'], d['chi_mean']):.2f} (all)", transform=ax.transAxes, va="top", fontsize=6)
    cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label(r"$\log_{10}\mathcal{M}_s$"); cb.solids.set_alpha(1)
    ax = axes[1]
    for s in np.unique(d["sim_id"]):
        m = d["sim_id"] == s
        x = float(d["xi_forcing"][m][0])
        ax.errorbar(np.median(d["chi_true"][m]), np.median(d["chi_mean"][m]),
                    yerr=[[np.median(d["chi_mean"][m]) - np.percentile(d["chi_mean"][m], 16)],
                          [np.percentile(d["chi_mean"][m], 84) - np.median(d["chi_mean"][m])]],
                    fmt="o", ms=3.5 + 1.2 * np.log10(np.median(d["ms_true"][m]) + 1) * 3, color=XI_COLORS[x],
                    mec="k", mew=0.3, elinewidth=0.5)
    for x in XI_VALUES:
        ax.plot([], [], "o", ms=3, color=XI_COLORS[x], label=fr"$\xi={x:g}$")
    ax.plot([0, 1], [0, 1], "k-", lw=0.6); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"true $\chi_v$ (run median)"); ax.set_ylabel(r"predicted $\chi_v$ (run median, 16–84%)")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    for name in ("xi_0_mach5.5", "xi_0_mach10"):
        g = list(d["sim_names"].astype(str)).index(name); m = d["sim_id"] == g
        ax.annotate(name, (np.median(d["chi_true"][m]), np.median(d["chi_mean"][m])),
                    xytext=(-62, 14), textcoords="offset points", fontsize=5.5,
                    arrowprops=dict(arrowstyle="-", lw=0.4))
    save(fig, "chi_scatter")


# ─── 6. xi confusion matrices (first net9010, random split) ──────────────────

def fig_xi_confusion():
    src = "/home/x-nbisht1/thesis/6669b1f35be7cdc2643f33d9/Figures/chapter3/xi_confusion_net9010_allmom_randomsplit.png"
    os.makedirs(OUT, exist_ok=True); shutil.copy(src, f"{OUT}/fig_xi_confusion.png"); print("copied", src)


# ─── 7. b Ms degeneracy: residual correlation by Ms bin ──────────────────────

def fig_degeneracy():
    d = pooled("")
    dl, dc = np.log(d["ms_mean"] / d["ms_true"]), d["chi_mean"] - d["chi_true"]
    fig, axes = plt.subplots(1, 5, figsize=(COL2, 1.75), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (lo, hi) in zip(axes, MS_BINS):
        m = (d["ms_true"] >= lo) & (d["ms_true"] < hi)
        ax.hexbin(dc[m], dl[m], gridsize=35, extent=(-0.8, 0.8, -1.2, 1.2), cmap="Greys", bins="log", mincnt=1,
                  linewidths=0, rasterized=True)
        ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
        ax.set_title(fr"${lo:g}\leq\mathcal{{M}}_s<{min(hi, 20):g}$", fontsize=6.5)
        ax.text(0.05, 0.05, f"$r={np.corrcoef(dl[m], dc[m])[0,1]:+.2f}$", transform=ax.transAxes, fontsize=6.5)
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-1.2, 1.2); ax.set_xlabel(r"$\Delta\chi_v$")
    axes[0].set_ylabel(r"$\Delta\ln\mathcal{M}_s$")
    save(fig, "degeneracy")


# ─── 8. Coverage: raw Gaussian vs conformal, per fold ────────────────────────

def fig_coverage():
    t = ""
    files = sorted(glob.glob(f"{MODELS}/net9010{t}_allmom_fold[0-9]_test.npz"))
    levels = np.linspace(0.5, 0.95, 10)
    fig, axes = plt.subplots(1, 2, figsize=(COL1 * 1.35, 2.0), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, key, lab in ((axes[0], "ms", r"$\ln\mathcal{M}_s$"), (axes[1], "chi", r"logit $\chi_v$")):
        for f in files:
            d = np.load(f)
            z = np.abs(d[f"t_{key}_y"] - d[f"t_{key}_mu"]) / d[f"t_{key}_sig"]
            raw = [np.mean(z <= norm.ppf(0.5 + a / 2)) for a in levels]
            ax.plot(levels, raw, color="#d95f02", lw=0.7, alpha=0.8)
            conf = [np.mean((d[f"{key}_true"] >= d[f"{key}_lo{L}"]) & (d[f"{key}_true"] <= d[f"{key}_hi{L}"]))
                    for L in (80, 90, 95)]
            ax.plot([0.8, 0.9, 0.95], conf, "o-", color="#1b9e77", ms=2.5, lw=0.7, alpha=0.9)
        ax.plot([0.5, 1], [0.5, 1], "k:", lw=0.6)
        ax.set_xlim(0.48, 0.97); ax.set_ylim(0.3, 1.0); ax.set_xlabel("nominal coverage"); ax.set_title(lab)
    axes[0].set_ylabel("empirical coverage (test fold)")
    axes[0].plot([], [], color="#d95f02", label="raw Gaussian"); axes[0].plot([], [], "o-", color="#1b9e77", ms=2.5, label="conformal")
    axes[0].legend(frameon=False, loc="lower right")
    save(fig, "coverage")


# ─── 9. Synthetic CO maps: run 2 at low and high density ─────────────────────

def fig_radmc_maps():
    f = h5py.File(f"{DATA}/p79d_mach_grid_256_co_v2_fcrao500.h5", "r")
    names = f["sims/name"][:].astype(str)
    g = np.where(names == "xi_0.5_mach40")[0][0]
    sid, los, n = f["samples/sim_id"][:], f["samples/los"][:], f["labels/n_H2"][:]
    ms = f["labels/Ms"][:]
    m = np.where((sid == g) & (los == 2))[0]
    picks = [m[np.argmin(n[m])], m[np.argmax(n[m])]]
    av = f["rt/AV_median"][:]
    fig, axes = plt.subplots(2, 4, figsize=(COL2, 3.7))
    for r, i in enumerate(picks):
        for c, (line, mom, lab, cmap) in enumerate([("13co", 0, r"$^{13}$CO $W$", "cividis"), ("13co", 2, r"$^{13}$CO $\sigma_v$", "magma"),
                                                     ("12co", 0, r"$^{12}$CO $W$", "cividis"), ("12co", 2, r"$^{12}$CO $\sigma_v$", "magma")]):
            a = f[f"images_obs_{line}"][i, mom]; mk = f[f"mask_obs_{line}"][i]
            a = np.where(mk, a, np.nan)
            ax = axes[r, c]; cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.85")
            im = ax.imshow(a, origin="lower", cmap=cm, vmin=np.nanpercentile(a, 1), vmax=np.nanpercentile(a, 99))
            ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6)
            ax.text(0.03, 0.03, f"{100*mk.mean():.0f}% detected", transform=ax.transAxes, fontsize=5.5,
                    color="w", bbox=dict(fc="k", alpha=0.4, lw=0, pad=1))
            if r == 0:
                ax.set_title(lab + (" [K km s$^{-1}$]" if mom == 0 else " [km s$^{-1}$]"))
        axes[r, 0].set_ylabel(f"$n={n[i]:.0f}$ cm$^{{-3}}$, $A_V={av[i]:.1f}$", fontsize=7)
    fig.suptitle(f"xi_0.5_mach40, $\\mathcal{{M}}_s = {ms[picks[0]]:.1f}$ and ${ms[picks[1]]:.1f}$, FCRAO-like noise", fontsize=7, y=0.99)
    fig.subplots_adjust(wspace=0.25, hspace=0.08)
    save(fig, "radmc_maps", png=True)


# ─── 10. Model comparison by Ms bin ──────────────────────────────────────────

def fig_model_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5), gridspec_kw={"wspace": 0.25})
    centers = np.arange(len(MS_BINS))
    colors = plt.cm.tab10(np.arange(len(MODELS_ORDER)))
    width = 0.8 / len(MODELS_ORDER)
    for j, (tag, label) in enumerate(MODELS_ORDER):
        d = pooled(tag)
        if d is None:
            continue
        err, r2c = [], []
        for lo, hi in MS_BINS:
            m = (d["ms_true"] >= lo) & (d["ms_true"] < hi)
            err.append(np.median(np.abs(d["ms_mean"][m] - d["ms_true"][m]) / d["ms_true"][m]))
            r2c.append(r2(d["chi_true"][m], d["chi_mean"][m]))
        x = centers - 0.4 + width * (j + 0.5)
        lab = label + (f" ({d['n_folds']} folds)" if d["n_folds"] < 5 else "")
        axes[0].bar(x, 100 * np.array(err), width, color=colors[j], label=lab)
        axes[1].bar(x, np.clip(r2c, -1, 1), width, color=colors[j])
    ticks = [fr"{lo:g}–{min(hi, 20):g}" for lo, hi in MS_BINS]
    for ax in axes:
        ax.set_xticks(centers); ax.set_xticklabels(ticks); ax.set_xlabel(r"$\mathcal{M}_s$ bin")
    axes[0].set_ylabel(r"median $|\Delta\mathcal{M}_s|/\mathcal{M}_s$ [%]"); axes[0].set_yscale("log")
    axes[1].set_ylabel(r"$R^2(\chi_v)$ (clipped at $-1$)"); axes[1].axhline(0, color="k", lw=0.5); axes[1].set_ylim(-1, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3,
               frameon=False, fontsize=6)
    save(fig, "model_comparison")


# ─── 11. Run 2: accuracy vs extinction ───────────────────────────────────────

def fig_av():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), gridspec_kw={"wspace": 0.25})
    edges = np.array([0.8, 1.5, 2.5, 4, 6, 10, 16, 30])
    cols = {"co2_13": "#1b9e77", "co2_12": "#d95f02", "co2_both": "#7570b3"}
    for tag, label in MODELS_ORDER[3:]:
        d = pooled(tag)
        if d is None:
            continue
        ok = d["ms_true"] >= 1
        av, e = d["AV_median"][ok], np.abs(d["ms_mean"][ok] - d["ms_true"][ok]) / d["ms_true"][ok]
        cx, med = binned(av, e, edges)
        axes[0].plot(cx, 100 * med, "o-", color=cols[tag], ms=3, lw=0.9, label=label)
        cx, mae = binned(av, np.abs(d["chi_mean"][ok] - d["chi_true"][ok]), edges, np.mean)
        axes[1].plot(cx, mae, "o-", color=cols[tag], ms=3, lw=0.9)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel(r"median $A_V$ of the image"); ax.axvline(5, color="0.5", ls=":", lw=0.7)
    for ax in axes:
        ax.text(5.4, 0.92, r"$^{13}$CO saturates $\rightarrow$", transform=ax.get_xaxis_transform(), fontsize=6, color="0.4")
    axes[0].set_ylabel(r"median $|\Delta\mathcal{M}_s|/\mathcal{M}_s$ [%] ($\mathcal{M}_s\geq1$)")
    axes[1].set_ylabel(r"MAE($\chi_v$) ($\mathcal{M}_s\geq1$)")
    axes[0].legend(frameon=False, fontsize=6)
    save(fig, "av")


# ─── 12. Observed clouds: tile maps (from predict_observations_net9010.py) ───

def fig_obs_maps():
    os.makedirs(OUT, exist_ok=True)
    for cloud in ("perseus", "orion"):
        src = f"{PLOTS}/observations_net9010/{cloud}_net9010.png"
        shutil.copy(src, f"{OUT}/fig_obs_{cloud}.png"); print("copied", src)


# ─── 13. Observed clouds: predictions vs matched-scale Ms ────────────────────

def fig_obs_compare():
    runs = [("projmask", "projected + mask aug."), ("13cov2", r"$^{13}$CO run 1 (v2)"), ("co2_13", r"run 2 $^{13}$CO")]
    marks = {"perseus": "o", "ophiuchus": "s", "orion": "D", "serpens": "^"}
    fig, axes = plt.subplots(1, len(runs), figsize=(COL2, 2.5), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (tag, label) in zip(axes, runs):
        d = f"{PLOTS}/observations_net9010" + (f"_{tag}" if tag else "")
        import pandas as pd
        for cloud, mk in marks.items():
            p = f"{d}/{cloud}_net9010_tiles.csv"
            if not os.path.exists(p):
                continue
            t = pd.read_csv(p)
            folds = [c for c in t.columns if c.startswith("Ms_fold") and c[7:].isdigit()]
            sc = ax.scatter(t["Ms_tile_spectrum"], t["Ms_foldall"], c=t["coverage"], cmap="viridis", vmin=0.4, vmax=1,
                            marker=mk, s=18, edgecolor="k", lw=0.3, zorder=3)
            ax.vlines(t["Ms_tile_spectrum"], t[folds].min(axis=1), t[folds].max(axis=1), color="0.6", lw=0.6, zorder=2)
            ax.scatter([], [], marker=mk, color="0.6", edgecolor="k", lw=0.3, label=cloud.capitalize())
        x = np.array([3, 30])
        ax.plot(x, x, "k-", lw=0.6); ax.fill_between(x, x / 1.2, x * 1.2, color="0.9", lw=0, zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(4, 25); ax.set_ylim(4, 25)
        ax.set_title(label); ax.set_xlabel(r"$\mathcal{M}_s^{\rm obs}=\sqrt{3}\,\sigma_{\rm tile}/c_s$")
        from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator([5, 10, 20])); axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(ScalarFormatter())
    axes[0].set_ylabel(r"predicted $\mathcal{M}_s$ (final model; bar: fold range)")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    cb = fig.colorbar(sc, ax=axes, pad=0.01, fraction=0.03); cb.set_label("detected fraction of tile")
    save(fig, "obs_compare")


# ─── 14. Perseus: observed Mach number depends on scale ──────────────────────

def fig_linewidth_scale():
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    L = np.logspace(-1.5, 1.7, 50)
    ax.plot(L, np.sqrt(3) * 0.7 * L ** 0.5 / 0.188, "k:", lw=0.8, label=r"Larson: $\sigma_{1D}=0.7\,L^{1/2}$ km s$^{-1}$")
    ax.plot(0.067, 4.8, "o", color="#1b9e77", ms=4, label="single beams (median)")
    arce = [(0.6, 7.4), (2.0, 7.6), (0.7, 6.5), (0.9, 7.2), (1.4, 5.4), (0.9, 5.8)]
    ax.plot(*zip(*[(2 * r, m) for r, m in arce]), "s", color="#d95f02", ms=3.5, label="regions (Arce+2010, $L=2R$)")
    import pandas as pd
    t = pd.read_csv(f"{PLOTS}/observations_net9010/perseus_net9010_tiles.csv")
    ax.plot(np.full(len(t), 4.3), t["Ms_tile_spectrum"], "D", color="#7570b3", ms=3.5, label="128-pixel tiles (this work)")
    ax.plot(30, 14, "^", color="#e7298a", ms=4.5, label="complex (Padoan+2003)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("scale $L$ [pc]"); ax.set_ylabel(r"$\mathcal{M}_s$ from linewidth")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    save(fig, "linewidth_scale")


# ─── 15. MWISP: network vs tile-spectrum Ms, by tile size ────────────────────

def mwisp_tiles(tag):
    """Every MWISP tile of one model with its cloud's tile size in pc (from the summary)."""
    import pandas as pd
    d = f"{PLOTS}/observations_net9010_mwisp_{tag}"
    size = {}
    for line in open(f"{d}/net9010_observations_summary.txt").read().splitlines()[1:]:
        p = line.split()
        if len(p) > 4 and p[0].startswith("G"):
            size[p[0]] = float(p[3])
    t = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{d}/G*_tiles.csv"))])
    t["tile_pc"] = t["cloud"].map(size)
    return t.dropna(subset=["tile_pc"])


def fig_mwisp():
    runs = [("co2_13", r"run 2, $^{13}$CO"), ("co2_both", r"run 2, $^{13}$CO + $^{12}$CO")]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7), sharey=True, gridspec_kw={"wspace": 0.06})
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    for ax, (tag, label) in zip(axes, runs):
        t = mwisp_tiles(tag)
        big = t["tile_pc"] > 10
        ax.scatter(t.loc[big, "Ms_tile_spectrum"], t.loc[big, "Ms_foldall"], s=10, marker="x", color="0.6", lw=0.6,
                   label=r"tile $>10$ pc", zorder=2)
        sc = ax.scatter(t.loc[~big, "Ms_tile_spectrum"], t.loc[~big, "Ms_foldall"], c=t.loc[~big, "tile_pc"],
                        cmap="viridis", vmin=2, vmax=10, s=16, edgecolor="k", lw=0.3, label=r"tile $\leq10$ pc", zorder=3)
        x = np.array([1, 60])
        ax.plot(x, x, "k-", lw=0.6); ax.plot(x, 1.4 * x, "k:", lw=0.6)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2, 50); ax.set_ylim(2, 30)
        ax.set_title(label); ax.set_xlabel(r"$\mathcal{M}_s^{\rm obs}=\sqrt{3}\,\sigma_{\rm tile}/c_s$")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator([2, 5, 10, 20])); axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(ScalarFormatter())
    axes[0].set_ylabel(r"predicted $\mathcal{M}_s$ (final model)")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    cb = fig.colorbar(sc, ax=axes, pad=0.01, fraction=0.03); cb.set_label("tile size [pc]")
    save(fig, "mwisp")


# ─── 16. Herschel dust: column-density model on the CO tiles ─────────────────

def fig_dust():
    import pandas as pd
    t = pd.read_csv(f"{PLOTS}/observations_net9010_dust/dust_matched_tiles.csv")
    marks = {"perseus": ("o", "Perseus"), "ophiuchus": ("s", "Ophiuchus"), "orion": ("D", "Orion A")}
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    for cloud, (mk, name) in marks.items():
        s = t[t["co_cloud"] == cloud]
        if s.empty:
            continue
        ax.scatter(s["Ms_CO_tile_spectrum"], s["Ms_kfold_mean"], marker=mk, s=18, color="#d95f02", edgecolor="k", lw=0.3,
                   label=f"{name}, dust", zorder=3)
        ax.scatter(s["Ms_CO_tile_spectrum"], s["Ms_CO_projmask_kfold"], marker=mk, s=18, facecolor="none",
                   edgecolor="#1b9e77", lw=0.7, label=f"{name}, $^{{13}}$CO", zorder=3)
        ax.plot(np.c_[s["Ms_CO_tile_spectrum"], s["Ms_CO_tile_spectrum"]].T,
                np.c_[s["Ms_kfold_mean"], s["Ms_CO_projmask_kfold"]].T, color="0.7", lw=0.5, zorder=1)
    x = np.array([1, 40])
    ax.plot(x, x, "k-", lw=0.6); ax.plot(x, 0.5 * x, "k:", lw=0.6)
    ax.text(18, 8.0, r"$\times0.5$", fontsize=6)
    ax.set_xlim(4, 22); ax.set_ylim(1.5, 22)
    ax.set_xlabel(r"$\mathcal{M}_s^{\rm obs}$ from the $^{13}$CO tile spectrum")
    ax.set_ylabel(r"predicted $\mathcal{M}_s$ (5-fold mean)")
    ax.legend(frameon=False, fontsize=5, loc="upper left", ncol=2)
    save(fig, "dust")


FIGS = {k[4:]: v for k, v in globals().items() if k.startswith("fig_") and callable(v)}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", choices=list(FIGS))
    a = p.parse_args()
    for name in (a.only or FIGS):
        try:
            FIGS[name]()
        except Exception as e:                      # keep going; report what failed
            print(f"FAILED {name}: {type(e).__name__}: {e}")
