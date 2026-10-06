"""
analyze_kfold_net9010.py
========================
Pool the per-fold test predictions written by ViT_setup_test_net9010.py
(net9010_<mode>_fold<k>_test.npz) into out-of-fold results.

  python analyze_kfold_net9010.py --mode allmom
  python analyze_kfold_net9010.py --mode allmom --compare mom0

Prints pooled metrics, per-Ms-bin and per-simulation tables, and saves to ~/plots:
  kfold_predictions_net9010_<mode>.png   pooled Ms and chi scatter
  kfold_persim_net9010_<mode>.png        per-simulation median prediction vs truth
  kfold_residuals_net9010_<mode>.png     d ln Ms vs d chi (degeneracy check), per Ms bin
  kfold_compare_net9010_<mode>_vs_<other>.png   (with --compare)
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["allmom", "mom0"], default="allmom")
p.add_argument("--compare", choices=["allmom", "mom0"], default=None)
p.add_argument("--tag", default="", help="run tag used in training, e.g. 13co")
p.add_argument("--model_dir", default="/home/x-nbisht1/projects/p79d_dataset/models")
p.add_argument("--plot_dir", default=os.path.join(os.environ["HOME"], "plots"))
args = p.parse_args()
os.makedirs(args.plot_dir, exist_ok=True)
MS_BINS = [(0, 0.5), (0.5, 1), (1, 3), (3, 7), (7, 20.01)]


def load(mode):
    tg = f"_{args.tag}" if args.tag else ""
    files = sorted(f for f in glob.glob(os.path.join(args.model_dir, f"net9010{tg}_{mode}_fold*_test.npz"))
                   if "foldall" not in f)
    if not files:
        raise SystemExit(f"No net9010_{mode}_fold*_test.npz in {args.model_dir}")
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    n = len(parts[0]["ms_true"])
    keys = [k for k, v in parts[0].items() if np.ndim(v) == 1 and len(v) == n]
    d = {k: np.concatenate([q[k] for q in parts]) for k in keys}
    d["folds"] = sorted(int(q["fold"]) for q in parts)
    d["sim_names"] = parts[0]["sim_names"]
    return d


def r2(t, p_):
    return 1 - np.sum((p_ - t) ** 2) / np.sum((t - t.mean()) ** 2)


def report(d, mode):
    lt, lp = np.log(d["ms_true"]), np.log(d["ms_mean"])
    frac = np.abs(d["ms_mean"] - d["ms_true"]) / d["ms_true"]
    print(f"\n{mode}: folds {d['folds']}, {len(lt)} held-out images, "
          f"{len(np.unique(d['sim_id']))} simulations")
    print(f"  ln Ms : R²={r2(lt, lp):.4f}  median |Δ|/Ms={np.median(frac):.4f}  "
          f"68th pct={np.percentile(frac, 68):.4f}")
    print(f"  chi   : R²={r2(d['chi_true'], d['chi_mean']):.4f}  "
          f"MAE={np.mean(np.abs(d['chi_mean'] - d['chi_true'])):.4f}  "
          f"r={np.corrcoef(d['chi_true'], d['chi_mean'])[0, 1]:.4f}")
    for key in ("ms", "chi"):
        for lev in (80, 90, 95):
            if f"{key}_lo{lev}" in d:
                c = np.mean((d[f"{key}_true"] >= d[f"{key}_lo{lev}"]) & (d[f"{key}_true"] <= d[f"{key}_hi{lev}"]))
                print(f"  {key:3s} {lev}% interval coverage = {c:.3f}", end="")
        print()
    print(f"\n  {'Ms bin':<13}{'N':>6}{'med|Δ|/Ms':>11}{'chi MAE':>9}{'chi R²':>8}{'r(dlnMs,dchi)':>15}")
    for lo, hi in MS_BINS:
        b = (d["ms_true"] >= lo) & (d["ms_true"] < hi)
        if b.sum() > 5:
            dl, dc = lp[b] - lt[b], d["chi_mean"][b] - d["chi_true"][b]
            print(f"  [{lo:>4}, {hi:>5.1f}){b.sum():6d}{np.median(frac[b]):11.3f}{np.mean(np.abs(dc)):9.3f}"
                  f"{r2(d['chi_true'][b], d['chi_mean'][b]):8.3f}{np.corrcoef(dl, dc)[0, 1]:15.3f}")

    print(f"\n  {'simulation':<20}{'xi':>5}{'<Ms>':>7}{'N':>6}{'Ms pred/true':>13}{'chi true':>9}{'chi pred':>9}")
    rows = []
    for s in np.unique(d["sim_id"]):
        m = d["sim_id"] == s
        rows.append((d["sim_names"][s], d["xi_forcing"][m][0], d["Ms_mean"][m][0], m.sum(),
                     np.median(d["ms_mean"][m] / d["ms_true"][m]),
                     np.median(d["chi_true"][m]), np.median(d["chi_mean"][m])))
    for r in sorted(rows, key=lambda r: (r[1], r[2])):
        print(f"  {r[0]:<20}{r[1]:5.2f}{r[2]:7.2f}{r[3]:6d}{r[4]:13.3f}{r[5]:9.3f}{r[6]:9.3f}")
    return rows


def path(kind, extra=""):
    tg = f"_{args.tag}" if args.tag else ""
    return os.path.join(args.plot_dir, f"{kind}_net9010{tg}_{args.mode}{extra}.png")


d = load(args.mode)
rows = report(d, args.mode)

# pooled scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
sc = ax.scatter(d["ms_true"], d["ms_mean"], s=3, alpha=0.4, c=d["xi_forcing"], cmap="coolwarm")
lim = [d["ms_true"].min(), d["ms_true"].max()]
ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"True $\mathcal{M}_s$"); ax.set_ylabel(r"Predicted $\mathcal{M}_s$")
fig.colorbar(sc, ax=ax, label=r"forcing $\xi$")
ax = axes[1]
sc = ax.scatter(d["chi_true"], d["chi_mean"], s=3, alpha=0.4, c=np.log10(d["ms_true"]), cmap="viridis")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel(r"True $\chi$"); ax.set_ylabel(r"Predicted $\chi$")
fig.colorbar(sc, ax=ax, label=r"$\log_{10}\mathcal{M}_s$")
for a in axes: a.grid(alpha=0.3)
fig.suptitle(f"net9010 {args.mode} — pooled held-out folds {d['folds']}")
fig.tight_layout(); fig.savefig(path("kfold_predictions"), dpi=150); plt.close(fig)

# per-simulation medians
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
R = np.array([r[1:] for r in rows], dtype=float)   # xi, <Ms>, N, ratio, chi_t, chi_p
axes[0].scatter(R[:, 1], R[:, 3], c=R[:, 0], cmap="coolwarm", s=40, edgecolor="k")
axes[0].axhline(1, color="k", ls="--"); axes[0].set_xscale("log")
axes[0].set_xlabel(r"$\langle\mathcal{M}_s\rangle$ of simulation"); axes[0].set_ylabel("median predicted / true Ms")
axes[1].scatter(R[:, 4], R[:, 5], c=R[:, 0], cmap="coolwarm", s=40, edgecolor="k")
axes[1].plot([0, 1], [0, 1], "k--"); axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
axes[1].set_xlabel(r"median true $\chi$"); axes[1].set_ylabel(r"median predicted $\chi$")
for a in axes: a.grid(alpha=0.3)
fig.suptitle(f"net9010 {args.mode} — one point per held-out simulation (colour: forcing ξ)")
fig.tight_layout(); fig.savefig(path("kfold_persim"), dpi=150); plt.close(fig)

# residual coupling (b·Ms-type degeneracy)
lt, lp = np.log(d["ms_true"]), np.log(d["ms_mean"])
fig, axes = plt.subplots(1, len(MS_BINS), figsize=(4 * len(MS_BINS), 3.8), sharex=True, sharey=True)
for ax, (lo, hi) in zip(axes, MS_BINS):
    b = (d["ms_true"] >= lo) & (d["ms_true"] < hi)
    if b.sum() < 5:
        continue
    dl, dc = lp[b] - lt[b], d["chi_mean"][b] - d["chi_true"][b]
    ax.hexbin(dc, dl, gridsize=35, cmap="Blues", mincnt=1, bins="log", extent=(-0.5, 0.5, -1, 1))
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_title(f"Ms {lo}–{hi:.0f}  r={np.corrcoef(dl, dc)[0, 1]:+.2f}", fontsize=10)
    ax.set_xlabel(r"$\hat\chi - \chi$")
axes[0].set_ylabel(r"$\ln\hat{\mathcal{M}}_s - \ln\mathcal{M}_s$")
fig.suptitle(f"net9010 {args.mode} — Ms vs chi residuals")
fig.tight_layout(); fig.savefig(path("kfold_residuals"), dpi=150); plt.close(fig)

if args.compare:
    o = load(args.compare)
    report(o, args.compare)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for dd, lbl in ((d, args.mode), (o, args.compare)):
        fr = np.abs(dd["ms_mean"] - dd["ms_true"]) / dd["ms_true"]
        dc = np.abs(dd["chi_mean"] - dd["chi_true"])
        mids, e1, e2 = [], [], []
        for lo, hi in MS_BINS:
            b = (dd["ms_true"] >= lo) & (dd["ms_true"] < hi)
            if b.sum() > 5:
                mids.append(np.sqrt(max(lo, 0.1) * hi)); e1.append(np.median(fr[b])); e2.append(np.mean(dc[b]))
        axes[0].plot(mids, e1, "o-", lw=2, label=lbl); axes[1].plot(mids, e2, "o-", lw=2, label=lbl)
    axes[0].set_ylabel(r"median $|\Delta\mathcal{M}_s|/\mathcal{M}_s$"); axes[1].set_ylabel(r"mean $|\Delta\chi|$")
    for a in axes:
        a.set_xscale("log"); a.set_xlabel(r"$\mathcal{M}_s$"); a.grid(alpha=0.3); a.legend()
    fig.suptitle(f"net9010 {args.mode} vs {args.compare} — held-out error by Mach number")
    fig.tight_layout(); fig.savefig(path("kfold_compare", f"_vs_{args.compare}"), dpi=150); plt.close(fig)

print("\nSaved plots to", args.plot_dir)
