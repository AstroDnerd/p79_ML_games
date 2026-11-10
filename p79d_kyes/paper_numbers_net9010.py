"""
paper_numbers_net9010.py
========================
Every number quoted in the Results section of the paper, from the pooled
out-of-fold test predictions of the final k-fold runs
(<model_dir>/net9010_<tag>_<mode>_fold<k>_test.npz), plus the linewidth
baseline computed from the run-2 maps.

  python paper_numbers_net9010.py [--no_linewidth] > paper_numbers.txt
"""
import argparse, glob, json, os
import numpy as np
import h5py

p = argparse.ArgumentParser()
p.add_argument("--model_dir", default="/home/x-nbisht1/projects/p79d_dataset/models")
p.add_argument("--data_dir", default="/home/x-nbisht1/scratch/projects/p79d_dataset")
p.add_argument("--no_linewidth", action="store_true", help="skip the linewidth baseline (reads the maps)")
args = p.parse_args()

MODELS = [("final_proj", "allmom", "Projected, full moments + mask aug."),
          ("final_mom0", "mom0", "Projected, column density only"),
          ("final_13co", "allmom", "13CO run 1 (FCRAO-like)"),
          ("final_co13", "allmom", "13CO run 2"),
          ("final_co12", "allmom", "12CO run 2"),
          ("final_coboth", "allmom", "13CO + 12CO run 2")]
BINS = [(0, 0.5), (0.5, 1), (1, 3), (3, 7), (7, 20.01)]
SUP = [(1, 3), (3, 7), (7, 20.01)]


def load(tag, mode):
    files = sorted(glob.glob(f"{args.model_dir}/net9010_{tag}_{mode}_fold[0-9]_test.npz"))
    out = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in d.files:
            if d[k].ndim == 1 and k not in ("sim_names", "fold"):
                out.setdefault(k, []).append(d[k])
        out.setdefault("fold", []).append(np.full(len(d["ms_true"]), int(d["fold"])))
        out["sim_names"] = d["sim_names"].astype(str)
    out = {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in out.items()}
    out["n_folds"] = len(files)
    out["q"] = [json.load(open(f"{args.model_dir}/conformal_quantiles_9010_{tag}_{mode}_fold{k}.json"))
                for k in range(len(files))]
    return out


def r2(y, x):
    return 1 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)


def frac(d, m=slice(None)):
    return np.abs(d["ms_mean"][m] / d["ms_true"][m] - 1)


def inbin(d, lo, hi):
    return (d["ms_true"] >= lo) & (d["ms_true"] < hi)


def cov(d, key, L, m=slice(None)):
    t = d[f"{key}_true"][m]
    return np.mean((t >= d[f"{key}_lo{L}"][m]) & (t <= d[f"{key}_hi{L}"][m]))


def per_run(d, m=None):
    """name -> (Ms_mean, xi, median pred/true Ms, mean chi true, median chi pred)."""
    m = np.ones(len(d["ms_true"]), bool) if m is None else m
    rows = {}
    for g in np.unique(d["sim_id"][m]):
        s = m & (d["sim_id"] == g)
        rows[d["sim_names"][g]] = (d["Ms_mean"][s][0], d["xi_forcing"][s][0], np.median(d["ms_mean"][s] / d["ms_true"][s]),
                                   d["chi_true"][s].mean(), np.median(d["chi_mean"][s]))
    return rows


D = {tag: load(tag, mode) for tag, mode, _ in MODELS}
P = D["final_proj"]
pr = lambda *a: print(*a)

pr("=" * 78 + "\nTABLE: out-of-fold Ms metrics\n" + "=" * 78)
pr(f"{'model':40s} {'R2lnMs':>7} {'med':>6} {'68th':>6} {'cov90':>6}  chi R2 (1-3/3-7/7-20)  folds")
for tag, mode, lab in MODELS:
    d = D[tag]
    c = [r2(d["chi_true"][inbin(d, *b)], d["chi_mean"][inbin(d, *b)]) for b in SUP]
    pr(f"{lab:40s} {r2(np.log(d['ms_true']), np.log(d['ms_mean'])):7.3f} {np.median(frac(d)):6.3f} "
       f"{np.percentile(frac(d), 68):6.3f} {cov(d, 'ms', 90):6.3f}  "
       f"{c[0]:.2f}/{c[1]:.2f}/{c[2]:.2f} range {min(c):.2f}-{max(c):.2f}   {d['n_folds']}")

pr("\n" + "=" * 78 + "\n4.1 Ms (final_proj)\n" + "=" * 78)
pr(f"N images {len(P['ms_true'])}, runs {len(np.unique(P['sim_id']))}")
for b in BINS:
    pr(f"  Ms {b}: median frac err {np.median(frac(P, inbin(P, *b))):.3f}  N={inbin(P, *b).sum()}")
runs = per_run(P)
sup = {k: v for k, v in runs.items() if 1 <= v[0] <= 20}
pr(f"  runs with 1<=<Ms><=20: {len(sup)}, max |ratio-1| = {max(abs(v[2] - 1) for v in sup.values()):.3f}")
for k, v in sorted(runs.items(), key=lambda kv: kv[1][0]):
    flag = "  <--" if abs(v[2] - 1) > 0.15 else ""
    pr(f"    {k:18s} xi={v[1]:.2f} <Ms>={v[0]:6.2f}  pred/true={v[2]:.3f}  chi {v[3]:.3f} -> {v[4]:.3f}{flag}")

pr("\n" + "=" * 78 + "\n4.2 chi_v (final_proj)\n" + "=" * 78)
u = P["ms_true"] >= 1
pr(f"  Ms>=1: R2={r2(P['chi_true'][u], P['chi_mean'][u]):.3f}  pooled all: R2={r2(P['chi_true'], P['chi_mean']):.3f}")
for b in BINS:
    m = inbin(P, *b)
    pr(f"  {b}: R2={r2(P['chi_true'][m], P['chi_mean'][m]):.3f}  MAE={np.mean(np.abs(P['chi_mean'][m] - P['chi_true'][m])):.3f}")
m = inbin(P, 0.5, 1)
rr = per_run(P, m)
worst = max(rr.items(), key=lambda kv: abs(kv[1][4] - kv[1][3]))
pr(f"  worst run at 0.5-1: {worst[0]} true chi {worst[1][3]:.3f} pred {worst[1][4]:.3f}")
# chi range by forcing at Ms>=3 (per-run means), labels only
m3 = P["ms_true"] >= 3
for x in sorted(np.unique(P["xi_forcing"])):
    s = m3 & (P["xi_forcing"] == x)
    vals = [P["chi_true"][s & (P["sim_id"] == g)].mean() for g in np.unique(P["sim_id"][s])]
    pr(f"  xi={x:.2f}: run-mean chi_v at Ms>=3 in [{min(vals):.3f}, {max(vals):.3f}] ({len(vals)} runs)")
pr(f"  chi subsonic (Ms<1) true range: {P['chi_true'][P['ms_true'] < 1].min():.3f}-{P['chi_true'][P['ms_true'] < 1].max():.3f}")
# per-run skill and within-run behaviour at Ms>=1
lg = lambda c: np.log(np.clip(c, 1e-3, 1 - 1e-3) / (1 - np.clip(c, 1e-3, 1 - 1e-3)))
gs = np.unique(P["sim_id"][u])
yt = np.array([P["chi_true"][u & (P["sim_id"] == g)].mean() for g in gs])
yp = np.array([np.median(P["chi_mean"][u & (P["sim_id"] == g)]) for g in gs])
pr(f"  per-run medians (Ms>=1, {len(gs)} runs): R2={r2(yt, yp):.3f}, within 0.1: {np.sum(abs(yp - yt) < 0.1)}, "
   f"within 0.05: {np.sum(abs(yp - yt) < 0.05)}, slope={np.polyfit(yt, yp, 1)[0]:.2f}")
rm = P["chi_mean"].copy()
for g in gs:
    s = u & (P["sim_id"] == g); rm[s] = np.median(P["chi_mean"][s])
pr(f"  replace preds by run medians: R2 {r2(P['chi_true'][u], P['chi_mean'][u]):.3f} -> {r2(P['chi_true'][u], rm[u]):.3f}")
wv = np.sum([np.sum((P["chi_true"][u & (P["sim_id"] == g)] - P["chi_true"][u & (P["sim_id"] == g)].mean()) ** 2) for g in gs])
pr(f"  within-run variance fraction of chi_v (Ms>=1): {wv / np.sum((P['chi_true'][u] - P['chi_true'][u].mean()) ** 2):.3f}"
   f"  -> perfect run-mean predictor R2 {1 - wv / np.sum((P['chi_true'][u] - P['chi_true'][u].mean()) ** 2):.3f}")
pr(f"  within-run std (median over runs): truth {np.median([P['chi_true'][u & (P['sim_id'] == g)].std() for g in gs]):.3f}"
   f"  pred {np.median([P['chi_mean'][u & (P['sim_id'] == g)].std() for g in gs]):.3f}"
   f"  truth logit {np.median([lg(P['chi_true'][u & (P['sim_id'] == g)]).std() for g in gs]):.3f}")
pr(f"  within-run r(pred,truth) median: "
   f"{np.median([np.corrcoef(P['chi_true'][u & (P['sim_id'] == g)], P['chi_mean'][u & (P['sim_id'] == g)])[0, 1] for g in gs]):.2f}")
# kNN reference in (ln Ms, xi), leave-run-out, k = 50
from sklearn.neighbors import KNeighborsRegressor
X = np.stack([np.log(P["ms_true"]), P["xi_forcing"]], 1)
kp = np.empty(len(X))
for g in np.unique(P["sim_id"]):
    s = P["sim_id"] == g
    kp[s] = KNeighborsRegressor(50).fit(X[~s], P["chi_true"][~s]).predict(X[s])
kin = KNeighborsRegressor(50).fit(X, P["chi_true"]).predict(X)
pr(f"  kNN(lnMs, xi) k=50 leave-run-out: R2(Ms>=1)={r2(P['chi_true'][u], kp[u]):.3f}  in-sample {r2(P['chi_true'][u], kin[u]):.3f}")

pr("\n" + "=" * 78 + "\n4.2b density head (final_proj)\n" + "=" * 78)
ls_t, ls_p = np.log(P["sigma_s_true"]), np.log(P["sigma_s_mean"])
lb_t, lb_p = np.log(P["b_lin_true"]), np.log(P["b_lin_mean"])
lm = np.log(P["ms_true"])
pr(f"  ln sigma_s: R2 all {r2(ls_t, ls_p):.3f}; cubic in ln Ms: {r2(ls_t, np.polyval(np.polyfit(lm, ls_t, 3), lm)):.3f}")
m2 = P["ms_true"] >= 2
pr(f"  Ms>=2: head {r2(ls_t[m2], ls_p[m2]):.3f}, cubic Ms-only {r2(ls_t[m2], np.polyval(np.polyfit(lm[m2], ls_t[m2], 3), lm[m2])):.3f},"
   f" median frac err {np.median(np.abs(P['sigma_s_mean'][m2] / P['sigma_s_true'][m2] - 1)):.3f}")
m12 = inbin(P, 1, 2)
pr(f"  ln b_lin: R2 all {r2(lb_t, lb_p):.3f}; Ms 1-2 {r2(lb_t[m12], lb_p[m12]):.3f}; Ms>=2 {r2(lb_t[m2], lb_p[m2]):.3f}, "
   f"median frac err {np.median(np.abs(P['b_lin_mean'][m2] / P['b_lin_true'][m2] - 1)):.3f}")
for x in sorted(np.unique(P["xi_forcing"]), reverse=True):
    s = m2 & (P["xi_forcing"] == x)
    pr(f"    xi={x:.2f}: b_lin true median {np.median(P['b_lin_true'][s]):.3f} pred {np.median(P['b_lin_mean'][s]):.3f}")

pr("\n" + "=" * 78 + "\n4.3 mom0 vs full\n" + "=" * 78)
M0 = D["final_mom0"]
pr(f"  mom0: R2 lnMs {r2(np.log(M0['ms_true']), np.log(M0['ms_mean'])):.3f}  median frac {np.median(frac(M0)):.3f}")
for name, d in (("mom0", M0), ("proj", P)):
    dl, dc = np.log(d["ms_mean"] / d["ms_true"]), d["chi_mean"] - d["chi_true"]
    s = "  " + name + " r(dlnMs,dchi): " + "  ".join(
        f"{b}:{np.corrcoef(dl[inbin(d, *b)], dc[inbin(d, *b)])[0, 1]:+.2f}" for b in BINS)
    m1 = d["ms_true"] < 1
    pr(s + f"  Ms<1:{np.corrcoef(dl[m1], dc[m1])[0, 1]:+.2f}")
for k, v in sorted(per_run(M0).items(), key=lambda kv: kv[1][0]):
    if 3 <= v[0] <= 7.5 or abs(v[2] - 1) > 0.4:
        pr(f"    mom0 {k:18s} xi={v[1]:.2f} <Ms>={v[0]:6.2f}  pred/true={v[2]:.2f}")
for b in BINS:
    m = inbin(M0, *b)
    pr(f"  mom0 chi {b}: R2={r2(M0['chi_true'][m], M0['chi_mean'][m]):.3f}")
pr(f"  chi MAE all: mom0 {np.mean(np.abs(M0['chi_mean'] - M0['chi_true'])):.3f}  proj {np.mean(np.abs(P['chi_mean'] - P['chi_true'])):.3f}"
   f"   pooled R2 mom0 {r2(M0['chi_true'], M0['chi_mean']):.3f} proj {r2(P['chi_true'], P['chi_mean']):.3f}"
   f"   Ms>=1 R2 mom0 {r2(M0['chi_true'][M0['ms_true'] >= 1], M0['chi_mean'][M0['ms_true'] >= 1]):.3f}")
for tag in ("final_mom0", "final_proj", "final_co13", "final_coboth"):
    d = D[tag]; m = (d["ms_true"] >= 2) & (d["ms_true"] <= 11)
    pr(f"  {tag}: median |dMs| at Ms 2-11 = {np.median(np.abs(d['ms_mean'][m] - d['ms_true'][m])):.2f}")

pr("\n" + "=" * 78 + "\n4.4 calibration (final_proj)\n" + "=" * 78)
from scipy.stats import norm
for tag in ("final_proj",):
    d = D[tag]
    z = np.abs(d["t_ms_y"] - d["t_ms_mu"]) / d["t_ms_sig"]
    pr(f"  raw Gaussian 90% coverage Ms: {np.mean(z <= norm.ppf(0.95)):.3f}")
    levels = np.linspace(0.5, 0.95, 10)
    for k in range(d["n_folds"]):
        f = d["fold"] == k
        for key in ("ms", "chi"):
            zz = np.abs(d[f"t_{key}_y"][f] - d[f"t_{key}_mu"][f]) / d[f"t_{key}_sig"][f]
            raw = np.mean([abs(np.mean(zz <= norm.ppf(0.5 + a / 2)) - a) for a in levels])
            con = np.mean([abs(cov(d, key, L, f) - L / 100) for L in (80, 90, 95)])
            pr(f"    fold {k} {key:3s}: q90={d['q'][k]['ms_ln' if key == 'ms' else 'chi_target']['0.9']:.2f}  "
               f"raw cov err {raw:.3f}  conformal cov err {con:.3f}  cov90 {cov(d, key, 90, f):.3f}")
    pr(f"  pooled conformal Ms 80/90/95: {cov(d, 'ms', 80):.3f} {cov(d, 'ms', 90):.3f} {cov(d, 'ms', 95):.3f}")
    pr(f"  pooled conformal chi 80/90/95: {cov(d, 'chi', 80):.3f} {cov(d, 'chi', 90):.3f} {cov(d, 'chi', 95):.3f}")
    for b in BINS:
        m = inbin(d, *b)
        pr(f"    Ms {b}: cov90 {cov(d, 'ms', 90, m):.3f}  median hi/lo {np.median(d['ms_hi90'][m] / d['ms_lo90'][m]):.2f}")
    pr(f"  chi median 90% width {np.median(d['chi_hi90'] - d['chi_lo90']):.3f}"
       f"  (Ms>=1: {np.median((d['chi_hi90'] - d['chi_lo90'])[d['ms_true'] >= 1]):.3f})")
for tag in ("final_13co", "final_co13", "final_co12", "final_coboth", "final_mom0"):
    pr(f"  {tag} cov90 Ms {cov(D[tag], 'ms', 90):.3f}  chi {cov(D[tag], 'chi', 90):.3f}  "
       f"q90 by fold " + " ".join(f"{q['ms_ln']['0.9']:.2f}" for q in D[tag]["q"]))

pr("\n" + "=" * 78 + "\n4.5 CO\n" + "=" * 78)
for tag in ("final_13co", "final_co13", "final_co12", "final_coboth"):
    d = D[tag]
    pr(f"  {tag}: R2 {r2(np.log(d['ms_true']), np.log(d['ms_mean'])):.3f} med {np.median(frac(d)):.3f}  by bin " +
       " ".join(f"{np.median(frac(d, inbin(d, *b))):.3f}" for b in BINS) +
       "   chi R2 by bin " + " ".join(f"{r2(d['chi_true'][inbin(d, *b)], d['chi_mean'][inbin(d, *b)]):.2f}" for b in BINS))
edges = np.array([0.8, 1.5, 2.5, 4, 6, 10, 16, 30])
for tag in ("final_co13", "final_co12", "final_coboth"):
    d = D[tag]; ok = d["ms_true"] >= 1
    s = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (d["AV_median"] >= lo) & (d["AV_median"] < hi)
        if m.sum() > 20:
            s.append(f"AV {lo:g}-{hi:g}: {np.median(frac(d, m)):.3f}")
    pr(f"  {tag} Ms>=1 median frac err by A_V: " + "; ".join(s))

if not args.no_linewidth:
    pr("\n  Linewidth baseline Ms_obs = sqrt(3) sigma_tile / c_s on run-2 images (masked, intensity-weighted)")
    with h5py.File(f"{args.data_dir}/p79d_mach_grid_256_co_v2_fcrao500.h5", "r") as f:
        cs = float(f.attrs["sound_speed_kms"])
        key = {(a, b, c): i for i, (a, b, c) in enumerate(zip(f["samples/sim_id"][:], f["samples/frame"][:], f["samples/los"][:]))}
        d = D["final_co13"]
        idx = np.array([key[(a, b, c)] for a, b, c in zip(d["sim_id"], d["frame"], d["los"])])
        order = np.argsort(idx)
        res = {}
        for line in ("13co", "12co"):
            sig = np.empty(len(idx))
            ims, msk = f[f"images_obs_{line}"], f[f"mask_obs_{line}"]
            for c0 in range(0, len(idx), 500):
                ii = idx[order[c0:c0 + 500]]
                a, mk = ims[np.sort(ii)], msk[np.sort(ii)]   # h5py needs increasing indices (already sorted)
                for j in range(len(ii)):
                    m = mk[j] & np.isfinite(a[j, 1]) & np.isfinite(a[j, 2])
                    w = np.clip(a[j, 0][m], 0, None)
                    vb = np.sum(w * a[j, 1][m]) / w.sum()
                    sig[order[c0 + j]] = np.sqrt(np.sum(w * (a[j, 2][m] ** 2 + (a[j, 1][m] - vb) ** 2)) / w.sum())
            res[line] = np.sqrt(3) * sig / cs
    for line, tag in (("13co", "final_co13"), ("12co", "final_co12")):
        dd = D[tag]
        e = np.abs(res[line] / d["ms_true"] - 1)
        pr(f"  linewidth {line}: R2 lnMs {r2(np.log(d['ms_true']), np.log(res[line])):.3f}; median frac err by bin " +
           " ".join(f"{b}:{np.median(e[inbin(d, *b)]):.3f}" for b in BINS) +
           f"; median bias at Ms>=3 {np.median(res[line][d['ms_true'] >= 3] / d['ms_true'][d['ms_true'] >= 3] - 1):+.3f}")
        pr(f"  network {tag}: by bin " + " ".join(f"{b}:{np.median(frac(dd, inbin(dd, *b))):.3f}" for b in BINS))
