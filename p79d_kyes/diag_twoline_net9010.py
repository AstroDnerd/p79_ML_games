"""
Why does the two-line (13CO + 12CO) model read higher than the 13CO model on real MWISP tiles
(1.56 vs 1.17 times the tile linewidth), when on synthetic MWISP-like maps it does not (1.06 vs 1.14)?

The network sees, per line, ln(W/<W>), v_c - <v_c> and ln(sigma) (units of c_s), so the absolute
12CO/13CO intensity ratio and any velocity offset between the lines are NOT inputs. What can differ
between real and synthetic pairs is the relation between the two lines' maps. This script

 1. computes summary statistics of the network inputs of each tile (per line and between lines) for
    the real MWISP tiles (dumped by predict_observations_net9010.py --dump_inputs) and for held-out
    synthetic MWISP-like images, and places each real tile in the synthetic distribution;
 2. intervenes on the inputs: moves one between-line statistic of the real tiles to the synthetic
    median (and the reverse on the synthetic images) and records how the two-line prediction moves.

Synthetic predictions are out of fold; real tiles use the all-runs model (foldfull).
"""
import argparse, json, os, sys
import numpy as np, h5py, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networks_nbisht.net9010 as net

p = argparse.ArgumentParser()
p.add_argument("--real", default="/home/x-nbisht1/projects/p79d_dataset/models/mwisp_inputs_final_coboth.npz")
p.add_argument("--syn", default="/home/x-nbisht1/scratch/projects/p79d_dataset/p79d_mach_grid_256_co_v2_mwisp.h5")
p.add_argument("--model_dir", default="/home/x-nbisht1/projects/p79d_dataset/models")
p.add_argument("--tag", default="final_coboth")
p.add_argument("--n_per_sim", type=int, default=30)
p.add_argument("--ms_min", type=float, default=2.0)
p.add_argument("--max_tile_pc", type=float, default=10.0)
p.add_argument("--out", default="/home/x-nbisht1/projects/p79d_dataset/models/diag_twoline.txt")
args = p.parse_args()
rng = np.random.default_rng(1)
OUT = open(args.out, "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); OUT.write(s + "\n"); OUT.flush()


net.n_input_channels, net.input_mode, net.n_lines, net.n_model_channels = 3, "allmom", 2, 7


def load(fold):
    q = json.load(open(f"{args.model_dir}/conformal_quantiles_9010_{args.tag}_allmom_fold{fold}.json"))
    net.dens_head, net.chi_field, net.chi_space = q.get("dens_head", False), q["chi_field"], q["chi_space"]
    sd = torch.load(f"{args.model_dir}/test9010_{args.tag}_allmom_fold{fold}.pth", map_location=net.device)
    m = net.thisnet()
    for k in ("train_curve", "val_curve"):
        setattr(m, k, torch.zeros_like(sd[k]))
    m.load_state_dict(sd); m.eval()
    return m


def predict(m, X):
    X = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return np.concatenate([m(X[a:a + 32].to(net.device))[0][0].view(-1).exp().cpu().numpy()
                               for a in range(0, len(X), 32)])


def stats(X, det):
    """X [N, 7, H, W]: (lnW, v_c, ln sigma) of 13CO then 12CO, then r. det [N, 2] detected fractions."""
    w13, c13, s13, w12, c12, s12 = (X[:, i] for i in range(6))
    f = lambda a: a.reshape(len(a), -1)
    corr = lambda a, b: np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(f(a), f(b))])
    rms = lambda a: np.sqrt((f(a) ** 2).mean(1))
    return {
        "std lnW 13":        f(w13).std(1),
        "std lnW 12":        f(w12).std(1),
        "std v_c 13":        f(c13).std(1),
        "std v_c 12":        f(c12).std(1),
        "mean ln sig 13":    f(s13).mean(1),
        "mean ln sig 12":    f(s12).mean(1),
        "d mean ln sig":     f(s12).mean(1) - f(s13).mean(1),          # ln of the 12CO/13CO width ratio
        "ln std v_c 12/13":  np.log(f(c12).std(1) / f(c13).std(1)),
        "ln std lnW 12/13":  np.log(f(w12).std(1) / f(w13).std(1)),
        "corr v_c":          corr(c13, c12),
        "corr lnW":          corr(w13, w12),
        "corr ln sig":       corr(s13, s12),
        "rms v_c12-v_c13":   rms(c12 - c13),
        "det 13":            det[:, 0],
        "det 12 - det 13":   det[:, 1] - det[:, 0],
    }


# ─── real tiles ──────────────────────────────────────────────────────────────
R = np.load(args.real, allow_pickle=True)
assert list(R["lines"]) == ["13co", "12co"], R["lines"]
keep = R["tile_pc"] <= args.max_tile_pc
Xr, detr, msobs_r, cloud_r = R["X"][keep], R["det"][keep], R["Ms_tile_spectrum"][keep], R["cloud"][keep]
m_full = load("full")
ms_r = predict(m_full, Xr)
say(f"real: {len(Xr)} tiles of {len(set(cloud_r))} clouds with tiles <= {args.max_tile_pc} pc; "
    f"two-line / Ms_obs: tile median {np.median(ms_r / msobs_r):.2f}")

# ─── synthetic MWISP-like images (held-out predictions) ──────────────────────
f = h5py.File(args.syn, "r")
cs = float(f.attrs["sound_speed_kms"])
sid, ms = f["samples/sim_id"][:], f["labels/Ms"][:]
fold_of = net.assign_sim_folds(f["sims/Ms_mean"][:], f["sims/xi"][:], 5)
pick = []
for g in np.unique(sid):
    c = np.where((sid == g) & (ms >= args.ms_min) & (ms <= net.ms_max))[0]
    if len(c):
        pick += list(rng.choice(c, min(args.n_per_sim, len(c)), replace=False))
pick = np.sort(np.array(pick))
Xs, dets, msobs_s = [], [], []
for i in pick:
    blocks, d = [], []
    for line in ("13co", "12co"):
        o, mk = f[f"images_obs_{line}"][i], f[f"mask_obs_{line}"][i]
        blocks.append(net.preprocess_maps(net.obs_to_training_units(o[None], mk[None], cs)[0]).numpy())
        d.append(mk.mean())
        if line == "13co":
            g_ = mk & np.isfinite(o[1]) & np.isfinite(o[2])
            w = np.clip(o[0][g_], 0, None); vb = (w * o[1][g_]).sum() / w.sum()
            msobs_s.append(np.sqrt(3) * np.sqrt((w * (o[2][g_] ** 2 + (o[1][g_] - vb) ** 2)).sum() / w.sum()) / cs)
    Xs.append(np.concatenate(blocks + [np.ones((1, 128, 128), np.float32)])); dets.append(d)
Xs, dets, msobs_s, ms_true = np.stack(Xs), np.array(dets), np.array(msobs_s), ms[pick]
folds = fold_of[sid[pick]]
models = {k: load(k) for k in sorted(set(folds))}
ms_s = np.empty(len(Xs))
for k, m in models.items():
    ms_s[folds == k] = predict(m, Xs[folds == k])
say(f"synthetic: {len(Xs)} images of {len(set(sid[pick]))} runs, Ms >= {args.ms_min}; two-line / Ms_obs median "
    f"{np.median(ms_s / msobs_s):.2f}, two-line / true {np.median(ms_s / ms_true):.2f}")

# ─── 1. where the real tiles sit in the synthetic distribution ───────────────
Sr, Ss = stats(Xr, detr), stats(Xs, dets)
ratio_r = ms_r / msobs_r
from scipy.stats import spearmanr
say("\nInput statistics. Synthetic images are restricted to the Ms_obs range of the real tiles "
    f"({msobs_r.min():.1f}-{msobs_r.max():.1f}).")
say(f"{'statistic':20s} {'real med':>9s} {'[16, 84]':>16s} {'syn med':>9s} {'[2.5, 97.5]':>18s} "
    f"{'real outside':>12s} {'rho(ratio)':>10s}")
inr = (msobs_s >= msobs_r.min()) & (msobs_s <= msobs_r.max())
for k in Sr:
    a, b = Sr[k], Ss[k][inr]
    lo, hi = np.percentile(b, [2.5, 97.5])
    outside = np.mean((a < lo) | (a > hi))
    say(f"{k:20s} {np.median(a):9.3f} {str(np.round(np.percentile(a, [16, 84]), 3)):>16s} {np.median(b):9.3f} "
        f"{str(np.round([lo, hi], 3)):>18s} {outside:12.2f} {spearmanr(a, ratio_r)[0]:10.2f}")
say("('real outside' = fraction of real tiles outside the central 95% of the synthetic images; "
    "rho = Spearman correlation of the statistic with two-line / Ms_obs over the real tiles)")


# ─── 2. interventions ────────────────────────────────────────────────────────
def shift_sig12(X, d):            # multiply the 12CO width by exp(d)
    Y = X.copy(); Y[:, 5] += d; return Y


def scale_vc12(X, fac):           # scale the 12CO centroid fluctuations
    Y = X.copy(); Y[:, 4] *= fac; return Y


def scale_w12(X, fac):            # scale the 12CO ln W contrast (then restore <W> = 1)
    Y = X.copy(); w = np.exp(Y[:, 3] * fac); Y[:, 3] = np.log(w / w.mean((1, 2), keepdims=True)); return Y


def copy13(X, ch):                # replace 12CO channels by the 13CO ones
    Y = X.copy()
    for c in ch:
        Y[:, 3 + c] = Y[:, c]
    return Y


d_sig = np.median(Sr["d mean ln sig"]) - np.median(Ss["d mean ln sig"][inr])
d_vc  = np.median(Sr["ln std v_c 12/13"]) - np.median(Ss["ln std v_c 12/13"][inr])
d_w   = np.median(Sr["ln std lnW 12/13"]) - np.median(Ss["ln std lnW 12/13"][inr])
say(f"\nReal minus synthetic medians: d ln(sig12/sig13) = {d_sig:+.3f}, d ln(std v_c 12/13) = {d_vc:+.3f}, "
    f"d ln(std lnW 12/13) = {d_w:+.3f}")
say("\nInterventions: median two-line / Ms_obs")
say(f"{'':52s} {'real':>6s} {'synthetic':>10s}")
ps = lambda X: np.concatenate([predict(m, X[folds == k]) for k, m in models.items()])
order = np.concatenate([np.where(folds == k)[0] for k in models])
ms_obs_s_o = msobs_s[order]
def line(name, Yr, Ys):
    a = np.median(predict(m_full, Yr) / msobs_r) if Yr is not None else np.nan
    b = np.median(ps(Ys) / ms_obs_s_o) if Ys is not None else np.nan
    say(f"{name:52s} {a:6.2f} {b:10.2f}")
line("as observed", Xr, Xs)
line("12CO width moved to the other set's 12/13 ratio", shift_sig12(Xr, -d_sig), shift_sig12(Xs, +d_sig))
line("12CO centroid spread moved likewise", scale_vc12(Xr, np.exp(-d_vc)), scale_vc12(Xs, np.exp(d_vc)))
line("12CO ln W contrast moved likewise", scale_w12(Xr, np.exp(-d_w)), scale_w12(Xs, np.exp(d_w)))
line("all three moved", scale_w12(scale_vc12(shift_sig12(Xr, -d_sig), np.exp(-d_vc)), np.exp(-d_w)),
     scale_w12(scale_vc12(shift_sig12(Xs, d_sig), np.exp(d_vc)), np.exp(d_w)))
line("12CO width map replaced by the 13CO one", copy13(Xr, [2]), copy13(Xs, [2]))
line("12CO centroid map replaced by the 13CO one", copy13(Xr, [1]), copy13(Xs, [1]))
line("12CO ln W map replaced by the 13CO one", copy13(Xr, [0]), copy13(Xs, [0]))
line("all 12CO maps replaced by the 13CO ones", copy13(Xr, [0, 1, 2]), copy13(Xs, [0, 1, 2]))
say("\n12CO width shifted by d (ln units): median two-line / Ms_obs")
for d in (-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4):
    line(f"  d = {d:+.1f}", shift_sig12(Xr, d), shift_sig12(Xs, d))
OUT.close()
