"""
robustness_net9010.py
=====================
Inference-only checks of the k-fold models on their held-out test folds.

1. Resolution: degrade every test image by factor f (as in training) and predict
   with the resolution channel set correctly (= f) and left at 1 (the network is
   not told). f = 1 is the undegraded baseline.
2. Uncertainty split: MC dropout on a random subset of each test fold, giving
   the aleatoric and epistemic sigma of ln Ms (and chi in network space).

  python robustness_net9010.py --tag final_proj --mode allmom \
      --data_file .../p79d_mach_grid_256_v1.h5 --image_key images --dens_head

Writes <model_dir>/robustness_9010_<tag>_<mode>.npz and a text summary
(.txt) next to it. Uses the trained weights and conformal quantiles of each fold.
"""

import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes/")
import networks_nbisht.net9010 as net

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["allmom", "mom0"], default="allmom")
p.add_argument("--tag", required=True)
p.add_argument("--data_file", required=True)
p.add_argument("--image_key", default="images")
p.add_argument("--dens_head", action="store_true")
p.add_argument("--folds", default="0,1,2,3,4")
p.add_argument("--factors", default="1.0,0.8,0.6,0.4")
p.add_argument("--n_mc_images", type=int, default=1500, help="test images per fold for MC dropout")
p.add_argument("--n_mc", type=int, default=30)
p.add_argument("--model_dir", default="/home/x-nbisht1/projects/p79d_dataset/models")
args = p.parse_args()

net.n_input_channels = 1 if args.mode == "mom0" else 3
net.input_mode       = args.mode
net.n_lines          = len(args.image_key.split("+"))
net.n_model_channels = net.n_lines * net.n_input_channels + 1
net.dens_head        = args.dens_head
net.data_file, net.image_key, net.run_tag = args.data_file, args.image_key, args.tag
factors = [float(f) for f in args.factors.split(",")]
print(f"robustness | {args.tag} {args.mode} | threads {torch.get_num_threads()} | factors {factors}", flush=True)


class FixedResDataset(Dataset):
    """Test images degraded by `factor`; the resolution channel is set to `res_value`."""
    def __init__(self, split, factor, res_value):
        self.data, self.factor, self.res_value = split["data"], factor, res_value
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, i):
        x = self.data[i][net.input_channel_index()].clone()
        if self.factor < 1.0:
            x = net._degrade_resolution(x, self.factor)
        x = net.preprocess_lines(x)
        r = torch.full((1, x.shape[-2], x.shape[-1]), self.res_value, dtype=x.dtype)
        return torch.cat([x, r], 0)


def run(model, ds):
    mu_ms, sg_ms, mu_c, sg_c = [], [], [], []
    with torch.no_grad():
        for xb in DataLoader(ds, batch_size=256, shuffle=False, num_workers=8):
            (a, la), (b, lb) = model(xb.to(net.device))[:2]
            mu_ms.append(a.view(-1)); sg_ms.append(torch.exp(0.5 * la).view(-1))
            mu_c.append(b.view(-1));  sg_c.append(torch.exp(0.5 * lb).view(-1))
    return [torch.cat(v).cpu().numpy().astype(np.float64) for v in (mu_ms, sg_ms, mu_c, sg_c)]


splits_cache = {}
out = {k: [] for k in ("fold", "ms_true", "chi_true", "sim_id")}
for f in factors:
    for r in ("ok", "one"):
        for k in ("mu_ms", "sg_ms", "mu_chi", "sg_chi"):
            out[f"{k}_{f:g}_{r}"] = []
mc = {k: [] for k in ("fold", "ms_true", "chi_true", "mu_ms", "alea_ms", "epis_ms", "mu_chi", "alea_chi", "epis_chi")}
q90 = {}

for fold in [int(s) for s in args.folds.split(",")]:
    t0 = time.time()
    splits = net.load_data_grouped(fold, 5, calib=True)
    test = splits["test"]
    model = net.thisnet()
    sd = torch.load(os.path.join(args.model_dir, f"test{net.idd}_{args.tag}_{args.mode}_fold{fold}.pth"),
                    map_location=net.device)
    for k in ("train_curve", "val_curve"):
        setattr(model, k, torch.zeros_like(sd[k]))
    model.load_state_dict(sd); model.eval()
    with open(os.path.join(args.model_dir, f"conformal_quantiles_{net.idd}_{args.tag}_{args.mode}_fold{fold}.json")) as fh:
        q90[fold] = json.load(fh)["ms_ln"]["0.9"]

    n = len(test["data"])
    out["fold"].append(np.full(n, fold)); out["ms_true"].append(test["Ms_act"])
    out["chi_true"].append(test["chi_act"]); out["sim_id"].append(test["sim_id"])
    for f in factors:
        for r, rv in (("ok", f), ("one", 1.0)):
            if f == 1.0 and r == "one":          # identical to the baseline
                res = run_ok
            else:
                res = run(model, FixedResDataset(test, f, rv))
            if f == 1.0 and r == "ok":
                run_ok = res
            for k, v in zip(("mu_ms", "sg_ms", "mu_chi", "sg_chi"), res):
                out[f"{k}_{f:g}_{r}"].append(v)
            e = np.median(np.abs(np.exp(res[0]) / test["Ms_act"] - 1))
            print(f"  fold {fold} f={f:g} res={r:3s}: median |dMs|/Ms = {e:.3f}  "
                  f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

    rng = np.random.default_rng(fold)
    sel = np.sort(rng.choice(n, min(args.n_mc_images, n), replace=False))
    xs = torch.stack([FixedResDataset(test, 1.0, 1.0)[i] for i in sel])
    parts = []
    for b in range(0, len(sel), 256):
        parts.append(model.mc_forward(xs[b:b + 256].to(net.device), args.n_mc))
    (m_ms, a_ms, e_ms, _), (m_c, a_c, e_c, _) = [
        [torch.cat([pp[h][j].view(-1) for pp in parts]).cpu().numpy() for j in range(4)] for h in (0, 1)]
    for k, v in zip(("mu_ms", "alea_ms", "epis_ms", "mu_chi", "alea_chi", "epis_chi"),
                    (m_ms, a_ms, e_ms, m_c, a_c, e_c)):
        mc[k].append(v)
    mc["fold"].append(np.full(len(sel), fold)); mc["ms_true"].append(test["Ms_act"][sel])
    mc["chi_true"].append(test["chi_act"][sel])
    print(f"  fold {fold} done in {(time.time() - t0) / 60:.1f} min", flush=True)

out = {k: np.concatenate(v) for k, v in out.items()}
mc = {k: np.concatenate(v) for k, v in mc.items()}
base = os.path.join(args.model_dir, f"robustness_{net.idd}_{args.tag}_{args.mode}")
np.savez(base + ".npz", factors=np.array(factors), q90=np.array([q90[k] for k in sorted(q90)]),
         **out, **{f"mc_{k}": v for k, v in mc.items()})

# ─── summary ────────────────────────────────────────────────────────────────
lines = [f"{args.tag} {args.mode}: {len(out['ms_true'])} test images, folds {sorted(q90)}", ""]
qf = np.array([q90[int(k)] for k in out["fold"]])
y = np.log(out["ms_true"])
lines.append(f"{'f':>5} {'res':>4} {'med|dMs|/Ms':>12} {'lnMs R2':>8} {'bias lnMs':>10} {'cov90':>6} "
             f"{'med|dMs|/Ms 2-11':>17}")
for f in factors:
    for r in (("ok",) if f == 1.0 else ("ok", "one")):
        mu, sg = out[f"mu_ms_{f:g}_{r}"], out[f"sg_ms_{f:g}_{r}"]
        err = np.abs(np.exp(mu) / out["ms_true"] - 1)
        r2 = 1 - np.sum((mu - y) ** 2) / np.sum((y - y.mean()) ** 2)
        cov = np.mean(np.abs(mu - y) <= qf * sg)
        mid = (out["ms_true"] >= 2) & (out["ms_true"] <= 11)
        lines.append(f"{f:5.2f} {r:>4} {np.median(err):12.3f} {r2:8.4f} {np.median(mu - y):10.3f} {cov:6.3f} "
                     f"{np.median(err[mid]):17.3f}")
lines += ["", "chi (Ms >= 1), R2 in chi:"]
u = out["ms_true"] >= 1
for f in factors:
    for r in (("ok",) if f == 1.0 else ("ok", "one")):
        c = net.chi_from_target(out[f"mu_chi_{f:g}_{r}"])
        t = out["chi_true"]
        lines.append(f"  f={f:4.2f} res={r:3s} R2={1 - np.sum((c[u] - t[u])**2) / np.sum((t[u] - t[u].mean())**2):.3f}")

lines += ["", "MC dropout, sigma of ln Ms by Ms bin (medians):",
          f"  {'Ms bin':<12}{'N':>6}{'alea':>8}{'epis':>8}{'epis/alea':>10}{'|resid|':>9}"]
for lo, hi in [(0, 0.5), (0.5, 1), (1, 3), (3, 7), (7, 20.01)]:
    b = (mc["ms_true"] >= lo) & (mc["ms_true"] < hi)
    if b.sum() > 2:
        res_ = np.abs(mc["mu_ms"][b] - np.log(mc["ms_true"][b]))
        lines.append(f"  [{lo:>4}, {hi:>5.1f}){b.sum():6d}{np.median(mc['alea_ms'][b]):8.3f}"
                     f"{np.median(mc['epis_ms'][b]):8.3f}{np.median(mc['epis_ms'][b] / mc['alea_ms'][b]):10.2f}"
                     f"{np.median(res_):9.3f}")
lines.append(f"  all: epistemic fraction of variance = "
             f"{np.median(mc['epis_ms']**2 / (mc['epis_ms']**2 + mc['alea_ms']**2)):.3f} (median)")
open(base + ".txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Saved {base}.npz/.txt")
