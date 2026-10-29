"""
compare_lvg_lte.py
==================
Size of the LTE bias in run 2 (step 8): the same frames rendered with LVG
level populations (radmc_co_lvg.json) against the LTE run-2 images.

Inputs (both re-observed with the same telescope setup, so beam, channels and
the noise realisation are identical; only the level populations differ):
  LVG  <output_dir>/p79d_mach_grid_256_co_lvg_<obs>.h5
  LTE  <output_dir>/p79d_mach_grid_256_co_v2_<obs>.h5

Per image and line:
  * noise-free peak T_B and integrated intensity <W> (LVG / LTE);
  * detected fraction after masking;
  * median sigma_v (moment 2) and the tile-spectrum Mach number
    Ms_obs = sqrt(3) sigma_tile / c_s;
  * predictions of the run-2 models (co2_13, co2_12, co2_both; --models fold
    tag, default foldall) on both versions.

  python extract_sim_co.py reobserve --config sims_mach_grid.json --rt radmc_co_lvg.json --obs fcrao500
  python extract_sim_co.py merge     --config sims_mach_grid.json --rt radmc_co_lvg.json --obs fcrao500
  python compare_lvg_lte.py --obs fcrao500

Writes <out>/lvg_lte_images.csv and lvg_lte_summary.txt.
"""
import argparse, csv, json, os, sys
import numpy as np
import h5py
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networks_nbisht.net9010 as net

p = argparse.ArgumentParser()
p.add_argument("--obs", default="fcrao500")
p.add_argument("--data_dir", default="/home/x-nbisht1/scratch/projects/p79d_dataset")
p.add_argument("--model_dir", default="/home/x-nbisht1/projects/p79d_dataset/models")
p.add_argument("--tags", nargs="*", default=["co2_13", "co2_12", "co2_both"])
p.add_argument("--models", default="foldall")
p.add_argument("--out", default="/home/x-nbisht1/plots/lvg_check")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
LINES = ("13co", "12co")


def load(path):
    f = h5py.File(path, "r")
    names = f["sims/name"][:].astype(str)
    key = list(zip(names[f["samples/sim_id"][:]], f["samples/frame"][:], f["samples/los"][:], f["samples/aug_index"][:]))
    return f, {k: i for i, k in enumerate(key)}


f_lvg, k_lvg = load(f"{args.data_dir}/p79d_mach_grid_256_co_lvg_{args.obs}.h5")
f_lte, k_lte = load(f"{args.data_dir}/p79d_mach_grid_256_co_v2_{args.obs}.h5")
pairs = [(k, i, k_lte[k]) for k, i in k_lvg.items() if k in k_lte]
print(f"{len(pairs)} images in both versions ({len(k_lvg)} LVG)")
cs = float(f_lte.attrs["sound_speed_kms"])
I_lvg, I_lte = np.array([p[1] for p in pairs]), np.array([p[2] for p in pairs])


def take(f, key, idx):
    o = np.argsort(idx)
    out = f[key][idx[o]]
    return out[np.argsort(o)]


rows = []
for (sim, frame, los, aug), il, it in pairs:
    rows.append({"sim": sim, "frame": int(frame), "los": int(los), "aug": int(aug),
                 "n_H2": float(f_lte["labels/n_H2"][it]), "Ms": float(f_lte["labels/Ms"][it]),
                 "chi_v": float(f_lte["labels/chi_v"][it]), "AV_median": float(f_lte["rt/AV_median"][it])})

for line in LINES:
    for tag, f, idx in (("lvg", f_lvg, I_lvg), ("lte", f_lte, I_lte)):
        clean = take(f, f"images_{line}", idx)               # noise-free [W/<W>, v_c/c_s, sigma/c_s]
        obs = take(f, f"images_obs_{line}", idx)             # [K km/s, km/s, km/s], NaN masked
        mask = take(f, f"mask_obs_{line}", idx)
        tpk = take(f, f"rt/Tpeak_{line}", idx)
        for j, r in enumerate(rows):
            W = obs[j, 0]
            r[f"Tpeak_{line}_{tag}"] = float(tpk[j])
            r[f"W_mean_{line}_{tag}"] = float(np.nanmean(np.where(mask[j], W, 0.0)))
            r[f"det_{line}_{tag}"] = float(mask[j].mean())
            r[f"sigma_med_{line}_{tag}"] = float(np.nanmedian(obs[j, 2][mask[j]])) if mask[j].any() else np.nan
            # tile spectrum from the moments: intensity-weighted centroid spread + internal widths
            if mask[j].any():
                w, v, s = W[mask[j]], obs[j, 1][mask[j]], obs[j, 2][mask[j]]
                vbar = np.sum(w * v) / w.sum()
                sig_tile = np.sqrt(np.sum(w * ((v - vbar)**2 + s**2)) / w.sum())
                r[f"Ms_obs_{line}_{tag}"] = float(np.sqrt(3) * sig_tile / cs)
            else:
                r[f"Ms_obs_{line}_{tag}"] = np.nan

# network predictions on both versions
for mt in args.tags:
    qf = f"{args.model_dir}/conformal_quantiles_9010_{mt}_allmom_{args.models}.json"
    ck = f"{args.model_dir}/test9010_{mt}_allmom_{args.models}.pth"
    if not (os.path.exists(qf) and os.path.exists(ck)):
        print(f"  skip {mt}: no {args.models} model"); continue
    q = json.load(open(qf))
    keys = q["image_key"].split("+")
    net.n_input_channels, net.input_mode = 3, "allmom"
    net.n_lines, net.dens_head = len(keys), q.get("dens_head", False)
    net.n_model_channels = net.n_lines * 3 + 1
    net.chi_field, net.chi_space = q["chi_field"], q["chi_space"]
    sd = torch.load(ck, map_location=net.device)
    m = net.thisnet()
    for k in ("train_curve", "val_curve"):
        setattr(m, k, torch.zeros_like(sd[k]))
    m.load_state_dict(sd); m.eval()
    for tag, f, idx in (("lvg", f_lvg, I_lvg), ("lte", f_lte, I_lte)):
        parts = [net.obs_to_training_units(take(f, k, idx), take(f, k.replace("images_obs", "mask_obs"), idx), cs)
                 for k in keys]
        x = net.preprocess_lines(torch.cat(parts, dim=1))
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1).to(net.device)    # native resolution: r = 1
        with torch.no_grad():
            res = [m(x[i:i + 32]) for i in range(0, len(x), 32)]
        ms = torch.cat([r_[0][0].view(-1) for r_ in res]).exp().cpu().numpy()
        chi = net.chi_from_target(torch.cat([r_[1][0].view(-1) for r_ in res]).cpu().numpy())
        for j, r in enumerate(rows):
            r[f"Ms_{mt}_{tag}"], r[f"chi_{mt}_{tag}"] = float(ms[j]), float(chi[j])

with open(os.path.join(args.out, "lvg_lte_images.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# summary per frame: medians over its images of LVG/LTE ratios and differences
lines_out = [f"LVG vs LTE, run 2, re-observed as {args.obs}; median over each frame's images "
             f"(ratio = LVG/LTE, d = LVG - LTE); models {args.models}", ""]
frames = sorted({(r["sim"], r["frame"]) for r in rows}, key=lambda k: np.median([r["n_H2"] for r in rows if (r["sim"], r["frame"]) == k]))
hdr = f"{'sim':18s} {'frame':>5s} {'n_H2':>6s} {'Ms':>5s} {'AV':>5s}"
for line in LINES:
    hdr += f" | {line}: {'Tpk':>5s} {'W':>5s} {'det_l/t':>9s} {'sig':>5s} {'Ms_obs':>6s}"
lines_out.append(hdr)
for sim, fr in frames:
    R = [r for r in rows if (r["sim"], r["frame"]) == (sim, fr)]
    med = lambda k: float(np.nanmedian([r[k] for r in R]))
    s = f"{sim:18s} {fr:5d} {med('n_H2'):6.0f} {med('Ms'):5.2f} {med('AV_median'):5.1f}"
    for line in LINES:
        rt = lambda k: float(np.nanmedian([r[f"{k}_{line}_lvg"] / r[f"{k}_{line}_lte"] for r in R]))
        s += (f" | {line}: {rt('Tpeak'):5.2f} {rt('W_mean'):5.2f} {med(f'det_{line}_lvg'):4.2f}/{med(f'det_{line}_lte'):4.2f}"
              f" {rt('sigma_med'):5.2f} {rt('Ms_obs'):6.2f}")
    lines_out.append(s)
lines_out += ["", "Network predictions (median over images): Ms LVG/LTE ratio, chi LVG - LTE, and Ms_pred/Ms_true for each"]
for mt in args.tags:
    if f"Ms_{mt}_lvg" not in rows[0]:
        continue
    lines_out.append(f"  {mt}:")
    for sim, fr in frames:
        R = [r for r in rows if (r["sim"], r["frame"]) == (sim, fr)]
        a = np.array([[r[f"Ms_{mt}_lvg"], r[f"Ms_{mt}_lte"], r[f"chi_{mt}_lvg"], r[f"chi_{mt}_lte"], r["Ms"]] for r in R])
        lines_out.append(f"    {sim:18s} {fr:5d} n={R[0]['n_H2']:6.0f}  Ms ratio {np.median(a[:, 0] / a[:, 1]):5.2f}  "
                         f"d chi {np.median(a[:, 2] - a[:, 3]):+6.3f}   Ms/true LVG {np.median(a[:, 0] / a[:, 4]):5.2f} "
                         f"LTE {np.median(a[:, 1] / a[:, 4]):5.2f}")
open(os.path.join(args.out, "lvg_lte_summary.txt"), "w").write("\n".join(lines_out) + "\n")
print("\n".join(lines_out))
