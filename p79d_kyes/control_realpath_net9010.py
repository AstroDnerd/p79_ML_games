"""
control_realpath_net9010.py
===========================
Control for the offset of the CO-trained models on real clouds.

On real FCRAO-like maps the beam spans 2 pixels, so predict_observations_net9010.py
upsamples the MOMENT MAPS by ~1.55 (nearest-pixel velocity fill, bilinear zoom) until
the beam spans the 3.1 pixels of the training maps. The training maps never went
through that path: they are made at 3.1 pixels per beam with white noise per pixel.
The simulated re-observations used so far as a control are fed directly, so they do
not test it.

Here synthetic 13CO cubes of held-out simulations are observed the way a real map is:

  1. beam convolution (training beam, 3.1 px, periodic);
  2. spatial resampling to 2 pixels per beam (like 46'' on 23'' pixels);
  3. white noise per coarse pixel and channel;
  4. the observation pipeline's moments (3 sigma, contiguous channels);
  5. the real-data path: nearest-pixel velocity fill, bilinear upsampling of the
     moment maps by 3.1/2, fill, training units, r = 1;
  6. prediction by the k-fold model that held the simulation out.

The same model's prediction on the standard training-style image of the same
snapshot and line of sight is the reference. If the ratio path/standard is ~1 the
real-data path is not the cause of the offset; if it is ~1.2 it is.

  python control_realpath_net9010.py [--tag final_co13] [--n_per_sim 12] [--noise 0.2]

Writes <out>/control_realpath_<tag>.csv and .txt.
"""
import argparse, contextlib, csv, io, json, os, sys
import numpy as np
import h5py
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networks_nbisht.net9010 as net
import get_Observation_Data as G

p = argparse.ArgumentParser()
p.add_argument("--tag", default="final_co13")
p.add_argument("--line", default="13co")
p.add_argument("--data", default="/home/x-nbisht1/scratch/projects/p79d_dataset/p79d_mach_grid_256_co_v2_fcrao500.h5")
p.add_argument("--model_dir", default="/home/x-nbisht1/projects/p79d_dataset/models")
p.add_argument("--n_per_sim", type=int, default=12)
p.add_argument("--ms_min", type=float, default=2.0)
p.add_argument("--beam_pix_obs", type=float, default=2.0, help="beam FWHM in pixels of the emulated real map")
p.add_argument("--noise", type=float, default=0.2, help="K per coarse pixel and channel")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out", default="/home/x-nbisht1/plots/control_realpath")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed)

f = h5py.File(args.data, "r")
rt, ob = json.loads(f.attrs["rt_json"]), json.loads(f.attrs["observe_version"])
cs = float(f.attrs["sound_speed_kms"])
pix = (rt["box_length_pc"] / rt["target_res"]) / ob["distance_pc"] * 206265.0
beam_pix = ob["beam_fwhm_arcsec"] / pix                       # 3.1 for fcrao500
down = args.beam_pix_obs / beam_pix                           # < 1: coarser grid, like the real map
up = beam_pix / args.beam_pix_obs                             # what predict_observations applies
n_run = G.min_channels_for(ob["min_channels"], ob["channel_width_kms"])
names = f["sims/name"][:].astype(str)
sim_fold = net.assign_sim_folds(f["sims/Ms_mean"][:], f["sims/xi"][:], 5)
sid, ms, los, aug = f["samples/sim_id"][:], f["labels/Ms"][:], f["samples/los"][:], f["samples/aug_index"][:]
cfile, cgroup = f["samples/cube_file"][:].astype(str), f["samples/cube_group"][:].astype(str)
cube_dir = f.attrs["cube_dir"]

pick = []
for g in np.unique(sid):
    c = np.where((sid == g) & (ms >= args.ms_min) & (ms <= net.ms_max))[0]
    if len(c):
        pick += list(rng.choice(c, min(args.n_per_sim, len(c)), replace=False))
pick = np.sort(np.array(pick))
print(f"{len(pick)} images of {len(np.unique(sid[pick]))} simulations; beam {beam_pix:.2f} px -> {args.beam_pix_obs} px "
      f"(x{down:.3f}), upsampled x{up:.3f}; min run {n_run} channels; noise {args.noise} K", flush=True)


def tile_ms(m0, m1, m2, mask):
    w, v, s = np.clip(m0[mask], 0, None), m1[mask], m2[mask]
    vb = np.sum(w * v) / w.sum()
    return float(np.sqrt(3) * np.sqrt(np.sum(w * (s ** 2 + (v - vb) ** 2)) / w.sum()) / cs)


def real_path(T, v):
    """Steps 1-5 of the docstring for one noise-free [x, y, v] cube. Returns the network input and diagnostics."""
    Tb = gaussian_filter(T, (beam_pix / 2.3548, beam_pix / 2.3548, 0), mode="wrap")
    Tc = zoom(Tb, (down, down, 1), order=1, mode="grid-wrap", grid_mode=True)
    Tc = Tc + rng.normal(0, args.noise, Tc.shape)
    with contextlib.redirect_stdout(io.StringIO()):
        m0, m1, m2, mask = G.compute_moment_maps(Tc.transpose(2, 1, 0), v, noise_threshold=ob["mask_sigma"], min_channels=n_run)
    m0, m1, m2, mask = m0.T, m1.T, m2.T, mask.T
    if mask.mean() < 0.05:
        return None
    ms_obs = tile_ms(m0, m1, m2, mask & np.isfinite(m1) & np.isfinite(m2))
    good = mask & np.isfinite(m1) & np.isfinite(m2)                 # regrid() of predict_observations_net9010.py
    _, (iy, ix) = distance_transform_edt(~good, return_indices=True)
    u1, u2 = zoom(m1[iy, ix], up, order=1), zoom(m2[iy, ix], up, order=1)
    umask = zoom(mask.astype(np.uint8), up, order=0).astype(bool)
    u0 = zoom(m0, up, order=1)
    u1, u2 = np.where(umask, u1, np.nan), np.where(umask, u2, np.nan)
    n = net.img_size
    if min(u0.shape) < n:
        return None
    u0, u1, u2, umask = u0[:n, :n], u1[:n, :n], u2[:n, :n], umask[:n, :n]
    floor = np.nanpercentile(u0[umask], 1)
    a0, a1, a2 = net.fill_masked_moments(u0, u1, u2, umask, floor)
    x = net.preprocess_maps(torch.tensor(np.stack([a0, a1 / cs, a2 / cs]), dtype=torch.float32))
    return x, ms_obs, float(umask.mean())


rows, X_path, X_std = [], [], []
for n_done, i in enumerate(pick):
    with h5py.File(os.path.join(cube_dir, cfile[i]), "r") as h:
        gq = h[cgroup[i]]
        T = gq[f"cube_{args.line}_{los[i]}_{aug[i]}"][()].astype(np.float32)
        v = gq[f"v_{args.line}_{los[i]}"][()].astype(np.float64)
    out = real_path(T, v)
    if out is None:
        continue
    o, mk = f[f"images_obs_{args.line}"][i], f[f"mask_obs_{args.line}"][i]
    x_std = net.preprocess_maps(net.obs_to_training_units(o[None], mk[None], cs)[0])
    good = mk & np.isfinite(o[1]) & np.isfinite(o[2])
    rows.append({"sim": names[sid[i]], "fold": int(sim_fold[sid[i]]), "Ms_true": float(ms[i]),
                 "det_path": out[2], "det_std": float(mk.mean()),
                 "Ms_obs_path": out[1], "Ms_obs_std": tile_ms(o[0], o[1], o[2], good)})
    X_path.append(out[0]); X_std.append(x_std)
    if n_done % 50 == 0:
        print(f"  {n_done}/{len(pick)}", flush=True)

net.n_input_channels, net.input_mode, net.n_lines = 3, "allmom", 1
net.n_model_channels = 4
folds = np.array([r["fold"] for r in rows])
for k in sorted(set(folds)):
    q = json.load(open(f"{args.model_dir}/conformal_quantiles_9010_{args.tag}_allmom_fold{k}.json"))
    net.dens_head, net.chi_field, net.chi_space = q.get("dens_head", False), q["chi_field"], q["chi_space"]
    sd = torch.load(f"{args.model_dir}/test9010_{args.tag}_allmom_fold{k}.pth", map_location=net.device)
    m = net.thisnet()
    for kk in ("train_curve", "val_curve"):
        setattr(m, kk, torch.zeros_like(sd[kk]))
    m.load_state_dict(sd); m.eval()
    sel = np.where(folds == k)[0]
    for key, X in (("path", X_path), ("std", X_std)):
        x = torch.stack([X[j] for j in sel])
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1).to(net.device)
        with torch.no_grad():
            res = [m(x[a:a + 32]) for a in range(0, len(x), 32)]
        mu = torch.cat([r_[0][0].view(-1) for r_ in res]).exp().cpu().numpy()
        ch = net.chi_from_target(torch.cat([r_[1][0].view(-1) for r_ in res]).cpu().numpy())
        for j, s in enumerate(sel):
            rows[s][f"Ms_{key}"], rows[s][f"chi_{key}"] = float(mu[j]), float(ch[j])

with open(os.path.join(args.out, f"control_realpath_{args.tag}.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "sim"}
med = lambda x: float(np.median(x))
L = [f"Real-data path control, model {args.tag} (out-of-fold), {len(rows)} images with Ms >= {args.ms_min}; "
     f"beam {args.beam_pix_obs} px upsampled x{up:.2f}, noise {args.noise} K",
     f"{'Ms bin':>10s} {'N':>5s} {'path/std':>9s} {'path/true':>10s} {'std/true':>9s} {'path/Msobs':>11s} {'std/Msobs':>10s} "
     f"{'Msobs p/true':>13s} {'det path':>9s} {'dchi':>6s}"]
for lo, hi in [(2, 4), (4, 7), (7, 11), (11, 20.01), (2, 20.01)]:
    b = (A["Ms_true"] >= lo) & (A["Ms_true"] < hi)
    if b.sum() < 5:
        continue
    L.append(f"{lo:4g}-{hi:<5g} {b.sum():5d} {med(A['Ms_path'][b] / A['Ms_std'][b]):9.3f} {med(A['Ms_path'][b] / A['Ms_true'][b]):10.3f} "
             f"{med(A['Ms_std'][b] / A['Ms_true'][b]):9.3f} {med(A['Ms_path'][b] / A['Ms_obs_path'][b]):11.3f} "
             f"{med(A['Ms_std'][b] / A['Ms_obs_std'][b]):10.3f} {med(A['Ms_obs_path'][b] / A['Ms_true'][b]):13.3f} "
             f"{med(A['det_path'][b]):9.2f} {med(A['chi_path'][b] - A['chi_std'][b]):+6.3f}")
open(os.path.join(args.out, f"control_realpath_{args.tag}.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
