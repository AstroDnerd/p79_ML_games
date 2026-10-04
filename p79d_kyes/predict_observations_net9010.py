"""
predict_observations_net9010.py
===============================
Apply net9010 (Ms + chi) to observed PPV cubes.

For each cloud in obs_clouds.json:
  1. moment maps with the observation pipeline (get_Observation_Data:
     load_ppv_cube, compute_moment_maps, upsample_moment_maps) — the same code
     used for the synthetic 13CO dataset;
  2. emission-rich 128x128 tiles (get_Observation_Data.prepare_tiled_model_input
     is used only to choose tile positions);
  3. per tile, the training units: [W, v_c / c_s, sigma_v / c_s] with
     c_s = sqrt(k T / mu m_H) at the cloud's assumed T, masked pixels filled
     (W with the detection floor, velocities with the nearest detected pixel),
     then net9010.preprocess_maps, plus the resolution channel
     r = clip(1 / beam_FWHM_in_pixels, 0.4, 1);
  4. predictions from every net9010 checkpoint found (k-fold models and the
     --fold all model), with 90% conformal intervals from each model's
     calibration quantiles.

  python predict_observations_net9010.py                      # all clouds, all models
  python predict_observations_net9010.py --clouds perseus --models foldall

Writes <output_dir>/<cloud>_net9010_tiles.csv, <cloud>_net9010.png and a
summary table net9010_observations_summary.txt.
"""
import argparse, contextlib, csv, glob, io, json, os, re, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networks_nbisht.net9010 as net
import get_Observation_Data as G

KB, MH = 1.380649e-16, 1.6735575e-24

p = argparse.ArgumentParser()
p.add_argument("--config", default="obs_clouds.json")
p.add_argument("--clouds", nargs="*", default=None)
p.add_argument("--models", nargs="*", default=None,
               help="model tags to use, e.g. fold0 foldall (default: every test9010_allmom_fold*.pth)")
p.add_argument("--model_dir", default="/home/x-nbisht1/projects/p79d_dataset/models")
p.add_argument("--mode", default="allmom", choices=["allmom", "mom0"])
args = p.parse_args()
cfg = json.load(open(args.config))
os.makedirs(cfg["output_dir"], exist_ok=True)
T_SIZE = cfg["tile_size"]

net.n_input_channels = 1 if args.mode == "mom0" else 3
net.input_mode, net.n_model_channels = args.mode, net.n_input_channels + 1


# ─── models ──────────────────────────────────────────────────────────────────
def load_models():
    out = {}
    for ck in sorted(glob.glob(os.path.join(args.model_dir, f"test9010_{args.mode}_fold*.pth"))):
        tag = re.search(r"_(fold\w+)\.pth$", ck).group(1)
        if args.models and tag not in args.models:
            continue
        qf = os.path.join(args.model_dir, f"conformal_quantiles_9010_{args.mode}_{tag}.json")
        if not os.path.exists(qf):
            print(f"  skip {tag}: no conformal quantiles"); continue
        q = json.load(open(qf))
        net.chi_field, net.chi_space = q["chi_field"], q["chi_space"]
        sd = torch.load(ck, map_location=net.device)
        m = net.thisnet()
        for k in ("train_curve", "val_curve"):
            setattr(m, k, torch.zeros_like(sd[k]))
        m.load_state_dict(sd); m.eval()
        out[tag] = (m, q)
    if not out:
        raise SystemExit(f"No net9010 {args.mode} checkpoints with quantiles in {args.model_dir}")
    print(f"Models: {', '.join(out)}")
    return out


# ─── one cloud ───────────────────────────────────────────────────────────────
def spatial_pixel_arcsec(hdr):
    for i in (1, 2, 3):
        ct = str(hdr.get(f"CTYPE{i}", ""))
        if ct and not ct.upper().startswith(("VELO", "VRAD", "FREQ", "VOPT")):
            return abs(float(hdr[f"CDELT{i}"])) * 3600.0
    raise ValueError("no spatial axis in header")


def fill_tile(m0, m1, m2, mask, floor):
    """Training maps have no blank pixels: fill W with the detection floor, velocities by nearest neighbour."""
    good = mask & np.isfinite(m1) & np.isfinite(m2)
    if good.sum() == 0:
        raise ValueError("tile without detected pixels")
    _, (iy, ix) = distance_transform_edt(~good, return_indices=True)
    return (np.where(good, np.maximum(m0, floor), floor), m1[iy, ix], m2[iy, ix])


def process_cloud(name, cc, models):
    print(f"\n=== {name} ({cc['tracer']}, {cc['telescope']}) ===")
    with contextlib.redirect_stdout(io.StringIO()):
        data, hdr, wcs, vax = G.load_ppv_cube(os.path.join(cfg["obs_dir"], cc["fits"]))
        m0, m1, m2, mask = G.compute_moment_maps(data, vax, noise_threshold=cfg["mask_sigma"])
    pix = spatial_pixel_arcsec(hdr)
    up = 1.0
    if min(m0.shape) < T_SIZE:
        with contextlib.redirect_stdout(io.StringIO()):
            u0, u1, u2, umask = G.upsample_moment_maps(m0, m1, m2, mask, target_size=T_SIZE)
        up = u0.shape[0] / m0.shape[0]
        m0, m1, m2, mask = u0, u1, u2, umask
    beam_pix = cc["beam_arcsec"] / pix * up
    r = float(np.clip(1.0 / beam_pix, 0.4, 1.0))
    cs = np.sqrt(KB * cc["T_K"] / (cfg["mu_sound"] * MH)) / 1e5
    tile_pc = T_SIZE * (pix / up) / 206265.0 * cc["distance_pc"]
    floor = np.nanpercentile(m0[mask], 1) if mask.any() else 1e-3
    print(f"  map {m0.shape}, pixel {pix:.1f}\" (upsampled x{up:.2f}), beam {beam_pix:.2f} px -> r = {r:.2f}, "
          f"c_s = {cs:.3f} km/s at {cc['T_K']} K, tile = {tile_pc:.1f} pc, detected {mask.mean():.0%}")

    with contextlib.redirect_stdout(io.StringIO()):
        td = G.prepare_tiled_model_input(m0, m1, m2, mask, n_tiles=cc["n_tiles"],
                                         min_coverage=cc["min_coverage"], max_overlap=cc["max_overlap"])
    rows, xs = [], []
    for (y, x), cov in zip(td["positions"], td["coverages"]):
        sl = (slice(y, y + T_SIZE), slice(x, x + T_SIZE))
        a0, a1, a2 = fill_tile(m0[sl], m1[sl], m2[sl], mask[sl], floor)
        raw = torch.tensor(np.stack([a0, a1 / cs, a2 / cs])[:net.n_input_channels], dtype=torch.float32)
        x_in = torch.cat([net.preprocess_maps(raw), torch.full((1, T_SIZE, T_SIZE), r)], dim=0)
        xs.append(x_in)
        rows.append({"cloud": name, "tile_y": int(y), "tile_x": int(x), "coverage": float(cov),
                     "sigma_v_median_kms": float(np.nanmedian(m2[sl][mask[sl]])),
                     "Ms_linewidth_naive": float(np.sqrt(3) * np.nanmedian(m2[sl][mask[sl]]) / cs)})
    X = torch.stack(xs).to(net.device)

    for tag, (m, q) in models.items():
        net.chi_field, net.chi_space = q["chi_field"], q["chi_space"]
        with torch.no_grad():
            (mu_ms, lv_ms), (mu_c, lv_c) = m(X)
        mu_ms, s_ms = mu_ms.view(-1).cpu().numpy(), torch.exp(0.5 * lv_ms).view(-1).cpu().numpy()
        mu_c,  s_c  = mu_c.view(-1).cpu().numpy(),  torch.exp(0.5 * lv_c).view(-1).cpu().numpy()
        q_ms, q_c = q["ms_ln"]["0.9"], q["chi_target"]["0.9"]
        for i, row in enumerate(rows):
            row[f"Ms_{tag}"] = float(np.exp(mu_ms[i]))
            row[f"Ms_lo90_{tag}"], row[f"Ms_hi90_{tag}"] = (float(np.exp(mu_ms[i] - q_ms * s_ms[i])),
                                                            float(np.exp(mu_ms[i] + q_ms * s_ms[i])))
            row[f"chi_{tag}"] = float(net.chi_from_target(mu_c[i]))
            row[f"chi_lo90_{tag}"] = float(net.chi_from_target(mu_c[i] - q_c * s_c[i]))
            row[f"chi_hi90_{tag}"] = float(net.chi_from_target(mu_c[i] + q_c * s_c[i]))
    kf = [t for t in models if t != "foldall"]
    for row in rows:
        if kf:
            lm = np.log([row[f"Ms_{t}"] for t in kf])
            row["Ms_kfold_mean"], row["Ms_kfold_spread"] = float(np.exp(lm.mean())), float(lm.std())
            row["chi_kfold_mean"] = float(np.mean([row[f"chi_{t}"] for t in kf]))

    with open(os.path.join(cfg["output_dir"], f"{name}_net9010_tiles.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # figure: integrated intensity with tiles labelled by the headline model
    head = "foldall" if "foldall" in models else None
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(np.log10(np.clip(m0, floor, None)), origin="lower", cmap="Greys")
    for row in rows:
        Ms  = row[f"Ms_{head}"] if head else row["Ms_kfold_mean"]
        chi = row[f"chi_{head}"] if head else row["chi_kfold_mean"]
        ax.add_patch(Rectangle((row["tile_x"], row["tile_y"]), T_SIZE, T_SIZE, fill=False, ec="tab:red", lw=1.5))
        ax.text(row["tile_x"] + 4, row["tile_y"] + T_SIZE - 4,
                f"Ms={Ms:.1f}\nχ={chi:.2f}\nlw={row['Ms_linewidth_naive']:.1f}",
                color="tab:red", fontsize=8, va="top")
    ax.set_title(f"{name}: net9010 {args.mode} ({head or 'k-fold mean'}), T={cc['T_K']} K, "
                 f"tile {tile_pc:.1f} pc, r={r:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(cfg["output_dir"], f"{name}_net9010.png"), dpi=130); plt.close(fig)
    return rows, {"cloud": name, "tile_pc": tile_pc, "r": r, "c_s": cs, "T_K": cc["T_K"], "tracer": cc["tracer"]}


models = load_models()
summary = []
for name, cc in cfg["clouds"].items():
    if args.clouds and name not in args.clouds:
        continue
    try:
        rows, info = process_cloud(name, cc, models)
    except Exception as e:
        print(f"  {name} FAILED: {e!r}"); continue
    for tag in models:
        Ms = np.array([r[f"Ms_{tag}"] for r in rows]); chi = np.array([r[f"chi_{tag}"] for r in rows])
        print(f"  {tag:8s} Ms median {np.median(Ms):6.2f} [tiles {Ms.min():.2f}–{Ms.max():.2f}]   "
              f"chi median {np.median(chi):.3f}")
    lw = np.median([r["Ms_linewidth_naive"] for r in rows])
    print(f"  naive linewidth Ms = sqrt(3) median(sigma_v) / c_s = {lw:.2f}")
    summary.append((info, rows))

with open(os.path.join(cfg["output_dir"], "net9010_observations_summary.txt"), "w") as f:
    f.write(f"{'cloud':10s} {'tracer':10s} {'T_K':>4s} {'tile_pc':>7s} {'r':>5s} {'n':>3s} "
            + " ".join(f"{'Ms_' + t:>12s}" for t in models) + f" {'chi_med':>8s} {'Ms_lw':>7s}\n")
    for info, rows in summary:
        tagc = "foldall" if "foldall" in models else next(iter(models))
        f.write(f"{info['cloud']:10s} {info['tracer']:10s} {info['T_K']:4.0f} {info['tile_pc']:7.1f} {info['r']:5.2f} "
                f"{len(rows):3d} " + " ".join(f"{np.median([r[f'Ms_{t}'] for r in rows]):12.2f}" for t in models)
                + f" {np.median([r[f'chi_{tagc}'] for r in rows]):8.3f}"
                + f" {np.median([r['Ms_linewidth_naive'] for r in rows]):7.2f}\n")
print(f"\nWrote {cfg['output_dir']}/net9010_observations_summary.txt")
