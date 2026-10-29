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
     r = clip(native beam of the model's training maps / beam_FWHM_in_pixels, 0.4, 1)
     (native beam = 1 px for the projected dataset, ~3.1 px for the 13CO mocks);
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
p.add_argument("--tag", default="", help="run tag of the models, e.g. 13co")
args = p.parse_args()
cfg = json.load(open(args.config))
if args.tag:
    cfg["output_dir"] = cfg["output_dir"].rstrip("/") + f"_{args.tag}"
os.makedirs(cfg["output_dir"], exist_ok=True)
T_SIZE = cfg["tile_size"]

net.n_input_channels = 1 if args.mode == "mom0" else 3
net.input_mode, net.n_model_channels = args.mode, net.n_input_channels + 1


# ─── models ──────────────────────────────────────────────────────────────────
def load_models():
    out = {}
    tg = f"_{args.tag}" if args.tag else ""
    for ck in sorted(glob.glob(os.path.join(args.model_dir, f"test9010{tg}_{args.mode}_fold*.pth"))):
        tag = re.search(r"_(fold\w+)\.pth$", ck).group(1)
        if args.models and tag not in args.models:
            continue
        qf = os.path.join(args.model_dir, f"conformal_quantiles_9010{tg}_{args.mode}_{tag}.json")
        if not os.path.exists(qf):
            print(f"  skip {tag}: no conformal quantiles"); continue
        q = json.load(open(qf))
        net.chi_field, net.chi_space = q["chi_field"], q["chi_space"]
        net.dens_head = q.get("dens_head", False)              # sets the heads thisnet() builds
        net.n_lines = q.get("n_lines", 1)
        net.n_model_channels = net.n_lines * net.n_input_channels + 1
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


fill_tile = net.fill_masked_moments      # identical filling for training mocks and real tiles


def model_lines(models):
    """Line tags the models expect, in channel order, from the training image_key:
    'images_obs_<line>' -> that line; 'images' / 'images_obs' (projected maps, 13CO run 1)
    -> 'any' (the cloud's primary line)."""
    keys = next(iter(models.values()))[1].get("image_key", "images").split("+")
    return [k.rsplit("_", 1)[1] if k.startswith("images_obs_") else "any" for k in keys]


def cloud_lines(cc):
    """{line: fits} of a cloud; a plain string is its only line (tag from 'tracer', e.g. 13CO(1-0) -> 13co)."""
    if isinstance(cc["fits"], dict):
        return cc["fits"]
    return {cc["tracer"].split("(")[0].lower(): cc["fits"]}


def line_moments(path, cc, tag):
    """Moment maps of one line cube with the observation pipeline; noise_rms_K (float or {line: K})
    replaces the edge-channel noise estimate for pre-masked cubes."""
    nz = cc.get("noise_rms_K")
    nz = nz.get(tag) if isinstance(nz, dict) else nz
    with contextlib.redirect_stdout(io.StringIO()):
        data, hdr, wcs, vax = G.load_ppv_cube(os.path.join(cfg["obs_dir"], path))
        if cc.get("blank_noisy_pixels"):       # noisy map edges: per-pixel rms (5+5 edge channels) > 3x median -> blank
            rms = np.nanstd(np.concatenate([data[:5], data[-5:]]), axis=0)
            data = np.where((rms > 3 * np.nanmedian(rms))[None], np.nan, data)
        n_run = G.min_channels_for(cc.get("min_channels", cfg.get("min_channels", 3)), np.median(np.abs(np.diff(vax))))
        m0, m1, m2, mask = G.compute_moment_maps(data, vax, noise_threshold=cfg["mask_sigma"], min_channels=n_run, noise=nz)
        dclean, vmask, _ = G.signal_mask(data, cfg["mask_sigma"], n_run, noise=nz)
    Tmasked = np.where(vmask, np.nan_to_num(dclean), 0.0).astype(np.float32)
    return dict(m0=m0, m1=m1, m2=m2, mask=mask, T=Tmasked, vax=vax, hdr=hdr, n_run=n_run)


def regrid(L, up):
    """Zoom the moment maps of one line by `up` (bilinear; mask nearest). Blank velocity pixels
    take the nearest detected value first: zeros there would pull edge centroids toward
    v_lsr = 0 (tens of c_s for a cloud at -10 km/s)."""
    from scipy.ndimage import zoom
    if up == 1.0:
        return L
    good = L["mask"] & np.isfinite(L["m1"]) & np.isfinite(L["m2"])
    _, (iy, ix) = distance_transform_edt(~good, return_indices=True)
    m1 = zoom(L["m1"][iy, ix], up, order=1); m2 = zoom(L["m2"][iy, ix], up, order=1)
    mask = zoom(L["mask"].astype(np.uint8), up, order=0).astype(bool)
    return {**L, "m0": zoom(L["m0"], up, order=1), "m1": np.where(mask, m1, np.nan),
            "m2": np.where(mask, m2, np.nan), "mask": mask}


def process_cloud(name, cc, models):
    files = cloud_lines(cc)
    lines = [next(iter(files)) if t == "any" else t for t in model_lines(models)]
    missing = [t for t in lines if t not in files]
    if missing:
        raise ValueError(f"no {'/'.join(missing)} cube for this cloud")
    print(f"\n=== {name} ({cc['tracer']}, {cc['telescope']}) — lines {'+'.join(lines)} ===")
    Ls = {t: line_moments(files[t], cc, t) for t in dict.fromkeys(lines)}
    L0 = Ls[lines[0]]
    vax, hdr = L0["vax"], L0["hdr"]
    print(f"  channel width {np.median(np.abs(np.diff(vax))):.3f} km/s -> min_channels {L0['n_run']}")
    pix = spatial_pixel_arcsec(hdr)
    native = next(iter(models.values()))[1].get("native_beam_pix", 1.0)
    beam_pix0 = cc["beam_arcsec"] / pix
    # regrid so the tile is at least 128 px and the beam spans at least the training beam
    # (native_beam_pix): the network sees the same beams per tile as in training
    up = max(1.0, T_SIZE / min(L0["m0"].shape), native / beam_pix0)
    if up < 1.02 and min(L0["m0"].shape) >= T_SIZE:
        up = 1.0
    elif min(L0["m0"].shape) * up < T_SIZE:
        up = (T_SIZE + 0.5) / min(L0["m0"].shape)
    Ls = {t: regrid(L, up) for t, L in Ls.items()}
    L0 = Ls[lines[0]]
    m0, m1, m2, mask = L0["m0"], L0["m1"], L0["m2"], L0["mask"]
    Tmasked = L0["T"]
    beam_pix = beam_pix0 * up
    r = float(np.clip(native / beam_pix, 0.4, 1.0))
    cs = np.sqrt(KB * cc["T_K"] / (cfg["mu_sound"] * MH)) / 1e5
    tile_pc = T_SIZE * (pix / up) / 206265.0 * cc["distance_pc"]
    floors = {t: (np.nanpercentile(L["m0"][L["mask"]], 1) if L["mask"].any() else 1e-3) for t, L in Ls.items()}
    floor = floors[lines[0]]
    print(f"  map {m0.shape}, pixel {pix:.1f}\" (regridded x{up:.2f}), beam {beam_pix:.2f} px -> r = {r:.2f}, "
          f"c_s = {cs:.3f} km/s at {cc['T_K']} K, tile = {tile_pc:.1f} pc, detected {mask.mean():.0%}")

    with contextlib.redirect_stdout(io.StringIO()):
        td = G.prepare_tiled_model_input(m0, m1, m2, mask, n_tiles=cc["n_tiles"],
                                         min_coverage=cc["min_coverage"], max_overlap=cc["max_overlap"])
    rows, xs = [], []
    for (y, x), cov in zip(td["positions"], td["coverages"]):
        sl = (slice(y, y + T_SIZE), slice(x, x + T_SIZE))
        chans = []
        for t in lines:                       # each line filled and preprocessed on its own, as in training
            L = Ls[t]
            if not L["mask"][sl].any():
                raise ValueError(f"tile ({y},{x}) has no {t} detection")
            a0, a1, a2 = fill_tile(L["m0"][sl], L["m1"][sl], L["m2"][sl], L["mask"][sl], floors[t])
            raw = torch.tensor(np.stack([a0, a1 / cs, a2 / cs])[:net.n_input_channels], dtype=torch.float32)
            chans.append(net.preprocess_maps(raw))
        x_in = torch.cat(chans + [torch.full((1, T_SIZE, T_SIZE), r)], dim=0)
        xs.append(x_in)
        # matched-scale observational estimate: width of the tile-averaged spectrum
        # (tile mapped back to native pixels when the map was upsampled)
        ys = slice(int(y / up), int((y + T_SIZE) / up)); xs_ = slice(int(x / up), int((x + T_SIZE) / up))
        spec = Tmasked[:, ys, xs_].mean(axis=(1, 2))
        mu_v = (spec * vax).sum() / spec.sum()
        sig_tile = np.sqrt((spec * (vax - mu_v)**2).sum() / spec.sum())
        rows.append({"cloud": name, "tile_y": int(y), "tile_x": int(x), "coverage": float(cov),
                     "sigma_v_median_kms": float(np.nanmedian(m2[sl][mask[sl]])),
                     "sigma_tile_spectrum_kms": float(sig_tile),
                     "Ms_tile_spectrum": float(np.sqrt(3) * sig_tile / cs)})
    X = torch.stack(xs).to(net.device)

    for tag, (m, q) in models.items():
        net.chi_field, net.chi_space = q["chi_field"], q["chi_space"]
        with torch.no_grad():
            res = m(X)
        (mu_ms, lv_ms), (mu_c, lv_c) = res[:2]
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
            # chi is only claimed where the model was trained for it (q["chi_ms_min"])
            row[f"chi_valid_{tag}"] = int(row[f"Ms_{tag}"] >= q.get("chi_ms_min", 0.0))
        if len(res) > 2:
            mu_d, s_d = res[2][0].cpu().numpy(), torch.exp(0.5 * res[2][1]).cpu().numpy()
            for j, k in enumerate(net.DENS_FIELDS):
                qd = q["dens_ln"][k]["0.9"]
                for i, row in enumerate(rows):
                    row[f"{k}_{tag}"] = float(np.exp(mu_d[i, j]))
                    row[f"{k}_lo90_{tag}"] = float(np.exp(mu_d[i, j] - qd * s_d[i, j]))
                    row[f"{k}_hi90_{tag}"] = float(np.exp(mu_d[i, j] + qd * s_d[i, j]))
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
                f"Ms={Ms:.1f}\nχ={chi:.2f}\nobs={row['Ms_tile_spectrum']:.1f}",
                color="tab:red", fontsize=8, va="top")
    ax.set_title(f"{name}: net9010 {args.mode} ({head or 'k-fold mean'}), T={cc['T_K']} K, "
                 f"tile {tile_pc:.1f} pc, r={r:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(cfg["output_dir"], f"{name}_net9010.png"), dpi=130); plt.close(fig)
    return rows, {"cloud": name, "tile_pc": tile_pc, "r": r, "c_s": cs, "T_K": cc["T_K"], "tracer": cc["tracer"],
                  "distance_pc": cc["distance_pc"], "up": up}


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
    for r in rows:
        print(f"    tile ({r['tile_y']:4d},{r['tile_x']:4d}) coverage {r['coverage']:.2f}: Ms from tile spectrum "
              f"{r['Ms_tile_spectrum']:5.1f}  " + "  ".join(f"{t} {r[f'Ms_{t}']:5.1f}" for t in models))
    summary.append((info, rows))

# per-cloud estimate: every tile x every k-fold model pooled; median and 16-84% range.
# chi only from (tile, model) pairs where that model's chi head applies.
kf = [t for t in models if t != "foldall"] or list(models)
cloud_rows = []
for info, rows in summary:
    rec = {"cloud": info["cloud"], "n_tiles": len(rows), "n_models": len(kf),
           "Ms_tile_spectrum_median": float(np.median([r["Ms_tile_spectrum"] for r in rows]))}
    for k in ["Ms", "chi"] + [d for d in net.DENS_FIELDS if f"{d}_{kf[0]}" in rows[0]]:
        v = np.array([r[f"{k}_{t}"] for r in rows for t in kf
                      if k != "chi" or r.get(f"chi_valid_{t}", 1)])
        rec[f"{k}_n"] = int(len(v))
        if len(v):
            rec[f"{k}_median"], rec[f"{k}_p16"], rec[f"{k}_p84"] = (float(x) for x in np.percentile(v, [50, 16, 84]))
    cloud_rows.append(rec)
    print(f"  {info['cloud']:10s} " + "  ".join(f"{k} {rec[f'{k}_median']:.3g} [{rec[f'{k}_p16']:.3g}, {rec[f'{k}_p84']:.3g}]"
                                          for k in ("Ms", "chi", "b_lin", "sigma_s") if f"{k}_median" in rec))
if cloud_rows:
    keys = sorted({k for r in cloud_rows for k in r}, key=lambda k: (k != "cloud", k))
    with open(os.path.join(cfg["output_dir"], "net9010_observations_per_cloud.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(cloud_rows)

with open(os.path.join(cfg["output_dir"], "net9010_observations_summary.txt"), "w") as f:
    f.write(f"{'cloud':10s} {'tracer':10s} {'T_K':>4s} {'tile_pc':>7s} {'r':>5s} {'n':>3s} "
            + " ".join(f"{'Ms_' + t:>12s}" for t in models) + f" {'chi_med':>8s} {'Ms_tilespec':>11s}\n")
    for info, rows in summary:
        tagc = "foldall" if "foldall" in models else next(iter(models))
        f.write(f"{info['cloud']:10s} {info['tracer']:10s} {info['T_K']:4.0f} {info['tile_pc']:7.1f} {info['r']:5.2f} "
                f"{len(rows):3d} " + " ".join(f"{np.median([r[f'Ms_{t}'] for r in rows]):12.2f}" for t in models)
                + f" {np.median([r[f'chi_{tagc}'] for r in rows]):8.3f}"
                + f" {np.median([r['Ms_tile_spectrum'] for r in rows]):11.2f}\n")
print(f"\nWrote {cfg['output_dir']}/net9010_observations_summary.txt")
