"""
predict_dust_net9010.py
=======================
Apply the net9010 column-density (mom0) model to Herschel dust maps (obs_dust.json).

The mom0 model sees only ln(Sigma / <Sigma>_tile) plus the resolution channel,
so any map proportional to the gas column works: HGBS N(H2) or the 353 GHz
optical depth of the Herschel+Planck maps.

Two sets of tiles per region:
  native  : the map regridded so the beam spans 2 pixels (r = 0.5), invalid
            pixels (NaN, <= 0, outside coverage) filled with the nearest valid
            pixel, the n_tiles highest-column 128x128 tiles with valid
            coverage >= min_coverage.
  matched : regions with a CO counterpart only. The CO tiles of
            predict_observations_net9010.py (read from --co_tiles_dir) are
            resampled from the dust map, smoothed to the coarser of the CO and
            dust beams, on the CO pixel grid; r from that beam. Same sky area,
            so Ms(dust) can be compared with the CO tile-spectrum Ms and the
            CO-based network predictions tile by tile.

Dust Ms is reported as a LOWER LIMIT. Dust traces all the column along the
line of sight (diffuse foreground/background and gas below the CO threshold),
which the periodic training boxes do not have, so ln N has less contrast than
the turbulent cloud itself. On the CO-matched tiles the mom0 model gives
Ms(13CO W) ~ Ms(CO spectrum) but Ms(dust) ~ 0.4-0.6 of it (Perseus,
Ophiuchus). Removing a constant background raises Ms only slightly, so the
offset is not a simple zero point.

Each tile is predicted with and without a background: "bgsub" removes the
map's 1st-percentile column (a proxy for the diffuse line-of-sight column that
lowers the contrast), floored at 5% of it.

  python predict_dust_net9010.py                              # all regions, mom0 fold models
  python predict_dust_net9010.py --regions perseus orionB --models fold0 foldall

Writes <output_dir>/<region>_dust_tiles.csv, <region>_dust.png,
dust_matched_tiles.csv and dust_summary.txt.
"""
import argparse, csv, glob, json, os, re, sys, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, zoom
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networks_nbisht.net9010 as net

warnings.filterwarnings("ignore", category=FITSFixedWarning)

p = argparse.ArgumentParser()
p.add_argument("--config", default="obs_dust.json")
p.add_argument("--co_config", default="obs_clouds.json")
p.add_argument("--co_tiles_dir", default="/home/x-nbisht1/plots/observations_net9010_projmask",
               help="CO tile CSVs of predict_observations_net9010.py (positions + CO Ms)")
p.add_argument("--regions", nargs="*", default=None)
p.add_argument("--models", nargs="*", default=None, help="e.g. fold0 foldall (default: all)")
p.add_argument("--model_dir", default="/home/x-nbisht1/projects/p79d_dataset/models")
p.add_argument("--tag", default="mom0", help="run tag of the mom0 models")
p.add_argument("--beam_pix", type=float, default=2.0, help="beam FWHM in pixels after regridding")
p.add_argument("--output_dir", default=None, help="overrides the config's output_dir")
args = p.parse_args()
cfg = json.load(open(args.config))
if args.output_dir:
    cfg["output_dir"] = args.output_dir
co_cfg = json.load(open(args.co_config))
os.makedirs(cfg["output_dir"], exist_ok=True)
T_SIZE = cfg["tile_size"]

net.n_input_channels = 1
net.input_mode, net.n_model_channels = "mom0", 2


# ─── models ──────────────────────────────────────────────────────────────────
def load_models():
    out = {}
    tg = f"_{args.tag}" if args.tag else ""
    for ck in sorted(glob.glob(os.path.join(args.model_dir, f"test9010{tg}_mom0_fold*.pth"))):
        tag = re.search(r"_(fold\w+)\.pth$", ck).group(1)
        if args.models and tag not in args.models:
            continue
        qf = os.path.join(args.model_dir, f"conformal_quantiles_9010{tg}_mom0_{tag}.json")
        if not os.path.exists(qf):
            print(f"  skip {tag}: no conformal quantiles"); continue
        q = json.load(open(qf))
        net.chi_field, net.chi_space = q["chi_field"], q["chi_space"]
        net.dens_head = q.get("dens_head", False)
        sd = torch.load(ck, map_location=net.device)
        m = net.thisnet()
        for k in ("train_curve", "val_curve"):
            setattr(m, k, torch.zeros_like(sd[k]))
        m.load_state_dict(sd); m.eval()
        out[tag] = (m, q)
    if not out:
        raise SystemExit(f"No net9010 mom0 checkpoints with quantiles in {args.model_dir}")
    print(f"Models: {', '.join(out)}")
    return out


def predict(models, tiles, r, rows, suffix=""):
    """tiles: list of [H, W] column maps (physical units). Adds Ms/chi columns to rows."""
    raw = torch.tensor(np.stack(tiles)[:, None], dtype=torch.float32)
    X = torch.cat([net.preprocess_maps(raw), torch.full_like(raw, r)], dim=1).to(net.device)
    for tag, (m, q) in models.items():
        net.chi_field, net.chi_space = q["chi_field"], q["chi_space"]
        with torch.no_grad():
            (mu_ms, lv_ms), (mu_c, lv_c) = m(X)[:2]
        mu_ms, s_ms = mu_ms.view(-1).cpu().numpy(), torch.exp(0.5 * lv_ms).view(-1).cpu().numpy()
        mu_c = mu_c.view(-1).cpu().numpy()
        q_ms = q["ms_ln"]["0.9"]
        for i, row in enumerate(rows):
            row[f"Ms{suffix}_{tag}"] = float(np.exp(mu_ms[i]))
            row[f"Ms{suffix}_lo90_{tag}"] = float(np.exp(mu_ms[i] - q_ms * s_ms[i]))
            row[f"Ms{suffix}_hi90_{tag}"] = float(np.exp(mu_ms[i] + q_ms * s_ms[i]))
            row[f"chi{suffix}_{tag}"] = float(net.chi_from_target(mu_c[i]))
    kf = [t for t in models if t != "foldall"] or list(models)
    for row in rows:
        row[f"Ms{suffix}_kfold_mean"] = float(np.exp(np.mean(np.log([row[f"Ms{suffix}_{t}"] for t in kf]))))
        row[f"chi{suffix}_kfold_mean"] = float(np.mean([row[f"chi{suffix}_{t}"] for t in kf]))


# ─── maps ────────────────────────────────────────────────────────────────────
def load_map(rc):
    hdu = fits.open(os.path.join(cfg["obs_dir"], rc["fits"]))[0]
    d = hdu.data
    img = np.asarray(d.reshape((-1,) + d.shape[-2:])[rc.get("plane", 0)], dtype=np.float64)
    wcs = WCS(hdu.header).celestial
    pix = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
    valid = np.isfinite(img) & (img > 0)
    return img, valid, wcs, pix


def fill_nearest(img, valid):
    if valid.all():
        return img
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    return img[iy, ix]


def smooth_valid(img, valid, sigma_pix):
    """Normalised convolution: invalid pixels carry no weight."""
    if sigma_pix <= 0:
        return np.where(valid, img, 0.0), valid.astype(float)
    w = gaussian_filter(valid.astype(float), sigma_pix)
    s = gaussian_filter(np.where(valid, img, 0.0), sigma_pix)
    return np.where(w > 1e-3, s / np.maximum(w, 1e-3), 0.0), w


def select_tiles(img, valid, n_tiles, min_cov, max_overlap, stride=16):
    """Highest-column tiles (sum of valid column) with valid fraction >= min_cov, greedy with limited overlap."""
    ny, nx = img.shape
    cs = np.pad(np.cumsum(np.cumsum(np.where(valid, img, 0.0), 0), 1), ((1, 0), (1, 0)))
    cv = np.pad(np.cumsum(np.cumsum(valid.astype(float), 0), 1), ((1, 0), (1, 0)))
    box = lambda a, y, x: a[y + T_SIZE, x + T_SIZE] - a[y, x + T_SIZE] - a[y + T_SIZE, x] + a[y, x]
    cand = []
    for y in range(0, ny - T_SIZE + 1, stride):
        for x in range(0, nx - T_SIZE + 1, stride):
            cov = box(cv, y, x) / T_SIZE**2
            if cov >= min_cov:
                cand.append((box(cs, y, x), y, x, cov))
    cand.sort(reverse=True)
    sel = []
    for s, y, x, cov in cand:
        ov = [max(0, T_SIZE - abs(y - b[0])) * max(0, T_SIZE - abs(x - b[1])) / T_SIZE**2 for b in sel]
        if all(o <= max_overlap for o in ov):
            sel.append((y, x, cov))
        if len(sel) >= n_tiles:
            break
    return sel


def with_bg(t, bg):
    return np.maximum(t - bg, 0.05 * bg)


# ─── native tiles ────────────────────────────────────────────────────────────
def process_native(name, rc, models):
    img, valid, wcs, pix = load_map(rc)
    bg = float(np.percentile(img[valid], 1))
    f = args.beam_pix * pix / rc["beam_arcsec"]          # < 1: coarser grid
    if f < 1:
        imgf = np.where(valid, img, 0.0)
        g = zoom(imgf, f, order=1, prefilter=False)
        w = zoom(valid.astype(float), f, order=1, prefilter=False)
        valid_r = w > 0.99
        img_r = np.where(valid_r, g / np.maximum(w, 1e-6), np.nan)
    else:
        f, img_r, valid_r = 1.0, img, valid
    pix_r = pix / f
    beam_pix = rc["beam_arcsec"] / pix_r
    r = float(np.clip(1.0 / beam_pix, 0.4, 1.0))         # native_beam_pix = 1 for the projected training maps
    tile_pc = T_SIZE * pix_r / 206265.0 * rc["distance_pc"]
    sel = select_tiles(img_r, valid_r, rc.get("n_tiles", cfg.get("n_tiles", 8)),
                       rc.get("min_coverage", cfg.get("min_coverage", 0.95)),
                       rc.get("max_overlap", cfg.get("max_overlap", 0.25)))
    print(f"\n=== {name} ({rc['units']}) ===\n  map {img.shape} -> {img_r.shape} at {pix_r:.1f}\" "
          f"(beam {beam_pix:.2f} px, r = {r:.2f}), tile = {tile_pc:.2f} pc, valid {valid_r.mean():.0%}, "
          f"{len(sel)} tiles, bg (p1) = {bg:.3g}")
    if not sel:
        return [], None
    filled = fill_nearest(img_r, valid_r)
    tiles = [filled[y:y + T_SIZE, x:x + T_SIZE] for y, x, _ in sel]
    rows = []
    for (y, x, cov), t in zip(sel, tiles):
        ln = np.log(t / t.mean())
        rows.append({"region": name, "tile_y": y, "tile_x": x, "coverage": float(cov), "tile_pc": tile_pc,
                     "col_median": float(np.median(t)), "sigma_lnN": float(ln.std()),
                     "sigma_lnN_bgsub": float(np.log(with_bg(t, bg) / with_bg(t, bg).mean()).std())})
    predict(models, tiles, r, rows)
    predict(models, [with_bg(t, bg) for t in tiles], r, rows, "_bgsub")
    for row in rows:
        print(f"    tile ({row['tile_y']:4d},{row['tile_x']:4d}) cov {row['coverage']:.2f}  "
              f"sigma_lnN {row['sigma_lnN']:.2f}  Ms k-fold {row['Ms_kfold_mean']:5.1f}  "
              f"bgsub {row['Ms_bgsub_kfold_mean']:5.1f}  chi {row['chi_kfold_mean']:.2f}")
    with open(os.path.join(cfg["output_dir"], f"{name}_dust_tiles.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 7))
    lo, hi = np.nanpercentile(img_r[valid_r], [1, 99.5])
    ax.imshow(np.log10(np.clip(img_r, lo, hi)), origin="lower", cmap="Greys")
    for row in rows:
        ax.add_patch(Rectangle((row["tile_x"], row["tile_y"]), T_SIZE, T_SIZE, fill=False, ec="tab:red", lw=1.2))
        ax.text(row["tile_x"] + 3, row["tile_y"] + T_SIZE - 3,
                f"Ms={row['Ms_kfold_mean']:.1f}\nbg-={row['Ms_bgsub_kfold_mean']:.1f}",
                color="tab:red", fontsize=7, va="top")
    ax.set_title(f"{name}: net9010 mom0 on {rc['units']} (k-fold mean), tile {tile_pc:.2f} pc, r={r:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(cfg["output_dir"], f"{name}_dust.png"), dpi=130); plt.close(fig)
    return rows, {"region": name, "tile_pc": tile_pc, "r": r, "units": rc["units"]}


# ─── CO-matched tiles ────────────────────────────────────────────────────────
def process_matched(name, rc, models):
    co = rc.get("co_counterpart")
    csv_co = os.path.join(args.co_tiles_dir, f"{co}_net9010_tiles.csv") if co else None
    if not co or not os.path.exists(csv_co):
        return []
    cc = co_cfg["clouds"][co]
    wco = WCS(fits.getheader(os.path.join(co_cfg["obs_dir"], cc["fits"]))).celestial
    pco = float(np.mean(proj_plane_pixel_scales(wco))) * 3600.0
    if min(wco.pixel_shape) < T_SIZE:                     # CO map was upsampled: tile grid is not the CO grid
        print(f"  {co}: CO map smaller than a tile (upsampled), no matched tiles")
        return []
    co_rows = list(csv.DictReader(open(csv_co)))
    img, valid, wd, pd = load_map(rc)
    bg = float(np.percentile(img[valid], 1))
    beam = max(cc["beam_arcsec"], rc["beam_arcsec"])
    sm, w = smooth_valid(img, valid, np.sqrt(max(beam**2 - rc["beam_arcsec"]**2, 0)) / 2.3548 / pd)
    r = float(np.clip(1.0 / (beam / pco), 0.4, 1.0))
    out, tiles = [], []
    for cr in co_rows:
        y0, x0 = int(cr["tile_y"]), int(cr["tile_x"])
        yy, xx = np.mgrid[y0:y0 + T_SIZE, x0:x0 + T_SIZE]
        px, py = wd.world_to_pixel(wco.pixel_to_world(xx, yy))
        inside = (px >= 0) & (py >= 0) & (px <= img.shape[1] - 1) & (py <= img.shape[0] - 1)
        ww = np.where(inside, map_coordinates(w, [py, px], order=1, mode="constant"), 0.0)
        ok = ww > 0.5
        cov = float(ok.mean())
        if cov < 0.9:
            print(f"    {co} tile ({y0},{x0}): dust coverage {cov:.2f} < 0.9, skipped")
            continue
        t = fill_nearest(np.where(ok, map_coordinates(sm, [py, px], order=1, mode="constant"), np.nan), ok)
        tiles.append(t)
        kf = [k for k in cr if re.fullmatch(r"Ms_fold\d", k)]
        out.append({"region": name, "co_cloud": co, "tile_y": y0, "tile_x": x0, "dust_coverage": cov,
                    "beam_arcsec": beam, "r": r,
                    "Ms_CO_tile_spectrum": float(cr["Ms_tile_spectrum"]),
                    "Ms_CO_projmask_kfold": float(np.exp(np.mean(np.log([float(cr[k]) for k in kf])))) if kf else np.nan})
    if not out:
        return []
    predict(models, tiles, r, out)
    predict(models, [with_bg(t, bg) for t in tiles], r, out, "_bgsub")
    print(f"  matched to {co} CO tiles (beam {beam:.1f}\" = {beam / pco:.2f} CO px, r = {r:.2f}):")
    for o in out:
        print(f"    tile ({o['tile_y']:4d},{o['tile_x']:4d})  CO spectrum {o['Ms_CO_tile_spectrum']:5.1f}  "
              f"CO projmask {o['Ms_CO_projmask_kfold']:5.1f}  dust mom0 {o['Ms_kfold_mean']:5.1f}  "
              f"bgsub {o['Ms_bgsub_kfold_mean']:5.1f}")
    return out


# ─── main ────────────────────────────────────────────────────────────────────
models = load_models()
kf = [t for t in models if t != "foldall"] or list(models)
summary, matched = [], []
for name, rc in cfg["regions"].items():
    if args.regions and name not in args.regions:
        continue
    rows, info = process_native(name, rc, models)
    if rows:
        rec = dict(info, n_tiles=len(rows))
        for sfx in ("", "_bgsub"):
            v = np.array([row[f"Ms{sfx}_{t}"] for row in rows for t in kf])
            rec[f"Ms{sfx}_med"], rec[f"Ms{sfx}_p16"], rec[f"Ms{sfx}_p84"] = np.percentile(v, [50, 16, 84])
        c = np.array([row[f"chi_{t}"] for row in rows for t in kf])
        rec["chi_med"], rec["chi_p16"], rec["chi_p84"] = np.percentile(c, [50, 16, 84])
        summary.append(rec)
    matched += process_matched(name, rc, models)

if matched:
    keys = list(dict.fromkeys(k for m in matched for k in m))
    with open(os.path.join(cfg["output_dir"], "dust_matched_tiles.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(matched)

with open(os.path.join(cfg["output_dir"], "dust_summary.txt"), "w") as fh:
    hdr = (f"{'region':16s} {'units':7s} {'tile_pc':>7s} {'r':>4s} {'n':>3s}  {'Ms [16-84]':>20s}  "
           f"{'Ms bgsub [16-84]':>20s}  {'chi [16-84]':>18s}  lower limit")
    lines = [f"net9010 mom0 (tag {args.tag}) on Herschel dust maps; every tile x k-fold model ({', '.join(kf)}) pooled",
             "Ms from dust is a LOWER LIMIT: line-of-sight column dilutes the ln N contrast (see the CO-matched tiles).",
             f"Adopted lower limit per region: the median of the bg-subtracted values (the dust estimate itself is biased low).", hdr]
    for s in summary:
        lines.append(f"{s['region']:16s} {s['units']:7s} {s['tile_pc']:7.2f} {s['r']:4.2f} {s['n_tiles']:3d}  "
                     f"{s['Ms_med']:6.1f} [{s['Ms_p16']:5.1f},{s['Ms_p84']:5.1f}]  "
                     f"{s['Ms_bgsub_med']:6.1f} [{s['Ms_bgsub_p16']:5.1f},{s['Ms_bgsub_p84']:5.1f}]  "
                     f"{s['chi_med']:6.2f} [{s['chi_p16']:4.2f},{s['chi_p84']:4.2f}]  Ms > {s['Ms_bgsub_med']:.1f}")
    if matched:
        lines += ["", "CO-matched tiles (same sky area; dust smoothed to the coarser beam on the CO grid)",
                  f"{'region':16s} {'tile':>11s} {'CO spec':>8s} {'CO proj':>8s} {'dust':>6s} {'dust bg-':>8s}"]
        for m in matched:
            lines.append(f"{m['region']:16s} ({m['tile_y']:4d},{m['tile_x']:4d}) {m['Ms_CO_tile_spectrum']:8.1f} "
                         f"{m['Ms_CO_projmask_kfold']:8.1f} {m['Ms_kfold_mean']:6.1f} {m['Ms_bgsub_kfold_mean']:8.1f}")
    fh.write("\n".join(lines) + "\n")
print("\n" + open(os.path.join(cfg["output_dir"], "dust_summary.txt")).read())
print(f"Wrote {cfg['output_dir']}")
