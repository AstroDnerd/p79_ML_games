"""
ViT_setup_test_net9010.py
=========================
Train, calibrate, and evaluate net9010 (dual-output: Ms + chi) on one
simulation-held-out fold.

  python ViT_setup_test_net9010.py --mode allmom --fold 0
  python ViT_setup_test_net9010.py --mode mom0 --fold 0 --eval_only
  python ViT_setup_test_net9010.py --mode allmom --fold all      # final model

--fold all trains the model to apply to observations: every simulation except
the calibration fold (fold 0 by default, --calib_fold) is used for training, and
there is no test set. Report performance from the k-fold runs, not from it.

Splits (net.load_data_grouped): test = one fold of simulations, calib = the
next fold (conformal calibration and diagnostic plots), valid = held-back
late frames of the training simulations (early stopping only), train = rest.

Outputs
  ~/plots/<kind>_net9010_<mode>[_<tag>].png
  <model_dir>/test9010_<mode>_fold<k>.pth                   weights
  <model_dir>/conformal_quantiles_9010_<mode>_fold<k>.json  q80/q90/q95 for Ms and chi
With --tag T every name becomes ..._9010_T_<mode>_... (e.g. --tag 13co for the
RADMC dataset: --data_file .../p79d_mach_grid_256_13co_v1.h5 --image_key images_obs).
  <model_dir>/net9010_<mode>_fold<k>_test.npz / .json       test predictions + metrics
Plots are not versioned: each run overwrites the previous one for that mode.
Pool the folds with analyze_kfold_net9010.py.
"""

import argparse, json, os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes/")
import networks_nbisht.net9010 as net

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["allmom", "mom0"], default="allmom")
p.add_argument("--fold", default="0", help="fold index, or 'all' for the final model")
p.add_argument("--calib_fold", type=int, default=0, help="calibration fold for --fold all")
p.add_argument("--n_folds", type=int, default=5)
p.add_argument("--chi_field", choices=["chi_v", "chi_w", "b", "b_lin"], default=net.chi_field,
               help="second target; b / b_lin need --chi_space log")
p.add_argument("--chi_space", choices=["logit", "linear", "log"], default=net.chi_space)
p.add_argument("--chi_smooth_tc", type=float, default=None,
               help="smooth chi_v/chi_w over this many crossing times of its run (see net.smoothed_chi)")
p.add_argument("--chi_ms_min", type=float, default=0.0,
               help="train, calibrate and score chi only on images with Ms >= this")
p.add_argument("--dens_head", action="store_true",
               help="add a third Gaussian head for ln b_lin and ln sigma_s")
p.add_argument("--mask_aug_prob", type=float, default=0.0,
               help="probability of the emission-mask augmentation per training image")
p.add_argument("--epochs", type=int, default=net.epochs)
p.add_argument("--eval_only", action="store_true", help="load weights instead of training")
p.add_argument("--no_plots", action="store_true", help="skip the diagnostic plots on calib")
p.add_argument("--workers", type=int, default=8, help="DataLoader workers for training")
p.add_argument("--model_dir", default="/home/x-nbisht1/projects/p79d_dataset/models")
p.add_argument("--data_file", default=net.data_file, help="dataset HDF5 (projected or 13CO)")
p.add_argument("--image_key", default="images",
               help="images | images_obs (13CO run 1) | images_obs_13co, images_obs_12co (run 2); "
                    "join with '+' to stack lines, e.g. images_obs_13co+images_obs_12co")
p.add_argument("--time_budget_h", type=float, default=None,
               help="stop training before this many hours (keeps the best weights), so the job "
                    "can be given a tight SLURM limit and still finish its evaluation")
p.add_argument("--tag", default="", help="label added to every output name, e.g. 13co")
args = p.parse_args()

# Module-level switches are read at call time: set them before building anything.
net.n_input_channels = 1 if args.mode == "mom0" else 3
net.input_mode       = args.mode
net.n_lines          = len(args.image_key.split("+"))
net.n_model_channels = net.n_lines * net.n_input_channels + 1
net.time_budget_h    = args.time_budget_h
net.chi_field, net.chi_space = args.chi_field, args.chi_space
net.CHI_LABEL = {"chi_v": r"$\chi_v$", "chi_w": r"$\chi_w$", "b": r"$b$", "b_lin": r"$b_{\rm lin}$"}[args.chi_field]
if args.chi_field in ("b", "b_lin") and args.chi_space == "logit":
    sys.exit("b / b_lin are not bounded by 1: use --chi_space log")
net.mask_aug_prob = args.mask_aug_prob
net.chi_smooth_tc, net.chi_ms_min, net.dens_head = args.chi_smooth_tc, args.chi_ms_min, args.dens_head
net.epochs = args.epochs
net.data_file, net.image_key, net.run_tag = args.data_file, args.image_key, args.tag
fold = None if args.fold == "all" else int(args.fold)

tag_      = f"_{args.tag}" if args.tag else ""
net_name  = f"net{net.idd}{tag_}_{net.input_mode}"
plot_dir  = os.path.join(os.environ["HOME"], "plots")
os.makedirs(args.model_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)
run_name  = f"{net_name}_fold{args.fold}"
ckpt_path = os.path.join(args.model_dir, f"test{net.idd}{tag_}_{net.input_mode}_fold{args.fold}.pth")

# native resolution of the training maps, in pixels: 1 for projected maps, the
# beam FWHM for the 13CO mocks (used to set the resolution channel on real data)
import h5py
with h5py.File(net.data_file, "r") as _f:
    _rt  = json.loads(_f.attrs["rt_json"]) if "rt_json" in _f.attrs else None
    # run-2 re-observed files carry their setup (beam, distance) in observe_version
    _obs = json.loads(_f.attrs["observe_version"]) if "observe_version" in _f.attrs else None
_beam = _obs if _obs is not None else (_rt if _rt and "beam_fwhm_arcsec" in _rt else None)
native_beam_pix = 1.0 if _beam is None else _beam["beam_fwhm_arcsec"] / (
    _rt["box_length_pc"] / _rt["target_res"] / _beam["distance_pc"] * 206265.0)
print(f"{net_name} | fold {args.fold}/{args.n_folds} | target {net.chi_field} ({net.chi_space}) | "
      f"device {net.device} | threads {torch.get_num_threads()}", flush=True)

# ─── DATA ─────────────────────────────────────────────────────────────────────
splits = net.load_data_grouped(fold, args.n_folds, calib=True, calib_fold=args.calib_fold)
tags   = ("train", "valid", "calib", "test") if fold is not None else ("train", "valid", "calib")
for tag in tags:
    s = splits[tag]
    print(f"  {tag:5s}: N={len(s['data']):6d}  Ms [{s['Ms_act'].min():.2f}, {s['Ms_act'].max():.2f}]  "
          f"chi [{s['chi_act'].min():.3f}, {s['chi_act'].max():.3f}]")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for tag, c in zip(tags, ("steelblue", "orange", "purple", "green")):
    axes[0].hist(np.log10(splits[tag]["Ms_act"]), bins=50, alpha=0.5, label=tag, color=c, density=True)
    axes[1].hist(splits[tag]["chi_act"], bins=50, alpha=0.5, label=tag, color=c, density=True)
axes[0].set_xlabel(r"$\log_{10}\mathcal{M}_s$"); axes[1].set_xlabel(net.CHI_LABEL)
for ax in axes: ax.legend(); ax.grid(True, alpha=0.3)
fig.suptitle(f"{net_name} — fold {args.fold} split distributions")
fig.savefig(os.path.join(plot_dir, f"ms_chi_distribution_{net_name}.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ─── MODEL ────────────────────────────────────────────────────────────────────
model = net.thisnet()
print(f"\nInstantiated {net_name} | params: {sum(q.numel() for q in model.parameters() if q.requires_grad):,}")

if args.eval_only:
    sd = torch.load(ckpt_path, map_location=net.device)
    for k in ("train_curve", "val_curve"):     # sized by the epoch count the checkpoint was trained with
        setattr(model, k, torch.zeros_like(sd[k]))
    model.load_state_dict(sd)
    print(f"Loaded weights from {ckpt_path}")
else:
    t0 = time.time()
    net.trainer(model, splits, epochs=net.epochs, lr=net.lr, batch_size=net.batch_size,
                weight_decay=net.weight_decay, num_workers=args.workers)
    print(f"Training complete: {(time.time() - t0) / 3600:.2f} h")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved weights → {ckpt_path}")
    net.plot_loss_curve(model)
model.eval()

loader = lambda tag: DataLoader(net.MachChiDataset(splits[tag], augment=False), batch_size=256, shuffle=False)
calib_loader, test_loader = loader("calib"), loader("test")

# ─── CONFORMAL CALIBRATION (held-out calib fold) ─────────────────────────────
s_ms, s_chi = net.compute_conformal_scores(model, calib_loader)
levels = (0.80, 0.90, 0.95)
q_ms   = {a: net.get_conformal_quantile(s_ms, a) for a in levels}
q_chi  = {a: net.get_conformal_quantile(s_chi, a) for a in levels}
print("\nConformal quantiles (calib fold):")
print("  Ms : " + "  ".join(f"q{int(a*100)}={q:.3f}" for a, q in q_ms.items()))
print("  chi: " + "  ".join(f"q{int(a*100)}={q:.3f}" for a, q in q_chi.items()))
q_dens = {}
if net.dens_head:
    pc = net.predict(model, calib_loader)
    for k in net.DENS_FIELDS:
        sc = torch.tensor(np.abs(np.log(pc[f"{k}_true"] / pc[f"{k}_mean"])) / (pc[f"{k}_sig_log"] + 1e-8))
        q_dens[k] = {str(a): net.get_conformal_quantile(sc, a) for a in levels}
        print(f"  ln {k}: " + "  ".join(f"q{int(float(a)*100)}={q:.3f}" for a, q in q_dens[k].items()))
print(f"  temperatures: T_ms={model.log_temp_ms.exp().item():.3f}  T_chi={model.log_temp_chi.exp().item():.3f}")
with open(os.path.join(args.model_dir, f"conformal_quantiles_{net.idd}{tag_}_{net.input_mode}_fold{args.fold}.json"), "w") as f:
    json.dump({"fold": args.fold, "chi_field": net.chi_field, "chi_space": net.chi_space,
               "data_file": net.data_file, "image_key": net.image_key, "n_lines": net.n_lines, "mask_aug_prob": net.mask_aug_prob,
               "native_beam_pix": native_beam_pix,
               "chi_smooth_tc": net.chi_smooth_tc, "chi_ms_min": net.chi_ms_min, "dens_head": net.dens_head,
               "dens_ln": q_dens,
               "ms_ln": {str(a): q for a, q in q_ms.items()},
               "chi_target": {str(a): q for a, q in q_chi.items()}}, f, indent=2)

# ─── DIAGNOSTIC PLOTS (calib fold) ───────────────────────────────────────────
if not args.no_plots:
    net.plot_predictions_dual(model, calib_loader, tag="calib")
    n_mc = min(1000, len(splits["calib"]["data"]))
    sub  = {k: v[:n_mc] for k, v in splits["calib"].items()}
    net.plot_uncertainty_analysis_dual(
        model, DataLoader(net.MachChiDataset(sub, augment=False), batch_size=128), tag="calib")
    # three fixed examples (low / mid / high Ms) so the files are overwritten, not accumulated
    order = np.argsort(splits["calib"]["Ms_act"])
    for j, i in enumerate(order[[len(order) // 10, len(order) // 2, (9 * len(order)) // 10]]):
        x = splits["calib"]["data"][i:i + 1, net.input_channel_index()].to(net.device)
        net.plot_resolution_robustness_dual(model, x, float(splits["calib"]["Ms_act"][i]),
                                            float(splits["calib"]["chi_act"][i]), tag=f"calib_{j}")

if fold is None:
    print(f"\n--fold all: final model saved to {ckpt_path}; conformal quantiles from fold "
          f"{args.calib_fold}. No test set — use the k-fold runs for performance.")
    sys.exit(0)

# ─── FINAL EVALUATION (held-out test fold) ───────────────────────────────────
print("\n" + "=" * 70 + "\nHELD-OUT TEST FOLD\n" + "=" * 70)
pt = net.predict(model, test_loader)
m  = net.regression_metrics(pt)
print(f"  ln Ms : R²={m['lnMs']['R2']:.4f}  r={m['lnMs']['r']:.4f}  RMSE={m['lnMs']['RMSE']:.4f}")
print(f"  Ms    : R²={m['Ms']['R2']:.4f}  MAE={m['Ms']['MAE']:.4f}  median |Δ|/Ms={m['Ms_median_frac_err']:.4f}")
print(f"  chi   : R²={m['chi']['R2']:.4f}  r={m['chi']['r']:.4f}  MAE={m['chi']['MAE']:.4f}  "
      f"(R² in {net.chi_space} space {m['chi_target_space']['R2']:.4f})")

print(f"\n  {'Ms bin':<12}{'N':>6}{'med|Δ|/Ms':>11}{'chi MAE':>9}{'chi r':>7}")
for lo, hi in [(0, 0.5), (0.5, 1), (1, 3), (3, 7), (7, 20.01)]:
    b = (pt["ms_true"] >= lo) & (pt["ms_true"] < hi)
    if b.sum() > 2:
        print(f"  [{lo:>4}, {hi:>5.1f}){b.sum():6d}"
              f"{np.median(np.abs(pt['ms_mean'][b] - pt['ms_true'][b]) / pt['ms_true'][b]):11.3f}"
              f"{np.mean(np.abs(pt['chi_mean'][b] - pt['chi_true'][b])):9.3f}"
              f"{np.corrcoef(pt['chi_true'][b], pt['chi_mean'][b])[0, 1]:7.3f}")

for k in net.DENS_FIELDS:
    if f"ln_{k}" in m:
        print(f"  ln {k:8s}: R²={m[f'ln_{k}']['R2']:.4f}  r={m[f'ln_{k}']['r']:.4f}  "
              f"median |Δ|/{k}={np.median(np.abs(pt[f'{k}_mean'] / pt[f'{k}_true'] - 1)):.3f}")
for key, mm in m.items():
    if key.startswith("chi_target_space_Ms"):
        print(f"  chi ({key[17:]}): R² in {net.chi_space} space {mm['R2']:.4f}")

# skill per simulation (median prediction per run vs its mean truth) and whether
# the network follows the time variation inside a run
test = splits["test"]
runm = {f"Ms>={c:g}": net.per_run_metrics(pt, test["sim_id"], c) for c in sorted({0.0, 1.0, net.chi_ms_min})}
print(f"\n  Per-run skill ({runm['Ms>=0']['n_runs']} runs):")
for cut, rm in runm.items():
    print(f"   {cut:7s} " + "  ".join(f"{k} R²={v['R2']:.3f} (within-run r {v['within_run_r']:.2f})"
                                     for k, v in rm.items() if isinstance(v, dict)))

ci = net.build_conformal_intervals(model, test_loader, q_ms, q_chi)
chi_ok = ci["ms_true"] >= net.chi_ms_min
cov = {k: {lev: float(np.mean(((ci[f"{k}_true"] >= ci[f"{k}_lo{lev}"]) & (ci[f"{k}_true"] <= ci[f"{k}_hi{lev}"]))
                              [chi_ok if k == "chi" else slice(None)]))
           for lev in (80, 90, 95)} for k in ("ms", "chi")}
print("\n  Conformal coverage on test (nominal 80/90/95): "
      f"Ms {cov['ms'][80]:.3f}/{cov['ms'][90]:.3f}/{cov['ms'][95]:.3f}   "
      f"chi {cov['chi'][80]:.3f}/{cov['chi'][90]:.3f}/{cov['chi'][95]:.3f}")

net.plot_predictions_dual(model, test_loader, tag="test")
cov_mae = net.plot_coverage_calibration(model, calib_loader, test_loader, tag="test")
net.plot_conformal_intervals_dual(ci, tag="test")

np.savez(os.path.join(args.model_dir, f"{run_name}_test.npz"),
         fold=args.fold, sim_id=test["sim_id"], frame=test["frame"], los=test["los"],
         xi_forcing=test["xi_forcing"], Ms_mean=test["Ms_mean"], sim_names=splits["sim_names"],
         n_H2=test["n_H2"], AV_median=test["AV_median"], chi_raw=test["chi_raw"],
         **pt, **{k: v for k, v in ci.items() if k.rsplit("_", 1)[-1][:2] in ("lo", "hi")})
with open(os.path.join(args.model_dir, f"{run_name}_test.json"), "w") as f:
    json.dump({"fold": args.fold, "chi_field": net.chi_field, "chi_space": net.chi_space,
               "chi_smooth_tc": net.chi_smooth_tc, "chi_ms_min": net.chi_ms_min, "dens_head": net.dens_head,
               "metrics": m, "per_run": runm, "coverage": cov, "coverage_mae_raw_conformal": cov_mae,
               "n_test": int(len(pt["ms_true"]))}, f, indent=2)
print(f"\nSaved test predictions + metrics → {args.model_dir}/{run_name}_test.{{npz,json}}")
print("All evaluation complete.")
