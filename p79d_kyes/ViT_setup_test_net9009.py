"""
run_ViT_test_net9009.py
=======================
Train, calibrate, and evaluate net9009 on mock-observation moment maps.

Workflow flags (set at the top):
  new_model   – instantiate architecture fresh
  load_model  – load weights from disk (for evaluation / fine-tuning)
  train_model – run training loop
  save_model  – persist best weights after training
  calibrate   – run conformal calibration on validation set
  plot_models – produce all diagnostic plots

Typical first run:  new_model=1, load_model=0, train_model=1, save_model=1,
                    calibrate=1, plot_models=1
Evaluation only:    new_model=1, load_model=1, train_model=0, save_model=0,
                    calibrate=1, plot_models=1
"""

from importlib import reload
import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
import loader
reload(loader)
# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, "/home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes/")

from torch.utils.data import DataLoader

# ── import the network ────────────────────────────────────────────────────────
import networks_nbisht.net9009 as net
reload(net)

# FLAGS
new_model   = 1
load_model  = 1
train_model = 0
save_model  = 0
calibrate   = 1      # compute conformal quantiles on validation set
plot_models = 1
final_eval = 1

net_name  = f"net{net.idd}"
model_dir = "/home/x-nbisht1/projects/p79d_dataset/models"
plot_dir  = os.path.join(os.environ["HOME"], "plots")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir,  exist_ok=True)


# DATA LOADING
if new_model or train_model or calibrate or plot_models:
    print(f"Loading data for {net_name}")
    all_data = net.load_data()

    print("\nTrain Ms range:",
          f"[{all_data['quantities']['train']['Ms_act'].min():.2f}, "
          f"{all_data['quantities']['train']['Ms_act'].max():.2f}]")
    print("Valid Ms range:",
          f"[{all_data['quantities']['valid']['Ms_act'].min():.2f}, "
          f"{all_data['quantities']['valid']['Ms_act'].max():.2f}]")
    if len(all_data["quantities"]["test"]["Ms_act"]) > 0:
        print("Test Ms range:",
            f"[{all_data['quantities']['test']['Ms_act'].min():.2f}, "
            f"{all_data['quantities']['test']['Ms_act'].max():.2f}]")
    else:
        print("Test Ms range: (held-out, loaded separately in final_eval)")

    # Quick distribution plot
    ms_train = all_data["quantities"]["train"]["Ms_act"]
    ms_valid = all_data["quantities"]["valid"]["Ms_act"]
    ms_test = all_data["quantities"]["test"]["Ms_act"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(ms_train, bins=50, alpha=0.6, label="Train", color='steelblue')
    axes[0].hist(ms_valid, bins=50, alpha=0.6, label="Valid", color='orange')
    if len(ms_test) > 0:
        axes[0].hist(ms_test, bins=50, alpha=0.6, label="Test", color='green')
    axes[0].set(xlabel="Ms", ylabel="Count", title="Data Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative
    axes[1].hist(ms_train, bins=50, alpha=0.6, cumulative=True, 
                label="Train", color='steelblue')
    axes[1].hist(ms_valid, bins=50, alpha=0.6, cumulative=True,
                label="Valid", color='orange')
    if len(ms_test) > 0:
        axes[1].hist(ms_test, bins=50, alpha=0.6, cumulative=True, label="Test", color='green')
    axes[1].set(xlabel="Ms", ylabel="Cumulative Count",
                title="Cumulative Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"ms_distribution_{net_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved distribution plot to {plot_dir}")

# MODEL INSTANTIATION
if new_model:
    model = net.thisnet()
    model.idd = net.idd
    print(f"Instantiated {net_name}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    print(f"log_temperature (init): {model.log_temperature.item():.4f}")

# LOAD WEIGHTS
if load_model:
    ckpt = os.path.join(model_dir, f"test{net.idd}.pth")
    if not os.path.exists(ckpt):
        print(f"Warning: Checkpoint {ckpt} not found!")
    else:
        model.load_state_dict(torch.load(ckpt, map_location=net.device))
        model = model.to(net.device)
        model.eval()
        print(f"Loaded weights from {ckpt}")

# TRAINING
if train_model:
    print(f"Training {net_name}")
    t0 = time.time()
    net.train(model, all_data)
    elapsed = time.time() - t0
    hrs, rem = divmod(int(elapsed), 3600)
    mnt, sec = divmod(rem, 60)
    print(f"\nTraining complete: {hrs:02d}:{mnt:02d}:{sec:02d}")

    if save_model:
        oname = os.path.join(model_dir, f"test{model.idd}.pth")
        torch.save(model.state_dict(), oname)
        print(f"Saved weights → {oname}")

    net.plot_loss_curve(model)

# BUILD DATA LOADERS FOR EVALUATION
print("\nCreating dataloaders for evaluation...")
ds_train = net.MockObsDataset(all_data["train"],
                               all_data["quantities"]["train"], augment=False)
ds_val   = net.MockObsDataset(all_data["valid"],
                               all_data["quantities"]["valid"], augment=False)
ds_test  = net.MockObsDataset(all_data["test"],
                               all_data["quantities"]["test"],  augment=False)

train_loader = DataLoader(ds_train, batch_size=128, shuffle=False, drop_last=False)
val_loader   = DataLoader(ds_val,   batch_size=128, shuffle=False, drop_last=False)
test_loader  = DataLoader(ds_test,  batch_size=128, shuffle=False, drop_last=False)

print(f"  Train loader: {len(train_loader)} batches")
print(f"  Val loader:   {len(val_loader)} batches")
print(f"  Test loader:  {len(test_loader)} batches")

# CONFORMAL CALIBRATION
q90, q95 = None, None

if calibrate:
    print("\n" + "=" * 70)
    print("Conformal calibration on validation set")
    print("=" * 70)
    model.eval()

    scores, cal_preds, cal_targets = net.compute_conformal_scores(model, val_loader)
    q90 = net.get_conformal_quantile(scores, coverage=0.90)
    q95 = net.get_conformal_quantile(scores, coverage=0.95)
    q80 = net.get_conformal_quantile(scores, coverage=0.80)

    print(f"Conformal quantiles:  q80 = {q80:.4f}  |  "
          f"q90 = {q90:.4f}  |  q95 = {q95:.4f}")
    print(f"Learnt temperature T = exp({model.log_temperature.item():.4f}) "
          f"= {model.log_temperature.exp().item():.4f}")

    # Check coverage on the calibration set itself (should match nominal)
    mae_cal = net.plot_coverage_calibration(model, val_loader, tag="val")
    print(f"Coverage calibration MAE on val set: {mae_cal:.4f}  "
          f"(target < 0.05 for good calibration)")

    # Save quantiles for re-use
    quantile_file = os.path.join(model_dir, f"conformal_quantiles_{net.idd}.npy")
    np.save(quantile_file, np.array([q80, q90, q95]))
    print(f"Saved conformal quantiles to {quantile_file}")

# PLOTS
if plot_models:
    print(f"Generating diagnostic plots for {net_name}")
    model.eval()

    # 1. Basic prediction quality on val set
    print("\n1. Prediction quality plots...")
    net.plot_predictions(model, val_loader, tag="val")
    model.eval()

    # 2. Conformal prediction intervals (requires calibration quantiles)
    if q90 is not None and q95 is not None:
        print("\n2. Conformal prediction intervals...")
        # Use first 200 test samples for the interval plot
        n_plot = min(200, len(all_data["valid"]))
        ds_sub = net.MockObsDataset(
            all_data["valid"][:n_plot],
            {"Ms_act": all_data["quantities"]["valid"]["Ms_act"][:n_plot],
             "Ma_act": all_data["quantities"]["valid"]["Ma_act"][:n_plot]},
            augment=False,
        )
        sub_loader = DataLoader(ds_sub, batch_size=64, shuffle=False)
        results    = net.build_conformal_intervals(
            model, sub_loader, q90, q95, res_factor=1.0)
        net.plot_conformal_intervals(results, tag=f"valid_{n_plot}")
        model.eval()

    # 3. Full uncertainty analysis (MC dropout)
    print("\n3. MC dropout uncertainty analysis (this may take a moment)...")
    n_mc = min(1000, len(all_data["valid"]))
    ds_mc = net.MockObsDataset(
        all_data["valid"][:n_mc],
        {"Ms_act": all_data["quantities"]["valid"]["Ms_act"][:n_mc],
         "Ma_act": all_data["quantities"]["valid"]["Ma_act"][:n_mc]},
        augment=False,
    )
    mc_loader = DataLoader(ds_mc, batch_size=64, shuffle=False)
    net.plot_uncertainty_analysis(model, mc_loader, tag=f"test_{n_mc}")
    model.eval()

    # 4. Resolution robustness on a few representative samples
    print("\n4. Resolution robustness plots...")
    for i, sample_idx in enumerate([0, 100, 500]):
        if sample_idx >= len(all_data["valid"]):
            continue
        x_sample = all_data["valid"][sample_idx:sample_idx + 1, 0:3].to(net.device)
        true_ms  = float(all_data["quantities"]["valid"]["Ms_act"][sample_idx])
        net.plot_resolution_robustness(model, x_sample, true_ms,
                                       tag=f"sample{sample_idx}")
        model.eval()

    # 5. Prediction quality summary stats
    print("Test set summary statistics:")
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            mu, _ = model(xb)
            all_preds.append(mu.squeeze().cpu())
            all_true.append(yb.squeeze().cpu())

    preds_np = torch.cat(all_preds).numpy()
    true_np  = torch.cat(all_true).numpy()
    
    preds_np = np.exp(preds_np)
    true_np  = np.exp(true_np)
    residuals = preds_np - true_np

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae  = float(np.mean(np.abs(residuals)))
    r, _ = pearsonr(true_np, preds_np)

    print(f"\nOverall metrics:")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  Pearson r: {r:.4f}")
    print(f"  R²       : {r**2:.4f}")

    # Per-bin breakdown
    bins = [0, 2, 4, 6, 8, 10, 15]
    print("\nPer-Mach-bin RMSE:")
    for i in range(len(bins) - 1):
        mask = (true_np >= bins[i]) & (true_np < bins[i + 1])
        if mask.sum() > 0:
            rmse_bin = float(np.sqrt(np.mean(residuals[mask] ** 2)))
            mae_bin = float(np.mean(np.abs(residuals[mask])))
            print(f"  Ms [{bins[i]:2d}–{bins[i+1]:2d}): "
                  f"N={mask.sum():4d}  RMSE={rmse_bin:.4f}  MAE={mae_bin:.4f}")

if final_eval:
    print("HELD-OUT TEST SET EVALUATION")

    #load the test file
    heldout_path = os.path.join(
        "/home/x-nbisht1/scratch/projects/radmc3d/p79d_dataset/",
        "p79d_mockobs_13CO_S128_test.h5"
    )
    print(f"Loading held-out test set: {heldout_path}")
    import h5py

    with h5py.File(heldout_path, 'r') as f:
        heldout_data = torch.from_numpy(f['subsets'][:]).float()
        heldout_ms   = f['Ms_act'][:]
        heldout_ma   = f['Ma_act'][:]

    ds_heldout = net.MockObsDataset(
        heldout_data,
        {"Ms_act": heldout_ms, "Ma_act": heldout_ma},
        augment=False,
    )
    heldout_loader = DataLoader(ds_heldout, batch_size=128,
                                shuffle=False, drop_last=False)
    print(f"Held-out samples: {len(ds_heldout)}")

    #collect predictions
    model.eval()
    ho_preds, ho_true, ho_sigma = [], [], []
    with torch.no_grad():
        for xb, yb in heldout_loader:
            mu, lv = model(xb)
            ho_preds.append(mu.squeeze().cpu())
            ho_true.append(yb.squeeze().cpu())
            ho_sigma.append(torch.exp(0.5 * lv).squeeze().cpu())

    ho_preds = torch.cat(ho_preds).numpy()
    ho_true  = torch.cat(ho_true).numpy()
    ho_sigma = torch.cat(ho_sigma).numpy()

    ho_preds = np.exp(ho_preds)
    ho_true  = np.exp(ho_true)
    ho_sigma = ho_preds * ho_sigma
    resid = ho_preds - ho_true

    #compute metrics
    from scipy.stats import pearsonr
    rmse_ho  = float(np.sqrt(np.mean(resid ** 2)))
    mae_ho   = float(np.mean(np.abs(resid)))
    mse_ho   = float(np.mean(resid ** 2))
    r_ho, _  = pearsonr(ho_true, ho_preds)
    ss_res   = np.sum(resid ** 2)
    ss_tot   = np.sum((ho_true - ho_true.mean()) ** 2)
    r2_ho    = float(1.0 - ss_res / ss_tot)

    print(f"\nHeld-out test set metrics:")
    print(f"  N        : {len(ho_true)}")
    print(f"  RMSE     : {rmse_ho:.4f}")
    print(f"  MAE      : {mae_ho:.4f}")
    print(f"  MSE      : {mse_ho:.4f}")
    print(f"  Pearson r: {r_ho:.4f}")
    print(f"  R²       : {r2_ho:.4f}")

    log_true  = np.log(ho_true)
    log_preds = np.log(ho_preds)
    log_resid = log_preds - log_true
    rmse_log  = float(np.sqrt(np.mean(log_resid**2)))
    mae_log   = float(np.mean(np.abs(log_resid)))
    ss_res_log = np.sum(log_resid**2)
    ss_tot_log = np.sum((log_true - log_true.mean())**2)
    r2_log    = float(1.0 - ss_res_log / ss_tot_log)
    print(f"  Log-space R²  : {r2_log:.4f}")
    print(f"  Log-space RMSE: {rmse_log:.4f}")
    print(f"  Log-space MAE : {mae_log:.4f}")

    # Per-bin breakdown
    bins = [0, 2, 4, 6, 8, 10, 15]
    print(f"\n  Per-Mach-bin breakdown:")
    print(f"  {'Bin':<12} {'N':>5}  {'RMSE':>7}  {'MAE':>7}")
    for i in range(len(bins) - 1):
        m = (ho_true >= bins[i]) & (ho_true < bins[i + 1])
        if m.sum() > 0:
            r_bin   = float(np.sqrt(np.mean(resid[m] ** 2)))
            mae_bin = float(np.mean(np.abs(resid[m])))
            print(f"  [{bins[i]:2d}–{bins[i+1]:2d})      "
                  f"{m.sum():5d}  {r_bin:7.4f}  {mae_bin:7.4f}")

    # ── main prediction plot with metrics embedded ───────────────────────────
    order = np.argsort(ho_true)
    lo    = min(ho_true.min(), ho_preds.min()) - 0.3
    hi    = max(ho_true.max(), ho_preds.max()) + 0.3

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # left panel: scatter pred vs truth
    ax = axes[0]
    sc = ax.scatter(ho_true, ho_preds, s=5, c=ho_sigma,
                    cmap="viridis", alpha=0.5, vmin=0, vmax=ho_sigma.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"Predicted $\sigma$", fontsize=10)

    metrics_text = (
        f"$R^2$  = {r2_ho:.4f}\n"
        f"$r$    = {r_ho:.4f}\n"
        f"MAE = {mae_ho:.4f}\n"
        f"MSE = {mse_ho:.4f}\n"
        f"RMSE = {rmse_ho:.4f}\n"
        f"N = {len(ho_true)}"
    )
    ax.text(0.04, 0.97, metrics_text,
            transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85),
            family="monospace")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Mach Number", fontsize=12)
    ax.set_ylabel("Predicted Mach Number", fontsize=12)
    ax.set_title(f"Held-Out Test: Pred vs Truth — net{net.idd}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # right panel: sorted predictions with per-bin RMSE annotations
    ax = axes[1]
    ax.errorbar(np.arange(len(order)), ho_preds[order],
                yerr=ho_sigma[order], fmt="none",
                ecolor="steelblue", alpha=0.25, elinewidth=0.4)
    ax.plot(ho_preds[order], "b.", ms=2, alpha=0.6, label="Predicted")
    ax.plot(ho_true[order],  "r-", lw=1.5, alpha=0.8, label="True")

    # annotate per-bin RMSE as horizontal text bands
    bin_colors = ["#e6f2ff", "#cce5ff", "#b3d9ff", "#99ccff", "#80bfff", "#66b2ff"]
    for i in range(len(bins) - 1):
        m = (ho_true >= bins[i]) & (ho_true < bins[i + 1])
        if m.sum() < 2:
            continue
        r_bin  = float(np.sqrt(np.mean(resid[m] ** 2)))
        # find x-range for this bin in sorted order
        idxs   = np.where(m[order])[0]
        x0, x1 = idxs.min(), idxs.max()
        ax.axvspan(x0, x1, alpha=0.07, color=bin_colors[i])
        ax.text((x0 + x1) / 2, hi - 0.5,
                f"RMSE\n{r_bin:.2f}",
                ha="center", va="top", fontsize=7,
                color="navy", alpha=0.75)

    ax.set_xlabel("Sample (sorted by true Ms)", fontsize=12)
    ax.set_ylabel("Mach Number", fontsize=12)
    ax.set_title(f"Sorted Predictions ± σ — net{net.idd}", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    oname = os.path.join(plot_dir, f"heldout_test_predictions_net{net.idd}.png")
    fig.savefig(oname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved held-out test prediction plot → {oname}")
    print("All evaluation complete.")