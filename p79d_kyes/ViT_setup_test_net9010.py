"""
ViT_setup_test_net9010.py
=========================
Train, calibrate, and evaluate net9010 (dual-output: Ms + xi).

Flags:
  new_model   – instantiate fresh
  load_model  – load weights from disk
  train_model – run training
  save_model  – persist best weights
  calibrate   – conformal calibration on validation set
  plot_models – diagnostic plots on validation set
  final_eval  – full metrics on held-out test set

Typical first run : new_model=1, load_model=0, train_model=1, save_model=1, calibrate=1, plot_models=1, final_eval=1
Evaluation only   : new_model=1, load_model=1, train_model=0, save_model=0, calibrate=1, plot_models=0, final_eval=1
"""

from importlib import reload
import sys, os, time
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, classification_report
torch.set_num_threads(128)
torch.set_num_interop_threads(8)

sys.path.insert(0, "/home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes/")
from torch.utils.data import DataLoader

import networks_nbisht.net9010 as net
reload(net)

# ─── FLAGS ────────────────────────────────────────────────────────────────────
new_model   = 1
load_model  = 0
train_model = 1
save_model  = 1
calibrate   = 1
plot_models = 1
final_eval  = 1

# ─── Paths ────────────────────────────────────────────────────────────────────
net_name  = f"net{net.idd}_{net.input_mode}"
model_dir = "/home/x-nbisht1/projects/p79d_dataset/models"
plot_dir  = os.path.join(os.environ["HOME"], "plots")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir,  exist_ok=True)
ckpt_path = os.path.join(model_dir, f"test{net.idd}_{net.input_mode}.pth")

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
print(f"Loading data for {net_name}")
splits = net.load_data()

print(f"\nMs ranges:")
for tag in ("train", "valid", "test"):
    ms = splits[tag]["Ms_act"]
    print(f"  {tag:5s}: [{ms.min():.2f}, {ms.max():.2f}]")

# Distribution plot for Ms and Xi
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = {"train": "steelblue", "valid": "orange", "test": "green"}
for tag in ("train", "valid", "test"):
    ms = splits[tag]["Ms_act"]
    xi = splits[tag]["xi_act"]

    # Ms PDF & CDF
    axes[0, 0].hist(ms, bins=50, alpha=0.6, label=tag, color=colors[tag])
    axes[0, 1].hist(ms, bins=50, alpha=0.6, cumulative=True, label=tag, color=colors[tag])

    # Xi PDF & CDF
    axes[1, 0].hist(xi, bins=25, alpha=0.6, label=tag, color=colors[tag])
    axes[1, 1].hist(xi, bins=25, alpha=0.6, cumulative=True, label=tag, color=colors[tag])

titles = [["Ms Distribution", "Ms Cumulative"], [r"$\xi$ Distribution", r"$\xi$ Cumulative"]]
xlabels = [["Ms", "Ms"], [r"$\xi$", r"$\xi$"]]

for i in range(2):
    for j in range(2):
        axes[i, j].set_xlabel(xlabels[i][j])
        axes[i, j].legend()
        axes[i, j].grid(True, alpha=0.3)
        axes[i, j].set_title(titles[i][j])

plt.tight_layout()
fig.savefig(os.path.join(plot_dir, f"ms_xi_distribution_{net_name}.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved Ms & Xi distribution plot.")

# ─── MODEL ────────────────────────────────────────────────────────────────────
if new_model:
    model = net.thisnet()
    model.idd = net.idd
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nInstantiated {net_name}  |  params: {n_params:,}")
    print(f"  log_temp_ms (init): {model.log_temp_ms.item():.4f}")
    print(f"  log_temp_xi (init): {model.log_temp_xi.item():.4f}")

if load_model:
    if not os.path.exists(ckpt_path):
        print(f"Warning: {ckpt_path} not found!")
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=net.device))
        model = model.to(net.device)
        model.eval()
        print(f"Loaded weights from {ckpt_path}")

# ─── TRAINING ─────────────────────────────────────────────────────────────────
if train_model:
    print(f"\nTraining {net_name}")
    t0 = time.time()
    net.train(model, splits)
    elapsed = time.time() - t0
    h, r = divmod(int(elapsed), 3600); m, s = divmod(r, 60)
    print(f"Training complete: {h:02d}:{m:02d}:{s:02d}")
    if save_model:
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved weights → {ckpt_path}")
    net.plot_loss_curve(model)

# ─── DATA LOADERS ─────────────────────────────────────────────────────────────
print("\nCreating dataloaders …")
ds_train = net.MachXiDataset(splits["train"], augment=False)
ds_val   = net.MachXiDataset(splits["valid"], augment=False)
ds_test  = net.MachXiDataset(splits["test"],  augment=False)

train_loader = DataLoader(ds_train, batch_size=128, shuffle=False, drop_last=False)
val_loader   = DataLoader(ds_val,   batch_size=128, shuffle=False, drop_last=False)
test_loader  = DataLoader(ds_test,  batch_size=128, shuffle=False, drop_last=False)

print(f"  Train: {len(train_loader)} batches  |  Val: {len(val_loader)}  |  Test: {len(test_loader)}")

# ─── CONFORMAL CALIBRATION ────────────────────────────────────────────────────
q90_ms = q95_ms = q90_xi = q95_xi = None

if calibrate:
    print("\n" + "=" * 70)
    print("Conformal calibration on validation set")
    print("=" * 70)
    model.eval()

    s_ms, s_xi, *_ = net.compute_conformal_scores(model, val_loader)

    q80_ms = net.get_conformal_quantile(s_ms, 0.80)
    q90_ms = net.get_conformal_quantile(s_ms, 0.90)
    q95_ms = net.get_conformal_quantile(s_ms, 0.95)

    q80_xi = net.get_conformal_quantile(s_xi, 0.80)
    q90_xi = net.get_conformal_quantile(s_xi, 0.90)
    q95_xi = net.get_conformal_quantile(s_xi, 0.95)

    print(f"Ms  quantiles:  q80={q80_ms:.4f}  q90={q90_ms:.4f}  q95={q95_ms:.4f}")
    print(f"xi  quantiles:  q80={q80_xi:.4f}  q90={q90_xi:.4f}  q95={q95_xi:.4f}")
    print(f"Temperatures:  T_ms={model.log_temp_ms.exp().item():.4f}"
          f"  T_xi={model.log_temp_xi.exp().item():.4f}")

    mae_ms, mae_xi = net.plot_coverage_calibration(model, val_loader, tag="val")
    model.eval()
    print(f"Coverage MAE — Ms: {mae_ms:.4f}  xi: {mae_xi:.4f}")

    np.save(os.path.join(model_dir, f"conformal_quantiles_{net.idd}_{net.input_mode}_ms.npy"),
            np.array([q80_ms, q90_ms, q95_ms]))
    np.save(os.path.join(model_dir, f"conformal_quantiles_{net.idd}_{net.input_mode}_xi.npy"),
            np.array([q80_xi, q90_xi, q95_xi]))
    print("Saved conformal quantiles.")

# ─── DIAGNOSTIC PLOTS (validation set) ───────────────────────────────────────
if plot_models:
    print(f"\nGenerating diagnostic plots for {net_name}")
    model.eval()
    net.plot_predictions_dual(model, val_loader, tag="val")
    model.eval()

    # 2. Conformal prediction intervals
    if q90_ms is not None and q90_xi is not None:
        print("\n2. Conformal prediction intervals...")
        n_plot    = min(200, len(splits["valid"]["data"]))
        sub_split = {"data":   splits["valid"]["data"][:n_plot],
                     "Ms_act": splits["valid"]["Ms_act"][:n_plot],
                     "xi_act": splits["valid"]["xi_act"][:n_plot]}
        sub_loader = DataLoader(net.MachXiDataset(sub_split, augment=False),
                                batch_size=64, shuffle=False)
        ci_results = net.build_conformal_intervals_dual(
            model, sub_loader, q90_ms, q95_ms, q90_xi, q95_xi)
        net.plot_conformal_intervals_dual(ci_results, tag=f"val_{n_plot}")
        model.eval()

    # 3. MC dropout uncertainty analysis
    print("\n3. MC dropout uncertainty analysis (this may take a moment)...")
    n_mc      = min(1000, len(splits["valid"]["data"]))
    mc_split  = {"data":   splits["valid"]["data"][:n_mc],
                 "Ms_act": splits["valid"]["Ms_act"][:n_mc],
                 "xi_act": splits["valid"]["xi_act"][:n_mc]}
    mc_loader = DataLoader(net.MachXiDataset(mc_split, augment=False),
                           batch_size=64, shuffle=False)
    net.plot_uncertainty_analysis_dual(model, mc_loader, tag=f"val_{n_mc}")
    model.eval()

    # 4. Resolution robustness
    print("\n4. Resolution robustness plots...")
    for sample_idx in [0, 100, 500]:
        if sample_idx >= len(splits["valid"]["data"]):
            continue
        x_sample = splits["valid"]["data"][
            sample_idx:sample_idx + 1, :net.n_input_channels].to(net.device)
        true_ms  = float(splits["valid"]["Ms_act"][sample_idx])
        true_xi  = float(splits["valid"]["xi_act"][sample_idx])
        net.plot_resolution_robustness_dual(model, x_sample, true_ms, true_xi,
                                            tag=f"sample{sample_idx}")
        model.eval()

    # 5. Summary stats on validation set
    print("\n5. Validation set summary statistics:")
    model.eval()
    ms_p, ms_t, xi_p, xi_t = [], [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            (mu_ms, _), pred_xi = model(xb)
            mu_xi, *_ = model.get_xi_stats(pred_xi)
            ms_p.append(mu_ms.squeeze().cpu()); ms_t.append(yb[:, 0].cpu())
            xi_p.append(mu_xi.squeeze().cpu()); xi_t.append(yb[:, 1].cpu())
    ms_pred_v = np.exp(torch.cat(ms_p).numpy())
    ms_true_v = np.exp(torch.cat(ms_t).numpy())
    xi_pred_v = torch.cat(xi_p).numpy()
    xi_true_v = torch.cat(xi_t).numpy()

    for pred, true, lbl in [(ms_pred_v, ms_true_v, "Ms"), (xi_pred_v, xi_true_v, "xi")]:
        resid  = pred - true
        r, _   = pearsonr(true, pred)
        ss_r   = np.sum(resid**2); ss_t = np.sum((true - true.mean())**2)
        r2     = float(1.0 - ss_r / ss_t)
        extra_str = ""
        if lbl == "xi":
            xi_grid = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
            xi_true_cls = np.argmin(np.abs(true[:, None] - xi_grid), axis=1)
            xi_pred_cls = np.argmin(np.abs(pred[:, None] - xi_grid), axis=1)
            acc = accuracy_score(xi_true_cls, xi_pred_cls)
            f1  = f1_score(xi_true_cls, xi_pred_cls, average="macro", zero_division=0)
            extra_str = f"  Acc={acc:.4f}  Macro-F1={f1:.4f}"

        print(f"  {lbl:2s}: RMSE={float(np.sqrt(np.mean(resid**2))):.4f}"
              f"  MAE={float(np.mean(np.abs(resid))):.4f}"
              f"  r={r:.4f}  R²={r2:.4f}{extra_str}")

# ─── FINAL EVALUATION (test set) ─────────────────────────────────────────────
if final_eval:
    print("\n" + "=" * 70)
    print("FINAL EVALUATION — HELD-OUT TEST SET")
    print("=" * 70)
    model.eval()

    ms_preds, xi_preds, xi_snaps = [], [], []
    ms_trues, xi_trues = [], []
    ms_sigmas, xi_sigmas = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            (mu_ms, lv_ms), pred_xi = model(xb)
            mu_xi, sig_xi, snap_xi = model.get_xi_stats(pred_xi)

            ms_preds.append(mu_ms.squeeze().cpu())
            xi_preds.append(mu_xi.squeeze().cpu())
            xi_snaps.append(snap_xi.squeeze().cpu())
            ms_sigmas.append(torch.exp(0.5 * lv_ms).squeeze().cpu())
            xi_sigmas.append(sig_xi.squeeze().cpu())
            ms_trues.append(yb[:, 0].cpu())
            xi_trues.append(yb[:, 1].cpu())

    # Convert Ms from log space
    ms_pred = np.exp(torch.cat(ms_preds).numpy())
    ms_true = np.exp(torch.cat(ms_trues).numpy())
    ms_sig  = ms_pred * torch.cat(ms_sigmas).numpy()

    xi_pred = torch.cat(xi_preds).numpy()
    xi_snap = torch.cat(xi_snaps).numpy()
    xi_true = torch.cat(xi_trues).numpy()
    xi_sig  = torch.cat(xi_sigmas).numpy()

    xi_grid = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    xi_true_cls = np.argmin(np.abs(xi_true[:, None] - xi_grid), axis=1)
    xi_pred_cls = np.argmin(np.abs(xi_snap[:, None] - xi_grid), axis=1)
    acc_xi = accuracy_score(xi_true_cls, xi_pred_cls)
    f1_xi  = f1_score(xi_true_cls, xi_pred_cls, average="macro", zero_division=0)

    def _metrics(pred, true, label):
        resid = pred - true
        rmse = float(np.sqrt(np.mean(resid**2)))
        mae  = float(np.mean(np.abs(resid)))
        r, _ = pearsonr(true, pred)
        ss_r = np.sum(resid**2); ss_t = np.sum((true - true.mean())**2)
        r2   = float(1.0 - ss_r / ss_t)
        print(f"\n  {label}:")
        print(f"    N={len(true):6d}  RMSE={rmse:.4f}  MAE={mae:.4f}"
              f"  r={r:.4f}  R²={r2:.4f}")

        # Per-bin (for Ms)
        if "Ms" in label:
            log_true = np.log(true); log_pred = np.log(pred)
            log_res  = log_pred - log_true
            r2_log   = float(1.0 - np.sum(log_res**2) / np.sum((log_true - log_true.mean())**2))
            print(f"    Log-space R²={r2_log:.4f}  RMSE={float(np.sqrt(np.mean(log_res**2))):.4f}"
                  f"  MAE={float(np.mean(np.abs(log_res))):.4f}")
            bins = [0, 2, 4, 6, 8, 10, 15, 20]
            print(f"    {'Bin':<10} {'N':>6}  {'RMSE':>7}  {'MAE':>7}")
            for i in range(len(bins)-1):
                m = (true >= bins[i]) & (true < bins[i+1])
                if m.sum() > 0:
                    r_b = float(np.sqrt(np.mean(resid[m]**2)))
                    m_b = float(np.mean(np.abs(resid[m])))
                    print(f"    [{bins[i]:2d}–{bins[i+1]:2d})    {m.sum():6d}  {r_b:7.4f}  {m_b:7.4f}")
        # Classification & Per-Grid metrics (for Xi)
        if "xi" in label or "Compressibility" in label:
            print(f"    Classification Accuracy = {acc_xi:.4f} ({acc_xi*100:.2f}%)")
            print(f"    Classification Macro-F1 = {f1_xi:.4f}")
            print("\n    Per-Grid Breakdown:")
            print(f"    {'Grid (ξ)':<10} {'N':>6}  {'Acc':>7}  {'RMSE':>7}  {'MAE':>7}  {'R²':>8}")
            for i, val in enumerate(xi_grid):
                m = (xi_true_cls == i)
                if m.sum() > 0:
                    c_acc = (xi_pred_cls[m] == i).mean()
                    res_b = pred[m] - true[m]
                    rmse_b = float(np.sqrt(np.mean(res_b**2)))
                    mae_b  = float(np.mean(np.abs(res_b)))
                    ss_r_b = np.sum(res_b**2)
                    ss_t_b = np.sum((true[m] - true[m].mean())**2)
                    r2_b   = float(1.0 - ss_r_b / ss_t_b) if ss_t_b > 1e-8 else float('nan')
                    print(f"    [{val:4.2f}]       {m.sum():6d}  {c_acc:7.4f}  {rmse_b:7.4f}  {mae_b:7.4f}  {r2_b:8.4f}")

        return rmse, mae, r, r2

    _metrics(ms_pred, ms_true, "Sonic Mach Number (Ms)")
    _metrics(xi_pred, xi_true, "Compressibility (ξ)")

    # ── Prediction plots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pred, true, sig, lbl in [
        (axes[0], ms_pred, ms_true, ms_sig, "Sonic Mach Number (Ms)"),
        (axes[1], xi_pred, xi_true, xi_sig, "Compressibility (ξ)"),
    ]:
        resid = pred - true
        rmse = float(np.sqrt(np.mean(resid**2)))
        mae  = float(np.mean(np.abs(resid)))
        r, _ = pearsonr(true, pred)
        ss_r = np.sum(resid**2); ss_t = np.sum((true - true.mean())**2)
        r2   = float(1.0 - ss_r / ss_t)

        lo = min(true.min(), pred.min()) - 0.05 * abs(true.max() - true.min())
        hi = max(true.max(), pred.max()) + 0.05 * abs(true.max() - true.min())

        sc = ax.scatter(true, pred, s=4, c=sig, cmap="viridis", alpha=0.5,
                        vmin=0, vmax=np.percentile(sig, 95))
        plt.colorbar(sc, ax=ax).set_label(r"$\sigma$", fontsize=9)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)

        if "Compressibility" in lbl or "xi" in lbl:
            txt = f"R²={r2:.4f}\nr={r:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}\nAcc={acc_xi:.4f}\nF1={f1_xi:.4f}\nN={len(true)}"
        else:
            txt = f"R²={r2:.4f}\nr={r:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}\nN={len(true)}"

        ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85),
                family="monospace")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"True {lbl}"); ax.set_ylabel(f"Predicted {lbl}")
        ax.set_title(f"Test Set — {lbl} — {net_name}"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    oname = os.path.join(plot_dir, f"heldout_test_predictions_{net_name}.png")
    fig.savefig(oname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved test prediction plot → {oname}")

    # ── Conformal intervals on test set ──────────────────────────────────────
    if q90_ms is not None:
        mae_ms_t, mae_xi_t = net.plot_coverage_calibration(model, test_loader, tag="test")
        model.eval()
        print(f"Test set coverage MAE — Ms: {mae_ms_t:.4f}  xi: {mae_xi_t:.4f}")

    print("\nAll evaluation complete.")