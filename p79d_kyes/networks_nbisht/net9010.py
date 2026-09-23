"""
net9010.py
==========
Hybrid CNN-ViT — dual-output version predicting:
  • Sonic Mach number  (Ms_act) — in log space
  • Compressibility    (xi)     — in linear space

Input modes (set n_input_channels at the top):
  1 → mom0 only  (density channel + res indicator)  → saved as *_mom0.*
  3 → all moments (mom0/mom1/mom2 + res indicator)  → saved as *_allmom.*

Data: combine two HDF5 files, split 80 / 15 / 5 (train / test / valid).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
import time
import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import h5py

idd  = 9010

# ─── Input mode ──────────────────────────────────────────────────────────────
n_input_channels = 3                              # 1 or 3
input_mode       = "allmom" if n_input_channels == 3 else "mom0"
n_model_channels = n_input_channels + 1           # +1 for resolution channel

what = (f"Hybrid CNN-ViT | dual-output Ms+xi | {input_mode} | "
        f"resolution-aware | temperature-scaled conformal | MC-dropout")

# ─── Data paths ──────────────────────────────────────────────────────────────
dirpath = "/home/x-nbisht1/projects/p79d_dataset/"
fnames  = [
    "p79d_dataset_S256_N1_xyz_down_128_mach_grid_tvs_second.h5",
    "p79d_dataset_S256_N1_xyz_down_128_mach_grid_tvs_first.h5",
]

# ─── Split fractions ─────────────────────────────────────────────────────────
frac_train = 0.80
frac_test  = 0.15
frac_valid = 0.05     # conformal calibration

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Training hyperparameters ─────────────────────────────────────────────────
epochs       = 100
lr           = 5e-4
batch_size   = 512
weight_decay = 0.01

# ─── Architecture ─────────────────────────────────────────────────────────────
img_size        = 128
cnn_output_size = 64
patch_size      = 8
embed_dim       = 384
depth           = 6
num_heads       = 6
mlp_ratio       = 4.0
dropout         = 0.1
mc_dropout_p    = 0.10

# ─── Resolution-aware training ────────────────────────────────────────────────
res_train_prob = 0.50
res_min_factor = 0.40

# ─── MC uncertainty ───────────────────────────────────────────────────────────
n_mc_samples = 30


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(seed: int = 42):
    """
    Combine both HDF5 files, filter Ms <= 20, stratify split across 2D bins (Ms x xi),
    and split 80 / 15 / 5 (train / test / valid).
    """
    print("Reading data …")
    parts_data, parts_ms, parts_ma, parts_xi = [], [], [], []

    for fname in fnames:
        path = dirpath + fname
        print(f"  Loading {fname} …")
        with h5py.File(path, "r") as f:
            parts_data.append(torch.from_numpy(f["subsets"][:]).float())
            parts_ms.append(f["Ms_act"][:])
            parts_ma.append(f["Ma_act"][:])
            parts_xi.append(f["xi"][:])

    data   = torch.cat(parts_data, dim=0)        # [N, 3, 128, 128]
    ms_act = np.concatenate(parts_ms)
    ma_act = np.concatenate(parts_ma)
    xi_act = np.concatenate(parts_xi)

    # Filter out extreme Mach values (Ms > 20)
    valid_mask = ms_act <= 20.0
    data   = data[valid_mask]
    ms_act = ms_act[valid_mask]
    ma_act = ma_act[valid_mask]
    xi_act = xi_act[valid_mask]
    n_total = len(data)

    print(f"\nCombined & Filtered (Ms <= 20): {n_total} samples")
    print(f"  Ms range : [{ms_act.min():.3f}, {ms_act.max():.3f}]")
    print(f"  xi range : [{xi_act.min():.4f}, {xi_act.max():.4f}]")

    # ── 2D Stratification: Bin by Ms and then by Xi ──
    ms_bins = [0, 2, 4, 6, 8, 10, 15, 20.01]
    xi_bins = [-0.01, 0.125, 0.375, 0.625, 0.875, 1.01]

    ms_b = np.digitize(ms_act, ms_bins) - 1
    xi_b = np.digitize(xi_act, xi_bins) - 1
    group_ids = ms_b * len(xi_bins) + xi_b

    train_idx, test_idx, valid_idx = [], [], []
    rng = np.random.default_rng(seed)

    for gid in np.unique(group_ids):
        g_indices = np.where(group_ids == gid)[0]
        rng.shuffle(g_indices)
        n_g = len(g_indices)
        n_tr = int(frac_train * n_g)
        n_te = int(frac_test * n_g)

        train_idx.extend(g_indices[:n_tr])
        test_idx.extend(g_indices[n_tr:n_tr + n_te])
        valid_idx.extend(g_indices[n_tr + n_te:])

    train_idx = rng.permutation(np.array(train_idx))
    test_idx  = rng.permutation(np.array(test_idx))
    valid_idx = rng.permutation(np.array(valid_idx))

    splits = {}
    for tag, sidx in [("train", train_idx),
                       ("valid", valid_idx),
                       ("test",  test_idx)]:
        splits[tag] = {
            "data":   data[sidx],
            "Ms_act": ms_act[sidx],
            "Ma_act": ma_act[sidx],
            "xi_act": xi_act[sidx],
        }

    print("\nSplit sizes and distributions:")
    xi_grid = [0.0, 0.25, 0.50, 0.75, 1.0]
    for tag in ("train", "valid", "test"):
        n  = len(splits[tag]["data"])
        ms = splits[tag]["Ms_act"]
        xi = splits[tag]["xi_act"]
        print(f"  {tag:5s}: {n:6d}")
        print(f"    Ms bins : ", end="")
        for i in range(len(ms_bins) - 1):
            m = int(((ms >= ms_bins[i]) & (ms < ms_bins[i + 1])).sum())
            print(f"[{ms_bins[i]:2d}-{ms_bins[i+1]:2.0f}): {m:5d}  ", end="")
        print(f"\n    xi bins : ", end="")
        for val in xi_grid:
            m = int(np.isclose(xi, val, atol=0.05).sum())
            print(f"[{val:.2f}]: {m:5d}  ", end="")
        print("\n")

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def _degrade_resolution(x: torch.Tensor, factor: float) -> torch.Tensor:
    squeeze = x.ndim == 3
    if squeeze:
        x = x.unsqueeze(0)
    H, W = x.shape[-2], x.shape[-1]
    sh, sw = max(4, int(H * factor)), max(4, int(W * factor))
    x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
    x = F.interpolate(x, size=(H,  W),  mode="bilinear", align_corners=False)
    if squeeze:
        x = x.squeeze(0)
    return x


class MachXiDataset(Dataset):
    """
    Returns:
        x       : [n_model_channels, 128, 128]  — input channels + res indicator
        targets : [2]                            — [log(Ms_act), xi_act]

    n_input_channels controls whether x uses mom0 only (1) or all moments (3).
    The resolution indicator is always appended as the last channel.
    """
    def __init__(self, split_dict, augment: bool = False):
        self.data    = split_dict["data"]     # [N, 3, 128, 128]
        self.ms_act  = split_dict["Ms_act"]
        self.xi_act  = split_dict["xi_act"]
        self.augment = augment

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx][:n_input_channels].clone()   # [C, H, W]
        res_factor = 1.0

        if self.augment:
            H, W = x.shape[-2], x.shape[-1]
            dy   = torch.randint(0, H, (1,)).item()
            dx   = torch.randint(0, W, (1,)).item()
            x    = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
            if torch.rand(1) > 0.5: x = torch.flip(x, dims=[-1])
            if torch.rand(1) > 0.5: x = torch.flip(x, dims=[-2])
            if torch.rand(1).item() < res_train_prob:
                res_factor = res_min_factor + (1.0 - res_min_factor) * torch.rand(1).item()
                x = _degrade_resolution(x, res_factor)

        res_ch = torch.full((1, x.shape[-2], x.shape[-1]),
                            fill_value=res_factor, dtype=x.dtype)
        x = torch.cat([x, res_ch], dim=0)   # [n_model_channels, H, W]

        ms  = float(self.ms_act[idx])
        xi  = float(self.xi_act[idx])
        tgt = torch.tensor([np.log(ms + 1e-6), xi], dtype=torch.float32)

        return x.to(device), tgt.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE COMPONENTS  (CNNStem uses n_model_channels; rest unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class CNNStem(nn.Module):
    def __init__(self, in_chans: int = 4, out_chans: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_chans, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(out_chans)
        self.pool  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.pool(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=64, embed_dim=384):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.fc2  = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, mc_drop=0.0):
        super().__init__()
        self.norm1   = nn.LayerNorm(dim)
        self.attn    = Attention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm2   = nn.LayerNorm(dim)
        self.mlp     = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.mc_drop = nn.Dropout(mc_drop)

    def forward(self, x):
        x = x + self.mc_drop(self.attn(self.norm1(x)))
        x = x + self.mc_drop(self.mlp(self.norm2(x)))
        return x


class OrdinalLoss(nn.Module):
    """
    Earth Mover's Distance (Wasserstein-1) Loss for Ordinal Classification.
    Penalizes class distance by computing L2 norm on cumulative distribution functions (CDFs).
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
        # logits: [B, K], target_cls: [B] (integer class indices 0..K-1)
        probs = F.softmax(logits, dim=-1)
        target_onehot = F.one_hot(target_cls, num_classes=self.num_classes).float()

        # Cumulative sums along class dimension
        cdf_pred = torch.cumsum(probs, dim=-1)
        cdf_true = torch.cumsum(target_onehot, dim=-1)

        # L2 distance between CDFs
        loss = torch.mean(torch.sum((cdf_pred - cdf_true) ** 2, dim=-1))
        return loss

# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class HybridCNNViT(nn.Module):
    """
    Net9010: dual-output version of net9009.
    Two separate output heads (Ms, xi) each predicting (mean, log_var).
    Two separate learnable temperature scalars for calibration.
    """
    def __init__(
        self,
        img_size=128, cnn_output_size=64, patch_size=8,
        in_chans=4, embed_dim=384, depth=6, num_heads=6,
        mlp_ratio=4.0, dropout=0.1, mc_drop=0.10,
    ):
        super().__init__()
        self.cnn_stem    = CNNStem(in_chans=in_chans, out_chans=64)
        self.patch_embed = PatchEmbed(cnn_output_size, patch_size, 64, embed_dim)
        n_patches        = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, mc_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Separate regression heads for each target: [mean, logvar]
        self.head_ms = nn.Linear(embed_dim, 2)   # [mean_ms, logvar_ms]
        self.head_xi = nn.Linear(embed_dim, 2)   # [mean_xi, logvar_xi]

        # Temperature scalars for scaling
        self.log_temp_ms = nn.Parameter(torch.zeros(1))
        self.log_temp_xi = nn.Parameter(torch.zeros(1))

        # Discrete grid reference for snapping
        self.register_buffer("xi_grid", torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0]))

        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve",   torch.zeros(epochs))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias); nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    def _backbone(self, x):
        B  = x.shape[0]
        x  = self.cnn_stem(x)
        x  = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x  = torch.cat([cls, x], dim=1) + self.pos_embed
        x  = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]   # CLS token  [B, D]

    def forward(self, x):
        """
        Returns:
            (mean_ms [B,1], lv_ms [B,1]),
            (mean_xi [B,1], lv_xi [B,1]),
        """
        feat = self._backbone(x)

        out_ms  = self.head_ms(feat)
        mean_ms = out_ms[:, 0:1]
        lv_ms   = (out_ms[:, 1:2] + 2.0 * self.log_temp_ms).clamp(-10, 5)

        out_xi  = self.head_xi(feat)
        mean_xi = out_xi[:, 0:1]
        lv_xi   = (out_xi[:, 1:2] + 2.0 * self.log_temp_xi).clamp(-10, 5)

        return (mean_ms, lv_ms), (mean_xi, lv_xi)


    def mc_forward(self, x, n_samples: int = n_mc_samples):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        with torch.no_grad():
            ms_means, ms_sigmas = [], []
            xi_means, xi_sigmas = [], []
            for _ in range(n_samples):
                (mu_ms, lv_ms), pred_xi = self(x)
                m_xi, s_xi, _ = self.get_xi_stats(pred_xi)

                ms_means.append(mu_ms)
                ms_sigmas.append(torch.exp(0.5 * lv_ms))
                xi_means.append(m_xi)
                xi_sigmas.append(s_xi)

        self.eval()

        def _stats(means, sigmas):
            stk_m = torch.stack(means,  dim=0)
            stk_s = torch.stack(sigmas, dim=0)
            mean  = stk_m.mean(0)
            epis  = stk_m.std(0)
            alea  = stk_s.mean(0)
            total = torch.sqrt(alea**2 + epis**2)
            return mean, alea, epis, total

        return _stats(ms_means, ms_sigmas), _stats(xi_means, xi_sigmas)

    def criterion(self, pred, target):
        """
        Combined Gaussian NLL for Ms (log space) and Xi (linear space).
        """
        (mean_ms, lv_ms), (mean_xi, lv_xi) = pred
        tgt_ms = target[:, 0:1]   # log(Ms)
        tgt_xi = target[:, 1:2]   # continuous xi in [0.0, 1.0]

        # 1. Ms Regression Loss
        nll_ms = 0.5 * (lv_ms + (tgt_ms - mean_ms)**2 * torch.exp(-lv_ms))
        w_ms   = torch.where(tgt_ms > np.log(10), 3.0,
                 torch.where(tgt_ms > np.log(4),  2.0, 1.0))
        aux_ms  = (w_ms * (mean_ms - tgt_ms)**2).mean()
        loss_ms = 0.9 * nll_ms.mean() + 0.1 * aux_ms

        # 2. Xi Continuous Regression Loss (Gaussian NLL)
        nll_xi  = 0.5 * (lv_xi + (tgt_xi - mean_xi)**2 * torch.exp(-lv_xi))
        loss_xi = nll_xi.mean()

        return loss_ms + loss_xi

    def get_xi_stats(self, pred_xi):
        """
        Accepts continuous prediction tuple (mean_xi, lv_xi).
        Returns continuous mean, standard deviation, and grid-snapped value.
        """
        mean_xi, lv_xi = pred_xi
        sig_xi = torch.exp(0.5 * lv_xi)

        # Distance calculation to grid {0.00, 0.25, 0.50, 0.75, 1.00}
        idx_snap = torch.argmin(torch.abs(mean_xi - self.xi_grid), dim=-1, keepdim=True)
        snap_xi  = self.xi_grid[idx_snap]

        return mean_xi, sig_xi, snap_xi


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def thisnet():
    return HybridCNNViT(
        img_size=img_size, cnn_output_size=cnn_output_size,
        patch_size=patch_size, in_chans=n_model_channels,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=mlp_ratio, dropout=dropout, mc_drop=mc_dropout_p,
    ).to(device)


def train(model, splits):
    trainer(model, splits, epochs=epochs, lr=lr,
            batch_size=batch_size, weight_decay=weight_decay)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed=8675309):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def trainer(model, splits, epochs=100, batch_size=64, lr=5e-4,
            weight_decay=0.01, patience=15):
    set_seed()

    ds_train = MachXiDataset(splits["train"], augment=True)
    ds_val   = MachXiDataset(splits["valid"], augment=False)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                            drop_last=False, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=4, pin_memory=True)

    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Adaptive LR reduction on validation loss plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    total_steps = epochs * max(1, len(train_loader))
    print(f"Total Steps {total_steps}  |  ntrain {len(ds_train)}  |  nvalid {len(ds_val)}  |  epochs {epochs}")
    print(f"Architecture: {input_mode} | {n_model_channels}-ch input | MC-dropout={mc_dropout_p}")

    best_val, best_state, bad_epochs = float("inf"), None, 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        train_loss = running / len(ds_train)
        model.train_curve[epoch - 1] = train_loss

        model.eval()
        with torch.no_grad():
            vtotal = sum(model.criterion(model(xb), yb).item() * xb.size(0)
                         for xb, yb in val_loader)
        val_loss = vtotal / len(ds_val)
        model.val_curve[epoch - 1] = val_loss

        # Step adaptive scheduler based on val_loss
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        now  = time.time()
        tpe  = (now - t0) / epoch
        left = tpe * (epochs - epoch)
        eta  = datetime.datetime.fromtimestamp(now + left)

        def fmt(s):
            m, s = divmod(int(s), 60); h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        print(f"[{epoch:3d}/{epochs}] net{idd}  "
              f"train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | bad {bad_epochs:02d} | "
              f"T_ms {model.log_temp_ms.item():.3f} T_xi {model.log_temp_xi.item():.3f} | "
              f"ETA {eta.strftime('%H:%M:%S')} | Remain {fmt(left)} | Sofar {fmt(now-t0)}")

        # Trigger Early Stopping
        if bad_epochs >= patience:
            print(f"\n[Early Stopping] Validation loss did not improve for {patience} epochs. Terminating early.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model | val loss {best_val:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL PREDICTION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_conformal_scores(model, cal_loader):
    """
    Returns nonconformity scores for both targets.
    scores_ms, scores_xi : |y - μ| / σ  in their respective spaces.
    Also returns raw preds and targets for diagnostics.
    """
    model.eval()
    s_ms, s_xi = [], []
    p_ms, p_xi, t_ms, t_xi = [], [], [], []

    with torch.no_grad():
        for xb, yb in cal_loader:
            (mu_ms, lv_ms), pred_xi = model(xb)
            mu_xi, sig_xi, _ = model.get_xi_stats(pred_xi)
            sig_ms = torch.exp(0.5 * lv_ms)
            y_ms   = yb[:, 0:1];  y_xi = yb[:, 1:2]

            s_ms.append(((y_ms - mu_ms).abs() / (sig_ms + 1e-8)).squeeze().cpu())
            s_xi.append(((y_xi - mu_xi).abs() / (sig_xi + 1e-8)).squeeze().cpu())
            p_ms.append(mu_ms.squeeze().cpu());  t_ms.append(y_ms.squeeze().cpu())
            p_xi.append(mu_xi.squeeze().cpu());  t_xi.append(y_xi.squeeze().cpu())

    return (torch.cat(s_ms),  torch.cat(s_xi),
            torch.cat(p_ms),  torch.cat(p_xi),
            torch.cat(t_ms),  torch.cat(t_xi))


def get_conformal_quantile(scores: torch.Tensor, coverage: float) -> float:
    n   = len(scores)
    lvl = min(math.ceil((n + 1) * coverage) / n, 1.0)
    return float(torch.quantile(scores, lvl).item())


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(model):
    import matplotlib.pyplot as plt, os
    n     = int((model.train_curve != 0).sum().item())
    train = model.train_curve[:n].cpu().numpy()
    val   = model.val_curve[:n].cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train, lw=2, label="Train Loss")
    ax.plot(val,   lw=2, label="Val Loss")
    ax.set_yscale("symlog")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Combined NLL")
    ax.set_title(f"Training Curves – net{idd} ({input_mode})"); ax.legend(); ax.grid(True, alpha=0.3)
    oname = f"{os.environ['HOME']}/plots/loss_curve_net{idd}_{input_mode}.png"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight"); print(f"Saved: {oname}"); plt.close(fig)


def plot_predictions_dual(model, loader, tag="val"):
    """Two-panel scatter plot: Ms predictions and xi predictions with classification inlay."""
    import matplotlib.pyplot as plt, os
    from scipy.stats import pearsonr
    model.eval()
    ms_pred, xi_pred, ms_true, xi_true = [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            (mu_ms, _), pred_xi = model(xb)
            mu_xi, *_ = model.get_xi_stats(pred_xi)
            ms_pred.append(mu_ms.squeeze().cpu())
            xi_pred.append(mu_xi.squeeze().cpu())
            ms_true.append(yb[:, 0].cpu())
            xi_true.append(yb[:, 1].cpu())
    ms_pred = np.exp(torch.cat(ms_pred).numpy())
    ms_true = np.exp(torch.cat(ms_true).numpy())
    xi_pred = torch.cat(xi_pred).numpy()
    xi_true = torch.cat(xi_true).numpy()

    xi_grid = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    xi_true_cls = np.argmin(np.abs(xi_true[:, None] - xi_grid), axis=1)
    xi_pred_cls = np.argmin(np.abs(xi_pred[:, None] - xi_grid), axis=1)
    acc_xi = accuracy_score(xi_true_cls, xi_pred_cls)
    f1_xi  = f1_score(xi_true_cls, xi_pred_cls, average="macro", zero_division=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pred, true, lbl in [
        (axes[0], ms_pred, ms_true, "Sonic Mach Number (Ms)"),
        (axes[1], xi_pred, xi_true, "Compressibility (ξ)"),
    ]:
        resid = pred - true
        rmse = float(np.sqrt(np.mean(resid**2)))
        mae  = float(np.mean(np.abs(resid)))
        r, _ = pearsonr(true, pred)
        ss_r = np.sum(resid**2); ss_t = np.sum((true - true.mean())**2)
        r2   = float(1.0 - ss_r / ss_t)

        ax.scatter(true, pred, s=4, alpha=0.4, color="steelblue")
        lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)

        if "Compressibility" in lbl or "xi" in lbl:
            txt = f"R²={r2:.4f}\nr={r:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}\nAcc={acc_xi:.4f}\nF1={f1_xi:.4f}\nN={len(true)}"
        else:
            txt = f"R²={r2:.4f}\nr={r:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}\nN={len(true)}"

        ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85),
                family="monospace")
        ax.set_xlabel(f"True {lbl}"); ax.set_ylabel(f"Predicted {lbl}")
        ax.set_title(f"net{idd} ({input_mode}) — {tag}"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    oname = f"{os.environ['HOME']}/plots/predictions_net{idd}_{input_mode}_{tag}.png"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight"); print(f"Saved: {oname}"); plt.close(fig)


def plot_coverage_calibration(model, cal_loader, tag="val"):
    """Coverage calibration for both targets; returns (mae_ms, mae_xi)."""
    import matplotlib.pyplot as plt, os
    s_ms, s_xi, *_ = compute_conformal_scores(model, cal_loader)
    nominal = np.arange(0.50, 1.00, 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    maes = []
    for ax, scores, lbl in [(axes[0], s_ms, "Ms"), (axes[1], s_xi, "xi")]:
        emp = np.array([(scores <= get_conformal_quantile(scores, a)).float().mean().item()
                        for a in nominal])
        mae = float(np.mean(np.abs(emp - nominal)))
        maes.append(mae)
        ax.plot(nominal, emp, "bo-", lw=2, label=f"Conformal ({lbl})")
        ax.plot([0.5, 1.0], [0.5, 1.0], "r--", lw=1.5)
        ax.annotate(f"MAE={mae:.3f}", xy=(0.55, 0.45), fontsize=10,
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
        ax.set_xlabel("Nominal Coverage"); ax.set_ylabel("Empirical Coverage")
        ax.set_title(f"Calibration – {lbl} ({tag})"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    oname = f"{os.environ['HOME']}/plots/coverage_calibration_net{idd}_{input_mode}_{tag}.png"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight")
    print(f"Saved: {oname}  (MAE_ms={maes[0]:.4f}  MAE_xi={maes[1]:.4f})")
    plt.close(fig)
    return maes[0], maes[1]


def build_conformal_intervals_dual(model, data_loader,
                                   q90_ms, q95_ms, q90_xi, q95_xi):
    """Conformal intervals for both Ms (returned in Ms units) and xi (linear)."""
    model.eval()
    ms_mu, ms_lo90, ms_hi90, ms_lo95, ms_hi95 = [], [], [], [], []
    xi_mu, xi_lo90, xi_hi90, xi_lo95, xi_hi95 = [], [], [], [], []
    ms_sig, xi_sig, ms_tr, xi_tr = [], [], [], []

    with torch.no_grad():
        for xb, yb in data_loader:
            (mu_ms, lv_ms), pred_xi = model(xb)
            m_xi, s_xi, _ = model.get_xi_stats(pred_xi)
            s_ms = torch.exp(0.5 * lv_ms).squeeze().cpu()
            s_xi = s_xi.squeeze().cpu()
            m_ms = mu_ms.squeeze().cpu()
            m_xi = m_xi.squeeze().cpu()
            ms_mu.append(m_ms);  ms_sig.append(s_ms)
            ms_lo90.append(m_ms - q90_ms * s_ms)
            ms_hi90.append(m_ms + q90_ms * s_ms)
            ms_lo95.append(m_ms - q95_ms * s_ms)
            ms_hi95.append(m_ms + q95_ms * s_ms)
            ms_tr.append(yb[:, 0].cpu())   # log(Ms)
            xi_mu.append(m_xi);  xi_sig.append(s_xi)
            xi_lo90.append(m_xi - q90_xi * s_xi)
            xi_hi90.append(m_xi + q90_xi * s_xi)
            xi_lo95.append(m_xi - q95_xi * s_xi)
            xi_hi95.append(m_xi + q95_xi * s_xi)
            xi_tr.append(yb[:, 1].cpu())   # xi

    def c(lst): return torch.cat(lst).numpy()
    return {
        # Ms — exp'd to physical units
        "ms_mean":  np.exp(c(ms_mu)),
        "ms_lo90":  np.exp(c(ms_lo90)), "ms_hi90": np.exp(c(ms_hi90)),
        "ms_lo95":  np.exp(c(ms_lo95)), "ms_hi95": np.exp(c(ms_hi95)),
        "ms_true":  np.exp(c(ms_tr)),   "ms_sigma": c(ms_sig),
        # xi — linear
        "xi_mean":  c(xi_mu),
        "xi_lo90":  c(xi_lo90), "xi_hi90": c(xi_hi90),
        "xi_lo95":  c(xi_lo95), "xi_hi95": c(xi_hi95),
        "xi_true":  c(xi_tr),   "xi_sigma": c(xi_sig),
    }


def plot_conformal_intervals_dual(results, tag="test"):
    """2-row conformal interval plot: Ms (top) and xi (bottom)."""
    import matplotlib.pyplot as plt, os
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    pairs = [
        ("ms", r"Sonic Mach Number $\mathcal{M}_s$"),
        ("xi", r"Compressibility $\xi$"),
    ]
    for ax, (key, lbl) in zip(axes, pairs):
        true  = results[f"{key}_true"]
        order = np.argsort(true)
        x     = np.arange(len(order))
        ax.fill_between(x, results[f"{key}_lo95"][order],
                           results[f"{key}_hi95"][order],
                           alpha=0.30, color="royalblue", label="95% CI")
        ax.fill_between(x, results[f"{key}_lo90"][order],
                           results[f"{key}_hi90"][order],
                           alpha=0.45, color="royalblue", label="90% CI")
        ax.plot(x, results[f"{key}_mean"][order], "b-", lw=1.5, label="Prediction")
        ax.scatter(x, true[order], s=8, c="red", zorder=3, label="True")
        ax.set_xlabel(f"Sample (sorted by true {lbl})", fontsize=10)
        ax.set_ylabel(lbl, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Conformal Intervals — net{idd} ({input_mode}) — {tag}", fontsize=13)
    plt.tight_layout()
    oname = f"{os.environ['HOME']}/plots/conformal_intervals_net{idd}_{input_mode}_{tag}.png"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight"); print(f"Saved: {oname}"); plt.close(fig)


def plot_uncertainty_analysis_dual(model, test_loader, tag="test"):
    """
    4-row × 2-column uncertainty analysis.
    Left column: Ms  |  Right column: xi
    Rows: sorted predictions, pred vs truth, epistemic vs aleatoric, histogram.
    Ms values converted to physical units via exp(); uncertainties shown in log units.
    """
    import matplotlib.pyplot as plt, os
    model.eval()
    ms_mu, ms_al, ms_ep, ms_tot, ms_tr = [], [], [], [], []
    xi_mu, xi_al, xi_ep, xi_tot, xi_tr = [], [], [], [], []

    for xb, yb in test_loader:
        ms_r, xi_r = model.mc_forward(xb, n_samples=n_mc_samples)
        ms_mu.append(ms_r[0].squeeze().cpu()); ms_al.append(ms_r[1].squeeze().cpu())
        ms_ep.append(ms_r[2].squeeze().cpu()); ms_tot.append(ms_r[3].squeeze().cpu())
        ms_tr.append(yb[:, 0].cpu())
        xi_mu.append(xi_r[0].squeeze().cpu()); xi_al.append(xi_r[1].squeeze().cpu())
        xi_ep.append(xi_r[2].squeeze().cpu()); xi_tot.append(xi_r[3].squeeze().cpu())
        xi_tr.append(yb[:, 1].cpu())

    # Ms: exp to physical; σ in log units (shown as-is — annotate axes accordingly)
    ms_means  = np.exp(torch.cat(ms_mu).numpy())
    ms_trues  = np.exp(torch.cat(ms_tr).numpy())
    ms_alea   = torch.cat(ms_al).numpy()
    ms_epis   = torch.cat(ms_ep).numpy()
    ms_totals = torch.cat(ms_tot).numpy()
    # σ_Ms ≈ Ms * σ_log  for sorted-prediction error bars
    ms_tot_ms = ms_means * ms_totals

    xi_means  = torch.cat(xi_mu).numpy()
    xi_trues  = torch.cat(xi_tr).numpy()
    xi_alea   = torch.cat(xi_al).numpy()
    xi_epis   = torch.cat(xi_ep).numpy()
    xi_totals = torch.cat(xi_tot).numpy()

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    data = [
        (ms_means, ms_alea, ms_epis, ms_totals, ms_tot_ms, ms_trues,
         r"$\mathcal{M}_s$", "log units"),
        (xi_means, xi_alea, xi_epis, xi_totals, xi_totals, xi_trues,
         r"$\xi$", "linear units"),
    ]

    for col, (means, alea, epis, totals_log, totals_phys, trues, lbl, ulbl) in enumerate(data):
        order = np.argsort(trues)
        x     = np.arange(len(order))

        ax = axes[0, col]
        ax.plot(trues[order], "r-", lw=1.5, label="True"); 
        ax.errorbar(x, means[order], yerr=totals_phys[order],
                    fmt="none", ecolor="steelblue", alpha=0.3, elinewidth=0.5)
        ax.plot(means[order], "b.", ms=2, alpha=0.5, label=r"Pred ± σ")
        ax.set_xlabel("Sample (sorted)"); ax.set_ylabel(lbl)
        ax.set_title(f"Sorted Predictions — {lbl}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.scatter(trues, means, s=3, alpha=0.3, color="steelblue")
        lo, hi = trues.min(), trues.max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        ax.set_xlabel(f"True {lbl}"); ax.set_ylabel(f"Predicted {lbl}")
        ax.set_title(f"Pred vs Truth — {lbl}"); ax.grid(True, alpha=0.3)

        ax = axes[2, col]
        ax.scatter(alea, epis, s=3, alpha=0.3, color="steelblue")
        ax.set_xlabel(f"Aleatoric σ ({ulbl})"); ax.set_ylabel(f"Epistemic σ ({ulbl})")
        ax.set_title(f"Uncertainty Decomposition — {lbl}"); ax.grid(True, alpha=0.3)

        ax = axes[3, col]
        ax.hist(alea,   bins=50, alpha=0.6, color="steelblue", label="Aleatoric")
        ax.hist(epis,   bins=50, alpha=0.6, color="orange",    label="Epistemic")
        ax.hist(totals_log, bins=50, alpha=0.6, color="green", label="Total")
        ax.set_xlabel(f"σ ({ulbl})"); ax.set_ylabel("Count")
        ax.set_title(f"Uncertainty Distribution — {lbl}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Uncertainty Analysis — net{idd} ({input_mode}) — {tag}", fontsize=14, y=1.01)
    plt.tight_layout()
    oname = f"{os.environ['HOME']}/plots/uncertainty_analysis_net{idd}_{input_mode}_{tag}.png"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight"); print(f"Saved: {oname}"); plt.close(fig)


def plot_resolution_robustness_dual(model, sample_x, true_ms, true_xi, tag=""):
    """
    2×2 resolution robustness plot.
    Top row: Ms prediction and σ vs resolution factor.
    Bottom row: xi prediction and σ vs resolution factor.
    sample_x : [1, n_input_channels, 128, 128]  — moment channels, no res channel yet.
    """
    import matplotlib.pyplot as plt, os
    factors   = [0.3, 0.5, 0.7, 0.9, 1.0]
    ms_p, ms_s, xi_p, xi_s = [], [], [], []

    model.eval()
    with torch.no_grad():
        for f in factors:
            x = sample_x.clone()
            if f < 1.0:
                x = _degrade_resolution(x, f)
            res_ch = torch.full((1, 1, x.shape[-2], x.shape[-1]),
                                f, dtype=x.dtype, device=x.device)
            x4 = torch.cat([x, res_ch], dim=1)
            (mu_ms, lv_ms), pred_xi = model(x4)
            m_xi, s_xi, _ = model.get_xi_stats(pred_xi)
            ms_val = float(np.exp(mu_ms.item()))
            ms_p.append(ms_val)
            ms_s.append(ms_val * float(torch.exp(0.5 * lv_ms).item()))  # σ in Ms units
            xi_p.append(float(m_xi.item()))
            xi_s.append(float(s_xi.item()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    rows = [
        (ms_p, ms_s, true_ms, r"$\mathcal{M}_s$"),
        (xi_p, xi_s, true_xi, r"$\xi$"),
    ]
    for row, (preds, sigmas, true_val, lbl) in enumerate(rows):
        ax = axes[row, 0]
        ax.errorbar(factors, preds, yerr=sigmas, fmt="bo-", lw=2, capsize=4,
                    label="Prediction ± σ")
        ax.axhline(true_val, color="r", ls="--", lw=1.5,
                   label=f"True: {true_val:.3f}")
        ax.set_xlabel("Resolution Factor"); ax.set_ylabel(f"Predicted {lbl}")
        ax.set_title(f"{lbl}: Prediction vs Resolution"); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        ax.plot(factors, sigmas, "go-", lw=2)
        ax.set_xlabel("Resolution Factor"); ax.set_ylabel(f"σ ({lbl})")
        ax.set_title(f"{lbl}: Uncertainty vs Resolution"); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Resolution Robustness — net{idd} ({input_mode}) — {tag}", fontsize=13)
    plt.tight_layout()
    oname = (f"{os.environ['HOME']}/plots/"
             f"resolution_awareness_net{idd}_{input_mode}"
             f"{('_' + tag) if tag else ''}.png")
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight"); print(f"Saved: {oname}"); plt.close(fig)