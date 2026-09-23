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
import loader
import h5py
idd  = 9009
what = ("Hybrid CNN-ViT | Resolution-Aware (net9008) + Mock-Obs dataset "
        "+ Temperature-Scaled Conformal + MC-Dropout epistemic uncertainty")

# ─── Data paths (Anvil scratch) ──────────────────────────────────────────────
# Update these to your actual Anvil paths for the RADMC3D mock-obs dataset.
# Expected HDF5 layout identical to simulation datasets:
#   data  : [N, C, H, W]  – C channels are moment maps (mom0, mom1, mom2)
#   Ms_act: [N]            – true sonic Mach number
dirpath    = "/home/x-nbisht1/projects/p79d_dataset/"
fname_train = "p79d_mockobs_13CO_S128_train.h5"
fname_valid = "p79d_mockobs_13CO_S128_valid.h5"
fname_test = "p79d_mockobs_13CO_S128_test.h5"

ntrain = 16783
nvalid = 1198
ntest  = 0 
device     = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Training hyperparameters ─────────────────────────────────────────────────
lr           = 5e-4
batch_size   = 64
epochs      = 100
lr_schedule = [50, 80]
weight_decay = 0.01

# ─── Architecture ─────────────────────────────────────────────────────────────
img_size        = 128
cnn_output_size = 64     # after the 2× maxpool in the CNN stem
patch_size      = 8
embed_dim       = 384
depth           = 6
num_heads       = 6
mlp_ratio       = 4.0
dropout         = 0.1
mc_dropout_p    = 0.10   # applied at inference for epistemic uncertainty

# ─── Resolution-aware training ────────────────────────────────────────────────
# During training a fraction of samples are randomly degraded:
#   1. downsample to res_factor × 128
#   2. upsample back to 128  (bilinear)
# The 4th input channel is filled with the scalar res_factor ∈ [res_min, 1].
# At inference on real observations set res_factor = beam_fwhm_pixels / 128.
res_train_prob = 0.50   # probability of applying degradation to a sample
res_min_factor = 0.40   # minimum resolution factor during augmentation

# ─── MC uncertainty ───────────────────────────────────────────────────────────
n_mc_samples = 30        # forward passes for epistemic uncertainty estimate


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Load mock observation data from pre-split train/valid HDF5 files.
    Splits valid file into validation (for calibration) and test sets.
    """
    print("Reading data …")
    
    # Load training file
    print(f"Loading {fname_train}...")
    with h5py.File(dirpath + fname_train, 'r') as f:
        train_data = torch.from_numpy(f['subsets'][:]).float()
        train_ms = f['Ms_act'][:]
        train_ma = f['Ma_act'][:]
    
    # Load validation file (will split into val + test)
    print(f"Loading {fname_valid}...")
    with h5py.File(dirpath + fname_valid, 'r') as f:
        valid_data = torch.from_numpy(f['subsets'][:]).float()
        valid_ms = f['Ms_act'][:]
        valid_ma = f['Ma_act'][:]
    
    n_train = min(ntrain, len(train_data))
    n_valid = min(nvalid, len(valid_data))
    n_test = 0
    
    print(f"\nDataset sizes:")
    print(f"  Train: {n_train} (from {len(train_data)} available)")
    print(f"  Valid: {n_valid} (for calibration)")
    print(f"  Test:  {n_test}")
    
    # Stratified validation split from valid file
    mach_bins = [0, 4, 6, 8, 10, 15]
    samples_per_bin = n_valid // (len(mach_bins) - 1)
    valid_indices = []
    
    for i in range(len(mach_bins) - 1):
        mask = (valid_ms >= mach_bins[i]) & (valid_ms < mach_bins[i + 1])
        bin_idx = np.where(mask)[0]
        if len(bin_idx) >= samples_per_bin:
            sel = np.random.choice(bin_idx, samples_per_bin, replace=False)
            valid_indices.extend(sel.tolist())
        else:
            valid_indices.extend(bin_idx.tolist())
            print(f"  Warning: only {len(bin_idx)} samples in "
                  f"Ms [{mach_bins[i]}, {mach_bins[i+1]})")
    
    valid_indices = np.array(valid_indices)
    all_valid_indices = np.arange(len(valid_data))
    test_mask = ~np.isin(all_valid_indices, valid_indices)
    test_indices = all_valid_indices[test_mask][:n_test]
    
    all_data = {
        "train": train_data[:n_train],
        "valid": valid_data[valid_indices],
        "test":  torch.zeros(0),
        "quantities": {
            "train": {
                "Ms_act": train_ms[:n_train],
                "Ma_act": train_ma[:n_train],
            },
            "valid": {
                "Ms_act": valid_ms[valid_indices],
                "Ma_act": valid_ma[valid_indices],
            },
            "test": {
                "Ms_act": np.array([]),
                "Ma_act": np.array([]),
            },
        },
    }
    
    print(f"\nFinal splits:")
    print(f"  Train: {len(all_data['train'])}")
    print(f"  Valid (stratified): {len(all_data['valid'])}")
    print(f"  Test:  {len(all_data['test'])}")
    
    print("\nValidation Mach distribution:")
    ms_v = valid_ms[valid_indices]
    for i in range(len(mach_bins) - 1):
        m = (ms_v >= mach_bins[i]) & (ms_v < mach_bins[i + 1])
        print(f"  Ms [{mach_bins[i]:2d}–{mach_bins[i+1]:2d}): {m.sum():3d}")
    
    print("Done loading data.")
    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def _degrade_resolution(x: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Downsample × factor then upsample back to original size.
    x: [C, H, W]  or  [1, C, H, W]
    """
    squeeze = x.ndim == 3
    if squeeze:
        x = x.unsqueeze(0)
    H, W = x.shape[-2], x.shape[-1]
    small_h = max(4, int(H * factor))
    small_w = max(4, int(W * factor))
    x = F.interpolate(x, size=(small_h, small_w), mode="bilinear", align_corners=False)
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    if squeeze:
        x = x.squeeze(0)
    return x


class MockObsDataset(Dataset):
    """
    Returns a 4-channel tensor:
        channels 0–2 : moment maps (mom0, mom1, mom2)
        channel  3   : resolution indicator (scalar, constant per sample)
    """
    def __init__(self, all_data, quan, augment: bool = False):
        self.data    = all_data
        self.quan    = quan
        self.augment = augment

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx][0:3].clone()   # [3, H, W]

        res_factor = 1.0

        if self.augment:
            # Spatial roll augmentation
            H, W = x.shape[-2], x.shape[-1]
            dy = torch.randint(0, H, (1,)).item()
            dx = torch.randint(0, W, (1,)).item()
            x = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))

            # Random flips
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[-1])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[-2])

            # Resolution degradation augmentation
            if torch.rand(1).item() < res_train_prob:
                res_factor = (
                    res_min_factor
                    + (1.0 - res_min_factor) * torch.rand(1).item()
                )
                x = _degrade_resolution(x, res_factor)

        # Append resolution channel  [4, H, W]
        res_ch = torch.full((1, x.shape[-2], x.shape[-1]),
                            fill_value=res_factor, dtype=x.dtype)
        x = torch.cat([x, res_ch], dim=0)

        ms = self.quan["Ms_act"][idx]
        return (
            x.to(device), torch.tensor([np.log(ms + 1e-6)], dtype=torch.float32).to(device)
        )


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class CNNStem(nn.Module):
    """
    Lightweight CNN front-end:  [B, 4, 128, 128]  →  [B, 64, 64, 64]
    4 input channels: mom0, mom1, mom2, res_indicator
    """
    def __init__(self, in_chans: int = 4, out_chans: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_chans, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(out_chans)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)          # 128 → 64
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 64, patch_size: int = 8,
                 in_chans: int = 64, embed_dim: int = 384):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                    # [B, D, n, n]
        return x.flatten(2).transpose(1, 2)  # [B, N, D]


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8,
                 qkv_bias: bool = False, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int,
                 dropout: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with optional MC dropout."""
    def __init__(self, dim: int, num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 mc_drop: float = 0.0):
        super().__init__()
        self.norm1   = nn.LayerNorm(dim)
        self.attn    = Attention(dim, num_heads=num_heads,
                                 attn_drop=dropout, proj_drop=dropout)
        self.norm2   = nn.LayerNorm(dim)
        self.mlp     = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.mc_drop = nn.Dropout(mc_drop)   # active at eval time via mc_forward

    def forward(self, x):
        x = x + self.mc_drop(self.attn(self.norm1(x)))
        x = x + self.mc_drop(self.mlp(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class HybridCNNViT(nn.Module):
    """
    Net9009: Resolution-Aware Hybrid CNN-ViT with:
      • 4-channel input (3 moment maps + resolution indicator)
      • Gaussian NLL head → (mean, log_var)
      • MC-Dropout blocks for epistemic uncertainty
      • Learnable temperature scalar for conformal calibration
    """

    def __init__(
        self,
        img_size: int        = 128,
        cnn_output_size: int = 64,
        patch_size: int      = 8,
        in_chans: int        = 4,    # 3 moment maps + 1 resolution channel
        num_classes: int     = 2,    # mean + log_variance
        embed_dim: int       = 384,
        depth: int           = 6,
        num_heads: int       = 6,
        mlp_ratio: float     = 4.0,
        dropout: float       = 0.1,
        mc_drop: float       = 0.10,
    ):
        super().__init__()

        # CNN front-end
        self.cnn_stem    = CNNStem(in_chans=in_chans, out_chans=64)

        # Patch embedding from CNN features
        self.patch_embed = PatchEmbed(
            img_size   = cnn_output_size,
            patch_size = patch_size,
            in_chans   = 64,
            embed_dim  = embed_dim,
        )
        num_patches = self.patch_embed.n_patches

        # ViT positional encoding & class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(p=dropout)

        # Transformer blocks (with MC dropout inside)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio,
                             dropout=dropout, mc_drop=mc_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Output head: mean + log_variance
        self.head = nn.Linear(embed_dim, num_classes)

        # Learnable temperature for uncertainty calibration (init = 1 → no scaling)
        self.log_temperature = nn.Parameter(torch.zeros(1))

        # Training curves stored in model for portability
        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve",   torch.zeros(epochs))

        # Weight initialisation
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    # ── weight init ──────────────────────────────────────────────────────────

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x):
        """
        x: [B, 4, 128, 128]
        Returns: (mean [B,1], log_var [B,1])
                 log_var is temperature-scaled so calibration is learnable.
        """
        B = x.shape[0]

        # CNN local feature extraction
        x = self.cnn_stem(x)          # [B, 64, 64, 64]

        # Patch tokenisation
        x = self.patch_embed(x)       # [B, N, D]

        # Prepend [CLS] token + positional encoding
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embed
        x   = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x[:, 0]                   # [B, D]  – class token

        out    = self.head(x)         # [B, 2]
        mean   = out[:, 0:1]          # [B, 1]

        # Temperature scaling: logvar_calibrated = logvar + 2 * log_temperature
        #   σ_cal² = σ² * exp(2*logT) = σ² * T²
        # Larger T → wider uncertainty bands → improves under-coverage.
        logvar = (out[:, 1:2]
                  + 2.0 * self.log_temperature).clamp(-10, 5)  # [B, 1]

        return mean, logvar

    # ── MC inference ──────────────────────────────────────────────────────────

    def mc_forward(self, x, n_samples: int = n_mc_samples):
        """
        Run n_samples stochastic forward passes with MC dropout active.
        Returns:
            mean_pred : [B, 1]   – mean of MC means
            aleatoric : [B, 1]   – average predicted σ (intrinsic noise)
            epistemic : [B, 1]   – std of MC means   (model uncertainty)
            sigma_total: [B, 1]  – combined uncertainty √(alea² + epis²)
        """
        #Enable only Dropout layers
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        
        with torch.no_grad():
            mc_means, mc_sigmas = [], []
            for _ in range(n_samples):
                mu, lv = self(x)
                mc_means.append(mu)
                mc_sigmas.append(torch.exp(0.5 * lv))
            mc_means  = torch.stack(mc_means,  dim=0)
            mc_sigmas = torch.stack(mc_sigmas, dim=0)
        
        self.eval()   # restore fully

        mean_pred   = mc_means.mean(dim=0)
        epistemic   = mc_means.std(dim=0)
        aleatoric   = mc_sigmas.mean(dim=0)
        sigma_total = torch.sqrt(aleatoric**2 + epistemic**2)
        return mean_pred, aleatoric, epistemic, sigma_total

    # ── loss ──────────────────────────────────────────────────────────────────

    def criterion(self, pred, target):
        """
        Gaussian NLL with mild high-Mach weighting.
        NLL = 0.5 * (logvar + (y - μ)² / σ²)
        """
        mean, logvar = pred

        nll = 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))

        # Slightly upweight high-Mach samples (Ms > 10)
        weights = torch.where(target > 2.30, 2.5, 1.0)   # Ms > 10 in log space
        weighted_mse = (weights * (mean - target) ** 2).mean()

        return 0.9 * nll.mean() + 0.1 * weighted_mse

    def criterion1(self, pred, target):
        """Diagnostic breakdown."""
        mean, logvar = pred
        return {
            "nll": 0.5 * (logvar + (target - mean) ** 2
                          * torch.exp(-logvar)).mean(),
            "mse": F.mse_loss(mean, target),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def thisnet():
    model = HybridCNNViT(
        img_size        = img_size,
        cnn_output_size = cnn_output_size,
        patch_size      = patch_size,
        in_chans        = 4,         # 3 moments + resolution indicator
        num_classes     = 2,
        embed_dim       = embed_dim,
        depth           = depth,
        num_heads       = num_heads,
        mlp_ratio       = mlp_ratio,
        dropout         = dropout,
        mc_drop         = mc_dropout_p,
    ).to(device)
    return model


def train(model, all_data):
    trainer(model, all_data,
            epochs=epochs, lr=lr, batch_size=batch_size,
            weight_decay=weight_decay, lr_schedule=lr_schedule)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 8675309):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trainer(model, all_data, epochs=50, batch_size=64, lr=5e-4,
            weight_decay=0.01, lr_schedule=(1000,)):
    set_seed()

    ds_train = MockObsDataset(all_data["train"],
                               all_data["quantities"]["train"], augment=True)
    ds_val   = MockObsDataset(all_data["valid"],
                               all_data["quantities"]["valid"], augment=False)

    train_loader = DataLoader(ds_train, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_val, batch_size=max(64, batch_size),
                              shuffle=False, drop_last=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(lr_schedule), gamma=0.1)

    total_steps = epochs * max(1, len(train_loader))
    print(f"Total Steps {total_steps}  |  ntrain {len(ds_train)}  |  "
          f"nvalid {len(ds_val)}  |  epochs {epochs}")
    print(f"Architecture: Hybrid CNN-ViT | 4-ch input | "
          f"MC-dropout={mc_dropout_p} | temperature scaling")

    best_val   = float("inf")
    best_state = None
    patience   = int(1e6)
    bad_epochs = 0
    t0         = time.time()

    for epoch in range(1, epochs + 1):
        # ── train step ──
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss  = model.criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        scheduler.step()
        train_loss = running / len(ds_train)
        model.train_curve[epoch - 1] = train_loss

        # ── val step ──
        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            for xb, yb in val_loader:
                preds = model(xb)
                vloss = model.criterion(preds, yb)
                vtotal += vloss.item() * xb.size(0)
        val_loss = vtotal / len(ds_val)
        model.val_curve[epoch - 1] = val_loss

        # ── early stopping bookkeeping ──
        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        # ── progress line ──
        now            = time.time()
        time_per_epoch = (now - t0) / epoch
        secs_left      = time_per_epoch * (epochs - epoch)
        etad           = datetime.datetime.fromtimestamp(now + secs_left)
        nowdate        = datetime.datetime.now()
        lr_cur         = optimizer.param_groups[0]["lr"]

        def fmt(s):
            m, sec = divmod(int(s), 60)
            h, m   = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"

        print(f"[{epoch:3d}/{epochs}] net{idd}  "
              f"train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {lr_cur:.2e} | bad {bad_epochs:02d} | "
              f"T {model.log_temperature.item():.3f} | "
              f"ETA {etad.strftime('%H:%M:%S')} | "
              f"Remain {fmt(secs_left)} | Sofar {fmt(now - t0)}")

        if nowdate.day != etad.day:
            print("  (running past midnight)")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}.")
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
    Compute nonconformity scores on a calibration set.
    score_i = |y_i - μ_i| / σ_i   (normalised absolute residual)
    Resolution-aware: σ_i is the model's predicted σ × temperature.
    Returns: scores [N], predictions [N], targets [N]
    """
    model.eval()
    scores, preds, targets = [], [], []

    with torch.no_grad():
        for xb, yb in cal_loader:
            mean, logvar = model(xb)
            sigma = torch.exp(0.5 * logvar)
            s = (torch.abs(yb - mean) / (sigma + 1e-8)).squeeze().cpu()
            scores.append(s)
            preds.append(mean.squeeze().cpu())
            targets.append(yb.squeeze().cpu())

    return (torch.cat(scores),
            torch.cat(preds),
            torch.cat(targets))


def get_conformal_quantile(scores: torch.Tensor,
                           coverage: float) -> float:
    """
    Return the (1 – α) quantile of the nonconformity scores,
    with finite-sample correction: q = ⌈(n+1)(1-α)⌉ / n.
    """
    n   = len(scores)
    lvl = math.ceil((n + 1) * coverage) / n
    lvl = min(lvl, 1.0)
    return float(torch.quantile(scores, lvl).item())


def build_conformal_intervals(model, data_loader,
                              q90: float, q95: float,
                              res_factor: float = 1.0):
    """
    Given pre-computed quantiles, produce conformal prediction intervals.
    Returns dict with arrays: mean, lower90, upper90, lower95, upper95,
                               true, sigma_aleatoric.
    res_factor: for display in titles only.
    """
    model.eval()
    means, lower90s, upper90s = [], [], []
    lower95s, upper95s, trues = [], [], []
    sigmas = []

    with torch.no_grad():
        for xb, yb in data_loader:
            mu, lv = model(xb)
            sigma  = torch.exp(0.5 * lv).squeeze().cpu()
            mu     = mu.squeeze().cpu()
            means.append(mu)
            sigmas.append(sigma)
            lower90s.append(mu - q90 * sigma)
            upper90s.append(mu + q90 * sigma)
            lower95s.append(mu - q95 * sigma)
            upper95s.append(mu + q95 * sigma)
            trues.append(yb.squeeze().cpu())

    return {
        "mean":    torch.cat(means).numpy(),
        "lower90": torch.cat(lower90s).numpy(),
        "upper90": torch.cat(upper90s).numpy(),
        "lower95": torch.cat(lower95s).numpy(),
        "upper95": torch.cat(upper95s).numpy(),
        "true":    torch.cat(trues).numpy(),
        "sigma":   torch.cat(sigmas).numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(model):
    import matplotlib.pyplot as plt
    import os

    n     = int((model.train_curve != 0).sum().item())
    train = model.train_curve[:n].cpu().numpy()
    val   = model.val_curve[:n].cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train, lw=2, label="Train Loss")
    ax.plot(val,   lw=2, label="Val Loss")
    ax.set_yscale("symlog")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("NLL Loss", fontsize=12)
    ax.set_title(f"Training Curves – net{idd}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    oname = f"{os.environ['HOME']}/plots/loss_curve_net{idd}.png"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight")
    print(f"Saved: {oname}")
    plt.close(fig)


def plot_predictions(model, test_loader, tag: str = "test"):
    """Scatter + sorted-error plot for test/obs data."""
    import matplotlib.pyplot as plt
    import os

    model.eval()
    means, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            mu, _ = model(xb)
            means.append(mu.squeeze().cpu())
            trues.append(yb.squeeze().cpu())

    means = torch.cat(means).numpy()
    trues = torch.cat(trues).numpy()
    order = np.argsort(trues)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(trues, means, s=4, alpha=0.4, color="steelblue")
    lo = min(trues.min(), means.min())
    hi = max(trues.max(), means.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
    ax.set_xlabel("True Mach Number", fontsize=11)
    ax.set_ylabel("Predicted Mach Number", fontsize=11)
    ax.set_title(f"Pred vs Truth – net{idd} ({tag})", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(trues[order], "r-", lw=1.5, label="True")
    ax.plot(means[order], "b-", lw=1,   alpha=0.7, label="Predicted")
    ax.set_xlabel(f"Sample (sorted by Ms)", fontsize=11)
    ax.set_ylabel("Mach Number", fontsize=11)
    ax.set_title(f"Sorted Predictions – net{idd}", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    oname = f"{os.environ['HOME']}/plots/predictions_net{idd}_{tag}.png"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight")
    print(f"Saved: {oname}")
    plt.close(fig)


def plot_conformal_intervals(results: dict, tag: str = "test"):
    """Mirror of net9008's conformal interval plot."""
    import matplotlib.pyplot as plt
    import os

    order  = np.argsort(results["true"])
    x      = np.arange(len(order))
    ms     = results["true"][order]
    mu     = results["mean"][order]
    lo90   = results["lower90"][order]
    hi90   = results["upper90"][order]
    lo95   = results["lower95"][order]
    hi95   = results["upper95"][order]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(x, lo95, hi95, alpha=0.30, color="royalblue",
                    label="95% Conformal Interval")
    ax.fill_between(x, lo90, hi90, alpha=0.45, color="royalblue",
                    label="90% Conformal Interval")
    ax.plot(x, mu, "b-", lw=1.5,  label="Mean Prediction")
    ax.scatter(x, ms, s=12, c="red", zorder=3, label="True Value")
    ax.set_xlabel("Sample (sorted by true Ms)", fontsize=11)
    ax.set_ylabel("Mach Number", fontsize=11)
    ax.set_title(f"Resolution-Aware Conformal Intervals – net{idd}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    oname = f"{os.environ['HOME']}/plots/conformal_intervals_net{idd}_{tag}.png"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight")
    print(f"Saved: {oname}")
    plt.close(fig)


def plot_coverage_calibration(model, cal_loader, tag: str = "test"):
    """
    Reproduce the coverage calibration check from net9008 output plots.
    Target: empirical coverage ≈ nominal (diagonal line).
    """
    import matplotlib.pyplot as plt
    import os

    scores, _, _ = compute_conformal_scores(model, cal_loader)
    nominal_levels = np.arange(0.50, 1.00, 0.05)
    empirical      = []

    for alpha in nominal_levels:
        q          = get_conformal_quantile(scores, alpha)
        # fraction of calibration points whose score ≤ q
        cov = (scores <= q).float().mean().item()
        empirical.append(cov)

    empirical = np.array(empirical)
    mae       = float(np.mean(np.abs(empirical - nominal_levels)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(nominal_levels, empirical, "bo-", lw=2,
            label="Resolution-Aware Conformal Prediction")
    ax.plot([0.25, 1.0], [0.25, 1.0], "r--", lw=1.5, label="Perfect Calibration")
    ax.annotate(f"Mean Absolute Error: {mae:.3f}",
                xy=(0.55, 0.43), fontsize=10,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
    ax.set_xlabel("Nominal Coverage", fontsize=12)
    ax.set_ylabel("Empirical Coverage", fontsize=12)
    ax.set_title(f"Coverage Calibration Check (net{idd})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    oname = f"{os.environ['HOME']}/plots/coverage_calibration_net{idd}_{tag}.png"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight")
    print(f"Saved: {oname}  (MAE={mae:.3f})")
    plt.close(fig)
    return mae


def plot_uncertainty_analysis(model, test_loader, tag: str = "test"):
    """
    Four-panel uncertainty analysis plot mirroring net9008 output:
      top-left  : Predictions with uncertainty (sorted)
      top-right : Pred vs Truth scatter
      bot-left  : Epistemic vs Aleatoric scatter
      bot-right : Uncertainty distribution histogram
    """
    import matplotlib.pyplot as plt
    import os

    model.eval()
    means, alea, epis, totals, trues = [], [], [], [], []

    for xb, yb in test_loader:
        mu, al, ep, tot = model.mc_forward(xb, n_samples=n_mc_samples)
        means.append(mu.squeeze().cpu())
        alea.append(al.squeeze().cpu())
        epis.append(ep.squeeze().cpu())
        totals.append(tot.squeeze().cpu())
        trues.append(yb.squeeze().cpu())

    means  = torch.cat(means).numpy()
    alea   = torch.cat(alea).numpy()
    epis   = torch.cat(epis).numpy()
    totals = torch.cat(totals).numpy()
    trues  = torch.cat(trues).numpy()
    order  = np.argsort(trues)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.errorbar(np.arange(len(order)), trues[order],
                fmt="r-", lw=1.5, label="True Value", zorder=3)
    ax.errorbar(np.arange(len(order)), means[order],
                yerr=totals[order], fmt="none",
                ecolor="steelblue", alpha=0.3, elinewidth=0.5)
    ax.plot(means[order], "b.", ms=2, alpha=0.5, label=r"Pred ± σ_total")
    ax.set_xlabel("Sample (sorted by Ms)")
    ax.set_ylabel("Mach Number")
    ax.set_title("Predictions with Uncertainty (Resolution-Aware)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(trues, means, s=3, alpha=0.3, color="steelblue")
    lo, hi = trues.min(), trues.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("True Mach Number")
    ax.set_ylabel("Predicted Mach Number")
    ax.set_title("Prediction vs Truth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(alea, epis, s=3, alpha=0.3, color="steelblue")
    ax.set_xlabel(r"Aleatoric Uncertainty (data noise)")
    ax.set_ylabel(r"Epistemic Uncertainty (model)")
    ax.set_title("Uncertainty Decomposition")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(alea,   bins=50, alpha=0.6, color="steelblue", label="Aleatoric")
    ax.hist(epis,   bins=50, alpha=0.6, color="orange",    label="Epistemic")
    ax.hist(totals, bins=50, alpha=0.6, color="green",     label="Total")
    ax.set_xlabel(r"Uncertainty (σ)")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Uncertainty Analysis – net{idd} ({tag})", fontsize=14, y=1.01)
    plt.tight_layout()

    oname = f"{os.environ['HOME']}/plots/uncertainty_analysis_net{idd}_{tag}.png"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight")
    print(f"Saved: {oname}")
    plt.close(fig)


def plot_resolution_robustness(model, sample_x, true_ms, tag: str = ""):
    """
    Test a single sample (or small batch) at different resolution factors.
    Mirrors net9008's resolution_awareness plot.
    sample_x : [1, 3, 128, 128]  – 3-channel moment map (no res channel yet)
    true_ms  : float
    """
    import matplotlib.pyplot as plt
    import os

    factors   = [0.3, 0.5, 0.7, 0.9, 1.0]
    preds_out = []
    sigma_out = []

    model.eval()
    with torch.no_grad():
        for f in factors:
            x = sample_x.clone()
            if f < 1.0:
                x = _degrade_resolution(x, f)
            res_ch = torch.full((1, 1, 128, 128), f, dtype=x.dtype, device=x.device)
            x4 = torch.cat([x, res_ch], dim=1)
            mu, lv = model(x4)
            preds_out.append(mu.item())
            sigma_out.append(torch.exp(0.5 * lv).item())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(factors, preds_out, "bo-", lw=2, label="Model Prediction")
    ax.axhline(true_ms, color="r", ls="--", lw=1.5,
               label=f"True Value: {true_ms:.2f}")
    ax.set_xlabel("Resolution Indicator (native_size / target_size)")
    ax.set_ylabel("Predicted Mach Number")
    ax.set_title("Predictions Change with Resolution Indicator")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(factors, sigma_out, "go-", lw=2, label="Model Uncertainty")
    ax.set_xlabel("Resolution Indicator (native_size / target_size)")
    ax.set_ylabel(r"Predicted Uncertainty (σ)")
    ax.set_title("Uncertainty Increases with Lower Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    oname = (f"{os.environ['HOME']}/plots/"
             f"resolution_awareness_net{idd}{('_' + tag) if tag else ''}.png")
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    fig.savefig(oname, dpi=150, bbox_inches="tight")
    print(f"Saved: {oname}")
    plt.close(fig)