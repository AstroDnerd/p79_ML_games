"""
net9010.py
==========
Hybrid CNN-ViT — dual-output version predicting:
  • Sonic Mach number          (Ms)  — Gaussian in ln(Ms)
  • Compressive KE fraction    (chi) — Gaussian in logit(chi) (or linear chi)

chi is the Helmholtz compressive fraction of the snapshot's velocity field
(labels/chi_v) or of sqrt(rho) v (labels/chi_w); see extract_sim_data.py.

Input modes (set n_input_channels at the top):
  1 → mom0 only  (density channel + res indicator)  → saved as *_mom0.*
  3 → all moments (mom0/mom1/mom2 + res indicator)  → saved as *_allmom.*

Data: the extract_sim_data.py dataset (one HDF5 file). Use load_data_grouped
for evaluation — it holds out whole simulations. load_data (random split over
images) is kept only to reproduce old numbers and leaks between train and test.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
import random
import time
import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py

idd  = 9010

# ─── Input mode ──────────────────────────────────────────────────────────────
n_input_channels = 3                              # 1 or 3
input_mode       = "allmom" if n_input_channels == 3 else "mom0"
n_model_channels = n_input_channels + 1           # +1 for resolution channel

what = (f"Hybrid CNN-ViT | dual-output Ms+chi | {input_mode} | "
        f"resolution-aware | temperature-scaled conformal | MC-dropout")

# ─── Data ────────────────────────────────────────────────────────────────────
data_file = os.environ.get("P79D_DATA_FILE",          # override for tests / other datasets
                           "/home/x-nbisht1/scratch/projects/p79d_dataset/p79d_mach_grid_256_v1.h5")
ms_max    = 20.0        # drop images with Ms above this (sparse, under-resolved)

# Compressibility target: "chi_v" (velocity) or "chi_w" (sqrt(rho) v, kinetic-
# energy weighted). chi_space: "logit" models logit(chi) with a Gaussian, which
# keeps predictions in (0, 1) and resolves the many small values; "linear"
# models chi itself.
chi_field = "chi_v"
chi_space = "logit"
chi_eps   = 1e-3          # clip chi to [eps, 1 - eps] before the logit

# Input transform (see preprocess_maps). "physical": ln(Sigma/<Sigma>),
# centroid minus its image mean, ln(sigma). "raw": the stored maps unchanged.
input_transform = "physical"

# ─── Random-split fractions (legacy load_data only) ──────────────────────────
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

def _read_dataset():
    """Images [N, 3, H, W] (float tensor) and a dict of per-sample metadata."""
    print(f"Reading {data_file} …")
    with h5py.File(data_file, "r") as f:
        data = torch.from_numpy(f["images"][:]).float()
        meta = {
            "Ms_act":  f["labels/Ms"][:],
            "chi_act": f[f"labels/{chi_field}"][:],
            "xi_forcing": f["labels/xi"][:],         # forcing parameter, used to stratify splits
            "sim_id":  f["samples/sim_id"][:].astype(int),
            "frame":   f["samples/frame"][:],
            "los":     f["samples/los"][:],
            "time":    f["labels/time"][:],
        }
        sim_names   = f["sims/name"][:].astype(str)
        sim_xi      = f["sims/xi"][:]
        sim_ms_mean = f["sims/Ms_mean"][:]
    keep = meta["Ms_act"] <= ms_max
    data = data[keep]
    meta = {k: v[keep] for k, v in meta.items()}
    meta["Ms_mean"] = sim_ms_mean[meta["sim_id"]]
    print(f"  {len(data)} images with Ms <= {ms_max}, chi target = {chi_field} ({chi_space})")
    return data, meta, {"name": sim_names, "xi": sim_xi, "Ms_mean": sim_ms_mean}


def _pack(data, meta, idx):
    idx = np.asarray(idx)
    return {"data": data[idx], **{k: v[idx] for k, v in meta.items()}}


def load_data(seed: int = 42):
    """
    LEGACY random split over images, stratified by (Ms, xi) bins.
    Frames, LOS and augmentations of one simulation land on both sides, so the
    test score is optimistic. Use load_data_grouped for anything reported.
    """
    data, meta, _ = _read_dataset()
    ms_bins = [0, 2, 4, 6, 8, 10, 15, ms_max + 0.01]
    xi_bins = [-0.01, 0.125, 0.375, 0.625, 0.875, 1.01]
    group_ids = (np.digitize(meta["Ms_act"], ms_bins) - 1) * len(xi_bins) + \
                (np.digitize(meta["xi_forcing"], xi_bins) - 1)

    rng = np.random.default_rng(seed)
    idx = {"train": [], "test": [], "valid": []}
    for gid in np.unique(group_ids):
        g = rng.permutation(np.where(group_ids == gid)[0])
        n_tr, n_te = int(frac_train * len(g)), int(frac_test * len(g))
        idx["train"].extend(g[:n_tr]); idx["test"].extend(g[n_tr:n_tr + n_te])
        idx["valid"].extend(g[n_tr + n_te:])
    splits = {tag: _pack(data, meta, rng.permutation(np.array(v, dtype=int))) for tag, v in idx.items()}
    for tag in ("train", "valid", "test"):
        print(f"  {tag:5s}: {len(splits[tag]['data']):6d}")
    return splits


def assign_sim_folds(sim_ms, sim_xi, n_folds: int = 5):
    """
    Deterministic fold per simulation: within each forcing class (xi), sims
    are sorted by Ms_mean and dealt round-robin, with the start offset carried
    across classes so every fold gets ~1 sim per forcing class spread over the
    Ms range.
    """
    fold   = np.zeros(len(sim_ms), dtype=int)
    offset = 0
    for x in np.unique(sim_xi):
        idx = np.where(sim_xi == x)[0]
        idx = idx[np.argsort(sim_ms[idx])]
        fold[idx] = (np.arange(len(idx)) + offset) % n_folds
        offset   += len(idx)
    return fold


def load_data_grouped(test_fold, n_folds: int = 5, calib: bool = True,
                      val_frac: float = 0.15, gap_frames: int = 5, calib_fold: int = 0):
    """
    Leakage-free split: every frame, LOS and augmentation of a simulation goes
    to the same side of each boundary.

      test  : all sims assigned to `test_fold` (see assign_sim_folds)
      calib : if calib, all sims of fold (test_fold + 1) % n_folds — the
              conformal calibration set, exchangeable with test because its
              sims are also unseen in training
      valid : the last `val_frac` of each remaining sim's frames, separated
              from train by `gap_frames`; used only for early stopping
      train : everything else

    test_fold=None ("all" mode, for the final model applied to observations):
    no test set; calib is fold `calib_fold` and every other sim is trained on.
    """
    data, meta, sims = _read_dataset()
    present = np.unique(meta["sim_id"])
    sim_fold = np.full(len(sims["name"]), -1)
    sim_fold[present] = assign_sim_folds(sims["Ms_mean"][present], sims["xi"][present], n_folds)
    sid = meta["sim_id"]

    if test_fold is None:
        test_fold = -1                        # matches no simulation
    else:
        calib_fold = (test_fold + 1) % n_folds
    is_test  = sim_fold[sid] == test_fold
    is_calib = (sim_fold[sid] == calib_fold) if calib else np.zeros(len(sid), bool)
    is_val   = np.zeros(len(sid), bool)
    is_gap   = np.zeros(len(sid), bool)
    for g in present[(sim_fold[present] != test_fold) &
                     ~((sim_fold[present] == calib_fold) & calib)]:
        m      = sid == g
        frames = np.unique(meta["frame"][m])
        cut    = frames[-max(1, int(round(val_frac * len(frames))))]
        is_val |= m & (meta["frame"] >= cut)
        is_gap |= m & (meta["frame"] < cut) & (meta["frame"] >= cut - gap_frames)
    is_train = ~is_test & ~is_calib & ~is_val & ~is_gap

    print(f"\nFold {'all' if test_fold < 0 else test_fold}/{n_folds} | {len(present)} sims | train {is_train.sum()}  "
          f"valid {is_val.sum()}  calib {is_calib.sum()}  test {is_test.sum()}  "
          f"(gap dropped {is_gap.sum()})")
    if calib:
        print("  Calib sims: " + ", ".join(f"{sims['name'][g]} (Ms={sims['Ms_mean'][g]:.2f})"
                                          for g in present[sim_fold[present] == calib_fold]))
    print("  Test sims: " + ", ".join(f"{sims['name'][g]} (Ms={sims['Ms_mean'][g]:.2f})"
                                     for g in present[sim_fold[present] == test_fold]))
    parts = [("train", is_train), ("valid", is_val), ("test", is_test)]
    if calib:
        parts.append(("calib", is_calib))
    splits = {tag: _pack(data, meta, np.where(m)[0]) for tag, m in parts}
    splits["sim_keys"] = np.stack([sims["Ms_mean"], sims["xi"]], axis=1)
    splits["sim_names"] = sims["name"]
    splits["sim_fold"] = sim_fold
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


def chi_to_target(chi):
    """chi (array or tensor) -> the space the network models."""
    if chi_space == "linear":
        return chi
    lib = torch if torch.is_tensor(chi) else np
    c = lib.clip(chi, chi_eps, 1 - chi_eps)
    return lib.log(c / (1 - c))


def chi_from_target(t, sig_t=None):
    """
    Network-space mean (and sigma) -> linear chi (and its sigma by the delta
    method, dchi/dt = chi (1 - chi) for the logit). Works on tensors or arrays.
    """
    if chi_space == "linear":
        return t if sig_t is None else (t, sig_t)
    lib = torch if torch.is_tensor(t) else np
    chi = 1 / (1 + lib.exp(-t))
    return chi if sig_t is None else (chi, chi * (1 - chi) * sig_t)


def preprocess_maps(x: torch.Tensor, mode: str = None) -> torch.Tensor:
    """
    Map-level input transform, shared with the observation pipeline so real
    maps are fed exactly like training maps. x: [..., C, H, W] with channels
    (column density, velocity centroid, velocity dispersion) in any subset
    starting from channel 0; velocities in units of the sound speed.

    "physical": ch0 -> ln(Sigma / <Sigma>_image)   (removes abundance / distance scaling)
                ch1 -> v_c - <v_c>_image          (removes the systemic velocity)
                ch2 -> ln(max(sigma, 1e-3))
    "raw"     : unchanged
    """
    mode = mode or input_transform
    if mode == "raw":
        return x
    x = x.clone()
    x[..., 0, :, :] = torch.log(x[..., 0, :, :].clamp_min(1e-12) /
                                x[..., 0, :, :].mean(dim=(-2, -1), keepdim=True))
    if x.shape[-3] > 1:
        x[..., 1, :, :] = x[..., 1, :, :] - x[..., 1, :, :].mean(dim=(-2, -1), keepdim=True)
    if x.shape[-3] > 2:
        x[..., 2, :, :] = torch.log(x[..., 2, :, :].clamp_min(1e-3))
    return x


class MachChiDataset(Dataset):
    """
    Returns:
        x       : [n_model_channels, 128, 128]  — input channels + res indicator
        targets : [2]                            — [ln(Ms), chi_to_target(chi)]

    n_input_channels controls whether x uses mom0 only (1) or all moments (3).
    Augmentation and resolution degradation act on the physical maps; the
    input transform (preprocess_maps) is applied last. The resolution
    indicator is always appended as the last channel.
    """
    def __init__(self, split_dict, augment: bool = False):
        self.data    = split_dict["data"]     # [N, 3, 128, 128]
        self.ms_act  = split_dict["Ms_act"]
        self.chi_t   = chi_to_target(np.asarray(split_dict["chi_act"], dtype=np.float64))
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

        x = preprocess_maps(x)
        res_ch = torch.full((1, x.shape[-2], x.shape[-1]),
                            fill_value=res_factor, dtype=x.dtype)
        x = torch.cat([x, res_ch], dim=0)   # [n_model_channels, H, W]

        ms  = float(self.ms_act[idx])
        tgt = torch.tensor([np.log(ms + 1e-6), float(self.chi_t[idx])], dtype=torch.float32)

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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class HybridCNNViT(nn.Module):
    """
    Net9010: CNN stem + ViT encoder with two Gaussian heads, ln(Ms) and the
    chi target (chi_to_target(chi)), each with its own learnable temperature.
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
        self.head_ms  = nn.Linear(embed_dim, 2)
        self.head_chi = nn.Linear(embed_dim, 2)

        # Temperature scalars for calibration
        self.log_temp_ms  = nn.Parameter(torch.zeros(1))
        self.log_temp_chi = nn.Parameter(torch.zeros(1))

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
            (mean_ms [B,1], lv_ms [B,1])     — ln(Ms)
            (mean_chi [B,1], lv_chi [B,1])   — chi in network space (see chi_space)
        """
        feat = self._backbone(x)

        out_ms  = self.head_ms(feat)
        mean_ms = out_ms[:, 0:1]
        lv_ms   = (out_ms[:, 1:2] + 2.0 * self.log_temp_ms).clamp(-10, 5)

        out_chi  = self.head_chi(feat)
        mean_chi = out_chi[:, 0:1]
        lv_chi   = (out_chi[:, 1:2] + 2.0 * self.log_temp_chi).clamp(-10, 5)

        return (mean_ms, lv_ms), (mean_chi, lv_chi)

    def mc_forward(self, x, n_samples: int = n_mc_samples):
        """
        MC dropout: only Dropout layers are switched to train mode (BatchNorm
        keeps its running statistics). Returns, for ln(Ms) and for chi in
        network space, (mean, aleatoric sigma, epistemic sigma, total sigma)
        with sigma_alea = sqrt(<sigma^2>) over the passes.
        """
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        with torch.no_grad():
            runs = [self(x) for _ in range(n_samples)]
        self.eval()

        def _stats(k):
            mu  = torch.stack([r[k][0] for r in runs])
            var = torch.stack([torch.exp(r[k][1]) for r in runs])
            mean = mu.mean(0)
            epis = mu.std(0)
            alea = var.mean(0).sqrt()
            return mean, alea, epis, torch.sqrt(alea**2 + epis**2)

        return _stats(0), _stats(1)

    def criterion(self, pred, target):
        """Gaussian NLL for ln(Ms) (plus weighted MSE) and for the chi target."""
        (mean_ms, lv_ms), (mean_chi, lv_chi) = pred
        tgt_ms  = target[:, 0:1]
        tgt_chi = target[:, 1:2]

        nll_ms = 0.5 * (lv_ms + (tgt_ms - mean_ms)**2 * torch.exp(-lv_ms))
        w_ms   = torch.where(tgt_ms > np.log(10), 3.0,
                 torch.where(tgt_ms > np.log(4),  2.0, 1.0))
        aux_ms  = (w_ms * (mean_ms - tgt_ms)**2).mean()
        loss_ms = 0.9 * nll_ms.mean() + 0.1 * aux_ms

        nll_chi = 0.5 * (lv_chi + (tgt_chi - mean_chi)**2 * torch.exp(-lv_chi))
        return loss_ms + nll_chi.mean()


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
            weight_decay=0.01, patience=15, num_workers=8):
    set_seed()

    ds_train = MachChiDataset(splits["train"], augment=True)
    ds_val   = MachChiDataset(splits["valid"], augment=False)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=num_workers,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=max(1, num_workers // 2),
                              pin_memory=(device == "cuda"))

    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"Total Steps {epochs * max(1, len(train_loader))}  |  ntrain {len(ds_train)}  |  "
          f"nvalid {len(ds_val)}  |  epochs {epochs}")
    print(f"Architecture: {input_mode} | {n_model_channels}-ch input | MC-dropout={mc_dropout_p} | "
          f"chi target = {chi_field} ({chi_space})")

    best_val, best_state, bad_epochs = float("inf"), None, 0
    t0 = time.time()

    def fmt(s):
        m, s = divmod(int(s), 60); h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

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
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        now  = time.time()
        left = (now - t0) / epoch * (epochs - epoch)
        eta  = datetime.datetime.fromtimestamp(now + left)
        print(f"[{epoch:3d}/{epochs}] net{idd}  "
              f"train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | bad {bad_epochs:02d} | "
              f"T_ms {model.log_temp_ms.item():.3f} T_chi {model.log_temp_chi.item():.3f} | "
              f"ETA {eta.strftime('%H:%M:%S')} | Remain {fmt(left)} | Sofar {fmt(now-t0)}",
              flush=True)

        if bad_epochs >= patience:
            print(f"\n[Early Stopping] Validation loss did not improve for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model | val loss {best_val:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION + CONFORMAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, loader):
    """
    Deterministic predictions for a loader. Returns a dict of numpy arrays:
      ms_true, ms_mean, ms_sig_log   (Ms in physical units, sigma in ln Ms)
      chi_true, chi_mean, chi_sig    (linear chi, delta-method sigma)
      t_ms_*, t_chi_*                (network-space mean, sigma, target)
    """
    model.eval()
    out = {k: [] for k in ("t_ms_mu", "t_ms_sig", "t_ms_y", "t_chi_mu", "t_chi_sig", "t_chi_y")}
    with torch.no_grad():
        for xb, yb in loader:
            (mu_ms, lv_ms), (mu_c, lv_c) = model(xb)
            out["t_ms_mu"].append(mu_ms.view(-1).cpu());  out["t_ms_sig"].append(torch.exp(0.5 * lv_ms).view(-1).cpu())
            out["t_chi_mu"].append(mu_c.view(-1).cpu());  out["t_chi_sig"].append(torch.exp(0.5 * lv_c).view(-1).cpu())
            out["t_ms_y"].append(yb[:, 0].cpu());         out["t_chi_y"].append(yb[:, 1].cpu())
    out = {k: torch.cat(v).numpy().astype(np.float64) for k, v in out.items()}
    out["ms_true"], out["ms_mean"], out["ms_sig_log"] = np.exp(out["t_ms_y"]), np.exp(out["t_ms_mu"]), out["t_ms_sig"]
    out["chi_true"] = chi_from_target(out["t_chi_y"])
    out["chi_mean"], out["chi_sig"] = chi_from_target(out["t_chi_mu"], out["t_chi_sig"])
    return out


def compute_conformal_scores(model, cal_loader):
    """Normalised residuals |y - mu| / sigma in network space for (ln Ms, chi target)."""
    p = predict(model, cal_loader)
    s_ms  = np.abs(p["t_ms_y"]  - p["t_ms_mu"])  / (p["t_ms_sig"]  + 1e-8)
    s_chi = np.abs(p["t_chi_y"] - p["t_chi_mu"]) / (p["t_chi_sig"] + 1e-8)
    return torch.tensor(s_ms), torch.tensor(s_chi)


def get_conformal_quantile(scores: torch.Tensor, coverage: float) -> float:
    n   = len(scores)
    lvl = min(math.ceil((n + 1) * coverage) / n, 1.0)
    return float(torch.quantile(scores, lvl).item())


def build_conformal_intervals(model, data_loader, q_ms: dict, q_chi: dict):
    """
    Conformal intervals mu +- q sigma in network space, mapped to physical
    units (exp for Ms, chi_from_target for chi; both monotone, so the mapped
    interval keeps its coverage). q_* map coverage (e.g. 0.9) -> quantile.
    """
    p = predict(model, data_loader)
    res = {k: p[k] for k in ("ms_true", "ms_mean", "chi_true", "chi_mean")}
    for cov, q in q_ms.items():
        res[f"ms_lo{int(cov*100)}"] = np.exp(p["t_ms_mu"] - q * p["t_ms_sig"])
        res[f"ms_hi{int(cov*100)}"] = np.exp(p["t_ms_mu"] + q * p["t_ms_sig"])
    for cov, q in q_chi.items():
        res[f"chi_lo{int(cov*100)}"] = chi_from_target(p["t_chi_mu"] - q * p["t_chi_sig"])
        res[f"chi_hi{int(cov*100)}"] = chi_from_target(p["t_chi_mu"] + q * p["t_chi_sig"])
    return res


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING  (files: ~/plots/<kind>_net9010_<mode>[_<tag>].png)
# ─────────────────────────────────────────────────────────────────────────────

CHI_LABEL = r"$\chi_v$" if chi_field == "chi_v" else r"$\chi_w$"


def _plot_path(kind, tag=""):
    import os
    d = os.path.join(os.environ["HOME"], "plots")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{kind}_net{idd}_{input_mode}{('_' + tag) if tag else ''}.png")


def _metrics(true, pred):
    from scipy.stats import pearsonr
    resid = pred - true
    ss_t  = np.sum((true - true.mean())**2)
    return {"R2": float(1 - np.sum(resid**2) / ss_t) if ss_t > 0 else np.nan,
            "r": float(pearsonr(true, pred)[0]) if len(true) > 2 else np.nan,
            "MAE": float(np.mean(np.abs(resid))), "RMSE": float(np.sqrt(np.mean(resid**2)))}


def regression_metrics(p):
    """Headline metrics for a predict() dict."""
    m_ms   = _metrics(np.log(p["ms_true"]), np.log(p["ms_mean"]))
    m_msl  = _metrics(p["ms_true"], p["ms_mean"])
    m_chi  = _metrics(p["chi_true"], p["chi_mean"])
    m_chit = _metrics(p["t_chi_y"], p["t_chi_mu"])
    return {"lnMs": m_ms, "Ms": m_msl,
            "Ms_median_frac_err": float(np.median(np.abs(p["ms_mean"] - p["ms_true"]) / p["ms_true"])),
            "chi": m_chi, "chi_target_space": m_chit}


def plot_loss_curve(model):
    import matplotlib.pyplot as plt
    n     = int((model.train_curve != 0).sum().item())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.train_curve[:n].cpu().numpy(), lw=2, label="Train Loss")
    ax.plot(model.val_curve[:n].cpu().numpy(),   lw=2, label="Val Loss")
    ax.set_yscale("symlog")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Combined NLL")
    ax.set_title(f"Training Curves – net{idd} ({input_mode})"); ax.legend(); ax.grid(True, alpha=0.3)
    oname = _plot_path("loss_curve")
    fig.savefig(oname, dpi=150, bbox_inches="tight"); print(f"Saved: {oname}"); plt.close(fig)


def plot_predictions_dual(model, loader, tag="val"):
    """Predicted vs true Ms (log axes) and chi, with metrics."""
    import matplotlib.pyplot as plt
    p = predict(model, loader)
    m = regression_metrics(p)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(p["ms_true"], p["ms_mean"], s=4, alpha=0.4, color="steelblue")
    lo, hi = min(p["ms_true"].min(), p["ms_mean"].min()), max(p["ms_true"].max(), p["ms_mean"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5); ax.set_xscale("log"); ax.set_yscale("log")
    txt = (f"R²(ln Ms)={m['lnMs']['R2']:.4f}\nR²(Ms)={m['Ms']['R2']:.4f}\n"
           f"median |Δ|/Ms={m['Ms_median_frac_err']:.3f}\nN={len(p['ms_true'])}")
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=8, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))
    ax.set_xlabel(r"True $\mathcal{M}_s$"); ax.set_ylabel(r"Predicted $\mathcal{M}_s$")
    ax.set_title(f"net{idd} ({input_mode}) — {tag}"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(p["chi_true"], p["chi_mean"], s=4, alpha=0.5, c=np.log10(p["ms_true"]), cmap="viridis")
    ax.plot([0, 1], [0, 1], "r--", lw=1.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    txt = (f"R²={m['chi']['R2']:.4f}\nr={m['chi']['r']:.4f}\nMAE={m['chi']['MAE']:.4f}\n"
           f"R²({chi_space})={m['chi_target_space']['R2']:.4f}\nN={len(p['chi_true'])}")
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=8, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}\mathcal{M}_s$")
    ax.set_xlabel(f"True {CHI_LABEL}"); ax.set_ylabel(f"Predicted {CHI_LABEL}")
    ax.set_title(f"net{idd} ({input_mode}) — {tag}"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    oname = _plot_path("predictions", tag)
    fig.savefig(oname, dpi=150, bbox_inches="tight"); print(f"Saved: {oname}"); plt.close(fig)
    return m


def plot_coverage_calibration(model, cal_loader, eval_loader, tag="test"):
    """
    Coverage of the uncertainties on `eval_loader`, for nominal levels 0.5-0.95:
      * raw Gaussian: fraction with |y - mu| <= z_a sigma (before conformal)
      * conformal: quantiles q_a from `cal_loader` scores, coverage measured on
        `eval_loader` (must be disjoint from cal_loader, or this is circular)
    Returns {"ms": (mae_raw, mae_conf), "chi": (...)} as mean |empirical - nominal|.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    c_ms, c_chi = compute_conformal_scores(model, cal_loader)
    e_ms, e_chi = compute_conformal_scores(model, eval_loader)
    nominal = np.arange(0.50, 0.951, 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    out = {}
    for ax, cal, ev, key, lbl in [(axes[0], c_ms, e_ms, "ms", "ln Ms"),
                                  (axes[1], c_chi, e_chi, "chi", f"{chi_field} ({chi_space})")]:
        raw  = np.array([(ev <= norm.ppf(0.5 + a / 2)).float().mean().item() for a in nominal])
        conf = np.array([(ev <= get_conformal_quantile(cal, a)).float().mean().item() for a in nominal])
        out[key] = (float(np.mean(np.abs(raw - nominal))), float(np.mean(np.abs(conf - nominal))))
        ax.plot(nominal, raw, "s--", color="gray", lw=1.5, label=f"Gaussian σ (MAE {out[key][0]:.3f})")
        ax.plot(nominal, conf, "bo-", lw=2, label=f"Conformal (MAE {out[key][1]:.3f})")
        ax.plot([0.5, 1.0], [0.5, 1.0], "r--", lw=1)
        ax.set_xlabel("Nominal Coverage"); ax.set_ylabel(f"Empirical Coverage ({tag})")
        ax.set_title(f"{lbl}: calibrated on calib, measured on {tag}")
        ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    oname = _plot_path("coverage_calibration", tag)
    fig.savefig(oname, dpi=150, bbox_inches="tight")
    print(f"Saved: {oname}  (coverage MAE raw/conformal — Ms {out['ms'][0]:.3f}/{out['ms'][1]:.3f}  "
          f"chi {out['chi'][0]:.3f}/{out['chi'][1]:.3f})")
    plt.close(fig)
    return out


def plot_conformal_intervals_dual(results, tag="test", levels=(90, 95)):
    """Sorted predictions with conformal bands: Ms (top, log) and chi (bottom)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for ax, key, lbl in [(axes[0], "ms", r"$\mathcal{M}_s$"), (axes[1], "chi", CHI_LABEL)]:
        true  = results[f"{key}_true"]
        order = np.argsort(true)
        x     = np.arange(len(order))
        for lev, a in zip(sorted(levels, reverse=True), (0.30, 0.45)):
            if f"{key}_lo{lev}" in results:
                ax.fill_between(x, results[f"{key}_lo{lev}"][order], results[f"{key}_hi{lev}"][order],
                                alpha=a, color="royalblue", label=f"{lev}% interval")
        ax.plot(x, results[f"{key}_mean"][order], "b-", lw=1, label="Prediction")
        ax.scatter(x, true[order], s=6, c="red", zorder=3, label="True")
        if key == "ms":
            ax.set_yscale("log")
        ax.set_xlabel(f"Sample (sorted by true {lbl})"); ax.set_ylabel(lbl)
        cov = {lev: float(np.mean((true >= results[f"{key}_lo{lev}"]) & (true <= results[f"{key}_hi{lev}"])))
               for lev in levels if f"{key}_lo{lev}" in results}
        ax.set_title("  ".join(f"{lev}% coverage = {c:.3f}" for lev, c in cov.items()), fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Conformal Intervals — net{idd} ({input_mode}) — {tag}", fontsize=13)
    plt.tight_layout()
    oname = _plot_path("conformal_intervals", tag)
    fig.savefig(oname, dpi=150, bbox_inches="tight"); print(f"Saved: {oname}"); plt.close(fig)


def plot_uncertainty_analysis_dual(model, test_loader, tag="test"):
    """
    MC-dropout uncertainty analysis, Ms (left) and chi (right): sorted
    predictions, pred vs truth, epistemic vs aleatoric, sigma histograms.
    Ms sigmas are in ln Ms; chi sigmas are converted to linear chi.
    """
    import matplotlib.pyplot as plt
    rows = {k: [] for k in ("ms_mu", "ms_al", "ms_ep", "ms_tot", "ms_y",
                            "c_mu", "c_al", "c_ep", "c_tot", "c_y")}
    for xb, yb in test_loader:
        ms_r, c_r = model.mc_forward(xb, n_samples=n_mc_samples)
        for k, v in zip(("ms_mu", "ms_al", "ms_ep", "ms_tot"), ms_r):
            rows[k].append(v.view(-1).cpu())
        for k, v in zip(("c_mu", "c_al", "c_ep", "c_tot"), c_r):
            rows[k].append(v.view(-1).cpu())
        rows["ms_y"].append(yb[:, 0].cpu()); rows["c_y"].append(yb[:, 1].cpu())
    r = {k: torch.cat(v).numpy().astype(np.float64) for k, v in rows.items()}

    ms_mean, ms_true = np.exp(r["ms_mu"]), np.exp(r["ms_y"])
    chi_mean, chi_al = chi_from_target(r["c_mu"], r["c_al"])
    _, chi_ep  = chi_from_target(r["c_mu"], r["c_ep"])
    _, chi_tot = chi_from_target(r["c_mu"], r["c_tot"])
    chi_true = chi_from_target(r["c_y"])

    data = [(ms_mean, r["ms_al"], r["ms_ep"], r["ms_tot"], ms_mean * r["ms_tot"], ms_true,
             r"$\mathcal{M}_s$", "ln Ms"),
            (chi_mean, chi_al, chi_ep, chi_tot, chi_tot, chi_true, CHI_LABEL, "linear")]
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    for col, (means, alea, epis, tot, tot_phys, trues, lbl, ulbl) in enumerate(data):
        order = np.argsort(trues); x = np.arange(len(order))
        ax = axes[0, col]
        ax.plot(trues[order], "r-", lw=1.5, label="True")
        ax.errorbar(x, means[order], yerr=tot_phys[order], fmt="none", ecolor="steelblue",
                    alpha=0.3, elinewidth=0.5)
        ax.plot(means[order], "b.", ms=2, alpha=0.5, label="Pred ± σ")
        if col == 0: ax.set_yscale("log")
        ax.set_xlabel("Sample (sorted)"); ax.set_ylabel(lbl)
        ax.set_title(f"Sorted Predictions — {lbl}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.scatter(trues, means, s=3, alpha=0.3, color="steelblue")
        lo, hi = trues.min(), trues.max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        if col == 0: ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"True {lbl}"); ax.set_ylabel(f"Predicted {lbl}")
        ax.set_title(f"Pred vs Truth — {lbl}"); ax.grid(True, alpha=0.3)

        ax = axes[2, col]
        ax.scatter(alea, epis, s=3, alpha=0.3, color="steelblue")
        ax.set_xlabel(f"Aleatoric σ ({ulbl})"); ax.set_ylabel(f"Epistemic σ ({ulbl})")
        ax.set_title(f"Uncertainty Decomposition — {lbl}"); ax.grid(True, alpha=0.3)

        ax = axes[3, col]
        for v, c, l in ((alea, "steelblue", "Aleatoric"), (epis, "orange", "Epistemic"), (tot, "green", "Total")):
            ax.hist(v, bins=50, alpha=0.6, color=c, label=l)
        ax.set_xlabel(f"σ ({ulbl})"); ax.set_ylabel("Count")
        ax.set_title(f"Uncertainty Distribution — {lbl}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Uncertainty Analysis — net{idd} ({input_mode}) — {tag}", fontsize=14, y=1.01)
    plt.tight_layout()
    oname = _plot_path("uncertainty_analysis", tag)
    fig.savefig(oname, dpi=150, bbox_inches="tight"); print(f"Saved: {oname}"); plt.close(fig)


def plot_resolution_robustness_dual(model, sample_x, true_ms, true_chi, tag=""):
    """
    Prediction and sigma vs resolution factor for one image.
    sample_x : [1, n_input_channels, H, W] raw (physical) maps, no res channel.
    Degradation happens on the physical maps, then preprocess_maps, exactly as
    in MachChiDataset.
    """
    import matplotlib.pyplot as plt
    factors = [0.3, 0.4, 0.5, 0.7, 0.9, 1.0]
    ms_p, ms_s, c_p, c_s = [], [], [], []
    model.eval()
    with torch.no_grad():
        for f in factors:
            x = sample_x.clone()
            if f < 1.0:
                x = _degrade_resolution(x, f)
            x = preprocess_maps(x)
            res_ch = torch.full((1, 1, x.shape[-2], x.shape[-1]), f, dtype=x.dtype, device=x.device)
            (mu_ms, lv_ms), (mu_c, lv_c) = model(torch.cat([x, res_ch], dim=1).to(device))
            ms_val = float(np.exp(mu_ms.item()))
            ms_p.append(ms_val); ms_s.append(ms_val * float(torch.exp(0.5 * lv_ms).item()))
            cm, cs = chi_from_target(mu_c.item(), float(torch.exp(0.5 * lv_c).item()))
            c_p.append(float(cm)); c_s.append(float(cs))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, (preds, sigmas, true_val, lbl) in enumerate(
            [(ms_p, ms_s, true_ms, r"$\mathcal{M}_s$"), (c_p, c_s, true_chi, CHI_LABEL)]):
        ax = axes[row, 0]
        ax.errorbar(factors, preds, yerr=sigmas, fmt="bo-", lw=2, capsize=4, label="Prediction ± σ")
        ax.axhline(true_val, color="r", ls="--", lw=1.5, label=f"True: {true_val:.3f}")
        ax.set_xlabel("Resolution Factor"); ax.set_ylabel(f"Predicted {lbl}")
        ax.set_title(f"{lbl}: Prediction vs Resolution"); ax.legend(); ax.grid(True, alpha=0.3)
        ax = axes[row, 1]
        ax.plot(factors, sigmas, "go-", lw=2)
        ax.set_xlabel("Resolution Factor"); ax.set_ylabel(f"σ ({lbl})")
        ax.set_title(f"{lbl}: Uncertainty vs Resolution"); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Resolution Robustness — net{idd} ({input_mode}) — {tag}", fontsize=13)
    plt.tight_layout()
    oname = _plot_path("resolution_awareness", tag)
    fig.savefig(oname, dpi=150, bbox_inches="tight"); print(f"Saved: {oname}"); plt.close(fig)
