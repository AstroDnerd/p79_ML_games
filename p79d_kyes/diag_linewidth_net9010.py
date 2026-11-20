"""
diag_linewidth_net9010.py
=========================
Where does the tile linewidth Ms_obs = sqrt(3) sigma_tile / c_s depart from the volume-weighted Ms?
For run-2 images, sigma_tile is computed at three stages and compared with the labels:

  rt   : noise-free 13CO moments (RADMC-3D, no beam, no noise); thermal width removed
  obs  : FCRAO-like observed 13CO moments (beam, noise, 3 sigma mask); thermal + channel removed

and with sqrt(3) x the mass- and volume-weighted line-of-sight Mach numbers of the snapshot. For the
projected maps (no radiative transfer) sigma_tile equals the mass-weighted LOS dispersion exactly
(law of total variance), so "sqrt3 Ms_los_mw" IS that stage. ("images" of the CO file is images_13co.)
Medians of each ratio to the true Ms by Ms bin, and at Ms 1-3 by density and forcing.

  python diag_linewidth_net9010.py [--step 3] > diag_linewidth.txt
"""
import argparse, json
import numpy as np, h5py
p = argparse.ArgumentParser()
p.add_argument("--data", default="/home/x-nbisht1/scratch/projects/p79d_dataset/p79d_mach_grid_256_co_v2_fcrao500.h5")
p.add_argument("--step", type=int, default=3)
a = p.parse_args()
f = h5py.File(a.data, "r")
cs = float(f.attrs["sound_speed_kms"]); rt = json.loads(f.attrs["rt_json"]); ob = json.loads(f.attrs["observe_version"])
sth = np.sqrt(1.380649e-16 * rt["T_K"] / (29.0 * 1.6735575e-24)) / 1e5 / cs      # 13CO thermal sigma, units of c_s
dv = ob["channel_width_kms"] / cs
L = {k: f[f"labels/{k}"][:] for k in ("Ms", "Ms_los_mw", "Ms_los_vw", "Ms_mw", "n_H2", "xi")}
idx = np.arange(0, len(L["Ms"]), a.step); idx = idx[(L["Ms"][idx] >= 0.5) & (L["Ms"][idx] <= 20)]


def sig(w, v, s, m=None):
    m = np.isfinite(v) & np.isfinite(s) & (w > 0) if m is None else m & np.isfinite(v) & np.isfinite(s)
    w = np.clip(w[m], 0, None); vb = np.sum(w * v[m]) / w.sum()
    return np.sqrt(np.sum(w * (v[m] - vb) ** 2) / w.sum()), np.sqrt(np.sum(w * s[m] ** 2) / w.sum())


R = {k: np.empty((len(idx), 2)) for k in ("rt", "obs")}
for c0 in range(0, len(idx), 400):
    ii = idx[c0:c0 + 400]
    T, O, M = f["images_13co"][ii], f["images_obs_13co"][ii], f["mask_obs_13co"][ii]
    for j in range(len(ii)):
        R["rt"][c0 + j] = sig(*T[j])
        R["obs"][c0 + j] = np.array(sig(O[j, 0], O[j, 1], O[j, 2], M[j])) / cs
ms = L["Ms"][idx]
tot = lambda r, sub=0.0: np.sqrt(3) * np.sqrt(np.clip(r[:, 0] ** 2 + r[:, 1] ** 2 - sub, 1e-8, None))
Q = {"rt": tot(R["rt"], sth ** 2), "obs": tot(R["obs"], sth ** 2 + dv ** 2 / 12),
     "rt centroid only": np.sqrt(3) * R["rt"][:, 0], "rt internal only": np.sqrt(3) * np.sqrt(np.clip(R["rt"][:, 1] ** 2 - sth ** 2, 1e-8, None)),
     "obs centroid only": np.sqrt(3) * R["obs"][:, 0],
     "sqrt3 Ms_los_mw": np.sqrt(3) * L["Ms_los_mw"][idx], "sqrt3 Ms_los_vw": np.sqrt(3) * L["Ms_los_vw"][idx], "Ms_mw": L["Ms_mw"][idx]}
print(f"{len(idx)} images (every {a.step}th); c_s {cs:.4f} km/s; 13CO thermal sigma {sth:.3f} c_s; channel {dv:.3f} c_s")
print("median ratio to the volume-weighted Ms; (in brackets) median |ratio - 1|")
BINS = [(0.5, 1), (1, 2), (2, 3), (3, 7), (7, 20.01)]
print(f"{'quantity':>20s} " + " ".join(f"{f'{lo:g}-{hi:g}':>15s}" for lo, hi in BINS))
for k, q in Q.items():
    print(f"{k:>20s} " + " ".join(f"{np.median((q / ms)[(ms >= lo) & (ms < hi)]):7.3f} ({np.median(np.abs(q / ms - 1)[(ms >= lo) & (ms < hi)]):.3f})" for lo, hi in BINS))
b = (ms >= 1) & (ms < 3); n, xi = L["n_H2"][idx], L["xi"][idx]
print("\nMs 1-3, by mean density (cm^-3): median ratio to Ms")
for lo, hi in [(100, 300), (300, 1000), (1000, 3001)]:
    s = b & (n >= lo) & (n < hi)
    print(f"  n {lo}-{hi} N={s.sum():5d}: " + "  ".join(f"{k} {np.median((Q[k] / ms)[s]):.3f}" for k in ("rt", "obs", "sqrt3 Ms_los_mw")))
print("all Ms bins, by mean density: median rt/Ms (noise-free line, thermal removed)")
for lo, hi in [(100, 300), (300, 1000), (1000, 3001)]:
    print(f"  n {lo}-{hi}: " + " ".join(f"{np.median((Q['rt'] / ms)[(ms >= l_) & (ms < h_) & (n >= lo) & (n < hi)]):7.3f}" for l_, h_ in BINS))
print("Ms 1-3, by forcing: median ratio to Ms")
for x in np.unique(xi):
    s = b & (xi == x)
    print(f"  xi {x:.2f} N={s.sum():5d}: " + "  ".join(f"{k} {np.median((Q[k] / ms)[s]):.3f}" for k in ("rt", "obs", "sqrt3 Ms_los_mw", "sqrt3 Ms_los_vw")))
print("Ms 1-3, by line of sight scatter: 16-84% of obs/Ms", np.percentile((Q["obs"] / ms)[b], [16, 50, 84]).round(3),
      " of sqrt3 Ms_los_vw/Ms", np.percentile((Q["sqrt3 Ms_los_vw"] / ms)[b], [16, 50, 84]).round(3))
