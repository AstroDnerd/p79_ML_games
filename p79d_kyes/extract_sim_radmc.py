"""
extract_sim_radmc.py
====================
Synthetic 13CO J=1-0 dataset of the mach_grid suite with RADMC-3D: the
radiative-transfer counterpart of extract_sim_data.py. Same simulations,
frames, lines of sight, periodic shifts, orientation and 3D labels, so every
image here pairs pixel-for-pixel with one in the projected dataset.

Per frame:
  1. load rho, v (yt covering grid, native 256^3) and compute the 3D labels
     (Ms, chi, b, ...) exactly as extract_sim_data.py does;
  2. scale to physical units (radmc_13co.json): box length, mean n(H2), and
     code velocity = isothermal sound speed at T; write binary RADMC-3D inputs;
  3. ray-trace a PPV cube along x, y and z (LTE, one ray per cell column), with
     an FCRAO-like channel width and a velocity window sized per line of sight;
  4. per image: apply the shift, block-average to 128^2, convolve with the beam
     (periodic), and form
       images      noise-free moments in training units:
                   [W/<W>, v_c/c_s, sigma_v/c_s]   (drop-in for net9010)
       images_obs  noise added, then the observation pipeline's own moment code
                   (get_Observation_Data.compute_moment_maps: 3-sigma voxel mask,
                   edge-channel noise estimate): [K km/s, km/s, km/s], NaN where
                   masked, with mask_obs.

  python extract_sim_radmc.py extract --config sims_mach_grid.json --rt radmc_13co.json --index 0
  python extract_sim_radmc.py extract ... --index 0 --chunk 2 --n_chunks 6     # frames[2::6] only
  python extract_sim_radmc.py merge   --config sims_mach_grid.json --rt radmc_13co.json

PPV cubes: each image's noise-free, block-averaged (128^2) brightness-temperature
cube BEFORE beam convolution, shifted like the image, is kept in the part files
(cube_<los>_<aug>, float16, K; velocity axis v_<los>, km/s) so other telescope
setups can be generated later without re-running RADMC-3D. The merged file
records where each sample's cube lives (samples/cube_file, samples/cube_key).
  python extract_sim_radmc.py extract ... --sim xi_0_mach10 --frames 150 --keep_cubes DIR   # test
"""

import argparse, contextlib, datetime, io, json, os, shutil, subprocess, tempfile, time, traceback, zlib
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter

import extract_sim_data as E
import get_Observation_Data as G

PC_CM, AU_CM, C_KMS = 3.0857e18, 1.495978707e13, 2.99792458e5
KB, MH, C_CGS = 1.380649e-16, 1.6735575e-24, 2.99792458e10
M_13CO = 29.0 * MH

# How RADMC-3D views the box for each of our lines of sight, determined on a
# test box (see the RADMC section of the paper notes): image axis 0 / 1 and the
# sign that turns RADMC's (receding-positive) velocity into +v along our axis.
#   LOS x: incl=90 phi=90 -> image (-y, z), radial velocity = +v_x
#   LOS y: incl=90 phi=0  -> image ( x, z), radial velocity = +v_y
#   LOS z: incl=0  phi=0  -> image ( x, y), radial velocity = -v_z
RT_VIEW = {"x": dict(incl=90, phi=90, flip0=True,  vsign=+1),
           "y": dict(incl=90, phi=0,  flip0=False, vsign=+1),
           "z": dict(incl=0,  phi=0,  flip0=False, vsign=-1)}


# ─────────────────────────────────────────────────────────────────────────────
# RADMC-3D INPUT / OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def sound_speed_kms(rc):
    return np.sqrt(KB * rc["T_K"] / (rc["mu_sound"] * MH)) / 1e5


def _write_binp(path, arrays):
    """RADMC-3D binary input: int64 [format=1, precision=8, ncells] + doubles (cells x-fastest)."""
    ncell = arrays[0].size
    with open(path, "wb") as f:
        np.array([1, 8, ncell], dtype=np.int64).tofile(f)
        if len(arrays) == 1:
            np.asarray(arrays[0], dtype=np.float64).ravel(order="F").tofile(f)
        else:
            np.stack([a.ravel(order="F") for a in arrays], axis=1).astype(np.float64).tofile(f)


def write_inputs(d, rho, v_kms, rc):
    """Write every RADMC-3D input file for one snapshot into directory d."""
    N = rho.shape[0]
    L = rc["box_length_pc"] * PC_CM
    edges = " ".join(f"{x:.10e}" for x in np.linspace(0.0, L, N + 1))
    with open(os.path.join(d, "amr_grid.inp"), "w") as f:
        f.write(f"1\n0\n1\n0\n1 1 1\n{N} {N} {N}\n{edges}\n{edges}\n{edges}\n")
    _write_binp(os.path.join(d, f"numberdens_{rc['molecule']}.binp"),
                [rc["X_13CO"] * rc["n_H2_mean_cm3"] * rho / rho.mean()])
    _write_binp(os.path.join(d, "gas_velocity.binp"), [vi * 1e5 for vi in v_kms])      # cm/s
    _write_binp(os.path.join(d, "gas_temperature.binp"), [np.full(rho.shape, rc["T_K"])])
    with open(os.path.join(d, "lines.inp"), "w") as f:
        f.write(f"2\n1\n{rc['molecule']}    leiden    0    0    0\n")
    with open(os.path.join(d, "radmc3d.inp"), "w") as f:
        f.write("lines_mode = 1\nnphot_scat = 0\ncamera_tracemode = 1\ncamera_nrrefine = 1\n"
                "tgas_eq_tdust = 0\nlines_show_pictograms = 0\n")
    lam0_um = C_CGS / rc["rest_frequency_hz"] * 1e4
    with open(os.path.join(d, "wavelength_micron.inp"), "w") as f:
        f.write(f"1\n{lam0_um:.6e}\n")
    shutil.copy(rc["molecule_file"], os.path.join(d, f"molecule_{rc['molecule']}.inp"))


def velocity_window(v_los_kms, rc):
    """Half-width W and channel count so the line plus emission-free edge channels fit."""
    dv  = rc["channel_width_kms"]
    sth = np.sqrt(KB * rc["T_K"] / M_13CO) / 1e5
    q   = np.quantile(np.abs(v_los_kms - v_los_kms.mean()), rc["velocity_quantile"])
    W   = q + rc["velocity_margin_thermal_sigmas"] * sth + rc["edge_channels"] * dv
    half = int(np.ceil(W / dv))
    return half * dv, 2 * half + 1


def launch_image(d, rc, view, W, nch):
    """Start RADMC-3D for one view in directory d (returns the Popen)."""
    N = rc["npix_rt"]
    au = rc["box_length_pc"] * PC_CM / AU_CM
    cmd = [rc["radmc3d_binary"], "image", "npix", str(N), "sizeau", f"{au:.8e}",
           "pointau", f"{au/2:.8e}", f"{au/2:.8e}", f"{au/2:.8e}",
           "incl", str(view["incl"]), "phi", str(view["phi"]), "iline", "1",
           "widthkms", f"{2 * W:.6f}", "linenlam", str(nch), "vkms", "0", "imageunform"]
    return subprocess.Popen(cmd, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def read_image(d, rc, view):
    """
    Read image.bout -> brightness-temperature cube T[i0, i1, v] (K) in OUR
    orientation for this LOS, and the velocity axis (km/s, ascending, +v along
    our axis).
    """
    raw = open(os.path.join(d, "image.bout"), "rb").read()
    _, nx, ny, nl = np.frombuffer(raw[:32], dtype=np.int64)
    lam = np.frombuffer(raw[48:48 + 8 * nl], dtype=np.float64)              # micron
    img = np.frombuffer(raw[48 + 8 * nl:], dtype=np.float64).reshape(nl, ny, nx)
    cube = img.transpose(2, 1, 0)                                           # [ix, iy, lam]
    lam0 = C_CGS / rc["rest_frequency_hz"] * 1e4
    v = view["vsign"] * C_KMS * (lam / lam0 - 1.0)
    T = C_CGS**2 / (2 * KB * rc["rest_frequency_hz"]**2) * cube             # Rayleigh-Jeans T_B
    if view["flip0"]:
        T = T[::-1]
    order = np.argsort(v)
    return np.ascontiguousarray(T[:, :, order]), v[order]


# ─────────────────────────────────────────────────────────────────────────────
# PER-IMAGE PRODUCTS
# ─────────────────────────────────────────────────────────────────────────────

def block_mean(cube, f):
    if f == 1:
        return cube
    n0, n1, nv = cube.shape
    return cube.reshape(n0 // f, f, n1 // f, f, nv).mean(axis=(1, 3))


def clean_moments(T, v, cs_kms):
    """Noise-free moments over all channels, in training units [W/<W>, v_c/c_s, sigma/c_s]."""
    dv  = abs(v[1] - v[0])
    m0  = T.sum(axis=2) * dv
    m1  = (T * v).sum(axis=2) * dv / np.maximum(m0, 1e-30)
    var = (T * v**2).sum(axis=2) * dv / np.maximum(m0, 1e-30) - m1**2
    return np.stack([m0 / m0.mean(), m1 / cs_kms, np.sqrt(np.maximum(var, 0)) / cs_kms]).astype(np.float32)


def observed_moments(T, v, rc, min_channels=1):
    """
    The observation pipeline's moments on the noisy cube: [K km/s, km/s, km/s],
    NaN masked. min_channels=1 reproduces the original extraction; reobserve
    uses the contiguous-channel mask (see get_Observation_Data).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        m0, m1, m2, mask = G.compute_moment_maps(T.transpose(2, 1, 0), v, noise_threshold=rc["mask_sigma"],
                                                 min_channels=min_channels)
    # compute_moment_maps works on [v, y, x] and returns [y, x]; transpose back to [i0, i1]
    return np.stack([m0.T, m1.T, m2.T]).astype(np.float32), mask.T


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT
# ─────────────────────────────────────────────────────────────────────────────

def rt_cfg(cfg, rc):
    """A copy of the suite config whose part/merged paths are those of the RT dataset."""
    c = dict(cfg)
    c["output_name"] = rc["output_name"]
    return c


def part_file(rcfg, name, chunk=None, n_chunks=1):
    p = E.part_path(rcfg, name)
    return p if n_chunks <= 1 else p[:-3] + f"_c{chunk:02d}of{n_chunks:02d}.h5"


def part_files(rcfg, name):
    """All part files of one simulation (unchunked and chunked)."""
    import glob, re
    p = E.part_path(rcfg, name)
    chunked = [q for q in glob.glob(p[:-3] + "_c*of*.h5")
               if re.fullmatch(re.escape(os.path.basename(p)[:-3]) + r"_c\d+of\d+\.h5", os.path.basename(q))]
    return sorted([q for q in [p] if os.path.exists(q)] + chunked)


def cmd_extract(args):
    cfg, rc = json.load(open(args.config)), json.load(open(args.rt))
    sims = cfg["sims"]
    sim  = sims[args.index] if args.sim is None else next(s for s in sims if s["name"] == args.sim)
    ex, name = cfg["extraction"], sim["name"]
    sim_dir = os.path.join(cfg["data_root"], name)
    cs_code = sim.get("sound_speed", 1.0)
    cs_kms  = sound_speed_kms(rc)
    out     = part_file(rt_cfg(cfg, rc), name, args.chunk, args.n_chunks)
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if not os.access(rc["radmc3d_binary"], os.X_OK):
        raise SystemExit(f"RADMC-3D binary not executable: {rc['radmc3d_binary']}")

    frames = [f for f in E.frame_dirs(sim_dir) if f >= ex.get("min_frame", 0)][::rc.get("frame_stride", 1)]
    if args.frames:
        frames = [f for f in frames if f in set(args.frames)]
    if args.n_chunks > 1:
        frames = frames[args.chunk::args.n_chunks]     # interleaved: every chunk spans the run
    if args.max_frames:
        frames = frames[:args.max_frames]
    with h5py.File(out, "a") as h:
        done = set(int(g[2:]) for g in h.keys() if g.startswith("DD"))
        h.attrs["sim_name"], h.attrs["sim_json"], h.attrs["rt_json"] = name, json.dumps(sim), json.dumps(rc)
    todo = [f for f in frames if f not in done]
    print(f"{name}: {len(frames)} frames, {len(done)} done, {len(todo)} to do → {out}\n"
          f"  c_s = {cs_kms:.4f} km/s per code velocity, workers={workers}", flush=True)

    tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    L_pc = rc["box_length_pc"]
    pix_arcsec  = (L_pc / rc["target_res"]) / rc["distance_pc"] * 206265.0
    beam_sigma  = rc["beam_fwhm_arcsec"] / 2.3548 / pix_arcsec
    factor      = rc["npix_rt"] // rc["target_res"]
    kg, skipped, t_start = None, [], time.time()

    for i, frame in enumerate(todo):
        t0 = time.time()
        if not E.frame_readable(sim_dir, frame):
            skipped.append((frame, "unreadable")); print(f"  DD{frame:04d} skipped: unreadable", flush=True)
            continue
        work = tempfile.mkdtemp(prefix=f"rt_{name}_{frame}_", dir=tmp_root)
        try:
            rho, v, t = E.load_frame(sim_dir, frame)
            N = rho.shape[0]
            if N != rc["npix_rt"]:
                raise ValueError(f"grid {N}^3 but npix_rt = {rc['npix_rt']}")
            if kg is None or kg.N != N:
                kg = E.KGrid(N)
            q = E.frame_scalars(rho, v, kg, cs_code, workers)
            q["time"] = t
            v_kms = [(vi - vi.mean()) / cs_code * cs_kms for vi in v]
            write_inputs(work, rho, v_kms, rc)

            # one RADMC-3D process per LOS, in parallel, sharing the input files
            procs, wins = {}, {}
            for il, ax in enumerate(E.LOS_AXES):
                d = os.path.join(work, ax); os.makedirs(d)
                for fn in os.listdir(work):
                    if os.path.isfile(os.path.join(work, fn)):
                        os.symlink(os.path.join(work, fn), os.path.join(d, fn))
                wins[ax] = velocity_window(v_kms[il], rc)
                procs[ax] = launch_image(d, rc, RT_VIEW[ax], *wins[ax])
            for ax, pr in procs.items():
                log, _ = pr.communicate()
                if pr.returncode != 0 or not os.path.exists(os.path.join(work, ax, "image.bout")):
                    raise RuntimeError(f"RADMC-3D failed for LOS {ax}:\n{log[-2000:]}")

            aug = E.augment_params(ex, name, frame, N)
            rng = np.random.default_rng([ex["seed"], zlib.crc32(name.encode()), frame, 13])
            imgs, obs, masks, meta, cubes, vaxes = [], [], [], [], {}, {}
            for il, ax in enumerate(E.LOS_AXES):
                T, vax = read_image(os.path.join(work, ax), rc, RT_VIEW[ax])
                if args.keep_cubes:
                    os.makedirs(args.keep_cubes, exist_ok=True)
                    np.savez_compressed(os.path.join(args.keep_cubes, f"{name}_DD{frame:04d}_{ax}.npz"),
                                        T=T.astype(np.float32), v=vax)
                ls = E.los_scalars(rho, v, ax, cs_code)
                for _, ia, dx, dy, k90 in (a for a in aug if a[0] == il):
                    Ts = np.rot90(np.roll(T, (dx, dy), axis=(0, 1)), k90, axes=(0, 1))
                    Tr = block_mean(Ts, factor)
                    cubes[f"cube_{il}_{ia}"] = Tr.astype(np.float16)
                    vaxes[f"v_{il}"] = vax.astype(np.float32)
                    Tb = gaussian_filter(Tr, sigma=(beam_sigma, beam_sigma, 0), mode="wrap")
                    imgs.append(clean_moments(Tb, vax, cs_kms))
                    o, mk = observed_moments(Tb + rng.normal(0.0, rc["noise_rms_K"], Tb.shape), vax, rc)
                    obs.append(o); masks.append(mk)
                    meta.append((il, ia, dx, dy, k90, ls["Ms_los_mw"], ls["Ms_los_vw"],
                                 wins[ax][0], wins[ax][1], float(Tb.max()), float(mk.mean())))
            del rho, v

            with h5py.File(out, "a") as h:
                g = h.create_group(f"DD{frame:04d}")
                g["images"], g["images_obs"] = np.stack(imgs), np.stack(obs)
                g["mask_obs"] = np.stack(masks)
                for k, c in cubes.items():
                    g.create_dataset(k, data=c, chunks=(c.shape[0], c.shape[1], min(16, c.shape[2])),
                                     compression="lzf")
                    g[k].attrs["description"] = ("noise-free 13CO T_B (K) at target_res, shifted like the "
                                                 "image, before beam convolution; axes [i0, i1, v]")
                for k, va in vaxes.items():
                    g[k] = va
                m = np.array(meta, dtype=np.float64)
                for j, k in enumerate(["los", "aug_index", "shift_x", "shift_y", "rot90"]):
                    g[k] = m[:, j].astype(np.int32)
                for j, k in enumerate(["Ms_los_mw", "Ms_los_vw", "v_window_kms", "n_channels",
                                       "Tpeak_K", "frac_detected"], start=5):
                    g[k] = m[:, j]
                for k, val in q.items():
                    g.attrs[k] = val
            rem = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1)
            print(f"  DD{frame:04d} Ms={q['Ms']:6.3f} chi_v={q['chi_v']:.3f} "
                  f"nch={'/'.join(str(wins[a][1]) for a in E.LOS_AXES)} "
                  f"Tpeak={max(r[9] for r in meta):5.2f}K det={np.mean([r[10] for r in meta]):.2f}  "
                  f"{time.time() - t0:5.1f}s  (~{rem / 60:.0f} min left)", flush=True)
        except Exception as e:
            skipped.append((frame, repr(e)))
            print(f"  DD{frame:04d} FAILED: {e!r}", flush=True)
            traceback.print_exc()
        finally:
            shutil.rmtree(work, ignore_errors=True)

    with h5py.File(out, "a") as h:
        h.attrs["skipped"] = json.dumps(json.loads(h.attrs.get("skipped", "[]")) + skipped)
    print(f"{name}: finished, {len(skipped)} skipped this run", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# REOBSERVE (stored cubes -> a new observed version, no ray tracing)
# ─────────────────────────────────────────────────────────────────────────────

def _reobserve_file(job):
    """Worker: add images_obs_<name>, mask_obs_<name>, frac_detected_<name> to every frame of one part file."""
    path, rc, ob, seed = job
    pix_arcsec = (rc["box_length_pc"] / rc["target_res"]) / ob["distance_pc"] * 206265.0
    beam_sigma = ob["beam_fwhm_arcsec"] / 2.3548 / pix_arcsec
    name, tag, n_img = None, ob["name"], 0
    with h5py.File(path, "r+") as h:
        name = h.attrs["sim_name"]
        for g in sorted(k for k in h if k.startswith("DD")):
            G_, frame = h[g], int(g[2:])
            obs, masks, fd = [], [], []
            for j in range(G_["images"].shape[0]):
                il, ia = int(G_["los"][j]), int(G_["aug_index"][j])
                T = G_[f"cube_{il}_{ia}"][()].astype(np.float32)
                v = G_[f"v_{il}"][()].astype(np.float64)
                Tb = gaussian_filter(T, sigma=(beam_sigma, beam_sigma, 0), mode="wrap")
                rng = np.random.default_rng([seed, zlib.crc32(name.encode()), frame, il, ia])
                o, mk = observed_moments(Tb + rng.normal(0.0, ob["noise_rms_K"], Tb.shape), v, rc,
                                         min_channels=ob["min_channels"])
                obs.append(o); masks.append(mk); fd.append(float(mk.mean()))
            for k, val in ((f"images_obs_{tag}", np.stack(obs)), (f"mask_obs_{tag}", np.stack(masks)),
                           (f"frac_detected_{tag}", np.array(fd))):
                if k in G_:
                    del G_[k]
                G_[k] = val
            n_img += len(obs)
        h.attrs[f"observe_{tag}"] = json.dumps(ob)
    return os.path.basename(path), n_img


def cmd_reobserve(args):
    """
    Re-observe every stored noise-free cube with the setup rc["observe"][args.obs]
    (beam, distance, noise, masking) and store the result next to the original
    images_obs. Parallel over part files with multiprocessing.
    """
    import multiprocessing as mp
    cfg, rc = json.load(open(args.config)), json.load(open(args.rt))
    ob = dict(rc["observe"][args.obs], name=args.obs)
    rcfg = rt_cfg(cfg, rc)
    files = [f for s in cfg["sims"] for f in part_files(rcfg, s["name"])]
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"reobserve '{args.obs}': {ob} on {len(files)} part files with {workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, (fn, n) in enumerate(pool.imap_unordered(_reobserve_file,
                                                        [(f, rc, ob, ob["seed"]) for f in files]), 1):
            print(f"  [{i:3d}/{len(files)}] {fn}: {n} images  ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────

RT_SAMPLE_FIELDS = {
    "v_window_kms":  "half-width of the RADMC-3D velocity window for this LOS (km/s)",
    "n_channels":    "number of velocity channels",
    "Tpeak_K":       "peak brightness temperature of the beam-convolved, noise-free cube (K)",
    "frac_detected": "fraction of pixels in the observation-pipeline emission mask",
}


def cmd_merge(args):
    cfg, rc = json.load(open(args.config)), json.load(open(args.rt))
    rcfg = rt_cfg(cfg, rc)
    sfx = f"_{args.obs}" if args.obs else ""          # which observed version to merge
    out = os.path.join(cfg["output_dir"], rc["output_name"][:-3] + sfx + ".h5")
    obs_key, mask_key = f"images_obs{sfx}", f"mask_obs{sfx}"
    sample_keys = {k: (f"{k}{sfx}" if k == "frac_detected" else k) for k in RT_SAMPLE_FIELDS}
    frames_rows, samples, sources, sim_rows, shape, cube_refs = [], [], [], [], None, []
    for sid, sim in enumerate(cfg["sims"]):
        files = part_files(rcfg, sim["name"])
        if not files:
            sim_rows.append((sid, sim, 0, np.nan)); continue
        groups_all = []                          # (frame group, part file), frames sorted across chunks
        for p in files:
            with h5py.File(p, "r") as h:
                groups_all += [(g, p) for g in h if g.startswith("DD")]
        groups_all.sort()
        n_frames_sim = len(groups_all)
        for g, p in groups_all:
            with h5py.File(p, "r") as h:
                G_ = h[g]; fid = len(frames_rows)
                frames_rows.append({"sim_id": sid, "frame": int(g[2:]), **dict(G_.attrs)})
                n = G_["images"].shape[0]
                shape = G_["images"].shape[1:]
                sources.append((p, g, n))
                for j in range(n):
                    cube_refs.append((os.path.basename(p), f"{g}/cube_{int(G_['los'][j])}_{int(G_['aug_index'][j])}"))
                    samples.append([sid, fid, int(g[2:])] +
                                   [int(G_[k][j]) for k in ("los", "aug_index", "shift_x", "shift_y", "rot90")] +
                                   [float(G_[k][j]) for k in ("Ms_los_mw", "Ms_los_vw")] +
                                   [float(G_[sample_keys[k]][j]) for k in RT_SAMPLE_FIELDS])
        ms = np.array([r["Ms"] for r in frames_rows if r["sim_id"] == sid])
        sim_rows.append((sid, sim, n_frames_sim, ms.mean() if len(ms) else np.nan))
        print(f"  {sim['name']:20s} frames={n_frames_sim:4d}  ({len(files)} part file(s))")
    if not samples:
        raise SystemExit("Nothing to merge.")
    S = np.array(samples, dtype=np.float64)
    sid_s, fid_s = S[:, 0].astype(np.int32), S[:, 1].astype(np.int64)
    sim_xi = np.array([s.get("xi", np.nan) for _, s, *_ in sim_rows])

    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as h:
        h.attrs["title"] = f"p79d {cfg['suite']} synthetic 13CO (RADMC-3D) dataset"
        h.attrs["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        h.attrs["generator"] = f"extract_sim_radmc.py (git {E.git_hash()})"
        h.attrs["config_json"], h.attrs["rt_json"] = json.dumps(cfg), json.dumps(rc)
        h.attrs["observe_version"] = (json.dumps(dict(rc["observe"][args.obs], name=args.obs)) if args.obs
                                      else "original extraction (beam/noise from rt_json, min_channels=1)")
        h.attrs["layout"] = ("Same layout and sample order conventions as the projected dataset "
                             "(extract_sim_data.py): images / samples / labels / frames / sims. "
                             "images are noise-free 13CO moments in training units; images_obs "
                             "and mask_obs are the noisy, masked observation-pipeline moments.")
        h.attrs["image_orientation"] = ("Identical to the projected dataset: LOS x -> [y, z], "
                                        "y -> [x, z], z -> [x, y]; centroid positive along +axis.")
        h.attrs["sound_speed_kms"] = sound_speed_kms(rc)
        n = len(samples)
        d  = h.create_dataset("images", (n, *shape), np.float32, chunks=(1, *shape))
        do = h.create_dataset("images_obs", (n, *shape), np.float32, chunks=(1, *shape))
        dm = h.create_dataset("mask_obs", (n, *shape[1:]), bool, chunks=(1, *shape[1:]))
        d.attrs["description"]  = ("noise-free, beam-convolved 13CO moments in training units: "
                                   "[W / <W>_image, v_c / c_s, sigma_v / c_s]")
        do.attrs["description"] = ("noisy 13CO moments from get_Observation_Data.compute_moment_maps: "
                                   "[W (K km/s), v_c (km/s), sigma_v (km/s)], NaN outside mask_obs")
        dm.attrs["description"] = "emission mask of the observation pipeline"
        i0 = 0
        for p, g, k in sources:
            with h5py.File(p, "r") as hp:
                d[i0:i0 + k] = hp[g]["images"][()]
                do[i0:i0 + k] = hp[g][obs_key][()]
                dm[i0:i0 + k] = hp[g][mask_key][()]
            i0 += k

        E.put(h, "samples/sim_id", sid_s, "index into sims/*")
        E.put(h, "samples/cube_file", np.array([c[0] for c in cube_refs], dtype=h5py.string_dtype()),
              "part file (in cube_dir) holding this sample's noise-free PPV cube")
        E.put(h, "samples/cube_key", np.array([c[1] for c in cube_refs], dtype=h5py.string_dtype()),
              "dataset path of the cube inside cube_file; velocity axis at <DDxxxx>/v_<los>")
        h.attrs["cube_dir"] = os.path.dirname(E.part_path(rcfg, cfg["sims"][0]["name"]))
        E.put(h, "samples/frame_id", fid_s, "index into frames/*")
        for j, (k, desc) in enumerate([("frame", "ENZO output number"), ("los", "0 = x, 1 = y, 2 = z"),
                                       ("aug_index", "augmentation copy"), ("shift_x", "periodic roll, image axis 0"),
                                       ("shift_y", "periodic roll, image axis 1"), ("rot90", "90-degree rotations")], start=2):
            E.put(h, f"samples/{k}", S[:, j].astype(np.int32), desc)
        for j, (k, desc) in enumerate(RT_SAMPLE_FIELDS.items(), start=10):
            E.put(h, f"rt/{k}", S[:, j], desc)
        E.put(h, "labels/xi", sim_xi[sid_s], "forcing solenoidal weight (ENZO DrivenFlowWeight)")
        E.put(h, "labels/Ms_los_mw", S[:, 8], E.LOS_FIELDS["Ms_los_mw"][0])
        E.put(h, "labels/Ms_los_vw", S[:, 9], E.LOS_FIELDS["Ms_los_vw"][0])
        for k, (desc, units) in E.FRAME_FIELDS.items():
            vals = np.array([r[k] for r in frames_rows])
            E.put(h, f"frames/{k}", vals, desc, units)
            if vals.ndim == 1:
                E.put(h, f"labels/{k}", vals[fid_s], desc, units)
        E.put(h, "frames/sim_id", np.array([r["sim_id"] for r in frames_rows], np.int32), "index into sims/*")
        E.put(h, "frames/frame", np.array([r["frame"] for r in frames_rows], np.int32), "ENZO output number")
        E.put(h, "sims/name", np.array([s["name"] for _, s, *_ in sim_rows], dtype=h5py.string_dtype()), "simulation")
        E.put(h, "sims/xi", sim_xi, "DrivenFlowWeight")
        E.put(h, "sims/n_frames", np.array([r[2] for r in sim_rows], np.int32), "frames extracted")
        E.put(h, "sims/Ms_mean", np.array([r[3] for r in sim_rows]), "mean of frames/Ms over extracted frames")
        h["subsets"], h["Ms_act"], h["xi"] = h["images"], h["labels/Ms"], h["labels/xi"]
        h["frame"], h["los"] = h["samples/frame"], h["samples/los"]
    os.replace(tmp, out)
    print(f"Wrote {out}: {n} images from {len(frames_rows)} frames")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--config", required=True); e.add_argument("--rt", required=True)
    e.add_argument("--index", type=int); e.add_argument("--sim")
    e.add_argument("--frames", type=int, nargs="*"); e.add_argument("--max_frames", type=int)
    e.add_argument("--workers", type=int, default=None)
    e.add_argument("--keep_cubes", default=None, help="also save the full-resolution PPV cubes here (testing)")
    e.add_argument("--chunk", type=int, default=0)
    e.add_argument("--n_chunks", type=int, default=1, help="split the frames of a sim over n jobs")
    m = sp.add_parser("merge")
    m.add_argument("--config", required=True); m.add_argument("--rt", required=True)
    m.add_argument("--obs", default=None, help="merge this re-observed version (from reobserve)")
    r = sp.add_parser("reobserve", help="new observed version from the stored cubes")
    r.add_argument("--config", required=True); r.add_argument("--rt", required=True)
    r.add_argument("--obs", required=True, help="name of a setup in rt['observe']")
    r.add_argument("--workers", type=int, default=None)
    args = p.parse_args()
    if args.cmd == "extract" and args.index is None and args.sim is None:
        p.error("extract needs --index or --sim")
    {"extract": cmd_extract, "merge": cmd_merge, "reobserve": cmd_reobserve}[args.cmd](args)


if __name__ == "__main__":
    main()
