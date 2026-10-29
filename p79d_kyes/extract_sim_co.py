"""
extract_sim_co.py
=================
RADMC-3D run 2 of the mach_grid suite (config: radmc_co_v2.json): several CO
lines (default 13CO and 12CO J=1-0) rendered from each snapshot, with the mean
H2 density drawn per frame so the dataset spans optically thin to thick gas.
Builds on extract_sim_radmc.py (run 1; views, binary I/O, moment code) and
extract_sim_data.py (frames, shifts, labels); images pair pixel-for-pixel with
the projected dataset.

Per frame:
  1. n(H2) = log-uniform draw seeded by (seed, simulation, frame);
  2. molecule, H2 (LVG only), velocity and temperature inputs; for LVG the level
     populations are computed once per frame and reused by every image;
  3. one image per (line, LOS) — all in parallel — with an FCRAO-like channel
     width and a per-LOS velocity window;
  4. per image and line: the shifted, block-averaged (128^2) noise-free cube is
     stored (cube_<tag>_<los>_<aug>, float16, K; v_<tag>_<los>, km/s) together with
     its noise-free moments in training units (images_<tag>: [W/<W>, v_c/c_s,
     sigma_v/c_s]; no beam, no noise). Telescope versions (beam, channel width,
     per-line noise, masking) are made later from the cubes: `reobserve`.

  python extract_sim_co.py extract   --config sims_mach_grid.json --rt radmc_co_v2.json --index 0 --chunk 0 --n_chunks 6
  python extract_sim_co.py reobserve --config sims_mach_grid.json --rt radmc_co_v2.json --obs fcrao500
  python extract_sim_co.py merge     --config sims_mach_grid.json --rt radmc_co_v2.json [--obs fcrao500]
"""
import argparse, datetime, glob, json, os, re, shutil, subprocess, tempfile, time, traceback, zlib
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter

import extract_sim_data as E
import extract_sim_radmc as R
import get_Observation_Data as G

AV_PER_NH2 = 1.0 / 9.4e20        # A_V per N(H2) [cm^-2] (Bohlin et al. 1978: N_H = 1.87e21 A_V)


# ─────────────────────────────────────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────────────────────────────────────

def draw_n_H2(rc, name, frame):
    s = rc["n_H2_sampling"]
    rng = np.random.default_rng([s["seed"], zlib.crc32(name.encode()), frame])
    if s["distribution"] == "fixed":
        return float(s["min_cm3"])
    return float(np.exp(rng.uniform(np.log(s["min_cm3"]), np.log(s["max_cm3"]))))


def mol_rc(rc, mol):
    """Per-molecule view of the config for extract_sim_radmc.read_image."""
    return {**rc, "rest_frequency_hz": mol["rest_frequency_hz"]}


def trim_lamda(src, dst, n_keep):
    """Copy a LAMDA file keeping the lowest n_keep levels and the transitions among them."""
    L = open(src).read().splitlines()
    out, i = [], 0
    def take_header(n):            # copy n lines verbatim
        nonlocal i
        out.extend(L[i:i + n]); i += n
    take_header(5)                                         # !MOLECULE, name, !WEIGHT, weight, !NUMBER OF ENERGY LEVELS
    n_lev = int(L[i].split()[0]); out.append(str(min(n_keep, n_lev))); i += 1
    take_header(1)
    out.extend(L[i:i + min(n_keep, n_lev)]); i += n_lev
    def block(up_col):                                     # count line, header, rows filtered on (up, low) <= n_keep
        nonlocal i
        hdr_count = L[i - 1]; n = int(L[i].split()[0]); i += 1
        hdr = L[i]; i += 1
        rows = [r for r in L[i:i + n] if int(r.split()[up_col]) <= n_keep and int(r.split()[up_col + 1]) <= n_keep]
        i += n
        rows = [f"{k + 1:5d} " + " ".join(r.split()[1:]) for k, r in enumerate(rows)]
        out.extend([str(len(rows)), hdr] + rows)
    take_header(1); block(1)                               # radiative transitions
    take_header(1); n_part = int(L[i].split()[0]); take_header(1)
    for _ in range(n_part):
        take_header(3)                                     # !COLLISIONS BETWEEN, partner, !NUMBER OF COLL TRANS
        n_line = i; i += 1
        take_header(4)                                     # !NUMBER OF COLL TEMPS, n, !COLL TEMPS, temps
        n = int(L[n_line].split()[0]); hdr = L[i]; i += 1
        rows = [r for r in L[i:i + n] if int(r.split()[1]) <= n_keep and int(r.split()[2]) <= n_keep]
        i += n
        out.insert(len(out) - 4, str(len(rows)))
        out.extend([hdr] + [f"{k + 1:5d} " + " ".join(r.split()[1:]) for k, r in enumerate(rows)])
    out.extend(L[i:])
    open(dst, "w").write("\n".join(out) + "\n")


def write_inputs(d, rho, v_kms, rc, n_H2):
    N = rho.shape[0]
    L = rc["box_length_pc"] * R.PC_CM
    edges = " ".join(f"{x:.10e}" for x in np.linspace(0.0, L, N + 1))
    with open(os.path.join(d, "amr_grid.inp"), "w") as f:
        f.write(f"1\n0\n1\n0\n1 1 1\n{N} {N} {N}\n{edges}\n{edges}\n{edges}\n")
    nH2 = n_H2 * rho / rho.mean()
    lvg = rc["lines_mode"] != 1
    lines = [f"2\n{len(rc['molecules'])}\n"]
    for mol in rc["molecules"]:
        R._write_binp(os.path.join(d, f"numberdens_{mol['name']}.binp"), [mol["abundance"] * nH2])
        if lvg and rc.get("lvg_max_levels"):
            trim_lamda(mol["file"], os.path.join(d, f"molecule_{mol['name']}.inp"), rc["lvg_max_levels"])
        else:
            shutil.copy(mol["file"], os.path.join(d, f"molecule_{mol['name']}.inp"))
        lines.append(f"{mol['name']}    leiden    0    0    {2 if lvg else 0}\n" + ("p-h2\no-h2\n" if lvg else ""))
    if lvg:
        opr = rc["ortho_para_ratio"]
        R._write_binp(os.path.join(d, "numberdens_p-h2.binp"), [nH2 / (1 + opr)])
        R._write_binp(os.path.join(d, "numberdens_o-h2.binp"), [nH2 * opr / (1 + opr)])
    open(os.path.join(d, "lines.inp"), "w").write("".join(lines))
    R._write_binp(os.path.join(d, "gas_velocity.binp"), [vi * 1e5 for vi in v_kms])
    R._write_binp(os.path.join(d, "gas_temperature.binp"), [np.full(rho.shape, rc["T_K"])])
    with open(os.path.join(d, "radmc3d.inp"), "w") as f:
        f.write(f"lines_mode = {rc['lines_mode']}\nnphot_scat = 0\ncamera_tracemode = 1\ncamera_nrrefine = 1\n"
                f"tgas_eq_tdust = 0\nlines_show_pictograms = 0\n")
        if lvg:
            f.write(f"lines_nonlte_maxiter = {rc['lvg_maxiter']}\nlines_nonlte_convcrit = {rc['lvg_convcrit']}\n")
    with open(os.path.join(d, "wavelength_micron.inp"), "w") as f:
        f.write(f"1\n{R.C_CGS / rc['molecules'][0]['rest_frequency_hz'] * 1e4:.6e}\n")


def prepare_populations(d, rc):
    """LVG: compute the level populations once (calcpop writepop) so every image reuses them."""
    if rc["lines_mode"] == 1:
        return
    r = subprocess.run([rc["radmc3d_binary"], "calcpop", "writepop"], cwd=d, capture_output=True, text=True)
    if r.returncode != 0 or not glob.glob(os.path.join(d, "levelpop_*")):
        raise RuntimeError("RADMC-3D calcpop failed:\n" + (r.stdout + r.stderr)[-2000:])
    # images read the stored populations (lines_mode 50); with lines_mode 3 every
    # image call would solve the LVG populations again (identical result, 6x the cost)
    p = os.path.join(d, "radmc3d.inp")
    txt = open(p).read()                    # read before reopening: open(p, "w") truncates
    if not re.search(r"lines_mode = \d+", txt):
        raise RuntimeError(f"no lines_mode in {p}")
    open(p, "w").write(re.sub(r"lines_mode = \d+", "lines_mode = 50", txt))


def launch_image(d, rc, view, W, nch, imol):
    """Start RADMC-3D for molecule number imol (1-based, order of rc['molecules']) and one view."""
    N, au = rc["npix_rt"], rc["box_length_pc"] * R.PC_CM / R.AU_CM
    cmd = [rc["radmc3d_binary"], "image", "npix", str(N), "sizeau", f"{au:.8e}",
           "pointau", f"{au/2:.8e}", f"{au/2:.8e}", f"{au/2:.8e}",
           "incl", str(view["incl"]), "phi", str(view["phi"]), "imolspec", str(imol), "iline", "1",
           "widthkms", f"{2 * W:.6f}", "linenlam", str(nch), "vkms", "0", "imageunform"]
    return subprocess.Popen(cmd, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def velocity_window(v_los_kms, rc):
    """Half-width and channel count: line (velocity quantile + thermal margin of the lightest molecule) + edge channels."""
    dv  = rc["channel_width_kms"]
    sth = max(np.sqrt(R.KB * rc["T_K"] / (m["mass_amu"] * R.MH)) / 1e5 for m in rc["molecules"])
    q   = np.quantile(np.abs(v_los_kms - v_los_kms.mean()), rc["velocity_quantile"])
    half = int(np.ceil((q + rc["velocity_margin_thermal_sigmas"] * sth + rc["edge_channels"] * dv) / dv))
    return half * dv, 2 * half + 1


# ─────────────────────────────────────────────────────────────────────────────
# LABELS (copied from the projected dataset)
# ─────────────────────────────────────────────────────────────────────────────

class LabelSource:
    """Frame labels from the projected dataset, keyed by (sim name, frame)."""
    def __init__(self, path):
        self.rows = {}
        if not path or not os.path.exists(path):
            return
        with h5py.File(path, "r") as f:
            names = f["sims/name"][:].astype(str)
            sid, fr = f["frames/sim_id"][:], f["frames/frame"][:]
            cols = {k: f[f"frames/{k}"][:] for k in E.FRAME_FIELDS}
        for i, (s, fm) in enumerate(zip(sid, fr)):
            self.rows[(names[s], int(fm))] = {k: v[i] for k, v in cols.items()}

    def get(self, name, frame):
        return self.rows.get((name, frame))


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT
# ─────────────────────────────────────────────────────────────────────────────

def cmd_extract(args):
    cfg, rc = json.load(open(args.config)), json.load(open(args.rt))
    sim = cfg["sims"][args.index] if args.sim is None else next(s for s in cfg["sims"] if s["name"] == args.sim)
    ex, name = cfg["extraction"], sim["name"]
    sim_dir = os.path.join(cfg["data_root"], name)
    cs_code, cs_kms = sim.get("sound_speed", 1.0), R.sound_speed_kms(rc)
    rcfg = R.rt_cfg(cfg, rc)
    out = R.part_file(rcfg, name, args.chunk, args.n_chunks)
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    labels = LabelSource(rc.get("labels_from"))
    mols = rc["molecules"]

    frames = [f for f in E.frame_dirs(sim_dir) if f >= ex.get("min_frame", 0)]
    if args.n_chunks > 1:
        frames = frames[args.chunk::args.n_chunks]
    if args.frames:
        frames = [f for f in frames if f in set(args.frames)]
    if args.max_frames:
        frames = frames[:args.max_frames]
    with h5py.File(out, "a") as h:
        done = {int(g[2:]) for g in h if g.startswith("DD")}
        h.attrs["sim_name"], h.attrs["sim_json"], h.attrs["rt_json"] = name, json.dumps(sim), json.dumps(rc)
    todo = [f for f in frames if f not in done]
    print(f"{name}: {len(frames)} frames, {len(done)} done, {len(todo)} to do → {out}\n"
          f"  lines {[m['tag'] for m in mols]}, lines_mode {rc['lines_mode']}, c_s = {cs_kms:.4f} km/s", flush=True)

    tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    factor = rc["npix_rt"] // rc["target_res"]
    kg, skipped, t_start = None, [], time.time()
    for i, frame in enumerate(todo):
        t0 = time.time()
        q = labels.get(name, frame)
        if q is not None and q["Ms"] > rc["ms_max_frame"]:
            skipped.append((frame, f"Ms {q['Ms']:.1f} > ms_max_frame")); continue
        if not E.frame_readable(sim_dir, frame):
            skipped.append((frame, "unreadable")); print(f"  DD{frame:04d} skipped: unreadable", flush=True)
            continue
        work = tempfile.mkdtemp(prefix=f"co_{name}_{frame}_", dir=tmp_root)
        try:
            rho, v, t = E.load_frame(sim_dir, frame)
            N = rho.shape[0]
            if q is None:                                  # frame missing from the projected dataset
                kg = kg if (kg is not None and kg.N == N) else E.KGrid(N)
                q = E.frame_scalars(rho, v, kg, cs_code, workers); q["time"] = t
                if q["Ms"] > rc["ms_max_frame"]:
                    skipped.append((frame, f"Ms {q['Ms']:.1f} > ms_max_frame")); continue
            n_H2 = draw_n_H2(rc, name, frame)
            v_kms = [(vi - vi.mean()) / cs_code * cs_kms for vi in v]
            write_inputs(work, rho, v_kms, rc, n_H2)
            t_pop = time.time(); prepare_populations(work, rc); t_pop = time.time() - t_pop

            procs, wins = {}, {}
            for il, ax in enumerate(E.LOS_AXES):
                wins[ax] = velocity_window(v_kms[il], rc)
                for im, mol in enumerate(mols, start=1):
                    d = os.path.join(work, f"{mol['tag']}_{ax}"); os.makedirs(d)
                    for fn in os.listdir(work):
                        if os.path.isfile(os.path.join(work, fn)):
                            os.symlink(os.path.join(work, fn), os.path.join(d, fn))
                    procs[(mol["tag"], ax)] = launch_image(d, rc, R.RT_VIEW[ax], *wins[ax], im)
            for key, pr in procs.items():
                log, _ = pr.communicate()
                if pr.returncode != 0 or not os.path.exists(os.path.join(work, f"{key[0]}_{key[1]}", "image.bout")):
                    raise RuntimeError(f"RADMC-3D failed for {key}:\n{log[-2000:]}")

            aug = E.augment_params(ex, name, frame, N)
            imgs = {m["tag"]: [] for m in mols}
            cubes, meta = {}, []
            for il, ax in enumerate(E.LOS_AXES):
                ls = E.los_scalars(rho, v, ax, cs_code)
                col = rho.sum(axis=il) / N                        # Sigma / <Sigma> at native res
                av = n_H2 * rc["box_length_pc"] * R.PC_CM * col / (rho.mean()) * AV_PER_NH2
                Ts = {m["tag"]: R.read_image(os.path.join(work, f"{m['tag']}_{ax}"), mol_rc(rc, m), R.RT_VIEW[ax])
                      for m in mols}
                for _, ia, dx, dy, k90 in (a for a in aug if a[0] == il):
                    tpk = []
                    for m in mols:
                        T, vax = Ts[m["tag"]]
                        Tr = R.block_mean(np.rot90(np.roll(T, (dx, dy), axis=(0, 1)), k90, axes=(0, 1)), factor)
                        cubes[f"cube_{m['tag']}_{il}_{ia}"] = Tr.astype(np.float16)
                        cubes[f"v_{m['tag']}_{il}"] = vax.astype(np.float32)
                        imgs[m["tag"]].append(R.clean_moments(Tr, vax, cs_kms))
                        tpk.append(float(Tr.max()))
                    avs = np.rot90(np.roll(av, (dx, dy), axis=(0, 1)), k90)
                    meta.append([il, ia, dx, dy, k90, ls["Ms_los_mw"], ls["Ms_los_vw"], wins[ax][0], wins[ax][1],
                                 float(np.median(avs)), float(np.mean(avs > 5.0))] + tpk)
            del rho, v

            with h5py.File(out, "a") as h:
                g = h.create_group(f"DD{frame:04d}")
                for m in mols:
                    g[f"images_{m['tag']}"] = np.stack(imgs[m["tag"]])
                for k, c in cubes.items():
                    if k.startswith("cube_"):
                        g.create_dataset(k, data=c, chunks=(c.shape[0], c.shape[1], min(16, c.shape[2])), compression="lzf")
                    else:
                        g[k] = c
                M = np.array(meta, dtype=np.float64)
                for j, k in enumerate(["los", "aug_index", "shift_x", "shift_y", "rot90"]):
                    g[k] = M[:, j].astype(np.int32)
                for j, k in enumerate(["Ms_los_mw", "Ms_los_vw", "v_window_kms", "n_channels", "AV_median", "AV_frac_gt5"], start=5):
                    g[k] = M[:, j]
                for j, m in enumerate(mols, start=11):
                    g[f"Tpeak_{m['tag']}"] = M[:, j]
                for k, val in q.items():
                    g.attrs[k] = val
                g.attrs["n_H2"] = n_H2
            rem = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1)
            print(f"  DD{frame:04d} Ms={q['Ms']:6.3f} n={n_H2:7.1f} AV~{np.median(M[:, 9]):5.1f} "
                  f"nch={'/'.join(str(wins[a][1]) for a in E.LOS_AXES)} "
                  + " ".join(f"Tpk_{m['tag']}={max(M[:, 11 + j]):4.2f}" for j, m in enumerate(mols))
                  + f"  pop {t_pop:4.0f}s  {time.time() - t0:5.0f}s (~{rem / 60:.0f} min left)", flush=True)
        except Exception as e:
            skipped.append((frame, repr(e))); print(f"  DD{frame:04d} FAILED: {e!r}", flush=True); traceback.print_exc()
        finally:
            shutil.rmtree(work, ignore_errors=True)
    with h5py.File(out, "a") as h:
        h.attrs["skipped"] = json.dumps(json.loads(h.attrs.get("skipped", "[]")) + skipped)
    print(f"{name}: finished, {len(skipped)} skipped this run", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# REOBSERVE
# ─────────────────────────────────────────────────────────────────────────────

def rebin_channels(T, v, dv_new):
    """Flux-conserving resampling of the velocity axis onto channels of width dv_new (centred on 0)."""
    dv = v[1] - v[0]
    if abs(dv_new - dv) < 1e-6:
        return T, v
    edges = np.concatenate([[v[0] - dv / 2], v + dv / 2])
    cum = np.concatenate([np.zeros(T.shape[:2] + (1,)), np.cumsum(T, axis=2) * dv], axis=2)
    half = int(np.floor((v[-1] + dv / 2) / dv_new - 0.5))
    v_new = np.arange(-half, half + 1) * dv_new
    e_new = np.concatenate([v_new - dv_new / 2, [v_new[-1] + dv_new / 2]])
    c_new = np.stack([np.interp(e_new, edges, c) for c in cum.reshape(-1, cum.shape[2])]).reshape(T.shape[:2] + (len(e_new),))
    return np.diff(c_new, axis=2) / dv_new, v_new


def _reobserve_file(job):
    path, rc, ob = job
    pix = (rc["box_length_pc"] / rc["target_res"]) / ob["distance_pc"] * 206265.0
    beam_sigma = ob["beam_fwhm_arcsec"] / 2.3548 / pix
    tag, n = ob["name"], 0
    with h5py.File(path, "r+") as h:
        sim = h.attrs["sim_name"]
        for g in sorted(k for k in h if k.startswith("DD")):
            G_, frame = h[g], int(g[2:])
            for m in rc["molecules"]:
                mt = m["tag"]; obs, masks = [], []
                for j in range(G_["los"].shape[0]):
                    il, ia = int(G_["los"][j]), int(G_["aug_index"][j])
                    T = G_[f"cube_{mt}_{il}_{ia}"][()].astype(np.float32)
                    Tb, v = rebin_channels(gaussian_filter(T, (beam_sigma, beam_sigma, 0), mode="wrap"),
                                           G_[f"v_{mt}_{il}"][()].astype(np.float64), ob["channel_width_kms"])
                    rng = np.random.default_rng([ob["seed"], zlib.crc32(sim.encode()), frame, il, ia, zlib.crc32(mt.encode())])
                    o, mk = R.observed_moments(Tb + rng.normal(0, ob["noise_rms_K"][mt], Tb.shape), v, rc,
                                               min_channels=G.min_channels_for(ob["min_channels"], ob["channel_width_kms"]))
                    obs.append(o); masks.append(mk)
                for k, val in ((f"images_obs_{mt}_{tag}", np.stack(obs)), (f"mask_obs_{mt}_{tag}", np.stack(masks))):
                    if k in G_:
                        del G_[k]
                    G_[k] = val
                n += len(obs)
        h.attrs[f"observe_{tag}"] = json.dumps(ob)
    return os.path.basename(path), n


def cmd_reobserve(args):
    import multiprocessing as mp
    cfg, rc = json.load(open(args.config)), json.load(open(args.rt))
    ob = dict(rc["observe"][args.obs], name=args.obs)
    rcfg = R.rt_cfg(cfg, rc)
    files = [f for s in cfg["sims"] for f in R.part_files(rcfg, s["name"])]
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    rc = {**rc, "mask_sigma": ob["mask_sigma"]}
    print(f"reobserve '{args.obs}': {ob} on {len(files)} part files, {workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, (fn, k) in enumerate(pool.imap_unordered(_reobserve_file, [(f, rc, ob) for f in files]), 1):
            print(f"  [{i:3d}/{len(files)}] {fn}: {k} images ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_FIELDS = {"v_window_kms": "half-width of the velocity window (km/s)", "n_channels": "velocity channels",
                 "AV_median": "median A_V of the image (N(H2) / 9.4e20 cm^-2)", "AV_frac_gt5": "fraction of pixels with A_V > 5"}


def cmd_merge(args):
    cfg, rc = json.load(open(args.config)), json.load(open(args.rt))
    rcfg, mols = R.rt_cfg(cfg, rc), [m["tag"] for m in rc["molecules"]]
    sfx = f"_{args.obs}" if args.obs else ""
    out = os.path.join(cfg["output_dir"], rc["output_name"][:-3] + sfx + ".h5")
    rows, samples, sources, sim_rows, shape, cube_refs = [], [], [], [], None, []
    for sid, sim in enumerate(cfg["sims"]):
        files = R.part_files(rcfg, sim["name"])
        groups = []
        for p in files:
            with h5py.File(p, "r") as h:
                groups += [(g, p) for g in h if g.startswith("DD")]
        for g, p in sorted(groups):
            with h5py.File(p, "r") as h:
                G_ = h[g]; fid = len(rows)
                rows.append({"sim_id": sid, "frame": int(g[2:]), **dict(G_.attrs)})
                k = G_["los"].shape[0]; shape = G_[f"images_{mols[0]}"].shape[1:]
                sources.append((p, g, k))
                for j in range(k):
                    cube_refs.append((os.path.basename(p), g, int(G_["los"][j]), int(G_["aug_index"][j])))
                    samples.append([sid, fid, int(g[2:])] + [int(G_[c][j]) for c in ("los", "aug_index", "shift_x", "shift_y", "rot90")]
                                   + [float(G_[c][j]) for c in ("Ms_los_mw", "Ms_los_vw", *SAMPLE_FIELDS)]
                                   + [float(G_[f"Tpeak_{m}"][j]) for m in mols])
        ms = [r["Ms"] for r in rows if r["sim_id"] == sid]
        sim_rows.append((sid, sim, len(groups), np.mean(ms) if ms else np.nan))
    if not samples:
        raise SystemExit("Nothing to merge.")
    S = np.array(samples, dtype=np.float64); n = len(S)
    sid_s, fid_s = S[:, 0].astype(np.int32), S[:, 1].astype(np.int64)
    sim_xi = np.array([s.get("xi", np.nan) for _, s, *_ in sim_rows])
    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as h:
        h.attrs["title"] = f"p79d {cfg['suite']} synthetic CO ({', '.join(mols)}; RADMC-3D run 2)"
        h.attrs["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        h.attrs["generator"] = f"extract_sim_co.py (git {E.git_hash()})"
        h.attrs["config_json"], h.attrs["rt_json"] = json.dumps(cfg), json.dumps(rc)
        h.attrs["sound_speed_kms"] = R.sound_speed_kms(rc)
        h.attrs["observe_version"] = json.dumps(dict(rc["observe"][args.obs], name=args.obs)) if args.obs else "noise-free only"
        h.attrs["cube_dir"] = os.path.dirname(E.part_path(rcfg, cfg["sims"][0]["name"]))
        h.attrs["layout"] = ("Same conventions as the projected dataset. images_<line>: noise-free moments in training "
                             "units, no beam. With --obs: images_obs_<line> [K km/s, km/s, km/s] (NaN masked) and "
                             "mask_obs_<line> for that telescope setup. 'images' = images_<first line>.")
        keys = [f"images_{m}" for m in mols] + ([f"images_obs_{m}" for m in mols] if args.obs else [])
        ds = {}
        for k in keys:
            ds[k] = h.create_dataset(k, (n, *shape), np.float32, chunks=(1, *shape))
        if args.obs:
            for m in mols:
                ds[f"mask_obs_{m}"] = h.create_dataset(f"mask_obs_{m}", (n, *shape[1:]), bool, chunks=(1, *shape[1:]))
        i0 = 0
        for p, g, k in sources:
            with h5py.File(p, "r") as hp:
                for key, d in ds.items():
                    src = key if not args.obs or not key.startswith(("images_obs", "mask_obs")) else f"{key}{sfx}"
                    d[i0:i0 + k] = hp[g][src][()]
            i0 += k
        h["images"] = h[f"images_{mols[0]}"]
        E.put(h, "samples/sim_id", sid_s, "index into sims/*"); E.put(h, "samples/frame_id", fid_s, "index into frames/*")
        for j, (k, d) in enumerate([("frame", "ENZO output"), ("los", "0=x,1=y,2=z"), ("aug_index", "augmentation copy"),
                                    ("shift_x", "roll, axis 0"), ("shift_y", "roll, axis 1"), ("rot90", "rotations")], start=2):
            E.put(h, f"samples/{k}", S[:, j].astype(np.int32), d)
        E.put(h, "samples/cube_file", np.array([c[0] for c in cube_refs], dtype=h5py.string_dtype()), "part file holding the cubes")
        E.put(h, "samples/cube_group", np.array([c[1] for c in cube_refs], dtype=h5py.string_dtype()),
              "frame group; cubes at cube_<line>_<los>_<aug>, velocity axes at v_<line>_<los>")
        E.put(h, "labels/xi", sim_xi[sid_s], "forcing solenoidal weight")
        E.put(h, "labels/Ms_los_mw", S[:, 8], E.LOS_FIELDS["Ms_los_mw"][0]); E.put(h, "labels/Ms_los_vw", S[:, 9], E.LOS_FIELDS["Ms_los_vw"][0])
        for j, (k, d) in enumerate(SAMPLE_FIELDS.items(), start=10):
            E.put(h, f"rt/{k}", S[:, j], d)
        for j, m in enumerate(mols, start=14):
            E.put(h, f"rt/Tpeak_{m}", S[:, j], f"peak T_B of the noise-free {m} cube (K)")
        n_H2 = np.array([r["n_H2"] for r in rows])
        E.put(h, "frames/n_H2", n_H2, "mean H2 density used for the radiative transfer (cm^-3)")
        E.put(h, "labels/n_H2", n_H2[fid_s], "mean H2 density of the frame (cm^-3)")
        for k, (d, u) in E.FRAME_FIELDS.items():
            vals = np.array([r[k] for r in rows]); E.put(h, f"frames/{k}", vals, d, u)
            if vals.ndim == 1:
                E.put(h, f"labels/{k}", vals[fid_s], d, u)
        E.put(h, "frames/sim_id", np.array([r["sim_id"] for r in rows], np.int32), "index into sims/*")
        E.put(h, "frames/frame", np.array([r["frame"] for r in rows], np.int32), "ENZO output")
        E.put(h, "sims/name", np.array([s["name"] for _, s, *_ in sim_rows], dtype=h5py.string_dtype()), "simulation")
        E.put(h, "sims/xi", sim_xi, "DrivenFlowWeight")
        E.put(h, "sims/n_frames", np.array([r[2] for r in sim_rows], np.int32), "frames extracted")
        E.put(h, "sims/Ms_mean", np.array([r[3] for r in sim_rows]), "mean Ms of extracted frames")
        h["Ms_act"], h["xi"], h["frame"], h["los"] = h["labels/Ms"], h["labels/xi"], h["samples/frame"], h["samples/los"]
    os.replace(tmp, out)
    print(f"Wrote {out}: {n} images from {len(rows)} frames")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    for a in ("--config", "--rt"):
        e.add_argument(a, required=True)
    e.add_argument("--index", type=int); e.add_argument("--sim")
    e.add_argument("--frames", type=int, nargs="*"); e.add_argument("--max_frames", type=int)
    e.add_argument("--chunk", type=int, default=0); e.add_argument("--n_chunks", type=int, default=1)
    e.add_argument("--workers", type=int)
    r = sp.add_parser("reobserve")
    for a in ("--config", "--rt", "--obs"):
        r.add_argument(a, required=True)
    r.add_argument("--workers", type=int)
    m = sp.add_parser("merge")
    m.add_argument("--config", required=True); m.add_argument("--rt", required=True); m.add_argument("--obs")
    args = p.parse_args()
    if args.cmd == "extract" and args.index is None and args.sim is None:
        p.error("extract needs --index or --sim")
    {"extract": cmd_extract, "reobserve": cmd_reobserve, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
