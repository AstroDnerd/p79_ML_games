"""
extract_sim_data.py
===================
Build the p79d training dataset directly from the raw ENZO outputs of a
turbulence suite (default: the 256^3 "mach_grid" suite), without the
p49d_turb_sims machinery.

For every readable frame of every simulation this computes, from the native
3D fields (density, velocity):

  * projected moment maps along x, y, z — column density, density-weighted
    velocity centroid and density-weighted velocity dispersion — all three
    from the same line-of-sight sums, so the channels are spatially aligned;
  * per-frame turbulence labels: sonic Mach number, Helmholtz compressive
    fraction chi (from v and sqrt(rho) v) with shell spectra, dilatation /
    vorticity, log-density moments and the density-variance parameter b;
  * the forcing parameter xi (ENZO DrivenFlowWeight, 1 = solenoidal).

Everything about the suite (paths, simulation names, forcing parameters) lives
in a JSON config, never in this file.

Sub-commands
------------
  config   scan a suite directory + ENZO parameter files and write the JSON
               python extract_sim_data.py config --data_root /anvil/.../256_mach_grid \\
                      --out sims_mach_grid.json
  extract  process one simulation into a part file (resumable; run one per sim)
               python extract_sim_data.py extract --config sims_mach_grid.json --index 0
  merge    combine the part files into the final dataset HDF5
               python extract_sim_data.py merge --config sims_mach_grid.json

The merged file documents every dataset in its HDF5 attributes; see
write_merged() for the layout.
"""

import argparse, datetime, glob, json, os, re, subprocess, time, traceback, zlib
import numpy as np
import h5py

DATASET_VERSION = "v1"
LOS_AXES = "xyz"
CHANNELS = ["column_density", "velocity_centroid", "velocity_dispersion"]
ENZO_PARAMS = {  # parameter-file key -> (json key, type)
    "DrivenFlowWeight":     ("xi", float),
    "DrivenFlowVelocity":   ("driving_velocity", float),
    "DrivenFlowProfile":    ("driving_profile", int),
    "DrivenFlowAutoCorrl":  ("driving_autocorrelation", float),
    "DrivenFlowSeed":       ("driving_seed", int),
    "dtDataDump":           ("dt_data_dump", float),
    "TopGridDimensions":    ("resolution", int),
    "IsothermalSoundSpeed": ("sound_speed", float),
    "HydroMethod":          ("hydro_method", int),
    "SelfGravity":          ("self_gravity", int),
    "Gamma":                ("gamma", float),
}


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

def frame_dirs(sim_dir):
    """Sorted frame numbers that have a DDxxxx directory."""
    out = []
    for d in glob.glob(os.path.join(sim_dir, "DD[0-9][0-9][0-9][0-9]")):
        out.append(int(os.path.basename(d)[2:]))
    return sorted(out)


def frame_path(sim_dir, frame):
    return os.path.join(sim_dir, f"DD{frame:04d}", f"data{frame:04d}")


def frame_readable(sim_dir, frame):
    p = frame_path(sim_dir, frame)
    return all(os.access(q, os.R_OK) for q in (p, p + ".hierarchy", p + ".cpu0000"))


def read_enzo_params(param_file):
    out = {}
    with open(param_file) as f:
        for line in f:
            if "=" not in line:
                continue
            key, val = (s.strip() for s in line.split("=", 1))
            if key in ENZO_PARAMS and ENZO_PARAMS[key][0] not in out:
                name, typ = ENZO_PARAMS[key]
                out[name] = typ(val.split()[0])      # first component for vectors
    return out


def cmd_config(args):
    sims = []
    for sim_dir in sorted(glob.glob(os.path.join(args.data_root, args.pattern))):
        if not os.path.isdir(sim_dir):
            continue
        name   = os.path.basename(sim_dir)
        frames = frame_dirs(sim_dir)
        ok     = [f for f in frames if frame_readable(sim_dir, f)]
        entry  = {"name": name, "n_frame_dirs": len(frames), "n_readable_at_config": len(ok),
                  "first_frame": frames[0] if frames else None,
                  "last_frame": frames[-1] if frames else None}
        if ok:
            entry.update(read_enzo_params(frame_path(sim_dir, ok[len(ok) // 2])))
            entry["xi_source"] = "DrivenFlowWeight in the ENZO parameter file"
        else:
            m = re.match(r"xi_([0-9.]+)_", name)
            entry["xi"] = float(m.group(1)) if m else None
            entry["xi_source"] = "simulation name (no readable parameter file at config time)"
            entry["note"] = "no readable frames when config was written; rerun config later"
        sims.append(entry)
        print(f"  {name:20s} frames={len(frames):4d} readable={len(ok):4d} xi={entry.get('xi')}")

    cfg = {
        "suite": args.suite,
        "description": ("Driven isothermal hydrodynamic turbulence, ENZO PPM, periodic unit box, "
                        "no self-gravity. xi = DrivenFlowWeight (solenoidal weight of the "
                        "forcing: 1 = purely solenoidal, 0 = purely compressive). Simulation "
                        "names are labels only; the 'mach' part is NOT the sonic Mach number."),
        "data_root": os.path.abspath(args.data_root),
        "code": "enzo",
        "box_length": 1.0,
        "output_dir": args.output_dir,
        "output_name": f"p79d_{args.suite}_{DATASET_VERSION}.h5",
        "extraction": {
            "target_res": 128,          # images are block-reduced from native to this
            "n_aug_per_frame": 1,       # random periodic-shift copies per (frame, LOS)
            "random_shift": True,       # periodic roll before block reduction
            "random_rot90": False,
            "min_frame": 1,             # DD0000 is the uniform initial condition
            "seed": 20260930,
            "store_native_res": False,  # also keep the un-reduced maps (4x the size)
            "spectrum_shells": "integer |k| in units of 2 pi / L, k = 1 .. N/2",
        },
        "sims": sims,
    }
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote {args.out} ({len(sims)} sims)")


# ═════════════════════════════════════════════════════════════════════════════
# PHYSICS
# ═════════════════════════════════════════════════════════════════════════════

def load_frame(sim_dir, frame):
    """rho, vx, vy, vz as float64 arrays indexed [x, y, z]; plus code time."""
    import yt
    yt.set_log_level(40)
    ds = yt.load(frame_path(sim_dir, frame))
    cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    rho = np.asarray(cg[("gas", "density")].v, dtype=np.float64)
    v   = [np.asarray(cg[("gas", f"velocity_{a}")].v, dtype=np.float64) for a in "xyz"]
    t   = float(ds.current_time.v)
    del cg, ds
    return rho, v, t


class KGrid:
    """Wavenumber grid for an N^3 periodic box, cached across frames."""
    def __init__(self, N):
        k1 = np.fft.fftfreq(N) * N
        self.KX, self.KY, self.KZ = np.meshgrid(k1, k1, k1, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2
        self.K2[0, 0, 0] = 1.0                       # avoid 0/0; k = 0 excluded below
        self.shell = np.rint(np.sqrt(self.K2)).astype(np.int32).ravel()
        self.shell[0] = 0
        self.N = N


def helmholtz(fields, kg, workers):
    """Compressive / solenoidal power of a periodic vector field (k = 0 dropped)."""
    import scipy.fft as sfft
    fx, fy, fz = (sfft.fftn(f, workers=workers) for f in fields)
    kdotf = (kg.KX * fx + kg.KY * fy + kg.KZ * fz) / kg.K2
    Pt = np.abs(fx)**2 + np.abs(fy)**2 + np.abs(fz)**2
    del fx, fy, fz
    Pc = (kg.KX**2 + kg.KY**2 + kg.KZ**2) * np.abs(kdotf)**2
    del kdotf
    Pc[0, 0, 0] = 0.0
    Pt[0, 0, 0] = 0.0
    Ps = Pt - Pc
    nb = kg.N // 2
    out = {
        "Ec": Pc.sum(), "Es": Ps.sum(),
        "Ec_k": np.bincount(kg.shell, Pc.ravel(), minlength=kg.N)[1:nb + 1],
        "Es_k": np.bincount(kg.shell, Ps.ravel(), minlength=kg.N)[1:nb + 1],
        "theta2": (kg.K2 * Pc).sum(), "omega2": (kg.K2 * Ps).sum(),
    }
    out["chi"] = out["Ec"] / (out["Ec"] + out["Es"])
    return out


def frame_scalars(rho, v, kg, cs, workers):
    """All per-frame 3D labels. Keys match FRAME_FIELDS."""
    rho0 = rho.mean()
    q = {}
    vstd = [vi.std() for vi in v]
    q["Ms"]      = np.sqrt(sum(s**2 for s in vstd)) / cs          # = Ms_act of the old dataset
    q["Ms_mw"]   = np.sqrt(sum(np.average((vi - np.average(vi, weights=rho))**2, weights=rho)
                               for vi in v)) / cs
    q["v_mean"]  = np.array([vi.mean() for vi in v])
    q["v_std"]   = np.array(vstd)

    hv = helmholtz([vi - vi.mean() for vi in v], kg, workers)
    sq = np.sqrt(rho / rho0)
    hw = helmholtz([sq * (vi - vi.mean()) for vi in v], kg, workers)
    for tag, h in (("v", hv), ("w", hw)):
        for k in ("Ec", "Es", "chi"):
            q[f"{k}_{tag}"] = h[k]
        q[f"Ec_{tag}_k"], q[f"Es_{tag}_k"] = h["Ec_k"], h["Es_k"]
    q["theta2"], q["omega2"] = hv["theta2"], hv["omega2"]
    q["R_div_curl"] = hv["theta2"] / hv["omega2"]

    s  = np.log(rho / rho0)
    ds = s - s.mean()
    ss = s.std()
    q["s_mean"], q["sigma_s"] = s.mean(), ss
    q["skew_s"] = (ds**3).mean() / ss**3
    q["kurt_s"] = (ds**4).mean() / ss**4
    q["sigma_rho"] = (rho / rho0).std()
    q["b"]     = np.sqrt(np.expm1(ss**2)) / q["Ms"]
    q["b_lin"] = q["sigma_rho"] / q["Ms"]
    return q


def los_sums(rho, v, axis):
    """S0, S1, S2 = sum(rho), sum(rho v), sum(rho v^2) along one axis (native res)."""
    n  = LOS_AXES.index(axis)
    vl = v[n]
    return rho.sum(axis=n), (rho * vl).sum(axis=n), (rho * vl * vl).sum(axis=n)


def block_sum(a, f):
    if f == 1:
        return a
    N = a.shape[0]
    return a.reshape(N // f, f, N // f, f).sum(axis=(1, 3))


def moment_maps(S0, S1, S2, npix_los, factor):
    """
    Moment maps from line-of-sight sums, reduced by `factor` in each image
    direction. Reducing the SUMS (not the maps) makes the low-res centroid and
    dispersion the density-weighted moments of all gas in the coarse pixel,
    like a spectrum averaged over a beam.
    """
    S0r, S1r, S2r = (block_sum(a, factor) for a in (S0, S1, S2))
    col  = S0r / (npix_los * factor * factor)          # mean-normalised: <col> = rho0 * L
    cen  = S1r / S0r
    disp = np.sqrt(np.maximum(S2r / S0r - cen**2, 0.0))
    return np.stack([col, cen, disp]).astype(np.float32)


def augment_params(ex, name, frame, N):
    """
    [(los_index, aug_index, shift_x, shift_y, rot90)] for one snapshot, drawn
    from an RNG seeded by (seed, sim name, frame). Shared with
    extract_sim_radmc.py so both datasets use identical shifts per image.
    """
    rng = np.random.default_rng([ex["seed"], zlib.crc32(name.encode()), frame])
    out = []
    for il in range(len(LOS_AXES)):
        for ia in range(ex["n_aug_per_frame"]):
            dx, dy = (rng.integers(0, N, 2) if ex["random_shift"] else (0, 0))
            k90    = int(rng.integers(0, 4)) if ex["random_rot90"] else 0
            out.append((il, ia, int(dx), int(dy), k90))
    return out


def los_scalars(rho, v, axis, cs):
    """Per-LOS observable-like labels."""
    n   = LOS_AXES.index(axis)
    vl  = v[n]
    mu  = np.average(vl, weights=rho)
    return {"Ms_los_mw": np.sqrt(np.average((vl - mu)**2, weights=rho)) / cs,
            "Ms_los_vw": vl.std() / cs}


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACT (one simulation → part file)
# ═════════════════════════════════════════════════════════════════════════════

FRAME_FIELDS = {  # name: (description, units)
    "time":        ("simulation time of the snapshot", "code time (L / c_s)"),
    "Ms":          ("sonic Mach number sqrt(sum_i std(v_i)^2)/c_s, volume-weighted, mean flow removed; "
                    "same definition as Ms_act in the p49d AverageQuantities files", ""),
    "Ms_mw":       ("mass-weighted sonic Mach number", ""),
    "v_mean":      ("volume-mean velocity (vx, vy, vz)", "c_s"),
    "v_std":       ("volume-weighted std of (vx, vy, vz)", "c_s"),
    "Ec_v":        ("compressive power sum_k |k (k.v_k)|^2/k^4 of v (mean removed, k=0 dropped)", "arb."),
    "Es_v":        ("solenoidal power of v", "arb."),
    "chi_v":       ("compressive fraction Ec_v / (Ec_v + Es_v); 1/3 for random isotropic", ""),
    "Ec_w":        ("compressive power of w = sqrt(rho/rho0) v", "arb."),
    "Es_w":        ("solenoidal power of w", "arb."),
    "chi_w":       ("kinetic-energy-weighted compressive fraction Ec_w / (Ec_w + Es_w)", ""),
    "Ec_v_k":      ("compressive shell spectrum of v, shells |k| = 1..N/2 (2 pi / L)", "arb."),
    "Es_v_k":      ("solenoidal shell spectrum of v", "arb."),
    "Ec_w_k":      ("compressive shell spectrum of w", "arb."),
    "Es_w_k":      ("solenoidal shell spectrum of w", "arb."),
    "theta2":      ("<(div v)^2> via Parseval, same normalisation as Ec_v", "arb."),
    "omega2":      ("<|curl v|^2> via Parseval", "arb."),
    "R_div_curl":  ("theta2 / omega2 (k^2-weighted compressive ratio)", ""),
    "s_mean":      ("volume mean of s = ln(rho/rho0)", ""),
    "sigma_s":     ("volume-weighted std of s", ""),
    "skew_s":      ("skewness of s", ""),
    "kurt_s":      ("kurtosis of s (3 for a Gaussian)", ""),
    "sigma_rho":   ("std of rho/rho0", ""),
    "b":           ("density-variance parameter sqrt(exp(sigma_s^2)-1)/Ms; ill-conditioned for Ms < ~1", ""),
    "b_lin":       ("sigma_rho / Ms (no lognormal assumption)", ""),
}
LOS_FIELDS = {
    "Ms_los_mw":   ("mass-weighted 1D velocity dispersion along the LOS over the box / c_s", ""),
    "Ms_los_vw":   ("volume-weighted 1D velocity dispersion along the LOS / c_s", ""),
}


def part_path(cfg, name):
    d = os.path.join(cfg["output_dir"], "parts_" + os.path.splitext(cfg["output_name"])[0])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}.h5")


def cmd_extract(args):
    cfg  = json.load(open(args.config))
    sims = cfg["sims"]
    sim  = sims[args.index] if args.sim is None else next(s for s in sims if s["name"] == args.sim)
    ex   = cfg["extraction"]
    name = sim["name"]
    sim_dir = os.path.join(cfg["data_root"], name)
    cs   = sim.get("sound_speed", 1.0)
    out  = part_path(cfg, name)
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    frames = [f for f in frame_dirs(sim_dir) if f >= ex.get("min_frame", 0)]
    if args.frames:
        frames = [f for f in frames if f in set(args.frames)]
    if args.max_frames:
        frames = frames[:args.max_frames]

    with h5py.File(out, "a") as h:
        done = set(int(g[2:]) for g in h.keys() if g.startswith("DD"))
        h.attrs["sim_name"] = name
        h.attrs["sim_json"] = json.dumps(sim)
    todo = [f for f in frames if f not in done]
    print(f"{name}: {len(frames)} frames, {len(done)} already done, {len(todo)} to do "
          f"(workers={workers}) → {out}", flush=True)

    kg = None
    skipped = []
    t_start = time.time()
    for i, frame in enumerate(todo):
        t0 = time.time()
        if not frame_readable(sim_dir, frame):
            skipped.append((frame, "unreadable"))
            print(f"  DD{frame:04d} skipped: unreadable", flush=True)
            continue
        try:
            rho, v, t = load_frame(sim_dir, frame)
            N = rho.shape[0]
            if kg is None or kg.N != N:
                kg = KGrid(N)
            factor = N // ex["target_res"] if ex["target_res"] else 1
            q  = frame_scalars(rho, v, kg, cs, workers)
            q["time"] = t

            # deterministic per-(sim, frame) RNG so reruns reproduce the same shifts
            aug = augment_params(ex, name, frame, N)
            imgs, native, meta = [], [], []
            for il, ax in enumerate(LOS_AXES):
                S0, S1, S2 = los_sums(rho, v, ax)
                ls = los_scalars(rho, v, ax, cs)
                for _, ia, dx, dy, k90 in (a for a in aug if a[0] == il):
                    sh = [np.rot90(np.roll(a, (dx, dy), axis=(0, 1)), k90) for a in (S0, S1, S2)]
                    imgs.append(moment_maps(*sh, N, factor))
                    if ex["store_native_res"]:
                        native.append(moment_maps(*sh, N, 1))
                    meta.append((il, ia, dx, dy, k90, ls["Ms_los_mw"], ls["Ms_los_vw"]))
            del rho, v

            with h5py.File(out, "a") as h:
                g = h.create_group(f"DD{frame:04d}")
                g["images"] = np.stack(imgs)
                if native:
                    g["images_native"] = np.stack(native)
                m = np.array(meta, dtype=np.float64)
                for j, k in enumerate(["los", "aug_index", "shift_x", "shift_y", "rot90"]):
                    g[k] = m[:, j].astype(np.int32)
                g["Ms_los_mw"], g["Ms_los_vw"] = m[:, 5], m[:, 6]
                for k, val in q.items():
                    g.attrs[k] = val
            el = time.time() - t0
            rem = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1)
            print(f"  DD{frame:04d} t={t:8.3f} Ms={q['Ms']:6.3f} chi_v={q['chi_v']:.3f} "
                  f"b={q['b']:.3f}  {el:5.1f}s  (~{rem/60:.0f} min left)", flush=True)
        except Exception as e:
            skipped.append((frame, repr(e)))
            print(f"  DD{frame:04d} FAILED: {e!r}", flush=True)
            traceback.print_exc()

    with h5py.File(out, "a") as h:
        prev = json.loads(h.attrs.get("skipped", "[]"))
        h.attrs["skipped"] = json.dumps(prev + skipped)
    print(f"{name}: finished, {len(skipped)} skipped this run", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# MERGE (part files → one documented dataset)
# ═════════════════════════════════════════════════════════════════════════════

def git_hash():
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                        "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def put(h, path, data, desc, units=""):
    d = h.create_dataset(path, data=data)
    d.attrs["description"] = desc
    if units:
        d.attrs["units"] = units
    return d


def cmd_merge(args):
    cfg = json.load(open(args.config))
    ex  = cfg["extraction"]
    out = os.path.join(cfg["output_dir"], cfg["output_name"])
    frames_rows, samples, sources = [], [], []   # sources: (part path, group, n_images)
    sim_rows = []
    shape = native_shape = None
    for sid, sim in enumerate(cfg["sims"]):
        p = part_path(cfg, sim["name"])
        if not os.path.exists(p):           # keep a row so sim_id always indexes the config order
            print(f"  {sim['name']}: no part file, skipped")
            sim_rows.append((sid, sim, 0, 0, np.nan))
            continue
        with h5py.File(p, "r") as h:
            groups = sorted(g for g in h.keys() if g.startswith("DD"))
            done_frames = {int(g[2:]) for g in groups}
            skipped = {f for f, _ in json.loads(h.attrs.get("skipped", "[]"))} - done_frames
            for g in groups:
                G  = h[g]
                fid = len(frames_rows)
                frames_rows.append({"sim_id": sid, "frame": int(g[2:]), **dict(G.attrs)})
                n = G["images"].shape[0]
                shape = G["images"].shape[1:]
                if "images_native" in G:
                    native_shape = G["images_native"].shape[1:]
                sources.append((p, g, n))
                for j in range(n):
                    samples.append((sid, fid, int(g[2:]), int(G["los"][j]), int(G["aug_index"][j]),
                                    int(G["shift_x"][j]), int(G["shift_y"][j]), int(G["rot90"][j]),
                                    float(G["Ms_los_mw"][j]), float(G["Ms_los_vw"][j])))
        ms = np.array([r["Ms"] for r in frames_rows if r["sim_id"] == sid])
        sim_rows.append((sid, sim, len(groups), len(skipped), ms.mean() if len(ms) else np.nan))
        print(f"  {sim['name']:20s} frames={len(groups):4d} skipped={len(skipped):3d} "
              f"Ms_mean={sim_rows[-1][4]:.3f}")

    if not samples:
        raise SystemExit("Nothing to merge.")
    S = np.array(samples, dtype=object)
    fr_sid = np.array([r["sim_id"] for r in frames_rows])
    sim_xi = {sid: s.get("xi", np.nan) for sid, s, *_ in sim_rows}
    sim_ms_mean = {sid: m for sid, _, _, _, m in sim_rows}
    sample_sid = S[:, 0].astype(np.int32)
    sample_fid = S[:, 1].astype(np.int64)

    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as h:
        h.attrs["title"] = f"p79d {cfg['suite']} projected-moment dataset {DATASET_VERSION}"
        h.attrs["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        h.attrs["generator"] = f"extract_sim_data.py (git {git_hash()})"
        h.attrs["config_json"] = json.dumps(cfg)
        h.attrs["layout"] = (
            "One SAMPLE = one (simulation, frame, line of sight, augmentation copy). "
            "images[i] is sample i; samples/* give its keys; labels/* are the per-sample "
            "targets (copied from its frame); frames/* hold one row per snapshot, sims/* one "
            "row per simulation. Group train/test splits by samples/sim_id: every frame, LOS "
            "and augmentation of a simulation is strongly correlated. Top-level Ms_act, xi, "
            "frame, los, subsets are links kept for net9010-style loaders.")
        h.attrs["image_orientation"] = (
            "Maps are sums over the LOS axis of arrays indexed [x, y, z]: LOS x -> image "
            "[y, z], LOS y -> [x, z], LOS z -> [x, y]. All three channels share one "
            "orientation (the old FITS-based dataset had column density transposed for x and z).")

        d = h.create_dataset("images", shape=(len(samples), *shape), dtype=np.float32,
                             chunks=(1, *shape))
        d.attrs["description"] = "moment maps, shape [N, 3, H, W]; channels = " + ", ".join(CHANNELS)
        dn = None
        if native_shape is not None:
            dn = h.create_dataset("images_native", shape=(len(samples), *native_shape),
                                  dtype=np.float32, chunks=(1, *native_shape))
            dn.attrs["description"] = "same maps at native resolution"
        i0 = 0
        for p, g, n in sources:          # stream part files; keeps memory at one frame
            with h5py.File(p, "r") as hp:
                d[i0:i0 + n] = hp[g]["images"][()]
                if dn is not None:
                    dn[i0:i0 + n] = hp[g]["images_native"][()]
            i0 += n
        d.attrs["channels"] = json.dumps({
            "column_density": "sum(rho dl) / (rho0 L): mean 1 over the unshifted map",
            "velocity_centroid": "sum(rho v_los) / sum(rho), units c_s",
            "velocity_dispersion": "sqrt(sum(rho v_los^2)/sum(rho) - centroid^2), units c_s "
                                   "(a dispersion, NOT the variance stored by the old dataset)"})
        d.attrs["resolution_note"] = (
            f"native {cfg['sims'][0].get('resolution')}^2 LOS sums block-summed by "
            f"{cfg['sims'][0].get('resolution', 0) // max(1, ex['target_res'] or 1)} before "
            "forming moments, so reduced pixels are density-weighted beam averages")

        sm = h.create_group("samples")
        sm.attrs["description"] = "one row per image"
        put(sm, "sim_id", sample_sid, "index into sims/*")
        put(sm, "frame_id", sample_fid, "index into frames/*")
        put(sm, "frame", S[:, 2].astype(np.int32), "ENZO output number (DDxxxx)")
        put(sm, "los", S[:, 3].astype(np.int8), "line of sight: 0 = x, 1 = y, 2 = z")
        put(sm, "aug_index", S[:, 4].astype(np.int16), "augmentation copy number for this (frame, LOS)")
        put(sm, "shift_x", S[:, 5].astype(np.int32), "periodic roll applied to image axis 0 (native pixels)")
        put(sm, "shift_y", S[:, 6].astype(np.int32), "periodic roll applied to image axis 1 (native pixels)")
        put(sm, "rot90", S[:, 7].astype(np.int8), "number of 90-degree rotations applied after the roll")

        lb = h.create_group("labels")
        lb.attrs["description"] = "per-sample targets; frame quantities repeated for each image"
        put(lb, "xi", np.array([sim_xi[s] for s in sample_sid]),
            "forcing solenoidal weight (ENZO DrivenFlowWeight): 1 = solenoidal, 0 = compressive")
        put(lb, "Ms_los_mw", S[:, 8].astype(np.float64), LOS_FIELDS["Ms_los_mw"][0])
        put(lb, "Ms_los_vw", S[:, 9].astype(np.float64), LOS_FIELDS["Ms_los_vw"][0])
        for k, (desc, units) in FRAME_FIELDS.items():
            vals = np.array([r[k] for r in frames_rows])
            if vals.ndim == 1:
                put(lb, k, vals[sample_fid], desc, units)

        fg = h.create_group("frames")
        fg.attrs["description"] = "one row per snapshot (all LOS and augmentations share it)"
        put(fg, "sim_id", fr_sid.astype(np.int32), "index into sims/*")
        put(fg, "frame", np.array([r["frame"] for r in frames_rows], dtype=np.int32), "ENZO output number")
        for k, (desc, units) in FRAME_FIELDS.items():
            put(fg, k, np.array([r[k] for r in frames_rows]), desc, units)
        fg["Ec_v_k"].attrs["k_shells"] = "index j <-> |k| = j + 1"

        sg = h.create_group("sims")
        sg.attrs["description"] = "one row per simulation; full JSON entries in sims/json"
        put(sg, "name", np.array([s["name"] for _, s, *_ in sim_rows], dtype=h5py.string_dtype()),
            "simulation directory name (a label: its 'mach' number is not Ms)")
        put(sg, "xi", np.array([s.get("xi", np.nan) for _, s, *_ in sim_rows]), "DrivenFlowWeight")
        put(sg, "driving_velocity", np.array([s.get("driving_velocity", np.nan) for _, s, *_ in sim_rows]),
            "DrivenFlowVelocity (first component)", "c_s")
        put(sg, "n_frames", np.array([n for _, _, n, _, _ in sim_rows], dtype=np.int32), "frames extracted")
        put(sg, "n_skipped", np.array([n for _, _, _, n, _ in sim_rows], dtype=np.int32),
            "frames skipped (unreadable or failed)")
        put(sg, "Ms_mean", np.array([sim_ms_mean[sid] for sid, *_ in sim_rows]),
            "mean of frames/Ms over all extracted frames (spin-up included)")
        put(sg, "json", np.array([json.dumps(s) for _, s, *_ in sim_rows], dtype=h5py.string_dtype()),
            "config entry for the simulation")

        # compatibility links for loaders written for the old files
        h["subsets"] = h["images"]
        h["Ms_act"]  = h["labels/Ms"]
        h["xi"]      = h["labels/xi"]
        h["frame"]   = h["samples/frame"]
        h["los"]     = h["samples/los"]
    os.replace(tmp, out)
    print(f"Wrote {out}: {len(samples)} samples from {len(frames_rows)} frames, {len(sim_rows)} sims")


# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    c = sp.add_parser("config", help="write the suite JSON from a data directory")
    c.add_argument("--data_root", required=True)
    c.add_argument("--suite", default="mach_grid_256")
    c.add_argument("--pattern", default="xi_*")
    c.add_argument("--output_dir", default="/home/x-nbisht1/scratch/projects/p79d_dataset")
    c.add_argument("--out", default="sims_mach_grid.json")

    e = sp.add_parser("extract", help="process one simulation into its part file")
    e.add_argument("--config", required=True)
    e.add_argument("--index", type=int, default=None, help="simulation index in the JSON")
    e.add_argument("--sim", default=None, help="simulation name (instead of --index)")
    e.add_argument("--frames", type=int, nargs="*", help="only these frame numbers")
    e.add_argument("--max_frames", type=int, default=None, help="stop after this many frames (testing)")
    e.add_argument("--workers", type=int, default=None, help="FFT threads (default: SLURM cpus)")

    m = sp.add_parser("merge", help="combine part files into the final dataset")
    m.add_argument("--config", required=True)

    args = p.parse_args()
    if args.cmd == "extract" and args.index is None and args.sim is None:
        p.error("extract needs --index or --sim")
    {"config": cmd_config, "extract": cmd_extract, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
