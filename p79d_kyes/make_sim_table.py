"""
make_sim_table.py
=================
Write the LaTeX table of the simulation suite (tab_sims.tex) for the paper.

  python make_sim_table.py --config sims_mach_grid.json --out ../p79d_paper/<id>/tab_sims.tex

Mach numbers come from the merged dataset if it exists (frames/Ms), otherwise
from the p49d AverageQuantities files next to the simulations (same definition:
sqrt(sum_i std(v_i)^2) / c_s).
"""
import argparse, glob, json, os
import numpy as np
import h5py

p = argparse.ArgumentParser()
p.add_argument("--config", default="sims_mach_grid.json")
p.add_argument("--products", default="/anvil/scratch/x-ux454321/Paper83/256_mach_grid/products")
p.add_argument("--out", default="tab_sims.tex")
args = p.parse_args()

cfg = json.load(open(args.config))
merged = os.path.join(cfg["output_dir"], cfg["output_name"])
ms_by_sim = {}
if os.path.exists(merged):
    with h5py.File(merged, "r") as f:
        names, sid, ms = f["sims/name"][:].astype(str), f["frames/sim_id"][:], f["frames/Ms"][:]
        for i, n in enumerate(names):
            ms_by_sim[n] = ms[sid == i]
    source = "the extracted dataset"
else:
    for s in cfg["sims"]:
        vals = []
        for fn in sorted(glob.glob(f"{args.products}/{s['name']}/DD*.products/data*.AverageQuantities.h5")):
            try:
                with h5py.File(fn, "r") as h:
                    vals.append(np.sqrt(sum(h[f"v{a}_std"][0] ** 2 for a in "xyz")))
            except OSError:
                pass
        ms_by_sim[s["name"]] = np.array(vals)
    source = "the AverageQuantities files"

rows = []
for s in cfg["sims"]:
    ms = ms_by_sim.get(s["name"], np.array([]))
    V, note = s.get("driving_velocity"), ""
    if V is None:                                   # parameter file unreadable: name = 10 V
        V, note = float(s["name"].split("mach")[1]) / 10, "$^a$"
    rows.append((s["xi"], np.mean(ms) if len(ms) else np.nan, s, V, note, ms))
rows.sort(key=lambda r: (r[0], r[1]))

lines = [
    r"\begin{deluxetable}{lccccc}",
    r"\tabletypesize{\scriptsize}",
    r"\tablecaption{The simulation suite. $V$ is the driving amplitude, $N$ the number of "
    r"snapshots after the initial conditions, $\langle\Ms\rangle$ the mean over all snapshots "
    r"(including the initial growth) and the last column the range over snapshots. \label{tab:sims}}",
    r"\tablehead{\colhead{Name} & \colhead{$\xi$} & \colhead{$V/c_s$} & \colhead{$N$} & "
    r"\colhead{$\langle\Ms\rangle$} & \colhead{$\Ms$ range}}",
    r"\startdata",
]
for xi, msm, s, V, note, ms in rows:
    name = s["name"].replace("_", r"\_")
    rng = f"{ms.min():.2f}--{ms.max():.2f}" if len(ms) else "---"
    lines.append(f"\\texttt{{{name}}} & {xi:.2f} & {V:.3g}{note} & {s['n_frame_dirs'] - 1} & "
                 f"{msm:.2f} & {rng} \\\\")
lines += [r"\enddata",
          r"\tablecomments{Mach numbers from " + source + ". The ``mach'' label in each name is "
          r"$10\,V/c_s$, not the realised Mach number."
          + (r" $^a$From the name; the parameter file was not readable." if any(r[4] for r in rows) else "")
          + "}",
          r"\end{deluxetable}"]
open(args.out, "w").write("\n".join(lines) + "\n")
print(f"Wrote {args.out} ({len(rows)} sims, Mach numbers from {source})")
