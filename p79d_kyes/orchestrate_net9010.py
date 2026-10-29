"""
orchestrate_net9010.py
======================
Run a list of net9010 trainings on the highmem partition within the per-user
limits (default: 4 jobs submitted, 2 running), resubmitting as slots free up.

  python orchestrate_net9010.py plan  --plan run2 --out orch_run2.json   # write the job list
  python orchestrate_net9010.py run   --state orch_run2.json             # loop until done
  python orchestrate_net9010.py status --state orch_run2.json

The loop is meant to run as a small long job (submit_orchestrator.sh), and it
can be restarted at any time: everything lives in the state file.

Every cycle it
  1. reads squeue: all of the user's highmem jobs count against the limits,
     including jobs from other projects;
  2. reads sacct for its own jobs: COMPLETED counts only if the expected
     output files exist; FAILED / TIMEOUT / OUT_OF_MEMORY / NODE_FAIL are
     retried (up to max_retries) with more time or memory as appropriate;
  3. re-sizes the jobs still to be submitted from finished ones of the same
     group (time = 1.25 x the longest, memory = 1.3 x the largest MaxRSS),
     so later jobs ask for what they need and schedule sooner;
  4. submits from the queue, in order, while there are free slots;
  5. pools the k-fold results (analyze_kfold_net9010.py) once a group is complete.
"""

import argparse, datetime, json, math, os, re, subprocess, sys, time

REPO      = "/home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes"
MODEL_DIR = "/home/x-nbisht1/projects/p79d_dataset/models"
LOG_DIR   = f"{MODEL_DIR}/logs"
DATA      = "/home/x-nbisht1/scratch/projects/p79d_dataset"
ACCOUNT   = "phy240036"
PARTITION = "highmem"
USER      = os.environ.get("USER", "x-nbisht1")
RETRY_STATES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def hms(hours):
    m = int(math.ceil(hours * 60))
    return f"{m // 60:02d}:{m % 60:02d}:00"


def to_hours(t):
    """sacct elapsed [D-]HH:MM:SS -> hours"""
    d = 0
    if "-" in t:
        d, t = t.split("-"); d = int(d)
    h, m, s = (int(x) for x in t.split(":"))
    return d * 24 + h + m / 60 + s / 3600


def to_gb(rss):
    if not rss:
        return 0.0
    u = rss[-1]
    v = float(rss[:-1]) if u.isalpha() else float(rss)
    return v * {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}.get(u, 1 / 1024**3)


# ─── PLANS ───────────────────────────────────────────────────────────────────

def plan_run2():
    """Step 3: run-2 fcrao500 maps, 13CO alone / 12CO alone / both lines; 5 folds + final model each."""
    f = f"{DATA}/p79d_mach_grid_256_co_v2_fcrao500.h5"
    variants = [  # (group/tag, image_key, memory guess GB)
        ("co2_13", "images_obs_13co", 56),
        ("co2_both", "images_obs_13co+images_obs_12co", 64),
        ("co2_12", "images_obs_12co", 56),
    ]
    jobs = []
    # fold-major order so all three variants become comparable early
    for fold in ["0", "1", "2", "3", "4", "all"]:
        for tag, key, mem in variants:
            out = (f"{MODEL_DIR}/net9010_{tag}_allmom_fold{fold}_test.json" if fold != "all"
                   else f"{MODEL_DIR}/conformal_quantiles_9010_{tag}_allmom_foldall.json")
            jobs.append({
                "name": f"{tag}_f{fold}", "group": tag, "fold": fold,
                "args": f"--mode allmom --fold {fold} --data_file {f} --image_key {key} --tag {tag}",
                "outputs": [out],
                "mem_gb": mem, "time_h": 9.0, "time_h0": 9.0,   # first guess; refined from finished jobs
                "state": "queued", "attempts": [],
            })
    return {"name": "run2", "max_submitted": 4, "max_running": 2, "max_retries": 2,
            "poll_s": 600, "cores": 128, "groups_pooled": [], "jobs": jobs}


def plan_tests1():
    """Mask augmentation (projected + run-2 13CO) and alternative compressibility targets (chi_w, b)."""
    proj = f"{DATA}/p79d_mach_grid_256_v1.h5"
    co2  = f"{DATA}/p79d_mach_grid_256_co_v2_fcrao500.h5"
    spec = [  # (name, group/tag, fold, extra args, memory GB)
        ("projmask_fall",   "projmask",   "all", f"--data_file {proj} --image_key images --mask_aug_prob 0.5", 56),
        ("co2_13mask_fall", "co2_13mask", "all", f"--data_file {co2} --image_key images_obs_13co --mask_aug_prob 0.5", 56),
        ("chiw_f0",         "chiw",       "0",   f"--data_file {proj} --chi_field chi_w", 56),
        ("b_f0",            "b",          "0",   f"--data_file {proj} --chi_field b --chi_space log", 56),
        ("projmask_f0",     "projmask",   "0",   f"--data_file {proj} --image_key images --mask_aug_prob 0.5", 56),
        ("co2_13mask_f0",   "co2_13mask", "0",   f"--data_file {co2} --image_key images_obs_13co --mask_aug_prob 0.5", 56),
        ("chiw_f1",         "chiw",       "1",   f"--data_file {proj} --chi_field chi_w", 56),
        ("b_f1",            "b",          "1",   f"--data_file {proj} --chi_field b --chi_space log", 56),
    ]
    jobs = []
    for name, tag, fold, extra, mem in spec:
        out = (f"{MODEL_DIR}/net9010_{tag}_allmom_fold{fold}_test.json" if fold != "all"
               else f"{MODEL_DIR}/conformal_quantiles_9010_{tag}_allmom_fold{fold}.json")
        jobs.append({"name": name, "group": tag, "fold": fold,
                     "args": f"--mode allmom --fold {fold} {extra} --tag {tag}", "outputs": [out],
                     "mem_gb": mem, "time_h": 9.0, "time_h0": 9.0, "state": "queued", "attempts": []})
    return {"name": "tests1", "max_submitted": 4, "max_running": 2, "max_retries": 2,
            "poll_s": 600, "cores": 128, "groups_pooled": [], "jobs": jobs}


def plan_tests2():
    """Mask augmentation on the remaining projected folds; the column-density (mom0) k-fold + final model."""
    proj = f"{DATA}/p79d_mach_grid_256_v1.h5"
    spec = [("projmask", "allmom", f, "--mask_aug_prob 0.5") for f in ("1", "2", "3", "4")]
    spec += [("mom0", "mom0", f, "") for f in ("0", "1", "2", "3", "4", "all")]
    order = [spec[i] for i in (0, 4, 1, 5, 2, 6, 3, 7, 8, 9)]      # interleave the two groups
    jobs = []
    for tag, mode, fold, extra in order:
        stem = f"9010_{tag}_{mode}"
        out = (f"{MODEL_DIR}/net{stem}_fold{fold}_test.json" if fold != "all"
               else f"{MODEL_DIR}/conformal_quantiles_{stem}_fold{fold}.json")
        jobs.append({"name": f"{tag}_f{fold}", "group": tag, "fold": fold, "mode": mode,
                     "args": f"--mode {mode} --fold {fold} --data_file {proj} --image_key images {extra} --tag {tag}",
                     "outputs": [out], "mem_gb": 56, "time_h": 9.0, "time_h0": 9.0,
                     "state": "queued", "attempts": []})
    return {"name": "tests2", "max_submitted": 4, "max_running": 2, "max_retries": 2,
            "poll_s": 600, "cores": 128, "groups_pooled": [], "jobs": jobs}


def plan_smooth():
    """Projected + mask augmentation with the time-smoothed chi_v (1 crossing time), chi loss
    only at Ms >= 2, and the density head (ln b_lin, ln sigma_s); 5 folds + final model."""
    proj  = f"{DATA}/p79d_mach_grid_256_v1.h5"
    extra = "--mask_aug_prob 0.5 --chi_smooth_tc 1.0 --chi_ms_min 2 --dens_head"
    jobs = []
    for fold in ("0", "1", "2", "3", "4", "all"):
        out = (f"{MODEL_DIR}/net9010_smooth_allmom_fold{fold}_test.json" if fold != "all"
               else f"{MODEL_DIR}/conformal_quantiles_9010_smooth_allmom_fold{fold}.json")
        jobs.append({"name": f"smooth_f{fold}", "group": "smooth", "fold": fold, "mode": "allmom",
                     "args": f"--mode allmom --fold {fold} --data_file {proj} --image_key images {extra} --tag smooth",
                     "outputs": [out], "mem_gb": 48, "time_h": 8.0, "time_h0": 8.0,
                     "state": "queued", "attempts": []})
    return {"name": "smooth", "max_submitted": 4, "max_running": 2, "max_retries": 2,
            "poll_s": 600, "cores": 128, "groups_pooled": [], "jobs": jobs}


def plan_final():
    """Final tables on the full suite (with the 3 resynced runs): every input type, 5 folds + final
    model each, all with the density head (ln b_lin, ln sigma_s); raw chi_v, no Ms cut in training."""
    proj, r1, r2 = (f"{DATA}/p79d_mach_grid_256_v1.h5", f"{DATA}/p79d_mach_grid_256_13co_v1_fcrao500_v2.h5",
                    f"{DATA}/p79d_mach_grid_256_co_v2_fcrao500.h5")
    variants = [  # (tag, mode, data, image_key, extra, memory GB)
        ("final_proj",   "allmom", proj, "images", "--mask_aug_prob 0.5", 56),
        ("final_mom0",   "mom0",   proj, "images", "", 52),
        ("final_13co",   "allmom", r1,   "images_obs", "", 56),
        ("final_co13",   "allmom", r2,   "images_obs_13co", "", 56),
        ("final_co12",   "allmom", r2,   "images_obs_12co", "", 56),
        ("final_coboth", "allmom", r2,   "images_obs_13co+images_obs_12co", "", 64),
    ]
    jobs = []
    for fold in ("0", "1", "2", "3", "4", "all"):            # interleaved, so every group finishes early folds first
        for tag, mode, data, key, extra, mem in variants:
            out = (f"{MODEL_DIR}/net9010_{tag}_{mode}_fold{fold}_test.json" if fold != "all"
                   else f"{MODEL_DIR}/conformal_quantiles_9010_{tag}_{mode}_fold{fold}.json")
            jobs.append({"name": f"{tag}_f{fold}", "group": tag, "fold": fold, "mode": mode,
                         "args": f"--mode {mode} --fold {fold} --data_file {data} --image_key {key} {extra} "
                                 f"--dens_head --tag {tag}".replace("  ", " "),
                         "outputs": [out], "mem_gb": mem, "time_h": 10.0, "time_h0": 10.0,
                         "state": "queued", "attempts": []})
    return {"name": "final", "max_submitted": 4, "max_running": 2, "max_retries": 2,
            "poll_s": 600, "cores": 128, "groups_pooled": [], "jobs": jobs}


PLANS = {"run2": plan_run2, "tests1": plan_tests1, "tests2": plan_tests2, "smooth": plan_smooth, "final": plan_final}


# ─── STATE ───────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as fh:
        return json.load(fh)


def save(state, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, path)


def highmem_jobs():
    """(n_running, n_total) of all the user's highmem jobs, any project."""
    rc, out, _ = sh(f"squeue -h -u {USER} -p {PARTITION} -o '%T'")
    if rc:
        return None
    states = out.split()
    return sum(s == "RUNNING" for s in states), len(states)


def sacct(jobid):
    """State, elapsed hours, MaxRSS GB of a job."""
    rc, out, _ = sh(f"sacct -j {jobid} -n -P -o JobID,State,Elapsed,MaxRSS")
    if rc or not out:
        return None
    state, elapsed, rss = None, 0.0, 0.0
    for line in out.splitlines():
        jid, st, el, mr = line.split("|")
        if jid == str(jobid):
            state, elapsed = st.split()[0], to_hours(el)
        rss = max(rss, to_gb(mr))
    return state, elapsed, rss


def resize(state):
    """Size queued jobs from finished jobs of the same group (fold-all jobs train on more data)."""
    for job in state["jobs"]:
        if job["state"] != "queued":
            continue
        done = [j for j in state["jobs"] if j["group"] == job["group"] and j["state"] == "done"
                and (j["fold"] == "all") == (job["fold"] == "all")]
        if not done:
            continue
        t = max(j["attempts"][-1]["elapsed_h"] for j in done)
        m = max(j["attempts"][-1]["maxrss_gb"] for j in done)
        # never below the planned limit: a fast first job must not cut later ones short
        # (training stops on its time budget before early stopping otherwise)
        job["time_h"] = round(min(14.0, max(job.get("time_h0", 9.0), 1.25 * t + 0.25)), 2)
        job["mem_gb"] = int(min(900, max(48, math.ceil(1.3 * m / 8) * 8)))


def submit(job, state):
    budget = max(1.0, job["time_h"] - 0.75)     # leave time for calibration + test evaluation
    log_file = f"{LOG_DIR}/net9010_{job['name']}_%j.out"
    wrap = (f"source ~/.bashrc; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS={state['cores']} "
            f"MKL_NUM_THREADS={state['cores']}; cd {REPO}; "
            f"python ViT_setup_test_net9010.py {job['args']} --time_budget_h {budget:.2f}")
    cmd = (f"sbatch --parsable -J n9010_{job['name']} -p {PARTITION} -N 1 -c {state['cores']} "
           f"--mem={job['mem_gb']}G -t {hms(job['time_h'])} -A {ACCOUNT} -o {log_file} "
           f"--wrap \"{wrap}\"")
    rc, out, err = sh(cmd)
    if rc:
        log(f"sbatch failed for {job['name']}: {err}")
        return False
    jid = out.split(";")[0]
    job["attempts"].append({"jobid": jid, "submitted": now(), "mem_gb": job["mem_gb"],
                            "time_h": job["time_h"], "budget_h": budget})
    job["state"] = "submitted"
    log(f"submitted {job['name']} as {jid} ({job['mem_gb']} GB, {hms(job['time_h'])}, budget {budget:.2f} h)")
    return True


def update(job, state):
    a = job["attempts"][-1]
    r = sacct(a["jobid"])
    if r is None:
        return
    st, el, rss = r
    a.update(sacct_state=st, elapsed_h=round(el, 3), maxrss_gb=round(rss, 1))
    if st in ("PENDING", "REQUEUED"):
        job["state"] = "submitted"
    elif st == "RUNNING":
        job["state"] = "running"
    elif st == "COMPLETED":
        missing = [o for o in job["outputs"] if not os.path.exists(o)]
        if missing:
            log(f"{job['name']} ({a['jobid']}) COMPLETED but outputs missing: {missing}")
            retry(job, state, "missing outputs")
        else:
            job["state"] = "done"
            log(f"done {job['name']} ({a['jobid']}): {el:.2f} h, MaxRSS {rss:.1f} GB")
    elif st in RETRY_STATES or st.startswith("CANCELLED"):
        retry(job, state, st)


def retry(job, state, why):
    if len(job["attempts"]) > state["max_retries"]:
        job["state"] = "failed"
        log(f"FAILED {job['name']} after {len(job['attempts'])} attempts ({why}); giving up")
        return
    a = job["attempts"][-1]
    if why == "OUT_OF_MEMORY" or (a.get("maxrss_gb", 0) > 0.9 * a["mem_gb"]):
        job["mem_gb"] = int(min(900, job["mem_gb"] * 1.5))
    if why == "TIMEOUT":
        job["time_h"] = round(min(16.0, job["time_h"] + 2.0), 2)
    job["state"] = "queued"
    log(f"retry {job['name']} ({why}) with {job['mem_gb']} GB, {hms(job['time_h'])}")


def pool_finished_groups(state):
    for g in sorted({j["group"] for j in state["jobs"]}):
        if g in state["groups_pooled"]:
            continue
        folds = [j for j in state["jobs"] if j["group"] == g and j["fold"] != "all"]
        if folds and all(j["state"] == "done" for j in folds):
            mode = folds[0].get("mode", "allmom")
            out = f"{MODEL_DIR}/kfold_summary_{g}_{mode}.txt"
            tag = f"--tag {g}" if g else ""
            rc, o, e = sh(f"cd {REPO} && python analyze_kfold_net9010.py --mode {mode} {tag} "
                          f"> {out} 2>&1")
            log(f"pooled {g} -> {out} (rc {rc})")
            state["groups_pooled"].append(g)


def cycle(state):
    for job in state["jobs"]:
        if job["state"] in ("submitted", "running"):
            update(job, state)
    resize(state)
    pool_finished_groups(state)
    hm = highmem_jobs()
    if hm is None:
        log("squeue failed; skipping submissions this cycle")
        return
    running, total = hm
    for job in state["jobs"]:
        if job["state"] != "queued":
            continue
        if total >= state["max_submitted"]:
            break
        # queued jobs only start when a running slot frees, so pending ones beyond
        # max_running just wait in SLURM's queue; that keeps the next start immediate
        if submit(job, state):
            total += 1


def summary(state):
    counts = {}
    for j in state["jobs"]:
        counts[j["state"]] = counts.get(j["state"], 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan"); a.add_argument("--plan", choices=list(PLANS)); a.add_argument("--out")
    a = sub.add_parser("run"); a.add_argument("--state"); a.add_argument("--once", action="store_true")
    a.add_argument("--max_hours", type=float, default=None,
                   help="exit after this long and resubmit the orchestrator job (short jobs schedule sooner)")
    a = sub.add_parser("status"); a.add_argument("--state")
    args = p.parse_args()

    if args.cmd == "plan":
        if os.path.exists(args.out):
            sys.exit(f"{args.out} exists; not overwriting a live state file")
        save(PLANS[args.plan](), args.out)
        print(f"wrote {args.out}")
        return

    state = load(args.state)
    if args.cmd == "status":
        print(summary(state))
        for j in state["jobs"]:
            a = j["attempts"][-1] if j["attempts"] else {}
            print(f"  {j['name']:14s} {j['state']:9s} job {a.get('jobid', '-'):>9s} "
                  f"{a.get('sacct_state', ''):10s} {a.get('elapsed_h', 0):5.2f} h "
                  f"{a.get('maxrss_gb', 0):6.1f}/{j['mem_gb']} GB  tries {len(j['attempts'])}")
        return

    log(f"orchestrating {state['name']}: {len(state['jobs'])} jobs, "
        f"limits {state['max_submitted']} submitted / {state['max_running']} running")
    t_start = time.time()
    while True:
        cycle(state)
        save(state, args.state)
        log(summary(state))
        if all(j["state"] in ("done", "failed") for j in state["jobs"]):
            log("all jobs finished")
            break
        if args.once:
            break
        if args.max_hours and time.time() - t_start + state["poll_s"] > args.max_hours * 3600:
            rc, out, err = sh(f"cd {REPO} && sh submit_orchestrator.sh {args.state}")
            log(f"time limit near: resubmitted orchestrator ({out or err})")
            break
        time.sleep(state["poll_s"])


if __name__ == "__main__":
    main()
