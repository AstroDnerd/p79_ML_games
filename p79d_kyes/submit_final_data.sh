#!/bin/sh
#-------------------------------------------------------------------------------
# Step 10: add the 3 resynced runs (config indices 8, 13, 21: xi_0.5_mach10,
# xi_0.5_mach20, xi_0.5_mach70) to every dataset, then train the final k-folds.
#
#   projected     extract (shared, per sim) -> merge
#   13CO run 1    RADMC (6 chunks per sim)  -> merge -> re-observe fcrao500_v2 + merge
#   CO run 2      RADMC (6 chunks per sim)  -> merge + fcrao500 / mwisp / nro_orion
#   training      orchestrate_net9010.py plan final (36 highmem jobs), after all of the above
#
# Every extraction is resumable (frames already in a part file are skipped), so
# this only renders the frames that were unreadable before.
#
#   sh submit_final_data.sh            # check the frames are readable first:
#   python -c "import extract_sim_data as E; ..."  (see check below)
#-------------------------------------------------------------------------------
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
CONFIG=sims_mach_grid.json
SIMS="8 13 21"
OUT=$(python -c "import json;print(json.load(open('$CONFIG'))['output_dir'])")
A=phy240036

# refuse to start while the resynced frames are still unreadable
python - <<EOF || exit 1
import json, os, sys, extract_sim_data as E
cfg = json.load(open("$CONFIG"))
bad = 0
for i in map(int, "$SIMS".split()):
    d = os.path.join(cfg["data_root"], cfg["sims"][i]["name"])
    fr = E.frame_dirs(d); n = sum(not E.frame_readable(d, f) for f in fr)
    print(f"{cfg['sims'][i]['name']}: {len(fr)} frames, {n} unreadable"); bad += n
sys.exit(1 if bad > 10 else 0)
EOF

# projected
P=$(sbatch --parsable -J extract_p79d -p shared -c 8 -t 06:00:00 -A $A --array=$(echo $SIMS | tr ' ' ,) \
    -o $OUT/logs_extract/extract_%a.out \
    --wrap "source ~/.bashrc; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8; \
            python extract_sim_data.py extract --config $CONFIG --index \$SLURM_ARRAY_TASK_ID")
PM=$(sbatch --parsable -J merge_p79d -p shared -c 4 --mem=16G -t 01:00:00 -A $A --dependency=afterok:$P \
     -o $OUT/logs_extract/merge.out --wrap "source ~/.bashrc; python extract_sim_data.py merge --config $CONFIG")

# RADMC tasks: sim i -> array tasks 6i .. 6i+5
TASKS=$(for i in $SIMS; do for c in 0 1 2 3 4 5; do printf "%s," $((6 * i + c)); done; done | sed 's/,$//')

# 13CO run 1
R1=$(sbatch --parsable -J radmc_p79d -p shared -c 8 --mem=14G -t 48:00:00 -A $A --array=$TASKS \
     -o $OUT/logs_radmc/radmc_%a.out \
     --wrap "source ~/.bashrc; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8; \
             export TMPDIR=/tmp/radmc_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}; mkdir -p \$TMPDIR; \
             python extract_sim_radmc.py extract --config $CONFIG --rt radmc_13co.json \
                    --index \$((SLURM_ARRAY_TASK_ID / 6)) --chunk \$((SLURM_ARRAY_TASK_ID % 6)) --n_chunks 6; \
             rm -rf \$TMPDIR")
R1P=$(sbatch --parsable -J post_13co -p shared -c 32 --mem=24G -t 02:00:00 -A $A --dependency=afterok:$R1 \
      -o $OUT/logs_radmc/post_final.out \
      --wrap "source ~/.bashrc; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1; set -e; \
              python extract_sim_radmc.py merge --config $CONFIG --rt radmc_13co.json; \
              python extract_sim_radmc.py reobserve --config $CONFIG --rt radmc_13co.json --obs fcrao500_v2; \
              python extract_sim_radmc.py merge --config $CONFIG --rt radmc_13co.json --obs fcrao500_v2")

# CO run 2
R2=$(sbatch --parsable -J co_p79d -p shared -c 10 --mem=20G -t 16:00:00 -A $A --array=$TASKS \
     -o $OUT/logs_co/co_%a.out \
     --wrap "source ~/.bashrc; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=10; \
             export TMPDIR=/tmp/co_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}; mkdir -p \$TMPDIR; \
             python extract_sim_co.py extract --config $CONFIG --rt radmc_co_v2.json \
                    --index \$((SLURM_ARRAY_TASK_ID / 6)) --chunk \$((SLURM_ARRAY_TASK_ID % 6)) --n_chunks 6; \
             rm -rf \$TMPDIR")
CMD="source ~/.bashrc; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1; set -e; \
     python extract_sim_co.py merge --config $CONFIG --rt radmc_co_v2.json;"
for S in fcrao500 mwisp nro_orion; do
  CMD="$CMD python extract_sim_co.py reobserve --config $CONFIG --rt radmc_co_v2.json --obs $S; \
            python extract_sim_co.py merge --config $CONFIG --rt radmc_co_v2.json --obs $S;"
done
R2P=$(sbatch --parsable -J co_post -p shared -c 32 --mem=60G -t 03:00:00 -A $A --dependency=afterok:$R2 \
      -o $OUT/logs_co/co_post_final.out --wrap "$CMD")

# final trainings once every dataset is merged
STATE=/home/x-nbisht1/projects/p79d_dataset/models/orch_final.json
[ -f $STATE ] || python orchestrate_net9010.py plan --plan final --out $STATE
O=$(sbatch --parsable -J orch_final -p shared -c 1 --mem=2G -t 12:00:00 -A $A \
    --dependency=afterok:$PM:$R1P:$R2P --open-mode=append \
    -o /home/x-nbisht1/projects/p79d_dataset/models/logs/orch_final.log \
    --wrap "source ~/.bashrc; export PYTHONUNBUFFERED=1; python orchestrate_net9010.py run --state $STATE --max_hours 11.5")
echo "projected $P -> merge $PM | 13CO $R1 -> post $R1P | CO $R2 -> post $R2P | orchestrator $O"
echo "The orchestrator stops after 11.5 h; resubmit with: sh submit_orchestrator.sh $STATE"
