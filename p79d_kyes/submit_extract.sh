#!/bin/sh
#-------------------------------------------------------------------------------
# Extract the mach_grid suite into the p79d dataset: one array task per sim,
# then a merge job once every task has ended.
#
#   sh submit_extract.sh                      # all sims in sims_mach_grid.json
#   sh submit_extract.sh 29,36                # only these config indices
#
# Tasks are resumable: rerunning skips frames already in the part files, so
# sims that were unreadable can simply be resubmitted later.
#-------------------------------------------------------------------------------
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
CONFIG=sims_mach_grid.json
NSIM=$(python -c "import json;print(len(json.load(open('$CONFIG'))['sims']))")
ARRAY=${1:-0-$((NSIM-1))}
LOGS=$(python -c "import json;print(json.load(open('$CONFIG'))['output_dir'])")/logs_extract
mkdir -p $LOGS

JID=$(sbatch --parsable -J extract_p79d -p shared -c 8 -t 06:00:00 -A phy240036 \
      --array=$ARRAY -o $LOGS/extract_%a.out \
      --wrap "source ~/.bashrc; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8; \
              python extract_sim_data.py extract --config $CONFIG --index \$SLURM_ARRAY_TASK_ID")
echo "extract array: $JID ($ARRAY)"
MID=$(sbatch --parsable -J merge_p79d -p shared -c 4 -t 00:20:00 -A phy240036 \
      --dependency=afterany:$JID -o $LOGS/merge.out \
      --wrap "source ~/.bashrc; python extract_sim_data.py merge --config $CONFIG")
echo "merge job: $MID (runs after the array ends)"
