#!/bin/sh
#-------------------------------------------------------------------------------
# Synthetic 13CO (RADMC-3D) extraction of the mach_grid suite.
# Array task i -> simulation i / NCHUNK, chunk i % NCHUNK (frames[chunk::NCHUNK]),
# then one merge job after every task has ended.
#
#   sh submit_radmc.sh                 # all 42 sims x 6 chunks = tasks 0-251
#   sh submit_radmc.sh 0-5             # only these array tasks (e.g. sim 0)
#   EXTRA="--max_frames 2" sh submit_radmc.sh 0     # quick test
#
# Resumable: frames already in a part file are skipped, so failed or
# unreadable chunks can simply be resubmitted with the same task ids.
#-------------------------------------------------------------------------------
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
CONFIG=sims_mach_grid.json
RT=radmc_13co.json
NCHUNK=6
NSIM=$(python -c "import json;print(len(json.load(open('$CONFIG'))['sims']))")
ARRAY=${1:-0-$((NSIM*NCHUNK-1))}
LOGS=$(python -c "import json;print(json.load(open('$CONFIG'))['output_dir'])")/logs_radmc
mkdir -p $LOGS

JID=$(sbatch --parsable -J radmc_p79d -p shared -c 8 --mem=14G -t 48:00:00 -A phy240036 \
      --array=$ARRAY -o $LOGS/radmc_%a.out \
      --wrap "source ~/.bashrc; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8; \
              export TMPDIR=/tmp/radmc_\${SLURM_JOB_ID}; mkdir -p \$TMPDIR; \
              python extract_sim_radmc.py extract --config $CONFIG --rt $RT \
                     --index \$((SLURM_ARRAY_TASK_ID / $NCHUNK)) \
                     --chunk \$((SLURM_ARRAY_TASK_ID % $NCHUNK)) --n_chunks $NCHUNK $EXTRA; \
              rm -rf \$TMPDIR")
echo "radmc array: $JID ($ARRAY)"
if [ -z "$EXTRA" ]; then
  MID=$(sbatch --parsable -J merge_radmc -p shared -c 2 --mem=8G -t 04:00:00 -A phy240036 \
        --dependency=afterany:$JID -o $LOGS/merge.out \
        --wrap "source ~/.bashrc; python extract_sim_radmc.py merge --config $CONFIG --rt $RT")
  echo "merge job: $MID (runs after the array ends)"
fi
