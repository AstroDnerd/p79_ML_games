#!/bin/sh
#-------------------------------------------------------------------------------
# Re-observe the stored 13CO cubes with a named setup from radmc_13co.json
# ("observe" section) and merge that version — one job, parallel inside.
#
#   sh submit_reobserve.sh fcrao500_v2
#
# Output: <output_dir>/p79d_mach_grid_256_13co_v1_<name>.h5
# (~34k images: ~20-30 min on 32 cores, <10 GB; merge ~3 min, <1 GB)
#-------------------------------------------------------------------------------
NAME=${1:?usage: sh submit_reobserve.sh <observe-setup-name>}
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
LOGS=$(python -c "import json;print(json.load(open('sims_mach_grid.json'))['output_dir'])")/logs_radmc
sbatch -J reobs_$NAME -p shared -c 32 --mem=16G -t 01:00:00 -A phy240036 -o $LOGS/reobserve_$NAME.out \
       --wrap "source ~/.bashrc; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1; \
               python extract_sim_radmc.py reobserve --config sims_mach_grid.json --rt radmc_13co.json --obs $NAME && \
               python extract_sim_radmc.py merge --config sims_mach_grid.json --rt radmc_13co.json --obs $NAME"
