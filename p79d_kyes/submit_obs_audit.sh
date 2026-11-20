#!/bin/sh
#-------------------------------------------------------------------------------
# Audit follow-up (2026-11-20), one job:
#   the mask-augmented run-2 CO models on the nearby clouds and on MWISP,
#   and the Herschel dust run again with the all-runs model as the adopted value.
# Outputs: ~/plots/observations_net9010[_mwisp]_final_co{13,both}_mask, ~/plots/observations_net9010_dust_final
#-------------------------------------------------------------------------------
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
P=/home/x-nbisht1/plots
CMD="source ~/.bashrc; export PYTHONUNBUFFERED=1; set -e;"
CMD="$CMD python predict_dust_net9010.py --tag final_mom0 --co_tiles_dir $P/observations_net9010_final_proj \
          --output_dir $P/observations_net9010_dust_final;"
for T in final_co13_mask final_coboth_mask; do
  CMD="$CMD python predict_observations_net9010.py --tag $T;"
  CMD="$CMD python predict_observations_net9010.py --config obs_mwisp.json --tag $T;"
done
sbatch -J obs_audit -p shared -c 32 --mem=60G -t ${TIME:-04:00:00} -A phy240036 \
       -o /home/x-nbisht1/projects/p79d_dataset/models/logs/obs_audit_%j.out --wrap "$CMD"
