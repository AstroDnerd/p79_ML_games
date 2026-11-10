#!/bin/sh
#-------------------------------------------------------------------------------
# Apply the final net9010 models to the observations, one job:
#   nearby clouds (obs_clouds.json) with every final model,
#   MWISP (obs_mwisp.json) with the run-2 CO models,
#   Herschel dust (obs_dust.json) with the mom0 model, matched to the final_proj CO tiles.
# Outputs: ~/plots/observations_net9010[_mwisp]_<tag>, ~/plots/observations_net9010_dust_final
#-------------------------------------------------------------------------------
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
P=/home/x-nbisht1/plots
CMD="source ~/.bashrc; export PYTHONUNBUFFERED=1; set -e;"
for T in final_proj final_13co final_co13 final_co12 final_coboth; do
  CMD="$CMD python predict_observations_net9010.py --tag $T;"
done
CMD="$CMD python predict_observations_net9010.py --tag final_mom0 --mode mom0;"
for T in final_co13 final_co12 final_coboth; do
  CMD="$CMD python predict_observations_net9010.py --config obs_mwisp.json --tag $T;"
done
CMD="$CMD python predict_dust_net9010.py --tag final_mom0 --co_tiles_dir $P/observations_net9010_final_proj \
          --output_dir $P/observations_net9010_dust_final;"
sbatch -J obs_final -p shared -c 32 --mem=60G -t ${TIME:-06:00:00} -A phy240036 \
       -o /home/x-nbisht1/projects/p79d_dataset/models/logs/obs_final_%j.out --wrap "$CMD"
