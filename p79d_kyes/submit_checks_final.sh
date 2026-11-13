#!/bin/sh
#-------------------------------------------------------------------------------
# Inference-only checks on the final k-fold models (no training), one job:
#   1. paper_numbers_net9010.py with the linewidth baseline (now also with the thermal
#      and channel widths removed, and against the mass-weighted LOS Mach number);
#   2. compare_lvg_lte.py with the final run-2 models, out of fold;
#   3. control_realpath_net9010.py: synthetic cubes through the real-data path
#      (2 px beam, upsampled moment maps), 13CO and at Ophiuchus-like noise.
# Outputs: models/paper_numbers_final2.txt, ~/plots/lvg_check_final, ~/plots/control_realpath
#-------------------------------------------------------------------------------
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
M=/home/x-nbisht1/projects/p79d_dataset/models
CMD="source ~/.bashrc; export PYTHONUNBUFFERED=1;"
CMD="$CMD python control_realpath_net9010.py --n_per_sim 15;"
CMD="$CMD python control_realpath_net9010.py --n_per_sim 15 --noise 0.37 --out /home/x-nbisht1/plots/control_realpath_noise037;"
CMD="$CMD python compare_lvg_lte.py --obs fcrao500;"
CMD="$CMD python paper_numbers_net9010.py > $M/paper_numbers_final2.txt"
sbatch -J checks_final -p shared -c 16 --mem=48G -t ${TIME:-03:00:00} -A phy240036 \
       -o $M/logs/checks_final_%j.out --wrap "$CMD"
