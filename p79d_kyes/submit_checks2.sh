#!/bin/sh
#-------------------------------------------------------------------------------
# Second set of checks on the final models, one shared job:
#   MWISP-setup control (synthetic cubes observed like MWISP DR1 and passed through the real-data path)
#     for the 13CO, two-line and 12CO run-2 models, out of fold; the reference ("std") is the
#     MWISP-like synthetic set fed directly;
#   the same for the FCRAO setup with the two-line model;
#   linewidth diagnostic (diag_linewidth_net9010.py).
#-------------------------------------------------------------------------------
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
D=/home/x-nbisht1/scratch/projects/p79d_dataset
O=/home/x-nbisht1/plots/control_realpath_mwisp
MW="--beam_pix_obs 1.67 --chan_kms 0.16 --std_data $D/p79d_mach_grid_256_co_v2_mwisp.h5 --out $O --label _mwisp"
CMD="source ~/.bashrc; export PYTHONUNBUFFERED=1; set -e;
python control_realpath_net9010.py --tag final_co13 --lines 13co --noise 0.25 $MW;
python control_realpath_net9010.py --tag final_coboth --lines 13co,12co --noise 0.25,0.47 $MW;
python control_realpath_net9010.py --tag final_co12 --lines 12co --noise 0.47 $MW;
python control_realpath_net9010.py --tag final_coboth --lines 13co,12co --noise 0.2,0.25 --out /home/x-nbisht1/plots/control_realpath;
python diag_linewidth_net9010.py > /home/x-nbisht1/projects/p79d_dataset/models/diag_linewidth.txt"
sbatch -J checks2 -p shared -c 32 --mem=48G -t 04:00:00 -A phy240036 \
       -o /home/x-nbisht1/projects/p79d_dataset/models/logs/checks2_%j.out --wrap "$CMD"
