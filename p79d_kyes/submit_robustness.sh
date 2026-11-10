#!/bin/sh
#-------------------------------------------------------------------------------
# Resolution-degradation and MC-dropout uncertainty checks of the k-fold
# net9010 models (robustness_net9010.py), one array task per model group.
#   sbatch --array=0-2 submit_robustness.sh
#-------------------------------------------------------------------------------
#SBATCH -J robust9010
#SBATCH -o /home/x-nbisht1/projects/p79d_dataset/models/logs/robust9010_%A_%a.out
#SBATCH -e /home/x-nbisht1/projects/p79d_dataset/models/logs/robust9010_%A_%a.out
#SBATCH -p wholenode
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --mail-user=npb22a@fsu.edu
#SBATCH --mail-type=end
#SBATCH -A phy240036

source ~/.bashrc
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
D=/home/x-nbisht1/scratch/projects/p79d_dataset
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
case $SLURM_ARRAY_TASK_ID in
  0) A="--tag final_proj --data_file $D/p79d_mach_grid_256_v1.h5 --image_key images" ;;
  1) A="--tag final_13co --data_file $D/p79d_mach_grid_256_13co_v1_fcrao500_v2.h5 --image_key images_obs" ;;
  2) A="--tag final_coboth --data_file $D/p79d_mach_grid_256_co_v2_fcrao500.h5 --image_key images_obs_13co+images_obs_12co" ;;
esac
python robustness_net9010.py --mode allmom --dens_head $A $EXTRA
