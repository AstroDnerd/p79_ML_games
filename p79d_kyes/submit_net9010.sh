#!/bin/sh
#-------------------------------------------------------------------------------
# Train + evaluate net9010 (Ms + chi) on simulation-held-out folds, one array
# task per fold, on a highmem node.
#
#   sbatch --export=ALL,MODE=allmom --array=0 submit_net9010.sh      # one fold
#   sbatch --export=ALL,MODE=mom0   --array=0-4 submit_net9010.sh    # all folds
#   (EXTRA="--chi_field chi_w" etc. passes options to the script)
#
# Then pool the folds:  python analyze_kfold_net9010.py --mode allmom
#-------------------------------------------------------------------------------
#SBATCH -J net9010
#SBATCH -o /home/x-nbisht1/projects/p79d_dataset/models/logs/net9010_%x_%A_%a.out
#SBATCH -e /home/x-nbisht1/projects/p79d_dataset/models/logs/net9010_%x_%A_%a.out
#SBATCH -p highmem
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --array=0
#SBATCH --mail-user=npb22a@fsu.edu
#SBATCH --mail-type=end
#SBATCH -A phy240036

source ~/.bashrc
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
python ViT_setup_test_net9010.py --mode ${MODE:-allmom} --fold $SLURM_ARRAY_TASK_ID $EXTRA
