#!/bin/sh
#-------------------------------------------------------------------------------
# Mirror the p79d scratch data (datasets, stored RADMC cubes, logs, tests) into
# projects, which is not purged. Incremental: rerun after new extractions.
# Files deleted on scratch are kept in the mirror (no --delete).
#
#   sh backup_scratch.sh            # submits one shared job
#-------------------------------------------------------------------------------
SRC=/home/x-nbisht1/scratch/projects/p79d_dataset/
DST=/home/x-nbisht1/projects/p79d_dataset/scratch_backup/
mkdir -p $DST
sbatch -J p79d_backup -p shared -c 4 --mem=8G -t ${TIME:-8:00:00} -A phy240036 \
       -o /home/x-nbisht1/projects/p79d_dataset/models/logs/backup_scratch_%j.out \
       --wrap "rsync -a --stats $SRC $DST && echo BACKUP_OK \$(date) && du -sh $DST"
