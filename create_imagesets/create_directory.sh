#! /bin/bash

#$ -N make_dir
#$ -q staging
#$ -l h_vmem=1G
#$ -l h_rt=06:00:00
#$ -M aes
#$ -cwd

. /etc/profile.d/modules.sh

CELL_LINE="$1"

python ~/nn_loo/create_rsync_commands.py "$CELL_LINE" > ~/nn_loo/rsync_cmnds_excl_"$CELL_LINE".sh

bash ~/nn_loo/rsync_cmnds_excl_"$CELL_LINE".sh
rm ~/nn_loo/rsync_cmnds_excl_"$CELL_LINE".sh
