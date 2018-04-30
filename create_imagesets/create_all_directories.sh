#! /bin/bash

#$ -N stage_excl_data
#$ -q staging
#$ -l h_rt=12:00:00
#$ -t 1-8
#$ -cwd

SEEDFILE="cell_lines.txt"
CELL_LINE=$(awk "NR==$SGE_TASK_ID" "$SEEDFILE")
RSYNC_FILE_PATH=rsync_exl_"$CELL_LINE".sh

./create_rsync_commands.py $CELL_LINE > $RSYNC_FILE_PATH
bash $RSYNC_FILE_PATH
rm $RSYNC_FILE_PATH
