#!/bin/bash

#$ -N predict_score_stage
#$ -l h_vmem=1G
#$ -l h_rt=24:00:00
#$ -t 1-8
#$ -q staging
#$ -tc 1
#$ -p -1000
#$ -cwd

CELL_LINE_LIST="cell_lines.txt"
CELL_LINE=$(awk "NR==$SGE_TASK_ID" "$CELL_LINE_LIST")

SOURCE=/exports/igmm/datastore/Drug-Discovery/scott/2018-04-24_nncell_data_300_"$CELL_LINE"
DESTINATION=/exports/eddie/scratch/s1027820/nncell_data_stage_"$CELL_LINE"

cd /exports/igmm/eddie/Drug-Discovery/scott/transfer_learning_across_cell_lines/create_imagesets

python create_test_dirs.py "$SOURCE" "$DESTINATION"

