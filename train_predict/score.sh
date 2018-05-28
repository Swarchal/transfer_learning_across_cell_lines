#!/bin/bash

#$ -N score_predictions
#$ -l h_vmem=2G
#$ -l h_rt=01:00:00
#$ -t 1-8
#$ -hold_jid_ad run_predict_score
#$ -cwd

. /etc/profile.d/modules.sh

module load anaconda
source activate /exports/igmm/eddie/Drug-Discovery/scott/keras_env2


CELL_LINE_LIST="cell_lines.txt"
CELL_LINE=$(awk "NR==$SGE_TASK_ID" "$CELL_LINE_LIST")

PREDICTION_LOC=/exports/eddie/scratch/s1027820/predictions_"$CELL_LINE".tsv
OUTPUT_LOC=/exports/eddie/scratch/s1027820/scores_"$CELL_LINE".json

cd /exports/igmm/eddie/Drug-Discovery/scott/transfer_learning_across_cell_lines/train_predict/

python score.py "$PREDICTION_LOC" >> "$OUTPUT_LOC"

