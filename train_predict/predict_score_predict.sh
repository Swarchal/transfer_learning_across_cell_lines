#!/bin/bash

#$ -N run_predict_score
#$ -l h_vmem=32G
#$ -l h_rt=12:00:00
#$ -t 1-8
#$ -hold_jid_ad predict_score_stage
#$ -cwd

. /etc/profile.d/modules.sh

module load anaconda
source activate /exports/igmm/eddie/Drug-Discovery/scott/keras_env2


CELL_LINE_LIST="cell_lines.txt"
CELL_LINE=$(awk "NR==$SGE_TASK_ID" "$CELL_LINE_LIST")

DATA_DIR=/exports/eddie/scratch/s1027820/nncell_data_stage_"$CELL_LINE"
CHECKPOINT=/exports/eddie/scratch/s1027820/checkpoint_history/checkpoints/"$CELL_LINE"_checkpoint
OUTPUT_LOC=/exports/eddie/scratch/s1027820/predictions_"$CELL_LINE".tsv


cd /exports/igmm/eddie/Drug-Discovery/scott/transfer_learning_across_cell_lines/train_predict/

OMP_NUM_THREADS=1 python predict_5_channel.py "$DATA_DIR" "$CHECKPOINT" >> "$OUTPUT_LOC"

rm -rf "$DATA_DIR"
