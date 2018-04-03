#!/bin/sh

# FIXME: this is really slow, probably due to writing lots of small files
#        individually in parallel.
# TODO:  use $TMPDIR or otherwise batch the writes.

#$ -N augment_imgs
#$ -l h_vmem=2G
#$ -l h_rt=24:00:00
#$ -t 1-128
#$ -cwd
#$ -j y
#$ -o ~/augment.log

. /etc/profile.d/modules.sh

module load anaconda
source activate /exports/igmm/eddie/Drug-Discovery/scott/keras_env2

INPUT_FILE="all_subclass_paths.txt"
IMG_DIR=$(awk "NR==$SGE_TASK_ID" "$INPUT_FILE")

python augment_imageset.py $IMG_DIR
