#!/bin/bash

declare -a arr=("MDA-231" "MDA-157" "KPL4" "MCF7" "SKBR3" "T47D" "HCC1569" "HCC1954")

for cell_line in "${arr[@]}"
do
    echo "running $cell_line\n"
    python load_model_not_exclusions.py $cell_line
done

