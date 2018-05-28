while read CELL_LINE; do
    JSON_PATH=/exports/eddie/scratch/s1027820/scores_"$CELL_LINE".json
    python plot_cm.py "$JSON_PATH"
done < cell_lines.txt
