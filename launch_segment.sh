#!/bin/bash
SEGMENT_FILE=/home/pub_766/MISDD-MM/.current_segment
SEGMENT=$(cat "$SEGMENT_FILE" 2>/dev/null || echo "1")

if [ "$SEGMENT" = "v4_1" ] || [ "$SEGMENT" = "v4_2" ] || [ "$SEGMENT" = "v4_3" ]; then
    echo "Launching V4 corrected-pipeline segment ${SEGMENT#v4_} at $(date)"
    bash /home/pub_766/MISDD-MM/run_v4_ablation.sh "${SEGMENT#v4_}"
elif [ "$SEGMENT" = "v3" ]; then
    echo "Launching V3 full_model run at $(date)"
    bash /home/pub_766/MISDD-MM/run_v3_fullmodel.sh
elif [ "$SEGMENT" = "rate_sweep" ]; then
    echo "Launching missing rate sweep at $(date)"
    bash /home/pub_766/MISDD-MM/run_missing_rate_sweep.sh
else
    echo "Systemd launching segment $SEGMENT at $(date)"
    bash /home/pub_766/MISDD-MM/run_v2_segment.sh $SEGMENT
fi
echo "Done at $(date)"
