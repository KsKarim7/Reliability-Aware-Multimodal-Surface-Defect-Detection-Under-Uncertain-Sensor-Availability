#!/bin/bash
SEGMENT_FILE=/home/pub_766/MISDD-MM/.current_segment
SEGMENT=$(cat "$SEGMENT_FILE" 2>/dev/null || echo "1")

# a WSL VM kill bypasses the runner's EXIT trap and can leave a swapped
# ablation model.py behind — always start from the full model
cp /home/pub_766/MISDD-MM/MISDD_MM/model_full.py /home/pub_766/MISDD-MM/MISDD_MM/model.py

if [ "$SEGMENT" = "v5k_all" ] || [ "$SEGMENT" = "v5k_1" ] || [ "$SEGMENT" = "v5k_2" ] || [ "$SEGMENT" = "v5k_3" ]; then
    echo "Launching V5-3NN final-protocol campaign (${SEGMENT#v5k_}) at $(date)"
    bash /home/pub_766/MISDD-MM/run_v5_3nn.sh "${SEGMENT#v5k_}"
elif [ "$SEGMENT" = "v5_1" ] || [ "$SEGMENT" = "v5_2" ] || [ "$SEGMENT" = "v5_3" ]; then
    echo "Launching V5 fixed-scoring segment ${SEGMENT#v5_} at $(date)"
    bash /home/pub_766/MISDD-MM/run_v5_ablation.sh "${SEGMENT#v5_}"
elif [ "$SEGMENT" = "v4_1" ] || [ "$SEGMENT" = "v4_2" ] || [ "$SEGMENT" = "v4_3" ]; then
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
# propagate the segment's exit code to systemd so Restart=on-failure can fire
RC=$?
echo "Done at $(date) (exit $RC)"
exit $RC
