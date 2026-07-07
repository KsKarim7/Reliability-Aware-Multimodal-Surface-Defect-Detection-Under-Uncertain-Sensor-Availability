#!/bin/bash
# Called by systemd — reads which segment to run from a state file
SEGMENT_FILE=/home/pub_766/MISDD-MM/.current_segment
if [ ! -f "$SEGMENT_FILE" ]; then
    echo "1" > "$SEGMENT_FILE"
fi
SEGMENT=$(cat "$SEGMENT_FILE")
echo "Systemd launching segment $SEGMENT at $(date)"
bash /home/pub_766/MISDD-MM/run_v2_segment.sh $SEGMENT
echo "Segment $SEGMENT finished at $(date)"
