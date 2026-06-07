#!/usr/bin/env bash
# Run from survey_work/common/ (after: conda activate neuralopv2)

set -euo pipefail

LOSS_PLOT="../problems/hyperelasticity/FNO/Results/loss_his.png"
RESULTS_DIR="../problems/hyperelasticity/FNO/Results"

while [[ ! -f "$LOSS_PLOT" ]]; do
  echo "Waiting for $LOSS_PLOT ..."
  inotifywait -q -e close_write,moved_to,create --include 'loss_his.png' "$RESULTS_DIR"
done

echo "Found $LOSS_PLOT — running pipeline"
python run_model_pipeline.py hyperelasticity --skip-training
