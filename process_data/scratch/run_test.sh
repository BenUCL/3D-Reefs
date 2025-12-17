#!/bin/bash
# Monitor training progress and document findings

echo "========================================================================"
echo "TRAINING TEST WITH CUDA FIXES"
echo "========================================================================"
echo "Date: $(date)"
echo ""
echo "Changes applied:"
echo "  1. CUDA cleanup (torch.cuda.empty_cache + synchronize) after each patch"
echo "  2. 5-second delay after successful training"
echo ""
echo "Starting batch training..."
echo "========================================================================"
echo ""

cd /home/ben/encode/code/3D-Reefs/process_data
./batch_train_splat.sh
