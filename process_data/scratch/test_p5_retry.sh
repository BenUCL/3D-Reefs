#!/bin/bash
# Test if p5 failure is deterministic or random

cd /home/ben/encode/code/3D-Reefs/process_data

echo "Testing p5 training 3 times to check if error is deterministic..."
echo ""

for i in {1..3}; do
    echo "=================================================================="
    echo "Attempt $i"
    echo "=================================================================="
    
    # Remove old output
    rm -rf /home/ben/encode/data/intermediate_data/colmap5/sparse_patches/p5/sparse/splat
    
    # Try training
    if conda run -n mast3r-slam-blackwell python train_splat.py --config splat_config.yml --patch p5 2>&1 | tail -20; then
        echo "✓ SUCCESS on attempt $i"
        break
    else
        echo "❌ FAILED on attempt $i"
    fi
    
    echo ""
    sleep 2
done
