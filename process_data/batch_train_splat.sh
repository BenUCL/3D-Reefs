#!/bin/bash
#
# batch_train_splat.sh
#
# Train gaussian splats for all patches in a directory.
# If a patch fails, it logs the error and continues to the next one automatically.

set -e  # Exit on error for script syntax, but we handle command errors manually below

# Configuration
LICHTFELD_BIN="/home/ben/encode/code/lichtfeld-studio/build/LichtFeld-Studio"
TRAIN_SCRIPT="$(cd "$(dirname "$0")" && pwd)/train_splat.py"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <patches_dir> <images_dir> [lichtfeld_args...]"
    echo ""
    exit 1
fi

PATCHES_DIR="$1"
IMAGES_DIR="$2"
shift 2
LICHTFELD_ARGS="$@"

# Setup logging
LOG_FILE="$PATCHES_DIR/splat_training_log.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo "Batch Training Log - $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

# Validate inputs
if [ ! -d "$PATCHES_DIR" ]; then echo "❌ Error: Patches directory not found: $PATCHES_DIR"; exit 1; fi
if [ ! -d "$IMAGES_DIR" ]; then echo "❌ Error: Images directory not found: $IMAGES_DIR"; exit 1; fi
if [ ! -f "$LICHTFELD_BIN" ]; then echo "❌ Error: LichtFeld binary not found: $LICHTFELD_BIN"; exit 1; fi
if [ ! -f "$TRAIN_SCRIPT" ]; then echo "❌ Error: train_splat.py not found: $TRAIN_SCRIPT"; exit 1; fi

# Find all patch directories (p0, p1, p2, ...)
PATCHES=($(find "$PATCHES_DIR" -maxdepth 1 -type d -name 'p[0-9]*' | sort -V))

if [ ${#PATCHES[@]} -eq 0 ]; then
    echo "❌ Error: No patch directories (p0, p1, ...) found in $PATCHES_DIR"
    exit 1
fi

echo "Found ${#PATCHES[@]} patches to process."
echo ""

# Track successes and failures
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_PATCHES=()

# Process each patch
for PATCH_DIR in "${PATCHES[@]}"; do
    PATCH_NAME=$(basename "$PATCH_DIR")
    SPARSE_DIR="$PATCH_DIR/sparse/0"
    OUTPUT_DIR="$PATCH_DIR/sparse/splat"
    # Assuming the python script generates a report file, defining it here for grep later
    REPORT_FILE="$OUTPUT_DIR/report.json" 
    
    echo "========================================================================"
    echo "Training $PATCH_NAME"
    echo "========================================================================"
    
    # Check if sparse/0 exists
    if [ ! -d "$SPARSE_DIR" ]; then
        echo "⚠️  Skipping $PATCH_NAME: sparse/0 not found"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_PATCHES+=("$PATCH_NAME (no sparse/0)")
        continue
    fi
    
    # Check if already trained
    if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/point_cloud.ply" ]; then
        echo "⚠️  Output already exists: $OUTPUT_DIR/point_cloud.ply"
        # NOTE: This still asks to overwrite. If you want this fully automated, 
        # comment out the 'read' and 'if' block below and just use 'continue'.
        read -p "Overwrite? [y/N]: " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping $PATCH_NAME"
            continue
        fi
        echo "Removing existing output..."
        rm -rf "$OUTPUT_DIR"
    fi
    
    # Train this patch
    START_TIME=$(date +%s)
    PATCH_START=$(date '+%Y-%m-%d %H:%M:%S')
    
    # We use 'if' here so we can catch the exit code without set -e killing the script
    if python "$TRAIN_SCRIPT" \
        --lichtfeld "$LICHTFELD_BIN" \
        --sparse "$SPARSE_DIR" \
        --images "$IMAGES_DIR" \
        --output "$OUTPUT_DIR" \
        -- $LICHTFELD_ARGS; then
        
        # --- SUCCESS BLOCK ---
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        PATCH_END=$(date '+%Y-%m-%d %H:%M:%S')
        
        echo ""
        echo "✓ $PATCH_NAME completed in ${ELAPSED}s"
        echo "  Started:  $PATCH_START"
        echo "  Finished: $PATCH_END"
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
    else
        # --- FAILURE BLOCK ---
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        PATCH_END=$(date '+%Y-%m-%d %H:%M:%S')
        
        echo ""
        echo "❌ $PATCH_NAME FAILED after ${ELAPSED}s (exit code: $EXIT_CODE)"
        echo "  Started:  $PATCH_START"
        echo "  Failed:   $PATCH_END"
        
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_PATCHES+=("$PATCH_NAME (exit code: $EXIT_CODE, ${ELAPSED}s)")
    fi
    
    echo ""
done

# Final summary
BATCH_END=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "========================================================================"
echo "Batch Training Complete - $BATCH_END"
echo "========================================================================"
echo "Total patches: ${#PATCHES[@]}"
echo "Successful:    $SUCCESS_COUNT"
echo "Failed:        $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "❌ Failed patches summary:"
    for FAILED in "${FAILED_PATCHES[@]}"; do
        echo "  - $FAILED"
    done
    exit 1
else
    echo ""
    echo "✓ All patches processed successfully."
    exit 0
fi