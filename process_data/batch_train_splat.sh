#!/bin/bash
#
# batch_train_splat.sh
#
# Train gaussian splats for all patches using a config file.
# If a patch fails, it logs the error and continues to the next one automatically.
#
# Usage: ./batch_train_splat.sh --config <path_to_config.yml>

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_splat.py"

# Parse arguments
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --config <path_to_config.yml>"
            exit 1
            ;;
    esac
done

# Check config argument was provided
if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: --config argument is required"
    echo ""
    echo "Usage: $0 --config <path_to_config.yml>"
    echo ""
    echo "Example:"
    echo "  $0 --config splat_config.yml"
    echo "  $0 --config /path/to/splat_config_redwood1.yml"
    exit 1
fi

# Check files exist
if [ ! -f "$TRAIN_SCRIPT" ]; then 
    echo "ERROR: train_splat.py not found: $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then 
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse config using Python to extract values
PATCHES_DIR=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['paths']['patches_dir'])")
RUN_BATCH=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(str(c['training'].get('run_batch', True)).lower())")
# single_patch can be string "p0" or list ["p0", "p7"] - normalize to space-separated string
SINGLE_PATCHES=$(python3 -c "
import yaml
c = yaml.safe_load(open('$CONFIG_FILE'))
sp = c['training'].get('single_patch', 'p0')
if isinstance(sp, list):
    print(' '.join(sp))
else:
    print(sp)
")

# Setup logging
LOG_FILE="$PATCHES_DIR/splat_training_log.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo "Batch Training Log - $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo "Config: $CONFIG_FILE"
echo "Run batch: $RUN_BATCH"

# Validate patches directory
if [ ! -d "$PATCHES_DIR" ]; then 
    echo "ERROR: Patches directory not found: $PATCHES_DIR"
    exit 1
fi

# Determine which patches to train
if [ "$RUN_BATCH" = "true" ]; then
    PATCHES=($(find "$PATCHES_DIR" -maxdepth 1 -type d -name 'p[0-9]*' | sort -V))
    
    if [ ${#PATCHES[@]} -eq 0 ]; then
        echo "ERROR: No patch directories (p0, p1, ...) found in $PATCHES_DIR"
        exit 1
    fi
    
    echo "Found ${#PATCHES[@]} patches to process"
else
    # Build array from space-separated patch names
    PATCHES=()
    for PATCH_NAME in $SINGLE_PATCHES; do
        PATCH_PATH="$PATCHES_DIR/$PATCH_NAME"
        if [ ! -d "$PATCH_PATH" ]; then
            echo "ERROR: Patch directory not found: $PATCH_PATH"
            exit 1
        fi
        PATCHES+=("$PATCH_PATH")
    done
    echo "Training ${#PATCHES[@]} patch(es): $SINGLE_PATCHES"
fi

echo ""

# Check for existing splats in patches
EXISTING_SPLATS=()
for PATCH_DIR in "${PATCHES[@]}"; do
    PATCH_NAME=$(basename "$PATCH_DIR")
    OUTPUT_DIR="$PATCH_DIR/sparse/splat"
    
    # Check if this patch has any splat_*.ply or prefixed p*_splat_*.ply files
    if [ -d "$OUTPUT_DIR" ] && (compgen -G "$OUTPUT_DIR/splat_*.ply" > /dev/null || compgen -G "$OUTPUT_DIR/${PATCH_NAME}_splat_*.ply" > /dev/null); then
        EXISTING_SPLATS+=("$PATCH_NAME")
    fi
done

# If some patches already have splats, ask user what to do
SKIP_EXISTING=false
if [ ${#EXISTING_SPLATS[@]} -gt 0 ]; then
    echo "⚠️  Found existing splats in ${#EXISTING_SPLATS[@]} patch(es):"
    for EXISTING in "${EXISTING_SPLATS[@]}"; do
        echo "    - $EXISTING"
    done
    echo ""
    read -p "Continue with remaining patches (press Y to skip completed, press N to retrain all)? [Y/n]: " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        SKIP_EXISTING=true
        echo "✓ Will skip patches with existing splats"
    else
        echo "✓ Will retrain all patches from the beginning"
    fi
    echo ""
fi

# Track successes and failures
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_PATCHES=()
SKIPPED_COUNT=0

# Process each patch
for PATCH_DIR in "${PATCHES[@]}"; do
    PATCH_NAME=$(basename "$PATCH_DIR")
    SPARSE_DIR="$PATCH_DIR/sparse/0"
    OUTPUT_DIR="$PATCH_DIR/sparse/splat"
    
    echo "========================================================================"
    echo "Training $PATCH_NAME"
    echo "========================================================================"
    
    # Skip if already exists and user chose to skip existing
    if [ "$SKIP_EXISTING" = true ] && [ -d "$OUTPUT_DIR" ] && (compgen -G "$OUTPUT_DIR/splat_*.ply" > /dev/null || compgen -G "$OUTPUT_DIR/${PATCH_NAME}_splat_*.ply" > /dev/null); then
        echo "✓ Skipping $PATCH_NAME (already trained)"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        echo ""
        continue
    fi
    
    # Check if sparse/0 exists
    if [ ! -d "$SPARSE_DIR" ]; then
        echo "WARNING: Skipping $PATCH_NAME: sparse/0 not found"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_PATCHES+=("$PATCH_NAME (no sparse/0)")
        continue
    fi
    
    # Check if already trained
    if [ -d "$OUTPUT_DIR" ] && (compgen -G "$OUTPUT_DIR/splat_*.ply" > /dev/null || compgen -G "$OUTPUT_DIR/${PATCH_NAME}_splat_*.ply" > /dev/null); then
        echo "⚠️  WARNING: Output already exists in $OUTPUT_DIR"
        read -p "Overwrite? [y/N]: " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping $PATCH_NAME"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            echo ""
            continue
        fi
        echo "Removing existing output..."
        rm -rf "$OUTPUT_DIR"
    fi
    
    # Train this patch
    START_TIME=$(date +%s)
    PATCH_START=$(date '+%Y-%m-%d %H:%M:%S')
    
    if python "$TRAIN_SCRIPT" --config "$CONFIG_FILE" --patch "$PATCH_NAME"; then
        
        # Success - extract metrics from run_report.txt
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        PATCH_END=$(date '+%Y-%m-%d %H:%M:%S')
        
        REPORT_FILE="$OUTPUT_DIR/run_report.txt"
        FINAL_LOSS=""
        FINAL_SPLATS=""
        NUM_IMAGES=""
        
        if [ -f "$REPORT_FILE" ]; then
            FINAL_LOSS=$(grep -oP 'Loss:\s*\K[0-9.eE+-]+' "$REPORT_FILE" | tail -1 || echo "")
            FINAL_SPLATS=$(grep -oP 'Splats:\s*\K[0-9]+' "$REPORT_FILE" | tail -1 || echo "")
            NUM_IMAGES=$(grep -oP '"num_images":\s*\K[0-9]+' "$REPORT_FILE" | head -1 || echo "")
        fi
        
        echo ""
        echo "SUCCESS: $PATCH_NAME completed in ${ELAPSED}s"
        [ -n "$FINAL_LOSS" ] && echo "  Final Loss: $FINAL_LOSS"
        [ -n "$FINAL_SPLATS" ] && echo "  Final Splats: $FINAL_SPLATS"
        [ -n "$NUM_IMAGES" ] && echo "  Images: $NUM_IMAGES"
        echo "  Started:  $PATCH_START"
        echo "  Finished: $PATCH_END"
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # Add delay to allow GPU memory to be fully freed
        echo "  Waiting 5s for GPU cleanup..."
        sleep 5
        
    else
        # Failure
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        PATCH_END=$(date '+%Y-%m-%d %H:%M:%S')
        
        echo ""
        echo "⚠️  FAILED: $PATCH_NAME after ${ELAPSED}s (exit code: $EXIT_CODE)"
        echo "  Started:  $PATCH_START"
        echo "  Failed:   $PATCH_END"
        echo ""
        echo "⚠️  Continuing to next patch..."
        
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
echo "Skipped:       $SKIPPED_COUNT"
echo "Failed:        $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed patches:"
    for FAILED in "${FAILED_PATCHES[@]}"; do
        echo "  - $FAILED"
    done
    exit 1
else
    echo ""
    echo "All patches processed successfully"
    exit 0
fi