#!/bin/bash
#
# batch_clean_splat.sh
#
# Clean gaussian splats for all patches using configuration from splat_config.yml.
# If a patch fails, it logs the error and continues to the next one automatically.
#
# TODO: Add support for saving disposed splats (filtered-out splats) to separate files
#       for quality verification and parameter tuning

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLEAN_SCRIPT="$SCRIPT_DIR/clean_splats.py"
CONFIG_FILE="$SCRIPT_DIR/splat_config.yml"

# Check arguments
if [ $# -ne 0 ]; then
    echo "Usage: $0"
    echo ""
    echo "Configuration is read from splat_config.yml"
    echo "Set cleanup.run_batch=true to clean all patches"
    echo "Set cleanup.run_batch=false to clean single patch specified in cleanup.single_patch"
    exit 1
fi

# Check files exist
if [ ! -f "$CLEAN_SCRIPT" ]; then 
    echo "ERROR: clean_splats.py not found: $CLEAN_SCRIPT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then 
    echo "ERROR: splat_config.yml not found: $CONFIG_FILE"
    exit 1
fi

# Parse config using Python to extract values
PATCHES_DIR=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['paths']['patches_dir'])")
RUN_BATCH=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(str(c['cleanup'].get('run_batch', True)).lower())")
SINGLE_PATCH=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['cleanup'].get('single_patch', 'p0'))")

# Setup logging
LOG_FILE="$PATCHES_DIR/splat_cleanup_log.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo "Batch Cleanup Log - $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo "Config: $CONFIG_FILE"
echo "Run batch: $RUN_BATCH"

# Validate patches directory
if [ ! -d "$PATCHES_DIR" ]; then 
    echo "ERROR: Patches directory not found: $PATCHES_DIR"
    exit 1
fi

# Determine which patches to clean
if [ "$RUN_BATCH" = "true" ]; then
    PATCHES=($(find "$PATCHES_DIR" -maxdepth 1 -type d -name 'p[0-9]*' | sort -V))
    
    if [ ${#PATCHES[@]} -eq 0 ]; then
        echo "ERROR: No patch directories (p0, p1, ...) found in $PATCHES_DIR"
        exit 1
    fi
    
    echo "Found ${#PATCHES[@]} patches to process"
else
    PATCH_PATH="$PATCHES_DIR/$SINGLE_PATCH"
    if [ ! -d "$PATCH_PATH" ]; then
        echo "ERROR: Patch directory not found: $PATCH_PATH"
        exit 1
    fi
    PATCHES=("$PATCH_PATH")
    echo "Cleaning single patch: $SINGLE_PATCH"
fi

echo ""

# Track successes and failures
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_PATCHES=()

# Process each patch
for PATCH_DIR in "${PATCHES[@]}"; do
    PATCH_NAME=$(basename "$PATCH_DIR")
    SPLAT_DIR="$PATCH_DIR/sparse/splat"
    
    echo "========================================================================"
    echo "Cleaning $PATCH_NAME"
    echo "========================================================================"
    
    # Check if splat directory exists
    if [ ! -d "$SPLAT_DIR" ]; then
        echo "WARNING: Skipping $PATCH_NAME: splat directory not found"
        echo "         Train the splat first using train_splat.py"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_PATCHES+=("$PATCH_NAME (no splat directory)")
        continue
    fi
    
    # Check if any splat_*.ply files exist
    SPLAT_FILES=$(find "$SPLAT_DIR" -maxdepth 1 -name 'splat_*.ply' -not -name '*_clean.ply' -not -name '*_disposed.ply' 2>/dev/null | wc -l)
    if [ "$SPLAT_FILES" -eq 0 ]; then
        echo "WARNING: Skipping $PATCH_NAME: no splat_*.ply files found"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_PATCHES+=("$PATCH_NAME (no splat files)")
        continue
    fi
    
    # Clean this patch
    START_TIME=$(date +%s)
    PATCH_START=$(date '+%Y-%m-%d %H:%M:%S')
    
    if python "$CLEAN_SCRIPT" --config "$CONFIG_FILE" --patch "$PATCH_NAME"; then
        
        # Success
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        PATCH_END=$(date '+%Y-%m-%d %H:%M:%S')
        
        echo ""
        echo "SUCCESS: $PATCH_NAME cleaned in ${ELAPSED}s"
        echo "  Started:  $PATCH_START"
        echo "  Finished: $PATCH_END"
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
    else
        # Failure
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        PATCH_END=$(date '+%Y-%m-%d %H:%M:%S')
        
        echo ""
        echo "FAILED: $PATCH_NAME after ${ELAPSED}s (exit code: $EXIT_CODE)"
        echo "  Started:  $PATCH_START"
        echo "  Failed:   $PATCH_END"
        
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_PATCHES+=("$PATCH_NAME (exit code: $EXIT_CODE, ${ELAPSED}s)")
        
        # Ask whether to continue (only in batch mode)
        if [ "$RUN_BATCH" = "true" ]; then
            read -p "Continue with remaining patches? [Y/n]: " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Nn]$ ]]; then
                echo "Aborting batch cleanup"
                break
            fi
        fi
    fi
    
    echo ""
done

# Final summary
BATCH_END=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "========================================================================"
echo "Batch Cleanup Complete - $BATCH_END"
echo "========================================================================"
echo "Total patches: ${#PATCHES[@]}"
echo "Successful:    $SUCCESS_COUNT"
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
