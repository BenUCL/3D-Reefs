#!/bin/bash
#
# merge_splats.sh
#
# Merge all gaussian splat patches into a single PLY file.
# Reads configuration from splat_config.yml.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MERGE_SCRIPT="$SCRIPT_DIR/merge_splats.py"
CONFIG_FILE="$SCRIPT_DIR/splat_config.yml"

# Check arguments
if [ $# -ne 0 ]; then
    echo "Usage: $0"
    echo ""
    echo "Configuration is read from splat_config.yml"
    echo "Set merge.output_file to specify output path"
    echo "Set merge.prefer_cleaned to prefer cleaned splats (default: true)"
    exit 1
fi

# Check files exist
if [ ! -f "$MERGE_SCRIPT" ]; then 
    echo "ERROR: merge_splats.py not found: $MERGE_SCRIPT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then 
    echo "ERROR: splat_config.yml not found: $CONFIG_FILE"
    exit 1
fi

# Parse config using Python to extract values
PATCHES_DIR=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['paths']['patches_dir'])")

# Setup logging
LOG_FILE="$PATCHES_DIR/splat_merge_log.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo "Merge Splats Log - $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo "Config: $CONFIG_FILE"

# Validate patches directory
if [ ! -d "$PATCHES_DIR" ]; then 
    echo "ERROR: Patches directory not found: $PATCHES_DIR"
    exit 1
fi

# Run merge
START_TIME=$(date +%s)

if python "$MERGE_SCRIPT" --config "$CONFIG_FILE"; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo "========================================================================"
    echo "Merge completed successfully in ${ELAPSED}s"
    echo "========================================================================"
    exit 0
else
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo "========================================================================"
    echo "Merge FAILED after ${ELAPSED}s (exit code: $EXIT_CODE)"
    echo "========================================================================"
    exit $EXIT_CODE
fi
