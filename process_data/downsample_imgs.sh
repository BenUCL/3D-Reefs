#!/bin/bash
#
# downsample_imgs.sh
#
# Preprocess images for gaussian splatting pipeline.
# Reads configuration from splat_config.yml.
#
# Usage:
#   ./downsample_imgs.sh [--config path/to/splat_config.yml]
#
# If --config is not provided, defaults to splat_config.yml in the same directory.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/splat_config.yml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config path/to/splat_config.yml]"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse YAML config using grep/sed (no Python dependencies)
# Extract values - paths are on same line as key
SOURCE_DIR=$(grep "raw_images_dir:" "$CONFIG_FILE" | sed 's/.*raw_images_dir:[[:space:]]*//' | sed 's/#.*//' | xargs)
OUTPUT_DIR=$(grep "processed_images_dir:" "$CONFIG_FILE" | sed 's/.*processed_images_dir:[[:space:]]*//' | sed 's/#.*//' | xargs)
MAX_DIM=$(grep "max_dimension:" "$CONFIG_FILE" | sed 's/.*max_dimension:[[:space:]]*//' | sed 's/#.*//' | xargs)
OUTPUT_FORMAT=$(grep "output_format:" "$CONFIG_FILE" | sed 's/.*output_format:[[:space:]]*//' | sed 's/#.*//' | xargs)
FILTER=$(grep -w "filter:" "$CONFIG_FILE" | sed 's/.*filter:[[:space:]]*//' | sed 's/#.*//' | xargs)

FILTER=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['image_preprocessing']['filter'])")

echo "========================================================================"
echo "Image Preprocessing for Gaussian Splatting Pipeline"
echo "========================================================================"
echo "Config:        $CONFIG_FILE"
echo "Source:        $SOURCE_DIR"
echo "Output:        $OUTPUT_DIR"
echo "Max dimension: ${MAX_DIM}px"
echo "Format:        $OUTPUT_FORMAT"
echo "Filter:        $FILTER"
echo ""

if ! command -v convert &> /dev/null; then
    echo "Error: ImageMagick is not installed. Please run: sudo apt install imagemagick"
    exit 1
fi

if ! command -v parallel &> /dev/null; then
    echo "Error: GNU parallel is not installed. Please run: sudo apt install parallel"
    exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source directory $SOURCE_DIR does not exist!"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

process_image() {
    local file="$1"
    local output_dir="$2"
    local max_dim="$3"
    local format="$4"
    local filter="$5"
    
    # Get filename without extension
    local filename=$(basename "$file")
    local name_no_ext="${filename%.*}"
    
    # Output file with specified format
    local output_file="$output_dir/${name_no_ext}.${format}"

    if [[ -f "$output_file" ]]; then
        echo "SKIP: $filename (already exists as $format)"
        return 0
    fi
    
    # Resize with specified filter and max dimension
    # Use single quotes to prevent shell interpretation of >
    if convert "$file" -filter "$filter" -resize "${max_dim}x${max_dim}>" "$output_file" 2>&1; then
        echo "✓ SUCCESS: $filename -> ${name_no_ext}.${format}"
        return 0
    else
        echo "✗ FAILED: $filename"
        return 1
    fi
}

export -f process_image

mapfile -t image_files < <(find "$SOURCE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tiff" -o -iname "*.tif" -o -iname "*.bmp" \))

total=${#image_files[@]}
echo "Found $total images to process..."

if [[ $total -eq 0 ]]; then
    echo "No images found in $SOURCE_DIR"
    exit 1
fi

num_cores=$(nproc)
max_jobs=$((num_cores > 1 ? num_cores - 1 : 1))
max_jobs=$((max_jobs > 8 ? 8 : max_jobs))

echo "Using $max_jobs parallel jobs"
echo "Starting parallel processing..."
echo ""

printf '%s\n' "${image_files[@]}" | parallel -j "$max_jobs" --bar process_image {} "$OUTPUT_DIR" "$MAX_DIM" "$OUTPUT_FORMAT" "$FILTER"

success_count=$(find "$OUTPUT_DIR" -name "*.${OUTPUT_FORMAT}" | wc -l)
echo ""
echo "========================================================================"
echo "Done! Processed $success_count images to $OUTPUT_FORMAT."
echo "Output directory: $OUTPUT_DIR"
echo "========================================================================"