#!/bin/bash

# Default values
ANALYSIS=""
BENCHMARK=""
TIMESTAMP=""
TYPE="evolution_g_space"  # Default type
FRAMERATE=10
OUTPUT_FILE=""
NUM_JOBS=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)  # Auto-detect CPU cores

# Function to display usage
usage() {
    echo "Usage: $0 --analysis=\"analysis\" --benchmark=\"benchmark\" --timestamp=\"timestamp\" [options]"
    echo "Options:"
    echo "  --type=TYPE             Type of animation (default: evolution_g_space)"
    echo "  --output-file=FILE      Output filename (default: evo_g_space_TIMESTAMP.mov)"
    echo "  --framerate=FPS         Frames per second (default: 10)"
    echo "  --jobs=N                Number of parallel conversion jobs (default: $NUM_JOBS)"
    echo "  --help                  Display this help message"
    exit 1
}

# Parse command line arguments
for i in "$@"; do
    case $i in
        --analysis=*)
            ANALYSIS="${i#*=}"
            shift
            ;;
        --benchmark=*)
            BENCHMARK="${i#*=}"
            shift
            ;;
        --timestamp=*)
            TIMESTAMP="${i#*=}"
            shift
            ;;
        --type=*)
            TYPE="${i#*=}"
            shift
            ;;
        --output-file=*)
            OUTPUT_FILE="${i#*=}"
            shift
            ;;
        --framerate=*)
            FRAMERATE="${i#*=}"
            shift
            ;;
        --jobs=*)
            NUM_JOBS="${i#*=}"
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $i"
            usage
            ;;
    esac
done

# Check required parameters
if [ -z "$ANALYSIS" ] || [ -z "$BENCHMARK" ] || [ -z "$TIMESTAMP" ]; then
    echo "Error: analysis, benchmark, and timestamp are required parameters."
    usage
fi

# Construct directory paths
INPUT_DIR="./results/${ANALYSIS}/${BENCHMARK}/${TIMESTAMP}/evolution/${TYPE}"
OUT_DIR="./results/${ANALYSIS}/${BENCHMARK}/${TIMESTAMP}/evolution"

# Make sure output directory exists
mkdir -p "$OUT_DIR"

# Set default output filename if not provided
if [ -z "$OUTPUT_FILE" ]; then
    # Extract the base name from TYPE without any directory components
    TYPE_BASE=$(basename "$TYPE")
    OUTPUT_FILE="${OUT_DIR}/${TYPE_BASE}_${TIMESTAMP}.mov"
else
    # If the output file doesn't include a path, put it in OUT_DIR
    if [[ "$OUTPUT_FILE" != */* ]]; then
        OUTPUT_FILE="${OUT_DIR}/${OUTPUT_FILE}"
    fi
fi

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

echo "Processing PDF frames from directory: $INPUT_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Framerate: $FRAMERATE fps"
echo "Analysis: $ANALYSIS"
echo "Benchmark: $BENCHMARK"
echo "Timestamp: $TIMESTAMP"
echo "Type: $TYPE"
echo "Using $NUM_JOBS parallel conversion jobs"

# Determine whether to use 'convert' or 'magick'
if command -v magick &> /dev/null; then
    echo "Using ImageMagick v7+ (magick command)"
    CONVERT_CMD="magick"
else
    echo "Using ImageMagick v6 or earlier (convert command)"
    CONVERT_CMD="convert"
fi

# Create a temporary directory for converted images
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Create a directory for processing logs
LOG_DIR="${TEMP_DIR}/logs"
mkdir -p "$LOG_DIR"

# Find all PDF files in the input directory and sort them numerically
echo "Finding PDF files..."
PDF_FILES_LIST=$(mktemp)
find "$INPUT_DIR" -name "*.pdf" | sort -V > "$PDF_FILES_LIST"

# Count how many files we found
NUM_FILES=$(wc -l < "$PDF_FILES_LIST")
if [ "$NUM_FILES" -eq 0 ]; then
    echo "Error: No PDF files found in $INPUT_DIR"
    rm "$PDF_FILES_LIST"
    rmdir "$TEMP_DIR"
    exit 1
fi

echo "Found $NUM_FILES PDF files"

# Function to convert a single PDF to PNG
convert_pdf() {
    local pdf_file="$1"
    local output_file="$2"
    local log_file="$3"

    if [ "$CONVERT_CMD" = "magick" ]; then
        magick -density 300 "$pdf_file" -quality 100 "$output_file" 2>"$log_file"
    else
        convert -density 300 "$pdf_file" -quality 100 "$output_file" 2>"$log_file"
    fi

    return $?
}

# Convert PDF files to PNG images using parallel processing
echo "Converting PDF files to images using $NUM_JOBS parallel jobs..."

# Function to wait until we have fewer than N processes running
wait_for_jobs() {
    local max_jobs=$1
    local current_jobs

    while true; do
        # Count running background jobs
        current_jobs=$(jobs -p | wc -l)

        if [ "$current_jobs" -lt "$max_jobs" ]; then
            break
        fi

        # Wait a little bit before checking again
        sleep 0.1
    done
}

# Process conversion in parallel
COUNTER=0
while IFS= read -r pdf_file; do
    output_file=$(printf "%s/frame_%04d.png" "$TEMP_DIR" "$COUNTER")
    log_file="${LOG_DIR}/convert_${COUNTER}.log"

    # Wait if we have too many jobs running
    wait_for_jobs "$NUM_JOBS"

    # Start conversion in background
    echo "Starting conversion of $(basename "$pdf_file") (job $COUNTER)"
    convert_pdf "$pdf_file" "$output_file" "$log_file" &

    COUNTER=$((COUNTER+1))
done < "$PDF_FILES_LIST"

# Wait for all background jobs to finish
echo "Waiting for all conversion jobs to complete..."
wait

# Check for conversion errors
echo "Checking for conversion errors..."
ERROR_COUNT=0
for log_file in "$LOG_DIR"/*.log; do
    if [ -s "$log_file" ]; then
        ERROR_COUNT=$((ERROR_COUNT+1))
        echo "Error in conversion job: $(cat "$log_file")"
    fi
done

if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "Warning: $ERROR_COUNT conversion jobs reported errors."
fi

# Clean up the files list
rm -f "$PDF_FILES_LIST"

# Count converted images
NUM_IMAGES=$(find "$TEMP_DIR" -maxdepth 1 -name "frame_*.png" | wc -l)
echo "Successfully converted $NUM_IMAGES PDF files to PNG images"

if [ $NUM_IMAGES -eq 0 ]; then
    echo "Error: No images were created"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Check if we converted fewer images than PDFs
if [ "$NUM_IMAGES" -lt "$NUM_FILES" ]; then
    echo "Warning: Only $NUM_IMAGES out of $NUM_FILES PDFs were successfully converted."
    echo "Continuing with available images..."
fi

# Create metadata file for easier handling
META_FILE="${TEMP_DIR}/metadata.txt"
echo ";FFMETADATA1" > "$META_FILE"
echo "title=Generated video" >> "$META_FILE"
echo "comment=Analysis: ${ANALYSIS}, Benchmark: ${BENCHMARK}, Timestamp: ${TIMESTAMP}" >> "$META_FILE"

# Run ffmpeg with metadata, force overwrite
echo "Running ffmpeg to create video..."
ffmpeg -y -framerate "$FRAMERATE" -i "${TEMP_DIR}/frame_%04d.png" \
       -i "$META_FILE" -map_metadata 1 \
       -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le \
       "$OUTPUT_FILE"

# Check if ffmpeg succeeded
FFMPEG_STATUS=$?

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

# Check the result
if [ $FFMPEG_STATUS -eq 0 ]; then
    # Convert to absolute path for better clickability in terminals
    ABSOLUTE_PATH=$(cd "$OUT_DIR" && pwd)
    echo "Conversion successful. Output saved to: $ABSOLUTE_PATH"
    exit 0
else
    echo "Error: ffmpeg conversion failed with status $FFMPEG_STATUS"
    exit 1
fi
