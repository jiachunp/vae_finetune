#!/bin/bash

# Check if raw_dir and output_dir are provided as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Please provide the raw videos directory and output directory as arguments."
    exit 1
fi

# Set the directory containing raw videos and the output directory
raw_dir=$1
output_dir=$2

# Create the merged_videos directory if it does not exist
mkdir -p "$output_dir"

# Loop through all mp4 video files
files=( "$raw_dir"/*.mp4 )
total=${#files[@]}
echo "Total files: $total"
current=0
merged=0

for video in "${files[@]}"; do
    # Extract the base name without the file extension and specific formats
    base_name=$(echo "$video" | sed -E 's/(.*)(\.[^.]+)\.mp4$/\1/')

    # Find the matching audio file
    audio=$(ls "${base_name}.mp4."* | head -n 1)

    if [[ -n "$audio" ]]; then
        # Define the output filename
        base_name=$(basename "$base_name")
        output="${output_dir}/${base_name}.mp4"
        # Run ffmpeg to merge the video and audio
        ffmpeg -i "$video" -i "$audio" -c:v copy -c:a aac -strict experimental "$output" -loglevel quiet
        merged=$((merged + 1))
    else
        echo "No matching audio file found for $video"
    fi
    current=$((current + 1))
    # Progress bar
    percent=$((100 * current / total))
    printf "\r[%-50s] %d%%" $(printf '#%.0s' $(seq 1 $((percent / 2)))) $percent
done

echo ""
echo "Total files: $total."
echo "Merged $merged videos."
