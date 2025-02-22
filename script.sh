#!/bin/bash

# Ensure an executable and result file are provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <executable> <result-file>"
    exit 1
fi

# Executable from command line
EXECUTABLE="$1"

# Base path
INPUT_DIR="../input"

# Output file
OUTPUT_FILE="$2"

# Write the header to the output file
echo "File-name,N,Sum,Num-of-runs,Mean-time(ms),Median-time(ms),Standard-deviation(ms)" > "$OUTPUT_FILE"

# Find and sort all valid input files
INPUT_FILES=$(ls "$INPUT_DIR"/int_*.txt 2>/dev/null | sed -E 's/.*int_([0-9]+)\.txt/\1 \0/' | sort -n | cut -d' ' -f2)

# Process each file
for INPUT_FILE in $INPUT_FILES; do
    echo "Processing: $INPUT_FILE"
    
    # Run the executable and capture the output
    OUTPUT=$("$EXECUTABLE" "$INPUT_FILE")
    
    # Extract the second line (ignoring the first line)
    RESULT=$(echo "$OUTPUT" | sed -n '2p')
    
    # Append the result to the output file
    echo "$RESULT" >> "$OUTPUT_FILE"

done

echo "Results have been saved to $OUTPUT_FILE"

