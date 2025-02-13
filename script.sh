#!/bin/bash

# Ensure an executable is provided
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

# Start, End, and Stride values
START=10000000
END=2000000000
STRIDE=80000000

# Write the header to the output file
echo "File-name,N,Sum,Num-of-runs,Mean-time(ms),Median-time(ms),Standard-deviation(ms)" > "$OUTPUT_FILE"

# Collect all valid file paths and run the executable
for ((i = START; i <= END; i += STRIDE)); do
    INPUT_FILE="${INPUT_DIR}/int_${i}.txt"

    # Check if file exists before proceeding
    if [[ -f "$INPUT_FILE" ]]; then
        echo "Processing: $INPUT_FILE"
        
        # Run the executable and capture the output
        OUTPUT=$("$EXECUTABLE" "$INPUT_FILE")
        
        # Extract the second line (ignoring the first line)
        RESULT=$(echo "$OUTPUT" | sed -n '2p')
        
        # Append the result to the output file
        echo "$RESULT" >> "$OUTPUT_FILE"
    else
        echo "Skipping: $INPUT_FILE (File not found)"
    fi
done

echo "Results have been saved to $OUTPUT_FILE"
