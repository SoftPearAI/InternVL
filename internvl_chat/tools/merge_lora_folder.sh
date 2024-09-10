#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_dir>"
    exit 1
fi

# Assign arguments to variables
INPUT_DIR=$1

# Iterate over files matching 'checkpoint*' in the input directory
for file in "$INPUT_DIR"/checkpoint*; do
    # Extract the base name of the file
    BASENAME=$(basename "$file")

    # Run the Python script with the current file and output directory
    python3 tools/merge_lora.py "$file" "$INPUT_DIR/$BASENAME-merged"

    # Check if the Python script ran successfully
    if [ $? -ne 0 ]; then
        echo "An error occurred while processing $file"
        exit 1
    fi
done

echo "All files processed successfully."
