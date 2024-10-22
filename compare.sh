#! /usr/bin/env bash


GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
RESET=$(tput sgr0)

DIR1="data"
DIR2="data_sequential"

if [ ! -d "$DIR1" ]; then
    echo "Directory $DIR1 does not exist."
    exit 1
fi

if [ ! -d "$DIR2" ]; then
    echo "Directory $DIR2 does not exist."
    exit 1
fi

found_difference=1
for file in "$DIR1"/*.dat; do
    # Extract the file name (basename)
    filename=$(basename "$file")
    
    # Check if the corresponding file exists in DIR2
    if [ -f "$DIR2/$filename" ]; then
        # Compare the two files using diff
        diff_output=$(diff "$file" "$DIR2/$filename")
        
        if [ -n "$diff_output" ]; then
            echo "$diff_output"
            found_difference=0
        fi
    else
        echo "File $filename does not exist in $DIR2"
    fi
done

if [ $found_difference -eq 1 ]; then
    echo
    echo
    echo "${GREEN}The sequential and parallel version produced mathcing output!${RESET}"
    echo
    echo
else
    echo
    echo
    echo "${RED}There were mismatches between the sequential and the parallel output.${RESET}"
    echo
    echo
fi
