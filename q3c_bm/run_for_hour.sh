#!/bin/bash

# Get the start time (in seconds since epoch)
start_time=$(date +%s)

duration=84600

# Loop until 30 minutes have passed
while true; do
    # Get current time
    current_time=$(date +%s)
    
    # Calculate elapsed time
    elapsed=$((current_time - start_time))
    
    # Check if 30 minutes have passed
    if [ $elapsed -ge $duration ]; then
        echo "30 minutes have passed. Stopping."
        break
    fi

    ./explain.sh sources/argh.h

done
