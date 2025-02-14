#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p ~/.pwr

# Initialize sequence number
seq_file=~/.pwr/.sequence
if [[ ! -f "$seq_file" ]]; then
    echo 0 > "$seq_file"
fi

# Function to get and increment sequence number atomically
get_next_sequence() {
    local seq
	seq=$(<"$seq_file")
	echo $((seq + 1)) > "$seq_file"
	echo "$seq"
}

# Function to extract and save individual measurements
process_powermetrics() {
    # Use tr to remove null bytes, then process line by line
    tr -d '\000' | while IFS= read -r line; do
        if [[ $line == "<?xml"* ]]; then
            timestamp=$(date +%Y%m%d_%H%M%S)
            seq=$(get_next_sequence)
            current_file=~/.pwr/${timestamp}.$(printf "%06d" "$seq").xml
            echo "$line" > "$current_file"
        else
            # Append to current file
            echo "$line" >> "$current_file"
        fi
    done
}

# Run powermetrics and pipe its output through our processing function
sudo powermetrics -f plist -s cpu_power,gpu_power,ane_power,network,disk -i 1000 | process_powermetrics
