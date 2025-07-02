#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Verifying single-line changes in argh/typos/*.h files"
echo "====================================================="

# Original file
ORIGINAL="argh/argh.h"

# Counter for summary
total_files=0
valid_files=0

# Loop through all modified files
for modified in argh/typos/argh_*.h; do
    if [ -f "$modified" ]; then
        total_files=$((total_files + 1))
        filename=$(basename "$modified")
        
        # Run diff and count changed lines
        # Using unified diff format with 0 context lines to get minimal output
        diff_output=$(diff -U0 "$ORIGINAL" "$modified" 2>/dev/null)
        
        # Count the number of changed lines (lines starting with + or - but not +++ or ---)
        changed_lines=$(echo "$diff_output" | grep -E '^[+-][^+-]' | wc -l)
        
        # Get the line number from diff output
        line_info=$(echo "$diff_output" | grep -E '^@@' | head -1)
        
        if [ "$changed_lines" -eq 2 ]; then
            # Extract line number from @@ -X,Y +A,B @@ format
            line_num=$(echo "$line_info" | sed -E 's/@@ -([0-9]+),[0-9]+ \+([0-9]+),[0-9]+ @@.*/\2/')
            
            # Get the actual changed lines
            old_line=$(echo "$diff_output" | grep '^-[^-]' | sed 's/^-//')
            new_line=$(echo "$diff_output" | grep '^+[^+]' | sed 's/^+//')
            
            echo -e "${GREEN}✓${NC} $filename - Exactly 1 line changed at line $line_num"
            echo -e "  ${RED}OLD:${NC} $old_line"
            echo -e "  ${GREEN}NEW:${NC} $new_line"
            valid_files=$((valid_files + 1))
        else
            echo -e "${RED}✗${NC} $filename - Expected 1 changed line, found $((changed_lines / 2))"
            
            # Show all changes if not exactly 1
            if [ "$changed_lines" -gt 0 ]; then
                echo "  Changes:"
                echo "$diff_output" | grep -E '^[+-][^+-]' | head -10
            fi
        fi
        echo ""
    fi
done

# Summary
echo "====================================================="
echo -e "Summary: ${GREEN}$valid_files${NC}/${total_files} files have exactly 1 line changed"

if [ "$valid_files" -eq "$total_files" ]; then
    echo -e "${GREEN}All files passed validation!${NC}"
    exit 0
else
    echo -e "${RED}Some files have more than 1 line changed!${NC}"
    exit 1
fi