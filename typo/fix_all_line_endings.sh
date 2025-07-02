#!/bin/bash

# Batch script to fix all line endings without prompting

echo "Fixing line endings for all files..."
echo ""

# Fix all files with CRLF line endings
find . -type f -print0 | while IFS= read -r -d '' file; do
    # Skip binary files
    if file -b "$file" | grep -q "binary"; then
        continue
    fi
    
    # Check if file has CRLF
    if file "$file" | grep -q "CRLF" || od -c "$file" 2>/dev/null | grep -q $'\r'; then
        echo "Fixing: $file"
        # Create backup
        cp "$file" "$file.bak"
        # Fix line endings
        sed -i '' 's/\r$//' "$file" 2>/dev/null || sed -i 's/\r$//' "$file"
    fi
done

echo ""
echo "Done! Running verification..."
echo ""

# Verify results
./check_and_fix_line_endings.sh