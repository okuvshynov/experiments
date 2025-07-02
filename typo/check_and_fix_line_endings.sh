#!/bin/bash

# Script to check and fix line endings in files
# Makes files more portable by converting to Unix (LF) line endings

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Line Ending Checker and Fixer"
echo "=============================="
echo ""

# Function to check line endings
check_line_endings() {
    local file="$1"
    
    # Check if file exists
    if [ ! -f "$file" ]; then
        return
    fi
    
    # Check for CRLF (Windows) line endings
    if file "$file" | grep -q "CRLF"; then
        echo -e "${RED}CRLF${NC} (Windows): $file"
        return 1
    fi
    
    # Check using od command for more detailed analysis
    if od -c "$file" 2>/dev/null | grep -q $'\r'; then
        echo -e "${RED}CR/CRLF${NC} detected: $file"
        return 1
    fi
    
    # Check file encoding
    encoding=$(file -b --mime-encoding "$file" 2>/dev/null)
    
    # If it's binary, skip it
    if file -b "$file" | grep -q "binary"; then
        echo -e "${YELLOW}Binary file${NC} (skipped): $file"
        return 2
    fi
    
    echo -e "${GREEN}OK${NC} (Unix LF, $encoding): $file"
    return 0
}

# Function to fix line endings
fix_line_endings() {
    local file="$1"
    
    # Skip binary files
    if file -b "$file" | grep -q "binary"; then
        echo "  Skipping binary file: $file"
        return
    fi
    
    # Create backup
    cp "$file" "$file.bak"
    
    # Convert to Unix line endings
    # Using multiple methods for compatibility
    if command -v dos2unix >/dev/null 2>&1; then
        dos2unix -q "$file"
        echo "  Fixed using dos2unix: $file"
    elif command -v sed >/dev/null 2>&1; then
        # Use sed to remove carriage returns
        sed -i '' 's/\r$//' "$file" 2>/dev/null || sed -i 's/\r$//' "$file"
        echo "  Fixed using sed: $file"
    elif command -v perl >/dev/null 2>&1; then
        # Use perl as fallback
        perl -pi -e 's/\r\n/\n/g' "$file"
        echo "  Fixed using perl: $file"
    else
        echo "  ERROR: No suitable tool found to fix line endings"
        mv "$file.bak" "$file"
        return 1
    fi
    
    # Remove backup if fix was successful
    rm -f "$file.bak"
}

# Main script
echo "Checking files in current directory and subdirectories..."
echo ""

# Arrays to store results
declare -a files_with_crlf=()
declare -a files_ok=()
declare -a binary_files=()

# Check all text files
while IFS= read -r -d '' file; do
    result=$(check_line_endings "$file")
    exit_code=$?
    
    if [ $exit_code -eq 1 ]; then
        files_with_crlf+=("$file")
    elif [ $exit_code -eq 0 ]; then
        files_ok+=("$file")
    else
        binary_files+=("$file")
    fi
done < <(find . -type f -print0)

echo ""
echo "Summary:"
echo "--------"
echo "Files with Unix line endings (LF): ${#files_ok[@]}"
echo "Files with Windows line endings (CRLF): ${#files_with_crlf[@]}"
echo "Binary files (skipped): ${#binary_files[@]}"
echo ""

# If there are files with CRLF, offer to fix them
if [ ${#files_with_crlf[@]} -gt 0 ]; then
    echo "Files that need fixing:"
    printf '%s\n' "${files_with_crlf[@]}"
    echo ""
    
    read -p "Do you want to fix these files? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Fixing line endings..."
        for file in "${files_with_crlf[@]}"; do
            fix_line_endings "$file"
        done
        
        echo ""
        echo "Done! Backups were created with .bak extension."
        echo "To remove all backups: find . -name '*.bak' -delete"
    else
        echo "No changes made."
    fi
else
    echo "All text files already have Unix line endings!"
fi

echo ""
echo "Additional file information:"
echo "---------------------------"
# Show encoding information for text files
find . -type f \( -name "*.txt" -o -name "*.sh" -o -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.c" \) -exec file {} \; | head -20