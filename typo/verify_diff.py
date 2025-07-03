#!/usr/bin/env python3

import sys
import re
from typing import Tuple, Optional, Dict, Any

def parse_diff(diff_content: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse diff output to extract change line and old/new content."""
    lines = diff_content.strip().split('\n')
    
    change_line = None
    old_line = None
    new_line = None
    
    for line in lines:
        # Match line number change (e.g., "362c362")
        if re.match(r'^\d+c\d+$', line):
            change_line = line
        # Match old line (starts with '<')
        elif line.startswith('< '):
            old_line = line[2:].strip()
        elif line.startswith('<\t'):
            old_line = line[2:].strip()
        # Match new line (starts with '>')
        elif line.startswith('> '):
            new_line = line[2:].strip()
        elif line.startswith('>\t'):
            new_line = line[2:].strip()
    
    return change_line, old_line, new_line

def extract_tags(text_content: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract content between <current> and <fixed> tags."""
    # Extract current
    current_match = re.search(r'<current>\s*(.*?)\s*</current>', text_content, re.DOTALL)
    current = current_match.group(1).strip() if current_match else None
    
    # Extract fixed
    fixed_match = re.search(r'<fixed>\s*(.*?)\s*</fixed>', text_content, re.DOTALL)
    fixed = fixed_match.group(1).strip() if fixed_match else None
    
    return current, fixed

def verify_diff_detailed(diff_file: str, text_file: str) -> Dict[str, Any]:
    """Verify if diff output matches the expected changes in text file and return detailed results."""
    result = {
        "success": False,
        "error": None,
        "diff_analysis": {},
        "llm_analysis": {},
        "mismatches": []
    }
    
    # Read diff file
    try:
        with open(diff_file, 'r') as f:
            diff_content = f.read()
    except Exception as e:
        result["error"] = f"Failed to read diff file: {e}"
        return result
    
    # Read text file
    try:
        with open(text_file, 'r') as f:
            text_content = f.read()
    except Exception as e:
        result["error"] = f"Failed to read text file: {e}"
        return result
    
    # Parse diff
    change_line, old_line, new_line = parse_diff(diff_content)
    
    if not change_line:
        result["error"] = "No change line found in diff"
        return result
    
    if old_line is None or new_line is None:
        result["error"] = "Could not extract old/new lines from diff"
        return result
    
    # Extract tags from text file
    current, fixed = extract_tags(text_content)
    
    if current is None or fixed is None:
        result["error"] = "Could not extract <current> or <fixed> tags from text file"
        return result
    
    # Store analysis details
    result["diff_analysis"] = {
        "change_line": change_line,
        "old_line": old_line,
        "new_line": new_line
    }
    
    result["llm_analysis"] = {
        "current": current,
        "fixed": fixed
    }
    
    # Verification
    matches = True
    
    if old_line != current:
        result["mismatches"].append({
            "type": "old_line_mismatch",
            "diff_old": old_line,
            "llm_current": current
        })
        matches = False
    
    if new_line != fixed:
        result["mismatches"].append({
            "type": "new_line_mismatch",
            "diff_new": new_line,
            "llm_fixed": fixed
        })
        matches = False
    
    result["success"] = matches
    return result

def verify_diff(diff_file: str, text_file: str) -> bool:
    """Verify if diff output matches the expected changes in text file."""
    # Read diff file
    try:
        with open(diff_file, 'r') as f:
            diff_content = f.read()
    except Exception as e:
        print(f"ERROR: Failed to read diff file: {e}")
        return False
    
    # Read text file
    try:
        with open(text_file, 'r') as f:
            text_content = f.read()
    except Exception as e:
        print(f"ERROR: Failed to read text file: {e}")
        return False
    
    # Parse diff
    change_line, old_line, new_line = parse_diff(diff_content)
    
    if not change_line:
        print("ERROR: No change line found in diff")
        return False
    
    if old_line is None or new_line is None:
        print("ERROR: Could not extract old/new lines from diff")
        return False
    
    # Extract tags from text file
    current, fixed = extract_tags(text_content)
    
    if current is None or fixed is None:
        print("ERROR: Could not extract <current> or <fixed> tags from text file")
        return False
    
    # Print analysis
    print("Diff Analysis:")
    print(f"  Change: {change_line}")
    print(f"  Old line from diff: '{old_line}'")
    print(f"  New line from diff: '{new_line}'")
    print()
    print("Text File Analysis:")
    print(f"  Current: '{current}'")
    print(f"  Fixed: '{fixed}'")
    print()
    
    # Verification
    matches = True
    
    if old_line != current:
        print("MISMATCH: Old line from diff doesn't match <current> tag")
        print(f"  Diff old: '{old_line}'")
        print(f"  Current:  '{current}'")
        matches = False
    
    if new_line != fixed:
        print("MISMATCH: New line from diff doesn't match <fixed> tag")
        print(f"  Diff new: '{new_line}'")
        print(f"  Fixed:    '{fixed}'")
        matches = False
    
    if matches:
        print("VERIFICATION PASSED: Diff matches the expected changes")
    else:
        print("VERIFICATION FAILED: Diff does not match expected changes")
    
    return matches

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <diff_output> <text_file>")
        sys.exit(1)
    
    diff_file = sys.argv[1]
    text_file = sys.argv[2]
    
    success = verify_diff(diff_file, text_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()