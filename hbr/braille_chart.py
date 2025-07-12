#!/usr/bin/env python3
"""
Braille horizontal bar chart visualization.
Uses 8-dot Braille Unicode characters to display horizontal bar charts.
Each character represents 4 horizontal lines.
"""

import sys
import argparse
from typing import List


def create_braille_chart(values: List[float], width: int) -> List[str]:
    """
    Create a horizontal bar chart using Braille characters.
    
    Args:
        values: List of values between 0.0 and 1.0
        width: Width of the chart in characters
        
    Returns:
        List of strings, each representing 4 horizontal bars
    """
    # Braille pattern base (8-dot)
    # The dots are numbered:
    # 1 4
    # 2 5
    # 3 6
    # 7 8
    
    # For horizontal lines, we use dots:
    # Line 1 (top): dots 1,4
    # Line 2: dots 2,5
    # Line 3: dots 3,6
    # Line 4 (bottom): dots 7,8
    
    # Base offset for 8-dot Braille
    BRAILLE_BASE = 0x2800
    
    # Dot patterns for each horizontal line
    LINE_DOTS = {
        0: 0x09,  # dots 1,4 (binary: 00001001)
        1: 0x12,  # dots 2,5 (binary: 00010010)
        2: 0x24,  # dots 3,6 (binary: 00100100)
        3: 0xC0,  # dots 7,8 (binary: 11000000)
    }
    
    # Process values in groups of 4
    lines = []
    for i in range(0, len(values), 4):
        group = values[i:i+4]
        line_chars = []
        
        for char_pos in range(width):
            # Calculate position in value range (0.0 to 1.0)
            pos = char_pos / (width - 1) if width > 1 else 0
            
            # Build the Braille character
            dots = 0
            for j, value in enumerate(group):
                if value >= pos:
                    dots |= LINE_DOTS[j]
            
            # Convert to Unicode character
            char = chr(BRAILLE_BASE + dots)
            line_chars.append(char)
        
        lines.append(''.join(line_chars))
    
    return lines


def main():
    parser = argparse.ArgumentParser(
        description='Create horizontal bar charts using Braille characters'
    )
    parser.add_argument(
        'values',
        nargs='+',
        type=float,
        help='Values between 0.0 and 1.0'
    )
    parser.add_argument(
        '-w', '--width',
        type=int,
        default=40,
        help='Width of the chart in characters (default: 40)'
    )
    
    args = parser.parse_args()
    
    # Validate values
    for value in args.values:
        if not 0.0 <= value <= 1.0:
            print(f"Error: Value {value} is not between 0.0 and 1.0", file=sys.stderr)
            sys.exit(1)
    
    # Create and display the chart
    chart = create_braille_chart(args.values, args.width)
    for line in chart:
        print(line)


if __name__ == '__main__':
    main()