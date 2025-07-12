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
    
    # Individual dot patterns for each line
    LINE_LEFT_DOTS = {
        0: 0x01,  # dot 1 (binary: 00000001)
        1: 0x02,  # dot 2 (binary: 00000010)
        2: 0x04,  # dot 3 (binary: 00000100)
        3: 0x40,  # dot 7 (binary: 01000000)
    }
    
    LINE_RIGHT_DOTS = {
        0: 0x08,  # dot 4 (binary: 00001000)
        1: 0x10,  # dot 5 (binary: 00010000)
        2: 0x20,  # dot 6 (binary: 00100000)
        3: 0x80,  # dot 8 (binary: 10000000)
    }
    
    # Process values in groups of 4
    lines = []
    for i in range(0, len(values), 4):
        group = values[i:i+4]
        line_chars = []
        
        for char_pos in range(width):
            # Calculate two positions within this character
            # Left position for left dot, right position for right dot
            left_pos = (char_pos * 2) / (width * 2 - 1) if width > 1 else 0
            right_pos = (char_pos * 2 + 1) / (width * 2 - 1) if width > 1 else 0
            
            # Build the Braille character
            dots = 0
            for j, value in enumerate(group):
                # Check if we should show the left dot
                if value >= left_pos:
                    dots |= LINE_LEFT_DOTS[j]
                # Check if we should show the right dot
                if value >= right_pos:
                    dots |= LINE_RIGHT_DOTS[j]
            
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