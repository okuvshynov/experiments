"""Terminal chart plotting using Unicode characters."""
import math
from typing import List, Tuple, Optional


def create_heatmap(
    data: List[Tuple],
    width: Optional[int] = None,
    height: Optional[int] = None,
    chars: str = " ░▒▓█"
) -> str:
    """
    Create a heatmap using Unicode block characters.
    
    Args:
        data: List of tuples (x, y, value) from SQL query
        width: Maximum width of the chart (default: auto)
        height: Maximum height of the chart (default: auto)
        chars: Characters to use for intensity levels
    
    Returns:
        String representation of the heatmap
    """
    if not data:
        return "No data to plot"
    
    # Extract unique x and y values
    x_values = sorted(list(set(row[0] for row in data)))
    y_values = sorted(list(set(row[1] for row in data)), reverse=True)  # Reverse for top-to-bottom
    
    # Create a mapping of (x, y) -> value
    value_map = {(str(row[0]), str(row[1])): row[2] for row in data}
    
    # Find min and max values for normalization
    values = []
    for row in data:
        if row[2] is not None:
            try:
                # Try to convert to float
                values.append(float(row[2]))
            except (ValueError, TypeError):
                # Skip non-numeric values
                pass
    
    if not values:
        return "No numeric values to plot"
    
    min_val = min(values)
    max_val = max(values)
    
    # Handle case where all values are the same
    if min_val == max_val:
        # Use middle character for all cells
        char_idx = len(chars) // 2
    else:
        char_idx = None
    
    # Calculate label widths
    x_label_width = max(len(str(x)) for x in x_values) if x_values else 0
    y_label_width = max(len(str(y)) for y in y_values) if y_values else 0
    
    # Build the heatmap
    lines = []
    
    # Add header with x labels (rotated would be better, but using simple layout)
    header = " " * (y_label_width + 1)  # Space for y labels
    for x in x_values:
        header += str(x).rjust(x_label_width + 1)
    lines.append(header)
    
    # Add separator line
    separator = " " * (y_label_width + 1) + "-" * (len(x_values) * (x_label_width + 1))
    lines.append(separator)
    
    # Add data rows
    for y in y_values:
        row = str(y).rjust(y_label_width) + "|"
        for x in x_values:
            key = (str(x), str(y))
            if key in value_map and value_map[key] is not None:
                value = value_map[key]
                try:
                    # Convert to float for normalization
                    numeric_value = float(value)
                    # Normalize value to character index
                    if char_idx is not None:
                        idx = char_idx
                    else:
                        normalized = (numeric_value - min_val) / (max_val - min_val)
                        idx = int(normalized * (len(chars) - 1))
                        idx = max(0, min(idx, len(chars) - 1))
                    char = chars[idx]
                except (ValueError, TypeError):
                    # Non-numeric value, use empty space
                    char = " "
            else:
                char = " "  # Empty cell
            
            # Repeat character to fill cell width
            row += (char * x_label_width) + " "
        lines.append(row)
    
    # Add value scale legend
    lines.append("")
    lines.append(f"Scale: {chars[0]}={min_val:.2f} to {chars[-1]}={max_val:.2f}")
    
    return "\n".join(lines)


def format_chart_output(chart_type: str, data: List[Tuple], **kwargs) -> str:
    """
    Format data into the requested chart type.
    
    Args:
        chart_type: Type of chart to create
        data: Query results
        **kwargs: Additional chart-specific options
    
    Returns:
        Formatted chart as string
    """
    if chart_type == "heatmap":
        return create_heatmap(data, **kwargs)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")