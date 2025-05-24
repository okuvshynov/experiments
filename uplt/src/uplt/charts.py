"""Terminal chart plotting using Unicode characters."""
import math
from typing import List, Tuple, Optional, Union, Dict


def is_numeric_axis(values: List) -> bool:
    """Check if all values in a list can be converted to numbers."""
    if not values:
        return False
    
    for val in values:
        try:
            float(val)
        except (ValueError, TypeError):
            return False
    return True


def create_numeric_scale(min_val: float, max_val: float, target_steps: int = 10) -> List[float]:
    """Create a nice numeric scale for an axis."""
    if min_val == max_val:
        # Single value, create scale around it
        if min_val == 0:
            return [0]
        return [min_val * 0.9, min_val, min_val * 1.1]
    
    range_val = max_val - min_val
    
    # Find a nice step size
    raw_step = range_val / target_steps
    magnitude = 10 ** math.floor(math.log10(raw_step))
    
    # Round to nice numbers (1, 2, 2.5, 5, 10)
    normalized_step = raw_step / magnitude
    if normalized_step <= 1:
        nice_step = 1
    elif normalized_step <= 2:
        nice_step = 2
    elif normalized_step <= 2.5:
        nice_step = 2.5
    elif normalized_step <= 5:
        nice_step = 5
    else:
        nice_step = 10
    
    step = nice_step * magnitude
    
    # Create scale
    start = math.floor(min_val / step) * step
    end = math.ceil(max_val / step) * step
    
    scale = []
    current = start
    while current <= end + step * 0.01:  # Small epsilon for floating point
        scale.append(current)
        current += step
    
    return scale


def find_bin_index(value: float, scale: List[float]) -> int:
    """Find which bin a value belongs to in a scale."""
    for i in range(len(scale) - 1):
        if scale[i] <= value < scale[i + 1]:
            return i
    # Handle edge case for maximum value
    if value == scale[-1]:
        return len(scale) - 2
    return -1


def create_heatmap(
    data: List[Tuple],
    width: Optional[int] = None,
    height: Optional[int] = None,
    chars: str = " ░▒▓█"
) -> str:
    """
    Create a heatmap using Unicode block characters.
    
    Automatically detects numeric vs categorical axes and handles them appropriately.
    For numeric axes, creates proper scales with binning.
    
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
    
    # Extract x and y values
    x_values_raw = [row[0] for row in data]
    y_values_raw = [row[1] for row in data]
    
    # Determine if axes are numeric
    x_is_numeric = is_numeric_axis(x_values_raw)
    y_is_numeric = is_numeric_axis(y_values_raw)
    
    # Process x-axis
    if x_is_numeric:
        x_numeric = [float(x) for x in x_values_raw]
        x_min, x_max = min(x_numeric), max(x_numeric)
        x_scale = create_numeric_scale(x_min, x_max, width or 20)
        x_labels = [f"{x:.6g}" for x in x_scale]
        x_bins = len(x_scale) - 1
    else:
        x_labels = sorted(list(set(str(x) for x in x_values_raw)))
        x_bins = len(x_labels)
    
    # Process y-axis
    if y_is_numeric:
        y_numeric = [float(y) for y in y_values_raw]
        y_min, y_max = min(y_numeric), max(y_numeric)
        y_scale = create_numeric_scale(y_min, y_max, height or 15)
        y_labels = [f"{y:.6g}" for y in y_scale]
        y_bins = len(y_scale) - 1
        # Reverse for top-to-bottom display
        y_labels = list(reversed(y_labels))
        y_scale = list(reversed(y_scale))
    else:
        y_labels = sorted(list(set(str(y) for y in y_values_raw)), reverse=True)
        y_bins = len(y_labels)
    
    # Create grid for accumulating values
    grid: Dict[Tuple[int, int], List[float]] = {}
    
    # Process data points
    for x_raw, y_raw, value in data:
        if value is None:
            continue
            
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            continue
        
        # Find grid position
        if x_is_numeric:
            x_idx = find_bin_index(float(x_raw), x_scale)
            if x_idx < 0:
                continue
        else:
            try:
                x_idx = x_labels.index(str(x_raw))
            except ValueError:
                continue
        
        if y_is_numeric:
            # y_scale is already reversed for display, so un-reverse it to find the bin
            original_scale = list(reversed(y_scale))
            bin_idx = find_bin_index(float(y_raw), original_scale)
            if bin_idx < 0:
                continue
            # Since y_scale is reversed, high values should have low y_idx
            y_idx = y_bins - 1 - bin_idx
            if y_idx < 0 or y_idx >= y_bins:
                continue
        else:
            try:
                y_idx = y_labels.index(str(y_raw))
            except ValueError:
                continue
        
        # Accumulate values in grid cell
        key = (x_idx, y_idx)
        if key not in grid:
            grid[key] = []
        grid[key].append(numeric_value)
    
    # Calculate aggregated values for each cell
    aggregated_grid = {}
    all_values = []
    
    for key, values in grid.items():
        if values:
            # Use mean for aggregation when multiple values in a bin
            avg_value = sum(values) / len(values)
            aggregated_grid[key] = avg_value
            all_values.append(avg_value)
    
    if not all_values:
        return "No numeric values to plot"
    
    # Find min and max for color scaling
    min_val = min(all_values)
    max_val = max(all_values)
    
    # Handle case where all values are the same
    if min_val == max_val:
        char_idx = len(chars) // 2
    else:
        char_idx = None
    
    # Calculate label widths
    x_label_width = max(len(label) for label in x_labels) if x_labels else 0
    y_label_width = max(len(label) for label in y_labels) if y_labels else 0
    
    # Build the heatmap
    lines = []
    
    # Add header with x labels
    if x_is_numeric:
        # For numeric x-axis, show scale indicators
        header = " " * (y_label_width + 1)
        for i, label in enumerate(x_labels[:-1]):  # Skip last label for bins
            header += label.rjust(x_label_width + 1)
        lines.append(header)
    else:
        # Categorical x-axis
        header = " " * (y_label_width + 1)
        for label in x_labels:
            header += label.rjust(x_label_width + 1)
        lines.append(header)
    
    # Add separator line
    separator = " " * (y_label_width + 1) + "-" * (x_bins * (x_label_width + 1))
    lines.append(separator)
    
    # Add data rows
    for y_idx in range(y_bins):
        # Get y label
        if y_is_numeric:
            # For numeric axes, show the upper boundary of each bin
            row_label = y_labels[y_idx]
        else:
            row_label = y_labels[y_idx]
        
        row = row_label.rjust(y_label_width) + "|"
        
        for x_idx in range(x_bins):
            key = (x_idx, y_idx)
            if key in aggregated_grid:
                value = aggregated_grid[key]
                # Normalize value to character index
                if char_idx is not None:
                    idx = char_idx
                else:
                    normalized = (value - min_val) / (max_val - min_val)
                    idx = int(normalized * (len(chars) - 1))
                    idx = max(0, min(idx, len(chars) - 1))
                char = chars[idx]
            else:
                char = " "  # Empty cell
            
            # Fill cell
            row += (char * x_label_width) + " "
        
        lines.append(row)
    
    # Add axis labels and scale info
    lines.append("")
    
    if x_is_numeric:
        lines.append(f"X-axis: {x_min:.6g} to {x_max:.6g}")
    if y_is_numeric:
        lines.append(f"Y-axis: {y_min:.6g} to {y_max:.6g}")
    
    lines.append(f"Value scale: {chars[0]}={min_val:.6g} to {chars[-1]}={max_val:.6g}")
    
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