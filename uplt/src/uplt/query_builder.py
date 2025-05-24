"""SQL query builder for chart commands."""
import re
from typing import List, Optional, Tuple


def parse_aggregation(field: str) -> Tuple[Optional[str], str]:
    """
    Parse aggregation function from field specification.
    
    Returns tuple of (aggregation_function, field_name)
    Examples:
        "avg(price)" -> ("avg", "price")
        "sum(total)" -> ("sum", "total")
        "price" -> (None, "price")
    """
    # Match patterns like avg(field), sum(field), etc.
    match = re.match(r'^(\w+)\((.+)\)$', field.strip())
    if match:
        func, field_name = match.groups()
        # Validate known aggregation functions
        valid_funcs = ['avg', 'sum', 'min', 'max', 'count']
        if func.lower() in valid_funcs:
            return func.lower(), field_name.strip()
    
    # No aggregation function found
    return None, field.strip()


def build_heatmap_query(
    x_field: str,
    y_field: str,
    value_field: Optional[str] = None,
    table_name: str = "data"
) -> str:
    """
    Build SQL query for heatmap data.
    
    Args:
        x_field: Field name for x-axis
        y_field: Field name for y-axis
        value_field: Optional field for aggregation (e.g., "avg(price)")
        table_name: Name of the table to query
    
    Returns:
        SQL query string
    """
    # Sanitize field names to prevent SQL injection
    x_field = x_field.strip()
    y_field = y_field.strip()
    
    if value_field:
        agg_func, field_name = parse_aggregation(value_field)
        if agg_func:
            # Use aggregation function
            value_expr = f"{agg_func.upper()}({field_name})"
        else:
            # Use field directly
            value_expr = field_name
    else:
        # Count rows by default
        value_expr = "COUNT(*)"
    
    query = f"""
    SELECT 
        {x_field} as x,
        {y_field} as y,
        {value_expr} as value
    FROM {table_name}
    GROUP BY {x_field}, {y_field}
    ORDER BY {y_field}, {x_field}
    """
    
    return query.strip()


def parse_chart_command(args: List[str]) -> Tuple[str, dict]:
    """
    Parse chart command arguments.
    
    Args:
        args: List of arguments after the chart type
    
    Returns:
        Tuple of (chart_type, options_dict)
    """
    if not args:
        raise ValueError("No chart type specified")
    
    chart_type = args[0]
    
    if chart_type == "heatmap":
        if len(args) < 3:
            raise ValueError("Heatmap requires at least x_field and y_field")
        
        options = {
            "x_field": args[1],
            "y_field": args[2],
            "value_field": args[3] if len(args) > 3 else None
        }
        return chart_type, options
    
    # Add more chart types here in the future
    raise ValueError(f"Unknown chart type: {chart_type}")