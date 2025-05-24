import sqlite3
import csv
import io
import re
from typing import List, Any, Optional, Tuple


def detect_delimiter(sample: str) -> str:
    """Detect the most likely delimiter in the CSV data."""
    delimiters = [',', ';', '\t', ' ', '|']
    delimiter_counts = {}
    
    # Count occurrences of each delimiter in the first few lines
    lines = sample.split('\n')[:5]  # Check first 5 lines
    for delimiter in delimiters:
        count = sum(line.count(delimiter) for line in lines)
        delimiter_counts[delimiter] = count
    
    # Return the delimiter with the highest count
    return max(delimiter_counts, key=delimiter_counts.get)


def sanitize_column_name(name: str) -> str:
    """Sanitize column names to be valid SQL identifiers."""
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[^\w]', '_', str(name).strip())
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = 'col_' + sanitized
    # Handle empty names
    if not sanitized:
        sanitized = 'unnamed_column'
    return sanitized


def infer_column_type(values: List[Any]) -> str:
    """Infer the SQL column type based on the values."""
    # Remove None/empty values for type inference
    non_empty_values = [v for v in values if v is not None and str(v).strip()]
    
    if not non_empty_values:
        return 'TEXT'
    
    # Check if all values can be integers
    try:
        for val in non_empty_values:
            int(str(val))
        return 'INTEGER'
    except ValueError:
        pass
    
    # Check if all values can be floats
    try:
        for val in non_empty_values:
            float(str(val))
        return 'REAL'
    except ValueError:
        pass
    
    return 'TEXT'


def create_table_from_csv(cursor: sqlite3.Cursor, csv_data: str, table_name: str = 'data') -> List[str]:
    """Create and populate an SQLite table from CSV data."""
    
    # Detect delimiter
    delimiter = detect_delimiter(csv_data)
    
    # Parse CSV
    csv_reader = csv.reader(io.StringIO(csv_data), delimiter=delimiter)
    
    try:
        # Get headers
        headers = next(csv_reader)
        headers = [sanitize_column_name(h) for h in headers]
        
        # Read all data to infer types
        rows = list(csv_reader)
        
        if not rows:
            raise ValueError("No data rows found in CSV")
        
        # Infer column types
        column_types = []
        for i, header in enumerate(headers):
            column_values = [row[i] if i < len(row) else None for row in rows]
            col_type = infer_column_type(column_values)
            column_types.append(f"{header} {col_type}")
        
        # Create table
        create_sql = f"CREATE TABLE {table_name} ({', '.join(column_types)})"
        cursor.execute(create_sql)
        
        # Insert data
        placeholders = ', '.join(['?' for _ in headers])
        insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
        
        for row in rows:
            # Pad row with None if it has fewer columns than headers
            padded_row = row + [None] * (len(headers) - len(row))
            # Truncate row if it has more columns than headers
            padded_row = padded_row[:len(headers)]
            cursor.execute(insert_sql, padded_row)
        
        return headers
        
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {e}")


def execute_query(cursor: sqlite3.Cursor, query: str) -> List[Tuple]:
    """Execute SQL query and return results."""
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        raise ValueError(f"SQL Error: {e}")


def format_output(results: List[Tuple], description: List[Tuple]) -> str:
    """Format query results as CSV."""
    if not results:
        return ""
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers
    headers = [desc[0] for desc in description]
    writer.writerow(headers)
    
    # Write data
    for row in results:
        writer.writerow(row)
    
    return output.getvalue()