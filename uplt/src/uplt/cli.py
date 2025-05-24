import sys
import sqlite3
import argparse
from .core import create_table_from_csv, execute_query, format_output


def main():
    parser = argparse.ArgumentParser(
        description='Execute SQL queries on CSV data from stdin',
        epilog='Example: cat data.csv | uplt "SELECT foo, bar, SUM(baz) FROM data GROUP BY foo, bar"'
    )
    parser.add_argument('query', help='SQL query to execute')
    parser.add_argument('--table-name', '-t', default='data', 
                       help='Name for the SQLite table (default: data)')
    parser.add_argument('--delimiter', '-d', 
                       help='CSV delimiter (auto-detected if not specified)')
    parser.add_argument('--no-headers', action='store_true',
                       help='Treat first row as data, not headers')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show additional information')
    
    args = parser.parse_args()
    
    try:
        # Read CSV data from stdin
        if sys.stdin.isatty():
            print("Error: No input data. Please pipe CSV data to this script.", file=sys.stderr)
            print("Example: cat data.csv | uplt \"SELECT * FROM data\"", file=sys.stderr)
            sys.exit(1)
        
        csv_data = sys.stdin.read().strip()
        
        if not csv_data:
            print("Error: No input data received.", file=sys.stderr)
            sys.exit(1)
        
        # Create in-memory SQLite database
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Create and populate table
        if args.verbose:
            print(f"Creating table '{args.table_name}'...", file=sys.stderr)
        
        headers = create_table_from_csv(cursor, csv_data, args.table_name)
        
        if args.verbose:
            print(f"Table created with columns: {', '.join(headers)}", file=sys.stderr)
            cursor.execute(f"SELECT COUNT(*) FROM {args.table_name}")
            count = cursor.fetchone()[0]
            print(f"Loaded {count} rows", file=sys.stderr)
        
        # Execute query
        results = execute_query(cursor, args.query)
        
        # Output results
        if results:
            output = format_output(results, cursor.description)
            print(output, end='')
        elif args.verbose:
            print("Query returned no results.", file=sys.stderr)
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()