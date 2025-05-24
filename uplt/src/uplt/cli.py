import sys
import sqlite3
import argparse
from .core import create_table_from_csv, execute_query, format_output
from .query_builder import parse_chart_command, build_heatmap_query
from .charts import format_chart_output


def main():
    parser = argparse.ArgumentParser(
        description='Execute SQL queries on CSV data from stdin or create terminal charts',
        epilog='Examples:\n'
               '  SQL query: cat data.csv | uplt query "SELECT * FROM data"\n'
               '  Heatmap: cat data.csv | uplt heatmap x_field y_field "avg(value)"\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Make command positional but with nargs='*' to handle variable arguments
    parser.add_argument('command', nargs='*', 
                       help='Command: "query" for SQL or chart type (e.g., "heatmap")')
    parser.add_argument('--table-name', '-t', default='data', 
                       help='Name for the SQLite table (default: data)')
    parser.add_argument('--delimiter', '-d', 
                       help='CSV delimiter (auto-detected if not specified)')
    parser.add_argument('--no-headers', action='store_true',
                       help='Treat first row as data, not headers')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show additional information')
    
    args = parser.parse_args()
    
    # Handle backward compatibility: if no command specified, treat as raw SQL
    if not args.command:
        print("Error: No command specified. Use 'query' for SQL or a chart type.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
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
        
        # Determine mode and execute
        command_type = args.command[0]
        
        if command_type == "query":
            # Raw SQL mode
            if len(args.command) < 2:
                print("Error: SQL query required after 'query'", file=sys.stderr)
                sys.exit(1)
            
            query = args.command[1]
            results = execute_query(cursor, query)
            
            # Output results as CSV
            if results:
                output = format_output(results, cursor.description)
                print(output, end='')
            elif args.verbose:
                print("Query returned no results.", file=sys.stderr)
        
        else:
            # Chart mode
            try:
                chart_type, options = parse_chart_command(args.command)
                
                # Build appropriate query based on chart type
                if chart_type == "heatmap":
                    query = build_heatmap_query(
                        options["x_field"],
                        options["y_field"],
                        options["value_field"],
                        args.table_name
                    )
                    
                    if args.verbose:
                        print(f"Generated query: {query}", file=sys.stderr)
                    
                    results = execute_query(cursor, query)
                    
                    if results:
                        chart = format_chart_output(chart_type, results)
                        print(chart)
                    else:
                        print("No data to plot.", file=sys.stderr)
                else:
                    print(f"Chart type '{chart_type}' not yet implemented", file=sys.stderr)
                    sys.exit(1)
                    
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()