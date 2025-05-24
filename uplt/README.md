# uplt

Execute SQL queries on CSV data from stdin.

## Installation

```bash
pip install uplt
```

or with uv:

```bash
uv pip install uplt
```

## Usage

Pipe CSV data to `uplt` and provide a SQL query:

```bash
cat data.csv | uplt "SELECT foo, bar, SUM(baz) FROM data GROUP BY foo, bar"
```

## Examples

### Basic query
```bash
cat data/test.csv | uplt "SELECT * FROM data WHERE age > 30"
```

### Aggregation
```bash
cat data/test.csv | uplt "SELECT department, AVG(salary) FROM data GROUP BY department"
```

### Custom table name
```bash
cat data.csv | uplt -t employees "SELECT * FROM employees WHERE department = 'Engineering'"
```

### Verbose mode
```bash
cat data.csv | uplt -v "SELECT COUNT(*) FROM data"
```

## Options

- `query`: SQL query to execute (required)
- `--table-name`, `-t`: Name for the SQLite table (default: data)
- `--delimiter`, `-d`: CSV delimiter (auto-detected if not specified)
- `--no-headers`: Treat first row as data, not headers
- `--verbose`, `-v`: Show additional information

## Features

- Automatic delimiter detection (comma, semicolon, tab, space, pipe)
- Column type inference (INTEGER, REAL, TEXT)
- Sanitized column names for valid SQL identifiers
- In-memory SQLite database for fast queries
- Standard CSV output format

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest
```

## License

MIT