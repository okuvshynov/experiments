import pytest
import sqlite3
from io import StringIO
from uplt.core import (
    detect_delimiter,
    sanitize_column_name,
    infer_column_type,
    create_table_from_csv,
    execute_query,
    format_output,
)


class TestDetectDelimiter:
    def test_comma_delimiter(self):
        sample = "a,b,c\n1,2,3\n4,5,6"
        assert detect_delimiter(sample) == ","
    
    def test_semicolon_delimiter(self):
        sample = "a;b;c\n1;2;3\n4;5;6"
        assert detect_delimiter(sample) == ";"
    
    def test_tab_delimiter(self):
        sample = "a\tb\tc\n1\t2\t3\n4\t5\t6"
        assert detect_delimiter(sample) == "\t"
    
    def test_pipe_delimiter(self):
        sample = "a|b|c\n1|2|3\n4|5|6"
        assert detect_delimiter(sample) == "|"
    
    def test_mixed_delimiters(self):
        # Should pick the most common one
        sample = "a,b;c\n1,2;3\n4,5;6"
        assert detect_delimiter(sample) == ","


class TestSanitizeColumnName:
    def test_normal_name(self):
        assert sanitize_column_name("column_name") == "column_name"
    
    def test_name_with_spaces(self):
        assert sanitize_column_name("column name") == "column_name"
    
    def test_name_with_special_chars(self):
        assert sanitize_column_name("column-name!") == "column_name_"
        assert sanitize_column_name("column@name#") == "column_name_"
    
    def test_name_starting_with_number(self):
        assert sanitize_column_name("123column") == "col_123column"
    
    def test_empty_name(self):
        assert sanitize_column_name("") == "unnamed_column"
        assert sanitize_column_name("   ") == "unnamed_column"
    
    def test_unicode_characters(self):
        # \w in Python regex includes unicode word characters by default
        assert sanitize_column_name("café") == "café"


class TestInferColumnType:
    def test_integer_column(self):
        values = ["1", "2", "3", "4", "5"]
        assert infer_column_type(values) == "INTEGER"
    
    def test_float_column(self):
        values = ["1.5", "2.7", "3.14", "4.0", "5.99"]
        assert infer_column_type(values) == "REAL"
    
    def test_text_column(self):
        values = ["hello", "world", "test", "data"]
        assert infer_column_type(values) == "TEXT"
    
    def test_mixed_numeric_types(self):
        # Integers can be floats, so this should be REAL
        values = ["1", "2.5", "3", "4.7"]
        assert infer_column_type(values) == "REAL"
    
    def test_empty_values(self):
        values = ["", None, "  ", None]
        assert infer_column_type(values) == "TEXT"
    
    def test_integers_with_empty(self):
        values = ["1", "", "3", None, "5"]
        assert infer_column_type(values) == "INTEGER"


class TestCreateTableFromCSV:
    def setup_method(self):
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
    
    def teardown_method(self):
        self.conn.close()
    
    def test_basic_csv(self):
        csv_data = "name,age,salary\nJohn,25,50000\nJane,30,65000"
        headers = create_table_from_csv(self.cursor, csv_data)
        
        assert headers == ["name", "age", "salary"]
        
        # Check table was created
        self.cursor.execute("SELECT COUNT(*) FROM data")
        assert self.cursor.fetchone()[0] == 2
        
        # Check data types
        self.cursor.execute("PRAGMA table_info(data)")
        columns = self.cursor.fetchall()
        assert columns[0][2] == "TEXT"  # name
        assert columns[1][2] == "INTEGER"  # age
        assert columns[2][2] == "INTEGER"  # salary
    
    def test_custom_table_name(self):
        csv_data = "col1,col2\nval1,val2"
        headers = create_table_from_csv(self.cursor, csv_data, "custom_table")
        
        # Check custom table was created
        self.cursor.execute("SELECT COUNT(*) FROM custom_table")
        assert self.cursor.fetchone()[0] == 1
    
    def test_csv_with_special_headers(self):
        csv_data = "First Name,Last-Name,Age (years),2024\nJohn,Doe,25,Yes"
        headers = create_table_from_csv(self.cursor, csv_data)
        
        assert headers == ["First_Name", "Last_Name", "Age__years_", "col_2024"]
    
    def test_csv_with_missing_values(self):
        csv_data = "a,b,c\n1,2,3\n4,,6\n7,8"
        headers = create_table_from_csv(self.cursor, csv_data)
        
        self.cursor.execute("SELECT * FROM data")
        rows = self.cursor.fetchall()
        assert len(rows) == 3
        # CSV reader returns empty strings for missing values, and types are inferred
        assert rows[1] == (4, '', 6)  # a and c are integers, b is empty string
        assert rows[2] == (7, 8, None)  # Last value is None due to padding
    
    def test_empty_csv_error(self):
        csv_data = "header1,header2"
        with pytest.raises(ValueError, match="No data rows found"):
            create_table_from_csv(self.cursor, csv_data)


class TestExecuteQuery:
    def setup_method(self):
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        # Create a test table
        self.cursor.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        self.cursor.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
    
    def teardown_method(self):
        self.conn.close()
    
    def test_select_query(self):
        results = execute_query(self.cursor, "SELECT * FROM test")
        assert len(results) == 2
        assert results[0] == (1, 'Alice')
        assert results[1] == (2, 'Bob')
    
    def test_aggregate_query(self):
        results = execute_query(self.cursor, "SELECT COUNT(*) FROM test")
        assert results[0][0] == 2
    
    def test_invalid_query(self):
        with pytest.raises(ValueError, match="SQL Error"):
            execute_query(self.cursor, "SELECT * FROM nonexistent")
    
    def test_syntax_error(self):
        with pytest.raises(ValueError, match="SQL Error"):
            execute_query(self.cursor, "SELCT * FROM test")


class TestFormatOutput:
    def test_basic_formatting(self):
        results = [(1, 'Alice'), (2, 'Bob')]
        description = [('id',), ('name',)]
        
        output = format_output(results, description)
        # Use splitlines() to handle different line endings properly
        lines = output.strip().splitlines()
        
        assert lines[0] == "id,name"
        assert lines[1] == "1,Alice"
        assert lines[2] == "2,Bob"
    
    def test_empty_results(self):
        results = []
        description = [('id',), ('name',)]
        
        output = format_output(results, description)
        assert output == ""
    
    def test_values_with_commas(self):
        results = [('John, Jr.', 'Doe')]
        description = [('first_name',), ('last_name',)]
        
        output = format_output(results, description)
        lines = output.strip().splitlines()
        
        assert lines[0] == "first_name,last_name"
        assert lines[1] == '"John, Jr.",Doe'
    
    def test_numeric_values(self):
        results = [(1, 2.5, 'test')]
        description = [('int_col',), ('float_col',), ('text_col',)]
        
        output = format_output(results, description)
        lines = output.strip().splitlines()
        
        assert lines[0] == "int_col,float_col,text_col"
        assert lines[1] == "1,2.5,test"