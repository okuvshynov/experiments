import pytest
from uplt.charts import create_heatmap, format_chart_output


class TestCreateHeatmap:
    def test_basic_heatmap(self):
        data = [
            ("A", "X", 10),
            ("A", "Y", 20),
            ("B", "X", 30),
            ("B", "Y", 40),
        ]
        result = create_heatmap(data)
        
        # Check that result contains expected elements
        assert "A" in result
        assert "B" in result
        assert "X" in result
        assert "Y" in result
        assert "Scale:" in result
        
        # Check structure
        lines = result.split("\n")
        assert len(lines) > 4  # At least header, separator, 2 data rows, and scale
    
    def test_heatmap_with_missing_cells(self):
        data = [
            ("A", "X", 10),
            # Missing (A, Y)
            ("B", "X", 30),
            ("B", "Y", 40),
        ]
        result = create_heatmap(data)
        
        # Should still work with missing cells
        assert "A" in result
        assert "B" in result
        lines = result.split("\n")
        assert len(lines) > 4
    
    def test_heatmap_with_none_values(self):
        data = [
            ("A", "X", 10),
            ("A", "Y", None),
            ("B", "X", 30),
            ("B", "Y", 40),
        ]
        result = create_heatmap(data)
        
        # Should handle None values gracefully
        assert "Scale:" in result
        assert "10.00" in result
        assert "40.00" in result
    
    def test_empty_data(self):
        data = []
        result = create_heatmap(data)
        assert result == "No data to plot"
    
    def test_all_none_values(self):
        data = [
            ("A", "X", None),
            ("A", "Y", None),
        ]
        result = create_heatmap(data)
        assert result == "No numeric values to plot"
    
    def test_single_value(self):
        data = [("A", "X", 25)]
        result = create_heatmap(data)
        
        # Should handle single value
        assert "A" in result
        assert "X" in result
        assert "25.00" in result
    
    def test_all_same_values(self):
        data = [
            ("A", "X", 50),
            ("A", "Y", 50),
            ("B", "X", 50),
            ("B", "Y", 50),
        ]
        result = create_heatmap(data)
        
        # Should handle all same values
        assert "50.00" in result
        # Should use middle character for all cells
        lines = result.split("\n")
        data_lines = [l for l in lines if "|" in l]
        assert len(data_lines) == 2  # Two Y values
    
    def test_custom_chars(self):
        data = [
            ("A", "X", 10),
            ("B", "Y", 90),
        ]
        result = create_heatmap(data, chars=".oO@")
        
        # Check custom characters are used
        assert "Scale: .=10.00 to @=90.00" in result
    
    def test_numeric_labels(self):
        data = [
            (1, 10, 100),
            (1, 20, 200),
            (2, 10, 150),
            (2, 20, 250),
        ]
        result = create_heatmap(data)
        
        # Should handle numeric labels
        assert "1" in result
        assert "2" in result
        assert "10" in result
        assert "20" in result
    
    def test_string_values_converted(self):
        # Data might come from SQL as strings
        data = [
            ("A", "X", "10"),
            ("A", "Y", "20"),
            ("B", "X", "30"),
            ("B", "Y", "40"),
        ]
        # This should work if the implementation converts strings to numbers
        # If not, it might need adjustment
        result = create_heatmap(data)
        assert "Scale:" in result


class TestFormatChartOutput:
    def test_heatmap_format(self):
        data = [
            ("A", "X", 10),
            ("B", "Y", 20),
        ]
        result = format_chart_output("heatmap", data)
        
        # Should call create_heatmap
        assert "Scale:" in result
        assert "A" in result
        assert "B" in result
    
    def test_unknown_chart_type(self):
        data = [("A", "X", 10)]
        
        with pytest.raises(ValueError, match="Unknown chart type: barchart"):
            format_chart_output("barchart", data)
    
    def test_format_with_kwargs(self):
        data = [
            ("A", "X", 10),
            ("B", "Y", 90),
        ]
        result = format_chart_output("heatmap", data, chars=".oO@")
        
        # Should pass kwargs to create_heatmap
        assert ".=10.00 to @=90.00" in result