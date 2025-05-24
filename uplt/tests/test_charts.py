import pytest
from uplt.charts import create_heatmap, format_chart_output, is_numeric_axis, create_numeric_scale, find_bin_index


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
        assert "Value scale:" in result
        
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
        assert "Value scale:" in result
        assert "10" in result
        assert "40" in result
    
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
        assert "25" in result
    
    def test_all_same_values(self):
        data = [
            ("A", "X", 50),
            ("A", "Y", 50),
            ("B", "X", 50),
            ("B", "Y", 50),
        ]
        result = create_heatmap(data)
        
        # Should handle all same values
        assert "50" in result
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
        assert "Value scale: .=10 to @=90" in result
    
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
        assert "Value scale:" in result


class TestNumericAxisFunctions:
    def test_is_numeric_axis(self):
        assert is_numeric_axis([1, 2, 3, 4]) == True
        assert is_numeric_axis(["1", "2", "3"]) == True
        assert is_numeric_axis([1.5, 2.5, 3.5]) == True
        assert is_numeric_axis(["1.5", "2.5", "3.5"]) == True
        
        assert is_numeric_axis(["a", "b", "c"]) == False
        assert is_numeric_axis([1, "two", 3]) == False
        assert is_numeric_axis([]) == False
        assert is_numeric_axis(["Engineering", "Marketing"]) == False
    
    def test_create_numeric_scale(self):
        # Basic scale
        scale = create_numeric_scale(0, 10)
        assert scale[0] == 0
        assert scale[-1] == 10
        assert len(scale) > 2
        
        # Single value
        scale = create_numeric_scale(5, 5)
        assert 5 in scale
        assert len(scale) >= 1
        
        # Large numbers
        scale = create_numeric_scale(1000, 5000)
        assert scale[0] <= 1000
        assert scale[-1] >= 5000
        
        # Decimals
        scale = create_numeric_scale(0.1, 0.9)
        assert scale[0] <= 0.1
        assert scale[-1] >= 0.89  # Allow for floating point precision
    
    def test_find_bin_index(self):
        scale = [0, 10, 20, 30, 40]
        
        assert find_bin_index(5, scale) == 0
        assert find_bin_index(15, scale) == 1
        assert find_bin_index(25, scale) == 2
        assert find_bin_index(35, scale) == 3
        
        # Edge cases
        assert find_bin_index(0, scale) == 0
        assert find_bin_index(40, scale) == 3  # Last value goes in last bin
        assert find_bin_index(-5, scale) == -1
        assert find_bin_index(45, scale) == -1


class TestNumericHeatmap:
    def test_numeric_axes_detection(self):
        # Both axes numeric
        data = [
            (10, 100, 5),
            (20, 200, 10),
            (30, 300, 15),
        ]
        result = create_heatmap(data)
        assert "X-axis:" in result
        assert "Y-axis:" in result
        assert "10" in result or "20" in result  # Should show numeric labels
    
    def test_mixed_axes(self):
        # X categorical, Y numeric
        data = [
            ("A", 10, 100),
            ("B", 20, 200),
            ("A", 30, 150),
        ]
        result = create_heatmap(data)
        assert "Y-axis:" in result
        assert "X-axis:" not in result  # Categorical axis doesn't show range
        assert "A" in result
        assert "B" in result
    
    def test_sparse_numeric_data(self):
        # Data with gaps
        data = [
            (1, 1, 10),
            (1, 5, 20),
            (5, 1, 30),
            (5, 5, 40),
        ]
        result = create_heatmap(data)
        # Should create a proper grid even with sparse data
        assert "X-axis: 1 to 5" in result
        assert "Y-axis: 1 to 5" in result
    
    def test_numeric_binning(self):
        # Test that close values get binned together
        data = [
            (1.1, 10, 100),
            (1.2, 10, 110),
            (1.3, 10, 120),
            (5.1, 10, 200),
            (5.2, 10, 210),
        ]
        result = create_heatmap(data)
        # Values 1.1-1.3 should be in same bin, 5.1-5.2 in another
        lines = result.split("\n")
        # Should show aggregated values
        assert "Value scale:" in result
    
    def test_edge_values_displayed(self):
        # Test that values at the edge of the scale are displayed
        data = [
            (25, 50000, 1),
            (28, 55000, 1),
            (30, 65000, 1),
            (35, 75000, 1),
        ]
        result = create_heatmap(data)
        # Count data rows (lines with |)
        data_rows = [line for line in result.split("\n") if "|" in line]
        # Should have at least 4 rows with data
        filled_rows = [row for row in data_rows if any(c in row for c in "░▒▓█")]
        assert len(filled_rows) == 4, f"Expected 4 filled rows, got {len(filled_rows)}"


class TestFormatChartOutput:
    def test_heatmap_format(self):
        data = [
            ("A", "X", 10),
            ("B", "Y", 20),
        ]
        result = format_chart_output("heatmap", data)
        
        # Should call create_heatmap
        assert "Value scale:" in result
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
        assert ".=10 to @=90" in result