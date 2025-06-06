import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from ui.components import show_data_preview, show_data_quality_summary

class TestUIComponents:
    """Test cases for UI components"""
    
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.dataframe')
    @patch('streamlit.metric')
    def test_show_data_preview(self, mock_metric, mock_dataframe, mock_subheader, mock_columns):
        """Test data preview component"""
        # Create test data
        test_data = pd.DataFrame({
            'col1': [1, 2, 3, None],
            'col2': [4, 5, 6, 7]
        })
        
        # Mock streamlit columns
        mock_col1, mock_col2 = Mock(), Mock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        # Call function
        show_data_preview(test_data)
        
        # Verify calls
        mock_columns.assert_called_once_with([2, 1])
        mock_subheader.assert_called_once_with("📊 Data Preview")
        mock_dataframe.assert_called_once()
        assert mock_metric.call_count == 3  # Rows, Columns, Missing Values
    
    @patch('streamlit.expander')
    @patch('streamlit.write')
    def test_show_data_quality_summary(self, mock_write, mock_expander):
        """Test data quality summary component"""
        # Create test data
        test_data = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0],
            'col2': [4.5, 5.5, 6.5]
        })
        
        # Mock expander
        mock_exp = Mock()
        mock_expander.return_value.__enter__.return_value = mock_exp
        
        # Call function
        show_data_quality_summary(test_data)
        
        # Verify calls
        mock_expander.assert_called_once_with("📈 Data Quality Summary")
        assert mock_write.call_count >= 2  # Header + column info
