"""
Unit tests for quantile-based scenario generation.

This module contains comprehensive tests for the QuantileScenarioGenerator class,
including edge cases and integration scenarios.
"""

import pytest
import polars as pl
import numpy as np
import sys
import os
from datetime import date, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scenario_generation.quantile_scenarios import QuantileScenarioGenerator


class TestQuantileScenarioGenerator:
    """Test cases for QuantileScenarioGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create sample energy data
        np.random.seed(42)  # For reproducible tests
        n_rows = 100
        
        # Create timestamps separately
        base_date = date(2023, 1, 1)
        timestamps = [base_date + timedelta(days=i) for i in range(n_rows)]
        
        self.sample_df = pl.DataFrame({
            'timestamp': timestamps,
            'SolarGen': np.random.gamma(2, 2, n_rows),  # Solar generation (kW)
            'HouseLoad': np.random.normal(5, 1.5, n_rows),  # House load (kW)
            'ImportEnergyPrice': np.random.uniform(0.1, 0.3, n_rows),  # Price ($/kWh)
            'ExportEnergyPrice': np.random.uniform(0.05, 0.15, n_rows),  # Price ($/kWh)
            'Customer': ['A'] * 50 + ['B'] * 50,  # Two customers
            'location_id': [1] * 25 + [2] * 25 + [1] * 25 + [2] * 25  # Two locations per customer
        })
        
        # Small dataset for edge case testing
        self.small_df = pl.DataFrame({
            'value': [1.0, 2.0, 3.0],
            'category': ['A', 'A', 'B']
        })
        
        # Single value dataset
        self.single_value_df = pl.DataFrame({
            'constant': [5.0, 5.0, 5.0, 5.0, 5.0]
        })
    
    def test_initialization_default(self):
        """Test default initialization."""
        generator = QuantileScenarioGenerator()
        
        assert generator.n_scenarios == 5
        assert generator.scenario_prefix == "scenario"
        assert len(generator.quantiles) == 5
        assert len(generator.quantiles) == 5
        expected_quantiles = [1/6, 2/6, 3/6, 4/6, 5/6]
        for actual, expected in zip(generator.quantiles, expected_quantiles):
            assert abs(actual - expected) < 1e-10
        
    def test_initialization_custom_scenarios(self):
        """Test initialization with custom number of scenarios."""
        generator = QuantileScenarioGenerator(n_scenarios=3)
        
        assert generator.n_scenarios == 3
        assert len(generator.quantiles) == 3
        assert len(generator.quantiles) == 3
        expected_quantiles = [1/4, 2/4, 3/4]
        for actual, expected in zip(generator.quantiles, expected_quantiles):
            assert abs(actual - expected) < 1e-10
    
    def test_initialization_single_scenario(self):
        """Test initialization with single scenario."""
        generator = QuantileScenarioGenerator(n_scenarios=1)
        
        assert generator.n_scenarios == 1
        assert generator.quantiles == [0.5]  # Median
    
    def test_initialization_custom_quantiles(self):
        """Test initialization with custom quantiles."""
        custom_quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        generator = QuantileScenarioGenerator(
            n_scenarios=5, 
            quantiles=custom_quantiles
        )
        
        assert generator.quantiles == custom_quantiles
    
    def test_initialization_custom_prefix(self):
        """Test initialization with custom scenario prefix."""
        generator = QuantileScenarioGenerator(scenario_prefix="test_scenario")
        
        assert generator.scenario_prefix == "test_scenario"
    
    def test_initialization_invalid_scenarios(self):
        """Test initialization with invalid number of scenarios."""
        with pytest.raises(ValueError, match="n_scenarios must be at least 1"):
            QuantileScenarioGenerator(n_scenarios=0)
        
        with pytest.raises(ValueError, match="n_scenarios must be at least 1"):
            QuantileScenarioGenerator(n_scenarios=-1)
    
    def test_initialization_invalid_quantiles(self):
        """Test initialization with invalid quantiles."""
        # Quantiles out of range
        with pytest.raises(ValueError, match="All quantiles must be between 0 and 1"):
            QuantileScenarioGenerator(quantiles=[0.1, 0.5, 1.1])
        
        with pytest.raises(ValueError, match="All quantiles must be between 0 and 1"):
            QuantileScenarioGenerator(quantiles=[-0.1, 0.5, 0.9])
        
        # Wrong number of quantiles
        with pytest.raises(ValueError, match="Number of quantiles .* must match n_scenarios"):
            QuantileScenarioGenerator(n_scenarios=3, quantiles=[0.1, 0.5])
    
    def test_auto_detect_scenario_columns(self):
        """Test automatic detection of scenario columns."""
        generator = QuantileScenarioGenerator()
        detected_columns = generator._auto_detect_scenario_columns(self.sample_df)
        
        expected_columns = ['SolarGen', 'HouseLoad', 'ImportEnergyPrice', 'ExportEnergyPrice', 'location_id']
        assert set(detected_columns) == set(expected_columns)
    
    def test_validate_columns_valid(self):
        """Test column validation with valid columns."""
        generator = QuantileScenarioGenerator()
        # Should not raise exception
        generator._validate_columns(self.sample_df, ['SolarGen', 'HouseLoad'])
    
    def test_validate_columns_missing(self):
        """Test column validation with missing columns."""
        generator = QuantileScenarioGenerator()
        
        with pytest.raises(ValueError, match="Columns not found in DataFrame"):
            generator._validate_columns(self.sample_df, ['NonExistentColumn'])
    
    def test_validate_columns_non_numeric(self):
        """Test column validation with non-numeric columns."""
        generator = QuantileScenarioGenerator()
        
        with pytest.raises(ValueError, match="Columns must be numeric for scenario generation"):
            generator._validate_columns(self.sample_df, ['Customer'])
    
    def test_generate_scenarios_basic(self):
        """Test basic scenario generation."""
        generator = QuantileScenarioGenerator(n_scenarios=3)
        columns = ['SolarGen', 'HouseLoad']
        
        result_df = generator.generate_scenarios(self.sample_df, columns)
        
        # Check that original columns are preserved
        for col in self.sample_df.columns:
            assert col in result_df.columns
        
        # Check that scenario columns are added
        expected_scenario_cols = [
            'scenario_1_SolarGen', 'scenario_2_SolarGen', 'scenario_3_SolarGen',
            'scenario_1_HouseLoad', 'scenario_2_HouseLoad', 'scenario_3_HouseLoad'
        ]
        for col in expected_scenario_cols:
            assert col in result_df.columns
        
        # Check that scenario values are reasonable (between min and max of original data)
        for base_col in columns:
            min_val = self.sample_df[base_col].min()
            max_val = self.sample_df[base_col].max()
            
            for i in range(1, 4):
                scenario_col = f'scenario_{i}_{base_col}'
                scenario_val = result_df[scenario_col].head(1).item()
                assert min_val <= scenario_val <= max_val
    
    def test_generate_scenarios_auto_detect(self):
        """Test scenario generation with auto-detected columns."""
        generator = QuantileScenarioGenerator(n_scenarios=2)
        
        result_df = generator.generate_scenarios(self.sample_df)
        
        # Should have scenario columns for all numeric columns
        expected_base_cols = ['SolarGen', 'HouseLoad', 'ImportEnergyPrice', 'ExportEnergyPrice', 'location_id']
        for base_col in expected_base_cols:
            assert f'scenario_1_{base_col}' in result_df.columns
            assert f'scenario_2_{base_col}' in result_df.columns
    
    def test_generate_scenarios_grouped(self):
        """Test scenario generation with grouping."""
        generator = QuantileScenarioGenerator(n_scenarios=2)
        columns = ['SolarGen']
        
        result_df = generator.generate_scenarios(
            self.sample_df, 
            columns=columns, 
            group_by='Customer'
        )
        
        # Check that scenario columns exist
        assert 'scenario_1_SolarGen' in result_df.columns
        assert 'scenario_2_SolarGen' in result_df.columns
        
        # Check that different groups have different scenario values
        customer_a_scenarios = result_df.filter(pl.col('Customer') == 'A')['scenario_1_SolarGen'].unique()
        customer_b_scenarios = result_df.filter(pl.col('Customer') == 'B')['scenario_1_SolarGen'].unique()
        
        # Each group should have consistent scenario values within the group
        assert len(customer_a_scenarios) == 1
        assert len(customer_b_scenarios) == 1
        
        # But scenario values between groups may be different
        # (This is probabilistic, but very likely with our test data)
    
    def test_generate_scenarios_custom_prefix(self):
        """Test scenario generation with custom prefix."""
        generator = QuantileScenarioGenerator(
            n_scenarios=2, 
            scenario_prefix="test_scenario"
        )
        columns = ['SolarGen']
        
        result_df = generator.generate_scenarios(self.sample_df, columns)
        
        assert 'test_scenario_1_SolarGen' in result_df.columns
        assert 'test_scenario_2_SolarGen' in result_df.columns
    
    def test_generate_scenarios_single_value(self):
        """Test scenario generation with constant values."""
        generator = QuantileScenarioGenerator(n_scenarios=3)
        
        result_df = generator.generate_scenarios(self.single_value_df, ['constant'])
        
        # All scenarios should have the same value for constant data
        for i in range(1, 4):
            scenario_col = f'scenario_{i}_constant'
            assert scenario_col in result_df.columns
            scenario_val = result_df[scenario_col].head(1).item()
            assert scenario_val == 5.0
    
    def test_generate_scenarios_small_dataset(self):
        """Test scenario generation with small dataset."""
        generator = QuantileScenarioGenerator(n_scenarios=2)
        
        result_df = generator.generate_scenarios(self.small_df, ['value'])
        
        # Should still work with small datasets
        assert 'scenario_1_value' in result_df.columns
        assert 'scenario_2_value' in result_df.columns
        
        # Verify scenario values are within expected range
        min_val = self.small_df['value'].min()
        max_val = self.small_df['value'].max()
        
        for i in range(1, 3):
            scenario_col = f'scenario_{i}_value'
            scenario_val = result_df[scenario_col].head(1).item()
            assert min_val <= scenario_val <= max_val
    
    def test_get_scenario_columns(self):
        """Test getting scenario column names."""
        generator = QuantileScenarioGenerator(
            n_scenarios=3, 
            scenario_prefix="test"
        )
        
        base_columns = ['SolarGen', 'HouseLoad']
        scenario_columns = generator.get_scenario_columns(base_columns)
        
        expected = [
            'test_1_SolarGen', 'test_2_SolarGen', 'test_3_SolarGen',
            'test_1_HouseLoad', 'test_2_HouseLoad', 'test_3_HouseLoad'
        ]
        assert scenario_columns == expected
    
    def test_summarize_scenarios(self):
        """Test scenario summary functionality."""
        generator = QuantileScenarioGenerator(n_scenarios=3)
        columns = ['SolarGen', 'HouseLoad']
        
        result_df = generator.generate_scenarios(self.sample_df, columns)
        summary_df = generator.summarize_scenarios(result_df, columns)
        
        # Check summary structure
        expected_cols = ['base_column', 'scenario', 'quantile', 'value']
        assert list(summary_df.columns) == expected_cols
        
        # Should have 6 rows (3 scenarios × 2 columns)
        assert len(summary_df) == 6
        
        # Check that all base columns are represented
        base_columns_in_summary = summary_df['base_column'].unique().to_list()
        assert set(base_columns_in_summary) == set(columns)
        
        # Check that scenario numbers are correct
        scenario_numbers = summary_df['scenario'].unique().to_list()
        assert set(scenario_numbers) == {1, 2, 3}
        
        # Check that quantiles match generator quantiles
        quantiles_in_summary = sorted(summary_df['quantile'].unique().to_list())
        assert quantiles_in_summary == sorted(generator.quantiles)
    
    def test_quantile_ordering(self):
        """Test that scenarios maintain quantile ordering."""
        generator = QuantileScenarioGenerator(n_scenarios=5)
        columns = ['SolarGen']
        
        result_df = generator.generate_scenarios(self.sample_df, columns)
        
        # Get scenario values
        scenario_values = []
        for i in range(1, 6):
            scenario_col = f'scenario_{i}_SolarGen'
            scenario_val = result_df[scenario_col].head(1).item()
            scenario_values.append(scenario_val)
        
        # Values should be in ascending order (since quantiles are sorted)
        assert scenario_values == sorted(scenario_values)
    
    def test_integration_with_polars_operations(self):
        """Test that generated scenarios work well with polars operations."""
        generator = QuantileScenarioGenerator(n_scenarios=3)
        columns = ['SolarGen', 'HouseLoad']
        
        result_df = generator.generate_scenarios(self.sample_df, columns)
        
        # Test basic polars operations still work
        # Filter operation
        filtered_df = result_df.filter(pl.col('Customer') == 'A')
        assert len(filtered_df) == 50
        
        # Group by operation
        grouped_df = result_df.group_by('Customer').agg([
            pl.col('scenario_1_SolarGen').mean().alias('avg_scenario_1_solar')
        ])
        assert len(grouped_df) == 2  # Two customers
        
        # Select operation with scenario columns
        selected_df = result_df.select([
            'timestamp', 'SolarGen', 'scenario_1_SolarGen', 'scenario_2_SolarGen'
        ])
        assert len(selected_df.columns) == 4


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])