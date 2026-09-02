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

from quantile_scenarios import QuantileScenarioGenerator


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
            'Customer': ['A'] * 50 + ['B'] * 50  # Two customers
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
    
    @pytest.mark.parametrize("n_scenarios,quantiles,prefix,expected_count,expected_quantiles", [
        (None, None, None, 5, [1/6, 2/6, 3/6, 4/6, 5/6]),
        (3, None, None, 3, [1/4, 2/4, 3/4]),
        (1, None, None, 1, [0.5]),
    ])
    def test_initialization(self, n_scenarios, quantiles, prefix, expected_count, expected_quantiles):
        kwargs = {}
        if n_scenarios is not None:
            kwargs['n_scenarios'] = n_scenarios
        if quantiles is not None:
            kwargs['quantiles'] = quantiles
        if prefix is not None:
            kwargs['scenario_prefix'] = prefix
        generator = QuantileScenarioGenerator(**kwargs)
        assert generator.n_scenarios == expected_count
        assert len(generator.quantiles) == expected_count
        for actual, expected in zip(generator.quantiles, expected_quantiles):
            assert abs(actual - expected) < 1e-10

    def test_initialization_custom_quantiles_precedence(self):
        generator = QuantileScenarioGenerator(n_scenarios=5, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
        assert generator.quantiles == [0.1, 0.3, 0.5, 0.7, 0.9]

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
    
if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])