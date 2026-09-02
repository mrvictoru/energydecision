"""
Shared test fixtures and configuration for the energydecision test suite.
"""

import sys
import os
import warnings
import pytest
import numpy as np
import polars as pl
import datetime as dt

from requests.exceptions import RequestsDependencyWarning

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Add scripts to path so script modules can be imported in isolation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)


@pytest.fixture
def sample_energy_df():
    """Create a sample energy DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100
    base_time = dt.datetime(2023, 1, 1, 0, 0)
    times = [base_time + dt.timedelta(minutes=i * 30) for i in range(n_rows)]
    
    return pl.DataFrame({
        'Time': times,
        'SolarGen': np.abs(np.random.normal(2.0, 1.0, n_rows)),
        'HouseLoad': np.abs(np.random.normal(4.0, 1.0, n_rows)),
        'ImportEnergyPrice': np.random.uniform(0.1, 0.3, n_rows),
        'ExportEnergyPrice': np.random.uniform(0.05, 0.15, n_rows),
    })


@pytest.fixture
def small_env_df():
    """Create a minimal DataFrame for quick environment testing."""
    return pl.DataFrame({
        'Time': ['2025-01-01T00:00', '2025-01-01T00:30'],
        'SolarGen': [2.0, 0.0],
        'HouseLoad': [1.0, 3.0],
        'ImportEnergyPrice': [0.3, 0.3],
        'ExportEnergyPrice': [0.05, 0.05],
    })
