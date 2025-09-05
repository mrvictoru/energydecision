"""
Scenario generation package for energy decision making.

This package provides tools for generating scenarios from historical data
to support uncertainty modeling in energy decision making applications.
"""

from .quantile_scenarios import QuantileScenarioGenerator

__all__ = ['QuantileScenarioGenerator']