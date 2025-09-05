"""
Quantile-based scenario generation for Polars DataFrames.

This module provides functionality to generate scenarios based on quantiles
of historical data, supporting uncertainty modeling for energy decision making.
"""

import polars as pl
import numpy as np
from typing import List, Optional, Union


class QuantileScenarioGenerator:
    """
    Generates scenarios for Polars DataFrames based on quantiles of the data.
    
    This class creates multiple scenarios by sampling from quantiles of the
    historical data distribution. By default, it generates 5 scenarios per
    variable using evenly spaced quantiles.
    
    Attributes:
        n_scenarios (int): Number of scenarios to generate per variable.
        quantiles (List[float]): Specific quantiles to use for scenario generation.
        scenario_prefix (str): Prefix for scenario column names.
    """
    
    def __init__(
        self, 
        n_scenarios: int = 5,
        quantiles: Optional[List[float]] = None,
        scenario_prefix: str = "scenario"
    ):
        """
        Initialize the QuantileScenarioGenerator.
        
        Args:
            n_scenarios (int): Number of scenarios to generate. Default is 5.
            quantiles (Optional[List[float]]): Custom quantiles to use. If None,
                will generate evenly spaced quantiles from 0.1 to 0.9.
            scenario_prefix (str): Prefix for scenario column names. Default is "scenario".
        
        Raises:
            ValueError: If n_scenarios < 1 or quantiles are not between 0 and 1.
        """
        if n_scenarios < 1:
            raise ValueError("n_scenarios must be at least 1")
        
        self.n_scenarios = n_scenarios
        self.scenario_prefix = scenario_prefix
        
        if quantiles is not None:
            if not all(0 <= q <= 1 for q in quantiles):
                raise ValueError("All quantiles must be between 0 and 1")
            if len(quantiles) != n_scenarios:
                raise ValueError(f"Number of quantiles ({len(quantiles)}) must match n_scenarios ({n_scenarios})")
            self.quantiles = sorted(quantiles)
        else:
            # Generate evenly spaced quantiles
            if n_scenarios == 1:
                self.quantiles = [0.5]  # Use median for single scenario
            else:
                self.quantiles = [i / (n_scenarios + 1) for i in range(1, n_scenarios + 1)]
    
    def generate_scenarios(
        self, 
        df: pl.DataFrame, 
        columns: Optional[List[str]] = None,
        group_by: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Generate scenarios for specified columns in a Polars DataFrame.
        
        Args:
            df (pl.DataFrame): Input DataFrame containing historical data.
            columns (Optional[List[str]]): List of column names to generate scenarios for.
                If None, will attempt to auto-detect numeric columns suitable for scenarios.
            group_by (Optional[str]): Column name to group by when calculating quantiles.
                Useful for generating scenarios by time period, location, etc.
        
        Returns:
            pl.DataFrame: Original DataFrame with additional scenario columns added.
        
        Raises:
            ValueError: If specified columns don't exist in DataFrame or contain non-numeric data.
        """
        if columns is None:
            columns = self._auto_detect_scenario_columns(df)
        
        # Validate columns exist and are numeric
        self._validate_columns(df, columns)
        
        result_df = df.clone()
        
        if group_by is not None:
            # Generate scenarios with grouping
            result_df = self._generate_scenarios_grouped(result_df, columns, group_by)
        else:
            # Generate scenarios for entire dataset
            result_df = self._generate_scenarios_global(result_df, columns)
        
        return result_df
    
    def _auto_detect_scenario_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Auto-detect numeric columns suitable for scenario generation.
        
        Args:
            df (pl.DataFrame): Input DataFrame.
        
        Returns:
            List[str]: List of column names suitable for scenario generation.
        """
        numeric_types = [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
        
        scenario_columns = []
        for col in df.columns:
            if df[col].dtype in numeric_types:
                # Skip columns that look like primary identifiers or timestamps
                col_lower = col.lower()
                if not any(keyword == col_lower or col_lower.endswith('_' + keyword) or col_lower.startswith(keyword + '_') 
                          for keyword in ['timestamp', 'time', 'date']):
                    # Don't skip location_id or similar meaningful numeric identifiers
                    if not (col_lower in ['id'] or col_lower.endswith('_id') and len(col_lower) <= 3):
                        scenario_columns.append(col)
        
        return scenario_columns
    
    def _validate_columns(self, df: pl.DataFrame, columns: List[str]) -> None:
        """
        Validate that specified columns exist and are numeric.
        
        Args:
            df (pl.DataFrame): Input DataFrame.
            columns (List[str]): Column names to validate.
        
        Raises:
            ValueError: If columns don't exist or aren't numeric.
        """
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
        
        numeric_types = [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
        
        non_numeric_columns = [col for col in columns if df[col].dtype not in numeric_types]
        if non_numeric_columns:
            raise ValueError(f"Columns must be numeric for scenario generation: {non_numeric_columns}")
    
    def _generate_scenarios_global(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """
        Generate scenarios using global quantiles across entire dataset.
        
        Args:
            df (pl.DataFrame): Input DataFrame.
            columns (List[str]): Column names to generate scenarios for.
        
        Returns:
            pl.DataFrame: DataFrame with scenario columns added.
        """
        # Calculate quantiles for each column
        quantile_exprs = []
        
        for col in columns:
            for i, quantile in enumerate(self.quantiles, 1):
                scenario_col_name = f"{self.scenario_prefix}_{i}_{col}"
                quantile_exprs.append(
                    pl.col(col).quantile(quantile).alias(scenario_col_name)
                )
        
        # Calculate all quantiles
        quantiles_df = df.select(quantile_exprs).head(1)
        
        # Add scenario columns to original dataframe
        result_df = df.clone()
        
        for col in columns:
            for i, quantile in enumerate(self.quantiles, 1):
                scenario_col_name = f"{self.scenario_prefix}_{i}_{col}"
                scenario_value = quantiles_df[scenario_col_name].item()
                result_df = result_df.with_columns(
                    pl.lit(scenario_value).alias(scenario_col_name)
                )
        
        return result_df
    
    def _generate_scenarios_grouped(
        self, 
        df: pl.DataFrame, 
        columns: List[str], 
        group_by: str
    ) -> pl.DataFrame:
        """
        Generate scenarios using quantiles calculated within groups.
        
        Args:
            df (pl.DataFrame): Input DataFrame.
            columns (List[str]): Column names to generate scenarios for.
            group_by (str): Column name to group by.
        
        Returns:
            pl.DataFrame: DataFrame with scenario columns added.
        """
        # Calculate quantiles by group
        quantile_exprs = []
        
        for col in columns:
            for i, quantile in enumerate(self.quantiles, 1):
                scenario_col_name = f"{self.scenario_prefix}_{i}_{col}"
                quantile_exprs.append(
                    pl.col(col).quantile(quantile).alias(scenario_col_name)
                )
        
        # Calculate quantiles by group
        group_quantiles = df.group_by(group_by).agg(quantile_exprs)
        
        # Join back to original dataframe
        result_df = df.join(group_quantiles, on=group_by, how="left")
        
        return result_df
    
    def get_scenario_columns(self, base_columns: List[str]) -> List[str]:
        """
        Get the names of scenario columns that would be generated for given base columns.
        
        Args:
            base_columns (List[str]): Base column names.
        
        Returns:
            List[str]: List of scenario column names.
        """
        scenario_columns = []
        for col in base_columns:
            for i in range(1, self.n_scenarios + 1):
                scenario_columns.append(f"{self.scenario_prefix}_{i}_{col}")
        return scenario_columns
    
    def summarize_scenarios(self, df: pl.DataFrame, base_columns: List[str]) -> pl.DataFrame:
        """
        Create a summary of generated scenarios showing quantile values.
        
        Args:
            df (pl.DataFrame): DataFrame with generated scenarios.
            base_columns (List[str]): Original column names scenarios were generated for.
        
        Returns:
            pl.DataFrame: Summary DataFrame with scenario statistics.
        """
        summary_data = []
        
        for col in base_columns:
            for i, quantile in enumerate(self.quantiles, 1):
                scenario_col = f"{self.scenario_prefix}_{i}_{col}"
                if scenario_col in df.columns:
                    scenario_value = df[scenario_col].head(1).item()
                    summary_data.append({
                        "base_column": col,
                        "scenario": i,
                        "quantile": quantile,
                        "value": scenario_value
                    })
        
        return pl.DataFrame(summary_data)