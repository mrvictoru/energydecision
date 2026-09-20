"""
Quantile-based scenario generation for Polars DataFrames.

This module provides functionality to generate scenarios based on quantiles
of historical data, supporting uncertainty modeling for energy decision making.
"""

import polars as pl
import numpy as np
from typing import List, Optional, Union
from typing import Dict, Tuple


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
    
    # New methods to satisfy the blueprint described in the project:
    # For each time step t produce arrays of values and probabilities for each variable
    # (Solar, Load, ImportPrice, ExportPrice). Probabilities are equal by default
    # (uniform over quantile bins) but callers may supply explicit quantiles per variable.
    def generate_time_step_scenarios(
        self,
        df: pl.DataFrame,
        time_index_col: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        per_variable_quantiles: Optional[Dict[str, List[float]]] = None,
        group_by: Optional[str] = None,
    ) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate per-time-step marginal scenario arrays and probabilities for the
        four variables required by the blueprint.

        Args:
            df: Input Polars DataFrame containing historical data.
            time_index_col: Optional column name to use as the time index. If None,
                row order (0..N-1) will be used as time steps.
            variables: Optional mapping of blueprint variable names to DataFrame
                columns. Expected keys: 'solar', 'load', 'import_price', 'export_price'.
                Defaults to common names if not provided.
            per_variable_quantiles: Optional dict mapping variable name to list of
                quantiles to use for that variable. If not provided the generator's
                configured quantiles are used for all variables.
            group_by: Optional column name to group by when computing quantiles.

        Returns:
            A dict keyed by time-step index (int). Each value is a dict mapping
            variable short names ('solar','load','import_price','export_price') to
            a tuple (values: np.ndarray, probs: np.ndarray).

        Notes:
            - If `group_by` is provided, quantiles are computed per-group and then
              attached to every row in that group (so values can change by group).
            - Probabilities are uniform across the quantile bins by default.
        """
        # Default variable column mapping
        default_vars = {
            'solar': 'SolarGen',
            'load': 'HouseLoad',
            'import_price': 'ImportEnergyPrice',
            'export_price': 'ExportEnergyPrice'
        }
        variables = variables or default_vars

        # Validate variable columns exist
        for vname, col in variables.items():
            if col not in df.columns:
                raise ValueError(f"Column for variable '{vname}' not found in DataFrame: {col}")

        # Determine per-variable quantiles
        var_quantiles: Dict[str, List[float]] = {}
        for vname in variables.keys():
            if per_variable_quantiles and vname in per_variable_quantiles:
                qs = per_variable_quantiles[vname]
            else:
                qs = self.quantiles
            if not qs:
                raise ValueError(f"No quantiles provided for variable {vname}")
            var_quantiles[vname] = sorted(qs)

        # Helper to build uniform probs for a list of values
        def _uniform_probs(n: int) -> np.ndarray:
            if n <= 0:
                return np.array([])
            return np.repeat(1.0 / n, n)

        # If grouping is requested compute quantiles per group else global
        if group_by is not None:
            if group_by not in df.columns:
                raise ValueError(f"group_by column '{group_by}' not found in DataFrame")

            # Build aggregation expressions to compute quantiles per variable
            agg_exprs = []
            for vname, col in variables.items():
                qs = var_quantiles[vname]
                for i, q in enumerate(qs, 1):
                    agg_exprs.append(pl.col(col).quantile(q).alias(f"{vname}_q{i}"))

            group_q = df.group_by(group_by).agg(agg_exprs)
            # Convert group quantiles to a mapping: group_value -> {vname: np.array(values)}
            # Build list of quantile column names in the same order they were aggregated
            quantile_col_names = []
            for vname in variables.keys():
                qs = var_quantiles[vname]
                for i in range(1, len(qs) + 1):
                    quantile_col_names.append(f"{vname}_q{i}")

            # Select group key plus quantile columns and convert to numpy array once
            sel_cols = [group_by] + quantile_col_names
            group_arr = group_q.select(sel_cols).to_numpy()

            group_mapping = {}
            for row in group_arr:
                key = row[0]
                vals = {}
                start = 1
                for vname in variables.keys():
                    qs = var_quantiles[vname]
                    n = len(qs)
                    values = np.array(row[start:start + n], dtype=float)
                    vals[vname] = values
                    start += n
                group_mapping[key] = vals

            # Build per-variable numpy arrays shaped (n_rows, n_quantiles_for_var)
            nrows = df.height
            result_mapping: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            group_keys = df.select(group_by).to_series().to_numpy()
            for vname in variables.keys():
                qs = var_quantiles[vname]
                n_q = len(qs)
                vals_arr = np.zeros((nrows, n_q), dtype=float)
                for idx, group_key in enumerate(group_keys):
                    if group_key not in group_mapping:
                        raise RuntimeError(f"Missing group quantiles for group {group_key}")
                    vals_arr[idx, :] = group_mapping[group_key][vname]
                probs_vec = _uniform_probs(n_q)
                probs_arr = np.tile(probs_vec.reshape(1, n_q), (nrows, 1))
                result_mapping[vname] = (vals_arr, probs_arr)
            return result_mapping

        # No grouping: compute global quantiles for each variable once
        # For no-group case, build per-variable arrays where each row repeats the global quantiles
        result_mapping: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        
        # Batch compute all quantiles in a single query for efficiency
        quantile_exprs = []
        for vname, col in variables.items():
            qs = var_quantiles[vname]
            for i, q in enumerate(qs):
                quantile_exprs.append(pl.col(col).quantile(q).alias(f"{vname}_q{i}"))
        
        # Single query to compute all quantiles
        quantiles_result = df.select(quantile_exprs)
        
        # Extract results from the single query
        global_vals: Dict[str, np.ndarray] = {}
        for vname, col in variables.items():
            qs = var_quantiles[vname]
            values = np.array([quantiles_result[f"{vname}_q{i}"].item() for i in range(len(qs))], dtype=float)
            global_vals[vname] = values

        # Build per-row mapping using time index or row order
        if time_index_col is not None and time_index_col not in df.columns:
            raise ValueError(f"time_index_col '{time_index_col}' not found in DataFrame")

        nrows = df.height
        for vname in variables.keys():
            values = global_vals[vname]
            n_q = len(values)
            vals_arr = np.tile(values.reshape(1, n_q), (nrows, 1))
            probs_arr = np.tile(_uniform_probs(n_q).reshape(1, n_q), (nrows, 1))
            result_mapping[vname] = (vals_arr, probs_arr)

        return result_mapping


