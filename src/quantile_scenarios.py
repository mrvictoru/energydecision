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
        
        if not columns:
            raise ValueError("No numeric columns available for scenario generation.")

        if group_by is not None and group_by not in df.columns:
            raise ValueError(f"group_by column '{group_by}' not found in DataFrame")
        
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
                # Skip obvious time/date columns but keep numeric ids and other numeric features
                col_lower = col.lower()
                if not any(
                    keyword == col_lower or col_lower.endswith('_' + keyword) or col_lower.startswith(keyword + '_')
                    for keyword in ['timestamp', 'time', 'date']
                ):
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
        # Calculate quantiles for each column and collect into a single row
        quantile_exprs = []

        for col in columns:
            for i, quantile in enumerate(self.quantiles, 1):
                scenario_col_name = f"{self.scenario_prefix}_{i}_{col}"
                quantile_exprs.append(pl.col(col).quantile(quantile).alias(scenario_col_name))

        quantiles_df = df.select(quantile_exprs).head(1)

        # Build literal expressions for all scenario columns at once for efficiency
        lit_exprs = []
        for col in columns:
            for i, _ in enumerate(self.quantiles, 1):
                scenario_col_name = f"{self.scenario_prefix}_{i}_{col}"
                scenario_value = quantiles_df[scenario_col_name].item()
                lit_exprs.append(pl.lit(scenario_value).alias(scenario_col_name))

        result_df = df.clone().with_columns(lit_exprs)
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
                quantile_exprs.append(pl.col(col).quantile(quantile).alias(scenario_col_name))

        # Use modern Polars API for grouping and aggregation
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

            # Now attach for each row in df the group's scenario arrays (avoid to_dicts())
            result: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
            group_keys = df.select(group_by).to_series().to_numpy()
            for idx, group_key in enumerate(group_keys):
                if group_key not in group_mapping:
                    raise RuntimeError(f"Missing group quantiles for group {group_key}")
                per_vars = {}
                for vname in variables.keys():
                    values = group_mapping[group_key][vname]
                    probs = _uniform_probs(len(values))
                    per_vars[vname] = (values, probs)
                result[idx] = per_vars
            return result

        # No grouping: compute global quantiles for each variable once
        result: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        # Precompute global quantile values per variable
        global_vals: Dict[str, np.ndarray] = {}
        for vname, col in variables.items():
            qs = var_quantiles[vname]
            values = np.array([df.select(pl.col(col).quantile(q)).to_series().item() for q in qs], dtype=float)
            global_vals[vname] = values

        # Build per-row mapping using time index or row order
        if time_index_col is not None and time_index_col not in df.columns:
            raise ValueError(f"time_index_col '{time_index_col}' not found in DataFrame")

        nrows = df.height
        for idx in range(nrows):
            per_vars = {}
            for vname in variables.keys():
                values = global_vals[vname]
                probs = _uniform_probs(len(values))
                per_vars[vname] = (values, probs)
            result[idx] = per_vars

        return result

    def expected_cost_cartesian(
        self,
        values_solar: np.ndarray,
        probs_solar: np.ndarray,
        values_load: np.ndarray,
        probs_load: np.ndarray,
        values_imp: np.ndarray,
        probs_imp: np.ndarray,
        values_exp: np.ndarray,
        probs_exp: np.ndarray,
        stage_cost_fn,
    ) -> float:
        """
        Compute expected stage cost by enumerating the Cartesian product of four
        independent marginal scenarios. The provided stage_cost_fn should accept
        arguments (solar, load, import_price, export_price) and return the cost
        (float) for that realization.

        This implements the blueprint's expectation: E[cost] = sum_{i,j,k,l}
        probs_solar[i]*probs_load[j]*probs_imp[k]*probs_exp[l] * stage_cost(...)
        """
        # Quick shape checks
        if len(values_solar) != len(probs_solar) or len(values_load) != len(probs_load) or \
           len(values_imp) != len(probs_imp) or len(values_exp) != len(probs_exp):
            raise ValueError("Values and probs arrays must have matching lengths for each variable")

        expected = 0.0
        # Enumerate cartesian product; keep it simple and readable
        for i, (vs, ps) in enumerate(zip(values_solar, probs_solar)):
            for j, (vl, pl) in enumerate(zip(values_load, probs_load)):
                for k, (vi, pi) in enumerate(zip(values_imp, probs_imp)):
                    for l, (ve, pe) in enumerate(zip(values_exp, probs_exp)):
                        p = ps * pl * pi * pe
                        if p == 0.0:
                            continue
                        c = stage_cost_fn(vs, vl, vi, ve)
                        expected += p * c

        return float(expected)

    def expected_cost_monte_carlo(
        self,
        values_solar: np.ndarray,
        probs_solar: np.ndarray,
        values_load: np.ndarray,
        probs_load: np.ndarray,
        values_imp: np.ndarray,
        probs_imp: np.ndarray,
        values_exp: np.ndarray,
        probs_exp: np.ndarray,
        stage_cost_fn,
        n_samples: int = 100,
        rng_seed: Optional[int] = None,
    ) -> float:
        """
        Approximate expected stage cost by Monte Carlo sampling of joint scenarios.

        Args:
            values_*/probs_*: Marginal discrete values and probabilities for each variable.
            stage_cost_fn: Callable(solar, load, import_price, export_price) -> float
            n_samples: Number of Monte Carlo joint samples (default 100).
            rng_seed: Optional integer seed for reproducibility.

        Returns:
            Approximate expected cost (float). If any sampled realization yields
            `np.inf` cost the method will return `np.inf` immediately because that
            indicates a positive-probability infeasible outcome.

        Notes:
            - This method samples independent marginals to form joint samples.
            - Accuracy improves with larger `n_samples`.
        """
        # basic validation
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")

        # ensure shapes match
        if len(values_solar) != len(probs_solar) or len(values_load) != len(probs_load) or \
           len(values_imp) != len(probs_imp) or len(values_exp) != len(probs_exp):
            raise ValueError("Values and probs arrays must have matching lengths for each variable")

        rng = np.random.default_rng(rng_seed)

        # Precompute cumulative distributions or just use choice with p=probs
        total = 0.0
        for _ in range(n_samples):
            i = rng.choice(len(values_solar), p=probs_solar)
            j = rng.choice(len(values_load), p=probs_load)
            k = rng.choice(len(values_imp), p=probs_imp)
            l = rng.choice(len(values_exp), p=probs_exp)

            vs = float(values_solar[i])
            vl = float(values_load[j])
            vi = float(values_imp[k])
            ve = float(values_exp[l])

            c = stage_cost_fn(vs, vl, vi, ve)
            if c == np.inf:
                # If any sampled realization is infeasible with infinite cost,
                # the true expectation (if that realization has >0 probability)
                # is infinite; return immediately.
                return np.inf
            total += float(c)

        return float(total / float(n_samples))