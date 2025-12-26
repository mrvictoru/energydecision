# Scenario Generation Support

This document describes the quantile-based scenario generation functionality for Polars DataFrames in the energydecision package.

> **Module Location:** [`src/quantile_scenarios.py`](src/quantile_scenarios.py)

## Overview

The scenario generation module provides tools for creating multiple scenarios from historical energy data based on quantiles of the data distribution. This is useful for uncertainty modeling in energy decision making applications, where you need to evaluate system performance under different possible future conditions.

## Key Features

- **Quantile-based scenario generation**: Creates scenarios using statistical quantiles of historical data
- **Configurable number of scenarios**: Default is 5 scenarios per variable, but can be customized
- **Automatic column detection**: Intelligently identifies numeric columns suitable for scenario generation
- **Grouped scenario generation**: Generate scenarios within data groups (e.g., by customer, location, time period)
- **Polars DataFrame integration**: Seamlessly integrates with existing Polars workflows
- **Monte Carlo expected cost computation**: Calculate expected costs using Monte Carlo sampling or Cartesian product methods
- **Time-step scenario arrays**: Generate per-timestep scenario arrays for SDP optimization

## Quick Start

```python
import polars as pl
from src.quantile_scenarios import QuantileScenarioGenerator

# Load your energy data
df = pl.read_csv("energy_data.csv")

# Create scenario generator (defaults to 5 scenarios)
generator = QuantileScenarioGenerator()

# Generate scenarios for specific columns
scenarios_df = generator.generate_scenarios(
    df, 
    columns=['SolarGen', 'HouseLoad', 'ImportEnergyPrice']
)

# The result includes original data plus scenario columns
print(scenarios_df.columns)
# Output: [...original columns..., 'scenario_1_SolarGen', 'scenario_2_SolarGen', ...]
```

## Basic Usage

### Default Configuration

By default, the `QuantileScenarioGenerator` creates 5 scenarios per variable using evenly spaced quantiles:

```python
generator = QuantileScenarioGenerator()
# Uses quantiles: [1/6, 2/6, 3/6, 4/6, 5/6] ≈ [0.167, 0.333, 0.5, 0.667, 0.833]
```

### Custom Number of Scenarios

```python
# Generate 3 scenarios per variable
generator = QuantileScenarioGenerator(n_scenarios=3)
# Uses quantiles: [0.25, 0.5, 0.75]

# Generate single scenario (median)
generator = QuantileScenarioGenerator(n_scenarios=1)
# Uses quantile: [0.5]
```

### Custom Quantiles

```python
# Specify exact quantiles to use
generator = QuantileScenarioGenerator(
    n_scenarios=5,
    quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]
)
```

### Custom Column Prefix

```python
# Change the prefix for scenario columns
generator = QuantileScenarioGenerator(scenario_prefix="forecast")
# Creates columns like: 'forecast_1_SolarGen', 'forecast_2_SolarGen', etc.
```

## Advanced Usage

### Automatic Column Detection

If you don't specify columns, the generator will automatically detect suitable numeric columns:

```python
# Automatically detects numeric columns (excludes timestamps, IDs, etc.)
scenarios_df = generator.generate_scenarios(df)
```

The auto-detection excludes:
- Timestamp and date columns
- Primary key columns (named 'id' or ending with '_id' and short)
- Non-numeric columns

### Grouped Scenario Generation

Generate different scenarios for different groups in your data:

```python
# Generate scenarios by customer
scenarios_df = generator.generate_scenarios(
    df, 
    columns=['SolarGen', 'HouseLoad'],
    group_by='Customer'
)

# Each customer will have their own scenario values based on their historical data
```

### Working with Energy Time Series Data

Common pattern for energy datasets:

```python
import polars as pl
from src.quantile_scenarios import QuantileScenarioGenerator

# Load energy data
df = pl.read_csv("solar_data.csv")

# Generate scenarios for key energy variables
generator = QuantileScenarioGenerator(n_scenarios=5)

energy_scenarios = generator.generate_scenarios(
    df,
    columns=[
        'SolarGen',           # Solar generation (kW)
        'HouseLoad',          # House load (kW)
        'ImportEnergyPrice',  # Grid import price ($/kWh)
        'ExportEnergyPrice'   # Grid export price ($/kWh)
    ]
)

# Use scenarios for planning and optimization
for scenario_num in range(1, 6):
    scenario_cols = [f'scenario_{scenario_num}_{col}' 
                    for col in ['SolarGen', 'HouseLoad', 'ImportEnergyPrice', 'ExportEnergyPrice']]
    
    scenario_data = energy_scenarios.select(['timestamp'] + scenario_cols)
    # Run your energy optimization using scenario_data
```

### Generate Time-Step Scenario Arrays for SDP

For use with Stochastic Dynamic Programming:

```python
from src.quantile_scenarios import QuantileScenarioGenerator

generator = QuantileScenarioGenerator(n_scenarios=5)

# Generate per-timestep scenario arrays
scenario_cache = generator.generate_time_step_scenarios(df)

# Returns dict with arrays for each variable:
# scenario_cache['solar'] -> (values_array, probabilities_array)
# scenario_cache['load'] -> (values_array, probabilities_array)
# etc.
```

### Expected Cost Computation

Calculate expected costs using Monte Carlo or Cartesian product methods:

```python
# Monte Carlo expected cost
mc_cost = generator.expected_cost_monte_carlo(
    values_solar, probs_solar,
    values_load, probs_load,
    values_imp, probs_imp,
    values_exp, probs_exp,
    stage_cost_function,
    n_samples=1000,
    rng_seed=42
)

# Exact Cartesian expected cost (for small scenario counts)
exact_cost = generator.expected_cost_cartesian(
    values_solar, probs_solar,
    values_load, probs_load,
    values_imp, probs_imp,
    values_exp, probs_exp,
    stage_cost_function
)
```

## Utility Functions

### Get Scenario Column Names

```python
base_columns = ['SolarGen', 'HouseLoad']
scenario_columns = generator.get_scenario_columns(base_columns)
print(scenario_columns)
# Output: ['scenario_1_SolarGen', 'scenario_2_SolarGen', ...]
```

### Summarize Generated Scenarios

```python
summary = generator.summarize_scenarios(scenarios_df, base_columns)
print(summary)
# Shows quantile values for each scenario and variable
```

## Integration with Existing Workflows

The scenario generation integrates seamlessly with existing Polars operations:

```python
# Filter scenarios
customer_a_scenarios = scenarios_df.filter(pl.col('Customer') == 'A')

# Aggregate scenarios
scenario_stats = scenarios_df.group_by('month').agg([
    pl.col('scenario_1_SolarGen').mean().alias('avg_low_solar'),
    pl.col('scenario_5_SolarGen').mean().alias('avg_high_solar')
])

# Select specific scenarios for analysis
selected_scenarios = scenarios_df.select([
    'timestamp', 'Customer',
    'scenario_1_SolarGen', 'scenario_3_SolarGen', 'scenario_5_SolarGen'
])
```

## Default Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `n_scenarios` | 5 | Number of scenarios to generate per variable |
| `quantiles` | Auto-generated | Evenly spaced quantiles based on n_scenarios |
| `scenario_prefix` | "scenario" | Prefix for scenario column names |
| `columns` | Auto-detected | Columns to generate scenarios for |
| `group_by` | None | Column to group by for scenario generation |

## Example Output

Given input data with columns `['timestamp', 'SolarGen', 'HouseLoad']` and using default settings:

**Input:**
```
┌─────────────┬───────────┬───────────┐
│ timestamp   ┆ SolarGen  ┆ HouseLoad │
│ ---         ┆ ---       ┆ ---       │
│ date        ┆ f64       ┆ f64       │
╞═════════════╪═══════════╪═══════════╡
│ 2023-01-01  ┆ 4.2       ┆ 6.1       │
│ 2023-01-02  ┆ 3.8       ┆ 5.9       │
│ ...         ┆ ...       ┆ ...       │
└─────────────┴───────────┴───────────┘
```

**Output (with additional scenario columns):**
```
┌─────────────┬───────────┬───────────┬───────────────────┬───────────────────┬─...─┐
│ timestamp   ┆ SolarGen  ┆ HouseLoad ┆ scenario_1_SolarG ┆ scenario_2_SolarG ┆     │
│ ---         ┆ ---       ┆ ---       ┆ en                ┆ en                ┆     │
│ date        ┆ f64       ┆ f64       ┆ ---               ┆ ---               ┆     │
│             ┆           ┆           ┆ f64               ┆ f64               ┆     │
╞═════════════╪═══════════╪═══════════╪═══════════════════╪═══════════════════╪═════╡
│ 2023-01-01  ┆ 4.2       ┆ 6.1       ┆ 2.1               ┆ 3.5               ┆ ... │
│ 2023-01-02  ┆ 3.8       ┆ 5.9       ┆ 2.1               ┆ 3.5               ┆ ... │
│ ...         ┆ ...       ┆ ...       ┆ ...               ┆ ...               ┆ ... │
└─────────────┴───────────┴───────────┴───────────────────┴───────────────────┴─────┘
```

## Error Handling

The module provides clear error messages for common issues:

- **Invalid number of scenarios**: Must be at least 1
- **Invalid quantiles**: Must be between 0 and 1, and match the number of scenarios
- **Missing columns**: Specified columns must exist in the DataFrame
- **Non-numeric columns**: Only numeric columns can be used for scenario generation

## Performance Considerations

- Scenario generation is efficient for typical energy datasets (thousands to millions of rows)
- Grouped scenario generation may be slower for datasets with many groups
- Generated scenarios add columns to your DataFrame, increasing memory usage
- Consider generating scenarios only for the columns you actually need

## Best Practices

1. **Start with defaults**: The default 5 scenarios work well for most applications
2. **Use meaningful groups**: Group by relevant dimensions like customer, location, or time period
3. **Validate scenarios**: Check that generated scenarios make sense for your domain
4. **Document your choices**: Record which quantiles and groupings you used for reproducibility
5. **Consider storage**: Large datasets with many scenarios can become memory-intensive

## Integration with Energy Decision Workflows

This scenario generation module is designed to work with the broader energydecision package:

```python
from src.quantile_scenarios import QuantileScenarioGenerator
from src.helper import transform_polars_df

# Transform raw energy data
cleaned_df = transform_polars_df(raw_df)

# Generate scenarios
generator = QuantileScenarioGenerator()
scenarios_df = generator.generate_scenarios(cleaned_df)

# Use with energy decision agents and environments
# (scenarios_df can be used with SolarBatteryEnv and Agent classes)
```

## Testing

The scenario generation module is thoroughly tested. Run tests with:

```bash
# Run all scenario tests
pytest tests/test_quantile_scenarios.py -v

# Run specific test
pytest tests/test_quantile_scenarios.py::TestQuantileScenarioGenerator::test_expected_cost_monte_carlo -v
```

See [`tests/test_quantile_scenarios.py`](tests/test_quantile_scenarios.py) for 21 comprehensive test cases covering:
- Initialization and configuration
- Column validation and auto-detection
- Scenario generation (basic, grouped, custom prefix)
- Edge cases (single value, small datasets)
- Monte Carlo expected cost computation
- Integration with Polars operations

## Related Documentation

- [README.md](README.md) - Main project documentation
- [MRDP_README.md](MRDP_README.md) - Multi-Resolution Dynamic Programming documentation
- [PERFORMANCE_IMPROVEMENTS.md](PERFORMANCE_IMPROVEMENTS.md) - Performance optimization details