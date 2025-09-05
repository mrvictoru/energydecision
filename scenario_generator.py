import polars as pl
import numpy as np

def add_quantile_scenarios(
    df: pl.DataFrame,
    variables: list[str],
    n_scenarios: int = 5,
) -> pl.DataFrame:
    """
    For each variable, generate quantile-based scenarios:
    - Compute quantile levels [0, 1/(n-1), ..., 1]
    - Use static quantiles over the entire df for simplicity.
    - Adds new columns `<var>_s0..s{n-1}` for values and `<var>_p0..p{n-1}` for probabilities (uniform).
    """
    # Quantile levels and uniform probabilities
    q_levels = [i/(n_scenarios-1) for i in range(n_scenarios)]
    probs = [1.0/n_scenarios] * n_scenarios

    for var in variables:
        # Extract series and compute quantiles
        arr = df[var].to_numpy()
        values = np.quantile(arr, q_levels)
        for idx, val in enumerate(values):
            df = df.with_column(pl.lit(float(val)).alias(f"{var}_s{idx}"))
            df = df.with_column(pl.lit(float(probs[idx])).alias(f"{var}_p{idx}"))
    return df
