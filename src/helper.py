import polars as pl
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
from EnergySimEnv import SolarBatteryEnv


# Helper: parse a time string like "7am" or "7:30am" into minutes since midnight.
def parse_time(time_str: str) -> int:
    time_str = time_str.strip().lower()
    try:
        if ":" in time_str:
            dt = datetime.strptime(time_str, "%I:%M%p")
        else:
            dt = datetime.strptime(time_str, "%I%p")
    except Exception:
        raise ValueError(f"Time format not recognized: {time_str}")
    return dt.hour * 60 + dt.minute

def transform_polars_df(
    df: pl.DataFrame,
    import_energy_price: float = 0.23, #in USD
    export_energy_price: float = 0.04, #in USD
    price_periods: str = None,  # Expects string in format "7am – 10am | 4pm – 9pm"
    default_import_energy_price: float = 0.1, #in USD
    default_export_energy_price: float = 0.02 #in USD
) -> pl.DataFrame:
    """
    Transforms an input Polars DataFrame into a format for the SolarBatteryEnv.
    Adds meta data columns: Customer, Postcode, DateRange.
    """
    # use regex to check if price_periods is in the correct format
    if price_periods is not None:
        if not re.match(r"(\d{1,2}(:\d{2})?[ap]m\s*–\s*\d{1,2}(:\d{2})?[ap]m\s*\|?\s*)+", price_periods):
            raise ValueError("price_periods should be in the format '7am – 10am | 4pm – 9pm'")

    # Define the known columns present in the input
    known_cols = {'Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'date', 'Row Quality'}

    # check if the known columns are present in the input
    if not all(col in df.columns for col in known_cols):
        raise ValueError("Input DataFrame is missing required columns")

    # Identify time columns (all columns not in known_cols)
    time_cols = [col for col in df.columns if col not in known_cols]
    
    # Unpivot the dataframe so that all time columns become rows.
    unpivoted = df.unpivot(index=["date", "Consumption Category"], on=time_cols,
                           variable_name="time", value_name="measurement")

    # Create a 'Time' column by concatenating 'date' and 'time'
    unpivoted = unpivoted.with_columns(
        (pl.col("date").cast(pl.Utf8) + " " + pl.col("time")).alias("Time")
    )

    # Convert the 'Time' column from string to datetime using the given format.
    unpivoted = unpivoted.with_columns(
        pl.col("Time").str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M", strict=False)
    )

    # Remove rows where time conversion failed.
    unpivoted = unpivoted.filter(pl.col("Time").is_not_null())

    # Preaggregate: for each Time and Consumption Category, sum the measurements.
    aggregated = unpivoted.group_by(["Time", "Consumption Category"]).agg(
        pl.col("measurement").sum().alias("measurement")
    )
    
    # Pivot the aggregated data so that each 'Consumption Category' becomes its own column.
    pivot = aggregated.pivot(
        index="Time",
        on="Consumption Category",
        values="measurement"
    )
    
    # Create SolarGen from 'GG'
    pivot = pivot.with_columns(
        pl.col("GG").fill_null(0.0).alias("SolarGen")
    )
    
    # Ensure 'CL' column exists; if not, create it with a default value of 0.0.
    if "CL" not in pivot.columns:
        pivot = pivot.with_columns(pl.lit(0.0).alias("CL"))
    
    # Create HouseLoad by summing 'GC' and 'CL'
    pivot = pivot.with_columns([
        (pl.col("GC").fill_null(0.0) + pl.col("CL").fill_null(0.0)).alias("HouseLoad")
    ])
    
    # Apply custom energy pricing based on the provided daily time periods.
    if price_periods is not None:
        # Extract minutes from the Time column.
        pivot = pivot.with_columns([
            pl.col("Time")
            .dt.strftime("%H")
            .cast(pl.Int32)
            .alias("hour"),
            pl.col("Time")
            .dt.strftime("%M")
            .cast(pl.Int32)
            .alias("minute")
        ]).with_columns(
            (pl.col("hour") * 60 + pl.col("minute")).alias("minutes")
        )
        
        # Parse the provided periods.
        periods = []
        for period in price_periods.split("|"):
            period = period.strip()
            # Split on the en-dash
            period_parts = re.split(r"[-–]", period)
            if len(period_parts) != 2:
                raise ValueError(f"Period format not recognized: {period}. Expected format like '7am – 10am'")
            start_minutes = parse_time(period_parts[0])
            end_minutes = parse_time(period_parts[1])
            periods.append((start_minutes, end_minutes))
        
        # Build a condition that checks if the current minute falls within any of the periods.
        condition = pl.lit(False)
        for start, end in periods:
            condition = condition | ((pl.col("minutes") >= start) & (pl.col("minutes") <= end))
        
        # Apply the custom prices when the condition is met.
        pivot = pivot.with_columns([
            pl.when(condition)
              .then(import_energy_price)
              .otherwise(default_import_energy_price)
              .alias("ImportEnergyPrice"),
            pl.when(condition)
              .then(export_energy_price)
              .otherwise(default_export_energy_price)
              .alias("ExportEnergyPrice")
        ])

        # Remove the helper "minutes" column.
        pivot = pivot.drop("minutes")
    else:
        print("No price periods provided, using default prices.")
        pivot = pivot.with_columns([
            pl.lit(import_energy_price).alias("ImportEnergyPrice"),
            pl.lit(export_energy_price).alias("ExportEnergyPrice")
        ])
    
    # Optionally drop original consumption category columns if they exist.
    pivot = pivot.drop(["GG", "GC", "CL"])
    
    # Sort by Time.
    pivot = pivot.sort("Time")

    # Add future columns defaulting to next values
    pivot = pivot.with_columns([
        pl.col("SolarGen").shift(-1).alias("FutureSolar"),
        pl.col("HouseLoad").shift(-1).alias("FutureLoad")
    ])
    # Remove last row to avoid NaN values in future columns
    pivot = pivot.head(pivot.height - 1)

    # Add a column with numerical timestamps while keeping the original Time column
    pivot = pivot.with_columns(
        pl.col("Time").dt.timestamp().alias("Timestamp")
    )
    """"
    # --- Add meta data columns ---
    # Get unique values for Customer and Postcode
    customer = df["Customer"][0] if "Customer" in df.columns else None
    postcode = df["Postcode"][0] if "Postcode" in df.columns else None
    # Get date range
    dates = df["date"].unique().to_list()
    if dates:
        try:
            date_objs = [datetime.strptime(d, "%d/%m/%Y") for d in dates]
            min_date = min(date_objs).strftime("%d/%m/%Y")
            max_date = max(date_objs).strftime("%d/%m/%Y")
            date_range = f"{min_date} - {max_date}"
        except Exception:
            date_range = ""
    else:
        date_range = ""

    pivot = pivot.with_columns([
        pl.lit(customer).alias("Customer"),
        pl.lit(postcode).alias("Postcode"),
        pl.lit(date_range).alias("DateRange")
    ])
    """
    # Regroup the columns
    pivot = pivot.select([
        #"Customer", "Postcode", "DateRange",
        "Timestamp", "SolarGen", "HouseLoad", "FutureSolar", "FutureLoad", "ImportEnergyPrice", "ExportEnergyPrice", "Time"
    ])
    
    return pivot

# Example usage:
# import polars as pl
# df_polars = pl.read_csv("path/to/your/data.csv")
# solar_df_polars = transform_polars_df(
#     df_polars,
#     import_energy_price=0.15,
#     export_energy_price=0.08,
#     price_periods="7am – 10am | 4pm – 9pm",
#     default_import_energy_price=0.1,
#     default_export_energy_price=0.05
# )

# Helper: create an environment instance from a dataset.
def make_env(dataset):
    def _init(max_step = None):
        if max_step is not None:
            env = SolarBatteryEnv(dataset, max_step=max_step)
        else:
            env = SolarBatteryEnv(dataset, max_step=len(dataset)-1)
        return env
    return _init

# Helper: plot a 48-hour window from agent logs.
def plot_48h_from_logs(
    logs_df,
    step_duration=0.5,
    start_step=0,
    logs_df2=None,
    label1="Agent 1",
    label2="Agent 2"
):
    """
    Plots battery, solar, load, grid, and action for a 48-hour window from agent logs.
    Optionally overlays battery level and agent action from a second logs DataFrame.
    
    logs_df: pandas or polars DataFrame with 'raw_observation', 'action', 'info' columns.
    step_duration: duration of each step in hours (default 0.5).
    start_step: index to start the 48-hour window (default 0).
    logs_df2: optional, second logs DataFrame for overlay.
    label1: legend label for first agent.
    label2: legend label for second agent.
    """
    # Convert to pandas if it's a polars DataFrame
    if hasattr(logs_df, "to_pandas"):
        df = logs_df.to_pandas()
    else:
        df = logs_df

    steps_per_hour = int(1 / step_duration)
    steps_48h = 48 * steps_per_hour
    end_step = start_step + steps_48h

    df_48h = df.iloc[start_step:end_step]

    battery = [obs[-2] for obs in df_48h['raw_observation']]
    solar = [obs[5] for obs in df_48h['raw_observation']]
    load = [obs[6] for obs in df_48h['raw_observation']]
    action = [a[0] for a in df_48h['action']]
    grid = [info.get('grid_energy', None) for info in df_48h['info']]

    # If overlay logs provided, extract their 48h window
    if logs_df2 is not None:
        if hasattr(logs_df2, "to_pandas"):
            df2 = logs_df2.to_pandas()
        else:
            df2 = logs_df2
        df2_48h = df2.iloc[start_step:end_step]
        battery2 = [obs[-2] for obs in df2_48h['raw_observation']]
        action2 = [a[0] for a in df2_48h['action']]
    else:
        battery2 = None
        action2 = None

    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(battery, label=f"Battery Level ({label1})", color="tab:blue")
    if battery2 is not None:
        plt.plot(battery2, label=f"Battery Level ({label2})", color="tab:orange", linestyle="--")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(solar, label="Solar Generation (kWh)", color="tab:green")
    plt.plot(load, label="House Load (kWh)", color="tab:red")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(grid, label="Grid Energy (kWh)", color="tab:purple")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(action, label=f"Agent Action ({label1})", color="tab:blue")
    if action2 is not None:
        plt.plot(action2, label=f"Agent Action ({label2})", color="tab:orange", linestyle="--")
    plt.legend()
    plt.xlabel(f"Step ({step_duration}h intervals)")

    plt.tight_layout()
    plt.show()