import polars as pl
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Optional, Any, List, Tuple, Union
from dataclasses import dataclass, field
from EnergySimEnv import SolarBatteryEnv
from pathlib import Path

try:
    from scipy.stats import wasserstein_distance  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SCIPY = False
    wasserstein_distance = None  # type: ignore


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
    price_periods: Optional[str] = None,  # Expects string in format "7am – 10am | 4pm – 9pm"
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
        
    # check if columns 'Row Quality' exists
    if "Row Quality" in df.columns:
        # if so remove the column
        df = df.drop("Row Quality")

    # Define the known columns present in the input
    known_cols = {'Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'date'}

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

    # check which string format is used for the date "%d-%b-%y %H:%M" or "%d/%m/%Y %H:%M"
    # get the one value of column 'Time' to check the format
    sample_time = unpivoted["Time"][0]
    if re.match(r"\d{1,2}-[a-zA-Z]{3}-\d{2} \d{1,2}:\d{2}", sample_time):
        unpivoted = unpivoted.with_columns(
            pl.col("Time").str.strptime(pl.Datetime, format="%d-%b-%y %H:%M", strict=False)
        )
    elif re.match(r"\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}", sample_time):
        # Convert the 'Time' column from string to datetime using the given format.
        unpivoted = unpivoted.with_columns(
            pl.col("Time").str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M", strict=False)
        )
    else:
        raise ValueError(f"Time format not recognized: {sample_time}. Expected format like '%d-%b-%y %H:%M' or '%d/%m/%Y %H:%M'.")

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

# Helper: flatten episode data created from run_sb3_model_on_vec_env() from decision.py into a Polars DataFrame
def flatten_episode_data(episode_data):
    dfs = []
    for i, traj in enumerate(episode_data):
        length = len(traj['norm_observation'])
        df = pl.DataFrame({
            'episode_id': [i for _ in range(length)],
            'step': list(range(length)),
            'norm_observation': traj['norm_observation'],
            'raw_observation': traj['raw_observation'],
            'action': traj['actions'],
            'reward': traj['rewards'],
            'info': traj['infos'],
        }, strict=False)
        dfs.append(df)
    return pl.concat(dfs)


import json
# Helper: evaluate a single experiment's episode logs
def evaluate_experiment_logs(
    logs: list[pl.DataFrame],
    target_return: float = 0.0 # for comparison, use the mean reward of a baseline agent's episode
) -> dict:
    """
    Compute key performance metrics for a single experiment's episode logs.
    Returns a dict of:
      - mean_reward, median_reward, std_reward
      - pct_5_reward, pct_95_reward
      - sharpe_ratio, sortino_ratio
      - avg_grid_cost, avg_grid_revenue, avg_degradation_cost
    """
    # total rewards per episode
    total_rewards = [df['reward'].sum() for df in logs]
    rewards_arr = np.array(total_rewards, dtype=float)
    mean_r = rewards_arr.mean()
    std_r = rewards_arr.std(ddof=0)
    # compute 5th, 50th (median), and 95th percentiles of episode rewards
    # Use Python list to satisfy numpy percentile type annotations
    rewards_list = rewards_arr.tolist()
    pct5 = float(np.percentile(rewards_list, 5))
    median_r = float(np.percentile(rewards_list, 50))
    pct95 = float(np.percentile(rewards_list, 95))

    # downside deviation for sortino
    # downside deviation for Sortino: only rewards below target_return
    downs = [r for r in rewards_arr if r < target_return]
    dd = float(np.std(downs, ddof=0)) if downs else 0.0
    sharpe = mean_r / std_r if std_r > 0 else float('nan')
    sortino = (mean_r - target_return) / dd if dd > 0 else float('nan')

    # compute cost components
    grid_costs = []
    grid_revenues = []
    deg_costs = []
    def safe_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    for df in logs:
        infos = df['info'].to_list()
        # Parse info strings to dicts if needed
        parsed_infos = []
        for info in infos:
            if isinstance(info, str):
                try:
                    parsed_infos.append(json.loads(info))
                except Exception:
                    parsed_infos.append({})
            else:
                parsed_infos.append(info)

        # Prefer new fields if available: 'grid_cost' (positive=cost, negative=revenue) and 'deg_cost'
        has_grid_cost_field = any(isinstance(i, dict) and 'grid_cost' in i for i in parsed_infos)
        has_deg_cost_field = any(isinstance(i, dict) and 'deg_cost' in i for i in parsed_infos)

        if has_grid_cost_field:
            total_grid_cost = sum(safe_float(i.get('grid_cost', 0.0)) for i in parsed_infos if safe_float(i.get('grid_cost', 0.0)) > 0)
            total_grid_revenue = -sum(safe_float(i.get('grid_cost', 0.0)) for i in parsed_infos if safe_float(i.get('grid_cost', 0.0)) < 0)
        else:
            # Fallback to older fields: compute cost from grid_reward where grid_energy>0 (importing)
            total_grid_cost = sum(safe_float(i.get('grid_reward', 0.0)) for i in parsed_infos if safe_float(i.get('grid_energy', 0)) > 0)
            total_grid_cost = float(abs(total_grid_cost))
            # revenue from exporting (grid_energy < 0)
            total_grid_revenue = sum(safe_float(i.get('grid_reward', 0.0)) for i in parsed_infos if safe_float(i.get('grid_energy', 0)) < 0)

        if has_deg_cost_field:
            total_deg_cost = sum(safe_float(i.get('deg_cost', 0.0)) for i in parsed_infos)
        else:
            # Fallback to older fields: battery_deg_penalty * battery_life_cost
            total_deg_cost = sum(safe_float(i.get('battery_deg_penalty', 0.0)) * safe_float(i.get('battery_life_cost', 1.0)) for i in parsed_infos)

        grid_costs.append(float(total_grid_cost))
        grid_revenues.append(float(total_grid_revenue))
        deg_costs.append(float(total_deg_cost))
    avg_gc = float(np.mean(grid_costs)) if grid_costs else 0.0
    avg_gr = float(np.mean(grid_revenues)) if grid_revenues else 0.0
    avg_dc = float(np.mean(deg_costs)) if deg_costs else 0.0

    # Average total operational cost per episode = grid_cost - grid_revenue + deg_cost
    operational_costs = [float(gc) - float(gr) + float(dc) for gc, gr, dc in zip(grid_costs, grid_revenues, deg_costs)]
    avg_operational_cost = float(np.mean(operational_costs)) if operational_costs else 0.0

    return {
        'mean_reward': mean_r,
        'median_reward': median_r,
        'std_reward': std_r,
        'pct_5_reward': pct5,
        'pct_95_reward': pct95,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'avg_grid_cost': avg_gc,
        'avg_grid_revenue': avg_gr,
        'avg_deg_cost': avg_dc,
        'avg_operational_cost': avg_operational_cost,
    }


# Helper: evaluate multiple experiments and return a pandas DataFrame

def evaluate_experiments(
    all_logs: dict[str, list[pl.DataFrame]],
    target_return: float = 0.0,
    make_plots: bool = True,
    return_figs: bool = False,
    figsize: tuple = (6,4),
    save_dir: Optional[str] = None,
    save_format: str = "svg",
    dpi: int = 200,
) -> pl.DataFrame | tuple[pl.DataFrame, dict]:
    """
    Given a dict mapping experiment names to lists of episode logs, compute evaluation metrics.
    Optionally create diagnostic plots:
      1) Mean reward bar (with std error bars)
      2) Stacked average cost components (grid vs degradation)
      3) Risk–return scatter (std vs mean, colour by Sharpe)
      4) Episode return distribution (box plot)

    Params:
        all_logs: mapping experiment name -> list of per-episode DataFrames
        target_return: baseline for Sortino (passed to evaluate_experiment_logs)
        make_plots: whether to generate and display plots
        figsize: base figure size (width, height)
        save_dir: if provided, save generated figures into this directory (created if needed)
        save_format: image format for saved figures (e.g., 'png', 'pdf', 'svg')
        dpi: dots-per-inch for saved raster images

    Returns:
        metrics_df (pl.DataFrame) OR (metrics_df, figs_dict) if return_figs=True
    """
    rows = []
    episode_rows = []  # for per-episode reward distribution
    for name, logs in all_logs.items():
        metrics = evaluate_experiment_logs(logs, target_return=target_return)
        metrics['experiment'] = name
        rows.append(metrics)
        # build per-episode total rewards
        for idx, ep_df in enumerate(logs):
            episode_rows.append({
                'experiment': name,
                'episode': idx,
                'total_reward': float(ep_df['reward'].sum())
            })
    metrics_df = pl.DataFrame(rows).sort('experiment')
    # If neither showing nor saving nor returning figs, skip figure generation entirely
    if not make_plots and not return_figs and save_dir is None:
        return metrics_df

    figs: dict[str, plt.Figure] = {}

    try:
        # 1) Mean reward bar with std
        fig1, ax1 = plt.subplots(figsize=figsize)
        x = np.arange(len(metrics_df))
        mean_rewards = metrics_df['mean_reward'].to_list()
        std_rewards = metrics_df['std_reward'].to_list()
        labels = metrics_df['experiment'].to_list()
        ax1.bar(x, mean_rewards, yerr=std_rewards, capsize=4, color='tab:blue', alpha=0.75)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Mean Episode Reward')
        ax1.set_title('Mean Reward ± Std')
        ax1.grid(axis='y', alpha=0.3)
        figs['mean_reward'] = fig1

        # 2) Stacked cost components with percentage annotations
        fig2, ax2 = plt.subplots(figsize=figsize)
        grid_cost = metrics_df['avg_grid_cost'].to_list()
        deg_cost = metrics_df['avg_deg_cost'].to_list()

        # Draw stacked bars
        bars_grid = ax2.bar(x, grid_cost, label='Grid Cost', color='tab:gray')
        bars_deg = ax2.bar(x, deg_cost, bottom=grid_cost, label='Degradation Cost', color='tab:orange')

        # Compute totals and annotate each segment with percent contribution
        totals = [g + d for g, d in zip(grid_cost, deg_cost)]
        # Find a reference height for placing small-label annotations
        max_total = max(totals) if totals else 1.0

        for i in range(len(x)):
            total = totals[i] if totals[i] != 0 else 0.0

            # Grid segment
            g = grid_cost[i]
            if total > 0:
                pct_g = 100.0 * (g / total)
            else:
                pct_g = 0.0
            # Position: middle of the grid segment
            y_g = g / 2.0
            # Choose text color for contrast
            color_g = 'white' if (g / max_total) > 0.12 else 'black'
            if g / max_total >= 0.03:
                ax2.text(x[i], y_g, f"{pct_g:.1f}%", ha='center', va='center', color=color_g, fontsize=8)
            else:
                # place small labels above the bar
                ax2.text(x[i], total + 0.01 * max_total, f"G:{pct_g:.1f}%", ha='center', va='bottom', color='black', fontsize=7)

            # Degradation segment
            d = deg_cost[i]
            if total > 0:
                pct_d = 100.0 * (d / total)
            else:
                pct_d = 0.0
            # Position: middle of the degradation segment
            y_d = g + d / 2.0
            color_d = 'white' if (d / max_total) > 0.12 else 'black'
            if d / max_total >= 0.03:
                ax2.text(x[i], y_d, f"{pct_d:.1f}%", ha='center', va='center', color=color_d, fontsize=8)
            else:
                # place small labels above the bar (if grid already used the space, offset slightly)
                ax2.text(x[i], total + 0.015 * max_total, f"D:{pct_d:.1f}%", ha='center', va='bottom', color='black', fontsize=7)

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Average Cost')
        ax2.set_title('Average Cost Components (percent annotated)')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        figs['cost_stack'] = fig2

        # 3) Risk–return scatter (std vs mean, colour by Sharpe)
        sharpe = metrics_df['sharpe_ratio'].to_list()
        fig3, ax3 = plt.subplots(figsize=figsize)
        sc = ax3.scatter(std_rewards, mean_rewards, c=sharpe, cmap='viridis', s=120, edgecolor='k')
        for i, lbl in enumerate(labels):
            ax3.text(std_rewards[i], mean_rewards[i], lbl, fontsize=8, ha='left', va='bottom')
        ax3.set_xlabel('Std Reward (Risk)')
        ax3.set_ylabel('Mean Reward')
        ax3.set_title('Risk–Return (Colour=Sharpe)')
        cbar = fig3.colorbar(sc, ax=ax3)
        cbar.set_label('Sharpe Ratio')
        ax3.grid(alpha=0.3)
        figs['risk_return'] = fig3

        # 4) Episode return distribution (box plot)
        if episode_rows:
            ep_df = pl.DataFrame(episode_rows)
            # pivot to pandas for boxplot simplicity
            pd_ep = ep_df.to_pandas()
            fig4, ax4 = plt.subplots(figsize=(max(figsize[0], len(labels)*0.6), figsize[1]))
            # group returns by experiment in order of metrics_df
            data_in_order = [pd_ep.loc[pd_ep['experiment']==lab, 'total_reward'].values for lab in labels]
            ax4.boxplot(data_in_order, labels=labels, showfliers=False)
            ax4.set_ylabel('Episode Total Reward')
            ax4.set_title('Episode Return Distribution (Box)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(axis='y', alpha=0.3)
            figs['episode_distribution'] = fig4

        # Save figures if requested
        if save_dir is not None and figs:
            try:
                out_dir = Path(save_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                for name, fig in figs.items():
                    filename = out_dir / f"{name}.{save_format}"
                    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            except Exception as se:
                print(f"Saving figures failed: {se}")

        if make_plots:
            plt.show(block=False)
    except Exception as e:
        print(f"Plot generation failed: {e}")

    return metrics_df

def evaluate_by_conditions(
    logs: list[pl.DataFrame],
    conditions: dict[str, callable]  # e.g., {"high_solar": lambda obs: obs[5] > threshold}
) -> dict:
    """
    Evaluate algorithm performance under different conditions.
    
    conditions example:
        {
            "high_solar": lambda obs: obs[5] > 2.0,
            "peak_price": lambda obs: obs[7] > 0.2,
            "low_battery": lambda obs: obs[-2] < 0.3
        }
    """
    results = {}
    for condition_name, condition_fn in conditions.items():
        filtered_rewards = []
        for ep_df in logs:
            mask = [condition_fn(obs) for obs in ep_df['raw_observation']]
            filtered_rewards.extend(ep_df['reward'][mask].to_list())
        
        results[condition_name] = {
            'mean_reward': np.mean(filtered_rewards) if filtered_rewards else 0.0,
            'count': len(filtered_rewards)
        }
    return results

def compute_decision_divergence(
    logs1: pl.DataFrame,
    logs2: pl.DataFrame,
    action_tolerance: float = 0.01
) -> dict:
    """
    Measure how often two algorithms take different actions in same states.
    """
    actions1 = np.array([a[0] if isinstance(a, list) else a for a in logs1['action']])
    actions2 = np.array([a[0] if isinstance(a, list) else a for a in logs2['action']])
    
    # Action difference metrics
    action_diff = np.abs(actions1 - actions2)
    
    return {
        'mean_absolute_diff': np.mean(action_diff),
        'max_diff': np.max(action_diff),
        'divergence_rate': np.mean(action_diff > action_tolerance),
        'correlation': np.corrcoef(actions1, actions2)[0, 1]
    }

# Helper: compare actions taken by different algorithms
@dataclass
class ActionComparisonConfig:
    """Configuration for action-level comparisons.

    Attributes:
        bins: histogram bin count for action distributions, or the string 'auto'. When 'auto' (or
            a non-positive integer) is used, an automatic selection using the Freedman–Diaconis
            rule is applied with a Sturges fallback.
        normalize_hist: if True produce density (probabilities) instead of raw counts.
        reference: algorithm name to use as reference for median-difference plots and vs-reference stats.
        step_duration: optional step duration (hours) used to label x axes.
        time_periods: optional list of (start, end) step windows to restrict analysis to; when provided
            the comparator will slice each episode into these windows (clamped to episode length) and
            concatenate slices for aggregated metrics. Per-window step profiles are also produced.
        max_episodes: cap number of episodes to use per algorithm (None = use all).
        compute_pairwise: whether to compute pairwise divergence metrics between action distributions.
        subsample_scatter: how many points to subsample for action vs SOC scatter plots.
        save_dir: directory to save generated figures (if provided).
        save_format: image format for saved figures.
        dpi: dpi for saved raster images.
        return_figs: if True, return an ActionComparisonResult (metrics + figure dict), otherwise return metrics dict only.
    """
    bins: Union[int, str] = 'auto'
    normalize_hist: bool = True
    reference: Optional[str] = None
    step_duration: Optional[float] = None
    # If provided, only inspect these (start_step, end_step) windows per episode.
    # Example: [(0, 96), (192, 288)]
    time_periods: Optional[List[Tuple[int, int]]] = None
    max_episodes: Optional[int] = None
    compute_pairwise: bool = True
    subsample_scatter: int = 2000
    save_dir: Optional[str] = None
    save_format: str = "png"
    dpi: int = 160
    return_figs: bool = True


@dataclass
class TemporalAnalysisConfig:
    """Configuration for temporal per-step analysis.

    Attributes:
        step_range: single (start, end) tuple used when time_periods is not provided.
        time_periods: optional list of (start, end) windows to analyze; windows are concatenated in
            the provided order for plotting/stats and the returned stats['step_range'] records the windows.
        annotate_states: whether to add SOC/grid subplots beneath the action plots.
        step_duration: optional step duration (hours) used to label x axes.
        reference: algorithm name used as reference for difference statistics.
        action_tolerance: tolerance threshold to compute divergence rate vs reference.
        save_path: optional file path used to save the produced figure.
    """
    step_range: Optional[tuple[int, int]] = None
    # If provided, analyze these windows (start_step, end_step) per episode and
    # concatenate them in order for plotting and stats.
    time_periods: Optional[List[Tuple[int, int]]] = None
    annotate_states: bool = True
    step_duration: Optional[float] = None
    reference: Optional[str] = None
    action_tolerance: float = 0.01
    save_path: Optional[str] = None


@dataclass
class ActionComparisonResult:
    metrics: dict[str, Any]
    figures: dict[str, plt.Figure] = field(default_factory=dict)


@dataclass
class TemporalAnalysisResult:
    figure: plt.Figure
    stats: dict[str, Any]


class AlgorithmActionComparator:
    """Encapsulates action-level and temporal comparisons across algorithms."""

    def __init__(
        self,
        action_logs: Optional[dict[str, list[pl.DataFrame]]] = None,
        random_seed: int = 1234,
    ) -> None:
        self.random_seed = random_seed
        self.action_logs = self._validate_action_logs(action_logs) if action_logs else None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_action_logs(
        logs_dict: Optional[dict[str, list[pl.DataFrame]]]
    ) -> dict[str, list[pl.DataFrame]]:
        if not logs_dict or not isinstance(logs_dict, dict):
            raise ValueError("logs_dict must be a non-empty dict[str, list[pl.DataFrame]].")
        for name, logs in logs_dict.items():
            if not isinstance(logs, list) or not logs:
                raise ValueError(f"Algorithm '{name}' must map to a non-empty list of Polars DataFrames.")
            for df in logs:
                if not isinstance(df, pl.DataFrame):
                    raise ValueError(f"Entries for algorithm '{name}' must be Polars DataFrames.")
        return logs_dict

    @staticmethod
    def _extract_action(val: Any) -> float:
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[0]) if len(val) > 0 else 0.0
        try:
            return float(val)
        except Exception:
            return 0.0

    @staticmethod
    def _safe_float(val: Any) -> float:
        try:
            return float(val)
        except Exception:
            return float("nan")

    @staticmethod
    def _parse_info(val: Any) -> dict[str, Any]:
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        x_f = x[mask]
        y_f = y[mask]
        if x_f.size < 3 or np.allclose(x_f.std(), 0.0) or np.allclose(y_f.std(), 0.0):
            return float("nan")
        return float(np.corrcoef(x_f, y_f)[0, 1])

    def _auto_bins(self, data: np.ndarray) -> int:
        """Estimate a sensible bin count for histogramming 'data'.

        Uses Freedman–Diaconis rule (bin width = 2*IQR / n^(1/3)). If the IQR is 0 or
        yields non-positive width, falls back to Sturges' rule (ceil(log2(n)+1)).
        Bounds the result to a reasonable range to avoid extreme bin counts.
        """
        finite = data[np.isfinite(data)]
        n = finite.size
        if n <= 1:
            return 10
        try:
            q75, q25 = np.percentile(finite, [75, 25])
            iqr = float(q75 - q25)
        except Exception:
            iqr = 0.0

        if iqr > 0:
            # Freedman–Diaconis bin width
            bw = 2.0 * iqr / (n ** (1.0 / 3.0))
            if bw > 0:
                data_range = float(np.nanmax(finite) - np.nanmin(finite))
                if data_range <= 0:
                    bins = int(np.ceil(np.log2(n) + 1))
                else:
                    bins = int(max(1, np.ceil(data_range / bw)))
            else:
                bins = int(np.ceil(np.log2(n) + 1))
        else:
            # Sturges' rule fallback
            bins = int(np.ceil(np.log2(n) + 1))

        # bound bins to avoid too small/large choices
        bins = max(5, min(bins, 500))
        return int(bins)

    # ------------------------------------------------------------------
    # Action-level comparison (Section A)
    # ------------------------------------------------------------------
    def compare(
        self,
        logs_dict: Optional[dict[str, list[pl.DataFrame]]] = None,
        config: Optional[ActionComparisonConfig] = None,
    ) -> ActionComparisonResult | dict:
        cfg = config or ActionComparisonConfig()
        data = self._validate_action_logs(logs_dict or self.action_logs)
        algo_names = list(data.keys())

        reference = cfg.reference or algo_names[0]
        if reference not in data:
            raise ValueError(f"Reference '{reference}' not found among algorithms: {algo_names}")

        per_algo_actions: dict[str, np.ndarray] = {}
        per_algo_soc: dict[str, np.ndarray] = {}
        per_algo_solar: dict[str, np.ndarray] = {}
        per_algo_load: dict[str, np.ndarray] = {}
        episode_lengths: dict[str, list[int]] = {}

        for algo, episode_list in data.items():
            max_eps = cfg.max_episodes if cfg.max_episodes is not None else len(episode_list)
            use_eps = episode_list[: max_eps]
            if not use_eps:
                raise ValueError(f"Algorithm '{algo}' has an empty episode subset after max_episodes filter.")

            all_actions: list[float] = []
            soc_vals: list[float] = []
            solar_vals: list[float] = []
            load_vals: list[float] = []
            lengths: list[int] = []

            # If time_periods provided, we will slice each episode into those windows
            # and concatenate slices. Also track per-window lengths for step profile.
            per_window_lengths: dict[int, list[int]] = {}
            if cfg.time_periods:
                for w_idx, (ws, we) in enumerate(cfg.time_periods):
                    per_window_lengths[w_idx] = []

            for ep_df in use_eps:
                full_len = ep_df.height
                lengths.append(full_len)

                if cfg.time_periods:
                    # For each window, slice safely
                    for w_idx, (ws, we) in enumerate(cfg.time_periods):
                        s = max(0, int(ws))
                        e = min(int(we), full_len)
                        if e <= s:
                            # empty window for this episode
                            per_window_lengths[w_idx].append(0)
                            continue
                        per_window_lengths[w_idx].append(e - s)
                        ep_actions = [self._extract_action(a) for a in ep_df['action'].to_list()[s:e]]
                        all_actions.extend(ep_actions)

                        if 'raw_observation' in ep_df.columns:
                            ro_list = ep_df['raw_observation'].to_list()[s:e]
                            for obs in ro_list:
                                if isinstance(obs, (list, tuple, np.ndarray)):
                                    soc_vals.append(self._safe_float(obs[-2]) if len(obs) >= 2 else float('nan'))
                                    solar_vals.append(self._safe_float(obs[5]) if len(obs) > 5 else float('nan'))
                                    load_vals.append(self._safe_float(obs[6]) if len(obs) > 6 else float('nan'))
                                else:
                                    soc_vals.append(float('nan'))
                                    solar_vals.append(float('nan'))
                                    load_vals.append(float('nan'))
                        else:
                            soc_vals.extend([float('nan')] * (e - s))
                            solar_vals.extend([float('nan')] * (e - s))
                            load_vals.extend([float('nan')] * (e - s))
                else:
                    # Whole-episode behavior
                    ep_actions = [self._extract_action(a) for a in ep_df["action"].to_list()]
                    all_actions.extend(ep_actions)

                    if "raw_observation" in ep_df.columns:
                        ro_list = ep_df["raw_observation"].to_list()
                        for obs in ro_list:
                            if isinstance(obs, (list, tuple, np.ndarray)):
                                soc_vals.append(self._safe_float(obs[-2]) if len(obs) >= 2 else float("nan"))
                                solar_vals.append(self._safe_float(obs[5]) if len(obs) > 5 else float("nan"))
                                load_vals.append(self._safe_float(obs[6]) if len(obs) > 6 else float("nan"))
                            else:
                                soc_vals.append(float("nan"))
                                solar_vals.append(float("nan"))
                                load_vals.append(float("nan"))
                    else:
                        soc_vals.extend([float("nan")] * ep_df.height)
                        solar_vals.extend([float("nan")] * ep_df.height)
                        load_vals.extend([float("nan")] * ep_df.height)

            per_algo_actions[algo] = np.array(all_actions, dtype=float)
            per_algo_soc[algo] = np.array(soc_vals, dtype=float)
            per_algo_solar[algo] = np.array(solar_vals, dtype=float)
            per_algo_load[algo] = np.array(load_vals, dtype=float)
            episode_lengths[algo] = lengths

        step_profile = {"windows": None, "median": {}, "iqr_low": {}, "iqr_high": {}}
        if cfg.time_periods:
            step_profile["windows"] = cfg.time_periods
            # compute per-window step profiles per algorithm
            for algo, lengths in episode_lengths.items():
                # lengths is full episode lengths list; we use per_window_lengths computed earlier
                # fetch per-window lengths map from earlier local variable? We'll recompute per-window arrays
                step_profile["median"][algo] = {}
                step_profile["iqr_low"][algo] = {}
                step_profile["iqr_high"][algo] = {}
                # iterate windows
                for w_idx, (ws, we) in enumerate(cfg.time_periods):
                    truncated_actions = []
                    for ep_df in data[algo][: len(episode_lengths[algo])]:
                        full_len = ep_df.height
                        s = max(0, int(ws))
                        e = min(int(we), full_len)
                        if e <= s:
                            continue
                        ep_actions = [self._extract_action(a) for a in ep_df["action"].to_list()[s:e]]
                        truncated_actions.append(ep_actions)
                    if not truncated_actions:
                        arr = np.empty((0, 0), dtype=float)
                    else:
                        # pad shorter episodes with nan to form rectangular array
                        maxw = max(len(a) for a in truncated_actions)
                        arr = np.array([np.pad(np.array(a, dtype=float), (0, maxw - len(a)), constant_values=np.nan) for a in truncated_actions], dtype=float)
                    if arr.size == 0:
                        step_profile["median"][algo][w_idx] = []
                        step_profile["iqr_low"][algo][w_idx] = []
                        step_profile["iqr_high"][algo][w_idx] = []
                    else:
                        step_profile["median"][algo][w_idx] = np.nanmedian(arr, axis=0).tolist()
                        step_profile["iqr_low"][algo][w_idx] = np.nanpercentile(arr, 25, axis=0).tolist()
                        step_profile["iqr_high"][algo][w_idx] = np.nanpercentile(arr, 75, axis=0).tolist()
        else:
            for algo, lengths in episode_lengths.items():
                min_len = min(lengths)
                truncated_actions = []
                for ep_df in data[algo][: len(lengths)]:
                    ep_actions = [self._extract_action(a) for a in ep_df["action"].to_list()]
                    truncated_actions.append(ep_actions[:min_len])
                arr = np.array(truncated_actions, dtype=float)
                step_profile["median"][algo] = np.nanmedian(arr, axis=0).tolist()
                step_profile["iqr_low"][algo] = np.nanpercentile(arr, 25, axis=0).tolist()
                step_profile["iqr_high"][algo] = np.nanpercentile(arr, 75, axis=0).tolist()
                if step_profile["windows"] is None:
                    step_profile["windows"] = [ (0, min_len) ]

        all_actions_concat = np.concatenate(list(per_algo_actions.values()))
        finite_actions = all_actions_concat[np.isfinite(all_actions_concat)]
        if finite_actions.size == 0:
            raise ValueError("No finite action values found.")
        global_min, global_max = finite_actions.min(), finite_actions.max()
        span = global_max - global_min if global_max > global_min else 1.0
        global_min -= 0.01 * span
        global_max += 0.01 * span
        # Determine bin count: allow cfg.bins to be an int>0 or 'auto' (or non-positive int)
        if isinstance(cfg.bins, int) and cfg.bins > 0:
            bins_count = int(cfg.bins)
        else:
            # auto-select based on data
            bins_count = self._auto_bins(finite_actions)
        bin_edges = np.linspace(global_min, global_max, bins_count + 1)

        def histogram(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            finite = arr[np.isfinite(arr)]
            counts, _ = np.histogram(finite, bins=bin_edges)
            if cfg.normalize_hist:
                total = counts.sum()
                probs = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
                return probs, counts
            return counts, counts

        metrics: dict[str, Any] = {
            "per_algorithm": {},
            "pairwise": {},
            "step_profile": step_profile,
            "config": {
                "bins_requested": cfg.bins,
                "bins_used": bins_count,
                "normalize_hist": cfg.normalize_hist,
                "reference": reference,
                "max_episodes": cfg.max_episodes,
                "compute_pairwise": cfg.compute_pairwise,
            },
        }

        for algo, arr in per_algo_actions.items():
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                finite = np.array([float("nan")])
            probs_or_counts, raw_counts = histogram(arr)
            # Per-window histograms (if requested): compute per-window concatenated arrays
            per_window_hist: list[dict[str, Any]] = []
            if cfg.time_periods:
                per_window_lists: list[np.ndarray] = []
                for w_idx, (ws, we) in enumerate(cfg.time_periods):
                    # collect actions across episodes for this algo/window
                    collected: list[float] = []
                    max_eps = cfg.max_episodes if cfg.max_episodes is not None else len(data[algo])
                    for ep_df in data[algo][:max_eps]:
                        full_len = ep_df.height
                        s = max(0, int(ws))
                        e = min(int(we), full_len)
                        if e <= s:
                            continue
                        collected.extend([self._extract_action(a) for a in ep_df['action'].to_list()[s:e]])
                    arr_win = np.array(collected, dtype=float) if collected else np.array([], dtype=float)
                    per_window_lists.append(arr_win)
                    # compute histogram for this window using same global bin edges
                    probs_win, counts_win = histogram(arr_win) if arr_win.size > 0 else (np.zeros(cfg.bins, dtype=float), np.zeros(cfg.bins, dtype=int))
                    per_window_hist.append({
                        'window': (int(ws), int(we)),
                        'hist_values': probs_win.tolist(),
                        'raw_counts': counts_win.tolist(),
                    })
            metrics["per_algorithm"][algo] = {
                "total_steps": int(arr.size),
                "total_episodes": int(len(episode_lengths[algo])),
                "mean_action": float(np.nanmean(finite)),
                "std_action": float(np.nanstd(finite)),
                "min_action": float(np.nanmin(finite)),
                "max_action": float(np.nanmax(finite)),
                "median_action": float(np.nanmedian(finite)),
                "pct_5_action": float(np.nanpercentile(finite, 5)),
                "pct_95_action": float(np.nanpercentile(finite, 95)),
                "fraction_positive": float(np.mean(finite > 0)) if finite.size else float("nan"),
                "hist_bin_edges": bin_edges.tolist(),
                "hist_values": probs_or_counts.tolist(),
                "raw_counts": raw_counts.tolist(),
                "per_window": per_window_hist,
                "corr_action_soc": self._safe_corr(arr, per_algo_soc[algo]),
                "corr_action_solar": self._safe_corr(arr, per_algo_solar[algo]),
                "corr_action_load": self._safe_corr(arr, per_algo_load[algo]),
            }

        if cfg.compute_pairwise:
            rng = np.random.default_rng(seed=self.random_seed)

            def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
                p = np.asarray(p, dtype=float)
                q = np.asarray(q, dtype=float)
                if p.sum() == 0 and q.sum() == 0:
                    return 0.0
                if not np.isclose(p.sum(), 1.0):
                    p = p / (p.sum() + 1e-12)
                if not np.isclose(q.sum(), 1.0):
                    q = q / (q.sum() + 1e-12)
                m = 0.5 * (p + q)
                mask_p = (p > 0) & (m > 0)
                mask_q = (q > 0) & (m > 0)
                kl_pm = float(np.sum(p[mask_p] * np.log(p[mask_p] / m[mask_p])))
                kl_qm = float(np.sum(q[mask_q] * np.log(q[mask_q] / m[mask_q])))
                return 0.5 * (kl_pm + kl_qm)

            def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
                p = np.asarray(p, dtype=float)
                q = np.asarray(q, dtype=float)
                if p.sum() == 0 or q.sum() == 0:
                    return float("nan")
                if not np.isclose(p.sum(), 1.0):
                    p = p / (p.sum() + 1e-12)
                if not np.isclose(q.sum(), 1.0):
                    q = q / (q.sum() + 1e-12)
                mask = (p > 0) & (q > 0)
                return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

            for i, a_i in enumerate(algo_names):
                arr_i = per_algo_actions[a_i]
                hist_i = np.array(metrics["per_algorithm"][a_i]["hist_values"], dtype=float)
                for j, a_j in enumerate(algo_names):
                    if j <= i:
                        continue
                    arr_j = per_algo_actions[a_j]
                    hist_j = np.array(metrics["per_algorithm"][a_j]["hist_values"], dtype=float)
                    finite_i = arr_i[np.isfinite(arr_i)]
                    finite_j = arr_j[np.isfinite(arr_j)]
                    sample_size = min(5000, finite_i.size, finite_j.size)
                    if sample_size > 0:
                        idx_i = rng.choice(finite_i.size, sample_size, replace=False)
                        idx_j = rng.choice(finite_j.size, sample_size, replace=False)
                        corr_val = (
                            float(np.corrcoef(finite_i[idx_i], finite_j[idx_j])[0, 1])
                            if sample_size > 2
                            else float("nan")
                        )
                        diff = finite_i[idx_i] - finite_j[idx_j]
                        mae_val = float(np.nanmean(np.abs(diff)))
                        rmse_val = float(np.sqrt(np.nanmean(diff**2)))
                    else:
                        corr_val = float("nan")
                        mae_val = float("nan")
                        rmse_val = float("nan")

                    js_val = js_divergence(hist_i, hist_j)
                    kl_ij_val = kl_divergence(hist_i, hist_j)
                    kl_ji_val = kl_divergence(hist_j, hist_i)
                    if _HAS_SCIPY and finite_i.size > 0 and finite_j.size > 0:
                        wdist = float(wasserstein_distance(finite_i, finite_j))  # type: ignore[arg-type]
                    else:
                        wdist = float("nan")

                    metrics["pairwise"][(a_i, a_j)] = {
                        "corr": corr_val,
                        "mae": mae_val,
                        "rmse": rmse_val,
                        "js_divergence": js_val,
                        "kl_i_j": kl_ij_val,
                        "kl_j_i": kl_ji_val,
                        "wasserstein": wdist,
                    }

        figs: dict[str, plt.Figure] = {}
        try:
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
            for algo in algo_names:
                vals = np.array(metrics["per_algorithm"][algo]["hist_values"])
                ax_hist.plot(centers, vals, label=algo, linewidth=1.6)
            ax_hist.set_title("Action Distribution (Histogram / Density)")
            ax_hist.set_xlabel("Action")
            ax_hist.set_ylabel("Probability" if cfg.normalize_hist else "Count")
            ax_hist.grid(alpha=0.3)
            ax_hist.legend()
            figs["action_hist"] = fig_hist

            fig_cdf, ax_cdf = plt.subplots(figsize=(7, 4))
            for algo in algo_names:
                actions_sorted = np.sort(per_algo_actions[algo][np.isfinite(per_algo_actions[algo])])
                if actions_sorted.size == 0:
                    continue
                cdf = np.linspace(0, 1, actions_sorted.size)
                ax_cdf.plot(actions_sorted, cdf, label=algo, linewidth=1.5)
            ax_cdf.set_title("Action CDF")
            ax_cdf.set_xlabel("Action")
            ax_cdf.set_ylabel("CDF")
            ax_cdf.grid(alpha=0.3)
            ax_cdf.legend()
            figs["action_cdf"] = fig_cdf

            # --- Per-window histograms and per-window step profiles ---
            if cfg.time_periods:
                for w_idx, (ws, we) in enumerate(cfg.time_periods):
                    # Per-window histogram across algorithms (same bin edges)
                    fig_w_hist, ax_w_hist = plt.subplots(figsize=(7, 4))
                    for algo in algo_names:
                        per_win = metrics["per_algorithm"][algo].get("per_window", [])
                        if w_idx < len(per_win):
                            vals = np.array(per_win[w_idx]["hist_values"], dtype=float)
                            ax_w_hist.plot(centers, vals, label=algo, linewidth=1.4)
                    ax_w_hist.set_title(f"Action Distribution (Window {w_idx}: {ws}-{we})")
                    ax_w_hist.set_xlabel("Action")
                    ax_w_hist.set_ylabel("Probability" if cfg.normalize_hist else "Count")
                    ax_w_hist.grid(alpha=0.3)
                    ax_w_hist.legend()
                    figs[f"action_hist_window_{w_idx}"] = fig_w_hist

                    # Per-window step profile (median + IQR) as separate figure
                    fig_w_profile, ax_w_profile = plt.subplots(figsize=(8, 3))
                    for algo in algo_names:
                        med = step_profile["median"][algo].get(w_idx, []) if cfg.time_periods else step_profile["median"][algo]
                        low = step_profile["iqr_low"][algo].get(w_idx, []) if cfg.time_periods else step_profile["iqr_low"][algo]
                        high = step_profile["iqr_high"][algo].get(w_idx, []) if cfg.time_periods else step_profile["iqr_high"][algo]
                        if not med:
                            continue
                        steps_local = list(range(len(med)))
                        ax_w_profile.plot(steps_local, med, label=f"{algo} median", linewidth=1.3)
                        if low and high:
                            ax_w_profile.fill_between(steps_local, low, high, alpha=0.12)
                    ax_w_profile.set_title(f"Per-Step Median Action (Window {w_idx}: {ws}-{we})")
                    xlabel_w = "Step"
                    if cfg.step_duration:
                        xlabel_w += " (hours)"
                    ax_w_profile.set_xlabel(xlabel_w)
                    ax_w_profile.set_ylabel("Action")
                    ax_w_profile.grid(alpha=0.3)
                    ax_w_profile.legend(ncol=min(len(algo_names), 3))
                    figs[f"step_profile_window_{w_idx}"] = fig_w_profile

            fig_profile, ax_profile = plt.subplots(figsize=(8, 4))
            # For multi-window support, concatenate per-window medians for plotting
            for algo in algo_names:
                if cfg.time_periods:
                    concat_med = []
                    concat_low = []
                    concat_high = []
                    for w_idx in range(len(cfg.time_periods)):
                        med = step_profile["median"][algo].get(w_idx, [])
                        low = step_profile["iqr_low"][algo].get(w_idx, [])
                        high = step_profile["iqr_high"][algo].get(w_idx, [])
                        concat_med.extend(med)
                        concat_low.extend(low)
                        concat_high.extend(high)
                    steps_local = list(range(len(concat_med)))
                    ax_profile.plot(steps_local, concat_med, label=f"{algo} median", linewidth=1.5)
                    if concat_low and concat_high:
                        ax_profile.fill_between(steps_local, concat_low, concat_high, alpha=0.15)
                else:
                    steps_local = list(range(len(step_profile["median"][algo])))
                    med = step_profile["median"][algo]
                    low = step_profile["iqr_low"][algo]
                    high = step_profile["iqr_high"][algo]
                    ax_profile.plot(steps_local, med, label=f"{algo} median", linewidth=1.5)
                    ax_profile.fill_between(steps_local, low, high, alpha=0.15)
            ax_profile.set_title("Per-Step Action Median (Shaded IQR)")
            xlabel = "Step"
            if cfg.step_duration:
                xlabel += " (hours)"
            ax_profile.set_xlabel(xlabel)
            ax_profile.set_ylabel("Action")
            ax_profile.grid(alpha=0.3)
            ax_profile.legend(ncol=min(len(algo_names), 3))
            figs["step_profile"] = fig_profile

            fig_diff, ax_diff = plt.subplots(figsize=(8, 3))
            # Build concatenated reference median
            if cfg.time_periods:
                ref_concat = []
                for w_idx in range(len(cfg.time_periods)):
                    ref_concat.extend(step_profile["median"][reference].get(w_idx, []))
                ref_med = np.array(ref_concat, dtype=float)
            else:
                ref_med = np.array(step_profile["median"][reference], dtype=float)
            steps = np.arange(len(ref_med))
            for algo in algo_names:
                if algo == reference:
                    continue
                if cfg.time_periods:
                    algo_concat = []
                    for w_idx in range(len(cfg.time_periods)):
                        algo_concat.extend(step_profile["median"][algo].get(w_idx, []))
                    med = np.array(algo_concat, dtype=float)
                else:
                    med = np.array(step_profile["median"][algo], dtype=float)
                # align lengths if necessary
                if med.size != ref_med.size:
                    # pad shorter with nan
                    maxlen = max(med.size, ref_med.size)
                    med = np.pad(med, (0, maxlen - med.size), constant_values=np.nan)
                    ref = np.pad(ref_med, (0, maxlen - ref_med.size), constant_values=np.nan)
                    ax_diff.plot(np.arange(len(ref)), med - ref, label=f"{algo} - {reference}", linewidth=1.4)
                else:
                    ax_diff.plot(steps, med - ref_med, label=f"{algo} - {reference}", linewidth=1.4)
            ax_diff.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
            ax_diff.set_title("Median Action Difference vs Reference")
            ax_diff.set_xlabel(xlabel)
            ax_diff.set_ylabel("Δ Action")
            ax_diff.grid(alpha=0.3)
            ax_diff.legend(ncol=min(len(algo_names) - 1, 3))
            figs["median_diff_vs_reference"] = fig_diff

            fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
            for algo in algo_names:
                actions = per_algo_actions[algo]
                socs = per_algo_soc[algo]
                mask = np.isfinite(actions) & np.isfinite(socs)
                actions = actions[mask]
                socs = socs[mask]
                if actions.size == 0:
                    continue
                if cfg.subsample_scatter and actions.size > cfg.subsample_scatter:
                    idx = np.random.default_rng(seed=42).choice(actions.size, cfg.subsample_scatter, replace=False)
                    actions = actions[idx]
                    socs = socs[idx]
                ax_scatter.scatter(socs, actions, s=10, alpha=0.4, label=algo)
            ax_scatter.set_title("Action vs Battery SOC (subsampled)")
            ax_scatter.set_xlabel("SOC")
            ax_scatter.set_ylabel("Action")
            ax_scatter.grid(alpha=0.3)
            ax_scatter.legend()
            figs["action_vs_soc"] = fig_scatter

            if cfg.compute_pairwise and metrics["pairwise"]:
                fig_heat, ax_heat = plt.subplots(figsize=(6, 5))
                n = len(algo_names)
                mat = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        if i < j:
                            val = metrics["pairwise"].get((algo_names[i], algo_names[j]), {}).get("js_divergence", float("nan"))
                            mat[i, j] = val
                            mat[j, i] = val
                im = ax_heat.imshow(mat, cmap="viridis")
                ax_heat.set_xticks(range(n))
                ax_heat.set_yticks(range(n))
                ax_heat.set_xticklabels(algo_names, rotation=45, ha="right")
                ax_heat.set_yticklabels(algo_names)
                ax_heat.set_title("Pairwise JS Divergence (Action Distributions)")
                mean_val = np.nanmean(mat)
                for i in range(n):
                    for j in range(n):
                        text_color = "white" if not np.isnan(mat[i, j]) and mat[i, j] > mean_val else "black"
                        ax_heat.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=8)
                fig_heat.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
                figs["pairwise_js_heatmap"] = fig_heat

            if cfg.save_dir and figs:
                out_dir = Path(cfg.save_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                for name, fig in figs.items():
                    fig.savefig(out_dir / f"{name}.{cfg.save_format}", dpi=cfg.dpi, bbox_inches="tight")
        except Exception as exc:
            print(f"Figure generation failed: {exc}")

        result = ActionComparisonResult(metrics=metrics, figures=figs)
        if cfg.return_figs:
            return result
        return metrics

    # ------------------------------------------------------------------
    # Temporal analysis (Section B)
    # ------------------------------------------------------------------
    def analyze_temporal(
        self,
        logs_dict: dict[str, pl.DataFrame],
        config: Optional[TemporalAnalysisConfig] = None,
    ) -> TemporalAnalysisResult:
        if not logs_dict or not isinstance(logs_dict, dict):
            raise ValueError("logs_dict must be a non-empty dict[str, pl.DataFrame].")
        for name, df in logs_dict.items():
            if not isinstance(df, pl.DataFrame):
                raise ValueError(f"Entry for algorithm '{name}' must be a Polars DataFrame.")

        cfg = config or TemporalAnalysisConfig()
        algo_names = list(logs_dict.keys())
        reference = cfg.reference or algo_names[0]
        if reference not in logs_dict:
            raise ValueError(f"Reference '{reference}' not found in logs_dict keys {algo_names}.")

        lengths = {k: v.height for k, v in logs_dict.items()}
        if any(l == 0 for l in lengths.values()):
            raise ValueError("One or more input logs are empty.")

        actions: dict[str, np.ndarray] = {}
        socs: dict[str, Optional[np.ndarray]] = {}
        grids: dict[str, Optional[np.ndarray]] = {}

        if cfg.time_periods:
            # Concatenate the requested windows (in order) for each algorithm
            total_len = 0
            window_ranges = []
            for (ws, we) in cfg.time_periods:
                if ws < 0 or we <= ws:
                    raise ValueError(f"Invalid time_period window {(ws, we)}")
                window_ranges.append((int(ws), int(we)))
            # Build arrays per algorithm
            for name, df in logs_dict.items():
                parts_actions = []
                parts_socs = []
                parts_grids = []
                for (s, e) in window_ranges:
                    s_loc = max(0, s)
                    e_loc = min(e, df.height)
                    if e_loc <= s_loc:
                        continue
                    df_slice = df.slice(s_loc, e_loc - s_loc)
                    parts_actions.extend([self._extract_action(a) for a in df_slice["action"].to_list()])
                    if "raw_observation" in df_slice.columns:
                        ro = df_slice["raw_observation"].to_list()
                        parts_socs.extend([obs[-2] if isinstance(obs, (list, np.ndarray)) and len(obs) >= 2 else float("nan") for obs in ro])
                    else:
                        parts_socs.extend([float("nan")] * (e_loc - s_loc))

                    if "info" in df_slice.columns:
                        info_col = df_slice["info"].to_list()
                        parsed = [self._parse_info(x) for x in info_col]
                        for info in parsed:
                            if isinstance(info, dict) and "grid_energy" in info:
                                try:
                                    parts_grids.append(float(info["grid_energy"]))
                                except Exception:
                                    parts_grids.append(float("nan"))
                            else:
                                parts_grids.append(float("nan"))
                    else:
                        parts_grids.extend([float("nan")] * (e_loc - s_loc))

                actions[name] = np.array(parts_actions, dtype=float)
                socs[name] = np.array(parts_socs, dtype=float) if parts_socs else None
                grids[name] = np.array(parts_grids, dtype=float) if parts_grids else None

            # steps become a simple range for concatenated windows
            total_len = max((arr.size for arr in actions.values()), default=0)
            steps = np.arange(0, total_len)
        else:
            if cfg.step_range is None:
                min_len = min(lengths.values())
                start, end = 0, min_len
            else:
                start, end = cfg.step_range
                if start < 0 or end <= start:
                    raise ValueError("Invalid step_range. Expect (start, end) with end>start.")
                end = min(end, min(lengths.values()))
                start = max(0, start)
                if end - start <= 0:
                    raise ValueError("Provided step_range has no overlap across logs.")

            steps = np.arange(start, end)

            for name, df in logs_dict.items():
                df_slice = df.slice(start, end - start)
                actions[name] = np.array([self._extract_action(a) for a in df_slice["action"].to_list()])

                if "raw_observation" in df_slice.columns:
                    socs_raw = df_slice["raw_observation"].to_list()
                    socs[name] = np.array([
                        obs[-2] if isinstance(obs, (list, np.ndarray)) and len(obs) >= 2 else float("nan")
                        for obs in socs_raw
                    ])
                else:
                    socs[name] = None

                if "info" in df_slice.columns:
                    info_col = df_slice["info"].to_list()
                    parsed = [self._parse_info(x) for x in info_col]
                    grid_vals = []
                    for info in parsed:
                        if isinstance(info, dict) and "grid_energy" in info:
                            try:
                                grid_vals.append(float(info["grid_energy"]))
                            except Exception:
                                grid_vals.append(float("nan"))
                        else:
                            grid_vals.append(float("nan"))
                    grids[name] = np.array(grid_vals)
                else:
                    grids[name] = None

        n_rows = 2
        if cfg.annotate_states:
            n_rows += 1
            if any(v is not None for v in grids.values()):
                n_rows += 1

        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.0 + 2.5 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        axis_idx = 0
        ax_act = axes[axis_idx]
        axis_idx += 1
        for name in algo_names:
            ax_act.plot(steps, actions[name], label=name, linewidth=1.6)
        ax_act.set_title("Action comparison across algorithms")
        ax_act.set_ylabel("Action")
        ax_act.grid(alpha=0.3)
        ax_act.legend(loc="best", ncol=min(len(algo_names), 3))

        ax_diff = axes[axis_idx]
        axis_idx += 1
        ref_actions = actions[reference]
        for name in algo_names:
            if name == reference:
                continue
            diff = actions[name] - ref_actions
            ax_diff.plot(steps, diff, label=f"{name} - {reference}", linewidth=1.4)
        ax_diff.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        ax_diff.axhline(cfg.action_tolerance, color="r", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_diff.axhline(-cfg.action_tolerance, color="r", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_diff.set_title(f"Action differences vs reference '{reference}'")
        ax_diff.set_ylabel("Δ Action")
        ax_diff.grid(alpha=0.3)
        ax_diff.legend(loc="best", ncol=min(len(algo_names) - 1, 3))

        if cfg.annotate_states:
            ax_soc = axes[axis_idx]
            axis_idx += 1
            any_soc = False
            for name in algo_names:
                if socs[name] is not None:
                    ax_soc.plot(steps, socs[name], label=name, linewidth=1.4)
                    any_soc = True
            ax_soc.set_title("Battery State of Charge")
            ax_soc.set_ylabel("SOC")
            ax_soc.grid(alpha=0.3)
            if any_soc:
                ax_soc.legend(loc="best", ncol=min(len(algo_names), 3))

            if any(v is not None for v in grids.values()):
                ax_grid = axes[axis_idx]
                axis_idx += 1
                for name in algo_names:
                    if grids[name] is not None:
                        ax_grid.plot(steps, grids[name], label=name, linewidth=1.2)
                ax_grid.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
                ax_grid.set_title("Grid energy (+ import, - export)")
                ax_grid.set_ylabel("kWh per step")
                ax_grid.grid(alpha=0.3)
                ax_grid.legend(loc="best", ncol=min(len(algo_names), 3))

        if cfg.step_duration and cfg.step_duration > 0:
            hours = (steps - steps[0]) * cfg.step_duration
            axes[-1].set_xlabel(f"Time (hours from t0, step={cfg.step_duration}h)")
            tick_step = max(1, len(steps) // 10)
            axes[-1].set_xticks(steps[::tick_step])
            axes[-1].set_xticklabels([f"{h:.1f}" for h in hours[::tick_step]])
        else:
            axes[-1].set_xlabel("Step")

        plt.tight_layout()

        stats: dict[str, Any] = {
            "per_algorithm": {},
            "vs_reference": {},
            "step_range": cfg.time_periods if cfg.time_periods is not None else (int(steps[0]), int(steps[-1]) + 1),
            "reference": reference,
            "action_tolerance": float(cfg.action_tolerance),
        }

        for name in algo_names:
            arr = actions[name]
            stats["per_algorithm"][name] = {
                "mean_action": float(np.nanmean(arr)),
                "std_action": float(np.nanstd(arr)),
                "min_action": float(np.nanmin(arr)),
                "max_action": float(np.nanmax(arr)),
            }

        ref_arr = ref_actions
        for name in algo_names:
            if name == reference:
                continue
            arr = actions[name]
            diff = arr - ref_arr
            with np.errstate(invalid="ignore"):
                mae = float(np.nanmean(np.abs(diff)))
                rmse = float(np.sqrt(np.nanmean(diff**2)))
                divergence = float(np.nanmean(np.abs(diff) > cfg.action_tolerance))
                try:
                    corr_val = float(np.corrcoef(arr, ref_arr)[0, 1])
                except Exception:
                    corr_val = float("nan")
            stats["vs_reference"][name] = {
                "mean_abs_diff": mae,
                "rmse": rmse,
                "divergence_rate": divergence,
                "correlation": corr_val,
            }

        if cfg.save_path is not None:
            try:
                Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(cfg.save_path, dpi=200, bbox_inches="tight")
            except Exception as exc:
                print(f"Failed to save analyze_temporal_actions figure: {exc}")

        return TemporalAnalysisResult(figure=fig, stats=stats)


def compare_actions_across_algorithms(
    logs_dict: dict[str, list[pl.DataFrame]],
    bins: int = 50,
    normalize_hist: bool = True,
    reference: Optional[str] = None,
    step_duration: Optional[float] = None,
    max_episodes: Optional[int] = None,
    time_periods: Optional[List[Tuple[int, int]]] = None,
    compute_pairwise: bool = True,
    subsample_scatter: int = 2000,
    save_dir: Optional[str] = None,
    save_format: str = "png",
    dpi: int = 160,
    return_figs: bool = True,
) -> tuple[dict, dict] | dict:
    """Backwards-compatible wrapper around AlgorithmActionComparator.compare.

    Parameters
    ----------
    logs_dict: dict[str, list[pl.DataFrame]]
        Mapping from algorithm name to a list of episode DataFrames (Polars). Each DataFrame
        should contain columns 'action', optionally 'raw_observation' and 'info'.
    bins: int
        Number of histogram bins for action distributions.
    normalize_hist: bool
        If True, histogram values are normalized to form a probability density.
    reference: Optional[str]
        Algorithm name to use as reference for median-difference plots. If None, the first
        algorithm in logs_dict is used.
    step_duration: Optional[float]
        Optional step duration (hours) used for labeling time axes.
    time_periods: Optional[List[Tuple[int,int]]]
        If provided, only these (start, end) step windows (per episode) are used. Windows are
        clamped to episode lengths and concatenated for aggregated metrics. Per-window histograms
        and per-window step-profile plots are also produced.
    max_episodes: Optional[int]
        Limit number of episodes per algorithm to use (None = all).
    compute_pairwise: bool
        Whether to compute pairwise divergence metrics between algorithms.
    subsample_scatter: int
        Number of points to subsample when making scatter plots (action vs SOC).
    save_dir: Optional[str]
        Directory to save generated figures (if provided).
    save_format: str
        Image format for saved figures (e.g., 'png').
    dpi: int
        Dots-per-inch for saved raster images.
    return_figs: bool
        If True, the function returns both metrics and a dict of Matplotlib figures;
        otherwise returns metrics only.
    """

    comparator = AlgorithmActionComparator(action_logs=logs_dict)
    cfg = ActionComparisonConfig(
        bins=bins,
        normalize_hist=normalize_hist,
        reference=reference,
        step_duration=step_duration,
        time_periods=time_periods,
        max_episodes=max_episodes,
        compute_pairwise=compute_pairwise,
        subsample_scatter=subsample_scatter,
        save_dir=save_dir,
        save_format=save_format,
        dpi=dpi,
        return_figs=return_figs,
    )

    result = comparator.compare(config=cfg)
    if return_figs:
        assert isinstance(result, ActionComparisonResult)
        return result.metrics, result.figures
    if not isinstance(result, dict):
        raise TypeError("Expected metrics dictionary when return_figs=False.")
    return result


def analyze_temporal_actions(
    logs_dict: dict[str, pl.DataFrame],
    step_range: Optional[tuple[int, int]] = None,
    time_periods: Optional[List[Tuple[int, int]]] = None,
    annotate_states: bool = True,
    step_duration: Optional[float] = None,
    reference: Optional[str] = None,
    action_tolerance: float = 0.01,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, dict]:
    """Backwards-compatible wrapper around AlgorithmActionComparator.analyze_temporal.

    Parameters
    ----------
    logs_dict: dict[str, pl.DataFrame]
        Mapping from algorithm name to a single episode DataFrame (Polars) for each algorithm.
    step_range: Optional[tuple[int,int]]
        Single (start, end) tuple selecting a contiguous block to analyze when time_periods is not
        provided. If None, the common minimum length across inputs is used.
    time_periods: Optional[List[Tuple[int,int]]]
        If provided, a list of (start, end) windows to analyze. Windows are clamped to episode
        lengths and concatenated in the given order for plotting and statistics. The returned
        stats['step_range'] will contain the list of windows.
    annotate_states: bool
        Whether to include SOC and grid subplots beneath the action plots.
    step_duration: Optional[float]
        Optional step duration (hours) used to label x axes.
    reference: Optional[str]
        Algorithm name to use as reference for difference statistics.
    action_tolerance: float
        Threshold used to compute divergence_rate vs reference.
    save_path: Optional[str]
        Path to save the produced figure (if provided).

    Returns
    -------
    tuple[plt.Figure, dict]
        Matplotlib figure and computed statistics dictionary.
    """

    comparator = AlgorithmActionComparator()
    cfg = TemporalAnalysisConfig(
        step_range=step_range,
        time_periods=time_periods,
        annotate_states=annotate_states,
        step_duration=step_duration,
        reference=reference,
        action_tolerance=action_tolerance,
        save_path=save_path,
    )

    result = comparator.analyze_temporal(logs_dict=logs_dict, config=cfg)
    return result.figure, result.stats


