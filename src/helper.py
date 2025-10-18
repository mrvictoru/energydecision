import polars as pl
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Optional
from EnergySimEnv import SolarBatteryEnv
from pathlib import Path


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
        import matplotlib.pyplot as plt  # already imported above, safeguard
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