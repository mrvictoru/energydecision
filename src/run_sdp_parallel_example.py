import polars as pl

from helper import transform_polars_df, make_env

from decision import run_episodes_parallel, Agent
from EnergySimEnv import SolarBatteryEnv
import argparse

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def main(datapath='../data/2010-2011 Solar home electricity data.csv', output_path='../data/sdp_all_episode_01_logs.parquet'):

    # skip the first line in csv and read the next line as column
    # then read the rest of the file and store as dataframe
    print(f"Reading data from {datapath}")
    df = pl.read_csv(datapath, skip_rows=1)

    # get all the unique customers as their own dataframes
    customers = df['Customer'].unique()

    # loop through each customer and use transform_polars_df to get the dataframe and store it in a list call dataset
    dataset = []
    print(f"Transforming data for {len(customers)} customers")
    for customer in customers:
        customer_df = df.filter(pl.col('Customer') == customer)
        try:
            newcustomerdf = transform_polars_df(customer_df, import_energy_price=0.23, export_energy_price=0.015, price_periods="7am – 10am | 4pm – 9pm", default_import_energy_price=0.15, default_export_energy_price=0.01)
        except Exception as e:
            print(f"Error with customer as training dataset: {customer}")
            print(e)
            break
        dataset.append(newcustomerdf)

    env_fns = [make_env(ds) for ds in dataset]

    num_step = None # pick the number of step for the simulation
    envs = [env_fn(num_step) for env_fn in env_fns]

    sdp_agent_kwargs = {
        'algorithm': 'sdp',
        'soc_resolution': 20,
        'action_resolution': 41,  # best to be 2*soc_resolution + 1
        'degradation_model': 'linear', # the other option being static degradation 'static'
        'linear_deg_cost_p_kwh': 0.2 # only needed if using linear
    }
    print(f"Running SDP agent with kwargs: {sdp_agent_kwargs}")
    sdp_episode_logs = run_episodes_parallel(Agent, envs, agent_kwargs=sdp_agent_kwargs, max_workers=2, use_notebook_tqdm=False)
    print(f"Completed running {len(sdp_episode_logs)} episodes.")
    # combine all the dataframes in sdp_episode_logs into one dataframe and add a column episode_id to identify each episode
    print(f"Combining all episode logs into one dataframe and saving to {output_path}")
    dfs_with_id = [df.with_columns(pl.lit(i).alias("episode_id")) for i, df in enumerate(sdp_episode_logs)]
    sdp_all_logs = pl.concat(dfs_with_id)
    sdp_all_logs.write_parquet(output_path)
    print(f"Saved all episode logs to {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SDP episodes in parallel.")
    parser.add_argument(
        "-d", "--datapath",
        default="../data/2010-2011 Solar home electricity data.csv",
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "-o", "--output-path",
        dest="output_path",
        default="../data/sdp_all_episode_01_logs.parquet",
        help="Path to the output Parquet file."
    )
    args = parser.parse_args()
    main(datapath=args.datapath, output_path=args.output_path)