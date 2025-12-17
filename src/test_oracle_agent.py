import numpy as np
import polars as pl

from EnergySimEnv import SolarBatteryEnv
from decision import Agent


def _build_test_dataframe(num_steps: int = 4) -> pl.DataFrame:
    timestamps = [f"2021-01-01T{hour:02d}:00:00" for hour in range(num_steps)]
    return pl.DataFrame({
        "Timestamp": list(range(num_steps)),
        "Time": timestamps,
        "SolarGen": [0.5, 1.5, 2.0, 1.0][:num_steps],
        "HouseLoad": [1.0, 0.5, 1.5, 0.5][:num_steps],
        "ImportEnergyPrice": [0.25, 0.3, 0.35, 0.4][:num_steps],
        "ExportEnergyPrice": [0.1, 0.12, 0.15, 0.18][:num_steps],
    })


def _make_env(df: pl.DataFrame) -> SolarBatteryEnv:
    return SolarBatteryEnv(
        df=df,
        battery_capacity=4.0,
        max_battery_flow=2.0,
        max_grid_flow=5.0,
        init_battery_level=1.5,
        max_step=len(df),
        battery_life_cost=1000.0,
        correction_interval=100,
        init_correction_steps=[50],
        dynamic_interval_min_ratio=1.0,
        step_duration=1.0,
    )


def test_oracle_matches_best_single_step_action():
    df = _build_test_dataframe()
    env = _make_env(df)
    env.reset()

    agent = Agent(
        env,
        algorithm='oracle',
        horizon=1,
        action_resolution=3,
        # oracle uses the same horizon/action_resolution by default
    )

    raw_obs = env.get_raw_obs()
    oracle_action = agent.choose_action(raw_obs)[0]

    candidate_actions = np.linspace(-1.0, 1.0, 3, dtype=np.float32)
    rewards = []
    for act in candidate_actions:
        test_env = _make_env(df)
        test_env.reset()
        _, reward, terminated, truncated, _ = test_env.step([float(act)])
        assert not terminated
        rewards.append(reward)

    best_action = candidate_actions[int(np.argmax(rewards))]
    assert np.isclose(float(oracle_action), float(best_action), atol=1e-6)
