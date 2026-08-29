import datetime as dt

import numpy as np
import polars as pl
import torch

from EnergySimEnv import SolarBatteryEnv
from decision import Agent
from decision_transformer import DecisionTransformer
from dt_artifacts import write_model_kwargs
from household_optimization import build_j_t_soc_prompt_provider
from household_optimization import bill_for_actions, bootstrap_mean_ci, optimize_dispatch
from household_replay import Tariff


def _frame():
    start = dt.datetime(2025, 6, 1)
    timestamps = [start + dt.timedelta(minutes=5 * index) for index in range(288)]
    solar = np.zeros(288)
    solar[132:168] = 4.0  # free-window solar surplus
    return pl.DataFrame({
        "Timestamp": timestamps,
        "HouseLoad": np.full(288, 1.0),
        "SolarGen": solar,
    })


def test_optimizer_reduces_bill_and_respects_hardware_limits():
    result = optimize_dispatch(
        _frame(),
        tariff=Tariff(),
        capacity_kwh=5.0,
        max_flow_kw=3.3,
        roundtrip_eff=0.8,
    )
    no_battery = bill_for_actions(_frame(), np.zeros(288), Tariff())
    assert result.bill_aud < no_battery
    assert np.abs(result.actions_kw).max() <= 3.3 + 1e-9
    assert result.soc_kwh.min() >= -1e-9
    assert result.soc_kwh.max() <= 5.0 + 1e-9
    assert result.cost_to_go.shape == (289, 31)
    assert result.soc_levels_kwh.shape == (31,)


def test_bootstrap_ci_is_reproducible():
    assert bootstrap_mean_ci([1.0, 2.0, 3.0], seed=4) == bootstrap_mean_ci([1.0, 2.0, 3.0], seed=4)


def test_environment_returns_terminal_observation_at_final_row():
    frame = _frame().head(2).with_columns([
        pl.col("SolarGen").alias("FutureSolar"),
        pl.col("HouseLoad").alias("FutureLoad"),
        pl.lit(0.30).alias("ImportEnergyPrice"),
        pl.lit(0.05).alias("ExportEnergyPrice"),
        pl.col("Timestamp").alias("Time"),
    ])
    env = SolarBatteryEnv(frame, max_step=2)
    env.reset()
    env.step([0.0])
    obs, _, _, truncated, _ = env.step([0.0])
    assert truncated
    assert obs.shape == env.observation_space.shape


def test_j_t_soc_provider_varies_by_state_and_agent_uses_it():
    frame = _frame().with_columns([
        (pl.col("HouseLoad") / 12.0).alias("HouseLoad"),
        (pl.col("SolarGen") / 12.0).alias("SolarGen"),
        pl.col("SolarGen").alias("FutureSolar"),
        pl.col("HouseLoad").alias("FutureLoad"),
        pl.lit(0.30).alias("ImportEnergyPrice"),
        pl.lit(0.05).alias("ExportEnergyPrice"),
        pl.col("Timestamp").alias("Time"),
    ])
    provider = build_j_t_soc_prompt_provider(
        frame, tariff=Tariff(), capacity_kwh=5.0, max_flow_kw=3.3
    )
    assert provider(0.0, 0) != provider(5.0, 0)

    class CaptureModel(torch.nn.Module):
        context_len = 1
        state_dim = 12
        act_dim = 1

        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))
            self.embed_timestep = torch.nn.Embedding(2, 1)
            self.prompts = []

        def get_action(self, states, actions, rtg, timesteps, attention_mask=None):
            self.prompts.append(float(rtg[0, -1, 0]))
            return torch.zeros(1, device=states.device)

    env = SolarBatteryEnv(frame.head(2), battery_capacity=5.0, max_step=2)
    model = CaptureModel()
    Agent(env, algorithm="dt", model=model, rtg_prompt_provider=lambda soc, step: -7.5).run_episode()
    assert model.prompts == [-7.5, -7.5]


def test_j_t_soc_provider_respects_roundtrip_efficiency():
    frame = _frame().with_columns([
        (pl.col("HouseLoad") / 12.0).alias("HouseLoad"),
        (pl.col("SolarGen") / 12.0).alias("SolarGen"),
    ])
    tariff = Tariff()
    lossy = build_j_t_soc_prompt_provider(
        frame, tariff=tariff, capacity_kwh=5.0, max_flow_kw=3.3,
        roundtrip_eff=0.80,
    )
    ideal = build_j_t_soc_prompt_provider(
        frame, tariff=tariff, capacity_kwh=5.0, max_flow_kw=3.3,
        roundtrip_eff=1.0,
    )
    assert np.isfinite(lossy(2.5, 0))
    assert lossy(2.5, 0) != ideal(2.5, 0)


def test_checkpoint_model_configuration_round_trip(tmp_path):
    kwargs = {
        "state_dim": 12, "act_dim": 1, "n_block": 2, "h_dim": 128,
        "context_len": 60, "n_heads": 8, "drop_p": 0.1,
        "max_timestep": 2016,
    }
    path = write_model_kwargs(tmp_path / "h2_sdp_jtsoc_model_kwargs.json", kwargs)
    loaded = __import__("json").loads(path.read_text())
    model = DecisionTransformer(**loaded)
    assert loaded == kwargs
    assert model.embed_timestep.num_embeddings == 2016
