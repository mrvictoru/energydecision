"""
AEMO Battery Trading Environment

A Gymnasium environment for battery energy storage systems (BESS) participating in
Australia's National Electricity Market (NEM), including both energy spot market
and Frequency Control Ancillary Services (FCAS) markets.

This environment extends SolarBatteryEnv to incorporate real AEMO market data.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from EnergySimEnv import SolarBatteryEnv
from aemo_data import fetch_aemo_data_bundle
from batterydeg import DegradationModel, RainflowCounter


class AEMODataPreprocessor:
    """
    Preprocessor for AEMO data to prepare it for the RL environment.
    Handles time alignment, missing data, and normalization.
    """
    
    def __init__(self, 
                 step_duration_hours: float = 0.5,
                 missing_data_method: str = 'interpolate',
                 add_normalized_features: bool = True,
                 update_stats_from_data: bool = True):
        self.step_duration_hours = step_duration_hours
        self.missing_data_method = missing_data_method
        self.add_normalized_features = add_normalized_features
        self.update_stats_from_data = update_stats_from_data
        
        # Normalization bounds (will be updated from actual data)
        self.stats = {
            'RRP': {'min': -100.0, 'max': 500.0},  # Energy can go negative
            'FCAS_PRICE': {'min': 0.0, 'max': 100.0},
            'TOTALDEMAND': {'min': 4000.0, 'max': 12000.0},
            'GENERATION': {'min': 0.0, 'max': 5000.0},
        }
    
    def preprocess_aemo_data(self, 
                             prices: pl.DataFrame,
                             fcas: pl.DataFrame,
                             generation: pl.DataFrame) -> pl.DataFrame:
        """
        Preprocess AEMO data for the environment.
        
        Args:
            prices: Energy price DataFrame from fetch_aemo_dispatch_price
            fcas: FCAS price DataFrame from fetch_aemo_fcas_price
            generation: Generation DataFrame from fetch_aemo_generation_by_fuel
            
        Returns:
            Preprocessed Polars DataFrame ready for environment use
        """
        prices_pl = prices
        fcas_pl = fcas
        gen_pl = generation

        # Resample to match environment step duration
        prices_resampled = self._resample_data(prices_pl, 'SETTLEMENTDATE')
        fcas_resampled = self._resample_fcas(fcas_pl)
        gen_resampled = self._resample_generation(gen_pl)
        
        # Merge all datasets
        merged = self._merge_datasets(prices_resampled, fcas_resampled, gen_resampled)
        
        # Handle missing data
        merged = self._handle_missing_data(merged)
        
        # Add time features
        merged = self._add_time_features(merged)
        
        # Update stats from data
        if self.update_stats_from_data:
            self._update_stats_from_data(merged)

        # Normalize features (adds normalized columns, keeps raw columns)
        if self.add_normalized_features:
            merged = self._normalize_features(merged)
        
        return merged

    def _update_stats_from_data(self, df: pl.DataFrame) -> None:
        """Update normalization stats based on actual data."""
        if df.height == 0:
            return

        if 'RRP' in df.columns:
            min_val = df.select(pl.col('RRP').min()).item()
            max_val = df.select(pl.col('RRP').max()).item()
            if min_val is not None and max_val is not None:
                self.stats['RRP']['min'] = float(min_val)
                self.stats['RRP']['max'] = float(max_val)

        if 'TOTALDEMAND' in df.columns:
            min_val = df.select(pl.col('TOTALDEMAND').min()).item()
            max_val = df.select(pl.col('TOTALDEMAND').max()).item()
            if min_val is not None and max_val is not None:
                self.stats['TOTALDEMAND']['min'] = float(min_val)
                self.stats['TOTALDEMAND']['max'] = float(max_val)

        fcas_cols = [c for c in df.columns if c.startswith('FCAS_') and not c.endswith('_normalized')]
        if fcas_cols:
            fcas_min_row = df.select([pl.col(c).min() for c in fcas_cols]).row(0)
            fcas_max_row = df.select([pl.col(c).max() for c in fcas_cols]).row(0)
            valid_fcas_mins = [v for v in fcas_min_row if v is not None]
            valid_fcas_maxs = [v for v in fcas_max_row if v is not None]
            if valid_fcas_mins and valid_fcas_maxs:
                self.stats['FCAS_PRICE']['min'] = float(min(valid_fcas_mins))
                self.stats['FCAS_PRICE']['max'] = float(max(valid_fcas_maxs))

        gen_cols = [c for c in df.columns if c.startswith('GEN_') and not c.endswith('_pct')]
        if gen_cols:
            gen_min_row = df.select([pl.col(c).min() for c in gen_cols]).row(0)
            gen_max_row = df.select([pl.col(c).max() for c in gen_cols]).row(0)
            valid_gen_mins = [v for v in gen_min_row if v is not None]
            valid_gen_maxs = [v for v in gen_max_row if v is not None]
            if valid_gen_mins and valid_gen_maxs:
                self.stats['GENERATION']['min'] = float(min(valid_gen_mins))
                self.stats['GENERATION']['max'] = float(max(valid_gen_maxs))
    
    def _every_str(self) -> str:
        minutes = int(round(self.step_duration_hours * 60))
        return f"{minutes}m"

    def _ensure_dt(self, df: pl.DataFrame, time_col: str) -> pl.DataFrame:
        if time_col not in df.columns:
            return df
        if df.schema.get(time_col) == pl.Datetime:
            return df
        if df.schema.get(time_col) == pl.Utf8:
            return df.with_columns(pl.col(time_col).str.strptime(pl.Datetime, strict=False))
        return df.with_columns(pl.col(time_col).cast(pl.Datetime, strict=False))

    def _resample_data(self, df: pl.DataFrame, time_col: str) -> pl.DataFrame:
        """Resample 5-minute data to environment step duration."""
        df = self._ensure_dt(df, time_col)
        if df.height == 0:
            return df

        df = df.sort(time_col)
        every = self._every_str()

        numeric_cols = [c for c, t in df.schema.items() if c != time_col and t.is_numeric()]
        other_cols = [c for c in df.columns if c not in numeric_cols and c != time_col]

        aggs: list[pl.Expr] = []
        aggs.extend([pl.col(c).mean().alias(c) for c in numeric_cols])
        aggs.extend([pl.col(c).first().alias(c) for c in other_cols])

        return df.group_by_dynamic(time_col, every=every, label='left', closed='left').agg(aggs)
    
    def _resample_fcas(self, fcas_df: pl.DataFrame) -> pl.DataFrame:
        """Resample and pivot FCAS data to wide format."""
        if fcas_df.height == 0:
            return pl.DataFrame({'SETTLEMENTDATE': []})

        fcas_df = self._ensure_dt(fcas_df, 'SETTLEMENTDATE').sort('SETTLEMENTDATE')
        every = self._every_str()

        needed = {'SETTLEMENTDATE', 'SERVICE', 'PRICE'}
        if not needed.issubset(set(fcas_df.columns)):
            return pl.DataFrame({'SETTLEMENTDATE': []})

        grouped = (
            fcas_df
            .with_columns(pl.col('PRICE').cast(pl.Float64, strict=False))
            .group_by_dynamic('SETTLEMENTDATE', every=every, by='SERVICE', label='left', closed='left')
            .agg(pl.col('PRICE').mean().alias('PRICE'))
        )

        wide = grouped.pivot(index='SETTLEMENTDATE', columns='SERVICE', values='PRICE')
        rename_map = {c: f"FCAS_{c}" for c in wide.columns if c != 'SETTLEMENTDATE'}
        return wide.rename(rename_map)
    
    def _resample_generation(self, gen_df: pl.DataFrame) -> pl.DataFrame:
        """Resample and pivot generation data to wide format."""
        if gen_df.height == 0:
            return pl.DataFrame({'SETTLEMENTDATE': []})

        needed = {'SETTLEMENTDATE', 'FUEL_TYPE', 'GENERATION'}
        if not needed.issubset(set(gen_df.columns)):
            return pl.DataFrame({'SETTLEMENTDATE': []})

        gen_df = self._ensure_dt(gen_df, 'SETTLEMENTDATE').sort('SETTLEMENTDATE')
        every = self._every_str()

        # First aggregate per 5-min interval & fuel type (sum across units), then average across the env interval.
        gen_5 = (
            gen_df
            .with_columns(pl.col('GENERATION').cast(pl.Float64, strict=False))
            .group_by(['SETTLEMENTDATE', 'FUEL_TYPE'])
            .agg(pl.col('GENERATION').sum().alias('GENERATION'))
        )

        gen_res = (
            gen_5
            .sort(['FUEL_TYPE', 'SETTLEMENTDATE'])
            .group_by_dynamic('SETTLEMENTDATE', every=every, by='FUEL_TYPE', label='left', closed='left')
            .agg(pl.col('GENERATION').mean().alias('GENERATION'))
        )

        wide = gen_res.pivot(index='SETTLEMENTDATE', columns='FUEL_TYPE', values='GENERATION')
        rename_map = {c: f"GEN_{c}" for c in wide.columns if c != 'SETTLEMENTDATE'}
        return wide.rename(rename_map)
    
    def _merge_datasets(self, prices: pl.DataFrame, fcas: pl.DataFrame, gen: pl.DataFrame) -> pl.DataFrame:
        """Merge all datasets on timestamp."""
        merged = prices

        if fcas.height > 0 and 'SETTLEMENTDATE' in fcas.columns:
            merged = merged.join(fcas, on='SETTLEMENTDATE', how='left')

        if gen.height > 0 and 'SETTLEMENTDATE' in gen.columns:
            merged = merged.join(gen, on='SETTLEMENTDATE', how='left')

        return merged
    
    def _handle_missing_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle missing values."""
        if df.height == 0:
            return df

        df = df.sort('SETTLEMENTDATE') if 'SETTLEMENTDATE' in df.columns else df

        numeric_cols = [c for c, t in df.schema.items() if t.is_numeric()]
        if self.missing_data_method == 'interpolate' and numeric_cols:
            df = df.with_columns([pl.col(c).interpolate() for c in numeric_cols])
            df = df.fill_null(strategy='forward').fill_null(strategy='backward')
        elif self.missing_data_method == 'forward_fill':
            df = df.fill_null(strategy='forward', limit=12)

        if numeric_cols:
            df = df.with_columns([pl.col(c).fill_null(0.0) for c in numeric_cols])

        return df
    
    def _add_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add cyclical time features."""
        if df.height == 0:
            return df

        df = self._ensure_dt(df, 'SETTLEMENTDATE')

        hour = (
            pl.col('SETTLEMENTDATE').dt.hour().cast(pl.Float64) +
            pl.col('SETTLEMENTDATE').dt.minute().cast(pl.Float64) / 60.0
        )
        day = pl.col('SETTLEMENTDATE').dt.weekday().cast(pl.Float64)

        df = df.with_columns([
            (pl.lit(2 * np.pi) * hour / 24.0).sin().alias('hour_sin'),
            (pl.lit(2 * np.pi) * hour / 24.0).cos().alias('hour_cos'),
            (pl.lit(2 * np.pi) * day / 7.0).sin().alias('day_sin'),
            (pl.lit(2 * np.pi) * day / 7.0).cos().alias('day_cos'),
            ((hour >= 6.0) & (hour <= 22.0)).cast(pl.Float64).alias('is_peak'),
        ])

        return df
    
    def _normalize_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize features to [0, 1] range."""
        if df.height == 0:
            return df

        out = df

        # Normalize RRP (can be negative)
        if 'RRP' in out.columns:
            denom = (self.stats['RRP']['max'] - self.stats['RRP']['min'])
            out = out.with_columns(
                ((pl.col('RRP').cast(pl.Float64, strict=False) - self.stats['RRP']['min']) / denom)
                .clip(0.0, 1.0)
                .alias('RRP_normalized')
            )

        # Normalize TOTALDEMAND
        if 'TOTALDEMAND' in out.columns:
            denom = (self.stats['TOTALDEMAND']['max'] - self.stats['TOTALDEMAND']['min'])
            out = out.with_columns(
                ((pl.col('TOTALDEMAND').cast(pl.Float64, strict=False) - self.stats['TOTALDEMAND']['min']) / denom)
                .clip(0.0, 1.0)
                .alias('DEMAND_normalized')
            )

        # Normalize FCAS prices
        fcas_cols = [col for col in out.columns if col.startswith('FCAS_')]
        if fcas_cols:
            denom = (self.stats['FCAS_PRICE']['max'] - self.stats['FCAS_PRICE']['min'])
            out = out.with_columns([
                ((pl.col(col).cast(pl.Float64, strict=False) - self.stats['FCAS_PRICE']['min']) / denom)
                .clip(0.0, 1.0)
                .alias(f'{col}_normalized')
                for col in fcas_cols
            ])

        # Normalize generation (as percentage of total)
        gen_cols = [col for col in out.columns if col.startswith('GEN_')]
        if gen_cols:
            total_gen = pl.sum_horizontal([pl.col(c).cast(pl.Float64, strict=False) for c in gen_cols]).alias('_total_gen')
            out = out.with_columns(total_gen)
            out = out.with_columns([
                (pl.col(col).cast(pl.Float64, strict=False) / (pl.col('_total_gen') + 1e-6)).alias(f'{col}_pct')
                for col in gen_cols
            ]).drop('_total_gen')

        return out


class AEMOBatteryTradingEnv(gym.Env):
    """
    Gymnasium environment for battery trading in AEMO markets.
    
    This environment simulates a battery energy storage system participating in:
    1. Energy spot market (buy low, sell high)
    2. FCAS markets (provide frequency regulation services)
    
    The agent must optimize battery dispatch across multiple revenue streams while
    managing degradation and operational constraints.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 aemo_data: pl.DataFrame,
                 battery_capacity: float = 10.0,  # MWh (grid-scale)
                 max_battery_flow: float = 5.0,   # MW
                 init_battery_level: float = 5.0,  # MWh
                 max_step: int = 1000,
                 step_duration: float = 0.5,  # hours
                 battery_life_cost: float = 1_000_000.0,  # USD for grid-scale
                 render_mode: Optional[str] = None,
                 action_mode: str = 'simple',  # 'simple' or 'multi_market'
                 normalize_obs: bool = True,
                 return_raw_obs: bool = False,
                 random_episode_start: bool = False,
                 degradation_mode: str = 'rainflow',  # 'rainflow' or 'simple'
                 degradation_temperature: float = 25.0):
        """
        Initialize AEMO Battery Trading Environment.
        
        Args:
            aemo_data: Preprocessed AEMO market data DataFrame
            battery_capacity: Battery capacity in MWh
            max_battery_flow: Maximum charge/discharge rate in MW
            init_battery_level: Initial battery charge in MWh
            max_step: Maximum steps per episode
            step_duration: Duration of each step in hours (default 0.5 = 30 min)
            battery_life_cost: Total battery replacement cost in USD
            render_mode: Rendering mode ('human' or None)
            action_mode: 'simple' for energy-only or 'multi_market' for energy+FCAS
            random_episode_start: If True, sample a random valid episode start on
                reset; otherwise default to starting at index 0
            degradation_mode: 'rainflow' for physics-based Muenzel et al. model
                with rainflow cycle counting (recommended), or 'simple' for the
                original linear DoD-based approximation
            degradation_temperature: Ambient temperature in °C for degradation model
        """
        super().__init__()
        
        self.aemo_data = aemo_data
        self.initial_battery_capacity = float(battery_capacity)
        self.battery_capacity = float(battery_capacity)
        self.max_battery_flow = max_battery_flow
        self.init_battery_level = init_battery_level
        self.max_step = max_step
        self.step_duration = step_duration
        self.battery_life_cost = battery_life_cost
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.normalize_obs = normalize_obs
        self.return_raw_obs = return_raw_obs
        self.random_episode_start = random_episode_start
        self.degradation_mode = degradation_mode
        self.degradation_temperature = float(degradation_temperature)

        self._fcas_services = [
            'RAISEREG', 'LOWERREG', 'RAISE6SEC', 'LOWER6SEC',
            'RAISE60SEC', 'LOWER60SEC', 'RAISE5MIN', 'LOWER5MIN'
        ]
        self._gen_fuels = ['solar', 'wind']
        self._raw_col_bounds = self._compute_raw_col_bounds()

        # Degradation model setup
        if self.degradation_mode == 'rainflow':
            self.degradation_model = DegradationModel()
            max_c_rate = self.max_battery_flow / self.initial_battery_capacity
            self._rainflow_counter = RainflowCounter(
                step_duration=self.step_duration, max_c_rate=max_c_rate
            )
        
        # State variables
        self.current_step = 0
        self.episode_start_idx = 0
        self.battery_soc = init_battery_level  # State of charge (MWh)
        self.total_revenue = 0.0
        self.total_degradation_cost = 0.0
        self.total_degradation = 0.0
        self._rainflow_deg_cumulative = 0.0
        self._rainflow_num_cycles = 0
        self.soc_history: List[float] = []
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_soc = []
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation space components:
        # - Time features: 5 (hour_sin, hour_cos, day_sin, day_cos, is_peak)
        # - Energy market: 2 (RRP_normalized, DEMAND_normalized)
        # - FCAS prices: 8 (one per service, normalized)
        # - Generation: 2 (solar_pct, wind_pct) if available
        # - Battery state: 1 (SOC normalized)
        # Total: ~18 features
        
        obs_dim = 18  # Base dimension
        if self.normalize_obs:
            obs_low = np.zeros(obs_dim, dtype=np.float32)
            obs_high = np.ones(obs_dim, dtype=np.float32)
            obs_low[0:4] = -1.0
            obs_high[0:4] = 1.0
            obs_low[4] = 0.0
            obs_high[4] = 1.0
        else:
            obs_low, obs_high = self._build_raw_obs_bounds()

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        if self.action_mode == 'simple':
            # Simple action: just battery charge/discharge
            # Action in [-1, 1]: -1 = max discharge, 0 = idle, +1 = max charge
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        else:
            # Multi-market action: [battery_dispatch, fcas_raise_bid, fcas_lower_bid]
            # battery_dispatch: [-1, 1]
            # fcas_raise_bid: [0, 1] (fraction of FCAS MW capability to bid)
            # fcas_lower_bid: [0, 1] (fraction of FCAS MW capability to bid)
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0]),
                shape=(3,),
                dtype=np.float32
            )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        if options and 'return_raw_obs' in options:
            self.return_raw_obs = bool(options.get('return_raw_obs'))

        # Determine valid episode start range.
        max_start_idx = len(self.aemo_data) - self.max_step - 1
        if max_start_idx < 1:
            max_start_idx = 0

        requested_start_idx = options.get('episode_start_idx') if options else None
        use_random_start = bool(options.get('random_episode_start')) if options and 'random_episode_start' in options else self.random_episode_start

        if requested_start_idx is not None:
            self.episode_start_idx = int(np.clip(requested_start_idx, 0, max_start_idx))
        elif use_random_start:
            self.episode_start_idx = int(self.np_random.integers(0, max_start_idx + 1))
        else:
            self.episode_start_idx = 0

        self.current_step = 0
        
        # Reset battery state
        self.battery_capacity = self.initial_battery_capacity
        self.battery_soc = self.init_battery_level
        self.total_revenue = 0.0
        self.total_degradation_cost = 0.0
        self.total_degradation = 0.0
        self._rainflow_deg_cumulative = 0.0
        self._rainflow_num_cycles = 0

        # Reset degradation tracking
        init_soc_pct = float((self.battery_soc / self.initial_battery_capacity) * 100.0)
        self.soc_history = [init_soc_pct]
        if self.degradation_mode == 'rainflow':
            max_c_rate = self.max_battery_flow / self.initial_battery_capacity
            self._rainflow_counter = RainflowCounter(
                step_duration=self.step_duration, max_c_rate=max_c_rate
            )
        
        # Reset episode tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_soc = [self.battery_soc]
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        if self.return_raw_obs:
            info['raw_obs'] = self.get_raw_obs()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step."""
        # Parse action
        if self.action_mode == 'simple':
            battery_dispatch = float(action[0])
            fcas_raise_bid = 0.0
            fcas_lower_bid = 0.0
        else:
            battery_dispatch = float(action[0])
            fcas_raise_bid = float(action[1])
            fcas_lower_bid = float(action[2])
        
        # Get current market data
        current_idx = self.episode_start_idx + self.current_step
        if current_idx >= len(self.aemo_data):
            # Episode exhausted data
            terminated = True
            obs = self._get_observation()
            return obs, 0.0, terminated, False, {}
        
        market_data = self.aemo_data.row(current_idx, named=True)
        
        # Convert normalized action to actual power (MW)
        power_command = battery_dispatch * self.max_battery_flow
        
        # Convert to energy for this step (MWh)
        energy_command = power_command * self.step_duration
        
        # Apply battery constraints
        old_soc = self.battery_soc
        new_soc = old_soc + energy_command
        
        # Clip to battery capacity limits
        new_soc = np.clip(new_soc, 0, self.battery_capacity)
        actual_energy = new_soc - old_soc
        actual_power = actual_energy / self.step_duration
        
        # Update battery SOC
        self.battery_soc = new_soc

        # --- Degradation calculation ---
        soc_pct = float((self.battery_soc / self.initial_battery_capacity) * 100.0)
        self.soc_history.append(soc_pct)

        step_degradation = 0.0
        if self.degradation_mode == 'rainflow':
            new_cycles = self._rainflow_counter.update(soc_pct)
            for SoC_avg, DoD, Id_cycle, Ich_cycle in new_cycles:
                inc, _ = self._safe_degradation_per_cycle(
                    Id_cycle, Ich_cycle, SoC_avg, DoD
                )
                step_degradation += inc
            self._rainflow_num_cycles += len(new_cycles)
            self._rainflow_deg_cumulative += step_degradation
        else:
            # Simplified degradation (original model)
            dod = abs(actual_energy) / self.initial_battery_capacity
            step_degradation = dod * 0.0001

        self.total_degradation = min(1.0, self.total_degradation + step_degradation)

        # Capacity fade
        self.battery_capacity = max(
            self.initial_battery_capacity * (1.0 - self.total_degradation), 1e-9
        )
        self.battery_soc = min(self.battery_soc, self.battery_capacity)

        degradation_cost = step_degradation * self.battery_life_cost
        
        # Calculate reward
        reward, energy_revenue, fcas_revenue = self._calculate_reward(
            market_data, actual_power, actual_energy,
            fcas_raise_bid, fcas_lower_bid, old_soc, new_soc,
            degradation_cost
        )
        
        # Check termination
        self.current_step += 1
        terminated = bool(
            (self.current_step >= self.max_step)
            or (current_idx + 1 >= len(self.aemo_data))
            or (self.total_degradation >= 1.0)
        )
        truncated = False
        
        # Track episode
        self.episode_rewards.append(reward)
        self.episode_actions.append(battery_dispatch)
        self.episode_soc.append(self.battery_soc)
        
        # Get next observation
        obs = self._get_observation()
        
        info = self._make_reward_info(
            battery_soc=self.battery_soc,
            battery_dispatch=actual_power,
            energy_price=market_data.get('RRP', 0),
            energy_revenue=energy_revenue,
            fcas_revenue=fcas_revenue,
            fcas_raise_bid=fcas_raise_bid,
            fcas_lower_bid=fcas_lower_bid,
            actual_energy=actual_energy,
            degradation_cost=degradation_cost,
            current_step=self.current_step,
            step_degradation=step_degradation,
            total_degradation=self.total_degradation,
            capacity_mwh=self.battery_capacity,
            rainflow_cumulative_deg=self._rainflow_deg_cumulative,
            rainflow_num_cycles=self._rainflow_num_cycles,
        )

        if self.return_raw_obs:
            info['raw_obs'] = self.get_raw_obs()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_raw_col_bounds(self) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
        if self.aemo_data.height == 0:
            return bounds

        def _get_min_max(col: str) -> Optional[Tuple[float, float]]:
            if col not in self.aemo_data.columns:
                return None
            min_val = self.aemo_data.select(pl.col(col).min()).item()
            max_val = self.aemo_data.select(pl.col(col).max()).item()
            if min_val is None or max_val is None:
                return None
            return float(min_val), float(max_val)

        for col in ['RRP', 'TOTALDEMAND']:
            mm = _get_min_max(col)
            if mm:
                bounds[col] = mm

        for service in self._fcas_services:
            col = f'FCAS_{service}'
            mm = _get_min_max(col)
            if mm:
                bounds[col] = mm

        for fuel in self._gen_fuels:
            col = f'GEN_{fuel}'
            mm = _get_min_max(col)
            if mm:
                bounds[col] = mm

        return bounds

    def _build_raw_obs_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        obs_dim = 18
        obs_low = np.zeros(obs_dim, dtype=np.float32)
        obs_high = np.zeros(obs_dim, dtype=np.float32)

        obs_low[0:4] = -1.0
        obs_high[0:4] = 1.0
        obs_low[4] = 0.0
        obs_high[4] = 1.0

        def _set_bounds(idx: int, col: str, default_low: float = 0.0, default_high: float = 1.0):
            if col in self._raw_col_bounds:
                lo, hi = self._raw_col_bounds[col]
            else:
                lo, hi = default_low, default_high
            obs_low[idx] = lo
            obs_high[idx] = hi

        _set_bounds(5, 'RRP', -100.0, 500.0)
        _set_bounds(6, 'TOTALDEMAND', 0.0, 12000.0)

        base_idx = 7
        for i, service in enumerate(self._fcas_services):
            _set_bounds(base_idx + i, f'FCAS_{service}', 0.0, 100.0)

        gen_idx = base_idx + len(self._fcas_services)
        for j, fuel in enumerate(self._gen_fuels):
            _set_bounds(gen_idx + j, f'GEN_{fuel}', 0.0, 5000.0)

        obs_low[-1] = 0.0
        obs_high[-1] = float(self.battery_capacity)

        return obs_low, obs_high

    def _get_observation(self) -> np.ndarray:
        """Construct observation from current state."""
        components = self._get_observation_components()
        _, normalized_obs = components
        if self.normalize_obs:
            return normalized_obs
        return components[0]

    def _get_observation_components(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (raw_obs, normalized_obs) for current state."""
        current_idx = self.episode_start_idx + self.current_step

        if current_idx >= len(self.aemo_data):
            zero = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return zero, zero

        market_data = self.aemo_data.row(current_idx, named=True)

        raw_obs: list[float] = []
        norm_obs: list[float] = []

        # Time features (5)
        time_vals = [
            market_data.get('hour_sin', 0),
            market_data.get('hour_cos', 0),
            market_data.get('day_sin', 0),
            market_data.get('day_cos', 0),
            market_data.get('is_peak', 0),
        ]
        raw_obs.extend(time_vals)
        norm_obs.extend(time_vals)

        # Energy market (2)
        raw_obs.extend([
            market_data.get('RRP', 0),
            market_data.get('TOTALDEMAND', 0),
        ])
        norm_obs.extend([
            market_data.get('RRP_normalized', 0),
            market_data.get('DEMAND_normalized', 0),
        ])

        # FCAS prices (8)
        for service in self._fcas_services:
            raw_obs.append(market_data.get(f'FCAS_{service}', 0))
            norm_obs.append(market_data.get(f'FCAS_{service}_normalized', 0))

        # Generation mix (2)
        for fuel in self._gen_fuels:
            raw_obs.append(market_data.get(f'GEN_{fuel}', 0))
            norm_obs.append(market_data.get(f'GEN_{fuel}_pct', 0))

        # Battery state (1)
        raw_obs.append(self.battery_soc)
        norm_obs.append(self.battery_soc / self.battery_capacity if self.battery_capacity > 0 else 0.0)

        return np.array(raw_obs, dtype=np.float32), np.array(norm_obs, dtype=np.float32)

    def get_raw_obs(self) -> np.ndarray:
        """Return raw observation at current step."""
        raw_obs, _ = self._get_observation_components()
        return raw_obs
    
    def _safe_degradation_per_cycle(self, Id: float, Ich: float, soc: float, DoD: float) -> Tuple[float, Optional[str]]:
        """Compute degradation per cycle; return (value, error_message)."""
        try:
            value = self.degradation_model.degradation_per_cycle(
                T=self.degradation_temperature,
                Id=Id,
                Ich=Ich,
                SOCav=soc,
                DOD=DoD,
            )
            return float(value), None
        except ValueError as exc:
            return 0.0, str(exc)

    def _calculate_reward(self, market_data, actual_power: float, actual_energy: float,
                         fcas_raise_bid: float, fcas_lower_bid: float,
                         old_soc: float, new_soc: float,
                         degradation_cost: float = 0.0) -> Tuple[float, float, float]:
        """Calculate reward for this step."""
        # Energy market revenue
        energy_price = market_data.get('RRP', 0)  # $/MWh
        
        if actual_power < 0:  # Discharging (selling)
            energy_revenue = abs(actual_energy) * energy_price
        else:  # Charging (buying)
            energy_revenue = -abs(actual_energy) * energy_price
        
        # FCAS revenue (simplified)
        # In reality, revenue depends on enablement (MW) × price ($/MW/h) × duration
        fcas_revenue = 0.0
        if self.action_mode == 'multi_market':
            requested_raise_mw = max(0.0, float(fcas_raise_bid)) * float(self.max_battery_flow)
            requested_lower_mw = max(0.0, float(fcas_lower_bid)) * float(self.max_battery_flow)

            # Simplified FCAS enablement model:
            # bids are fractions of inverter power capability (MW), then capped by
            # one-step SOC headroom so the enabled regulation service is physically feasible.
            if self.step_duration > 0:
                available_raise_mw = max(0.0, float(self.battery_soc) / float(self.step_duration))
                available_lower_mw = max(
                    0.0,
                    float(self.battery_capacity - self.battery_soc) / float(self.step_duration),
                )
            else:
                available_raise_mw = float(self.max_battery_flow)
                available_lower_mw = float(self.max_battery_flow)

            fcas_raise_capacity = min(requested_raise_mw, float(self.max_battery_flow), available_raise_mw)
            fcas_lower_capacity = min(requested_lower_mw, float(self.max_battery_flow), available_lower_mw)
            
            # Get FCAS prices ($/MW/h)
            raisereg_price = market_data.get('FCAS_RAISEREG', 0)
            lowerreg_price = market_data.get('FCAS_LOWERREG', 0)
            
            # Calculate revenue for this step
            fcas_revenue = (fcas_raise_capacity * raisereg_price * self.step_duration +
                           fcas_lower_capacity * lowerreg_price * self.step_duration)
        
        # SOC violation penalty
        soc_penalty = 0.0
        if self.battery_soc < 0.1 * self.battery_capacity or self.battery_soc > 0.9 * self.battery_capacity:
            soc_penalty = -10.0  # Small penalty for operating at extremes
        
        # Total reward
        reward = energy_revenue + fcas_revenue - degradation_cost + soc_penalty
        
        # Track totals
        self.total_revenue += energy_revenue + fcas_revenue
        self.total_degradation_cost += degradation_cost

        normalized_reward = reward / 1000.0
        return normalized_reward, energy_revenue, fcas_revenue

    def _make_reward_info(self,
                          battery_soc: float,
                          battery_dispatch: float,
                          energy_price: float,
                          energy_revenue: float,
                          fcas_revenue: float,
                          fcas_raise_bid: float,
                          fcas_lower_bid: float,
                          actual_energy: float,
                          degradation_cost: float,
                          current_step: int,
                          step_degradation: float = 0.0,
                          total_degradation: float = 0.0,
                          capacity_mwh: float = 0.0,
                          rainflow_cumulative_deg: float = 0.0,
                          rainflow_num_cycles: int = 0) -> Dict[str, float]:
        """Return a compact reward/info dict for debugging and tracking."""
        return {
            'battery_soc': battery_soc,
            'battery_dispatch': battery_dispatch,
            'energy_price': energy_price,
            'energy_revenue': energy_revenue,
            'fcas_revenue': fcas_revenue,
            'fcas_raise_bid': fcas_raise_bid,
            'fcas_lower_bid': fcas_lower_bid,
            'actual_energy': actual_energy,
            'degradation_cost': degradation_cost,
            'step_degradation': step_degradation,
            'total_degradation': total_degradation,
            'capacity_mwh': capacity_mwh,
            'rainflow_cumulative_deg': rainflow_cumulative_deg,
            'rainflow_num_cycles': rainflow_num_cycles,
            'total_revenue': self.total_revenue,
            'total_degradation_cost': self.total_degradation_cost,
            'current_step': current_step,
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, SOC: {self.battery_soc:.2f} MWh, "
                  f"Revenue: ${self.total_revenue:.2f}")


def create_aemo_env_from_data(start_date: datetime,
                               end_date: datetime,
                               region: str = "NSW1",
                               cache_dir: str = "data/aemo",
                               **env_kwargs) -> AEMOBatteryTradingEnv:
    """
    Convenience function to create AEMO environment with data fetching.
    
    Args:
        start_date: Start date for AEMO data
        end_date: End date for AEMO data
        region: AEMO region
        cache_dir: Cache directory for AEMO data
        **env_kwargs: Additional arguments for AEMOBatteryTradingEnv
        
    Returns:
        Initialized AEMOBatteryTradingEnv
        
    Example:
        >>> from datetime import datetime
        >>> env = create_aemo_env_from_data(
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 1, 7),
        ...     region="NSW1",
        ...     battery_capacity=10.0,
        ...     action_mode='multi_market'
        ... )
        >>> obs, info = env.reset()
        >>> for _ in range(100):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     if terminated or truncated:
        ...         break
    """
    print(f"Fetching AEMO data for {region} from {start_date.date()} to {end_date.date()}...")
    
    # Fetch AEMO data
    data = fetch_aemo_data_bundle(
        start_date=start_date,
        end_date=end_date,
        region=region,
        fcas_services=["RAISEREG", "LOWERREG", "RAISE6SEC", "LOWER6SEC",
                      "RAISE60SEC", "LOWER60SEC", "RAISE5MIN", "LOWER5MIN"],
        fuel_types=["solar", "wind"],
        cache_dir=cache_dir,
    )
    
    # Preprocess data
    print("Preprocessing AEMO data...")
    preprocessor = AEMODataPreprocessor(
        step_duration_hours=env_kwargs.get('step_duration', 0.5)
    )
    
    processed_data = preprocessor.preprocess_aemo_data(
        prices=data['prices'],
        fcas=data['fcas'],
        generation=data['generation']
    )
    
    print(f"Processed {len(processed_data)} time steps")
    
    # Create environment
    env = AEMOBatteryTradingEnv(
        aemo_data=processed_data,
        **env_kwargs
    )
    
    return env
