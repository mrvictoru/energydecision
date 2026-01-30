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
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from EnergySimEnv import SolarBatteryEnv
from aemo_data import fetch_aemo_data_bundle


class AEMODataPreprocessor:
    """
    Preprocessor for AEMO data to prepare it for the RL environment.
    Handles time alignment, missing data, and normalization.
    """
    
    def __init__(self, 
                 step_duration_hours: float = 0.5,
                 missing_data_method: str = 'interpolate'):
        self.step_duration_hours = step_duration_hours
        self.missing_data_method = missing_data_method
        
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
                             generation: pl.DataFrame) -> pd.DataFrame:
        """
        Preprocess AEMO data for the environment.
        
        Args:
            prices: Energy price DataFrame from fetch_aemo_dispatch_price
            fcas: FCAS price DataFrame from fetch_aemo_fcas_price
            generation: Generation DataFrame from fetch_aemo_generation_by_fuel
            
        Returns:
            Preprocessed pandas DataFrame ready for environment use
        """
        # Convert to pandas for easier manipulation
        prices_pdf = prices.to_pandas()
        fcas_pdf = fcas.to_pandas()
        gen_pdf = generation.to_pandas()
        
        # Resample to match environment step duration
        prices_resampled = self._resample_data(prices_pdf, 'SETTLEMENTDATE')
        fcas_resampled = self._resample_fcas(fcas_pdf)
        gen_resampled = self._resample_generation(gen_pdf)
        
        # Merge all datasets
        merged = self._merge_datasets(prices_resampled, fcas_resampled, gen_resampled)
        
        # Handle missing data
        merged = self._handle_missing_data(merged)
        
        # Add time features
        merged = self._add_time_features(merged)
        
        # Normalize features
        merged = self._normalize_features(merged)
        
        return merged
    
    def _resample_data(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Resample 5-minute data to environment step duration."""
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        
        resample_freq = f"{int(self.step_duration_hours * 60)}min"
        
        # Aggregate numeric columns
        agg_dict = {}
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'first'
        
        resampled = df.resample(resample_freq).agg(agg_dict)
        return resampled.reset_index()
    
    def _resample_fcas(self, fcas_df: pd.DataFrame) -> pd.DataFrame:
        """Resample and pivot FCAS data to wide format."""
        fcas_df = fcas_df.copy()
        fcas_df['SETTLEMENTDATE'] = pd.to_datetime(fcas_df['SETTLEMENTDATE'])
        fcas_df = fcas_df.set_index('SETTLEMENTDATE')
        
        resample_freq = f"{int(self.step_duration_hours * 60)}min"
        
        # Pivot to wide format
        fcas_wide = fcas_df.pivot_table(
            index=fcas_df.index,
            columns='SERVICE',
            values='PRICE',
            aggfunc='mean'
        )
        
        # Resample
        fcas_resampled = fcas_wide.resample(resample_freq).mean()
        
        # Rename columns to have FCAS_ prefix
        fcas_resampled.columns = [f'FCAS_{col}' for col in fcas_resampled.columns]
        
        return fcas_resampled.reset_index()
    
    def _resample_generation(self, gen_df: pd.DataFrame) -> pd.DataFrame:
        """Resample and pivot generation data to wide format."""
        if len(gen_df) == 0:
            return pd.DataFrame({'SETTLEMENTDATE': []})
        
        gen_df = gen_df.copy()
        gen_df['SETTLEMENTDATE'] = pd.to_datetime(gen_df['SETTLEMENTDATE'])
        gen_df = gen_df.set_index('SETTLEMENTDATE')
        
        resample_freq = f"{int(self.step_duration_hours * 60)}min"
        
        # Pivot to wide format
        gen_wide = gen_df.pivot_table(
            index=gen_df.index,
            columns='FUEL_TYPE',
            values='GENERATION',
            aggfunc='sum'
        )
        
        # Resample
        gen_resampled = gen_wide.resample(resample_freq).mean()
        
        # Rename columns to have GEN_ prefix
        gen_resampled.columns = [f'GEN_{col}' for col in gen_resampled.columns]
        
        return gen_resampled.reset_index()
    
    def _merge_datasets(self, prices: pd.DataFrame, fcas: pd.DataFrame, gen: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets on timestamp."""
        merged = prices.copy()
        
        if len(fcas) > 0:
            merged = merged.merge(fcas, on='SETTLEMENTDATE', how='left')
        
        if len(gen) > 0:
            merged = merged.merge(gen, on='SETTLEMENTDATE', how='left')
        
        return merged
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        df = df.copy()
        
        if self.missing_data_method == 'interpolate':
            # Interpolate numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        elif self.missing_data_method == 'forward_fill':
            df = df.fillna(method='ffill', limit=12)
        
        # Fill any remaining NaNs with 0
        df = df.fillna(0)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time features."""
        df = df.copy()
        
        # Ensure SETTLEMENTDATE is datetime
        df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
        
        # Hour of day (0-23)
        hour = df['SETTLEMENTDATE'].dt.hour + df['SETTLEMENTDATE'].dt.minute / 60
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6)
        day = df['SETTLEMENTDATE'].dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * day / 7)
        df['day_cos'] = np.cos(2 * np.pi * day / 7)
        
        # Peak period indicator (6-22)
        df['is_peak'] = ((hour >= 6) & (hour <= 22)).astype(float)
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to [0, 1] range."""
        df = df.copy()
        
        # Normalize RRP (can be negative)
        if 'RRP' in df.columns:
            df['RRP_normalized'] = (df['RRP'] - self.stats['RRP']['min']) / \
                                   (self.stats['RRP']['max'] - self.stats['RRP']['min'])
            df['RRP_normalized'] = df['RRP_normalized'].clip(0, 1)
        
        # Normalize TOTALDEMAND
        if 'TOTALDEMAND' in df.columns:
            df['DEMAND_normalized'] = (df['TOTALDEMAND'] - self.stats['TOTALDEMAND']['min']) / \
                                      (self.stats['TOTALDEMAND']['max'] - self.stats['TOTALDEMAND']['min'])
            df['DEMAND_normalized'] = df['DEMAND_normalized'].clip(0, 1)
        
        # Normalize FCAS prices
        fcas_cols = [col for col in df.columns if col.startswith('FCAS_')]
        for col in fcas_cols:
            df[f'{col}_normalized'] = (df[col] - self.stats['FCAS_PRICE']['min']) / \
                                      (self.stats['FCAS_PRICE']['max'] - self.stats['FCAS_PRICE']['min'])
            df[f'{col}_normalized'] = df[f'{col}_normalized'].clip(0, 1)
        
        # Normalize generation (as percentage of total)
        gen_cols = [col for col in df.columns if col.startswith('GEN_')]
        if gen_cols:
            total_gen = df[gen_cols].sum(axis=1)
            for col in gen_cols:
                df[f'{col}_pct'] = df[col] / (total_gen + 1e-6)  # Avoid division by zero
        
        return df


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
                 aemo_data: pd.DataFrame,
                 battery_capacity: float = 10.0,  # MWh (grid-scale)
                 max_battery_flow: float = 5.0,   # MW
                 init_battery_level: float = 5.0,  # MWh
                 max_step: int = 1000,
                 step_duration: float = 0.5,  # hours
                 battery_life_cost: float = 1_000_000.0,  # USD for grid-scale
                 render_mode: Optional[str] = None,
                 action_mode: str = 'simple'):  # 'simple' or 'multi_market'
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
        """
        super().__init__()
        
        self.aemo_data = aemo_data
        self.battery_capacity = battery_capacity
        self.max_battery_flow = max_battery_flow
        self.init_battery_level = init_battery_level
        self.max_step = max_step
        self.step_duration = step_duration
        self.battery_life_cost = battery_life_cost
        self.render_mode = render_mode
        self.action_mode = action_mode
        
        # State variables
        self.current_step = 0
        self.episode_start_idx = 0
        self.battery_soc = init_battery_level  # State of charge (MWh)
        self.total_revenue = 0.0
        self.total_degradation_cost = 0.0
        
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
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
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
            # fcas_raise_bid: [0, 1] (fraction of capacity to bid)
            # fcas_lower_bid: [0, 1] (fraction of capacity to bid)
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0]),
                shape=(3,),
                dtype=np.float32
            )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Randomly sample episode start (leave buffer for data availability)
        max_start_idx = len(self.aemo_data) - self.max_step - 1
        if max_start_idx < 1:
            max_start_idx = 0
        
        self.episode_start_idx = self.np_random.integers(0, max(1, max_start_idx))
        self.current_step = 0
        
        # Reset battery state
        self.battery_soc = self.init_battery_level
        self.total_revenue = 0.0
        self.total_degradation_cost = 0.0
        
        # Reset episode tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_soc = [self.battery_soc]
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
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
        
        market_data = self.aemo_data.iloc[current_idx]
        
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
        
        # Calculate reward
        reward = self._calculate_reward(market_data, actual_power, actual_energy,
                                        fcas_raise_bid, fcas_lower_bid, old_soc, new_soc)
        
        # Check termination
        self.current_step += 1
        terminated = (self.current_step >= self.max_step) or (current_idx + 1 >= len(self.aemo_data))
        truncated = False
        
        # Track episode
        self.episode_rewards.append(reward)
        self.episode_actions.append(battery_dispatch)
        self.episode_soc.append(self.battery_soc)
        
        # Get next observation
        obs = self._get_observation()
        
        info = {
            'battery_soc': self.battery_soc,
            'battery_dispatch': actual_power,
            'energy_price': market_data.get('RRP', 0),
            'total_revenue': self.total_revenue,
            'total_degradation': self.total_degradation_cost,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation from current state."""
        current_idx = self.episode_start_idx + self.current_step
        
        if current_idx >= len(self.aemo_data):
            # Return zero observation if out of bounds
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        market_data = self.aemo_data.iloc[current_idx]
        
        obs = []
        
        # Time features (5)
        obs.extend([
            market_data.get('hour_sin', 0),
            market_data.get('hour_cos', 0),
            market_data.get('day_sin', 0),
            market_data.get('day_cos', 0),
            market_data.get('is_peak', 0),
        ])
        
        # Energy market (2)
        obs.extend([
            market_data.get('RRP_normalized', 0),
            market_data.get('DEMAND_normalized', 0),
        ])
        
        # FCAS prices (8)
        for service in ['RAISEREG', 'LOWERREG', 'RAISE6SEC', 'LOWER6SEC',
                       'RAISE60SEC', 'LOWER60SEC', 'RAISE5MIN', 'LOWER5MIN']:
            obs.append(market_data.get(f'FCAS_{service}_normalized', 0))
        
        # Generation mix (2)
        obs.extend([
            market_data.get('GEN_solar_pct', 0),
            market_data.get('GEN_wind_pct', 0),
        ])
        
        # Battery state (1)
        obs.append(self.battery_soc / self.battery_capacity)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self, market_data, actual_power: float, actual_energy: float,
                         fcas_raise_bid: float, fcas_lower_bid: float,
                         old_soc: float, new_soc: float) -> float:
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
            # Calculate available capacity for FCAS
            fcas_raise_capacity = min(fcas_raise_bid * self.battery_capacity,
                                     self.battery_capacity - self.battery_soc)
            fcas_lower_capacity = min(fcas_lower_bid * self.battery_capacity,
                                     self.battery_soc)
            
            # Get FCAS prices ($/MW/h)
            raisereg_price = market_data.get('FCAS_RAISEREG', 0)
            lowerreg_price = market_data.get('FCAS_LOWERREG', 0)
            
            # Calculate revenue for this step
            fcas_revenue = (fcas_raise_capacity * raisereg_price * self.step_duration +
                           fcas_lower_capacity * lowerreg_price * self.step_duration)
        
        # Battery degradation cost (simplified)
        # Use depth of discharge and cycle count
        dod = abs(new_soc - old_soc) / self.battery_capacity
        degradation_cost = dod * self.battery_life_cost * 0.0001  # Simplified model
        
        # SOC violation penalty
        soc_penalty = 0.0
        if self.battery_soc < 0.1 * self.battery_capacity or self.battery_soc > 0.9 * self.battery_capacity:
            soc_penalty = -10.0  # Small penalty for operating at extremes
        
        # Total reward
        reward = energy_revenue + fcas_revenue - degradation_cost + soc_penalty
        
        # Track totals
        self.total_revenue += energy_revenue + fcas_revenue
        self.total_degradation_cost += degradation_cost
        
        # Normalize reward to more manageable scale
        # Typical revenue per step might be in hundreds of dollars
        # Normalize to roughly [-1, 1] range
        normalized_reward = reward / 1000.0
        
        return normalized_reward
    
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
