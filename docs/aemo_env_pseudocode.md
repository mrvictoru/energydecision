# AEMO Environment Pseudocode

This document outlines the design for a future gym environment that incorporates AEMO market data for energy trading and FCAS participation. The environment extends the existing `SolarBatteryEnv` to include wholesale market dynamics.

## Overview

The AEMO environment simulates a battery energy storage system (BESS) that participates in both the National Electricity Market (NEM) energy spot market and the Frequency Control Ancillary Services (FCAS) markets. The agent must optimize battery dispatch across multiple revenue streams while managing degradation and operational constraints.

## Environment Name: `AEMOBatteryTradingEnv`

Extends: `SolarBatteryEnv`

## 1. Observation Space Additions

Building on the existing observation space from `SolarBatteryEnv`, add the following components:

### Current Market State (Real-time, 5-minute resolution)

```python
# Energy Market
- current_energy_price       # Current Regional Reference Price ($/MWh), normalized [0, 1]
                             # Note: Energy prices can go negative in renewable-heavy grids.
                             # Consider clipping to [min_observed, max_observed] before normalization
                             # or using symmetric range [-1, 1] to handle negative prices
- forecast_energy_price_1h   # Forecast price 1 hour ahead, normalized [0, 1]
- forecast_energy_price_4h   # Forecast price 4 hours ahead, normalized [0, 1]
- regional_demand            # Total regional demand (MW), normalized [0, 1]

# FCAS Market Prices (8 services)
- raisereg_price            # Regulation raise ($/MW/h), normalized [0, 1]
- lowerreg_price            # Regulation lower ($/MW/h), normalized [0, 1]
- raise6sec_price           # Fast raise ($/MW/h), normalized [0, 1]
- lower6sec_price           # Fast lower ($/MW/h), normalized [0, 1]
- raise60sec_price          # Slow raise ($/MW/h), normalized [0, 1]
- lower60sec_price          # Slow lower ($/MW/h), normalized [0, 1]
- raise5min_price           # Delayed raise ($/MW/h), normalized [0, 1]
- lower5min_price           # Delayed lower ($/MW/h), normalized [0, 1]

# Generation Mix (indicates price volatility and renewable penetration)
- solar_generation_pct      # Solar as % of total generation, [0, 1]
- wind_generation_pct       # Wind as % of total generation, [0, 1]
- renewable_penetration     # (Solar + Wind) / Total, [0, 1]

# Time Features (existing but noted for completeness)
- hour_sin, hour_cos        # Cyclical time encoding
- day_sin, day_cos          # Cyclical day encoding
- is_peak_period           # Boolean indicator for peak demand times

# Battery State (existing, from SolarBatteryEnv)
- battery_soc              # State of charge, normalized [0, 1]
- battery_degradation_cost # Cumulative degradation, normalized [0, 1]
```

### Historical Context Window (optional, for recurrent policies)

For advanced agents (e.g., LSTM, Transformer-based), include a sliding window of past observations:

```python
# Example: Last 12 time steps (1 hour of 5-minute data)
- price_history[12]           # Historical energy prices
- fcas_price_history[12, 8]   # Historical FCAS prices for all services
- demand_history[12]          # Historical demand
```

## 2. Action Space Additions

Extend the action space from simple battery charge/discharge to include FCAS market participation.

### Option A: Discrete Action Space (Simplified)

```python
# Action: Integer in [0, N_ACTIONS-1]
# Example with N_ACTIONS = 7:
0: Charge at max rate (energy arbitrage)
1: Charge at half rate
2: Idle (no dispatch)
3: Discharge at half rate (energy arbitrage)
4: Discharge at max rate (energy arbitrage)
5: Reserve for FCAS raise (hold capacity, bid into raise markets)
6: Reserve for FCAS lower (hold capacity, bid into lower markets)
```

### Option B: Continuous Multi-Dimensional Action Space (Recommended)

```python
# Action: Vector of continuous values, each in [-1, 1] or [0, 1]

action = [
    battery_dispatch,        # [-1, 1]: -1=max discharge, 0=idle, +1=max charge
    fcas_raise_bid,         # [0, 1]: Fraction of capacity to bid for raise regulation
    fcas_lower_bid,         # [0, 1]: Fraction of capacity to bid for lower regulation
    energy_bid_aggressive,  # [0, 1]: Aggressiveness of energy market bidding
]

# Constraints:
# - battery_dispatch and fcas_bids must respect physical limits
# - Total FCAS commitments cannot exceed battery headroom
# - If providing FCAS, battery dispatch range is restricted
```

### Action Interpretation Logic

```python
def step(action):
    battery_dispatch = action[0]  # Normalized in [-1, 1]
    fcas_raise_bid = action[1]    # Fraction in [0, 1]
    fcas_lower_bid = action[2]    # Fraction in [0, 1]
    
    # Convert normalized action to power (kW)
    power_command = battery_dispatch * max_battery_flow
    
    # Calculate FCAS enablement (MW)
    max_fcas_raise = min(
        fcas_raise_bid * battery_capacity,
        battery_capacity - battery_soc  # Headroom for charging
    )
    max_fcas_lower = min(
        fcas_lower_bid * battery_capacity,
        battery_soc  # Available energy for discharging
    )
    
    # Restrict battery dispatch to respect FCAS commitments
    # If FCAS committed, must maintain headroom/available energy
    constrained_power = constrain_dispatch(
        power_command, 
        max_fcas_raise, 
        max_fcas_lower,
        battery_soc,
        battery_capacity
    )
    
    # Execute dispatch and accumulate revenue
    ...
```

## 3. Reward Function Adjustments

The reward function must account for multiple revenue streams and costs.

### Components

```python
def calculate_reward(state, action, next_state):
    # 1. Energy Market Revenue
    if battery_discharge > 0:
        energy_revenue = battery_discharge * current_energy_price * step_duration
    else:
        energy_revenue = 0
    
    # 2. Energy Market Cost (when charging)
    if battery_charge > 0:
        energy_cost = battery_charge * current_energy_price * step_duration
    else:
        energy_cost = 0
    
    # 3. FCAS Revenue (for capacity reservation)
    fcas_raise_revenue = fcas_raise_enablement * raisereg_price * step_duration
    fcas_lower_revenue = fcas_lower_enablement * lowerreg_price * step_duration
    
    # 4. Battery Degradation Cost (existing mechanism)
    degradation_cost = calculate_degradation(
        battery_soc_before,
        battery_soc_after,
        charge_rate,
        discharge_rate
    )
    
    # 5. Penalties
    # Penalty for violating SOC constraints
    soc_violation_penalty = 0
    if battery_soc < min_soc or battery_soc > max_soc:
        soc_violation_penalty = VIOLATION_PENALTY
    
    # Penalty for failing to deliver FCAS when called upon (simulation)
    # (Simplified: assume occasional random FCAS calls based on market need)
    fcas_delivery_penalty = 0
    if fcas_called and not can_deliver(fcas_raise_enablement, battery_soc):
        fcas_delivery_penalty = -100  # Financial penalty for non-delivery
    
    # Total reward
    reward = (
        energy_revenue
        - energy_cost
        + fcas_raise_revenue
        + fcas_lower_revenue
        - degradation_cost
        + soc_violation_penalty
        + fcas_delivery_penalty
    )
    
    return reward
```

### Reward Normalization

```python
# Similar to existing SolarBatteryEnv, normalize reward to [-1, 1] range
# Use historical statistics to determine scaling factors
max_possible_revenue_per_step = (
    max_discharge_power * max_energy_price * step_duration +
    battery_capacity * max_fcas_price * step_duration
)

normalized_reward = reward / max_possible_revenue_per_step
```

## 4. Data Alignment and Missing Data Handling

### Time Resolution

- **AEMO Data**: 5-minute dispatch intervals (288 intervals per day)
- **SolarBatteryEnv**: Configurable step duration (typically 30 minutes)

**Alignment Strategy:**

```python
# Option 1: Downsample AEMO data to match env step duration
# Average/aggregate 5-minute data into 30-minute intervals

def align_aemo_data(aemo_df, env_step_duration_hours):
    """
    Aggregate 5-minute AEMO data to match environment step duration.
    
    Args:
        aemo_df: AEMO data with 5-minute intervals
        env_step_duration_hours: Environment step duration (e.g., 0.5 for 30 min)
    
    Returns:
        Aggregated dataframe aligned to env time resolution
    """
    resample_freq = f"{int(env_step_duration_hours * 60)}min"
    
    aligned_df = aemo_df.set_index('SETTLEMENTDATE').resample(resample_freq).agg({
        'RRP': 'mean',              # Average price over interval
        'TOTALDEMAND': 'mean',      # Average demand
        'FCAS_PRICE': 'mean',       # Average FCAS prices
        'GENERATION': 'sum',        # Total generation
    }).reset_index()
    
    return aligned_df

# Option 2: Use 5-minute resolution in environment
# Modify SolarBatteryEnv to support finer time steps
# This increases computational cost but improves realism
```

### Missing Data Handling

```python
def handle_missing_data(df, method='interpolate'):
    """
    Handle missing values in AEMO data.
    
    Methods:
        - 'interpolate': Linear interpolation between known values
        - 'forward_fill': Carry forward last known value
        - 'drop': Skip episodes with missing data
        - 'typical': Use typical values for time-of-day
    """
    if method == 'interpolate':
        # Linear interpolation for prices and demand
        df = df.interpolate(method='linear', limit_direction='both')
    
    elif method == 'forward_fill':
        # Forward fill up to a maximum gap
        df = df.fillna(method='ffill', limit=12)  # Max 1 hour gap
    
    elif method == 'typical':
        # Use historical typical values by hour-of-day
        for col in ['RRP', 'TOTALDEMAND']:
            typical_values = df.groupby(df.index.hour)[col].transform('mean')
            df[col].fillna(typical_values, inplace=True)
    
    # Drop any remaining NaN rows
    df = df.dropna()
    
    return df
```

### Data Preprocessing Pipeline

```python
class AEMODataPipeline:
    """
    Preprocessing pipeline for AEMO data to be used in RL environment.
    """
    
    def __init__(self, 
                 cache_dir='data/aemo',
                 step_duration_hours=0.5,
                 missing_data_method='interpolate'):
        self.cache_dir = cache_dir
        self.step_duration_hours = step_duration_hours
        self.missing_data_method = missing_data_method
        
        # Normalization statistics (learned from training data)
        self.stats = {
            'RRP': {'mean': 80.0, 'std': 50.0, 'min': 0.0, 'max': 500.0},
            'FCAS_PRICE': {'mean': 15.0, 'std': 10.0, 'min': 0.0, 'max': 100.0},
            'DEMAND': {'mean': 7000.0, 'std': 2000.0, 'min': 4000.0, 'max': 12000.0},
        }
    
    def fetch_and_preprocess(self, start_date, end_date, region='NSW1'):
        """
        Fetch AEMO data and preprocess for RL environment.
        """
        # 1. Fetch raw data
        data = fetch_aemo_data_bundle(
            start_date=start_date,
            end_date=end_date,
            region=region,
            cache_dir=self.cache_dir
        )
        
        # 2. Align time resolution
        prices = align_aemo_data(data['prices'], self.step_duration_hours)
        fcas = align_aemo_data(data['fcas'], self.step_duration_hours)
        generation = align_aemo_data(data['generation'], self.step_duration_hours)
        
        # 3. Handle missing data
        prices = handle_missing_data(prices, method=self.missing_data_method)
        fcas = handle_missing_data(fcas, method=self.missing_data_method)
        generation = handle_missing_data(generation, method=self.missing_data_method)
        
        # 4. Merge datasets
        merged_df = self._merge_datasets(prices, fcas, generation)
        
        # 5. Normalize features
        normalized_df = self._normalize_features(merged_df)
        
        return normalized_df
    
    def _merge_datasets(self, prices, fcas, generation):
        # Join on timestamp
        # Pivot FCAS to wide format (one column per service)
        # Pivot generation to wide format (one column per fuel type)
        ...
    
    def _normalize_features(self, df):
        # Apply min-max or z-score normalization
        # Clip outliers to prevent extreme values
        ...
```

## 5. Episode Structure

```python
class AEMOBatteryTradingEnv(SolarBatteryEnv):
    """
    Gym environment for battery trading in AEMO markets.
    """
    
    def __init__(self, 
                 aemo_data_df,           # Preprocessed AEMO data
                 battery_capacity=10.0,   # MWh (larger scale for grid participation)
                 max_battery_flow=5.0,    # MW
                 init_battery_level=5.0,  # MWh
                 step_duration=0.5,       # hours (30 min aligned with AEMO)
                 **kwargs):
        
        super().__init__(**kwargs)
        self.aemo_data = aemo_data_df
        self.battery_capacity = battery_capacity
        self.max_battery_flow = max_battery_flow
        # ... initialize additional state variables
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        Randomly sample a start date for episode diversity.
        """
        super().reset(seed=seed)
        
        # Sample episode start (avoid first/last few days for forecast availability)
        max_start = len(self.aemo_data) - self.max_step - 12
        self.episode_start_idx = self.np_random.integers(12, max_start)
        self.current_step = 0
        
        # Reset battery state
        self.battery_soc = self.init_battery_level
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # 1. Parse action
        battery_dispatch, fcas_raise_bid, fcas_lower_bid = self._parse_action(action)
        
        # 2. Get current market state
        current_idx = self.episode_start_idx + self.current_step
        market_data = self.aemo_data.iloc[current_idx]
        
        # 3. Calculate constrained dispatch
        actual_dispatch = self._constrain_dispatch(
            battery_dispatch, fcas_raise_bid, fcas_lower_bid
        )
        
        # 4. Update battery state
        self._update_battery_state(actual_dispatch)
        
        # 5. Calculate reward
        reward = self._calculate_reward(
            market_data, actual_dispatch, fcas_raise_bid, fcas_lower_bid
        )
        
        # 6. Check termination
        self.current_step += 1
        terminated = (self.current_step >= self.max_step)
        truncated = False
        
        # 7. Get next observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Construct observation vector from current state.
        """
        current_idx = self.episode_start_idx + self.current_step
        market_data = self.aemo_data.iloc[current_idx]
        
        obs = np.array([
            # Time features
            market_data['hour_sin'],
            market_data['hour_cos'],
            market_data['day_sin'],
            market_data['day_cos'],
            
            # Energy market
            market_data['RRP_normalized'],
            market_data['RRP_forecast_1h'],
            market_data['RRP_forecast_4h'],
            market_data['TOTALDEMAND_normalized'],
            
            # FCAS prices
            market_data['RAISEREG_price'],
            market_data['LOWERREG_price'],
            # ... other FCAS services
            
            # Generation mix
            market_data['solar_pct'],
            market_data['wind_pct'],
            market_data['renewable_pct'],
            
            # Battery state
            self.battery_soc / self.battery_capacity,  # Normalized SOC
            self.battery_degradation_cost / self.battery_life_cost,
        ])
        
        return obs
```

## 6. Training Considerations

### Curriculum Learning

```python
# Stage 1: Energy arbitrage only (no FCAS)
# - Simpler action space: just charge/discharge decisions
# - Learn basic price patterns and battery constraints

# Stage 2: Add FCAS regulation services
# - Introduce FCAS bidding to action space
# - Learn to balance energy trading with FCAS revenue

# Stage 3: Full multi-market optimization
# - All FCAS services available
# - Complex interactions and constraints
```

### Exploration Strategies

```python
# Challenge: FCAS markets have lower magnitude but steadier revenue
# Risk: Agent may ignore FCAS due to higher variance in energy market

# Solution 1: Shaped reward with FCAS bonus
reward = energy_revenue + alpha * fcas_revenue - degradation

# Solution 2: Separate critics for energy and FCAS (multi-objective RL)

# Solution 3: Curriculum with FCAS-only episodes
```

### Offline RL Considerations

```python
# Leverage historical data:
# 1. Collect expert demonstrations (rule-based strategies)
# 2. Use decision transformer or CQL for offline training
# 3. Fine-tune with online RL if safe to do so

# Expert strategies:
# - "Peak shaver": Discharge during high price periods
# - "FCAS specialist": Primarily bid into FCAS markets
# - "Solar follower": Charge when solar generation high (low prices)
```

## 7. Validation and Testing

### Backtesting Framework

```python
def backtest_agent(agent, aemo_data, start_date, end_date):
    """
    Backtest trained agent on historical AEMO data.
    """
    env = AEMOBatteryTradingEnv(aemo_data)
    
    total_revenue = 0
    total_degradation = 0
    
    obs, info = env.reset()
    done = False
    
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_revenue += info['energy_revenue'] + info['fcas_revenue']
        total_degradation += info['degradation_cost']
    
    net_profit = total_revenue - total_degradation
    
    return {
        'total_revenue': total_revenue,
        'energy_revenue': info['total_energy_revenue'],
        'fcas_revenue': info['total_fcas_revenue'],
        'degradation_cost': total_degradation,
        'net_profit': net_profit,
        'roi': net_profit / env.battery_life_cost,
    }
```

### Performance Metrics

```python
# 1. Financial Metrics
- Total revenue (energy + FCAS)
- Net profit (revenue - costs)
- Return on investment (ROI)
- Revenue breakdown (energy vs. FCAS %)

# 2. Operational Metrics
- Cycle count (full depth-of-discharge equivalents)
- Average state of charge
- FCAS delivery rate (when called upon)
- Constraint violations

# 3. Market Metrics
- Correlation with price patterns
- Response to renewable penetration
- Peak vs. off-peak dispatch ratio
```

## 8. Implementation Roadmap

### Phase 1: Data Infrastructure (Current PR)
- ✅ AEMO data fetching module (`src/aemo_data.py`)
- ✅ Data exploration notebook (`test_aemo_data.ipynb`)
- ✅ Pseudocode documentation (this file)

### Phase 2: Basic Environment
- Extend `SolarBatteryEnv` to include AEMO price data
- Simplified action space (energy arbitrage only)
- Basic reward function (energy revenue - costs)
- Test with rule-based baselines

### Phase 3: FCAS Integration
- Add FCAS prices to observation space
- Expand action space to include FCAS bidding
- Implement FCAS revenue calculations
- Validate against known strategies

### Phase 4: Advanced Features
- Forecast integration (price and generation predictions)
- Multi-region support
- Battery degradation with C-rate impacts
- Realistic operational constraints (ramp rates, response times)

### Phase 5: Training and Evaluation
- Collect diverse behavioral policies
- Train offline RL agents (Decision Transformer)
- Benchmark against baselines
- Backtesting on held-out historical periods

## 9. References and Resources

### AEMO Documentation
- [AEMO Market Data](https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem)
- [FCAS Services Overview](https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/system-operations/ancillary-services)
- [MMS Data Model](https://nemweb.com.au/Reports/Current/MMSDataModelReport/)

### Related Implementations
- [opennem/nemweb](https://github.com/opennem/nemweb) - AEMO data parser
- [UNSW-CEEM/NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS) - Historical NEM data
- [energy-market-deep-learning](https://github.com/sustainable-computing/energy-market-deep-learning) - ML for energy markets

### Papers
- "Optimal Battery Trading in Frequency Regulation Markets" (Jia et al., 2020)
- "Deep Reinforcement Learning for Energy Storage Arbitrage in Australia's NEM" (Perera et al., 2021)
- "Multi-Market Participation of Battery Storage in Australia" (Nguyen et al., 2022)

---

**Note**: This pseudocode is intentionally high-level to guide implementation. Actual code will need careful tuning of normalization constants, reward shaping, and constraint handling based on empirical testing.
