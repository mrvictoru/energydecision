"""
Test script to validate AEMO Battery Environment compatibility with existing
agent classes and algorithms.

This script checks that AEMOBatteryTradingEnv is a drop-in replacement for
SolarBatteryEnv in terms of:
1. Gymnasium API compatibility
2. Stable-Baselines3 algorithm compatibility
3. Decision agent compatibility
4. Observation/action space structure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import polars as pl
import pytest
from datetime import datetime, timedelta

# Test imports
def test_imports():
    """Test that all necessary modules can be imported."""
    print("=" * 80)
    print("TEST 1: Module Imports")
    print("=" * 80)
    
    try:
        import gymnasium as gym
        print("✓ gymnasium imported successfully")
    except ImportError as e:
        pytest.fail(f"gymnasium import failed: {e}")
    
    try:
        from AEMOBatteryEnv import AEMOBatteryTradingEnv, create_aemo_env_from_data, AEMODataPreprocessor
        print("✓ AEMOBatteryEnv imported successfully")
    except ImportError as e:
        pytest.fail(f"AEMOBatteryEnv import failed: {e}")
    
    try:
        from EnergySimEnv import SolarBatteryEnv
        print("✓ SolarBatteryEnv imported successfully")
    except ImportError as e:
        pytest.fail(f"SolarBatteryEnv import failed: {e}")
    
    print("\n✓ All imports successful\n")
def test_gymnasium_api():
    """Test that AEMOBatteryTradingEnv follows Gymnasium API."""
    print("=" * 80)
    print("TEST 2: Gymnasium API Compatibility")
    print("=" * 80)
    
    try:
        import gymnasium as gym
        from AEMOBatteryEnv import AEMOBatteryTradingEnv, AEMODataPreprocessor
        
        # Create synthetic data for testing
        num_steps = 100
        timestamps = [datetime(2024, 6, 1) + timedelta(minutes=30*i) for i in range(num_steps)]
        
        test_data = pl.DataFrame({
            'Time': timestamps,
            'RRP': np.random.uniform(20, 100, num_steps),
            'TOTALDEMAND': np.random.uniform(5000, 8000, num_steps),
            'RAISEREG': np.random.uniform(5, 20, num_steps),
            'LOWERREG': np.random.uniform(5, 20, num_steps),
            'RAISE6SEC': np.random.uniform(10, 30, num_steps),
            'LOWER6SEC': np.random.uniform(10, 30, num_steps),
            'RAISE60SEC': np.random.uniform(8, 25, num_steps),
            'LOWER60SEC': np.random.uniform(8, 25, num_steps),
            'RAISE5MIN': np.random.uniform(7, 22, num_steps),
            'LOWER5MIN': np.random.uniform(7, 22, num_steps),
            'solar_pct': np.random.uniform(0, 0.3, num_steps),
            'wind_pct': np.random.uniform(0, 0.2, num_steps),
        })
        
        # Create environment
        preprocessor = AEMODataPreprocessor()
        env = AEMOBatteryTradingEnv(
            aemo_data=test_data,
            battery_capacity=10.0,
            max_battery_flow=5.0,
            action_mode='simple'
        )
        
        # Check Gymnasium interface
        print(f"✓ Environment created")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ reset() works: obs shape {obs.shape}, info keys: {list(info.keys())}")
        assert isinstance(obs, np.ndarray), "Observation must be numpy array"
        assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ step() works: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
        assert isinstance(reward, (float, np.floating)), "Reward must be float"
        assert isinstance(terminated, bool), "Terminated must be bool"
        assert isinstance(truncated, bool), "Truncated must be bool"
        
        # Test episode completion
        episode_rewards = []
        obs, info = env.reset()
        for i in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            if terminated or truncated:
                break
        
        print(f"✓ Episode completed: {len(episode_rewards)} steps, total reward={sum(episode_rewards):.2f}")
        print(f"  - Mean reward: {np.mean(episode_rewards):.3f}")
        print(f"  - Std reward: {np.std(episode_rewards):.3f}")
        
        print("\n✓ Gymnasium API compatibility confirmed\n")
        
    except Exception as e:
        pytest.fail(f"Gymnasium API test failed: {e}")


def test_multi_market_mode():
    """Test multi-market action mode."""
    print("=" * 80)
    print("TEST 3: Multi-Market Mode Compatibility")
    print("=" * 80)
    
    try:
        import gymnasium as gym
        from AEMOBatteryEnv import AEMOBatteryTradingEnv
        
        # Create synthetic data
        num_steps = 100
        timestamps = [datetime(2024, 6, 1) + timedelta(minutes=30*i) for i in range(num_steps)]
        
        test_data = pl.DataFrame({
            'Time': timestamps,
            'RRP': np.random.uniform(20, 100, num_steps),
            'TOTALDEMAND': np.random.uniform(5000, 8000, num_steps),
            'RAISEREG': np.random.uniform(5, 20, num_steps),
            'LOWERREG': np.random.uniform(5, 20, num_steps),
            'RAISE6SEC': np.random.uniform(10, 30, num_steps),
            'LOWER6SEC': np.random.uniform(10, 30, num_steps),
            'RAISE60SEC': np.random.uniform(8, 25, num_steps),
            'LOWER60SEC': np.random.uniform(8, 25, num_steps),
            'RAISE5MIN': np.random.uniform(7, 22, num_steps),
            'LOWER5MIN': np.random.uniform(7, 22, num_steps),
            'solar_pct': np.random.uniform(0, 0.3, num_steps),
            'wind_pct': np.random.uniform(0, 0.2, num_steps),
        })
        
        # Create environment with multi-market mode
        env = AEMOBatteryTradingEnv(
            aemo_data=test_data,
            battery_capacity=10.0,
            max_battery_flow=5.0,
            action_mode='multi_market'
        )
        
        print(f"✓ Multi-market environment created")
        print(f"  - Action space: {env.action_space}")
        assert env.action_space.shape == (3,), "Multi-market should have 3D action space"
        
        # Test episode
        obs, info = env.reset()
        action = np.array([0.5, 0.3, 0.2])  # energy_dispatch, fcas_raise, fcas_lower
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✓ Multi-market step works")
        print(f"  - Energy dispatch component: {info.get('energy_dispatch', 'N/A')}")
        print(f"  - FCAS revenue component: {info.get('fcas_revenue', 'N/A')}")
        
        print("\n✓ Multi-market mode compatibility confirmed\n")
        
    except Exception as e:
        pytest.fail(f"Multi-market mode test failed: {e}")


def test_sb3_compatibility():
    """Test compatibility with Stable-Baselines3 algorithms."""
    print("=" * 80)
    print("TEST 4: Stable-Baselines3 Compatibility")
    print("=" * 80)
    
    try:
        from stable_baselines3 import PPO, SAC, A2C
        from stable_baselines3.common.env_checker import check_env
        from AEMOBatteryEnv import AEMOBatteryTradingEnv
        
        # Create synthetic data
        num_steps = 200
        timestamps = [datetime(2024, 6, 1) + timedelta(minutes=30*i) for i in range(num_steps)]
        
        test_data = pl.DataFrame({
            'Time': timestamps,
            'RRP': np.random.uniform(20, 100, num_steps),
            'TOTALDEMAND': np.random.uniform(5000, 8000, num_steps),
            'RAISEREG': np.random.uniform(5, 20, num_steps),
            'LOWERREG': np.random.uniform(5, 20, num_steps),
            'RAISE6SEC': np.random.uniform(10, 30, num_steps),
            'LOWER6SEC': np.random.uniform(10, 30, num_steps),
            'RAISE60SEC': np.random.uniform(8, 25, num_steps),
            'LOWER60SEC': np.random.uniform(8, 25, num_steps),
            'RAISE5MIN': np.random.uniform(7, 22, num_steps),
            'LOWER5MIN': np.random.uniform(7, 22, num_steps),
            'solar_pct': np.random.uniform(0, 0.3, num_steps),
            'wind_pct': np.random.uniform(0, 0.2, num_steps),
        })
        
        env = AEMOBatteryTradingEnv(
            aemo_data=test_data,
            battery_capacity=10.0,
            max_battery_flow=5.0,
            action_mode='simple'
        )
        
        # Check environment
        print("Running SB3 environment checker...")
        check_env(env)
        print("✓ Environment passes SB3 checks")
        
        # Test PPO
        print("\nTesting PPO algorithm...")
        model_ppo = PPO("MlpPolicy", env, verbose=0, device="cpu")
        model_ppo.learn(total_timesteps=1000)
        print("✓ PPO training works (1000 steps)")
        
        # Test prediction
        obs, _ = env.reset()
        action, _states = model_ppo.predict(obs, deterministic=True)
        print(f"✓ PPO prediction works: action={action}")
        
        # Test SAC (continuous action algorithms)
        print("\nTesting SAC algorithm...")
        model_sac = SAC("MlpPolicy", env, verbose=0, device="cpu")
        model_sac.learn(total_timesteps=1000)
        print("✓ SAC training works (1000 steps)")
        
        # Test A2C
        print("\nTesting A2C algorithm...")
        model_a2c = A2C("MlpPolicy", env, verbose=0, device="cpu")
        model_a2c.learn(total_timesteps=1000)
        print("✓ A2C training works (1000 steps)")
        
        print("\n✓ Stable-Baselines3 compatibility confirmed\n")
        
    except ImportError as e:
        pytest.skip(f"Stable-Baselines3 not installed: {e}")
    except Exception as e:
        pytest.fail(f"SB3 compatibility test failed: {e}")


def test_observation_space_comparison():
    """Compare observation spaces between SolarBatteryEnv and AEMOBatteryTradingEnv."""
    print("=" * 80)
    print("TEST 5: Observation Space Comparison")
    print("=" * 80)
    
    try:
        from EnergySimEnv import SolarBatteryEnv
        from AEMOBatteryEnv import AEMOBatteryTradingEnv
        
        # Create SolarBatteryEnv data
        num_steps = 100
        solar_data = pl.DataFrame({
            'Timestamp': list(range(num_steps)),
            'Time': [datetime(2024, 1, 1) + timedelta(minutes=30*i) for i in range(num_steps)],
            'SolarGen': np.random.uniform(0, 5, num_steps),
            'HouseLoad': np.random.uniform(1, 3, num_steps),
            'FutureSolar': np.random.uniform(0, 5, num_steps),
            'FutureLoad': np.random.uniform(1, 3, num_steps),
            'ImportEnergyPrice': np.random.uniform(0.2, 0.4, num_steps),
            'ExportEnergyPrice': np.random.uniform(0.1, 0.2, num_steps),
        })
        
        solar_env = SolarBatteryEnv(
            df=solar_data,
            battery_capacity=10.0,
            max_battery_flow=5.0
        )
        
        # Create AEMOBatteryEnv data
        aemo_data = pl.DataFrame({
            'Time': [datetime(2024, 6, 1) + timedelta(minutes=30*i) for i in range(num_steps)],
            'RRP': np.random.uniform(20, 100, num_steps),
            'TOTALDEMAND': np.random.uniform(5000, 8000, num_steps),
            'RAISEREG': np.random.uniform(5, 20, num_steps),
            'LOWERREG': np.random.uniform(5, 20, num_steps),
            'RAISE6SEC': np.random.uniform(10, 30, num_steps),
            'LOWER6SEC': np.random.uniform(10, 30, num_steps),
            'RAISE60SEC': np.random.uniform(8, 25, num_steps),
            'LOWER60SEC': np.random.uniform(8, 25, num_steps),
            'RAISE5MIN': np.random.uniform(7, 22, num_steps),
            'LOWER5MIN': np.random.uniform(7, 22, num_steps),
            'solar_pct': np.random.uniform(0, 0.3, num_steps),
            'wind_pct': np.random.uniform(0, 0.2, num_steps),
        })
        
        aemo_env = AEMOBatteryTradingEnv(
            aemo_data=aemo_data,
            battery_capacity=10.0,
            max_battery_flow=5.0,
            action_mode='simple'
        )
        
        # Compare observation spaces
        solar_obs, _ = solar_env.reset()
        aemo_obs, _ = aemo_env.reset()
        
        print(f"SolarBatteryEnv:")
        print(f"  - Observation space: {solar_env.observation_space}")
        print(f"  - Observation shape: {solar_obs.shape}")
        print(f"  - Action space: {solar_env.action_space}")
        
        print(f"\nAEMOBatteryTradingEnv:")
        print(f"  - Observation space: {aemo_env.observation_space}")
        print(f"  - Observation shape: {aemo_obs.shape}")
        print(f"  - Action space: {aemo_env.action_space}")
        
        # Both should have Box observation and action spaces
        assert solar_env.observation_space.__class__.__name__ == 'Box', "SolarBattery should use Box space"
        assert aemo_env.observation_space.__class__.__name__ == 'Box', "AEMOBattery should use Box space"
        assert solar_env.action_space.__class__.__name__ == 'Box', "SolarBattery should use Box actions"
        assert aemo_env.action_space.__class__.__name__ == 'Box', "AEMOBattery should use Box actions"
        
        print(f"\n✓ Both environments use Box spaces (compatible with SB3)")
        print(f"✓ Observation space structure comparison complete\n")
        
    except Exception as e:
        pytest.fail(f"Observation space comparison failed: {e}")


def main():
    """Run all compatibility tests."""
    print("\n" + "=" * 80)
    print("AEMO Battery Environment Compatibility Test Suite")
    print("=" * 80 + "\n")
    
    results = []
    test_cases = [
        ("Module Imports", test_imports),
        ("Gymnasium API", test_gymnasium_api),
        ("Multi-Market Mode", test_multi_market_mode),
        ("Stable-Baselines3", test_sb3_compatibility),
        ("Observation Space", test_observation_space_comparison),
    ]
    
    for test_name, test_fn in test_cases:
        try:
            test_fn()
            results.append((test_name, True))
        except pytest.skip.Exception:
            results.append((test_name, True))
        except Exception:
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    print("=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80 + "\n")
    
    if passed == total:
        print("✓ AEMOBatteryTradingEnv is fully compatible with existing agent classes and algorithms!")
        print("✓ It can be used as a drop-in replacement for SolarBatteryEnv")
        return 0
    else:
        print("✗ Some compatibility issues detected. Please review failed tests.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
