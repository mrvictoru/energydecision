import unittest
import numpy as np
import polars as pl
from EnergySimEnv import SolarBatteryEnv

class TestSolarBatteryEnv(unittest.TestCase):
    def setUp(self):
        # Minimal DataFrame for 2 time steps
        data = {
            'Time': ['2025-01-01T00:00', '2025-01-01T00:30'],
            'SolarGen': [2.0, 0.0],  # kWh
            'HouseLoad': [1.0, 3.0],  # kWh
            'ImportEnergyPrice': [0.3, 0.3],  # $/kWh
            'ExportEnergyPrice': [0.05, 0.05],  # $/kWh
        }
        df = pl.DataFrame(data)
        self.env = SolarBatteryEnv(df, battery_capacity=4.0, max_battery_flow=2.0, max_grid_flow=4.0, init_battery_level=2.0, max_step=2, battery_life_cost=1000.0)
        self.env.reset()

    def test_import_price_selection(self):
        # Action: Discharge battery to cover house load (should result in grid import)
        action = np.array([-1.0])  # full discharge
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertGreaterEqual(info['grid_energy'], 0)  # importing from grid
        self.assertEqual(info['energy_price'], 0.3)  # uses import price

    def test_export_price_selection(self):
        # Set up for exporting energy (surplus solar)
        # Next step: SolarGen=0, HouseLoad=3, battery level will be 0 after discharge, so action must be 0 (no discharge), so grid will import.
        # Let's manually charge at second step to simulate export (if possible).
        self.env.reset()
        # First, charge battery fully
        self.env.battery_level = self.env.battery_capacity
        # Set solar high and load low for export
        self.env.df = self.env.df.with_columns([
            pl.lit(10.0).alias("SolarGen"),
            pl.lit(1.0).alias("HouseLoad"),
        ])
        action = np.array([0.0])  # no battery flow, all surplus solar should export
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertLess(info['grid_energy'], 0)  # exporting to grid
        self.assertEqual(info['energy_price'], 0.05)  # uses export price

    def test_degradation_cost_sensible(self):
        # Discharge action: check degradation cost is positive and small
        self.env.reset()
        action = np.array([-1.0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        deg_cost = info['deg_cost']
        self.assertGreaterEqual(deg_cost, 0)
        # Should be much less than battery_life_cost per step
        self.assertLess(deg_cost, self.env.battery_life_cost * 0.01)

    def test_reward_sign_and_magnitude(self):
        # Charge battery: grid cost should be negative (cost), reward should be negative (cost + degradation)
        self.env.reset()
        action = np.array([1.0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertLessEqual(info['grid_reward'], 0)
        self.assertLessEqual(reward, 0)

    def test_dynamic_correction_factor_behavior(self):
        # Simulate several steps to trigger dynamic correction
        self.env.reset()
        # Manually set init_correction_steps to [1] for quick test
        self.env.init_correction_steps = [1]
        for i in range(2):
            obs, reward, terminated, truncated, info = self.env.step(np.array([-0.5]))
        # After step 1, correction_factor should be updated
        self.assertTrue('correction_factor' in info)
        self.assertGreaterEqual(info['correction_factor'], 0)
        self.assertLessEqual(info['correction_factor'], 10)

    def test_violation_penalty(self):
        # Force grid_energy violation by using excessive battery action
        self.env.reset()
        # Set max_grid_energy low for test
        self.env.max_grid_energy = 0.1
        action = np.array([1.0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertTrue(reward <= -1000)  # Should get heavy negative reward
        self.assertTrue(info['energy_conservation_violation'] or abs(info['grid_energy']) > self.env.max_grid_energy)

if __name__ == "__main__":
    unittest.main()
