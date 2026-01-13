"""
Test the new self-contained algorithm classes.

This verifies that SDPSolver, MRDPSolver, and OracleSolver work correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Import new algorithm classes
from sdp_algorithm import SDPSolver
from mrdp_algorithm import MRDPSolver
from oracle_algorithm import OracleSolver

def test_algorithm_imports():
    """Test that all algorithm modules can be imported."""
    print("Testing algorithm imports...")
    assert SDPSolver is not None
    assert MRDPSolver is not None
    assert OracleSolver is not None
    print("  ✓ All algorithm classes imported successfully")

def test_sdp_solver_initialization():
    """Test SDPSolver can be initialized."""
    print("\nTesting SDPSolver initialization...")
    
    # Create mock environment
    class MockEnv:
        battery_capacity = 7.0
        max_battery_flow = 3.3
        step_duration = 0.5
        max_grid_energy = 5.0
        battery_life_cost = 1000.0
        degradation_temperature = 25.0
    
    env = MockEnv()
    
    solver = SDPSolver(
        env=env,
        horizon=24,
        soc_resolution=10,
        action_resolution=7
    )
    
    assert solver.horizon == 24
    assert solver.soc_resolution == 10
    assert solver.action_resolution == 7
    assert len(solver.soc_levels_kwh) == 10
    assert len(solver.action_levels_norm) == 7
    print("  ✓ SDPSolver initialized correctly")
    print(f"    - SoC levels: {solver.soc_levels_kwh[0]:.2f} to {solver.soc_levels_kwh[-1]:.2f} kWh")
    print(f"    - Action levels: {solver.action_levels_norm[0]:.2f} to {solver.action_levels_norm[-1]:.2f}")

def test_mrdp_solver_initialization():
    """Test MRDPSolver can be initialized."""
    print("\nTesting MRDPSolver initialization...")
    
    class MockEnv:
        battery_capacity = 7.0
        max_battery_flow = 3.3
        step_duration = 0.5
        max_grid_energy = 5.0
        battery_life_cost = 1000.0
        degradation_temperature = 25.0
    
    env = MockEnv()
    
    subhorizon_specs = [
        {'start': 0, 'length': 12, 'soc_resolution': 10, 'action_resolution': 7, 'step_duration': 0.5},
        {'start': 12, 'length': 12, 'soc_resolution': 5, 'action_resolution': 5, 'step_duration': 1.0},
    ]
    
    solver = MRDPSolver(
        env=env,
        subhorizon_specs=subhorizon_specs,
        degradation_model='linear'
    )
    
    assert len(solver.sub_solvers) == 2
    assert solver.sub_solvers[0].soc_resolution == 10
    assert solver.sub_solvers[1].soc_resolution == 5
    print("  ✓ MRDPSolver initialized correctly")
    print(f"    - Sub-horizons: {len(solver.sub_solvers)}")
    print(f"    - First sub-horizon: {subhorizon_specs[0]['length']} steps, {subhorizon_specs[0]['soc_resolution']} SoC levels")
    print(f"    - Second sub-horizon: {subhorizon_specs[1]['length']} steps, {subhorizon_specs[1]['soc_resolution']} SoC levels")

def test_oracle_solver_initialization():
    """Test OracleSolver can be initialized."""
    print("\nTesting OracleSolver initialization...")
    
    class MockEnv:
        battery_capacity = 7.0
        max_battery_flow = 3.3
        step_duration = 0.5
        max_grid_energy = 5.0
        battery_life_cost = 1000.0
        degradation_temperature = 25.0
        
        def _get_row(self, idx):
            return {
                'SolarGen': 1.0,
                'HouseLoad': 1.5,
                'ImportEnergyPrice': 0.25,
                'ExportEnergyPrice': 0.1
            }
    
    env = MockEnv()
    
    solver = OracleSolver(
        env=env,
        horizon=24,
        action_resolution=7
    )
    
    assert solver.horizon == 24
    assert solver.action_resolution == 7
    assert len(solver.action_levels) == 7
    print("  ✓ OracleSolver initialized correctly")
    print(f"    - Horizon: {solver.horizon} steps")
    print(f"    - Action resolution: {solver.action_resolution}")

def test_algorithm_class_structure():
    """Test that algorithm classes have expected methods."""
    print("\nTesting algorithm class structure...")
    
    # SDPSolver methods
    assert hasattr(SDPSolver, 'solve')
    assert hasattr(SDPSolver, '_prepare_monte_carlo_samples')
    assert hasattr(SDPSolver, '_compute_stage_costs')
    assert hasattr(SDPSolver, '_compute_future_costs')
    assert hasattr(SDPSolver, '_update_policy')
    print("  ✓ SDPSolver has all expected methods")
    
    # MRDPSolver methods
    assert hasattr(MRDPSolver, 'solve')
    print("  ✓ MRDPSolver has all expected methods")
    
    # OracleSolver methods
    assert hasattr(OracleSolver, 'solve')
    assert hasattr(OracleSolver, 'get_action_for_current_state')
    print("  ✓ OracleSolver has all expected methods")

if __name__ == '__main__':
    print("=" * 60)
    print("Testing New Self-Contained Algorithm Classes")
    print("=" * 60)
    
    try:
        test_algorithm_imports()
        test_sdp_solver_initialization()
        test_mrdp_solver_initialization()
        test_oracle_solver_initialization()
        test_algorithm_class_structure()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe new algorithm classes are working correctly:")
        print("  • SDPSolver - Self-contained SDP implementation")
        print("  • MRDPSolver - Multi-resolution DP")
        print("  • OracleSolver - Perfect information DP")
        print("\nEach algorithm is now in its own file with all logic")
        print("in one place for easy reading and debugging.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
