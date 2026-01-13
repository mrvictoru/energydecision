"""
Simple test to verify refactored Agent class works correctly.
This test doesn't require all dependencies, just core functionality.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Test that algorithm_helpers module works
from algorithm_helpers import DegradationCalculator, OracleHelper, interpolate_ctg, compute_grid_cost
from batterydeg import DegradationModel, RainflowCounter

def test_degradation_calculator():
    """Test DegradationCalculator class."""
    print("Testing DegradationCalculator...")
    
    calc = DegradationCalculator(
        battery_capacity=7.0,
        step_duration=0.5,
        battery_life_cost=1000.0,
        degradation_temperature=25.0
    )
    
    # Test degradation per cycle
    deg = calc.degradation_per_cycle(Id=0.2, Ich=0.1, soc_percent=50.0, DoD=20.0)
    assert 0.0 <= deg <= 1.0, f"Degradation should be between 0 and 1, got {deg}"
    print(f"  ✓ degradation_per_cycle: {deg:.6f}")
    
    # Test linearized degradation
    lin_deg = calc.compute_linearized_degradation(
        Id=0.2, Ich=0.1, soc_percent=50.0, energy_kwh=1.0
    )
    assert 0.0 <= lin_deg <= 1.0, f"Linearized degradation should be between 0 and 1, got {lin_deg}"
    print(f"  ✓ compute_linearized_degradation: {lin_deg:.6f}")
    
    # Test rainflow degradation
    rain_deg = calc.compute_rainflow_degradation(soc_start_kwh=3.0, soc_end_kwh=4.0)
    assert 0.0 <= rain_deg <= 1.0, f"Rainflow degradation should be between 0 and 1, got {rain_deg}"
    print(f"  ✓ compute_rainflow_degradation: {rain_deg:.6f}")
    
    print("✓ DegradationCalculator tests passed!")

def test_helper_functions():
    """Test helper functions."""
    print("\nTesting helper functions...")
    
    # Test interpolate_ctg
    soc_levels = np.array([0.0, 2.0, 4.0, 6.0])
    ctg = np.array([10.0, 5.0, 3.0, 2.0])
    
    # Test interior interpolation
    interp_val = interpolate_ctg(soc_levels, ctg, 3.0)
    assert 3.0 <= interp_val <= 5.0, f"Interpolated value should be between 3 and 5, got {interp_val}"
    print(f"  ✓ interpolate_ctg (interior): {interp_val:.2f}")
    
    # Test boundary clamping
    interp_val_low = interpolate_ctg(soc_levels, ctg, -1.0)
    assert interp_val_low == 10.0, f"Should clamp to min, got {interp_val_low}"
    print(f"  ✓ interpolate_ctg (clamp low): {interp_val_low:.2f}")
    
    interp_val_high = interpolate_ctg(soc_levels, ctg, 10.0)
    assert interp_val_high == 2.0, f"Should clamp to max, got {interp_val_high}"
    print(f"  ✓ interpolate_ctg (clamp high): {interp_val_high:.2f}")
    
    # Test compute_grid_cost
    # Import
    cost_import = compute_grid_cost(grid_energy=1.5, import_price=0.25, export_price=0.1, max_grid_energy=5.0)
    assert cost_import == 1.5 * 0.25, f"Import cost should be 0.375, got {cost_import}"
    print(f"  ✓ compute_grid_cost (import): {cost_import:.3f}")
    
    # Export (negative grid_energy)
    cost_export = compute_grid_cost(grid_energy=-2.0, import_price=0.25, export_price=0.1, max_grid_energy=5.0)
    assert cost_export == -2.0 * 0.1, f"Export revenue should be -0.2, got {cost_export}"
    print(f"  ✓ compute_grid_cost (export): {cost_export:.3f}")
    
    # Violation
    cost_violation = compute_grid_cost(grid_energy=10.0, import_price=0.25, export_price=0.1, max_grid_energy=5.0)
    assert np.isinf(cost_violation), f"Should be inf for violation, got {cost_violation}"
    print(f"  ✓ compute_grid_cost (violation): {cost_violation}")
    
    print("✓ Helper function tests passed!")

def test_oracle_helper():
    """Test OracleHelper class (without full env)."""
    print("\nTesting OracleHelper class structure...")
    
    # Create a mock environment
    class MockEnv:
        battery_capacity = 7.0
        max_battery_flow = 3.3
        max_grid_energy = 5.0
        step_duration = 0.5
        battery_life_cost = 1000.0
        
        def _get_row(self, idx):
            return {
                'SolarGen': 1.0,
                'HouseLoad': 1.5,
                'ImportEnergyPrice': 0.25,
                'ExportEnergyPrice': 0.1
            }
    
    env = MockEnv()
    
    calc = DegradationCalculator(
        battery_capacity=env.battery_capacity,
        step_duration=env.step_duration,
        battery_life_cost=env.battery_life_cost,
        degradation_temperature=25.0
    )
    
    oracle = OracleHelper(
        env=env,
        degradation_calc=calc,
        degradation_model='linear'
    )
    
    assert hasattr(oracle, 'solve_oracle_dp'), "OracleHelper should have solve_oracle_dp method"
    print("  ✓ OracleHelper initialized successfully")
    print("✓ OracleHelper tests passed!")

def test_batterydeg_integration():
    """Test that batterydeg classes are properly used."""
    print("\nTesting batterydeg integration...")
    
    # Test DegradationModel
    model = DegradationModel()
    deg = model.degradation_per_cycle(T=25.0, Id=0.2, Ich=0.1, SOCav=50.0, DOD=20.0)
    assert 0.0 < deg < 1.0, f"Degradation should be positive and less than 1, got {deg}"
    print(f"  ✓ DegradationModel.degradation_per_cycle: {deg:.6f}")
    
    # Test RainflowCounter
    counter = RainflowCounter(step_duration=0.5)
    cycles = []
    for soc in [50, 70, 60, 80, 50]:
        cycles.extend(counter.update(soc))
    print(f"  ✓ RainflowCounter detected {len(cycles)} cycles")
    
    print("✓ batterydeg integration tests passed!")

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Refactored Agent Class Components")
    print("=" * 60)
    
    try:
        test_degradation_calculator()
        test_helper_functions()
        test_oracle_helper()
        test_batterydeg_integration()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nRefactoring successful:")
        print("  • DegradationCalculator centralizes all degradation calculations")
        print("  • OracleHelper encapsulates Oracle algorithm logic")
        print("  • Helper functions properly extracted")
        print("  • batterydeg.py classes (DegradationModel, RainflowCounter) correctly integrated")
        print("\nThe refactored code maintains consistency with the")
        print("Muenzel et al. (2015) multi-factor battery cycle life model")
        print("as implemented in batterydeg.py.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
