"""
Helper classes for energy decision algorithms (SDP, MRDP, Oracle).

This module extracts algorithm-specific logic from the main Agent class to improve
readability and maintainability. Each algorithm helper encapsulates its own methods
while using shared degradation models from batterydeg.py.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from batterydeg import DegradationModel


class DegradationCalculator:
    """
    Centralized degradation calculation for all algorithms.
    
    Uses DegradationModel and RainflowCounter from batterydeg.py to ensure
    consistency with the multi-factor battery cycle life prediction methodology
    from Muenzel et al. (2015).
    """
    
    def __init__(self, battery_capacity: float, step_duration: float, 
                 battery_life_cost: float, degradation_temperature: float = 25.0,
                 calibration: float = 1.0):
        """
        Initialize degradation calculator.

        Args:
            battery_capacity: Battery capacity in kWh
            step_duration: Time step duration in hours
            battery_life_cost: Total cost of battery replacement in $
            degradation_temperature: Operating temperature in °C
            calibration: Multiplier applied to the per-step wear estimate so the
                planner's total wear can be matched to the environment's realized
                wear (the env charges per closed rainflow cycle, while the planner
                accumulates a per-step half-cycle; see known_issues B4).
        """
        self.battery_capacity = battery_capacity
        self.step_duration = step_duration
        self.battery_life_cost = battery_life_cost
        self.degradation_temperature = degradation_temperature
        self.calibration = float(calibration)

        # Initialize the class-based degradation model from batterydeg.py
        self.cycle_degradation_model = DegradationModel()
    
    def degradation_per_cycle(self, Id: float, Ich: float, soc_percent: float, DoD: float) -> float:
        """
        Calculate degradation fraction per cycle using the class-based model.
        
        Args:
            Id: Discharge current C-rate
            Ich: Charge current C-rate
            soc_percent: Average state of charge in percent (0-100)
            DoD: Depth of discharge in percent (0-100)
            
        Returns:
            Degradation fraction (0-1) for this cycle
        """
        return self.cycle_degradation_model.degradation_per_cycle(
            T=self.degradation_temperature,
            Id=Id,
            Ich=Ich,
            SOCav=soc_percent,
            DOD=DoD,
        )
    
    def compute_step_degradation(self, soc_start_kwh: float, soc_end_kwh: float) -> float:
        """Marginal degradation fraction for one step's SoC transition.

        A single step is treated as a **half-cycle** whose depth is the SoC
        excursion. The multi-factor per-cycle model is evaluated at the
        transition's average SoC and C-rate, then halved.

        The previous implementation fed a 3-point sequence (start, end, start)
        to ``RainflowCounter``, which can never close a cycle (that needs four
        turning points), so this estimator always returned 0 — the SDP/MRDP/
        Oracle stage costs were effectively degradation-blind (known_issues A5).
        """
        if self.battery_capacity <= 0:
            return 0.0

        delta = float(soc_end_kwh) - float(soc_start_kwh)
        if abs(delta) <= 1e-12:
            return 0.0

        dod = abs(delta) / self.battery_capacity * 100.0
        soc_avg = (float(soc_start_kwh) + float(soc_end_kwh)) / 2.0 / self.battery_capacity * 100.0
        c_rate = abs(delta) / self.battery_capacity / max(self.step_duration, 1e-12)
        if delta > 0:
            Id, Ich = 0.0, c_rate
        else:
            Id, Ich = c_rate, 0.0

        deg = self.degradation_per_cycle(Id, Ich, soc_avg, dod)

        # Sanitize: ensure non-negative and finite
        if not np.isfinite(deg) or deg <= 0.0:
            return 0.0
        return float(min(0.5 * deg * self.calibration, 1.0))

    def compute_rainflow_degradation(self, soc_start_kwh: float, soc_end_kwh: float) -> float:
        """Deprecated alias for :meth:`compute_step_degradation`."""
        return self.compute_step_degradation(soc_start_kwh, soc_end_kwh)


def interpolate_ctg(soc_levels_kwh: np.ndarray, ctg_array: np.ndarray, soc_value: float) -> float:
    """
    Linearly interpolate cost-to-go values for a continuous SoC between discrete levels.
    
    Clamps at the ends if soc_value is outside the range.
    
    Args:
        soc_levels_kwh: Array of discrete SoC levels in kWh
        ctg_array: Array of cost-to-go values corresponding to soc_levels_kwh
        soc_value: Continuous SoC value to interpolate at
    
    Returns:
        Interpolated cost-to-go value
    """
    soc_value = np.clip(soc_value, soc_levels_kwh[0], soc_levels_kwh[-1])
    return np.interp(soc_value, soc_levels_kwh, ctg_array)


def compute_grid_cost(grid_energy: float, import_price: float, export_price: float, 
                     max_grid_energy: float) -> float:
    """
    Compute grid cost with explicit import/export semantics and grid limit checking.
    
    Args:
        grid_energy: Grid energy (positive = import, negative = export)
        import_price: Price per kWh for importing energy
        export_price: Price per kWh for exporting energy (revenue)
        max_grid_energy: Maximum allowed grid energy (absolute value)
    
    Returns:
        Grid cost (positive = cost, negative = revenue, np.inf if limit exceeded)
    """
    # Check grid limits first
    if abs(grid_energy) > max_grid_energy + 1e-6:  # Add small tolerance
        return np.inf
    
    if grid_energy > 0:  # Importing energy
        return grid_energy * import_price
    else:  # Exporting energy (grid_energy is negative)
        # Export generates revenue, so cost is negative (revenue reduces total cost)
        export_revenue = abs(grid_energy) * export_price
        return -export_revenue

