"""
Helper classes for energy decision algorithms (SDP, MRDP, Oracle).

This module extracts algorithm-specific logic from the main Agent class to improve
readability and maintainability. Each algorithm helper encapsulates its own methods
while using shared degradation models from batterydeg.py.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from batterydeg import DegradationModel, RainflowCounter


class DegradationCalculator:
    """
    Centralized degradation calculation for all algorithms.
    
    Uses DegradationModel and RainflowCounter from batterydeg.py to ensure
    consistency with the multi-factor battery cycle life prediction methodology
    from Muenzel et al. (2015).
    """
    
    def __init__(self, battery_capacity: float, step_duration: float, 
                 battery_life_cost: float, degradation_temperature: float = 25.0):
        """
        Initialize degradation calculator.
        
        Args:
            battery_capacity: Battery capacity in kWh
            step_duration: Time step duration in hours
            battery_life_cost: Total cost of battery replacement in $
            degradation_temperature: Operating temperature in °C
        """
        self.battery_capacity = battery_capacity
        self.step_duration = step_duration
        self.battery_life_cost = battery_life_cost
        self.degradation_temperature = degradation_temperature
        
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
    
    def compute_linearized_degradation(self, Id: float, Ich: float, soc_percent: float, 
                                      energy_kwh: float, base_DoD: float = 80.0,
                                      correction_factor: float = 1.0) -> float:
        """
        Compute degradation fraction using linearized per-kWh model.
        
        This method converts a representative full-cycle wear into a per-kWh wear 
        using a base DoD, then scales by the energy moved in this step.
        
        Args:
            Id: Discharge current C-rate
            Ich: Charge current C-rate
            soc_percent: Average state of charge in percent (0-100)
            energy_kwh: Energy throughput in kWh
            base_DoD: Base depth of discharge for reference cycle (default 80%)
            correction_factor: Static correction factor to apply
            
        Returns:
            Degradation fraction (0-1) for this energy throughput
        """
        if energy_kwh <= 0:
            return 0.0
        
        # Calculate energy for a full base cycle (charge + discharge)
        energy_full_base_cycle = self.battery_capacity * (base_DoD / 100.0) * 2.0
        if energy_full_base_cycle <= 0:
            return 0.0
        
        # Get degradation for one full cycle at base DoD
        cycle_wear = self.degradation_per_cycle(Id, Ich, soc_percent, base_DoD)
        
        # Convert to per-kWh wear and apply to actual energy
        wear_per_kwh = cycle_wear / energy_full_base_cycle
        frac = wear_per_kwh * energy_kwh * correction_factor
        
        # Sanitize: ensure non-negative and finite
        if not np.isfinite(frac) or frac <= 0.0:
            return 0.0
        return float(min(frac, 1.0))
    
    def compute_rainflow_degradation(self, soc_start_kwh: float, soc_end_kwh: float) -> float:
        """
        Estimate degradation for a single step using rainflow counting.
        
        Uses the RainflowCounter class from batterydeg.py to detect cycles
        and calculate degradation.
        
        Args:
            soc_start_kwh: Starting state of charge in kWh
            soc_end_kwh: Ending state of charge in kWh
            
        Returns:
            Degradation fraction (0-1) for this SoC transition
        """
        if self.battery_capacity <= 0:
            return 0.0
        
        # Convert to percentages
        start_pct = np.clip((soc_start_kwh / self.battery_capacity) * 100.0, 0.0, 100.0)
        end_pct = np.clip((soc_end_kwh / self.battery_capacity) * 100.0, 0.0, 100.0)
        
        # Use RainflowCounter to detect cycles
        counter = RainflowCounter(step_duration=self.step_duration)
        cycles = []
        for val in (start_pct, end_pct, start_pct):
            cycles.extend(counter.update(val))
        
        # Sum degradation from all detected cycles
        deg_frac = 0.0
        for SoC_avg, DoD, Id_cycle, Ich_cycle in cycles:
            deg_frac += self.degradation_per_cycle(Id_cycle, Ich_cycle, SoC_avg, DoD)
        
        # Sanitize: ensure non-negative and finite
        if not np.isfinite(deg_frac) or deg_frac <= 0.0:
            return 0.0
        return float(min(deg_frac, 1.0))


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

