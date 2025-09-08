import time
import psutil
import os
from typing import Dict, Any

class EnhancedEnergyTracker:
    """Track energy consumption and CO2 emissions during model inference"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_cpu = None
        self.end_cpu = None
        self.start_memory = None
        self.end_memory = None
        self.co2_emission_factor = 0.0004  # kg CO2 per kWh (average)
        
    def start_tracking(self):
        """Start tracking energy consumption"""
        self.start_time = time.time()
        self.start_cpu = psutil.cpu_percent(interval=None)
        
        # Get memory usage
        process = psutil.Process(os.getpid())
        self.start_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
        
    def stop_tracking(self) -> Dict[str, Any]:
        """Stop tracking and calculate metrics"""
        self.end_time = time.time()
        self.end_cpu = psutil.cpu_percent(interval=None)
        
        # Get final memory usage
        process = psutil.Process(os.getpid())
        self.end_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
        
        # Calculate metrics
        processing_time = self.end_time - self.start_time
        
        # Estimate power consumption (very rough)
        # Based on CPU usage and processing time
        avg_cpu_usage = (self.start_cpu + self.end_cpu) / 2
        estimated_power_watts = (avg_cpu_usage / 100) * 65  # Assume 65W max CPU power
        energy_kwh = (estimated_power_watts * processing_time) / (1000 * 3600)  # Convert to kWh
        
        # Calculate CO2 emissions
        co2_emissions_kg = energy_kwh * self.co2_emission_factor
        co2_emissions_g = co2_emissions_kg * 1000
        
        # Calculate efficiency score (lower CO2 = higher score)
        efficiency_score = max(0, 10 - (co2_emissions_g * 2))  # Scale 0-10
        
        return {
            "processing_time_seconds": processing_time,
            "energy_consumption_kwh": energy_kwh,
            "co2_emissions_kg": co2_emissions_kg,
            "co2_emissions_g": co2_emissions_g,
            "cpu_utilization_start": self.start_cpu,
            "cpu_utilization_end": self.end_cpu,
            "memory_used_gb": max(self.start_memory, self.end_memory),
            "energy_efficiency_score": efficiency_score,
            "gpu_metrics": {}  # Placeholder for GPU tracking
        }