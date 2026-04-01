"""
Fitness evaluation for Genetic Algorithm.
"""

import numpy as np
from typing import List, Dict, Tuple
import pandas as pd

from config.constants import *
from src.greedy_scheduler import simulate_greedy_with_order


class FitnessEvaluator:
    """Evaluate fitness of chromosomes."""
    
    def __init__(self, debris_df: pd.DataFrame, night_date: str = None):
        """
        Initialize fitness evaluator.
        
        Args:
            debris_df: DataFrame of debris to schedule
            night_date: Reference date for night window
        """
        self.debris_df = debris_df
        self.night_date = night_date
        self.num_debris = len(debris_df)
        
        # Pre-compute elevation weights for tie-breaking
        self.elevation_weights = self._compute_elevation_weights()
    
    def _compute_elevation_weights(self) -> np.ndarray:
        """Compute normalized elevation weights for fitness scoring."""
        elevations = self.debris_df['peak_elevation'].values
        
        if len(elevations) == 0:
            return np.array([])
        
        # Normalize elevations to [0, 1]
        min_elev = np.min(elevations)
        max_elev = np.max(elevations)
        
        if max_elev > min_elev:
            norm_elev = (elevations - min_elev) / (max_elev - min_elev)
        else:
            norm_elev = np.ones_like(elevations) * 0.5
        
        # Scale to have small impact (0.1 per debris max)
        return norm_elev * 0.1
    
    def evaluate(self, chromosome) -> float:
        """
        Evaluate fitness of a chromosome.
        
        Fitness = number of scheduled debris + elevation bonus
        
        Args:
            chromosome: Chromosome object
            
        Returns:
            Fitness score (higher is better)
        """
        # Simulate greedy scheduling with chromosome order
        schedule = simulate_greedy_with_order(
            self.debris_df, 
            chromosome.genes,
            self.night_date
        )
        
        # Base fitness: number of scheduled debris
        base_fitness = len(schedule)
        
        if base_fitness == 0:
            return 0.0
        
        # Add elevation bonus for scheduled debris
        elevation_bonus = 0.0
        scheduled_indices = set(schedule.index)
        
        for debris_idx in scheduled_indices:
            if debris_idx < len(self.elevation_weights):
                elevation_bonus += self.elevation_weights[debris_idx]
        
        return base_fitness + elevation_bonus
    
    def batch_evaluate(self, chromosomes: List) -> np.ndarray:
        """Evaluate multiple chromosomes efficiently."""
        return np.array([self.evaluate(chrom) for chrom in chromosomes])