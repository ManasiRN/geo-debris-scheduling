"""
Greedy baseline scheduler for one night.
Implements Earliest Deadline First with elevation tie-breaking.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Set
import copy

from config.constants import *
from src.utils import parse_utc_datetime, create_night_boundaries
from src.constraints import TimeSlotChecker


class GreedyScheduler:
    """Greedy baseline scheduler for single night."""
    
    def __init__(self, night_date: str = None):
        """
        Initialize scheduler for a specific night.
        
        Args:
            night_date: Reference date for night window
        """
        self.night_date = night_date
        self.night_start, self.night_end = create_night_boundaries(
            night_date if night_date else datetime.utcnow().strftime("%Y-%m-%d")
        )
    
    def schedule_night(self, debris_df: pd.DataFrame) -> pd.DataFrame:
        """
        Schedule observations for one night using greedy algorithm.
        
        Args:
            debris_df: DataFrame of debris to schedule
            
        Returns:
            DataFrame of scheduled observations
        """
        if debris_df.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying original
        df = debris_df.copy()
        
        # Sort by: 1) earliest visibility end, 2) highest peak elevation
        df['visibility_end_sec'] = df['visibility_end_utc'].apply(
            lambda x: parse_utc_datetime(x).timestamp()
        )
        df = df.sort_values(['visibility_end_sec', 'peak_elevation'], 
                          ascending=[True, False])
        
        # Initialize schedule
        schedule = []
        current_time = self.night_start
        
        # Maximum slots per night (physical limit)
        max_slots = int((self.night_end - self.night_start).total_seconds() / 
                       (OBS_DURATION_SEC + SLEW_GAP_SEC))
        
        for _, debris in df.iterrows():
            if len(schedule) >= max_slots:
                break  # Physical limit reached
            
            vis_start = parse_utc_datetime(debris['visibility_start_utc'])
            vis_end = parse_utc_datetime(debris['visibility_end_utc'])
            
            # If current time is before visibility start, wait
            if current_time < vis_start:
                current_time = vis_start
            
            # Check if we can schedule this debris
            proposed_end = current_time + timedelta(seconds=OBS_DURATION_SEC)
            
            if proposed_end <= vis_end and proposed_end <= self.night_end:
                # Schedule this observation
                obs_start = current_time
                obs_end = proposed_end
                
                schedule.append({
                    'norad_id': debris['norad_id'],
                    'satellite_name': debris['satellite_name'],
                    'observation_start_utc': obs_start.strftime(TIME_FORMAT),
                    'observation_end_utc': obs_end.strftime(TIME_FORMAT),
                    'peak_elevation': debris['peak_elevation'],
                    'visibility_start_utc': debris['visibility_start_utc'],
                    'visibility_end_utc': debris['visibility_end_utc'],
                    'scheduled_day': 0  # Will be set by multi-day scheduler
                })
                
                # Advance time for next observation
                current_time = obs_end + timedelta(seconds=SLEW_GAP_SEC)
                
                # If we've reached night end, stop
                if current_time >= self.night_end:
                    break
        
        return pd.DataFrame(schedule)
    
    def evaluate_schedule_quality(self, schedule: pd.DataFrame, 
                                 all_debris: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate quality of a schedule.
        
        Args:
            schedule: Scheduled observations
            all_debris: All available debris
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            'scheduled_count': len(schedule),
            'total_available': len(all_debris),
            'coverage_ratio': len(schedule) / len(all_debris) if len(all_debris) > 0 else 0,
            'time_utilization': 0.0,
            'avg_elevation': 0.0
        }
        
        if len(schedule) > 0:
            # Time utilization
            total_obs_time = len(schedule) * OBS_DURATION_SEC
            night_duration = (self.night_end - self.night_start).total_seconds()
            metrics['time_utilization'] = total_obs_time / night_duration
            
            # Average elevation
            metrics['avg_elevation'] = schedule['peak_elevation'].mean()
        
        return metrics


def simulate_greedy_with_order(debris_df: pd.DataFrame, 
                              order: List[int],
                              night_date: str = None) -> pd.DataFrame:
    """
    Simulate greedy scheduling with a specific debris order.
    Used by GA fitness evaluation.
    
    Args:
        debris_df: DataFrame of debris
        order: List of debris indices in scheduling order
        night_date: Reference date for night
        
    Returns:
        DataFrame of scheduled observations
    """
    scheduler = GreedyScheduler(night_date)
    
    # Reorder debris according to chromosome
    ordered_df = debris_df.iloc[order].copy()
    
    # Run greedy scheduling
    schedule = scheduler.schedule_night(ordered_df)
    
    return schedule