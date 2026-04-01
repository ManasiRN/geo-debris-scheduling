"""
Improved greedy scheduler that distributes debris across nights intelligently.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
from src.greedy_scheduler import GreedyScheduler
from src.constraints import TimeSlot


class ImprovedGreedyScheduler(GreedyScheduler):
    """Improved scheduler that balances debris across nights."""
    
    def distribute_debris_across_nights(self, 
                                       debris_data: pd.DataFrame,
                                       max_nights: int = 5) -> Dict[int, pd.DataFrame]:
        """
        Intelligently distribute debris across nights.
        
        Strategy:
        1. Sort debris by visibility START time
        2. Assign to nights in round-robin fashion
        3. This spreads different visibility times across nights
        """
        # Sort by visibility start time
        sorted_debris = debris_data.sort_values('visibility_start').copy()
        
        # Initialize night buckets
        night_buckets = {i: [] for i in range(1, max_nights + 1)}
        
        # Round-robin assignment
        for idx, (_, row) in enumerate(sorted_debris.iterrows()):
            night_num = (idx % max_nights) + 1
            night_buckets[night_num].append(row)
        
        # Convert to DataFrames
        night_dataframes = {}
        for night_num, rows in night_buckets.items():
            if rows:
                night_dataframes[night_num] = pd.DataFrame(rows)
            else:
                night_dataframes[night_num] = pd.DataFrame(columns=debris_data.columns)
        
        return night_dataframes
    
    def multi_night_schedule(self,
                            debris_data: pd.DataFrame,
                            max_nights: int = 5) -> Dict[int, List[TimeSlot]]:
        """
        Schedule across multiple nights with better distribution.
        """
        # First, distribute debris across nights
        night_dataframes = self.distribute_debris_across_nights(debris_data, max_nights)
        
        all_schedules = {}
        
        for night_num in range(1, max_nights + 1):
            night_df = night_dataframes[night_num]
            
            if night_df.empty:
                continue
            
            # Get night start time
            reference_date = debris_data['visibility_start'].iloc[0].date()
            night_start = datetime.combine(reference_date + timedelta(days=night_num-1), 
                                          time(12, 30, 0))
            
            # Schedule this night
            scheduled_slots, _ = self.schedule_night(night_df, night_start)
            all_schedules[night_num] = scheduled_slots
        
        return all_schedules