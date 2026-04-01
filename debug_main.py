#!/usr/bin/env python3
"""
Debug version to identify issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os
from pathlib import Path
import sys
import traceback

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.constants import OBS_DURATION_SEC, SLEW_GAP_SEC, MAX_NIGHTS
from config.night_window import NIGHT_START_UTC, NIGHT_END_UTC

from src.preprocessing import load_and_preprocess, prepare_night_data
from src.greedy_scheduler import GreedyScheduler

def debug_single_night():
    """Debug scheduling for a single night."""
    print("Debugging single night scheduling...")
    print("=" * 60)
    
    # Load data
    csv_path = "data/raw/geo_visibility.csv"
    df = load_and_preprocess(csv_path)
    
    print(f"\nLoaded {len(df)} debris")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 debris:")
    print(df[['norad_id', 'visibility_start', 'visibility_end', 'peak_elevation']].head())
    
    # Test night 1
    print(f"\n{'='*60}")
    print("Testing Night 1")
    print("=" * 60)
    
    night_data = prepare_night_data(df, day_offset=0)
    
    if night_data.empty:
        print("No debris for night 1 after filtering!")
        return
    
    print(f"\nAfter night filtering: {len(night_data)} debris")
    print(f"\nSample of night data:")
    print(night_data[['norad_id', 'visibility_start', 'visibility_end', 'peak_elevation']].head(10))
    
    # Check visibility durations
    night_data['duration_hours'] = (night_data['visibility_end'] - night_data['visibility_start']) / np.timedelta64(1, 'h')
    print(f"\nVisibility duration statistics (hours):")
    print(night_data['duration_hours'].describe())
    
    # Try scheduling
    print(f"\n{'='*60}")
    print("Attempting scheduling...")
    print("=" * 60)
    
    scheduler = GreedyScheduler()
    
    # Get night start time
    reference_date = df['visibility_start'].iloc[0].date()
    night_start = datetime.combine(reference_date, NIGHT_START_UTC)
    
    print(f"Night start: {night_start}")
    print(f"Night end: {night_start + timedelta(hours=12)}")
    
    try:
        scheduled_slots, unscheduled = scheduler.schedule_night(night_data, night_start)
        
        print(f"\nScheduled {len(scheduled_slots)} debris")
        print(f"Unscheduled: {len(unscheduled)} debris")
        
        if scheduled_slots:
            print(f"\nFirst 10 scheduled observations:")
            for i, slot in enumerate(scheduled_slots[:10]):
                print(f"{i+1:3d}. {slot.norad_id}: {slot.start_time.strftime('%H:%M:%S')} - {slot.end_time.strftime('%H:%M:%S')} "
                      f"(visible: {slot.visibility_start.strftime('%H:%M:%S')} - {slot.visibility_end.strftime('%H:%M:%S')})")
            
            # Convert to DataFrame
            schedule_df = scheduler.convert_slots_to_dataframe(scheduled_slots)
            
            print(f"\nSchedule DataFrame shape: {schedule_df.shape}")
            print(f"\nFirst few rows of schedule:")
            print(schedule_df.head())
            
            # Validate
            from src.utils import validate_schedule
            is_valid, errors = validate_schedule(schedule_df, 1)
            
            print(f"\nSchedule valid: {is_valid}")
            if errors:
                print(f"Errors: {errors[:5]}")  # Show first 5 errors
        
    except Exception as e:
        print(f"\nERROR during scheduling: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_night()