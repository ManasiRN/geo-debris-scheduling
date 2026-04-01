"""
Data preprocessing module for GEO satellite scheduling.
Updated to handle the specific CSV format with additional columns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
import os
import warnings
warnings.filterwarnings('ignore')

from config.constants import *
from src.utils import parse_utc_datetime, is_within_night_window, create_night_boundaries


def load_visibility_data(filepath: str = None) -> pd.DataFrame:
    """
    Load and validate visibility data from CSV.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with parsed timestamps
    """
    if filepath is None:
        filepath = RAW_DATA_PATH
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Visibility data file not found: {filepath}")
    
    # Load CSV
    df = pd.read_csv(filepath)
    
    print(f"Original CSV columns: {df.columns.tolist()}")
    
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    
    # Map column names based on your input format
    column_mapping = {}
    
    # Try to identify columns
    for col in df.columns:
        col_lower = col.lower()
        if 'norad' in col_lower or 'id' in col_lower:
            column_mapping[col] = 'norad_id'
        elif 'satellite' in col_lower or 'name' in col_lower:
            column_mapping[col] = 'satellite_name'
        elif 'station' in col_lower:
            column_mapping[col] = 'station_name'
        elif 'duration' in col_lower and 'sec' in col_lower:
            column_mapping[col] = 'duration_sec'
        elif 'duration' in col_lower and 'min' in col_lower:
            column_mapping[col] = 'duration_min'
        elif 'start' in col_lower and 'time' in col_lower:
            column_mapping[col] = 'visibility_start_utc'
        elif 'end' in col_lower and 'time' in col_lower:
            column_mapping[col] = 'visibility_end_utc'
        elif 'peak' in col_lower or 'elevation' in col_lower:
            column_mapping[col] = 'peak_elevation'
    
    # Rename columns
    df.rename(columns=column_mapping, inplace=True)
    
    print(f"Mapped columns: {column_mapping}")
    
    # Check for required columns
    required_columns = ['norad_id', 'satellite_name', 'visibility_start_utc', 
                       'visibility_end_utc', 'peak_elevation']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns {missing_columns}. Available: {df.columns.tolist()}")
        # Try to use alternative naming
        for col in ['Start Time (UTC)', 'End Time (UTC)', 'Peak Elevation (deg)']:
            if col in df.columns:
                clean_name = col.lower().replace(' (utc)', '').replace(' (deg)', '').replace(' ', '_')
                df.rename(columns={col: clean_name}, inplace=True)
    
    # Now check again
    for col in ['visibility_start_utc', 'visibility_end_utc', 'peak_elevation']:
        if col not in df.columns:
            # Last resort: try to find by partial match
            for actual_col in df.columns:
                if 'start' in actual_col.lower() and 'time' in actual_col.lower():
                    df.rename(columns={actual_col: 'visibility_start_utc'}, inplace=True)
                    break
                elif 'end' in actual_col.lower() and 'time' in actual_col.lower():
                    df.rename(columns={actual_col: 'visibility_end_utc'}, inplace=True)
                    break
                elif 'elevation' in actual_col.lower():
                    df.rename(columns={actual_col: 'peak_elevation'}, inplace=True)
                    break
    
    # Parse timestamps
    try:
        df['visibility_start_utc'] = pd.to_datetime(df['visibility_start_utc'], utc=True)
        df['visibility_end_utc'] = pd.to_datetime(df['visibility_end_utc'], utc=True)
    except Exception as e:
        print(f"Error parsing timestamps: {e}")
        print("Sample timestamps:")
        print(df[['visibility_start_utc', 'visibility_end_utc']].head())
        raise
    
    # Convert peak elevation to numeric
    if 'peak_elevation' in df.columns:
        df['peak_elevation'] = pd.to_numeric(df['peak_elevation'], errors='coerce')
    
    # Calculate visibility duration
    df['visibility_duration_sec'] = (df['visibility_end_utc'] - df['visibility_start_utc']).dt.total_seconds()
    
    print(f"Data summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Time range: {df['visibility_start_utc'].min()} to {df['visibility_end_utc'].max()}")
    print(f"  Average duration: {df['visibility_duration_sec'].mean():.0f} seconds")
    
    # Filter out debris with visibility windows shorter than observation duration
    initial_count = len(df)
    df = df[df['visibility_duration_sec'] >= OBS_DURATION_SEC].copy()
    filtered_count = initial_count - len(df)
    
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} debris with visibility < {OBS_DURATION_SEC}s")
    
    # Add unique ID for internal reference
    df['debris_id'] = df.index
    
    # Ensure norad_id is integer
    if 'norad_id' in df.columns:
        df['norad_id'] = pd.to_numeric(df['norad_id'], errors='coerce').astype('Int64')
    
    return df


def filter_by_night_window(df: pd.DataFrame, date_str: str = None) -> pd.DataFrame:
    """
    Filter debris visibility windows to only include portions within night window.
    
    Args:
        df: DataFrame with visibility windows
        date_str: Reference date for night window
        
    Returns:
        Filtered DataFrame with clipped visibility windows
    """
    if df.empty:
        return df
    
    # Get night boundaries for the reference date
    # Use the date from the first visibility window if not provided
    if date_str is None:
        first_date = df['visibility_start_utc'].iloc[0].date()
        date_str = first_date.strftime("%Y-%m-%d")
    
    night_start, night_end = create_night_boundaries(date_str)
    
    print(f"Night window for {date_str}: {night_start} to {night_end}")
    
    results = []
    
    for idx, row in df.iterrows():
        vis_start = row['visibility_start_utc']
        vis_end = row['visibility_end_utc']
        
        # Check if visibility window overlaps with night window
        # Four cases for overlap:
        # 1. vis_start <= night_start <= vis_end
        # 2. vis_start <= night_end <= vis_end
        # 3. night_start <= vis_start <= night_end
        # 4. night_start <= vis_end <= night_end
        
        if vis_end <= night_start or vis_start >= night_end:
            # No overlap
            continue
        
        # Clip visibility window to night boundaries
        clipped_start = max(vis_start, night_start)
        clipped_end = min(vis_end, night_end)
        
        # Skip if clipped window is too short
        clipped_duration = (clipped_end - clipped_start).total_seconds()
        if clipped_duration < OBS_DURATION_SEC:
            continue
        
        # Create new row with clipped window
        new_row = row.copy()
        new_row['visibility_start_utc'] = clipped_start
        new_row['visibility_end_utc'] = clipped_end
        new_row['clipped_duration_sec'] = clipped_duration
        new_row['original_visibility_start'] = vis_start
        new_row['original_visibility_end'] = vis_end
        
        results.append(new_row)
    
    if results:
        filtered_df = pd.DataFrame(results)
        filtered_df.reset_index(drop=True, inplace=True)
        
        print(f"Night window filtering:")
        print(f"  Input debris: {len(df)}")
        print(f"  Within night window: {len(filtered_df)}")
        print(f"  Clipped out: {len(df) - len(filtered_df)}")
        
        return filtered_df
    else:
        print("No debris within night window")
        return pd.DataFrame(columns=df.columns)


def preprocess_pipeline(filepath: str = None, reference_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline.
    
    Args:
        filepath: Path to input CSV
        reference_date: Date for night window filtering
        
    Returns:
        Tuple of (filtered_df, unschedulable_df)
    """
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    print("Loading visibility data...")
    df = load_visibility_data(filepath)
    
    print(f"\nLoaded {len(df)} debris objects.")
    print(f"Data columns: {df.columns.tolist()}")
    
    # Filter by night window
    print("\nFiltering by night window...")
    filtered_df = filter_by_night_window(df, reference_date)
    
    # Identify unschedulable debris
    unschedulable_rows = []
    
    # 1. Debris with visibility windows shorter than observation duration
    duration_mask = df['visibility_duration_sec'] < OBS_DURATION_SEC
    unschedulable_duration = df[duration_mask].copy()
    unschedulable_duration['unschedulable_reason'] = 'visibility_window_too_short'
    unschedulable_duration['unschedulable_details'] = f'Duration: {unschedulable_duration["visibility_duration_sec"].iloc[0]:.0f}s < {OBS_DURATION_SEC}s'
    unschedulable_rows.append(unschedulable_duration)
    
    # 2. Debris completely outside night window
    if not filtered_df.empty:
        night_filtered_ids = set(df['debris_id']) - set(filtered_df['debris_id'])
        unschedulable_night = df[df['debris_id'].isin(night_filtered_ids)].copy()
        unschedulable_night['unschedulable_reason'] = 'outside_night_window'
        unschedulable_night['unschedulable_details'] = 'No overlap with 12:30-00:30 UTC night window'
        unschedulable_rows.append(unschedulable_night)
    else:
        # All debris are outside night window
        unschedulable_night = df.copy()
        unschedulable_night['unschedulable_reason'] = 'outside_night_window'
        unschedulable_night['unschedulable_details'] = 'No overlap with 12:30-00:30 UTC night window'
        unschedulable_rows.append(unschedulable_night)
    
    # Combine unschedulable reasons
    if unschedulable_rows:
        unschedulable_df = pd.concat(unschedulable_rows, ignore_index=True)
        unschedulable_df.drop_duplicates(subset=['debris_id', 'norad_id'], inplace=True)
    else:
        unschedulable_df = pd.DataFrame()
    
    print(f"\nPreprocessing results:")
    print(f"  Total debris: {len(df)}")
    print(f"  Schedulable debris (within night window): {len(filtered_df)}")
    print(f"  Unschedulable debris: {len(unschedulable_df)}")
    
    if len(unschedulable_df) > 0:
        reasons = unschedulable_df['unschedulable_reason'].value_counts()
        for reason, count in reasons.items():
            print(f"    - {reason}: {count}")
    
    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    if not filtered_df.empty:
        # Save only essential columns
        save_cols = ['norad_id', 'satellite_name', 'visibility_start_utc', 
                    'visibility_end_utc', 'peak_elevation', 'visibility_duration_sec']
        save_df = filtered_df[save_cols].copy()
        save_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"\nSaved filtered data to {PROCESSED_DATA_PATH}")
    
    return filtered_df, unschedulable_df