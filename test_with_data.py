"""
Test script to verify the system works with the provided CSV format.
"""

import pandas as pd
from datetime import datetime, time, timedelta
import sys
import os
import pytz

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """Test loading the provided CSV data."""
    print("Testing data loading...")
    
    # Direct CSV loading for testing
    filepath = "data/raw/geo_visibility.csv"
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Please place your geo_visibility.csv in data/raw/ directory")
        return None
    
    # Load CSV
    df = pd.read_csv(filepath)
    
    print(f"\nOriginal CSV loaded with {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check column names
    print("\nColumn details:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}, sample: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")
    
    return df

def parse_timestamps(df):
    """Parse timestamp columns with UTC timezone."""
    print("\nParsing timestamps...")
    
    # Find timestamp columns
    time_cols = []
    for col in df.columns:
        if 'start' in col.lower() and 'time' in col.lower():
            time_cols.append(col)
        elif 'end' in col.lower() and 'time' in col.lower():
            time_cols.append(col)
    
    print(f"Timestamp columns found: {time_cols}")
    
    for col in time_cols:
        try:
            # Parse with UTC timezone
            df[col] = pd.to_datetime(df[col], utc=True)
            print(f"  ✓ {col}: parsed successfully (timezone: {df[col].dt.tz})")
        except Exception as e:
            print(f"  ✗ {col}: error - {e}")
    
    return df

def test_night_window():
    """Test night window calculation with proper timezone."""
    print("\nTesting night window...")
    
    # From constants
    NIGHT_START_UTC = "12:30"
    NIGHT_END_UTC = "00:30"
    
    print(f"Night window: {NIGHT_START_UTC} to {NIGHT_END_UTC} UTC")
    
    # Test date
    test_date = "2025-10-07"
    print(f"Test date: {test_date}")
    
    # Parse night times
    night_start_time = datetime.strptime(NIGHT_START_UTC, "%H:%M").time()
    night_end_time = datetime.strptime(NIGHT_END_UTC, "%H:%M").time()
    
    # Create datetime objects with UTC timezone
    ref_date = datetime.strptime(test_date, "%Y-%m-%d")
    
    # Create timezone-aware datetimes
    night_start = datetime.combine(ref_date, night_start_time).replace(tzinfo=pytz.UTC)
    
    # End time is on next day (crosses midnight)
    night_end = datetime.combine(ref_date + timedelta(days=1), night_end_time).replace(tzinfo=pytz.UTC)
    
    print(f"Night start: {night_start}")
    print(f"Night end: {night_end}")
    print(f"Night duration: {(night_end - night_start).total_seconds()/3600:.2f} hours")
    
    return night_start, night_end

def check_visibility_vs_night(df, night_start, night_end):
    """Check which debris are visible during night."""
    print("\nChecking visibility vs night window...")
    
    # Find start and end time columns
    start_col = None
    end_col = None
    
    for col in df.columns:
        if 'start' in col.lower() and 'time' in col.lower():
            start_col = col
        elif 'end' in col.lower() and 'time' in col.lower():
            end_col = col
    
    if not start_col or not end_col:
        print("ERROR: Could not find start/end time columns")
        return [], []
    
    print(f"Using columns: {start_col}, {end_col}")
    
    # Check each debris
    within_night = []
    outside_night = []
    
    for idx, row in df.iterrows():
        vis_start = row[start_col]
        vis_end = row[end_col]
        
        # Check if visibility window overlaps with night window
        # Overlap if: vis_start < night_end AND vis_end > night_start
        if vis_start < night_end and vis_end > night_start:
            within_night.append(idx)
        else:
            outside_night.append(idx)
    
    print(f"\nResults:")
    print(f"  Total debris: {len(df)}")
    print(f"  Within night window: {len(within_night)}")
    print(f"  Outside night window: {len(outside_night)}")
    
    if within_night:
        print("\nFirst 5 within night window:")
        within_df = df.loc[within_night[:5]]
        # Format times for display
        display_df = within_df.copy()
        display_df['Start'] = display_df[start_col].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        display_df['End'] = display_df[end_col].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        print(display_df[['NORAD ID', 'Satellite Name', 'Start', 'End']])
    
    return within_night, outside_night

def calculate_capacity(night_start, night_end):
    """Calculate physical scheduling capacity."""
    print("\n" + "="*60)
    print("PHYSICAL CAPACITY CALCULATION")
    print("="*60)
    
    OBS_DURATION = 90  # seconds
    SLEW_GAP = 120     # seconds
    SLOT_DURATION = OBS_DURATION + SLEW_GAP
    
    # Night duration
    night_duration = (night_end - night_start).total_seconds()
    
    print(f"\nPhysical constraints:")
    print(f"  Observation duration: {OBS_DURATION}s")
    print(f"  Slew/settling time: {SLEW_GAP}s")
    print(f"  Total per observation: {SLOT_DURATION}s")
    print(f"  Night duration: {night_duration:.0f}s ({night_duration/3600:.2f} hours)")
    
    max_slots = int(night_duration // SLOT_DURATION)
    print(f"  Maximum slots per night: {max_slots}")
    
    # For 5 nights
    print(f"\n5-night capacity:")
    print(f"  Maximum observations in 5 nights: {max_slots * 5}")
    print(f"  With margin (90%): {int(max_slots * 5 * 0.9)}")
    
    return max_slots

def analyze_visibility_windows(df, within_indices, start_col, end_col):
    """Analyze visibility window characteristics."""
    print("\n" + "="*60)
    print("VISIBILITY WINDOW ANALYSIS")
    print("="*60)
    
    within_df = df.loc[within_indices].copy() if within_indices else pd.DataFrame()
    
    if len(within_df) == 0:
        print("No debris within night window to analyze")
        return
    
    # Calculate window durations
    within_df['window_duration_sec'] = (within_df[end_col] - within_df[start_col]).dt.total_seconds()
    
    print(f"Analysis of {len(within_df)} debris within night window:")
    print(f"  Minimum visibility window: {within_df['window_duration_sec'].min():.0f}s")
    print(f"  Maximum visibility window: {within_df['window_duration_sec'].max():.0f}s")
    print(f"  Average visibility window: {within_df['window_duration_sec'].mean():.0f}s")
    print(f"  Median visibility window: {within_df['window_duration_sec'].median():.0f}s")
    
    # Check for windows shorter than observation duration
    OBS_DURATION = 90
    too_short = within_df[within_df['window_duration_sec'] < OBS_DURATION]
    print(f"\nDebris with visibility < {OBS_DURATION}s (cannot be scheduled): {len(too_short)}")
    
    if len(too_short) > 0:
        print("These debris have insufficient visibility windows:")
        print(too_short[['NORAD ID', 'Satellite Name', 'window_duration_sec']].head(10).to_string(index=False))
    
    # Check peak elevations
    if 'Peak Elevation (deg)' in within_df.columns:
        print(f"\nPeak elevation statistics:")
        print(f"  Minimum: {within_df['Peak Elevation (deg)'].min():.1f}°")
        print(f"  Maximum: {within_df['Peak Elevation (deg)'].max():.1f}°")
        print(f"  Average: {within_df['Peak Elevation (deg)'].mean():.1f}°")
        print(f"  Median: {within_df['Peak Elevation (deg)'].median():.1f}°")

def main():
    """Run all tests."""
    print("="*60)
    print("GEO Satellite Scheduling System - Data Verification")
    print("="*60)
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Check if data exists
    data_file = "data/raw/geo_visibility.csv"
    if not os.path.exists(data_file):
        print(f"\nERROR: {data_file} not found!")
        print("\nPlease:")
        print("1. Create 'data/raw/' directory")
        print("2. Copy your geo_visibility.csv to data/raw/")
        print("\nCurrent directory:", os.getcwd())
        return
    
    # Install pytz if needed
    try:
        import pytz
    except ImportError:
        print("\nInstalling pytz for timezone support...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytz"])
        import pytz
    
    # Run tests
    try:
        df = test_data_loading()
        if df is None or df.empty:
            return
        
        df = parse_timestamps(df)
        night_start, night_end = test_night_window()
        within_night, outside_night = check_visibility_vs_night(df, night_start, night_end)
        
        # Find column names for analysis
        start_col = None
        end_col = None
        for col in df.columns:
            if 'start' in col.lower() and 'time' in col.lower():
                start_col = col
            elif 'end' in col.lower() and 'time' in col.lower():
                end_col = col
        
        if start_col and end_col:
            analyze_visibility_windows(df, within_night, start_col, end_col)
        
        max_slots = calculate_capacity(night_start, night_end)
        
        # Summary
        print("\n" + "="*60)
        print("SCHEDULING SUMMARY")
        print("="*60)
        print(f"Total debris in CSV: {len(df)}")
        print(f"Visible during night: {len(within_night)}")
        print(f"Maximum schedulable per night: {max_slots}")
        
        if len(within_night) > max_slots:
            nights_needed = (len(within_night) + max_slots - 1) // max_slots  # Ceiling division
            print(f"Minimum nights needed: {nights_needed} (max {max_slots} per night)")
        else:
            print("All visible debris can be scheduled in 1 night")
        
        if len(within_night) > max_slots * 5:
            print(f"\n⚠️  WARNING: {len(within_night)} debris visible, but only {max_slots * 5} can be scheduled in 5 nights")
            print(f"           {len(within_night) - max_slots * 5} debris will remain unscheduled")
        else:
            print(f"\n✓ All {len(within_night)} visible debris can be scheduled within 5 nights")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Run the full scheduler:")
        print("   python main.py --date 2025-10-07")
        print("\n2. To disable GA (use greedy only):")
        print("   python main.py --date 2025-10-07 --no-ga")
        print("\n3. To see scheduling statistics:")
        print("   Check outputs/scheduling_statistics.json after running")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()