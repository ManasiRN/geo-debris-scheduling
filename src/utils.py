"""
Utility functions for the scheduling system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config.constants import *


def parse_utc_datetime(dt_str: str) -> datetime:
    """Parse UTC datetime string with timezone awareness."""
    try:
        return datetime.strptime(dt_str, TIME_FORMAT)
    except ValueError:
        # Try alternative format if needed
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))


def time_to_seconds(t: time) -> float:
    """Convert time object to seconds since midnight."""
    return t.hour * 3600 + t.minute * 60 + t.second


def seconds_to_time(seconds: float) -> time:
    """Convert seconds since midnight to time object."""
    seconds = int(seconds) % 86400
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return time(hour=hours, minute=minutes, second=secs)


# Update the is_within_night_window function in utils.py
def is_within_night_window(start_dt: datetime, end_dt: datetime) -> bool:
    """
    Check if a time interval falls within the night window.
    Handles midnight crossing generically.
    
    Args:
        start_dt: Start datetime
        end_dt: End datetime
        
    Returns:
        True if interval is completely within night window
    """
    night_start_time = datetime.strptime(NIGHT_START_UTC, TIME_FORMAT_HM).time()
    night_end_time = datetime.strptime(NIGHT_END_UTC, TIME_FORMAT_HM).time()
    
    start_time = start_dt.time()
    end_time = end_dt.time()
    
    # Handle midnight crossing
    if night_end_time < night_start_time:  # Window crosses midnight
        # Case 1: Interval doesn't cross midnight
        if start_time < end_time:
            return (start_time >= night_start_time or start_time < night_end_time) and \
                   (end_time > night_start_time or end_time <= night_end_time)
        # Case 2: Interval crosses midnight
        else:
            return (start_time >= night_start_time and end_time <= night_end_time)
    else:  # Normal window (doesn't cross midnight)
        return night_start_time <= start_time < night_end_time and \
               night_start_time < end_time <= night_end_time


def calculate_available_slots(night_start: datetime, night_end: datetime) -> int:
    """Calculate maximum possible observation slots per night."""
    night_duration = (night_end - night_start).total_seconds()
    if night_duration < 0:
        night_duration += 86400  # Handle midnight crossing
    
    slot_duration = OBS_DURATION_SEC + SLEW_GAP_SEC
    return int(night_duration // slot_duration)


def create_night_boundaries(date_str: str) -> Tuple[datetime, datetime]:
    """
    Create night start and end datetimes for a given date.
    Handles midnight crossing properly with UTC timezone.
    """
    try:
        # Parse the reference date
        ref_date = datetime.strptime(date_str, "%Y-%m-%d")
    except:
        # If no date provided, use today
        ref_date = datetime.utcnow()
    
    # Parse night times
    night_start_time = datetime.strptime(NIGHT_START_UTC, TIME_FORMAT_HM).time()
    night_end_time = datetime.strptime(NIGHT_END_UTC, TIME_FORMAT_HM).time()
    
    # Create datetime objects with UTC timezone
    try:
        import pytz
        tz = pytz.UTC
    except ImportError:
        # Fallback if pytz not available
        from datetime import timezone
        tz = timezone.utc
    
    night_start = datetime.combine(ref_date, night_start_time).replace(tzinfo=tz)
    
    # End time might be on next day if it crosses midnight
    if night_end_time < night_start_time:
        night_end = datetime.combine(ref_date + timedelta(days=1), night_end_time).replace(tzinfo=tz)
    else:
        night_end = datetime.combine(ref_date, night_end_time).replace(tzinfo=tz)
    
    # Ensure night_end is after night_start
    if night_end <= night_start:
        night_end = night_end + timedelta(days=1)
    
    return night_start, night_end


def plot_multi_day_timeline(schedules: Dict[int, pd.DataFrame], 
                           night_starts: Dict[int, datetime],
                           night_ends: Dict[int, datetime]):
    """
    Create Gantt-style timeline visualization for multi-day schedule.
    """
    fig, axes = plt.subplots(len(schedules), 1, figsize=(15, 3 * len(schedules)))
    if len(schedules) == 1:
        axes = [axes]
    
    # Create consistent color palette
    cmap = plt.cm.get_cmap('tab20c')
    
    for idx, (day, schedule) in enumerate(schedules.items()):
        ax = axes[idx]
        
        if schedule.empty:
            ax.text(0.5, 0.5, f'Day {day}: No observations scheduled',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue
        
        # Convert times to minutes since night start
        night_start = night_starts[day]
        schedule['start_seconds'] = schedule['observation_start_utc'].apply(
            lambda x: (parse_utc_datetime(x) - night_start).total_seconds()
        )
        schedule['end_seconds'] = schedule['observation_end_utc'].apply(
            lambda x: (parse_utc_datetime(x) - night_start).total_seconds()
        )
        
        # Sort by start time for plotting
        schedule = schedule.sort_values('start_seconds')
        
        # Plot each observation as a bar
        for i, row in schedule.iterrows():
            start_sec = row['start_seconds']
            end_sec = row['end_seconds']
            duration = end_sec - start_sec
            
            # Use NORAD ID for color consistency
            color_idx = int(row['norad_id']) % 20
            color = cmap(color_idx / 20)
            
            ax.barh(y=0, width=duration/60, left=start_sec/60, 
                   height=0.8, color=color, edgecolor='black')
            
            # Label with NORAD ID
            if duration/60 > 2:  # Only label if bar is wide enough
                ax.text(start_sec/60 + duration/120, 0, 
                       str(int(row['norad_id'])), 
                       ha='center', va='center', fontsize=8)
        
        # Set labels and limits
        ax.set_ylabel(f'Day {day}', fontsize=12)
        ax.set_xlabel('Time since night start (minutes)', fontsize=10)
        
        night_duration = (night_ends[day] - night_start).total_seconds()
        if night_duration < 0:
            night_duration += 86400
        
        ax.set_xlim(0, night_duration/60)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add night boundaries
        ax.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Night start')
        ax.axvline(x=night_duration/60, color='red', linestyle='--', alpha=0.5, label='Night end')
    
    plt.suptitle('Multi-Day Observation Timeline', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/multi_day_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_ga_progress(progress_data: Dict[int, List[float]]):
    """
    Plot GA convergence progress for each day.
    """
    fig, axes = plt.subplots(len(progress_data), 1, figsize=(10, 3 * len(progress_data)))
    if len(progress_data) == 1:
        axes = [axes]
    
    for idx, (day, fitness_values) in enumerate(progress_data.items()):
        ax = axes[idx]
        generations = list(range(1, len(fitness_values) + 1))
        
        ax.plot(generations, fitness_values, 'b-', linewidth=2, label='Best Fitness')
        ax.fill_between(generations, 
                       [min(fitness_values)] * len(fitness_values),
                       fitness_values, alpha=0.3)
        
        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel('Fitness (Scheduled Count)', fontsize=10)
        ax.set_title(f'Day {day}: GA Convergence', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Mark best value
        best_idx = np.argmax(fitness_values)
        ax.plot(generations[best_idx], fitness_values[best_idx], 
               'ro', markersize=8, label=f'Best: {fitness_values[best_idx]:.0f}')
    
    plt.tight_layout()
    plt.savefig('outputs/ga_progress_daywise.png', dpi=150, bbox_inches='tight')
    plt.close()