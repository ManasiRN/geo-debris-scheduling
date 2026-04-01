#!/usr/bin/env python3
"""
Analyze why debris remain unscheduled.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# Load the combined schedule
schedule = pd.read_csv('outputs/multi_day_schedule.csv')
scheduled_ids = set(schedule['norad_id'])

# Load original data
debris_data = pd.read_csv('data/raw/geo_visibility.csv')
debris_data['Start Time (UTC)'] = pd.to_datetime(debris_data['Start Time (UTC)'])
debris_data['End Time (UTC)'] = pd.to_datetime(debris_data['End Time (UTC)'])

# Find unscheduled
unscheduled = debris_data[~debris_data['NORAD ID'].astype(str).isin(scheduled_ids)]

print(f"Total debris: {len(debris_data)}")
print(f"Scheduled: {len(scheduled_ids)}")
print(f"Unscheduled: {len(unscheduled)}")

print("\n=== ANALYSIS OF UNSCHEDULED DEBRIS ===")

# Calculate visibility duration
unscheduled['duration_hours'] = (unscheduled['End Time (UTC)'] - 
                                 unscheduled['Start Time (UTC)']) / pd.Timedelta(hours=1)

print("\n1. Visibility Duration Statistics:")
print(unscheduled['duration_hours'].describe())

print("\n2. Start Time Distribution:")
unscheduled['start_hour'] = unscheduled['Start Time (UTC)'].dt.hour + \
                           unscheduled['Start Time (UTC)'].dt.minute / 60
print(unscheduled['start_hour'].describe())

print("\n3. End Time Distribution:")
unscheduled['end_hour'] = unscheduled['End Time (UTC)'].dt.hour + \
                         unscheduled['End Time (UTC)'].dt.minute / 60
print(unscheduled['end_hour'].describe())

print("\n4. Shortest Windows (< 1 hour):")
short_windows = unscheduled[unscheduled['duration_hours'] < 1]
print(f"Count: {len(short_windows)}")
print(short_windows[['NORAD ID', 'Start Time (UTC)', 'End Time (UTC)', 'duration_hours']].head(10))