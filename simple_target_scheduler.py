"""
Simple but effective target scheduler for GEO debris observation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add config to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config.constants import *
    print(f"✓ Loaded configuration")
except ImportError as e:
    print(f"ERROR: Could not load configuration: {e}")
    print("Please run setup_project.py first")
    sys.exit(1)

# Import required modules
try:
    import pytz
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytz"])
    import pytz


class SimpleScheduler:
    """Simple scheduler that maximizes debris count per night."""
    
    def __init__(self):
        self.obs_duration = OBS_DURATION_SEC
        self.slew_gap = SLEW_GAP_SEC
        self.slot_duration = self.obs_duration + self.slew_gap
        
        print(f"Observation: {self.obs_duration}s, Slew: {self.slew_gap}s")
        print(f"Slot duration: {self.slot_duration}s")
    
    def max_slots_per_night(self) -> int:
        """Calculate maximum possible observations per night."""
        night_duration = 12 * 3600  # 12 hours in seconds
        return int(night_duration // self.slot_duration)
    
    def create_night_boundaries(self, night_date: str) -> Tuple[datetime, datetime]:
        """Create night start and end datetimes."""
        try:
            import pytz
            tz = pytz.UTC
        except ImportError:
            from datetime import timezone
            tz = timezone.utc
        
        ref_date = datetime.strptime(night_date, "%Y-%m-%d")
        
        night_start_time = datetime.strptime(NIGHT_START_UTC, TIME_FORMAT_HM).time()
        night_end_time = datetime.strptime(NIGHT_END_UTC, TIME_FORMAT_HM).time()
        
        night_start = datetime.combine(ref_date, night_start_time).replace(tzinfo=tz)
        night_end = datetime.combine(ref_date + timedelta(days=1), night_end_time).replace(tzinfo=tz)
        
        return night_start, night_end
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess data."""
        print(f"\nLoading data from {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} debris objects")
        
        # Parse timestamps
        df['visibility_start_utc'] = pd.to_datetime(df['Start Time (UTC)'], utc=True)
        df['visibility_end_utc'] = pd.to_datetime(df['End Time (UTC)'], utc=True)
        
        # Standardize column names
        df.rename(columns={
            'NORAD ID': 'norad_id',
            'Satellite Name': 'satellite_name',
            'Peak Elevation (deg)': 'peak_elevation'
        }, inplace=True)
        
        return df
    
    def shift_visibility(self, df: pd.DataFrame, night_offset: int) -> pd.DataFrame:
        """Shift visibility windows for subsequent nights."""
        if night_offset == 0:
            return df.copy()
        
        shifted_df = df.copy()
        shift_hours = 24 * night_offset
        
        shifted_df['visibility_start_utc'] = shifted_df['visibility_start_utc'] + pd.Timedelta(hours=shift_hours)
        shifted_df['visibility_end_utc'] = shifted_df['visibility_end_utc'] + pd.Timedelta(hours=shift_hours)
        
        return shifted_df
    
    def filter_by_night_window(self, df: pd.DataFrame, night_start: datetime, 
                              night_end: datetime) -> pd.DataFrame:
        """Filter debris visible during a specific night."""
        results = []
        
        for idx, row in df.iterrows():
            vis_start = row['visibility_start_utc']
            vis_end = row['visibility_end_utc']
            
            # Check if visibility window overlaps with night window
            if vis_start < night_end and vis_end > night_start:
                # Clip to night boundaries
                clipped_start = max(vis_start, night_start)
                clipped_end = min(vis_end, night_end)
                
                # Check if clipped window is long enough
                clipped_duration = (clipped_end - clipped_start).total_seconds()
                if clipped_duration >= self.obs_duration:
                    new_row = row.copy()
                    new_row['visibility_start_utc'] = clipped_start
                    new_row['visibility_end_utc'] = clipped_end
                    new_row['clipped_duration_sec'] = clipped_duration
                    results.append(new_row)
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def greedy_schedule_night(self, debris_df: pd.DataFrame, night_start: datetime,
                            night_end: datetime) -> pd.DataFrame:
        """
        Greedy scheduling for one night.
        Returns schedule DataFrame.
        """
        if debris_df.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying original
        df = debris_df.copy()
        
        # Sort by: 1) earliest visibility end, 2) highest peak elevation
        df = df.sort_values(['visibility_end_utc', 'peak_elevation'], 
                          ascending=[True, False])
        
        schedule_rows = []
        current_time = night_start
        max_slots = self.max_slots_per_night()
        scheduled_count = 0
        
        for idx, debris in df.iterrows():
            if scheduled_count >= max_slots:
                break  # Physical limit reached
            
            vis_start = debris['visibility_start_utc']
            vis_end = debris['visibility_end_utc']
            
            # If current time is before visibility start, wait
            if current_time < vis_start:
                current_time = vis_start
            
            # Check if we can schedule this debris
            proposed_end = current_time + timedelta(seconds=self.obs_duration)
            
            if proposed_end <= vis_end and proposed_end <= night_end:
                # Schedule this observation
                obs_start = current_time
                obs_end = proposed_end
                
                schedule_rows.append({
                    'norad_id': debris['norad_id'],
                    'satellite_name': debris['satellite_name'],
                    'observation_start_utc': obs_start.strftime(TIME_FORMAT),
                    'observation_end_utc': obs_end.strftime(TIME_FORMAT),
                    'peak_elevation': debris['peak_elevation']
                })
                
                # Advance time for next observation
                current_time = obs_end + timedelta(seconds=self.slew_gap)
                scheduled_count += 1
                
                # If we've reached night end, stop
                if current_time >= night_end:
                    break
        
        return pd.DataFrame(schedule_rows) if schedule_rows else pd.DataFrame()
    
    def optimize_with_ga(self, debris_df: pd.DataFrame, night_start: datetime,
                        night_end: datetime, max_iterations: int = 100) -> pd.DataFrame:
        """
        Simple optimization using random search (simplified GA).
        """
        if debris_df.empty or len(debris_df) < 10:
            return self.greedy_schedule_night(debris_df, night_start, night_end)
        
        print(f"  Running optimization ({len(debris_df)} debris)...")
        
        best_schedule = None
        best_count = 0
        
        # Try different random orders
        for iteration in range(max_iterations):
            # Create random order
            indices = list(range(len(debris_df)))
            random.shuffle(indices)
            
            # Schedule with this order
            ordered_df = debris_df.iloc[indices].copy()
            schedule = self.greedy_schedule_night(ordered_df, night_start, night_end)
            
            # Check if this is better
            scheduled_count = len(schedule)
            if scheduled_count > best_count:
                best_count = scheduled_count
                best_schedule = schedule
                
                # Early exit if we hit maximum
                max_possible = self.max_slots_per_night()
                if best_count >= max_possible:
                    print(f"    Iteration {iteration+1}: Hit maximum {max_possible}")
                    break
            
            # Progress reporting
            if (iteration + 1) % 20 == 0:
                print(f"    Iteration {iteration+1}: Best so far = {best_count}")
        
        print(f"  Optimization completed: {best_count} debris scheduled")
        return best_schedule if best_schedule is not None else pd.DataFrame()
    
    def schedule_all_nights(self, debris_df: pd.DataFrame, max_nights: int = 5, 
                           use_optimization: bool = True) -> Dict:
        """
        Schedule across all nights.
        """
        print(f"\n{'='*60}")
        print(f"SCHEDULING ALL NIGHTS")
        print(f"{'='*60}")
        
        total_debris = len(debris_df)
        remaining_df = debris_df.copy()
        all_schedules = {}
        night_stats = {}
        
        print(f"Total debris: {total_debris}")
        print(f"Maximum per night: {self.max_slots_per_night()}")
        print(f"Maximum in {max_nights} nights: {self.max_slots_per_night() * max_nights}")
        
        for night in range(1, max_nights + 1):
            if remaining_df.empty:
                print(f"\nAll debris scheduled. Completed in {night-1} nights.")
                break
            
            # Calculate date for this night
            night_date = (datetime.strptime("2025-10-07", "%Y-%m-%d") + 
                         timedelta(days=night-1)).strftime("%Y-%m-%d")
            
            # Create night boundaries
            night_start, night_end = self.create_night_boundaries(night_date)
            
            print(f"\n{'='*40}")
            print(f"Night {night} ({night_date})")
            print(f"{'='*40}")
            print(f"Remaining debris: {len(remaining_df)}")
            
            # Shift and filter debris for this night
            night_debris = self.shift_visibility(remaining_df, night-1)
            night_debris = self.filter_by_night_window(night_debris, night_start, night_end)
            
            print(f"Visible this night: {len(night_debris)}")
            
            if night_debris.empty:
                print("No debris visible this night")
                continue
            
            # Schedule this night
            if use_optimization:
                schedule = self.optimize_with_ga(night_debris, night_start, night_end)
            else:
                schedule = self.greedy_schedule_night(night_debris, night_start, night_end)
            
            if schedule.empty:
                print("No debris could be scheduled this night")
                continue
            
            scheduled_count = len(schedule)
            print(f"Scheduled: {scheduled_count} debris")
            
            # Calculate statistics
            avg_elevation = schedule['peak_elevation'].mean()
            print(f"Average elevation: {avg_elevation:.1f}°")
            
            # Store results
            all_schedules[night] = schedule
            night_stats[night] = {
                'date': night_date,
                'scheduled': scheduled_count,
                'remaining_before': len(remaining_df),
                'avg_elevation': avg_elevation
            }
            
            # Remove scheduled debris
            scheduled_ids = set(schedule['norad_id'])
            remaining_df = remaining_df[~remaining_df['norad_id'].isin(scheduled_ids)]
            
            # Save per-night schedule
            self.save_night_schedule(schedule, night)
        
        return {
            'schedules': all_schedules,
            'stats': night_stats,
            'remaining': remaining_df,
            'total_debris': total_debris
        }
    
    def save_night_schedule(self, schedule: pd.DataFrame, night: int):
        """Save schedule for a specific night."""
        if schedule.empty:
            return
        
        output_dir = "outputs/simple"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f"{output_dir}/schedule_night{night}.csv"
        schedule.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
    
    def generate_summary_report(self, results: Dict):
        """Generate summary report."""
        print(f"\n{'='*60}")
        print(f"SCHEDULING SUMMARY")
        print(f"{'='*60}")
        
        schedules = results['schedules']
        stats = results['stats']
        remaining = results['remaining']
        total_debris = results['total_debris']
        
        total_scheduled = sum(len(sched) for sched in schedules.values())
        nights_used = len(schedules)
        
        print(f"\nOverall Results:")
        print(f"  Total debris: {total_debris}")
        print(f"  Scheduled: {total_scheduled} ({total_scheduled/total_debris*100:.1f}%)")
        print(f"  Remaining: {len(remaining)} ({len(remaining)/total_debris*100:.1f}%)")
        print(f"  Nights used: {nights_used}")
        
        if total_scheduled > 0:
            print(f"\nNight-by-Night Breakdown:")
            for night, stat in stats.items():
                print(f"  Night {night}: {stat['scheduled']} debris, "
                      f"Avg elevation: {stat['avg_elevation']:.1f}°")
        
        # Calculate physical limits
        max_per_night = self.max_slots_per_night()
        total_capacity = max_per_night * nights_used
        utilization = (total_scheduled / total_capacity * 100) if total_capacity > 0 else 0
        
        print(f"\nPhysical Limits Analysis:")
        print(f"  Maximum per night: {max_per_night}")
        print(f"  Total capacity used: {total_scheduled}/{total_capacity}")
        print(f"  Capacity utilization: {utilization:.1f}%")
        
        if len(remaining) > 0:
            print(f"\nWhy some debris couldn't be scheduled:")
            print(f"  1. Physical limit: Max {max_per_night} per night")
            print(f"  2. Limited to {nights_used} nights (out of 5)")
            print(f"  3. Visibility window conflicts")
            print(f"  4. Need for more optimization")
        
        # Save summary
        summary = {
            'total_debris': total_debris,
            'total_scheduled': total_scheduled,
            'remaining': len(remaining),
            'coverage_percentage': total_scheduled/total_debris*100,
            'nights_used': nights_used,
            'nightly_stats': stats,
            'physical_limits': {
                'max_per_night': max_per_night,
                'total_capacity': total_capacity,
                'utilization_percentage': utilization
            }
        }
        
        summary_file = "outputs/simple/scheduling_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nSummary saved to {summary_file}")
        
        # Also save combined schedule
        self.save_combined_schedule(schedules)
        
        return summary
    
    def save_combined_schedule(self, schedules: Dict):
        """Save all schedules combined."""
        if not schedules:
            return
        
        combined_rows = []
        for night, schedule in schedules.items():
            for _, row in schedule.iterrows():
                combined_row = row.copy()
                combined_row['night'] = night
                combined_rows.append(combined_row)
        
        if combined_rows:
            combined_df = pd.DataFrame(combined_rows)
            combined_file = "outputs/simple/combined_schedule.csv"
            combined_df.to_csv(combined_file, index=False)
            print(f"Combined schedule saved to {combined_file}")


def analyze_visibility_data(df: pd.DataFrame):
    """Analyze visibility data characteristics."""
    print(f"\n{'='*60}")
    print(f"VISIBILITY DATA ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate window durations
    df['window_duration_sec'] = (df['visibility_end_utc'] - df['visibility_start_utc']).dt.total_seconds()
    
    print(f"Total debris: {len(df)}")
    print(f"\nVisibility Window Statistics:")
    print(f"  Minimum: {df['window_duration_sec'].min():.0f}s ({df['window_duration_sec'].min()/60:.1f} min)")
    print(f"  Maximum: {df['window_duration_sec'].max():.0f}s ({df['window_duration_sec'].max()/60:.1f} min)")
    print(f"  Average: {df['window_duration_sec'].mean():.0f}s ({df['window_duration_sec'].mean()/60:.1f} min)")
    print(f"  Median: {df['window_duration_sec'].median():.0f}s ({df['window_duration_sec'].median()/60:.1f} min)")
    
    # Count by duration categories
    categories = {
        'Very short (<5 min)': (df['window_duration_sec'] < 300).sum(),
        'Short (5-15 min)': ((df['window_duration_sec'] >= 300) & (df['window_duration_sec'] < 900)).sum(),
        'Medium (15-60 min)': ((df['window_duration_sec'] >= 900) & (df['window_duration_sec'] < 3600)).sum(),
        'Long (1-3 hours)': ((df['window_duration_sec'] >= 3600) & (df['window_duration_sec'] < 10800)).sum(),
        'Very long (>3 hours)': (df['window_duration_sec'] >= 10800).sum()
    }
    
    print(f"\nWindow Duration Categories:")
    for category, count in categories.items():
        percentage = count / len(df) * 100
        print(f"  {category}: {count} debris ({percentage:.1f}%)")
    
    # Elevation statistics
    print(f"\nElevation Statistics:")
    print(f"  Minimum: {df['peak_elevation'].min():.1f}°")
    print(f"  Maximum: {df['peak_elevation'].max():.1f}°")
    print(f"  Average: {df['peak_elevation'].mean():.1f}°")
    print(f"  Median: {df['peak_elevation'].median():.1f}°")
    
    # Time distribution
    df['start_hour'] = df['visibility_start_utc'].dt.hour + df['visibility_start_utc'].dt.minute / 60
    df['end_hour'] = df['visibility_end_utc'].dt.hour + df['visibility_end_utc'].dt.minute / 60
    
    print(f"\nTime Distribution:")
    print(f"  Earliest start: {df['start_hour'].min():.2f} hours")
    print(f"  Latest end: {df['end_hour'].max():.2f} hours")
    
    # Check night window overlap
    night_start_hour = 12.5  # 12:30
    night_end_hour = 24.5    # 00:30 next day
    
    overlapping = df[
        (df['start_hour'] < night_end_hour) & 
        (df['end_hour'] > night_start_hour)
    ]
    
    print(f"\nNight Window Overlap (12:30-00:30 UTC):")
    print(f"  Debris overlapping night window: {len(overlapping)}/{len(df)} ({len(overlapping)/len(df)*100:.1f}%)")
    
    return df


def main():
    """Main entry point."""
    print("="*60)
    print("GEO SATELLITE DEBRIS - SIMPLE SCHEDULER")
    print("="*60)
    
    # Create output directory
    os.makedirs("outputs/simple", exist_ok=True)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Simple GEO Scheduler')
    parser.add_argument('--date', type=str, default="2025-10-07",
                       help='Start date for scheduling')
    parser.add_argument('--data', type=str, default="data/raw/geo_visibility.csv",
                       help='Path to input CSV')
    parser.add_argument('--nights', type=int, default=5,
                       help='Maximum nights')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Disable optimization (use greedy only)')
    
    args = parser.parse_args()
    
    print(f"Start date: {args.date}")
    print(f"Data file: {args.data}")
    print(f"Maximum nights: {args.nights}")
    print(f"Optimization: {'Enabled' if not args.no_optimize else 'Disabled'}")
    
    try:
        # Create scheduler
        scheduler = SimpleScheduler()
        
        # Load data
        debris_df = scheduler.load_data(args.data)
        
        if debris_df.empty:
            print("No debris loaded. Exiting.")
            return
        
        # Analyze data
        analyze_visibility_data(debris_df)
        
        # Run scheduling
        print(f"\n{'='*60}")
        print(f"STARTING SCHEDULING")
        print(f"{'='*60}")
        
        results = scheduler.schedule_all_nights(
            debris_df, 
            max_nights=args.nights,
            use_optimization=not args.no_optimize
        )
        
        # Generate summary
        summary = scheduler.generate_summary_report(results)
        
        # Final message
        print(f"\n{'='*60}")
        print(f"SCHEDULING COMPLETE")
        print(f"{'='*60}")
        
        if summary['remaining'] == 0:
            print(f"✅ SUCCESS: All {summary['total_debris']} debris scheduled!")
            print(f"   Used {summary['nights_used']} nights")
        else:
            print(f"⚠️  PARTIAL: {summary['total_scheduled']}/{summary['total_debris']} debris scheduled")
            print(f"   {summary['remaining']} debris remaining")
            print(f"   Used {summary['nights_used']} of {args.nights} nights")
        
        print(f"\nOutput files in outputs/simple/:")
        print(f"  - schedule_nightX.csv (individual schedules)")
        print(f"  - combined_schedule.csv (all schedules combined)")
        print(f"  - scheduling_summary.json (detailed statistics)")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()