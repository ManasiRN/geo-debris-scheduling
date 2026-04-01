"""
Main orchestrator for GEO satellite debris observation scheduling.
Handles single-day visibility data by shifting visibility windows to subsequent nights.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
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
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytz", "matplotlib", "seaborn"])
    import pytz
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates


class MultiNightScheduler:
    """Scheduler that can handle single-day visibility data by shifting windows."""
    
    def __init__(self, start_date: str = "2025-10-07"):
        self.start_date = start_date
        self.obs_duration = OBS_DURATION_SEC
        self.slew_gap = SLEW_GAP_SEC
        self.slot_duration = self.obs_duration + self.slew_gap
        
        print(f"Starting from date: {start_date}")
        print(f"Maximum slots per night: {self.max_slots_per_night()}")
    
    def max_slots_per_night(self):
        """Calculate maximum possible observations per night."""
        # Calculate night duration (12 hours from 12:30 to 00:30)
        night_duration = 12 * 3600  # 12 hours in seconds
        return int(night_duration // self.slot_duration)
    
    def create_night_boundaries(self, night_date: str):
        """Create night start and end datetimes for a specific date."""
        ref_date = datetime.strptime(night_date, "%Y-%m-%d")
        
        # Parse night times
        night_start_time = datetime.strptime(NIGHT_START_UTC, TIME_FORMAT_HM).time()
        night_end_time = datetime.strptime(NIGHT_END_UTC, TIME_FORMAT_HM).time()
        
        # Create UTC timezone-aware datetimes
        night_start = datetime.combine(ref_date, night_start_time).replace(tzinfo=pytz.UTC)
        
        # End time is on next day (crosses midnight)
        night_end = datetime.combine(ref_date + timedelta(days=1), night_end_time).replace(tzinfo=pytz.UTC)
        
        return night_start, night_end
    
    def load_data(self, filepath: str = "data/raw/geo_visibility.csv"):
        """Load and preprocess data."""
        print(f"\nLoading data from {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Load CSV
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
        
        # Check date range
        start_dates = df['visibility_start_utc'].dt.date.unique()
        end_dates = df['visibility_end_utc'].dt.date.unique()
        
        print(f"Visibility date range:")
        print(f"  Start dates: {sorted(start_dates)}")
        print(f"  End dates: {sorted(end_dates)}")
        
        # Check if all data is for a single day
        if len(start_dates) == 1:
            data_date = start_dates[0]
            print(f"\n⚠️  NOTE: All visibility data is for a single date: {data_date}")
            print(f"   The scheduler will assume similar visibility patterns on subsequent nights.")
            print(f"   For accurate multi-night scheduling, visibility data for multiple days is needed.")
        
        return df
    
    def shift_visibility_for_night(self, df: pd.DataFrame, night_offset: int) -> pd.DataFrame:
        """
        Shift visibility windows for subsequent nights.
        This is a workaround when we only have single-day visibility data.
        
        Args:
            df: Original DataFrame with visibility windows
            night_offset: Number of nights to shift (0 for first night, 1 for second, etc.)
            
        Returns:
            DataFrame with shifted visibility windows
        """
        if night_offset == 0:
            return df.copy()
        
        shifted_df = df.copy()
        
        # Shift visibility windows by 24 hours for each night offset
        shift_hours = 24 * night_offset
        
        shifted_df['visibility_start_utc'] = shifted_df['visibility_start_utc'] + pd.Timedelta(hours=shift_hours)
        shifted_df['visibility_end_utc'] = shifted_df['visibility_end_utc'] + pd.Timedelta(hours=shift_hours)
        
        return shifted_df
    
    def filter_by_night_window(self, df: pd.DataFrame, night_start: datetime, 
                              night_end: datetime) -> pd.DataFrame:
        """Filter debris that are visible during a specific night."""
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
        
        filtered_df = pd.DataFrame(results) if results else pd.DataFrame()
        
        return filtered_df
    
    def greedy_schedule_night(self, debris_df: pd.DataFrame, night_start: datetime,
                            night_end: datetime) -> pd.DataFrame:
        """Greedy scheduling for one night."""
        if debris_df.empty:
            return pd.DataFrame()
        
        # Sort by end time (earliest first), then by elevation (highest first)
        df = debris_df.copy()
        df = df.sort_values(['visibility_end_utc', 'peak_elevation'], 
                          ascending=[True, False])
        
        schedule = []
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
                
                schedule.append({
                    'norad_id': debris['norad_id'],
                    'satellite_name': debris['satellite_name'],
                    'observation_start_utc': obs_start.strftime(TIME_FORMAT),
                    'observation_end_utc': obs_end.strftime(TIME_FORMAT),
                    'peak_elevation': debris['peak_elevation'],
                    'visibility_start_utc': debris['visibility_start_utc'].strftime(TIME_FORMAT),
                    'visibility_end_utc': debris['visibility_end_utc'].strftime(TIME_FORMAT)
                })
                
                # Advance time for next observation
                current_time = obs_end + timedelta(seconds=self.slew_gap)
                scheduled_count += 1
                
                # If we've reached night end, stop
                if current_time >= night_end:
                    break
        
        return pd.DataFrame(schedule)
    
    def schedule_multiple_nights(self, debris_df: pd.DataFrame, max_nights: int = 5) -> Dict:
        """Schedule across multiple nights."""
        print(f"\n{'='*60}")
        print(f"MULTI-NIGHT SCHEDULING (max {max_nights} nights)")
        print(f"{'='*60}")
        
        remaining_df = debris_df.copy()
        all_schedules = {}
        night_boundaries = {}
        
        for night in range(1, max_nights + 1):
            if remaining_df.empty:
                print(f"\nAll debris scheduled. Completed in {night-1} nights.")
                break
            
            # Calculate date for this night
            night_date = (datetime.strptime(self.start_date, "%Y-%m-%d") + 
                         timedelta(days=night-1)).strftime("%Y-%m-%d")
            
            # Create night boundaries for this night
            night_start, night_end = self.create_night_boundaries(night_date)
            night_boundaries[night] = (night_start, night_end)
            
            print(f"\nNight {night} ({night_date}):")
            print(f"  Night window: {night_start.strftime('%Y-%m-%d %H:%M')} to {night_end.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Remaining debris: {len(remaining_df)}")
            
            # For multi-night scheduling with single-day data, shift visibility windows
            night_debris = self.shift_visibility_for_night(remaining_df, night - 1)
            
            # Filter debris for this night (after shifting)
            night_debris = self.filter_by_night_window(night_debris, night_start, night_end)
            
            print(f"  Visible this night: {len(night_debris)}")
            
            if night_debris.empty:
                print(f"  No debris visible this night")
                if night == 1:
                    print(f"  ERROR: No debris visible on first night!")
                    print(f"  Check that visibility windows overlap with night window (12:30-00:30 UTC)")
                break
            
            # Schedule this night
            night_schedule = self.greedy_schedule_night(night_debris, night_start, night_end)
            
            if night_schedule.empty:
                print(f"  Could not schedule any debris this night")
                print(f"  This may be due to visibility window constraints")
                continue
            
            print(f"  Scheduled: {len(night_schedule)} debris")
            
            # Store schedule
            all_schedules[night] = night_schedule
            
            # Remove scheduled debris from remaining
            scheduled_ids = set(night_schedule['norad_id'])
            remaining_df = remaining_df[~remaining_df['norad_id'].isin(scheduled_ids)]
            
            # Save per-night schedule
            self.save_night_schedule(night, night_schedule)
            
            # Stop if we've reached the maximum we can schedule in one night
            # This prevents unnecessary iterations
            if len(night_schedule) < self.max_slots_per_night() * 0.5:  # Less than 50% capacity used
                print(f"  Note: Low scheduling efficiency on this night")
        
        return all_schedules, remaining_df, night_boundaries
    
    def save_night_schedule(self, night: int, schedule: pd.DataFrame):
        """Save schedule for a specific night."""
        if schedule.empty:
            return
        
        output_cols = ['norad_id', 'satellite_name', 'observation_start_utc',
                      'observation_end_utc', 'peak_elevation']
        
        output_file = f"outputs/schedule_night{night}.csv"
        schedule[output_cols].to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
    
    def combine_schedules(self, all_schedules: Dict):
        """Combine all night schedules into one."""
        if not all_schedules:
            return pd.DataFrame()
        
        combined = []
        for night, schedule in all_schedules.items():
            schedule_with_day = schedule.copy()
            schedule_with_day['night'] = night
            combined.append(schedule_with_day)
        
        combined_df = pd.concat(combined, ignore_index=True)
        
        # Save combined schedule
        output_file = "outputs/combined_schedule.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\n✓ Combined schedule saved to {output_file}")
        
        return combined_df
    
    def generate_statistics(self, all_schedules: Dict, initial_count: int, 
                          remaining_df: pd.DataFrame) -> Dict:
        """Generate scheduling statistics."""
        total_scheduled = sum(len(sched) for sched in all_schedules.values())
        max_per_night = self.max_slots_per_night()
        
        stats = {
            'total_debris': initial_count,
            'scheduled_debris': total_scheduled,
            'remaining_debris': len(remaining_df),
            'nights_used': len(all_schedules),
            'scheduled_per_night': {night: len(sched) 
                                   for night, sched in all_schedules.items()},
            'coverage_percentage': (total_scheduled / initial_count * 100 
                                  if initial_count > 0 else 0),
            'physical_limit_per_night': max_per_night,
            'max_possible_in_5_nights': max_per_night * 5,
            'data_assumption': 'single_day_shifted' if initial_count > 0 else 'unknown'
        }
        
        # Calculate slot usage efficiency
        total_slots_used = total_scheduled
        total_slots_available = max_per_night * len(all_schedules) if all_schedules else 0
        stats['slot_efficiency_percentage'] = (total_slots_used / total_slots_available * 100) if total_slots_available > 0 else 0
        
        # Save statistics
        stats_file = "outputs/scheduling_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"✓ Statistics saved to {stats_file}")
        
        return stats
    
    def plot_schedule(self, all_schedules: Dict, night_boundaries: Dict):
        """Create a visualization of the schedule."""
        if not all_schedules:
            print("No schedule to plot")
            return
        
        fig, axes = plt.subplots(len(all_schedules), 1, figsize=(14, 3 * len(all_schedules)))
        if len(all_schedules) == 1:
            axes = [axes]
        
        # Create color map for debris
        cmap = plt.cm.get_cmap('tab20c')
        
        for idx, (night, schedule) in enumerate(all_schedules.items()):
            ax = axes[idx]
            
            if schedule.empty:
                ax.text(0.5, 0.5, f'Night {night}: No observations',
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            night_start, night_end = night_boundaries[night]
            
            # Convert to hours since night start
            schedule['start_hours'] = pd.to_datetime(schedule['observation_start_utc']).apply(
                lambda x: (x - night_start).total_seconds() / 3600
            )
            schedule['duration_hours'] = self.obs_duration / 3600
            
            # Sort by start time for better visualization
            schedule = schedule.sort_values('start_hours')
            
            # Plot each observation
            for i, row in schedule.iterrows():
                # Use NORAD ID for consistent color
                color_idx = int(row['norad_id']) % 20
                color = cmap(color_idx / 20)
                
                ax.barh(y=0, width=row['duration_hours'], left=row['start_hours'],
                       height=0.5, edgecolor='black', alpha=0.7, color=color)
                
                # Add NORAD ID label if space allows
                if row['duration_hours'] > 0.05:  # More than 3 minutes
                    ax.text(row['start_hours'] + row['duration_hours']/2, 0,
                           str(int(row['norad_id'])), ha='center', va='center',
                           fontsize=6, color='white', fontweight='bold')
            
            ax.set_xlabel(f'Night {night}: Hours since {night_start.strftime("%H:%M")} UTC')
            ax.set_yticks([])
            ax.set_xlim(0, (night_end - night_start).total_seconds() / 3600)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add night boundaries
            ax.axvline(x=0, color='blue', linestyle='--', alpha=0.5, linewidth=0.5)
            ax.axvline(x=(night_end - night_start).total_seconds() / 3600, 
                      color='red', linestyle='--', alpha=0.5, linewidth=0.5)
        
        plt.suptitle('GEO Satellite Observation Schedule', fontsize=14, y=1.02)
        plt.tight_layout()
        
        plot_file = "outputs/schedule_timeline.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Schedule visualization saved to {plot_file}")
        
        # Also create a summary bar chart
        self.plot_summary_chart(all_schedules)
    
    def plot_summary_chart(self, all_schedules: Dict):
        """Create summary bar chart."""
        if not all_schedules:
            return
        
        nights = list(all_schedules.keys())
        counts = [len(sched) for sched in all_schedules.values()]
        max_capacity = self.max_slots_per_night()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(nights, counts, color='steelblue', edgecolor='black')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Night Number', fontsize=12)
        ax.set_ylabel('Number of Observations', fontsize=12)
        ax.set_title('Observations per Night', fontsize=14)
        ax.set_xticks(nights)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line for maximum capacity
        ax.axhline(y=max_capacity, color='red', linestyle='--', alpha=0.7, 
                  label=f'Max Capacity: {max_capacity}')
        
        # Add target line for total debris
        total_debris = sum(counts)
        ax.axhline(y=total_debris/len(nights) if nights else 0, color='green', 
                  linestyle=':', alpha=0.7, label=f'Average: {total_debris/len(nights):.0f}')
        
        ax.legend()
        
        plt.tight_layout()
        
        summary_file = "outputs/summary_chart.png"
        plt.savefig(summary_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Summary chart saved to {summary_file}")


def main():
    """Main entry point."""
    print("="*60)
    print("GEO SATELLITE DEBRIS OBSERVATION SCHEDULER")
    print("="*60)
    print("NOTE: This version handles single-day visibility data")
    print("      by shifting visibility windows to subsequent nights.")
    print("="*60)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='GEO Satellite Debris Scheduler')
    parser.add_argument('--date', type=str, default="2025-10-07",
                       help='Start date for scheduling (YYYY-MM-DD)')
    parser.add_argument('--data', type=str, default="data/raw/geo_visibility.csv",
                       help='Path to input CSV file')
    parser.add_argument('--nights', type=int, default=5,
                       help='Maximum number of nights to schedule')
    
    args = parser.parse_args()
    
    print(f"Start date: {args.date}")
    print(f"Data file: {args.data}")
    print(f"Maximum nights: {args.nights}")
    print(f"Observation duration: {OBS_DURATION_SEC}s")
    print(f"Slew/settle time: {SLEW_GAP_SEC}s")
    print(f"Slot duration: {OBS_DURATION_SEC + SLEW_GAP_SEC}s")
    
    try:
        # Create scheduler
        scheduler = MultiNightScheduler(start_date=args.date)
        
        # Load data
        debris_df = scheduler.load_data(args.data)
        
        if debris_df.empty:
            print("No debris loaded. Exiting.")
            return
        
        print(f"\nStarting scheduling for {len(debris_df)} debris...")
        
        # Schedule across multiple nights
        all_schedules, remaining_df, night_boundaries = scheduler.schedule_multiple_nights(
            debris_df, max_nights=args.nights
        )
        
        if not all_schedules:
            print("\nNo debris could be scheduled. Check night window constraints.")
            return
        
        # Combine schedules
        combined_df = scheduler.combine_schedules(all_schedules)
        
        # Generate statistics
        stats = scheduler.generate_statistics(
            all_schedules, len(debris_df), remaining_df
        )
        
        # Create visualizations
        scheduler.plot_schedule(all_schedules, night_boundaries)
        
        # Print summary
        print("\n" + "="*60)
        print("SCHEDULING COMPLETE")
        print("="*60)
        print(f"\nSummary:")
        print(f"  Total debris: {stats['total_debris']}")
        print(f"  Scheduled: {stats['scheduled_debris']}")
        print(f"  Remaining: {stats['remaining_debris']}")
        print(f"  Nights used: {stats['nights_used']}")
        print(f"  Coverage: {stats['coverage_percentage']:.1f}%")
        
        print(f"\nPer night breakdown:")
        for night, count in stats['scheduled_per_night'].items():
            print(f"  Night {night}: {count} observations")
        
        print(f"\nPhysical limits:")
        print(f"  Max per night: {stats['physical_limit_per_night']}")
        print(f"  Max in {args.nights} nights: {stats['max_possible_in_5_nights']}")
        
        if stats['remaining_debris'] > 0:
            print(f"\n⚠️  {stats['remaining_debris']} debris could not be scheduled")
            print("   Possible reasons:")
            print("   1. Visibility windows too short or don't align with available slots")
            print("   2. Overlapping windows causing scheduling conflicts")
            print("   3. Reached maximum number of nights (5)")
            if stats['data_assumption'] == 'single_day_shifted':
                print("   4. Using shifted visibility windows (single-day data assumption)")
        
        print(f"\nEfficiency: {stats['slot_efficiency_percentage']:.1f}% of available slots used")
        
        print(f"\nOutput files in 'outputs/' directory:")
        print("  - schedule_nightX.csv (individual night schedules)")
        print("  - combined_schedule.csv (all observations)")
        print("  - schedule_timeline.png (detailed timeline)")
        print("  - summary_chart.png (summary bar chart)")
        print("  - scheduling_statistics.json (detailed stats)")
        
        if stats['data_assumption'] == 'single_day_shifted':
            print(f"\n📝 IMPORTANT NOTE:")
            print(f"   Used single-day visibility data with time-shifted windows.")
            print(f"   For accurate multi-night scheduling, provide visibility data")
            print(f"   computed for each night separately.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()