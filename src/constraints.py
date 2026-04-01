"""
Constraint checking and validation module.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd

from config.constants import *
from src.utils import parse_utc_datetime


class ScheduleValidator:
    """Validator for schedule constraints."""
    
    @staticmethod
    def validate_schedule(schedule: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate a complete schedule against all constraints.
        
        Args:
            schedule: DataFrame with scheduled observations
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        if schedule.empty:
            return True, violations
        
        # 1. Check observation duration
        for _, row in schedule.iterrows():
            start_time = parse_utc_datetime(row['observation_start_utc'])
            end_time = parse_utc_datetime(row['observation_end_utc'])
            duration = (end_time - start_time).total_seconds()
            
            if abs(duration - OBS_DURATION_SEC) > 1:  # Allow 1-second tolerance
                violations.append(f"Observation duration violation: {duration}s (expected {OBS_DURATION_SEC}s)")
        
        # 2. Check non-overlap
        schedule_sorted = schedule.sort_values('observation_start_utc')
        prev_end = None
        
        for _, row in schedule_sorted.iterrows():
            start_time = parse_utc_datetime(row['observation_start_utc'])
            
            if prev_end is not None:
                gap = (start_time - prev_end).total_seconds()
                if gap < SLEW_GAP_SEC - 1:  # Allow 1-second tolerance
                    violations.append(f"Insufficient slew gap: {gap}s (minimum {SLEW_GAP_SEC}s)")
            
            end_time = parse_utc_datetime(row['observation_end_utc'])
            prev_end = end_time
        
        # 3. Check within visibility windows
        if 'visibility_start_utc' in schedule.columns and 'visibility_end_utc' in schedule.columns:
            for _, row in schedule.iterrows():
                obs_start = parse_utc_datetime(row['observation_start_utc'])
                obs_end = parse_utc_datetime(row['observation_end_utc'])
                vis_start = parse_utc_datetime(row['visibility_start_utc'])
                vis_end = parse_utc_datetime(row['visibility_end_utc'])
                
                if obs_start < vis_start:
                    violations.append(f"Observation starts before visibility window")
                if obs_end > vis_end:
                    violations.append(f"Observation ends after visibility window")
        
        # 4. Check within night window
        night_start_time = datetime.strptime(NIGHT_START_UTC, TIME_FORMAT_HM).time()
        night_end_time = datetime.strptime(NIGHT_END_UTC, TIME_FORMAT_HM).time()
        
        for _, row in schedule.iterrows():
            obs_start = parse_utc_datetime(row['observation_start_utc'])
            obs_end = parse_utc_datetime(row['observation_end_utc'])
            
            start_time = obs_start.time()
            end_time = obs_end.time()
            
            # Handle midnight crossing
            if night_end_time < night_start_time:
                if not (start_time >= night_start_time or start_time < night_end_time):
                    violations.append(f"Observation start outside night window: {start_time}")
                if not (end_time > night_start_time or end_time <= night_end_time):
                    violations.append(f"Observation end outside night window: {end_time}")
            else:
                if not (night_start_time <= start_time < night_end_time):
                    violations.append(f"Observation start outside night window: {start_time}")
                if not (night_start_time < end_time <= night_end_time):
                    violations.append(f"Observation end outside night window: {end_time}")
        
        # 5. Check unique debris (for single night)
        norad_ids = schedule['norad_id'].tolist()
        if len(norad_ids) != len(set(norad_ids)):
            violations.append("Duplicate debris observations")
        
        return len(violations) == 0, violations
    
    @staticmethod
    def validate_multi_day_schedule(schedules: Dict[int, pd.DataFrame]) -> Tuple[bool, List[str]]:
        """
        Validate multi-day schedule.
        
        Args:
            schedules: Dict mapping day number to schedule
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check each day's schedule
        for day, schedule in schedules.items():
            is_valid, day_violations = ScheduleValidator.validate_schedule(schedule)
            if not is_valid:
                for violation in day_violations:
                    violations.append(f"Day {day}: {violation}")
        
        # Check debris appears at most once across all days
        all_norad_ids = []
        for schedule in schedules.values():
            all_norad_ids.extend(schedule['norad_id'].tolist())
        
        if len(all_norad_ids) != len(set(all_norad_ids)):
            violations.append("Debris observed multiple times across different days")
        
        return len(violations) == 0, violations


class TimeSlotChecker:
    """Check time slot feasibility."""
    
    @staticmethod
    def can_schedule(proposed_start: datetime, 
                    visibility_start: datetime,
                    visibility_end: datetime,
                    current_time: datetime = None) -> Tuple[bool, str]:
        """
        Check if a time slot is feasible for observation.
        
        Args:
            proposed_start: Proposed observation start time
            visibility_start: Visibility window start
            visibility_end: Visibility window end
            current_time: Current telescope time (for slew gap check)
            
        Returns:
            Tuple of (is_feasible, reason)
        """
        # Check within visibility window
        if proposed_start < visibility_start:
            return False, "Before visibility window"
        
        proposed_end = proposed_start + timedelta(seconds=OBS_DURATION_SEC)
        if proposed_end > visibility_end:
            return False, "After visibility window"
        
        # Check slew gap if current_time provided
        if current_time is not None:
            gap = (proposed_start - current_time).total_seconds()
            if gap < SLEW_GAP_SEC:
                return False, f"Insufficient slew gap: {gap}s"
        
        # Check within night window
        from src.utils import is_within_night_window
        if not is_within_night_window(proposed_start, proposed_end):
            return False, "Outside night window"
        
        return True, "Feasible"