"""
Fixed Genetic Algorithm scheduler focusing on quality optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import random
import math
from typing import Dict, List, Tuple, Optional, Set
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
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytz", "matplotlib", "seaborn"])
    import pytz
    import matplotlib.pyplot as plt


class QualityFocusedGAScheduler:
    """
    Genetic Algorithm scheduler that focuses on quality optimization.
    """
    
    def __init__(self, population_size: int = GA_POPULATION_SIZE, 
                 generations: int = GA_GENERATIONS):
        self.population_size = population_size
        self.generations = generations
        self.obs_duration = OBS_DURATION_SEC
        self.slew_gap = SLEW_GAP_SEC
        self.slot_duration = self.obs_duration + self.slew_gap
        
        # GA parameters
        self.tournament_size = GA_TOURNAMENT_SIZE
        self.mutation_rate = GA_MUTATION_RATE
        self.elite_size = max(2, population_size // 10)
        self.early_stop_patience = GA_EARLY_STOP_PATIENCE
        
        # Fitness weights (α >> β >> γ)
        self.alpha = 1000.0    # Weight for debris count (primary)
        self.beta = 1.0        # Weight for elevation (secondary)
        self.gamma = 0.01      # Weight for idle time (tertiary)
        self.delta = 0.1       # Weight for tight window preference
        
        print(f"GA Fitness Weights: α={self.alpha}, β={self.beta}, γ={self.gamma}, δ={self.delta}")
    
    def max_slots_per_night(self):
        """Calculate maximum possible observations per night."""
        night_duration = 12 * 3600  # 12 hours
        return int(night_duration // self.slot_duration)
    
    class Chromosome:
        """Chromosome for GA representing debris order."""
        
        def __init__(self, genes: List[int] = None, num_genes: int = None):
            if genes is not None:
                self.genes = genes.copy()
                self.num_genes = len(genes)
            elif num_genes is not None:
                self.num_genes = num_genes
                self.genes = list(range(num_genes))
                random.shuffle(self.genes)
            else:
                raise ValueError("Either genes or num_genes must be provided")
        
        def __len__(self):
            return len(self.genes)
        
        def copy(self):
            return self.__class__(genes=self.genes)
        
        @classmethod
        def create_random(cls, num_genes: int):
            genes = list(range(num_genes))
            random.shuffle(genes)
            return cls(genes=genes)
    
    def greedy_schedule_with_order(self, debris_df: pd.DataFrame, order: List[int],
                                  night_start: datetime, night_end: datetime) -> Tuple[pd.DataFrame, Dict]:
        """
        Schedule using greedy algorithm with specific order.
        Returns schedule and quality metrics.
        """
        if debris_df.empty or not order:
            return pd.DataFrame(), {}
        
        # Reorder debris according to chromosome
        ordered_df = debris_df.iloc[order].reset_index(drop=True)
        
        # Calculate window tightness (for preference)
        ordered_df['window_tightness'] = (
            (ordered_df['visibility_end_utc'] - ordered_df['visibility_start_utc']).dt.total_seconds()
        ) / 3600  # in hours
        
        # Greedy scheduling
        schedule_rows = []
        current_time = night_start
        max_slots = self.max_slots_per_night()
        scheduled_count = 0
        
        # Track quality metrics
        idle_times = []
        elevation_sum = 0.0
        tight_window_count = 0
        
        for idx, debris in ordered_df.iterrows():
            if scheduled_count >= max_slots:
                break
            
            vis_start = debris['visibility_start_utc']
            vis_end = debris['visibility_end_utc']
            
            # Check for idle time before this observation
            if current_time < vis_start:
                idle_time = (vis_start - current_time).total_seconds()
                idle_times.append(idle_time)
                current_time = vis_start
            
            # Check if we can schedule this debris
            proposed_end = current_time + timedelta(seconds=self.obs_duration)
            
            if proposed_end <= vis_end and proposed_end <= night_end:
                obs_start = current_time
                obs_end = proposed_end
                
                schedule_rows.append({
                    'norad_id': debris['norad_id'],
                    'satellite_name': debris['satellite_name'],
                    'observation_start_utc': obs_start.strftime(TIME_FORMAT),
                    'observation_end_utc': obs_end.strftime(TIME_FORMAT),
                    'peak_elevation': debris['peak_elevation']
                })
                
                # Update quality metrics
                elevation_sum += debris['peak_elevation']
                
                # Count tight windows (less than 30 minutes)
                if debris['window_tightness'] < 0.5:  # 30 minutes
                    tight_window_count += 1
                
                # Advance time for next observation
                current_time = obs_end + timedelta(seconds=self.slew_gap)
                scheduled_count += 1
                
                if current_time >= night_end:
                    break
        
        schedule_df = pd.DataFrame(schedule_rows) if schedule_rows else pd.DataFrame()
        
        # Calculate quality metrics
        total_idle_time = sum(idle_times) if idle_times else 0
        
        # Calculate end-of-night idle time
        if schedule_rows and current_time < night_end:
            total_idle_time += (night_end - current_time).total_seconds()
        
        scheduled_count_val = len(schedule_df)
        
        metrics = {
            'scheduled_count': scheduled_count_val,
            'total_elevation': elevation_sum,
            'avg_elevation': elevation_sum / scheduled_count_val if scheduled_count_val > 0 else 0,
            'total_idle_time': total_idle_time,
            'avg_idle_time': total_idle_time / max(1, len(idle_times)),
            'tight_window_count': tight_window_count,
            'night_utilization': (scheduled_count_val * self.slot_duration) / 
                               (night_end - night_start).total_seconds() * 100
        }
        
        return schedule_df, metrics
    
    def evaluate_fitness(self, chromosome, debris_df: pd.DataFrame,
                        night_start: datetime, night_end: datetime) -> float:
        """
        Evaluate fitness with quality-focused metrics.
        
        fitness = α*scheduled_count + β*elevation_score - γ*idle_time + δ*tight_window_bonus
        where α >> β >> γ >> δ
        """
        # Simulate scheduling with this chromosome order
        schedule, metrics = self.greedy_schedule_with_order(
            debris_df, chromosome.genes, night_start, night_end
        )
        
        # Primary term: debris count (dominant)
        primary_score = self.alpha * metrics['scheduled_count']
        
        # Secondary term: elevation (normalized to [0, 1])
        max_possible_elevation = debris_df['peak_elevation'].max() * metrics['scheduled_count']
        elevation_score = (metrics['total_elevation'] / max_possible_elevation 
                          if max_possible_elevation > 0 else 0)
        
        # Tertiary term: idle time (penalty, normalized)
        night_duration = (night_end - night_start).total_seconds()
        idle_penalty = (metrics['total_idle_time'] / night_duration 
                       if night_duration > 0 else 0)
        
        # Quaternary term: tight window preference
        tight_window_bonus = (metrics['tight_window_count'] / metrics['scheduled_count'] 
                            if metrics['scheduled_count'] > 0 else 0)
        
        # Combined fitness
        fitness = (
            primary_score +
            self.beta * elevation_score -
            self.gamma * idle_penalty +
            self.delta * tight_window_bonus
        )
        
        # Store metrics for analysis
        chromosome.metrics = metrics
        
        return fitness
    
    def tournament_selection(self, population, fitness_values):
        """Tournament selection."""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        
        best_idx = tournament_indices[0]
        best_fitness = fitness_values[best_idx]
        
        for idx in tournament_indices[1:]:
            if fitness_values[idx] > best_fitness:
                best_idx = idx
                best_fitness = fitness_values[idx]
        
        return population[best_idx].copy()
    
    def order_crossover(self, parent1, parent2):
        """Order Crossover (OX)."""
        n = len(parent1)
        cx1, cx2 = sorted(random.sample(range(n), 2))
        
        child1_genes = [-1] * n
        child2_genes = [-1] * n
        
        # Copy segment between cx1 and cx2
        child1_genes[cx1:cx2] = parent1.genes[cx1:cx2]
        child2_genes[cx1:cx2] = parent2.genes[cx1:cx2]
        
        # Fill remaining positions
        child1_pos = cx2
        child2_pos = cx2
        
        for i in range(n):
            parent2_idx = (cx2 + i) % n
            parent1_idx = (cx2 + i) % n
            
            if parent2.genes[parent2_idx] not in child1_genes:
                child1_genes[child1_pos] = parent2.genes[parent2_idx]
                child1_pos = (child1_pos + 1) % n
            
            if parent1.genes[parent1_idx] not in child2_genes:
                child2_genes[child2_pos] = parent1.genes[parent1_idx]
                child2_pos = (child2_pos + 1) % n
        
        child1 = self.Chromosome(genes=child1_genes)
        child2 = self.Chromosome(genes=child2_genes)
        
        return child1, child2
    
    def swap_mutation(self, chromosome):
        """Swap mutation."""
        if random.random() > self.mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        n = len(mutated)
        
        i, j = random.sample(range(n), 2)
        mutated.genes[i], mutated.genes[j] = mutated.genes[j], mutated.genes[i]
        
        return mutated
    
    def run_ga(self, debris_df: pd.DataFrame, night_start: datetime,
              night_end: datetime) -> Tuple[pd.DataFrame, Dict, List[float]]:
        """
        Run Genetic Algorithm optimization.
        Returns schedule, metrics, and fitness history.
        """
        print(f"  Starting quality-focused GA...")
        print(f"  Debris count: {len(debris_df)}")
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            population.append(self.Chromosome.create_random(len(debris_df)))
        
        # Evaluate initial population
        fitness_values = []
        for chrom in population:
            fitness = self.evaluate_fitness(chrom, debris_df, night_start, night_end)
            fitness_values.append(fitness)
        
        best_idx = np.argmax(fitness_values)
        best_chromosome = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        best_metrics = getattr(best_chromosome, 'metrics', {})
        
        fitness_history = [best_fitness]
        metrics_history = [best_metrics.copy() if best_metrics else {}]
        
        # Evolution loop
        no_improvement_count = 0
        
        for generation in range(self.generations):
            new_population = []
            
            # Elitism: keep best solutions
            elite_indices = np.argsort(fitness_values)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Fill rest of population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                
                # Crossover
                if random.random() < 0.8:
                    child1, child2 = self.order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self.swap_mutation(child1)
                child2 = self.swap_mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            # Evaluate new population
            fitness_values = []
            for chrom in population:
                fitness = self.evaluate_fitness(chrom, debris_df, night_start, night_end)
                fitness_values.append(fitness)
            
            # Update best solution
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            current_metrics = getattr(population[current_best_idx], 'metrics', {})
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = population[current_best_idx].copy()
                best_metrics = current_metrics.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            fitness_history.append(best_fitness)
            metrics_history.append(best_metrics.copy() if best_metrics else {})
            
            # Early stopping
            if no_improvement_count >= self.early_stop_patience:
                print(f"  Early stopping at generation {generation+1}")
                break
            
            # Progress reporting
            if (generation + 1) % 20 == 0:
                count = best_metrics.get('scheduled_count', 0)
                elev = best_metrics.get('avg_elevation', 0)
                idle = best_metrics.get('total_idle_time', 0)
                print(f"    Gen {generation+1}: Best={count} debris, "
                      f"Elev={elev:.1f}°, Idle={idle:.0f}s")
        
        # Generate schedule from best chromosome
        best_schedule, final_metrics = self.greedy_schedule_with_order(
            debris_df, best_chromosome.genes, night_start, night_end
        )
        
        print(f"  GA completed:")
        print(f"    Scheduled: {final_metrics['scheduled_count']} debris")
        print(f"    Avg elevation: {final_metrics['avg_elevation']:.1f}°")
        print(f"    Total idle time: {final_metrics['total_idle_time']:.0f}s")
        print(f"    Night utilization: {final_metrics['night_utilization']:.1f}%")
        
        return best_schedule, final_metrics, fitness_history


class GreedyBaselineScheduler:
    """Greedy baseline scheduler for comparison."""
    
    def __init__(self):
        self.obs_duration = OBS_DURATION_SEC
        self.slew_gap = SLEW_GAP_SEC
        self.slot_duration = self.obs_duration + self.slew_gap
    
    def max_slots_per_night(self):
        """Calculate maximum possible observations per night."""
        night_duration = 12 * 3600
        return int(night_duration // self.slot_duration)
    
    def create_night_boundaries(self, night_date: str):
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
    
    def schedule_night(self, debris_df: pd.DataFrame, night_start: datetime,
                      night_end: datetime) -> pd.DataFrame:
        """Greedy scheduling for one night."""
        if debris_df.empty:
            return pd.DataFrame()
        
        # Sort by end time (earliest first), then by elevation (highest first)
        df = debris_df.copy()
        df = df.sort_values(['visibility_end_utc', 'peak_elevation'], 
                          ascending=[True, False])
        
        schedule_rows = []
        current_time = night_start
        max_slots = self.max_slots_per_night()
        scheduled_count = 0
        
        for idx, debris in df.iterrows():
            if scheduled_count >= max_slots:
                break
            
            vis_start = debris['visibility_start_utc']
            vis_end = debris['visibility_end_utc']
            
            if current_time < vis_start:
                current_time = vis_start
            
            proposed_end = current_time + timedelta(seconds=self.obs_duration)
            
            if proposed_end <= vis_end and proposed_end <= night_end:
                obs_start = current_time
                obs_end = proposed_end
                
                schedule_rows.append({
                    'norad_id': debris['norad_id'],
                    'satellite_name': debris['satellite_name'],
                    'observation_start_utc': obs_start.strftime(TIME_FORMAT),
                    'observation_end_utc': obs_end.strftime(TIME_FORMAT),
                    'peak_elevation': debris['peak_elevation']
                })
                
                current_time = obs_end + timedelta(seconds=self.slew_gap)
                scheduled_count += 1
                
                if current_time >= night_end:
                    break
        
        return pd.DataFrame(schedule_rows) if schedule_rows else pd.DataFrame()


class QualityComparisonScheduler:
    """Main scheduler that compares Greedy vs Quality-focused GA."""
    
    def __init__(self, start_date: str = "2025-10-07"):
        self.start_date = start_date
        self.greedy_scheduler = GreedyBaselineScheduler()
        self.ga_scheduler = QualityFocusedGAScheduler()
        
        print(f"Starting from date: {start_date}")
        print(f"Maximum slots per night: {self.greedy_scheduler.max_slots_per_night()}")
        print(f"GA focus: Quality optimization when count cannot be improved")
    
    def load_data(self, filepath: str = "data/raw/geo_visibility.csv"):
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
            
            if vis_start < night_end and vis_end > night_start:
                clipped_start = max(vis_start, night_start)
                clipped_end = min(vis_end, night_end)
                
                clipped_duration = (clipped_end - clipped_start).total_seconds()
                if clipped_duration >= self.greedy_scheduler.obs_duration:
                    new_row = row.copy()
                    new_row['visibility_start_utc'] = clipped_start
                    new_row['visibility_end_utc'] = clipped_end
                    new_row['clipped_duration_sec'] = clipped_duration
                    results.append(new_row)
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def calculate_quality_metrics(self, schedule: pd.DataFrame, 
                                 night_start: datetime, night_end: datetime) -> Dict:
        """Calculate quality metrics for a schedule."""
        if schedule.empty:
            return {
                'scheduled_count': 0,
                'avg_elevation': 0,
                'total_idle_time': 0,
                'night_utilization': 0,
                'avg_idle_per_slot': 0
            }
        
        # Calculate average elevation
        avg_elevation = schedule['peak_elevation'].mean()
        
        # Calculate idle time
        if not schedule.empty:
            schedule_times = pd.to_datetime(schedule['observation_start_utc'])
            schedule_times_sorted = np.sort(schedule_times)
            
            idle_time = 0
            current_time = night_start
            
            for obs_start in schedule_times_sorted:
                if current_time < obs_start:
                    idle_time += (obs_start - current_time).total_seconds()
                
                # Move to next slot
                current_time = obs_start + timedelta(
                    seconds=self.greedy_scheduler.obs_duration + self.greedy_scheduler.slew_gap
                )
            
            # Add end-of-night idle time
            if current_time < night_end:
                idle_time += (night_end - current_time).total_seconds()
            
            # Calculate night utilization
            total_scheduled_time = len(schedule) * self.greedy_scheduler.slot_duration
            night_duration = (night_end - night_start).total_seconds()
            utilization = (total_scheduled_time / night_duration) * 100 if night_duration > 0 else 0
            
            return {
                'scheduled_count': len(schedule),
                'avg_elevation': avg_elevation,
                'total_idle_time': idle_time,
                'night_utilization': utilization,
                'avg_idle_per_slot': idle_time / max(1, len(schedule))
            }
        else:
            return {
                'scheduled_count': 0,
                'avg_elevation': 0,
                'total_idle_time': 0,
                'night_utilization': 0,
                'avg_idle_per_slot': 0
            }
    
    def run_quality_comparison(self, debris_df: pd.DataFrame, max_nights: int = 5) -> Dict:
        """Run quality-focused comparison between Greedy and GA."""
        print(f"\n{'='*60}")
        print(f"QUALITY-FOCUSED COMPARISON (max {max_nights} nights)")
        print(f"{'='*60}")
        
        # Results storage
        results = {
            'greedy': {'schedules': {}, 'statistics': {}, 'quality_metrics': {}},
            'ga': {'schedules': {}, 'statistics': {}, 'quality_metrics': {}, 
                  'fitness_history': {}, 'comparisons': {}}
        }
        
        # Run greedy first to get baseline
        print(f"\n{'='*40}")
        print(f"Running GREEDY algorithm (baseline)")
        print(f"{'='*40}")
        
        greedy_remaining = debris_df.copy()
        
        for night in range(1, max_nights + 1):
            if greedy_remaining.empty:
                break
            
            night_date = (datetime.strptime(self.start_date, "%Y-%m-%d") + 
                         timedelta(days=night-1)).strftime("%Y-%m-%d")
            
            night_start, night_end = self.greedy_scheduler.create_night_boundaries(night_date)
            
            # Shift and filter debris
            night_debris = self.shift_visibility(greedy_remaining, night-1)
            night_debris = self.filter_by_night_window(night_debris, night_start, night_end)
            
            if night_debris.empty:
                print(f"\nNight {night}: No debris visible")
                continue
            
            print(f"\nNight {night}: {len(night_debris)} debris visible")
            
            # Run greedy
            greedy_schedule = self.greedy_scheduler.schedule_night(
                night_debris, night_start, night_end
            )
            
            print(f"  Greedy: {len(greedy_schedule)} debris scheduled")
            
            results['greedy']['schedules'][night] = greedy_schedule
            
            # Calculate quality metrics for greedy
            greedy_metrics = self.calculate_quality_metrics(
                greedy_schedule, night_start, night_end
            )
            results['greedy']['quality_metrics'][night] = greedy_metrics
            
            # Remove scheduled debris
            if not greedy_schedule.empty:
                scheduled_ids = set(greedy_schedule['norad_id'])
                greedy_remaining = greedy_remaining[~greedy_remaining['norad_id'].isin(scheduled_ids)]
        
        # Run GA with greedy baseline for comparison
        print(f"\n{'='*40}")
        print(f"Running QUALITY-FOCUSED GA")
        print(f"{'='*40}")
        
        ga_remaining = debris_df.copy()
        
        for night in range(1, max_nights + 1):
            if ga_remaining.empty:
                break
            
            night_date = (datetime.strptime(self.start_date, "%Y-%m-%d") + 
                         timedelta(days=night-1)).strftime("%Y-%m-%d")
            
            night_start, night_end = self.greedy_scheduler.create_night_boundaries(night_date)
            
            # Shift and filter debris
            night_debris = self.shift_visibility(ga_remaining, night-1)
            night_debris = self.filter_by_night_window(night_debris, night_start, night_end)
            
            if night_debris.empty:
                print(f"\nNight {night}: No debris visible")
                continue
            
            print(f"\nNight {night}: {len(night_debris)} debris visible")
            
            # Get greedy baseline for this night if available
            greedy_baseline = results['greedy']['schedules'].get(night, pd.DataFrame())
            
            # Run quality-focused GA
            ga_schedule, ga_metrics, fitness_history = self.ga_scheduler.run_ga(
                night_debris, night_start, night_end
            )
            
            print(f"  GA: {len(ga_schedule)} debris scheduled")
            
            results['ga']['schedules'][night] = ga_schedule
            results['ga']['quality_metrics'][night] = ga_metrics
            results['ga']['fitness_history'][night] = fitness_history
            
            # Compare with greedy
            comparison = self.compare_schedules(greedy_baseline, ga_schedule, 
                                               greedy_metrics, ga_metrics)
            results['ga']['comparisons'][night] = comparison
            
            # Remove scheduled debris
            if not ga_schedule.empty:
                scheduled_ids = set(ga_schedule['norad_id'])
                ga_remaining = ga_remaining[~ga_remaining['norad_id'].isin(scheduled_ids)]
        
        # Calculate statistics
        self.calculate_statistics(results, debris_df)
        
        return results
    
    def compare_schedules(self, greedy_schedule: pd.DataFrame, ga_schedule: pd.DataFrame,
                         greedy_metrics: Dict, ga_metrics: Dict) -> Dict:
        """Compare greedy and GA schedules."""
        greedy_count = len(greedy_schedule)
        ga_count = len(ga_schedule)
        
        comparison = {
            'greedy_count': greedy_count,
            'ga_count': ga_count,
            'count_difference': ga_count - greedy_count,
            'count_improvement': ga_count > greedy_count,
            'count_equal': ga_count == greedy_count
        }
        
        # Compare quality metrics
        if greedy_count > 0 and ga_count > 0:
            comparison.update({
                'greedy_avg_elevation': greedy_metrics['avg_elevation'],
                'ga_avg_elevation': ga_metrics['avg_elevation'],
                'greedy_idle_time': greedy_metrics['total_idle_time'],
                'ga_idle_time': ga_metrics['total_idle_time'],
                'elevation_difference': ga_metrics['avg_elevation'] - greedy_metrics['avg_elevation'],
                'idle_time_difference': greedy_metrics['total_idle_time'] - ga_metrics['total_idle_time'],
                'elevation_improved': ga_metrics['avg_elevation'] > greedy_metrics['avg_elevation'],
                'idle_time_improved': ga_metrics['total_idle_time'] < greedy_metrics['total_idle_time'],
                'quality_improved': (ga_metrics['avg_elevation'] > greedy_metrics['avg_elevation'] or
                                   ga_metrics['total_idle_time'] < greedy_metrics['total_idle_time'])
            })
        
        return comparison
    
    def calculate_statistics(self, results: Dict, debris_df: pd.DataFrame):
        """Calculate overall statistics."""
        total_debris = len(debris_df)
        
        for algorithm in ['greedy', 'ga']:
            schedules = results[algorithm]['schedules']
            
            total_scheduled = sum(len(sched) for sched in schedules.values())
            nights_used = len(schedules)
            
            # Calculate overall quality metrics
            all_elevations = []
            total_idle_time = 0
            
            for night, schedule in schedules.items():
                if not schedule.empty:
                    all_elevations.extend(schedule['peak_elevation'].tolist())
                
                # Add idle time from quality metrics
                if night in results[algorithm]['quality_metrics']:
                    metrics = results[algorithm]['quality_metrics'][night]
                    total_idle_time += metrics.get('total_idle_time', 0)
            
            avg_elevation = np.mean(all_elevations) if all_elevations else 0
            
            stats = {
                'total_scheduled': total_scheduled,
                'remaining': total_debris - total_scheduled,
                'nights_used': nights_used,
                'coverage_percentage': (total_scheduled / total_debris * 100) if total_debris > 0 else 0,
                'avg_elevation': avg_elevation,
                'total_idle_time': total_idle_time,
                'scheduled_per_night': {night: len(sched) for night, sched in schedules.items()}
            }
            
            results[algorithm]['statistics'] = stats
    
    def save_results(self, results: Dict):
        """Save all results to files."""
        print(f"\n{'='*60}")
        print(f"SAVING RESULTS")
        print(f"{'='*60}")
        
        # Create directories
        for algo in ['greedy', 'ga']:
            os.makedirs(f"outputs/{algo}", exist_ok=True)
            os.makedirs(f"outputs/{algo}/quality", exist_ok=True)
        
        # Save schedules
        for algorithm in ['greedy', 'ga']:
            schedules = results[algorithm]['schedules']
            
            for night, schedule in schedules.items():
                if not schedule.empty:
                    # Save schedule
                    schedule_file = f"outputs/{algorithm}/schedule_night{night}.csv"
                    schedule.to_csv(schedule_file, index=False)
                    
                    # Save quality metrics
                    if night in results[algorithm]['quality_metrics']:
                        metrics = results[algorithm]['quality_metrics'][night]
                        metrics_file = f"outputs/{algorithm}/quality/night{night}_metrics.json"
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics, f, indent=2, default=str)
            
            # Save fitness history for GA
            if algorithm == 'ga' and results['ga']['fitness_history']:
                fitness_file = "outputs/ga/fitness_history.json"
                with open(fitness_file, 'w') as f:
                    # Convert to list for JSON serialization
                    fitness_data = {}
                    for night, history in results['ga']['fitness_history'].items():
                        fitness_data[str(night)] = history
                    json.dump(fitness_data, f, indent=2)
        
        # Save comparison results
        comparison_results = {
            'greedy_statistics': results['greedy']['statistics'],
            'ga_statistics': results['ga']['statistics'],
            'nightly_comparisons': results['ga']['comparisons']
        }
        
        comparison_file = "outputs/quality_comparison_results.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        print(f"✓ Schedules saved to outputs/greedy/ and outputs/ga/")
        print(f"✓ Quality metrics saved to outputs/*/quality/")
        print(f"✓ Comparison results saved to {comparison_file}")
    
    def generate_quality_report(self, results: Dict, debris_df: pd.DataFrame):
        """Generate quality-focused comparison report."""
        print(f"\n{'='*60}")
        print(f"QUALITY COMPARISON REPORT")
        print(f"{'='*60}")
        
        greedy_stats = results['greedy']['statistics']
        ga_stats = results['ga']['statistics']
        
        # Count improvements
        count_improvements = 0
        quality_improvements = 0
        total_nights = max(len(results['greedy']['schedules']), 
                          len(results['ga']['schedules']))
        
        for night in range(1, total_nights + 1):
            comparison = results['ga']['comparisons'].get(night, {})
            if comparison.get('count_improvement', False):
                count_improvements += 1
            if comparison.get('quality_improved', False):
                quality_improvements += 1
        
        # Generate report
        report = {
            'summary': {
                'total_debris': len(debris_df),
                'greedy_scheduled': greedy_stats['total_scheduled'],
                'ga_scheduled': ga_stats['total_scheduled'],
                'count_difference': ga_stats['total_scheduled'] - greedy_stats['total_scheduled'],
                'greedy_nights': greedy_stats['nights_used'],
                'ga_nights': ga_stats['nights_used'],
                'greedy_avg_elevation': greedy_stats['avg_elevation'],
                'ga_avg_elevation': ga_stats['avg_elevation'],
                'greedy_idle_time': greedy_stats['total_idle_time'],
                'ga_idle_time': ga_stats['total_idle_time']
            },
            'improvements': {
                'count_improvement_nights': count_improvements,
                'quality_improvement_nights': quality_improvements,
                'total_nights_compared': total_nights,
                'count_improved': ga_stats['total_scheduled'] > greedy_stats['total_scheduled'],
                'quality_improved': (ga_stats['avg_elevation'] > greedy_stats['avg_elevation'] or
                                   ga_stats['total_idle_time'] < greedy_stats['total_idle_time'])
            }
        }
        
        # Save report
        report_file = "outputs/quality_comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✓ Quality report saved to {report_file}")
        
        # Generate visualizations
        self.plot_quality_comparison(results, report)
        
        return report
    
    def plot_quality_comparison(self, results: Dict, report: Dict):
        """Create quality comparison visualizations."""
        # 1. Fitness convergence plots
        if results['ga']['fitness_history']:
            nights_with_history = list(results['ga']['fitness_history'].keys())
            
            if nights_with_history:
                fig, axes = plt.subplots(len(nights_with_history), 1, 
                                        figsize=(12, 3 * len(nights_with_history)))
                if len(nights_with_history) == 1:
                    axes = [axes]
                
                for idx, night in enumerate(nights_with_history):
                    ax = axes[idx]
                    fitness_history = results['ga']['fitness_history'][night]
                    
                    if fitness_history:
                        generations = range(1, len(fitness_history) + 1)
                        
                        ax.plot(generations, fitness_history, 'b-', linewidth=2)
                        ax.fill_between(generations, 0, fitness_history, alpha=0.3)
                        ax.set_xlabel('Generation', fontsize=10)
                        ax.set_ylabel('Fitness', fontsize=10)
                        ax.set_title(f'Night {night}: GA Fitness Convergence', fontsize=12)
                        ax.grid(True, alpha=0.3)
                
                plt.suptitle('GA Fitness Convergence by Night', fontsize=14, y=1.02)
                plt.tight_layout()
                plt.savefig('outputs/ga_fitness_convergence.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ GA fitness convergence plots saved")
        
        # 2. Quality metrics comparison
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Subplot 1: Scheduled count comparison
            ax1 = axes[0, 0]
            nights = list(range(1, max(len(results['greedy']['schedules']), 
                                     len(results['ga']['schedules'])) + 1))
            
            greedy_counts = [len(results['greedy']['schedules'].get(n, pd.DataFrame())) for n in nights]
            ga_counts = [len(results['ga']['schedules'].get(n, pd.DataFrame())) for n in nights]
            
            x = np.arange(len(nights))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, greedy_counts, width, label='Greedy', color='skyblue')
            bars2 = ax1.bar(x + width/2, ga_counts, width, label='GA', color='lightcoral')
            
            ax1.set_xlabel('Night', fontsize=12)
            ax1.set_ylabel('Debris Scheduled', fontsize=12)
            ax1.set_title('Scheduled Count per Night', fontsize=14)
            ax1.set_xticks(x)
            ax1.set_xticklabels(nights)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Subplot 2: Average elevation comparison
            ax2 = axes[0, 1]
            greedy_elevations = []
            ga_elevations = []
            
            for night in nights:
                if night in results['greedy']['quality_metrics']:
                    greedy_elevations.append(results['greedy']['quality_metrics'][night]['avg_elevation'])
                else:
                    greedy_elevations.append(0)
                
                if night in results['ga']['quality_metrics']:
                    ga_elevations.append(results['ga']['quality_metrics'][night]['avg_elevation'])
                else:
                    ga_elevations.append(0)
            
            bars3 = ax2.bar(x - width/2, greedy_elevations, width, label='Greedy', color='lightgreen')
            bars4 = ax2.bar(x + width/2, ga_elevations, width, label='GA', color='lightyellow')
            
            ax2.set_xlabel('Night', fontsize=12)
            ax2.set_ylabel('Average Elevation (°)', fontsize=12)
            ax2.set_title('Average Elevation per Night', fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels(nights)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Subplot 3: Idle time comparison
            ax3 = axes[1, 0]
            greedy_idle = []
            ga_idle = []
            
            for night in nights:
                if night in results['greedy']['quality_metrics']:
                    greedy_idle.append(results['greedy']['quality_metrics'][night]['total_idle_time'] / 60)  # minutes
                else:
                    greedy_idle.append(0)
                
                if night in results['ga']['quality_metrics']:
                    ga_idle.append(results['ga']['quality_metrics'][night]['total_idle_time'] / 60)  # minutes
                else:
                    ga_idle.append(0)
            
            bars5 = ax3.bar(x - width/2, greedy_idle, width, label='Greedy', color='lightgray')
            bars6 = ax3.bar(x + width/2, ga_idle, width, label='GA', color='wheat')
            
            ax3.set_xlabel('Night', fontsize=12)
            ax3.set_ylabel('Idle Time (minutes)', fontsize=12)
            ax3.set_title('Total Idle Time per Night', fontsize=14)
            ax3.set_xticks(x)
            ax3.set_xticklabels(nights)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Subplot 4: Overall comparison
            ax4 = axes[1, 1]
            categories = ['Scheduled', 'Avg Elevation', 'Idle Time']
            
            # Normalize values for comparison
            max_scheduled = max(report['summary']['greedy_scheduled'], report['summary']['ga_scheduled'])
            max_elevation = max(report['summary']['greedy_avg_elevation'], report['summary']['ga_avg_elevation'])
            max_idle = max(report['summary']['greedy_idle_time'], report['summary']['ga_idle_time'])
            
            greedy_normalized = [
                report['summary']['greedy_scheduled'] / max_scheduled if max_scheduled > 0 else 0,
                report['summary']['greedy_avg_elevation'] / max_elevation if max_elevation > 0 else 0,
                1 - (report['summary']['greedy_idle_time'] / max_idle if max_idle > 0 else 0)  # Lower idle is better
            ]
            
            ga_normalized = [
                report['summary']['ga_scheduled'] / max_scheduled if max_scheduled > 0 else 0,
                report['summary']['ga_avg_elevation'] / max_elevation if max_elevation > 0 else 0,
                1 - (report['summary']['ga_idle_time'] / max_idle if max_idle > 0 else 0)  # Lower idle is better
            ]
            
            x_cat = np.arange(len(categories))
            bars7 = ax4.bar(x_cat - width/2, greedy_normalized, width, label='Greedy', color='skyblue', alpha=0.7)
            bars8 = ax4.bar(x_cat + width/2, ga_normalized, width, label='GA', color='lightcoral', alpha=0.7)
            
            ax4.set_xlabel('Metrics', fontsize=12)
            ax4.set_ylabel('Normalized Score', fontsize=12)
            ax4.set_title('Overall Performance (Normalized)', fontsize=14)
            ax4.set_xticks(x_cat)
            ax4.set_xticklabels(categories)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Quality-Focused GA vs Greedy Comparison', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig('outputs/quality_comparison_charts.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Quality comparison charts saved")
        except Exception as e:
            print(f"Warning: Could not generate all charts: {e}")
    
    def print_quality_summary(self, report: Dict):
        """Print quality-focused summary."""
        print("\n" + "="*60)
        print("QUALITY-FOCUSED OPTIMIZATION SUMMARY")
        print("="*60)
        
        summary = report['summary']
        improvements = report['improvements']
        
        print(f"\nOverall Results:")
        print(f"  Total debris: {summary['total_debris']}")
        print(f"  Greedy scheduled: {summary['greedy_scheduled']} "
              f"({summary['greedy_scheduled']/summary['total_debris']*100:.1f}%)")
        print(f"  GA scheduled: {summary['ga_scheduled']} "
              f"({summary['ga_scheduled']/summary['total_debris']*100:.1f}%)")
        
        if summary['count_difference'] > 0:
            print(f"  ✅ GA scheduled {summary['count_difference']} MORE debris than Greedy")
        elif summary['count_difference'] < 0:
            print(f"  ⚠️  GA scheduled {-summary['count_difference']} FEWER debris than Greedy")
        else:
            print(f"  ⚖️  Both algorithms scheduled the SAME number of debris")
        
        print(f"\nQuality Metrics:")
        print(f"  Greedy avg elevation: {summary['greedy_avg_elevation']:.1f}°")
        print(f"  GA avg elevation: {summary['ga_avg_elevation']:.1f}°")
        
        elevation_diff = summary['ga_avg_elevation'] - summary['greedy_avg_elevation']
        if elevation_diff > 0:
            print(f"  ✅ GA achieved {elevation_diff:.1f}° HIGHER average elevation")
        elif elevation_diff < 0:
            print(f"  ⚠️  GA achieved {-elevation_diff:.1f}° LOWER average elevation")
        else:
            print(f"  ⚖️  Both algorithms achieved the SAME average elevation")
        
        print(f"\n  Greedy total idle time: {summary['greedy_idle_time']:.0f}s "
              f"({summary['greedy_idle_time']/3600:.2f}h)")
        print(f"  GA total idle time: {summary['ga_idle_time']:.0f}s "
              f"({summary['ga_idle_time']/3600:.2f}h)")
        
        idle_diff = summary['greedy_idle_time'] - summary['ga_idle_time']
        if idle_diff > 0:
            print(f"  ✅ GA reduced idle time by {idle_diff:.0f}s ({idle_diff/60:.1f} minutes)")
        elif idle_diff < 0:
            print(f"  ⚠️  GA increased idle time by {-idle_diff:.0f}s ({-idle_diff/60:.1f} minutes)")
        else:
            print(f"  ⚖️  Both algorithms had the SAME idle time")
        
        print(f"\nNight-by-Night Improvements:")
        print(f"  Nights where GA scheduled more debris: {improvements['count_improvement_nights']}/"
              f"{improvements['total_nights_compared']}")
        print(f"  Nights where GA improved quality (same count): {improvements['quality_improvement_nights']}/"
              f"{improvements['total_nights_compared']}")
        
        print(f"\nConclusion:")
        if improvements['count_improved']:
            print(f"  ✅ GA IMPROVED both quantity AND quality")
        elif improvements['quality_improved']:
            print(f"  ✅ GA MAINTAINED quantity while IMPROVING quality")
        else:
            print(f"  ⚠️  GA did not improve over greedy baseline")
        
        print(f"\nOutput files:")
        print(f"  - outputs/quality_comparison_report.json (detailed report)")
        print(f"  - outputs/quality_comparison_results.json (raw data)")
        print(f"  - outputs/quality_comparison_charts.png (visualizations)")
        print(f"  - outputs/ga_fitness_convergence.png (GA convergence)")
        print(f"  - outputs/greedy/ (greedy schedules and metrics)")
        print(f"  - outputs/ga/ (GA schedules and metrics)")


def main():
    """Main entry point."""
    print("="*60)
    print("GEO SATELLITE - QUALITY-FOCUSED GA OPTIMIZATION")
    print("="*60)
    print("GA focuses on QUALITY when COUNT cannot be improved")
    print("="*60)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Quality-Focused GA Optimization')
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
    print(f"GA weights: α=1000 (count), β=1 (elevation), γ=0.01 (idle), δ=0.1 (tight windows)")
    
    try:
        # Create quality-focused scheduler
        scheduler = QualityComparisonScheduler(start_date=args.date)
        
        # Load data
        debris_df = scheduler.load_data(args.data)
        
        if debris_df.empty:
            print("No debris loaded. Exiting.")
            return
        
        print(f"\nStarting quality-focused comparison for {len(debris_df)} debris...")
        
        # Run quality comparison
        results = scheduler.run_quality_comparison(debris_df, max_nights=args.nights)
        
        # Save results
        scheduler.save_results(results)
        
        # Generate quality report
        report = scheduler.generate_quality_report(results, debris_df)
        
        # Print summary
        scheduler.print_quality_summary(report)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()