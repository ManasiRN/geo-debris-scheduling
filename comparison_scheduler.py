"""
Comparison scheduler that runs both Greedy and GA algorithms
and compares their performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import random
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
    import matplotlib.dates as mdates
    import seaborn as sns
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytz", "matplotlib", "seaborn"])
    import pytz
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns


class GreedyScheduler:
    """Greedy scheduler implementation."""
    
    def __init__(self):
        self.obs_duration = OBS_DURATION_SEC
        self.slew_gap = SLEW_GAP_SEC
        self.slot_duration = self.obs_duration + self.slew_gap
    
    def max_slots_per_night(self):
        """Calculate maximum possible observations per night."""
        night_duration = 12 * 3600  # 12 hours in seconds
        return int(night_duration // self.slot_duration)
    
    def create_night_boundaries(self, night_date: str):
        """Create night start and end datetimes."""
        ref_date = datetime.strptime(night_date, "%Y-%m-%d")
        
        night_start_time = datetime.strptime(NIGHT_START_UTC, TIME_FORMAT_HM).time()
        night_end_time = datetime.strptime(NIGHT_END_UTC, TIME_FORMAT_HM).time()
        
        night_start = datetime.combine(ref_date, night_start_time).replace(tzinfo=pytz.UTC)
        night_end = datetime.combine(ref_date + timedelta(days=1), night_end_time).replace(tzinfo=pytz.UTC)
        
        return night_start, night_end
    
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
                if clipped_duration >= self.obs_duration:
                    new_row = row.copy()
                    new_row['visibility_start_utc'] = clipped_start
                    new_row['visibility_end_utc'] = clipped_end
                    new_row['clipped_duration_sec'] = clipped_duration
                    results.append(new_row)
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def schedule_night(self, debris_df: pd.DataFrame, night_start: datetime,
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
                break
            
            vis_start = debris['visibility_start_utc']
            vis_end = debris['visibility_end_utc']
            
            if current_time < vis_start:
                current_time = vis_start
            
            proposed_end = current_time + timedelta(seconds=self.obs_duration)
            
            if proposed_end <= vis_end and proposed_end <= night_end:
                obs_start = current_time
                obs_end = proposed_end
                
                schedule.append({
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
        
        return pd.DataFrame(schedule)


class GeneticAlgorithmScheduler:
    """Genetic Algorithm scheduler implementation."""
    
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
    
    def max_slots_per_night(self):
        """Calculate maximum possible observations per night."""
        night_duration = 12 * 3600
        return int(night_duration // self.slot_duration)
    
    def create_night_boundaries(self, night_date: str):
        """Create night start and end datetimes."""
        ref_date = datetime.strptime(night_date, "%Y-%m-%d")
        
        night_start_time = datetime.strptime(NIGHT_START_UTC, TIME_FORMAT_HM).time()
        night_end_time = datetime.strptime(NIGHT_END_UTC, TIME_FORMAT_HM).time()
        
        night_start = datetime.combine(ref_date, night_start_time).replace(tzinfo=pytz.UTC)
        night_end = datetime.combine(ref_date + timedelta(days=1), night_end_time).replace(tzinfo=pytz.UTC)
        
        return night_start, night_end
    
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
                if clipped_duration >= self.obs_duration:
                    new_row = row.copy()
                    new_row['visibility_start_utc'] = clipped_start
                    new_row['visibility_end_utc'] = clipped_end
                    new_row['clipped_duration_sec'] = clipped_duration
                    results.append(new_row)
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
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
                                  night_start: datetime, night_end: datetime) -> pd.DataFrame:
        """Schedule using greedy algorithm with specific order."""
        if debris_df.empty:
            return pd.DataFrame()
        
        # Reorder debris according to chromosome
        ordered_df = debris_df.iloc[order].copy()
        
        # Greedy scheduling
        schedule = []
        current_time = night_start
        max_slots = self.max_slots_per_night()
        scheduled_count = 0
        
        for idx, debris in ordered_df.iterrows():
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
                
                schedule.append({
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
        
        return pd.DataFrame(schedule)
    
    def evaluate_fitness(self, chromosome, debris_df: pd.DataFrame,
                        night_start: datetime, night_end: datetime) -> float:
        """Evaluate fitness of a chromosome."""
        schedule = self.greedy_schedule_with_order(debris_df, chromosome.genes,
                                                  night_start, night_end)
        
        # Base fitness: number of scheduled debris
        base_fitness = len(schedule)
        
        if base_fitness == 0:
            return 0.0
        
        # Add elevation bonus (small weight)
        elevation_bonus = schedule['peak_elevation'].sum() * 0.001
        
        return base_fitness + elevation_bonus
    
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
              night_end: datetime) -> Tuple[pd.DataFrame, List[float]]:
        """Run Genetic Algorithm optimization."""
        print(f"  Starting GA with {len(debris_df)} debris...")
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            population.append(self.Chromosome.create_random(len(debris_df)))
        
        # Evaluate initial population
        fitness_values = [self.evaluate_fitness(chrom, debris_df, night_start, night_end) 
                         for chrom in population]
        
        best_idx = np.argmax(fitness_values)
        best_chromosome = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        fitness_history = [best_fitness]
        
        # Evolution loop
        for generation in range(self.generations):
            new_population = []
            
            # Elitism
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
            fitness_values = [self.evaluate_fitness(chrom, debris_df, night_start, night_end) 
                            for chrom in population]
            
            # Update best solution
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = population[current_best_idx].copy()
            
            fitness_history.append(best_fitness)
            
            # Early stopping check
            if generation > 50 and len(set(fitness_history[-20:])) == 1:
                break
        
        # Generate schedule from best chromosome
        best_schedule = self.greedy_schedule_with_order(debris_df, best_chromosome.genes,
                                                       night_start, night_end)
        
        print(f"  GA completed: {len(best_schedule)} debris scheduled")
        
        return best_schedule, fitness_history
    
    def schedule_night(self, debris_df: pd.DataFrame, night_start: datetime,
                      night_end: datetime, use_ga: bool = True) -> pd.DataFrame:
        """Schedule one night using GA or fallback to greedy."""
        if debris_df.empty:
            return pd.DataFrame()
        
        if not use_ga or len(debris_df) < 10:
            # Use greedy for small problems
            greedy = GreedyScheduler()
            return greedy.schedule_night(debris_df, night_start, night_end)
        
        # Run GA
        schedule, _ = self.run_ga(debris_df, night_start, night_end)
        
        # Fallback to greedy if GA performs worse
        greedy = GreedyScheduler()
        greedy_schedule = greedy.schedule_night(debris_df, night_start, night_end)
        
        if len(greedy_schedule) > len(schedule):
            print(f"  Using greedy (GA got {len(schedule)} vs greedy {len(greedy_schedule)})")
            return greedy_schedule
        
        return schedule


class ComparisonScheduler:
    """Main comparison scheduler that runs both algorithms."""
    
    def __init__(self, start_date: str = "2025-10-07"):
        self.start_date = start_date
        self.greedy_scheduler = GreedyScheduler()
        self.ga_scheduler = GeneticAlgorithmScheduler()
        
        print(f"Starting from date: {start_date}")
        print(f"Maximum slots per night: {self.greedy_scheduler.max_slots_per_night()}")
    
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
    
    def run_comparison(self, debris_df: pd.DataFrame, max_nights: int = 5) -> Dict:
        """Run comparison between Greedy and GA algorithms."""
        print(f"\n{'='*60}")
        print(f"ALGORITHM COMPARISON (max {max_nights} nights)")
        print(f"{'='*60}")
        
        # Results storage
        results = {
            'greedy': {'schedules': {}, 'statistics': {}},
            'ga': {'schedules': {}, 'statistics': {}}
        }
        
        # Run both algorithms
        for algorithm in ['greedy', 'ga']:
            print(f"\n{'='*40}")
            print(f"Running {algorithm.upper()} algorithm")
            print(f"{'='*40}")
            
            remaining_df = debris_df.copy()
            schedules = {}
            night_boundaries = {}
            
            for night in range(1, max_nights + 1):
                if remaining_df.empty:
                    break
                
                # Calculate date for this night
                night_date = (datetime.strptime(self.start_date, "%Y-%m-%d") + 
                            timedelta(days=night-1)).strftime("%Y-%m-%d")
                
                # Create night boundaries
                night_start, night_end = self.greedy_scheduler.create_night_boundaries(night_date)
                night_boundaries[night] = (night_start, night_end)
                
                print(f"\nNight {night} ({night_date}):")
                print(f"  Remaining debris: {len(remaining_df)}")
                
                # Shift visibility for subsequent nights
                if algorithm == 'greedy':
                    night_debris = self.greedy_scheduler.shift_visibility(remaining_df, night-1)
                    night_debris = self.greedy_scheduler.filter_by_night_window(night_debris, night_start, night_end)
                    schedule = self.greedy_scheduler.schedule_night(night_debris, night_start, night_end)
                else:
                    night_debris = self.ga_scheduler.shift_visibility(remaining_df, night-1)
                    night_debris = self.ga_scheduler.filter_by_night_window(night_debris, night_start, night_end)
                    schedule = self.ga_scheduler.schedule_night(night_debris, night_start, night_end, use_ga=True)
                
                if schedule.empty:
                    print(f"  No debris scheduled this night")
                    continue
                
                print(f"  Scheduled: {len(schedule)} debris")
                
                schedules[night] = schedule
                
                # Remove scheduled debris
                scheduled_ids = set(schedule['norad_id'])
                remaining_df = remaining_df[~remaining_df['norad_id'].isin(scheduled_ids)]
                
                # Save per-night schedule
                self.save_schedule(schedule, algorithm, night)
            
            # Store results
            results[algorithm]['schedules'] = schedules
            results[algorithm]['remaining'] = remaining_df
            results[algorithm]['night_boundaries'] = night_boundaries
            
            # Calculate statistics
            total_scheduled = sum(len(sched) for sched in schedules.values())
            max_per_night = self.greedy_scheduler.max_slots_per_night()
            
            stats = {
                'total_scheduled': total_scheduled,
                'remaining': len(remaining_df),
                'nights_used': len(schedules),
                'scheduled_per_night': {night: len(sched) for night, sched in schedules.items()},
                'coverage_percentage': (total_scheduled / len(debris_df) * 100) if len(debris_df) > 0 else 0,
                'slot_efficiency': (total_scheduled / (max_per_night * len(schedules)) * 100) if schedules else 0
            }
            
            results[algorithm]['statistics'] = stats
            
            print(f"\n{algorithm.upper()} results:")
            print(f"  Total scheduled: {total_scheduled}")
            print(f"  Remaining: {len(remaining_df)}")
            print(f"  Nights used: {len(schedules)}")
            print(f"  Coverage: {stats['coverage_percentage']:.1f}%")
        
        return results
    
    def save_schedule(self, schedule: pd.DataFrame, algorithm: str, night: int):
        """Save schedule to CSV."""
        if schedule.empty:
            return
        
        output_dir = f"outputs/{algorithm}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f"{output_dir}/schedule_night{night}.csv"
        schedule.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
    
    def generate_comparison_report(self, results: Dict, debris_df: pd.DataFrame):
        """Generate comparison report and visualizations."""
        print(f"\n{'='*60}")
        print(f"GENERATING COMPARISON REPORT")
        print(f"{'='*60}")
        
        # Create comparison statistics
        comparison_stats = {
            'total_debris': len(debris_df),
            'algorithms': {}
        }
        
        for algorithm in ['greedy', 'ga']:
            stats = results[algorithm]['statistics']
            comparison_stats['algorithms'][algorithm] = {
                'scheduled': stats['total_scheduled'],
                'remaining': stats['remaining'],
                'nights': stats['nights_used'],
                'coverage': stats['coverage_percentage'],
                'efficiency': stats['slot_efficiency']
            }
        
        # Calculate improvements
        greedy_scheduled = comparison_stats['algorithms']['greedy']['scheduled']
        ga_scheduled = comparison_stats['algorithms']['ga']['scheduled']
        
        comparison_stats['improvement'] = {
            'absolute': ga_scheduled - greedy_scheduled,
            'percentage': ((ga_scheduled - greedy_scheduled) / greedy_scheduled * 100) if greedy_scheduled > 0 else 0
        }
        
        # Save comparison statistics
        stats_file = "outputs/comparison_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(comparison_stats, f, indent=2, default=str)
        
        print(f"✓ Comparison statistics saved to {stats_file}")
        
        # Create comparison visualizations
        self.plot_comparison(results, comparison_stats)
        
        return comparison_stats
    
    def plot_comparison(self, results: Dict, stats: Dict):
        """Create comparison visualizations."""
        # 1. Bar chart comparing scheduled counts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Total scheduled comparison
        ax1 = axes[0, 0]
        algorithms = ['Greedy', 'GA']
        scheduled_counts = [stats['algorithms']['greedy']['scheduled'], 
                          stats['algorithms']['ga']['scheduled']]
        
        bars = ax1.bar(algorithms, scheduled_counts, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Number of Debris Scheduled', fontsize=12)
        ax1.set_title('Total Scheduled Debris Comparison', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, scheduled_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontsize=11)
        
        # Subplot 2: Nights used comparison
        ax2 = axes[0, 1]
        nights_used = [stats['algorithms']['greedy']['nights'], 
                      stats['algorithms']['ga']['nights']]
        
        bars2 = ax2.bar(algorithms, nights_used, color=['lightblue', 'lightpink'])
        ax2.set_ylabel('Number of Nights Used', fontsize=12)
        ax2.set_title('Nights Used Comparison', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars2, nights_used):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=11)
        
        # Subplot 3: Coverage percentage comparison
        ax3 = axes[1, 0]
        coverage = [stats['algorithms']['greedy']['coverage'], 
                   stats['algorithms']['ga']['coverage']]
        
        bars3 = ax3.bar(algorithms, coverage, color=['lightgreen', 'lightyellow'])
        ax3.set_ylabel('Coverage Percentage (%)', fontsize=12)
        ax3.set_title('Coverage Comparison', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, perc in zip(bars3, coverage):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{perc:.1f}%', ha='center', va='bottom', fontsize=11)
        
        # Subplot 4: Efficiency comparison
        ax4 = axes[1, 1]
        efficiency = [stats['algorithms']['greedy']['efficiency'], 
                     stats['algorithms']['ga']['efficiency']]
        
        bars4 = ax4.bar(algorithms, efficiency, color=['lightgray', 'wheat'])
        ax4.set_ylabel('Slot Efficiency (%)', fontsize=12)
        ax4.set_title('Slot Efficiency Comparison', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, eff in zip(bars4, efficiency):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{eff:.1f}%', ha='center', va='bottom', fontsize=11)
        
        plt.suptitle('Greedy vs Genetic Algorithm Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        comparison_file = "outputs/algorithm_comparison.png"
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison chart saved to {comparison_file}")
        
        # 2. Timeline comparison (first 3 nights)
        self.plot_timeline_comparison(results)
        
        # 3. Improvement summary
        self.plot_improvement_summary(stats)
    
    def plot_timeline_comparison(self, results: Dict):
        """Plot timeline comparison for first few nights."""
        max_nights_to_plot = min(3, max(len(results['greedy']['schedules']), 
                                       len(results['ga']['schedules'])))
        
        if max_nights_to_plot == 0:
            return
        
        fig, axes = plt.subplots(max_nights_to_plot, 2, figsize=(16, 4 * max_nights_to_plot))
        
        for night in range(1, max_nights_to_plot + 1):
            for col, algorithm in enumerate(['greedy', 'ga']):
                ax = axes[night-1, col] if max_nights_to_plot > 1 else axes[col]
                
                schedules = results[algorithm]['schedules']
                night_boundaries = results[algorithm]['night_boundaries']
                
                if night in schedules and night in night_boundaries:
                    schedule = schedules[night]
                    night_start, night_end = night_boundaries[night]
                    
                    # Convert to hours since night start
                    schedule['start_hours'] = pd.to_datetime(schedule['observation_start_utc']).apply(
                        lambda x: (x - night_start).total_seconds() / 3600
                    )
                    schedule['duration_hours'] = OBS_DURATION_SEC / 3600
                    
                    # Sort by start time
                    schedule = schedule.sort_values('start_hours')
                    
                    # Plot each observation
                    for _, row in schedule.iterrows():
                        ax.barh(y=0, width=row['duration_hours'], left=row['start_hours'],
                               height=0.5, edgecolor='black', alpha=0.7)
                    
                    ax.set_xlabel(f'Hours since {night_start.strftime("%H:%M")}', fontsize=10)
                    ax.set_title(f'{algorithm.upper()}: Night {night} ({len(schedule)} obs)', fontsize=11)
                    ax.set_yticks([])
                    ax.set_xlim(0, (night_end - night_start).total_seconds() / 3600)
                    ax.grid(True, alpha=0.3, axis='x')
                else:
                    ax.text(0.5, 0.5, f'No schedule\nfor Night {night}',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{algorithm.upper()}: Night {night}', fontsize=11)
        
        plt.suptitle('Schedule Timeline Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        
        timeline_file = "outputs/timeline_comparison.png"
        plt.savefig(timeline_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Timeline comparison saved to {timeline_file}")
    
    def plot_improvement_summary(self, stats: Dict):
        """Plot improvement summary."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Data
        categories = ['Scheduled Count', 'Coverage %', 'Slot Efficiency %']
        greedy_values = [
            stats['algorithms']['greedy']['scheduled'],
            stats['algorithms']['greedy']['coverage'],
            stats['algorithms']['greedy']['efficiency']
        ]
        ga_values = [
            stats['algorithms']['ga']['scheduled'],
            stats['algorithms']['ga']['coverage'],
            stats['algorithms']['ga']['efficiency']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, greedy_values, width, label='Greedy', color='skyblue')
        bars2 = ax.bar(x + width/2, ga_values, width, label='GA', color='lightcoral')
        
        # Add improvement annotations
        improvement = stats['improvement']
        if improvement['absolute'] > 0:
            ax.text(0, max(greedy_values[0], ga_values[0]) + 5,
                   f"GA improvement: +{improvement['absolute']} debris ({improvement['percentage']:.1f}%)",
                   ha='center', fontsize=11, fontweight='bold', color='green')
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title('Greedy vs GA Performance Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        improvement_file = "outputs/improvement_summary.png"
        plt.savefig(improvement_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Improvement summary saved to {improvement_file}")
    
    def print_final_report(self, comparison_stats: Dict):
        """Print final comparison report."""
        print("\n" + "="*60)
        print("FINAL COMPARISON REPORT")
        print("="*60)
        
        total_debris = comparison_stats['total_debris']
        
        print(f"\nTotal debris to schedule: {total_debris}")
        print(f"Maximum per night: {self.greedy_scheduler.max_slots_per_night()}")
        print(f"Maximum in 5 nights: {self.greedy_scheduler.max_slots_per_night() * 5}")
        
        print(f"\n{'Algorithm':<15} {'Scheduled':<12} {'Remaining':<12} {'Nights':<8} {'Coverage':<10} {'Efficiency':<10}")
        print("-" * 70)
        
        for algorithm in ['greedy', 'ga']:
            algo_stats = comparison_stats['algorithms'][algorithm]
            print(f"{algorithm.upper():<15} {algo_stats['scheduled']:<12} {algo_stats['remaining']:<12} "
                  f"{algo_stats['nights']:<8} {algo_stats['coverage']:<10.1f}% {algo_stats['efficiency']:<10.1f}%")
        
        improvement = comparison_stats['improvement']
        print(f"\nGA Improvement over Greedy:")
        print(f"  Absolute: +{improvement['absolute']} debris")
        print(f"  Percentage: {improvement['percentage']:.1f}%")
        
        print(f"\nOutput files in 'outputs/' directory:")
        print("  For Greedy algorithm:")
        print("    - outputs/greedy/schedule_nightX.csv")
        print("  For GA algorithm:")
        print("    - outputs/ga/schedule_nightX.csv")
        print("  Comparison files:")
        print("    - outputs/algorithm_comparison.png")
        print("    - outputs/timeline_comparison.png")
        print("    - outputs/improvement_summary.png")
        print("    - outputs/comparison_statistics.json")
        
        if improvement['absolute'] > 0:
            print(f"\n✅ GA performed better than Greedy by {improvement['absolute']} debris")
        elif improvement['absolute'] < 0:
            print(f"\n⚠️  Greedy performed better than GA by {-improvement['absolute']} debris")
        else:
            print(f"\n⚖️  Both algorithms performed equally")


def main():
    """Main entry point."""
    print("="*60)
    print("GEO SATELLITE - GREEDY vs GA ALGORITHM COMPARISON")
    print("="*60)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Greedy vs GA Algorithm Comparison')
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
    
    try:
        # Create comparison scheduler
        scheduler = ComparisonScheduler(start_date=args.date)
        
        # Load data
        debris_df = scheduler.load_data(args.data)
        
        if debris_df.empty:
            print("No debris loaded. Exiting.")
            return
        
        print(f"\nStarting algorithm comparison for {len(debris_df)} debris...")
        
        # Run comparison
        results = scheduler.run_comparison(debris_df, max_nights=args.nights)
        
        # Generate comparison report
        comparison_stats = scheduler.generate_comparison_report(results, debris_df)
        
        # Print final report
        scheduler.print_final_report(comparison_stats)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()