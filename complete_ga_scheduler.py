"""
Complete Genetic Algorithm Scheduler for GEO Satellite Debris Observation.
Produces proper output with schedules, statistics, and visualizations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add config to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config.constants import *
    print(f"Night window: {NIGHT_START_UTC} to {NIGHT_END_UTC} UTC")
except ImportError as e:
    print(f"ERROR: Could not load configuration: {e}")
    sys.exit(1)

# Import required modules
try:
    import pytz
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytz"])
    import pytz


class Chromosome:
    """Chromosome representing scheduling order."""
    
    def __init__(self, genes: List[int] = None, num_genes: int = None):
        if genes is not None:
            self.genes = genes.copy()
            self.length = len(genes)
        elif num_genes is not None:
            self.length = num_genes
            self.genes = list(range(num_genes))
            random.shuffle(self.genes)
        else:
            raise ValueError("Provide either genes or num_genes")
        
        self.fitness = 0.0
        self.metrics = {}
    
    def __len__(self):
        return self.length
    
    def copy(self):
        new_chrom = Chromosome(genes=self.genes)
        new_chrom.fitness = self.fitness
        new_chrom.metrics = self.metrics.copy()
        return new_chrom
    
    @classmethod
    def create_random(cls, num_genes: int):
        genes = list(range(num_genes))
        random.shuffle(genes)
        return cls(genes=genes)


class GeneticAlgorithm:
    """Genetic Algorithm for scheduling optimization."""
    
    def __init__(self, population_size: int = 30, generations: int = 50):
        self.population_size = population_size
        self.generations = generations
        self.obs_duration = OBS_DURATION_SEC
        self.slew_gap = SLEW_GAP_SEC
        self.slot_duration = self.obs_duration + self.slew_gap
        
        # GA parameters
        self.tournament_size = 3
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8
        self.elite_size = 2
        
        # Fitness weights
        self.count_weight = 1000.0
        self.elevation_weight = 1.0
        self.idle_weight = 0.01
        
        print(f"GA initialized: Pop={population_size}, Gen={generations}")
    
    def max_slots_per_night(self):
        """Maximum possible observations per night."""
        night_duration = 12 * 3600  # 12 hours
        return int(night_duration // self.slot_duration)
    
    def greedy_schedule(self, debris_df: pd.DataFrame, order: List[int],
                       night_start: datetime, night_end: datetime) -> Tuple[pd.DataFrame, Dict]:
        """Greedy scheduling with given order."""
        if debris_df.empty or not order:
            return pd.DataFrame(), {}
        
        # Apply order
        ordered_df = debris_df.iloc[order].reset_index(drop=True)
        
        schedule_rows = []
        current_time = night_start
        scheduled = 0
        max_slots = self.max_slots_per_night()
        
        idle_times = []
        elevation_sum = 0.0
        
        for idx, debris in ordered_df.iterrows():
            if scheduled >= max_slots:
                break
            
            vis_start = debris['visibility_start_utc']
            vis_end = debris['visibility_end_utc']
            
            # Wait if needed
            if current_time < vis_start:
                idle = (vis_start - current_time).total_seconds()
                idle_times.append(idle)
                current_time = vis_start
            
            # Check feasibility
            proposed_end = current_time + timedelta(seconds=self.obs_duration)
            
            if proposed_end <= vis_end and proposed_end <= night_end:
                # Schedule it
                obs_start = current_time
                obs_end = proposed_end
                
                schedule_rows.append({
                    'norad_id': debris['norad_id'],
                    'satellite_name': debris['satellite_name'],
                    'observation_start_utc': obs_start.strftime(TIME_FORMAT),
                    'observation_end_utc': obs_end.strftime(TIME_FORMAT),
                    'peak_elevation': debris['peak_elevation']
                })
                
                elevation_sum += debris['peak_elevation']
                current_time = obs_end + timedelta(seconds=self.slew_gap)
                scheduled += 1
                
                if current_time >= night_end:
                    break
        
        schedule_df = pd.DataFrame(schedule_rows) if schedule_rows else pd.DataFrame()
        
        # Calculate metrics
        total_idle = sum(idle_times) if idle_times else 0
        if schedule_rows and current_time < night_end:
            total_idle += (night_end - current_time).total_seconds()
        
        scheduled_count = len(schedule_df)
        metrics = {
            'scheduled_count': scheduled_count,
            'total_elevation': elevation_sum,
            'avg_elevation': elevation_sum / scheduled_count if scheduled_count > 0 else 0,
            'total_idle_time': total_idle,
            'night_utilization': (scheduled_count * self.slot_duration) / 
                               (night_end - night_start).total_seconds() * 100
        }
        
        return schedule_df, metrics
    
    def evaluate_fitness(self, chromosome: Chromosome, debris_df: pd.DataFrame,
                        night_start: datetime, night_end: datetime):
        """Evaluate chromosome fitness."""
        schedule, metrics = self.greedy_schedule(
            debris_df, chromosome.genes, night_start, night_end
        )
        
        # Store metrics
        chromosome.metrics = metrics
        
        # Calculate fitness
        count_score = self.count_weight * metrics['scheduled_count']
        elevation_score = self.elevation_weight * metrics['total_elevation']
        idle_penalty = self.idle_weight * metrics['total_idle_time']
        
        fitness = count_score + elevation_score - idle_penalty
        chromosome.fitness = fitness
        
        return fitness
    
    def tournament_selection(self, population: List[Chromosome]) -> Chromosome:
        """Tournament selection."""
        tournament = random.sample(population, self.tournament_size)
        best = tournament[0]
        for chrom in tournament[1:]:
            if chrom.fitness > best.fitness:
                best = chrom
        return best.copy()
    
    def order_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Order Crossover (OX)."""
        n = len(parent1)
        cx1, cx2 = sorted(random.sample(range(n), 2))
        
        # Initialize children
        child1_genes = [-1] * n
        child2_genes = [-1] * n
        
        # Copy segment
        child1_genes[cx1:cx2] = parent1.genes[cx1:cx2]
        child2_genes[cx1:cx2] = parent2.genes[cx1:cx2]
        
        # Fill remaining
        pos1 = cx2
        pos2 = cx2
        
        for i in range(n):
            idx = (cx2 + i) % n
            
            # For child1
            gene = parent2.genes[idx]
            if gene not in child1_genes:
                child1_genes[pos1] = gene
                pos1 = (pos1 + 1) % n
            
            # For child2
            gene = parent1.genes[idx]
            if gene not in child2_genes:
                child2_genes[pos2] = gene
                pos2 = (pos2 + 1) % n
        
        return Chromosome(genes=child1_genes), Chromosome(genes=child2_genes)
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """Swap mutation."""
        if random.random() > self.mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        n = len(mutated)
        i, j = random.sample(range(n), 2)
        mutated.genes[i], mutated.genes[j] = mutated.genes[j], mutated.genes[i]
        return mutated
    
    def run(self, debris_df: pd.DataFrame, night_start: datetime, 
           night_end: datetime) -> Tuple[pd.DataFrame, Dict, List[float]]:
        """Run GA optimization."""
        print(f"  Running GA for {len(debris_df)} debris...")
        
        # Initialize population
        population = [Chromosome.create_random(len(debris_df)) 
                     for _ in range(self.population_size)]
        
        # Evaluate initial population
        for chrom in population:
            self.evaluate_fitness(chrom, debris_df, night_start, night_end)
        
        # Track best
        best_chrom = max(population, key=lambda x: x.fitness).copy()
        fitness_history = [best_chrom.fitness]
        
        # Evolution loop
        for generation in range(self.generations):
            new_population = []
            
            # Elitism
            elites = sorted(population, key=lambda x: x.fitness, reverse=True)[:self.elite_size]
            new_population.extend([chrom.copy() for chrom in elites])
            
            # Fill rest
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            # Evaluate
            for chrom in population:
                self.evaluate_fitness(chrom, debris_df, night_start, night_end)
            
            # Update best
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_chrom.fitness:
                best_chrom = current_best.copy()
            
            fitness_history.append(best_chrom.fitness)
            
            # Progress
            if (generation + 1) % 10 == 0:
                print(f"    Gen {generation+1}: Best={best_chrom.metrics['scheduled_count']} debris")
        
        # Generate final schedule
        schedule, final_metrics = self.greedy_schedule(
            debris_df, best_chrom.genes, night_start, night_end
        )
        
        print(f"  GA completed: {final_metrics['scheduled_count']} debris scheduled")
        print(f"    Avg elevation: {final_metrics['avg_elevation']:.1f}°")
        print(f"    Utilization: {final_metrics['night_utilization']:.1f}%")
        
        return schedule, final_metrics, fitness_history


class GreedyBaseline:
    """Greedy baseline scheduler."""
    
    def __init__(self):
        self.obs_duration = OBS_DURATION_SEC
        self.slew_gap = SLEW_GAP_SEC
        self.slot_duration = self.obs_duration + self.slew_gap
    
    def max_slots_per_night(self):
        night_duration = 12 * 3600
        return int(night_duration // self.slot_duration)
    
    def schedule_night(self, debris_df: pd.DataFrame, night_start: datetime,
                      night_end: datetime) -> pd.DataFrame:
        """Simple greedy scheduling."""
        if debris_df.empty:
            return pd.DataFrame()
        
        # Sort by end time, then elevation
        df = debris_df.copy()
        df = df.sort_values(['visibility_end_utc', 'peak_elevation'], 
                          ascending=[True, False])
        
        schedule_rows = []
        current_time = night_start
        max_slots = self.max_slots_per_night()
        scheduled = 0
        
        for idx, debris in df.iterrows():
            if scheduled >= max_slots:
                break
            
            vis_start = debris['visibility_start_utc']
            vis_end = debris['visibility_end_utc']
            
            if current_time < vis_start:
                current_time = vis_start
            
            proposed_end = current_time + timedelta(seconds=self.obs_duration)
            
            if proposed_end <= vis_end and proposed_end <= night_end:
                schedule_rows.append({
                    'norad_id': debris['norad_id'],
                    'satellite_name': debris['satellite_name'],
                    'observation_start_utc': current_time.strftime(TIME_FORMAT),
                    'observation_end_utc': proposed_end.strftime(TIME_FORMAT),
                    'peak_elevation': debris['peak_elevation']
                })
                
                current_time = proposed_end + timedelta(seconds=self.slew_gap)
                scheduled += 1
                
                if current_time >= night_end:
                    break
        
        return pd.DataFrame(schedule_rows) if schedule_rows else pd.DataFrame()


class CompleteScheduler:
    """Complete scheduler with GA and proper output."""
    
    def __init__(self, start_date: str = "2025-10-07"):
        self.start_date = start_date
        self.greedy = GreedyBaseline()
        self.ga = GeneticAlgorithm()
        
        print(f"\nStarting scheduler for {start_date}")
        print(f"Maximum per night: {self.greedy.max_slots_per_night()}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare data."""
        print(f"\nLoading data from {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} debris objects")
        
        # Parse timestamps
        df['visibility_start_utc'] = pd.to_datetime(df['Start Time (UTC)'], utc=True)
        df['visibility_end_utc'] = pd.to_datetime(df['End Time (UTC)'], utc=True)
        
        # Standardize columns
        df.rename(columns={
            'NORAD ID': 'norad_id',
            'Satellite Name': 'satellite_name',
            'Peak Elevation (deg)': 'peak_elevation'
        }, inplace=True)
        
        return df
    
    def create_night_boundaries(self, night_date: str) -> Tuple[datetime, datetime]:
        """Create night boundaries."""
        try:
            tz = pytz.UTC
        except:
            from datetime import timezone
            tz = timezone.utc
        
        ref_date = datetime.strptime(night_date, "%Y-%m-%d")
        
        night_start_time = datetime.strptime(NIGHT_START_UTC, TIME_FORMAT_HM).time()
        night_end_time = datetime.strptime(NIGHT_END_UTC, TIME_FORMAT_HM).time()
        
        night_start = datetime.combine(ref_date, night_start_time).replace(tzinfo=tz)
        night_end = datetime.combine(ref_date + timedelta(days=1), night_end_time).replace(tzinfo=tz)
        
        return night_start, night_end
    
    def shift_visibility(self, df: pd.DataFrame, night_offset: int) -> pd.DataFrame:
        """Shift visibility windows."""
        if night_offset == 0:
            return df.copy()
        
        shifted_df = df.copy()
        shift_hours = 24 * night_offset
        
        shifted_df['visibility_start_utc'] += pd.Timedelta(hours=shift_hours)
        shifted_df['visibility_end_utc'] += pd.Timedelta(hours=shift_hours)
        
        return shifted_df
    
    def filter_by_night(self, df: pd.DataFrame, night_start: datetime, 
                       night_end: datetime) -> pd.DataFrame:
        """Filter debris visible during night."""
        results = []
        
        for idx, row in df.iterrows():
            vis_start = row['visibility_start_utc']
            vis_end = row['visibility_end_utc']
            
            if vis_start < night_end and vis_end > night_start:
                clipped_start = max(vis_start, night_start)
                clipped_end = min(vis_end, night_end)
                
                clipped_duration = (clipped_end - clipped_start).total_seconds()
                if clipped_duration >= self.greedy.obs_duration:
                    new_row = row.copy()
                    new_row['visibility_start_utc'] = clipped_start
                    new_row['visibility_end_utc'] = clipped_end
                    results.append(new_row)
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def run_comparison(self, debris_df: pd.DataFrame, max_nights: int = 5) -> Dict:
        """Run comparison between Greedy and GA."""
        print(f"\n{'='*60}")
        print(f"GREEDY vs GA COMPARISON")
        print(f"{'='*60}")
        
        results = {
            'greedy': {'schedules': {}, 'metrics': {}, 'total': 0},
            'ga': {'schedules': {}, 'metrics': {}, 'fitness_history': {}, 'total': 0}
        }
        
        remaining_greedy = debris_df.copy()
        remaining_ga = debris_df.copy()
        
        # Run Greedy
        print(f"\n{'='*40}")
        print(f"RUNNING GREEDY ALGORITHM")
        print(f"{'='*40}")
        
        for night in range(1, max_nights + 1):
            if remaining_greedy.empty:
                break
            
            night_date = (datetime.strptime(self.start_date, "%Y-%m-%d") + 
                         timedelta(days=night-1)).strftime("%Y-%m-%d")
            
            night_start, night_end = self.create_night_boundaries(night_date)
            
            # Prepare debris for this night
            night_debris = self.shift_visibility(remaining_greedy, night-1)
            night_debris = self.filter_by_night(night_debris, night_start, night_end)
            
            if night_debris.empty:
                print(f"\nNight {night}: No debris visible")
                continue
            
            print(f"\nNight {night}: {len(night_debris)} debris visible")
            
            # Run greedy
            schedule = self.greedy.schedule_night(night_debris, night_start, night_end)
            scheduled = len(schedule)
            
            print(f"  Greedy scheduled: {scheduled} debris")
            
            if scheduled > 0:
                results['greedy']['schedules'][night] = schedule
                results['greedy']['total'] += scheduled
                
                # Save schedule
                self.save_schedule(schedule, 'greedy', night)
                
                # Remove scheduled
                scheduled_ids = set(schedule['norad_id'])
                remaining_greedy = remaining_greedy[~remaining_greedy['norad_id'].isin(scheduled_ids)]
        
        # Run GA
        print(f"\n{'='*40}")
        print(f"RUNNING GENETIC ALGORITHM")
        print(f"{'='*40}")
        
        for night in range(1, max_nights + 1):
            if remaining_ga.empty:
                break
            
            night_date = (datetime.strptime(self.start_date, "%Y-%m-%d") + 
                         timedelta(days=night-1)).strftime("%Y-%m-%d")
            
            night_start, night_end = self.create_night_boundaries(night_date)
            
            # Prepare debris
            night_debris = self.shift_visibility(remaining_ga, night-1)
            night_debris = self.filter_by_night(night_debris, night_start, night_end)
            
            if night_debris.empty:
                print(f"\nNight {night}: No debris visible")
                continue
            
            print(f"\nNight {night}: {len(night_debris)} debris visible")
            
            # Run GA
            schedule, metrics, fitness_history = self.ga.run(
                night_debris, night_start, night_end
            )
            
            scheduled = len(schedule)
            print(f"  GA scheduled: {scheduled} debris")
            
            if scheduled > 0:
                results['ga']['schedules'][night] = schedule
                results['ga']['metrics'][night] = metrics
                results['ga']['fitness_history'][night] = fitness_history
                results['ga']['total'] += scheduled
                
                # Save schedule
                self.save_schedule(schedule, 'ga', night)
                
                # Remove scheduled
                scheduled_ids = set(schedule['norad_id'])
                remaining_ga = remaining_ga[~remaining_ga['norad_id'].isin(scheduled_ids)]
        
        return results
    
    def save_schedule(self, schedule: pd.DataFrame, algorithm: str, night: int):
        """Save schedule to CSV."""
        if schedule.empty:
            return
        
        output_dir = f"outputs/{algorithm}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f"{output_dir}/schedule_night{night}.csv"
        schedule.to_csv(output_file, index=False)
        print(f"    Saved to {output_file}")
    
    def generate_reports(self, results: Dict, total_debris: int):
        """Generate all reports and visualizations."""
        print(f"\n{'='*60}")
        print(f"GENERATING REPORTS")
        print(f"{'='*60}")
        
        # 1. Summary report
        self.generate_summary_report(results, total_debris)
        
        # 2. Comparison report
        self.generate_comparison_report(results, total_debris)
        
        # 3. Visualizations
        self.generate_visualizations(results)
        
        # 4. Combined schedules
        self.generate_combined_schedules(results)
    
    def generate_summary_report(self, results: Dict, total_debris: int):
        """Generate summary report."""
        greedy_total = results['greedy']['total']
        ga_total = results['ga']['total']
        
        greedy_nights = len(results['greedy']['schedules'])
        ga_nights = len(results['ga']['schedules'])
        
        report = {
            'total_debris': total_debris,
            'greedy': {
                'scheduled': greedy_total,
                'remaining': total_debris - greedy_total,
                'coverage': greedy_total / total_debris * 100,
                'nights_used': greedy_nights,
                'avg_per_night': greedy_total / greedy_nights if greedy_nights > 0 else 0
            },
            'ga': {
                'scheduled': ga_total,
                'remaining': total_debris - ga_total,
                'coverage': ga_total / total_debris * 100,
                'nights_used': ga_nights,
                'avg_per_night': ga_total / ga_nights if ga_nights > 0 else 0
            },
            'comparison': {
                'difference': ga_total - greedy_total,
                'improvement_percentage': (ga_total - greedy_total) / greedy_total * 100 if greedy_total > 0 else 0
            }
        }
        
        # Save report
        report_file = "outputs/summary_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✓ Summary report saved to {report_file}")
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"  Total debris: {total_debris}")
        print(f"  Greedy: {greedy_total} scheduled ({report['greedy']['coverage']:.1f}%)")
        print(f"  GA: {ga_total} scheduled ({report['ga']['coverage']:.1f}%)")
        
        if report['comparison']['difference'] > 0:
            print(f"  ✅ GA improved by {report['comparison']['difference']} debris "
                  f"({report['comparison']['improvement_percentage']:.1f}%)")
        elif report['comparison']['difference'] < 0:
            print(f"  ⚠️  Greedy better by {-report['comparison']['difference']} debris")
        else:
            print(f"  ⚖️  Both algorithms performed equally")
    
    def generate_comparison_report(self, results: Dict, total_debris: int):
        """Generate detailed comparison report."""
        comparison = {}
        
        # Night-by-night comparison
        all_nights = set(list(results['greedy']['schedules'].keys()) + 
                        list(results['ga']['schedules'].keys()))
        
        nightly_comparison = {}
        for night in sorted(all_nights):
            greedy_schedule = results['greedy']['schedules'].get(night, pd.DataFrame())
            ga_schedule = results['ga']['schedules'].get(night, pd.DataFrame())
            
            nightly_comparison[night] = {
                'greedy_count': len(greedy_schedule),
                'ga_count': len(ga_schedule),
                'difference': len(ga_schedule) - len(greedy_schedule)
            }
        
        comparison['nightly'] = nightly_comparison
        
        # Save comparison
        comparison_file = "outputs/comparison_report.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        print(f"✓ Comparison report saved to {comparison_file}")
    
    def generate_visualizations(self, results: Dict):
        """Generate visualization plots."""
        try:
            # 1. Fitness convergence plots
            if results['ga']['fitness_history']:
                self.plot_fitness_convergence(results['ga']['fitness_history'])
            
            # 2. Nightly comparison bar chart
            self.plot_nightly_comparison(results)
            
            # 3. Overall comparison
            self.plot_overall_comparison(results)
            
        except Exception as e:
            print(f"Warning: Could not generate all visualizations: {e}")
    
    def plot_fitness_convergence(self, fitness_history: Dict):
        """Plot GA fitness convergence."""
        nights = list(fitness_history.keys())
        
        fig, axes = plt.subplots(len(nights), 1, figsize=(10, 3 * len(nights)))
        if len(nights) == 1:
            axes = [axes]
        
        for idx, night in enumerate(nights):
            ax = axes[idx]
            history = fitness_history[night]
            
            generations = range(1, len(history) + 1)
            ax.plot(generations, history, 'b-', linewidth=2)
            ax.fill_between(generations, 0, history, alpha=0.3)
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title(f'Night {night}: GA Fitness Convergence')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('GA Fitness Convergence by Night', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('outputs/ga_fitness_convergence.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Fitness convergence plot saved")
    
    def plot_nightly_comparison(self, results: Dict):
        """Plot nightly comparison."""
        all_nights = set(list(results['greedy']['schedules'].keys()) + 
                        list(results['ga']['schedules'].keys()))
        
        if not all_nights:
            return
        
        nights = sorted(all_nights)
        greedy_counts = [len(results['greedy']['schedules'].get(n, pd.DataFrame())) for n in nights]
        ga_counts = [len(results['ga']['schedules'].get(n, pd.DataFrame())) for n in nights]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(nights))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, greedy_counts, width, label='Greedy', color='skyblue')
        bars2 = ax.bar(x + width/2, ga_counts, width, label='GA', color='lightcoral')
        
        ax.set_xlabel('Night')
        ax.set_ylabel('Debris Scheduled')
        ax.set_title('Nightly Schedule Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(nights)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('outputs/nightly_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Nightly comparison plot saved")
    
    def plot_overall_comparison(self, results: Dict):
        """Plot overall comparison."""
        greedy_total = results['greedy']['total']
        ga_total = results['ga']['total']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = ['Greedy', 'GA']
        values = [greedy_total, ga_total]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black')
        
        ax.set_ylabel('Total Debris Scheduled')
        ax.set_title('Overall Performance Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/overall_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Overall comparison plot saved")
    
    def generate_combined_schedules(self, results: Dict):
        """Generate combined schedule files."""
        # Combined greedy schedule
        greedy_rows = []
        for night, schedule in results['greedy']['schedules'].items():
            for _, row in schedule.iterrows():
                combined_row = row.copy()
                combined_row['night'] = night
                greedy_rows.append(combined_row)
        
        if greedy_rows:
            greedy_combined = pd.DataFrame(greedy_rows)
            greedy_combined.to_csv('outputs/greedy_combined_schedule.csv', index=False)
            print(f"✓ Combined greedy schedule saved")
        
        # Combined GA schedule
        ga_rows = []
        for night, schedule in results['ga']['schedules'].items():
            for _, row in schedule.iterrows():
                combined_row = row.copy()
                combined_row['night'] = night
                ga_rows.append(combined_row)
        
        if ga_rows:
            ga_combined = pd.DataFrame(ga_rows)
            ga_combined.to_csv('outputs/ga_combined_schedule.csv', index=False)
            print(f"✓ Combined GA schedule saved")
    
    def print_final_output(self, results: Dict, total_debris: int):
        """Print final output summary."""
        print(f"\n{'='*60}")
        print(f"FINAL OUTPUT")
        print(f"{'='*60}")
        
        greedy_total = results['greedy']['total']
        ga_total = results['ga']['total']
        
        print(f"\nResults:")
        print(f"  Algorithm       Scheduled   Coverage   Nights Used")
        print(f"  ---------       ---------   --------   ----------")
        print(f"  Greedy          {greedy_total:>7}     {greedy_total/total_debris*100:>6.1f}%      {len(results['greedy']['schedules']):>5}")
        print(f"  Genetic Alg.    {ga_total:>7}     {ga_total/total_debris*100:>6.1f}%      {len(results['ga']['schedules']):>5}")
        
        print(f"\nOutput Files Generated:")
        print(f"  Schedule files:")
        print(f"    - outputs/greedy/schedule_nightX.csv")
        print(f"    - outputs/ga/schedule_nightX.csv")
        print(f"    - outputs/greedy_combined_schedule.csv")
        print(f"    - outputs/ga_combined_schedule.csv")
        
        print(f"\n  Report files:")
        print(f"    - outputs/summary_report.json")
        print(f"    - outputs/comparison_report.json")
        
        print(f"\n  Visualization files:")
        print(f"    - outputs/ga_fitness_convergence.png")
        print(f"    - outputs/nightly_comparison.png")
        print(f"    - outputs/overall_comparison.png")
        
        if ga_total > greedy_total:
            improvement = ga_total - greedy_total
            print(f"\n✅ SUCCESS: GA improved scheduling by {improvement} debris!")
        elif ga_total < greedy_total:
            print(f"\n⚠️  NOTE: Greedy performed better than GA")
        else:
            print(f"\n⚖️  NOTE: Both algorithms performed equally")


def main():
    """Main entry point."""
    print("="*60)
    print("COMPLETE GA SCHEDULER FOR GEO SATELLITE DEBRIS")
    print("="*60)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Complete GA Scheduler')
    parser.add_argument('--date', type=str, default="2025-10-07",
                       help='Start date for scheduling')
    parser.add_argument('--data', type=str, default="data/raw/geo_visibility.csv",
                       help='Path to input CSV')
    parser.add_argument('--nights', type=int, default=5,
                       help='Maximum nights')
    
    args = parser.parse_args()
    
    print(f"Start date: {args.date}")
    print(f"Data file: {args.data}")
    print(f"Maximum nights: {args.nights}")
    
    try:
        # Create scheduler
        scheduler = CompleteScheduler(start_date=args.date)
        
        # Load data
        debris_df = scheduler.load_data(args.data)
        
        if debris_df.empty:
            print("No debris loaded. Exiting.")
            return
        
        total_debris = len(debris_df)
        print(f"\nTotal debris to schedule: {total_debris}")
        
        # Run comparison
        results = scheduler.run_comparison(debris_df, max_nights=args.nights)
        
        # Generate reports
        scheduler.generate_reports(results, total_debris)
        
        # Print final output
        scheduler.print_final_output(results, total_debris)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()