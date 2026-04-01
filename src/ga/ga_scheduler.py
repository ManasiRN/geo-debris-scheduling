"""
Genetic Algorithm scheduler for single night.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import pandas as pd
from datetime import datetime
import time

from config.constants import *
from src.ga.chromosome import Chromosome
from src.ga.fitness import FitnessEvaluator
from src.ga.operators import Selection, Crossover, Mutation
from src.greedy_scheduler import GreedyScheduler


class GeneticAlgorithmScheduler:
    """Genetic Algorithm optimizer for single night scheduling."""
    
    def __init__(self, debris_df: pd.DataFrame, night_date: str = None):
        """
        Initialize GA scheduler.
        
        Args:
            debris_df: DataFrame of debris to schedule
            night_date: Reference date for night window
        """
        self.debris_df = debris_df
        self.night_date = night_date
        self.num_debris = len(debris_df)
        
        # GA parameters
        self.population_size = GA_POPULATION_SIZE
        self.generations = GA_GENERATIONS
        self.tournament_size = GA_TOURNAMENT_SIZE
        self.mutation_rate = GA_MUTATION_RATE
        self.elite_size = max(2, self.population_size // 10)
        self.early_stop_patience = GA_EARLY_STOP_PATIENCE
        
        # Fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(debris_df, night_date)
        
        # Progress tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Create greedy scheduler for baseline comparison
        self.greedy_scheduler = GreedyScheduler(night_date)
    
    def initialize_population(self) -> List[Chromosome]:
        """Initialize random population."""
        population = []
        
        # Include greedy ordering as one candidate
        greedy_schedule = self.greedy_scheduler.schedule_night(self.debris_df)
        if not greedy_schedule.empty:
            # Create chromosome from scheduled debris order
            scheduled_indices = list(greedy_schedule.index)
            # Add unscheduled debris in random order
            all_indices = list(range(self.num_debris))
            unscheduled = [idx for idx in all_indices if idx not in scheduled_indices]
            random.shuffle(unscheduled)
            greedy_genes = scheduled_indices + unscheduled
            
            # Ensure valid permutation
            if set(greedy_genes) == set(range(self.num_debris)):
                population.append(Chromosome(genes=greedy_genes))
        
        # Fill rest with random chromosomes
        while len(population) < self.population_size:
            population.append(Chromosome.create_random(self.num_debris))
        
        return population[:self.population_size]
    
    def run(self) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Run Genetic Algorithm optimization.
        
        Returns:
            Tuple of (best_schedule, statistics)
        """
        print(f"Starting GA for {self.num_debris} debris...")
        start_time = time.time()
        
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial population
        fitness_values = self.fitness_evaluator.batch_evaluate(population)
        
        # Track best solution
        best_idx = np.argmax(fitness_values)
        best_chromosome = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        # Early stopping tracking
        no_improvement_count = 0
        
        # Evolution loop
        for generation in range(self.generations):
            # Create new population
            new_population = []
            
            # Elitism: keep best solutions
            elites = Selection.elitist_selection(population, fitness_values, self.elite_size)
            new_population.extend(elites)
            
            # Fill rest of population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = Selection.tournament_selection(
                    population, fitness_values, self.tournament_size
                )
                parent2 = Selection.tournament_selection(
                    population, fitness_values, self.tournament_size
                )
                
                # Crossover
                if random.random() < 0.8:  # Crossover probability
                    child1, child2 = Crossover.order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = Mutation.swap_mutation(child1, self.mutation_rate)
                child2 = Mutation.swap_mutation(child2, self.mutation_rate)
                
                new_population.extend([child1, child2])
            
            # Trim if we have too many
            population = new_population[:self.population_size]
            
            # Evaluate new population
            fitness_values = self.fitness_evaluator.batch_evaluate(population)
            
            # Update best solution
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = population[current_best_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Track progress
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))
            
            # Print progress
            if (generation + 1) % 10 == 0:
                print(f"  Generation {generation + 1}: "
                     f"Best = {best_fitness:.2f}, "
                     f"Avg = {np.mean(fitness_values):.2f}")
            
            # Early stopping
            if no_improvement_count >= self.early_stop_patience:
                print(f"  Early stopping at generation {generation + 1}")
                break
        
        # Generate schedule from best chromosome
        from src.greedy_scheduler import simulate_greedy_with_order
        best_schedule = simulate_greedy_with_order(
            self.debris_df, 
            best_chromosome.genes,
            self.night_date
        )
        
        # Calculate statistics
        run_time = time.time() - start_time
        greedy_schedule = self.greedy_scheduler.schedule_night(self.debris_df)
        
        statistics = {
            'run_time_seconds': run_time,
            'generations_completed': len(self.best_fitness_history),
            'final_best_fitness': best_fitness,
            'greedy_baseline_count': len(greedy_schedule),
            'ga_improvement': len(best_schedule) - len(greedy_schedule),
            'fitness_history': self.best_fitness_history.copy()
        }
        
        print(f"GA completed in {run_time:.2f}s. "
              f"Scheduled {len(best_schedule)} debris "
              f"(Greedy: {len(greedy_schedule)}, "
              f"Improvement: {statistics['ga_improvement']})")
        
        return best_schedule, statistics