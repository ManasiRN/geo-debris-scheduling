"""
Genetic Algorithm operators: selection, crossover, mutation.
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from src.ga.chromosome import Chromosome


class Selection:
    """Selection operators."""
    
    @staticmethod
    def tournament_selection(population: List[Chromosome], 
                           fitness_values: List[float],
                           tournament_size: int = 3) -> Chromosome:
        """
        Tournament selection.
        
        Args:
            population: List of chromosomes
            fitness_values: List of fitness values
            tournament_size: Size of tournament
            
        Returns:
            Selected chromosome
        """
        # Randomly select tournament participants
        tournament_indices = random.sample(range(len(population)), tournament_size)
        
        # Find the best in tournament
        best_idx = tournament_indices[0]
        best_fitness = fitness_values[best_idx]
        
        for idx in tournament_indices[1:]:
            if fitness_values[idx] > best_fitness:
                best_idx = idx
                best_fitness = fitness_values[idx]
        
        return population[best_idx].copy()
    
    @staticmethod
    def elitist_selection(population: List[Chromosome],
                         fitness_values: List[float],
                         elite_size: int = 2) -> List[Chromosome]:
        """
        Select elite chromosomes.
        
        Args:
            population: List of chromosomes
            fitness_values: List of fitness values
            elite_size: Number of elite to select
            
        Returns:
            List of elite chromosomes
        """
        # Sort by fitness
        sorted_indices = np.argsort(fitness_values)[::-1]
        
        elites = []
        for i in range(min(elite_size, len(population))):
            elites.append(population[sorted_indices[i]].copy())
        
        return elites


class Crossover:
    """Crossover operators for permutation chromosomes."""
    
    @staticmethod
    def order_crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Order Crossover (OX) for permutation chromosomes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of (child1, child2)
        """
        n = len(parent1)
        
        # Select random crossover points
        cx1, cx2 = sorted(random.sample(range(n), 2))
        
        # Create children
        child1_genes = [-1] * n
        child2_genes = [-1] * n
        
        # Copy segment between cx1 and cx2
        child1_genes[cx1:cx2] = parent1.genes[cx1:cx2]
        child2_genes[cx1:cx2] = parent2.genes[cx1:cx2]
        
        # Fill remaining positions from other parent
        child1_pos = cx2
        child2_pos = cx2
        
        for i in range(n):
            parent2_idx = (cx2 + i) % n
            parent1_idx = (cx2 + i) % n
            
            # For child1
            if parent2.genes[parent2_idx] not in child1_genes:
                child1_genes[child1_pos] = parent2.genes[parent2_idx]
                child1_pos = (child1_pos + 1) % n
            
            # For child2
            if parent1.genes[parent1_idx] not in child2_genes:
                child2_genes[child2_pos] = parent1.genes[parent1_idx]
                child2_pos = (child2_pos + 1) % n
        
        # Create child chromosomes
        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)
        
        return child1, child2
    
    @staticmethod
    def partially_mapped_crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Partially Mapped Crossover (PMX).
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        n = len(parent1)
        
        # Select random crossover points
        cx1, cx2 = sorted(random.sample(range(n), 2))
        
        # Initialize children
        child1_genes = [-1] * n
        child2_genes = [-1] * n
        
        # Copy segment between cx1 and cx2
        child1_genes[cx1:cx2] = parent1.genes[cx1:cx2]
        child2_genes[cx1:cx2] = parent2.genes[cx1:cx2]
        
        # Create mapping for remaining positions
        mapping1 = {}
        mapping2 = {}
        
        for i in range(cx1, cx2):
            mapping1[parent2.genes[i]] = parent1.genes[i]
            mapping2[parent1.genes[i]] = parent2.genes[i]
        
        # Fill remaining positions
        for i in list(range(0, cx1)) + list(range(cx2, n)):
            # For child1
            gene2 = parent2.genes[i]
            while gene2 in child1_genes:
                gene2 = mapping1.get(gene2, gene2)
            child1_genes[i] = gene2
            
            # For child2
            gene1 = parent1.genes[i]
            while gene1 in child2_genes:
                gene1 = mapping2.get(gene1, gene1)
            child2_genes[i] = gene1
        
        # Create child chromosomes
        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)
        
        return child1, child2


class Mutation:
    """Mutation operators."""
    
    @staticmethod
    def swap_mutation(chromosome: Chromosome, mutation_rate: float = 0.1) -> Chromosome:
        """
        Swap mutation: randomly swap two genes.
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated chromosome
        """
        if random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        n = len(mutated)
        
        # Select two distinct positions
        i, j = random.sample(range(n), 2)
        
        # Swap genes
        mutated.genes[i], mutated.genes[j] = mutated.genes[j], mutated.genes[i]
        
        return mutated
    
    @staticmethod
    def scramble_mutation(chromosome: Chromosome, mutation_rate: float = 0.1) -> Chromosome:
        """
        Scramble mutation: randomly shuffle a segment.
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated chromosome
        """
        if random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        n = len(mutated)
        
        # Select random segment
        start, end = sorted(random.sample(range(n), 2))
        
        # Scramble segment
        segment = mutated.genes[start:end]
        random.shuffle(segment)
        mutated.genes[start:end] = segment
        
        return mutated
    
    @staticmethod
    def inversion_mutation(chromosome: Chromosome, mutation_rate: float = 0.1) -> Chromosome:
        """
        Inversion mutation: reverse a segment.
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated chromosome
        """
        if random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        n = len(mutated)
        
        # Select random segment
        start, end = sorted(random.sample(range(n), 2))
        
        # Reverse segment
        mutated.genes[start:end] = reversed(mutated.genes[start:end])
        
        return mutated