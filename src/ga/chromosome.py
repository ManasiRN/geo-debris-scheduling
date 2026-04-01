"""
Chromosome representation for Genetic Algorithm.
"""

import numpy as np
from typing import List, Tuple, Optional
import random


class Chromosome:
    """Chromosome representing debris scheduling order."""
    
    def __init__(self, genes: List[int] = None, num_genes: int = None):
        """
        Initialize chromosome.
        
        Args:
            genes: List of gene indices (debris indices)
            num_genes: Number of genes (if genes not provided)
        """
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
    
    def __getitem__(self, index):
        return self.genes[index]
    
    def __setitem__(self, index, value):
        self.genes[index] = value
    
    def copy(self):
        """Create a deep copy of the chromosome."""
        return Chromosome(genes=self.genes)
    
    @classmethod
    def create_random(cls, num_genes: int) -> 'Chromosome':
        """Create random chromosome."""
        genes = list(range(num_genes))
        random.shuffle(genes)
        return cls(genes=genes)
    
    def is_valid(self) -> bool:
        """Check if chromosome represents a valid permutation."""
        if len(self.genes) != len(set(self.genes)):
            return False
        if set(self.genes) != set(range(self.num_genes)):
            return False
        return True
    
    def repair(self):
        """Repair chromosome to ensure valid permutation."""
        # Find missing and duplicate values
        seen = set()
        duplicates = []
        missing = set(range(self.num_genes))
        
        for i, gene in enumerate(self.genes):
            if gene in seen:
                duplicates.append(i)
            else:
                seen.add(gene)
                missing.discard(gene)
        
        # Replace duplicates with missing values
        missing_list = list(missing)
        for idx in duplicates:
            if missing_list:
                self.genes[idx] = missing_list.pop()
            else:
                # This shouldn't happen if chromosome was valid initially
                self.genes[idx] = random.randint(0, self.num_genes - 1)