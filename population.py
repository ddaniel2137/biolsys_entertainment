import numpy as np
from typing import List, Callable
from icecream import ic
import copy
import streamlit
from scipy.spatial.distance import cdist


class Population:
    def __init__(self, init_population: int, num_genes: int, optimal_genotype: np.ndarray,
                 fitness_coefficient: float, max_population: int, mutation_probability: float,
                 mutation_effect: float, max_num_children: int, interaction_value: float, **kwargs):
        if kwargs.get("seed", None) is not None:
            np.random.seed(kwargs.get("seed"))
        self.genotypes = np.random.uniform(-1, 1, (init_population, num_genes))
        self.num_genes = num_genes
        self.generation = 1
        self.optimal_genotype = optimal_genotype
        self.fitness_coefficient = fitness_coefficient
        self.size = init_population
        self.max_population = max_population
        self.mutation_probability = mutation_probability
        self.mutation_effect = mutation_effect
        self.max_num_children = max_num_children
        self.fitnesses = np.exp(-np.linalg.norm(self.genotypes - self.optimal_genotype, axis=1) / (2 * self.fitness_coefficient ** 2))
        self.prev_mean_fitness = 0.0
        self.prev_size = 0
        self.mean_fitness = float(np.mean(self.fitnesses))
        self.interaction_value = interaction_value
        
        
    
    def evaluate(self, mean_fitness_other: float, size_other: int):
        prev_mean_fitness = self.mean_fitness
        total_size = self.size + size_other
        
        if self.genotypes.shape[0] > 0:
            distances = cdist(self.genotypes, self.optimal_genotype.reshape(1, -1), metric='euclidean').flatten()
            genotypic_fitnesses = np.exp(-distances / (2 * self.fitness_coefficient ** 2))
            freq_other = size_other / total_size
            freq_self = self.size / total_size
            interaction_fitnesses = mean_fitness_other * self.interaction_value * freq_other / freq_self
            fitnesses = np.clip(genotypic_fitnesses + interaction_fitnesses, 0, 1)
            mean_fitness = np.mean(fitnesses)
        else:
            fitnesses = np.array([])
            mean_fitness = 0.0
        return fitnesses, mean_fitness, prev_mean_fitness
    
    def mutate(self):
        mask = np.random.uniform(0, 1, self.size * self.num_genes) < self.mutation_probability
        mutated_genotypes = self.genotypes + (mask * np.random.normal(0, self.mutation_effect,
                                                                     self.size * self.num_genes)).reshape(self.size, self.num_genes)
        return mutated_genotypes
    
    def reproduce(self):
        indices = self.select()
        np.random.shuffle(indices)
        if indices.shape[0] % 2 != 0:
            indices = indices[:-1]
        parents1, parents2 = np.array_split(indices, 2)
        offspring_numbers = np.random.poisson(self.max_num_children, parents1.shape[0])
        offspring_list = []
        for p1, p2, o_n in zip(parents1, parents2, offspring_numbers):
            crossover_points = np.random.randint(0, self.num_genes, o_n)
            for c in crossover_points:
                offspring = np.concatenate((self.genotypes[p1, :c], self.genotypes[p2, c:]))
                offspring_list.append(offspring)
        new_genotypes = np.concatenate([self.genotypes[indices], np.array(offspring_list).reshape(-1, self.num_genes)])
        new_genotypes = new_genotypes[:self.max_population]
        new_generation = self.generation + 1
        new_size = new_genotypes.shape[0]

        return new_genotypes, new_generation, new_size, copy.copy(self.size)
    
    def select(self):
        #ic(np.where(np.random.rand(self.size) < self.fitnesses))
        return np.where(np.random.rand(self.size) < np.exp(-self.size / self.max_population) * self.fitnesses)[0]
