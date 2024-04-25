import numpy as np
from typing import List, Callable
from population import Population
import math
from icecream import ic
from functools import partial
import copy


class Environment():
    
    def __init__(self, init_populations: List[int], num_genes: List[int], optimal_genotypes: List[np.ndarray],
                 fitness_coefficients: List[float], max_populations: List[int], mutation_probabilities: List[float],
                 mutation_effects: List[float], max_num_children: List[int], interaction_values: List[float], scenario: str,
                 meteor_impact_strategy: str,
                 num_generations: int, **kwargs):

        self.populations = [Population(init_populations[i], num_genes[i], optimal_genotypes[i], fitness_coefficients[i],
                                       max_populations[i], mutation_probabilities[i], mutation_effects[i],
                                       max_num_children[i], interaction_values[i])
                            for i in range(len(init_populations))]
        self.scenario = scenario
        self. meteor_impact_strategy = meteor_impact_strategy
        match scenario:
            case "global_warming":
                self.global_warming_scale = kwargs.get("global_warming_scale", 0.01)
                self.global_warming_var = kwargs.get("global_warming_var", 0.1)
                self._execute_scenario = partial(
                    self._change_optimal_genotypes, self.global_warming_scale,
                    self.global_warming_var, when=range(0, num_generations), distribution=np.random.uniform
                )
            case _:
                self._execute_scenario = lambda current: None
        match meteor_impact_strategy:
            case "every":
                self.meteor_impact_every = kwargs.get("meteor_impact_every", 20)
                self._execute_meteor = partial(
                    self._change_optimal_genotypes, 1, 0.2,
                    when=range(0, num_generations, self.meteor_impact_every),
                    distribution=np.random.normal
                )
            case "at":
                self.meteor_impact_at = kwargs.get("meteor_impact_at", [20, 40])
                self._execute_meteor = partial(
                    self._change_optimal_genotypes, 1, 0.2, when=self.meteor_impact_at,
                    distribution=np.random.normal
                )
            case _:
                self._execute_meteor = lambda current: None
                
             
    def evaluate(self):
        other_mean_fitnesses = [p.prev_mean_fitness for p in self.populations[-1::-1]]
        size_other = self.populations[-1].size
        for i, p in enumerate(self.populations):
            p.fitnesses, p.mean_fitness, p.prev_mean_fitness = p.evaluate(other_mean_fitnesses[i], size_other)
        
    def mutate(self):
        for i, _ in enumerate(self.populations):
            self.populations[i].genotypes = self.populations[i].mutate()
    
    def reproduce(self):
        for i, _ in enumerate(self.populations):
            new_genotypes, new_generation, new_size = self.populations[i].reproduce()
            self.populations[i].genotypes = new_genotypes
            self.populations[i].generation = new_generation
            self.populations[i].size = new_size
    
    def run(self, num_generations: int):
        stats_stacked = {'mean_fitness': {'prey': [], 'predator': []},
                         'size': {'prey': [], 'predator': []},
                         'generation': {'prey': [], 'predator': []},
                         'genotypes': {'prey': [], 'predator': []},
                         'fitnesses': {'prey': [], 'predator': []},
                         'optimal_genotype': {'prey': [], 'predator': []}}
        
        for i in range(num_generations):
            stats = self.log_stats()
            for key in stats_stacked.keys():
                for subkey in stats_stacked[key].keys():
                    stats_stacked[key][subkey].append(stats[key][subkey])
            self.mutate()
            self.reproduce()
            self.evaluate()
            self._execute_scenario(current=i)
            self._execute_meteor(current=i)
            
            progress = (i + 1) / num_generations
            yield progress, stats_stacked
        
    def log_stats(self):
        populations_copy = copy.deepcopy(self.populations)
        stats = {'mean_fitness': {'prey': populations_copy[0].mean_fitness, 'predator': populations_copy[1].mean_fitness},
                 'size': {'prey': populations_copy[0].size, 'predator': populations_copy[1].size},
                 'generation': {'prey': populations_copy[0].generation, 'predator': populations_copy[1].generation},
                 'genotypes': {'prey': populations_copy[0].genotypes, 'predator': populations_copy[1].genotypes},
                 'fitnesses': {'prey': populations_copy[0].fitnesses, 'predator': populations_copy[1].fitnesses},
                 'optimal_genotype': {'prey': populations_copy[0].optimal_genotype, 'predator': populations_copy[1].optimal_genotype}}
        return stats
    
    def _change_optimal_genotypes(
          self, scale: float, var: float, current: int = -1, when: List[int] = None,
          distribution: Callable = np.random.normal, **kwargs):
        if when is None:
            return
        else:
            if current in when:
                for p in self.populations:
                    #ic(distribution(0, var, p.num_genes))
                    p.optimal_genotype += scale * distribution(0, var, p.num_genes)
                    p.optimal_genotype = np.clip(p.optimal_genotype, -1, 1)
                       