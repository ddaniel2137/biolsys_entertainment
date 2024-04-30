from itertools import product
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import ParameterGrid
from stqdm import stqdm

from app import preprocess_data
from environment import Environment


def run_simulation(env, num_generations):
    #progress_bar = st.progress(0)
    #status_text = st.empty()
    
    for progress, stats_stacked in env.run(num_generations):
        continue
        #progress_bar.progress(progress)
        #status_text.text(f'Generation {int(progress * num_generations)}/{num_generations}')
    
    #progress_bar.empty()
    # status_text.empty()
    
    return stats_stacked


@st.cache_data
def run_simulations(params, fitness_coefficients):
    results = []
    roles = ['prey', 'predator']
    col_name = "fitness_coefficients"
    for p, q in stqdm(fitness_coefficients):
        params['fitness_coefficients'] = [p, q]
        env = Environment(**params)
        stats_stacked = run_simulation(env, params['num_generations'])
        df = preprocess_data(stats_stacked, roles)
        #ic([[p, q]]* len(df))
        #df[col_name] =  {p},{q}']* len(df)
        df.columns = pd.MultiIndex.from_arrays([df.columns, [''] * len(df.columns)])
        df[(col_name, 'prey')] = [p] * len(df)
        df[(col_name, 'predator')] = [q] * len(df)
        
        #ic(df.info())
        
        results.append(df)
        #progress_bar.progress((fitness_coefficients.index((p, q)) + 1) / len(fitness_coefficients))
    
    #progress_bar.empty()
    return pd.concat(results)


def run_simulation_grid(params):
    env = Environment(**params)
    stats_stacked = run_simulation(env, params['num_generations'])
    df = preprocess_data(stats_stacked, params['roles'])
    return df


def search_optimal_parameters(
      roles: List[str] = ['prey', 'predator'],
      num_populations: int = 2,
      num_generations: int = 200,
      scenario: str = 'global_warming',
      global_warming_scale: float = 1.0,
      global_warming_var: float = 0.05,
      meteor_impact_strategy: str = None,
      meteor_impact_every: int = None,
      meteor_impact_at: List[int] = None,
      seed: int = 42,
      optimal_genotype: np.ndarray = None,
      num_genes: List[int] = [5, 5],
      init_populations: List[int] = [200, 200],
      max_num_children: List[int] = [2, 2],
      max_populations: List[int] = [10000, 10000],
      **kwargs
):
    """
    Searches for optimal parameter settings that maximize population survival across generations.
    
    Fixed parameters:
        - roles: ['prey', 'predator']
        - num_populations: 2
        - num_generations: 200
        - scenario: global_warming
        - global_warming_scale: 1.0
        - global_warming_var: 0.05
        - meteor_impact_strategy: None
        - meteor_impact_every: None
        - meteor_impact_at: None
        - seed: 42
        - optimal_genotype: numpy.NDArray[float]
        - num_genes: 5
        - init_populations: [200, 200]
        - max_num_children: [2, 2]
        -
    Tunable parameters:
        
        - fitness_coefficients (dict): Fitness coefficients for each gene in the genotype. Range from 0 to 2.
            Default value is randomly drawn between the range using a dice button.
            
        - mutation_probabilities (dict): Probability of a gene mutating. Range from 0 to 1.
            Default value is randomly drawn between the range using a dice button.
            
        - mutation_effects (dict): Effect of a gene mutating. Range from 0 to 1.
            Default value is randomly drawn between the range using a dice button.
        
        - interaction_values (dict): Interaction values for each role. Range from -1 to 1.
            Default value is randomly drawn between the range using a dice button.

    """
    #### Extract fixed parameters from kwargs
    fixed_params = {
        'roles': roles,
        'num_populations': num_populations,
        'num_generations': num_generations,
        'scenario': scenario,
        'global_warming_scale': global_warming_scale,
        'global_warming_var': global_warming_var,
        'meteor_impact_strategy': meteor_impact_strategy,
        'meteor_impact_every': meteor_impact_every,
        'meteor_impact_at': meteor_impact_at,
        'seed': seed,
        'optimal_genotype': optimal_genotype,
        'num_genes': num_genes,
        'init_populations': init_populations,
        'max_num_children': max_num_children,
        'max_populations': max_populations,
        'optimal_genotypes': [np.zeros(num_genes[i]) for i, _ in enumerate(roles)]
    }
    
    #### Extract tunable parameters from kwargs
    param_grid = {
        'fitness_coefficients': list(kwargs.get('fitness_coefficients', product([0.1, 0.3, 0.5, 0.75, 1.0, 5.0, 10.0,
                                                                                 100.0], repeat=2))),
        'mutation_effects': list(kwargs.get('mutation_effects', product([0.1, 0.3, 0.5, 0.75, 1.0], repeat=2))),
        'interaction_values': list(kwargs.get('interaction_values', product([-0.7, -0.5, -0.3, -0.1,
                                                                             0.0], [0.0, 0.3, 0.5, 0.7, 0.9]))),
        'mutation_probabilities': list(
            kwargs.get('mutation_probabilities', product([0.1, 0.25, 0.5, 0.75, 0.9], repeat=2))),
    }
    
    #### Run simulations for each parameter setting
    results = []
    for params in stqdm(ParameterGrid(param_grid)):
        params.update(fixed_params)
        results.append(run_simulation_grid(params))
    
    #### Concatenate results into a single DataFrame
    return pd.concat(results)
