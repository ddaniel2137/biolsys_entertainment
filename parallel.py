# parallel.py
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from environment import Environment, run_simulation
from itertools import product
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from app import preprocess_data
from typing import List

def run_simulation_grid(params):
    env = Environment(**params)
    stats_stacked = run_simulation(env, params['num_generations'])
    df = preprocess_data(stats_stacked, params['roles'])
    return df

def search_optimal_parameters_parallel(
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
        'optimal_genotype': optimal_genotype or np.zeros((num_populations, max(num_genes))),
        'num_genes': num_genes,
        'init_populations': init_populations,
        'max_num_children': max_num_children,
        'max_populations': max_populations,
    }

    # Extract tunable parameters from kwargs
    param_grid = {
        # The following is just an example. You would set up your actual parameter grid here.
        'fitness_coefficients': list(kwargs.get('fitness_coefficients', [0.5])),
        'mutation_probabilities': list(kwargs.get('mutation_probabilities', [0.01])),
        # Add more parameters as needed
    }

    # Create all combinations of parameters
    param_list = list(ParameterGrid(param_grid))

    # Run simulations in parallel
    results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_simulation_grid, {**fixed_params, **params}): params for params in param_list}
        for future in concurrent.futures.as_completed(futures):
            param_set = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print('%r generated an exception: %s' % (param_set, exc))

    # Combine results into a single DataFrame
    return pd.concat(results, ignore_index=True)

# Example usage:
# optimal_params_result = search_optimal_parameters_parallel(fitness_coefficients=[0.1, 0.2, 0.3])