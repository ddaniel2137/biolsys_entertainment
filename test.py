import inspect
import streamlit as st
import numpy as np
import pandas as pd
from environment import Environment
from icecream import ic# Assuming this is your environment setup
import streamlit.components.v1 as stc
import pygwalker as pyg


st.set_page_config(
    page_title="Population Simulation",
    layout="wide"
)

def main():
    st.title("Population Simulation")
    setup_sidebar_controls()
    params_required = list(inspect.signature(Environment.__init__).parameters.keys())[1:-1]
    params_required.extend(
        ['seed', 'meteor_impact_strategy', 'global_warming_scale', 'global_warming_var', 'meteor_impact_every',
         'meteor_impact_at'])
    params_provided = {k: st.session_state[k] for k in params_required}
    np.random.seed(st.session_state['seed'])
    # Main layout with three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Column 1")
        # Assume some information or controls specific to Column 1
        st.write("Details or outputs for the first segment of the analysis")

    with col2:
        st.header("Column 2")
        # Assume some information or controls specific to Column 2
        if st.button('Run Simulation'):
            st.session_state['run_simulation'] = True
        
        if 'run_simulation' in st.session_state and st.session_state['run_simulation']:
            env = Environment(**params_provided)
            stats_stacked = run_simulation(env, st.session_state['num_generations'])
            stats_df = preprocess_data(stats_stacked, st.session_state['roles'])
            st.session_state['stats_df'] = stats_df
            st.session_state['run_simulation'] = False
        
        '''if 'stats_df' in st.session_state:
            if st.button('Show animations'):
                st.session_state['show_animations'] = True
            
            if 'show_animations' in st.session_state and st.session_state['show_animations']:
                st.write("Show animations here")
                tabs_animations = st.multiselect("Select animation", ['prey', 'predator'])
                ic2(tabs_animations)'''
                    
                        
                 

    with col3:
        st.header("Column 3")
        # Additional outputs or interactive widgets
        st.write("Interactive widgets or results display here")
    
    if 'stats_df' in st.session_state:
        df = st.session_state['stats_df']
        st.dataframe(df)
        pyg_html = pyg.walk(df, return_html=True).to_html()
        ic2(pyg_html)
        stc.html(pyg_html, height=1000, scrolling=True)
    # Optional: Adding a footer or additional information
    st.sidebar.write("Set parameters on the left to update the charts and data.")


def setup_sidebar_controls() -> None:
    params = {
        'roles': ['prey', 'predator'],
        'num_populations': None,
        'init_populations': None,
        'num_genes': None,
        'optimal_genotypes': None,
        'fitness_coefficients': None,
        'max_populations': None,
        'mutation_probabilities': None,
        'mutation_effects': None,
        'max_num_children': None,
        'interaction_values': None,
        'num_generations': None,
        'scenario': None,
        'seed': None,
        'meteor_impact_strategy': None,
        'global_warming_scale': None,
        'global_warming_var': None,
        'meteor_impact_every': None,
        'meteor_impact_at': None
    }
    roles = ['prey', 'predator']
    params['seed'] = st.sidebar.number_input('Seed', 0, 1000, 42)
    params['num_populations'] = st.sidebar.slider('Number of populations', 1, 10, 2)
    params['init_populations'] = [st.sidebar.slider(f'Initial population {rol}', 1, 1000, 30 * (3 * (2 - i))) for i, rol
                                  in enumerate(roles)]
    params['num_genes'] = [st.sidebar.slider(f'Number of genes {rol}', 1, 10, 3) for i, rol in enumerate(roles)]
    params['optimal_genotypes'] = [np.zeros(params['num_genes'][i]) for i, _ in enumerate(roles)]
    params['fitness_coefficients'] = [st.sidebar.slider(f'Fitness coefficient {i}', 0.1, 10.0, 0.75) for i, rol in
                                      enumerate(roles)]
    params['max_populations'] = [st.sidebar.slider(f'Max population {rol}', 100, 100000, 1000) for i, rol in
                                 enumerate(roles)]
    params['mutation_probabilities'] = [st.sidebar.slider(f'Mutation probability {rol}', 0.0, 1.0, 0.2) for i, rol in
                                        enumerate(roles)]
    params['mutation_effects'] = [st.sidebar.slider(f'Mutation effect {rol}', 0.0, 1.0, 0.1) for i, rol in
                                  enumerate(roles)]
    params['max_num_children'] = [st.sidebar.slider(f'Max children {rol}', 1, 10, 2 * (2 - i)) for i, rol in
                                  enumerate(roles)]
    params['interaction_values'] = [
        st.sidebar.slider(f'Interaction value {role}', i - 1.0, float(i), (-1) ** (i + 1) * 0.4 * (i + 1)) for i, role
        in enumerate(roles)]
    params['num_generations'] = st.sidebar.slider('Number of generations', 1, 1000, 150)
    params['scenario'] = st.sidebar.selectbox('Scenario', ['global_warming', 'None'])
    params['meteor_impact_strategy'] = st.sidebar.selectbox('Meteor impact strategy', ['every', 'at', 'None'])
    if params['scenario'] == 'global_warming':
        params['global_warming_scale'] = st.sidebar.slider('Global warming scale', 0.0, 1.0, 0.01)
        params['global_warming_var'] = st.sidebar.slider('Global warming variance', 0.0, 1.0, 0.1)
    else:
        params['global_warming_scale'] = None
        params['global_warming_var'] = None
    
    if params['meteor_impact_strategy'] == 'every':
        params['meteor_impact_every'] = st.sidebar.slider('Meteor impact every', 1, 100, 20)
        params['meteor_impact_at'] = None
    elif params['meteor_impact_strategy'] == 'at':
        params['meteor_impact_at'] = st.sidebar.multiselect('Meteor impact at',
                                                            list(range(1, params['num_generations'])), [20, 40])
        params['meteor_impact_every'] = None
    else:
        params['meteor_impact_every'] = None
        params['meteor_impact_at'] = None
    
    for key, value in params.items():
        st.session_state[key] = value
    
    return

def run_simulation(env: Environment, num_generations: int) -> dict:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for progress, stats_stacked in env.run(num_generations):
        progress_bar.progress(progress)
        status_text.text(f'Generation {int(progress * num_generations)}/{num_generations}')
    
    progress_bar.empty()
    status_text.empty()
    
    return stats_stacked

def preprocess_data(stats_stacked: dict, roles: list) -> pd.DataFrame:
    """Preprocess nested data into a flat structure suitable for a DataFrame."""
    data = []
    # dataframe with rows: individuals, columns: genotype
    data_pop = []
    for role in roles:
        for gen in range(len(stats_stacked['generation'][role])):
            entry = {
                'role': role,
                'generation': gen,
                'mean_fitness': stats_stacked['mean_fitness'][role][gen],
                'size': stats_stacked['size'][role][gen],
                'optimal_genotype': stats_stacked['optimal_genotype'][role][gen],
                # Flatten genotypes and fitnesses if they are stored as lists or arrays
            }
            # create new
            data.append(entry)
    return pd.DataFrame(data)

def ic2(obj, *args, **kwargs):
    class_name = obj.__class__.__name__
    hierarchy = inspect.getmro(obj.__class__)
    hierarchy_str = " -> ".join(cls.__name__ for cls in hierarchy)
    ic(f"{class_name} ({hierarchy_str})", obj, *args, **kwargs)

if __name__ == "__main__":
    main()