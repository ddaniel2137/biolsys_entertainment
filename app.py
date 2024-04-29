import plotly.graph_objects as go
import numpy as np
import streamlit as st
from environment import Environment
from typing import List, Dict, Union
from icecream import ic
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from plotly.graph_objs import Figure, Layout, Frame, Scatter, Scattergl
import plotly.express as px
import pandas as pd
import inspect
import matplotlib.pyplot as plt
import streamlit.components.v1 as stc
import pygwalker as pyg
import re
from stqdm import stqdm
from itertools import product
import copy
from sklearn.model_selection import ParameterGrid
from concurrent.futures import ThreadPoolExecutor


def main():
    setup_interface()
    params = setup_sidebar_controls()
    params_required = list(inspect.signature(Environment.__init__).parameters.keys())[1:-1]
    params_required.extend(
        ['seed', 'meteor_impact_strategy', 'global_warming_scale', 'global_warming_var', 'meteor_impact_every',
         'meteor_impact_at'])
    params_provided = {k: st.session_state[k] for k in params_required}
    np.random.seed(st.session_state['seed'])

    col = st.columns((1.5, 4.5, 2), gap='medium')
    col0_placeholder = col[0].empty()
    col2_placeholder = col[2].empty()
    
    with col0_placeholder.container():
        st.header('Prey parameters')
        st.write(f"Number of genes: {params['num_genes'][0]}")
        st.write(f"Fitness coefficients: {params['fitness_coefficients'][0]}")
        st.write(f"Mutation probabilities: {params['mutation_probabilities'][0]}")
        st.write(f"Mutation effects: {params['mutation_effects'][0]}")
        st.write(f"Max number of children: {params['max_num_children'][0]}")
        st.write(f"Interaction values: {params['interaction_values'][0]}")
        st.write(f"Initial populations: {params['init_populations'][0]}")
        

    

    with col[1]:
        
        st.header('Simulation')
        if st.button('🎲', key='dice_all'):
            st.session_state["random_all"] = True
            st.experimental_rerun()

        if "random_all" in st.session_state and st.session_state["random_all"]:
            st.session_state["random_all"] = False
        
        if st.button('Run Simulation'):
            st.session_state['run_simulation'] = True
            
        if 'run_simulation' in st.session_state and st.session_state['run_simulation']:
            try:
                env = Environment(**params_provided)
                stats_stacked = run_simulation(env, st.session_state['num_generations'])
                stats_df = preprocess_data(stats_stacked, params['roles'])
                # Assuming stats_stacked is structured as: { 'stat_name': [values_over_time], ... }
                # Direct conversion to DataFrame
                # df = preprocess_data(stats_stacked, params['roles'])
                # df_new = expand_variable_length_columns(df, 'element')
                # ic(df['genotypes'])
                # df.drop(columns=[('genotypes', 'prey'), ('genotypes', 'predator'), ('fitnesses', 'prey'), ('fitnesses', 'predator')], inplace=True)
                # ic(df['genotypes'])
                # ic(df)
                # ic(df.columns)
                # ic(df.head())
                # ic(df.genotypes)
                # ic(df.element_genotypes)
                # st.data_editor(df)
                display_results(stats_stacked)
                st.session_state['stats_df'] = stats_df
                st.session_state['stats_stacked'] = stats_stacked
                st.session_state['run_simulation'] = False
                
            except Exception as e:
                st.error(f"Failed to run simulation: {str(e)}")
                st.session_state['run_simulation'] = False
                
        if 'stats_df' in st.session_state:
            if st.button('Show animations'):
                st.session_state['show_animations'] = True
            
            if 'show_animations' in st.session_state and st.session_state['show_animations']:
                with st.form('animation_speed_form'):
                    animation_speed_prey = st.slider('Animation Speed for Prey (ms per frame)', min_value=0,
                                                     max_value=2000, value=800, step=100)
                    animation_speed_predator = st.slider('Animation Speed for Predator (ms per frame)', min_value=0,
                                                         max_value=2000, value=800, step=100)
                    submit_animation = st.form_submit_button('Submit')
                
                if submit_animation:
                    frames_prey = create_frames(st.session_state['stats_stacked'], 'prey')
                    frames_predator = create_frames(st.session_state['stats_stacked'], 'predator')
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.header('Prey Population')
                        fig_prey = build_figure(frames_prey, 'Prey', animation_speed_prey)
                        st.plotly_chart(fig_prey, use_container_width=True)
                    
                    with col2:
                        st.header('Predator Population')
                        fig_predator = build_figure(frames_predator, 'Predator', animation_speed_predator)
                        st.plotly_chart(fig_predator, use_container_width=True)
    
    with col2_placeholder.container():
        st.header('Predator parameters')
        st.write(f"Number of genes: {params['num_genes'][1]}")
        st.write(f"Fitness coefficients: {params['fitness_coefficients'][1]}")
        st.write(f"Mutation probabilities: {params['mutation_probabilities'][1]}")
        st.write(f"Mutation effects: {params['mutation_effects'][1]}")
        st.write(f"Max number of children: {params['max_num_children'][1]}")
        st.write(f"Interaction values: {params['interaction_values'][1]}")
        st.write(f"Initial populations: {params['init_populations'][1]}")
        
        
    if st.button('Open pygwalker (app layout embedded)'):
        st.session_state['show_analysis'] = True
    
    if 'show_analysis' in st.session_state and st.session_state['show_analysis']:
        df = st.session_state['stats_df']
        pyg_html = pyg.walk(df, return_html=True).to_html()
        stc.html(pyg_html, height=1000, scrolling=True)
    
    tabs_stats = st.tabs(['Fitness Coefficients', 'Grid Search'])
    
    with (tabs_stats[0]):
        with st.form('fitness_coefficients_form'):
            st.write('Select the fitness coefficients for the prey and predator populations')
            
            default_prey_fitness_range = [0.1, 0.5, 1.0, 5.0, 10.0]
            default_predator_fitness_range = default_prey_fitness_range.copy()
            
            custom_prey_fitness_range_input = st.text_input('Custom prey fitness range (comma separated values)')
            custom_predator_fitness_range_input = st.text_input(
                'Custom predator fitness range (comma separated values)')
            
            prey_fitness_range = get_fitness_range(default_prey_fitness_range, custom_prey_fitness_range_input)
            predator_fitness_range = get_fitness_range(default_predator_fitness_range,
                                                       custom_predator_fitness_range_input)
            
            submit_coeffs = st.form_submit_button('Submit')
        
        if submit_coeffs:
            st.session_state['run_simulations'] = True
        
        if st.session_state.get('run_simulations', False):
            fitness_coefficients = [(p, q) for p in prey_fitness_range for q in
                                    predator_fitness_range]
            try:
                st.session_state['results_df'] = run_simulations(params_provided, fitness_coefficients)
                # ic(st.session_state['results_df'].index)
                # ic(st.session_state['results_df'].columns)
                # ic(st.session_state['results_df'].head())
                # ic(st.session_state['results_df'].info())
            except Exception as e:
                st.error(f"Error running simulations: {e}")
            st.session_state['run_simulations'] = False
        
        if 'results_df' in st.session_state:
            if st.button('Display pygwalker (app layout embedded)'):
                st.session_state['pygwalker_coefs'] = True
        
        if 'pygwalker_coefs' in st.session_state and st.session_state['pygwalker_coefs']:
            results_df = st.session_state['results_df']
            try:
                pyg_html2 = pyg.walk(results_df, env='Streamlit').to_html()
                stc.html(pyg_html2, height=1000, scrolling=True)
            except Exception as e:
                st.error(f"Error generating HTML: {e}")

            if st.button('Close pygwalker'):
                st.session_state['pygwalker_coefs'] = False
        
        if st.button('Show visual analysis'):
            st.session_state['show_visual_analysis'] = True
        
        # Example usage within the main application
        if 'show_visual_analysis' in st.session_state and st.session_state['show_visual_analysis']:
            results_df = st.session_state['results_df']
            results_df.columns = ['_'.join(col).strip() for col in results_df.columns.values]
            st.write(results_df.columns)
            plot_population_contour(
                df=st.session_state['results_df'],
                index_col='generation_',
                pivot_col='fitness_coefficients_prey',
                value_col='size_',
                plot_title='Prey Population Sizes',
                xaxis_title='Generation',
                yaxis_title='Fitness Coefficients'
            )
            st.session_state['show_visual_analysis'] = False
    
    with (tabs_stats[1]):
        st.header('Grid Search for Optimal Parameters')
        st.write('This section performs a grid search to find the optimal parameter settings that maximize population survival across generations.')
        st.write('The simulation is run with fixed parameters and tunable parameters are varied across a grid of values.')
        st.write('The results are displayed in a table for further analysis.')
        
        st.write('Do you wanna run the grid search?')
        if st.button('Yeah'):
            st.session_state['run_grid_search'] = True
            
        if 'run_grid_search' in st.session_state and st.session_state['run_grid_search']:
            results = search_optimal_parameters()
            results.to_csv('results_grid.csv', index=False)
            st.write('Grid search complete!')
            st.session_state['grid_search_results'] = results
            
            st.write('End of grid search results.')
    
            
@st.cache_data
def create_frames(stats_stacked, role):
    frames = []
    all_genotypes = np.vstack([g for g in stats_stacked['genotypes'][role] if g.size > 0])
    all_optimal_genotypes = np.vstack(stats_stacked['optimal_genotype'][role])
    lengths = np.cumsum([0, *[g.shape[0] for g in stats_stacked['genotypes'][role]]])
    pca = TruncatedSVD(n_components=2)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    all_genotypes_scaled = scaler.fit_transform(pca.fit_transform(np.vstack([all_genotypes, all_optimal_genotypes])))
    all_genotypes_scaled, all_optimal_genotypes_scaled = np.split(all_genotypes_scaled, [len(all_genotypes)])
    #ic(all_genotypes_scaled)
    #ic(all_optimal_genotypes_scaled)
    for gen in range(len(stats_stacked['generation'][role])):
        pca_population = all_genotypes_scaled[lengths[gen]:lengths[gen + 1]]
        #ic(pca_population, pca_population.shape)
        pca_optimal = all_optimal_genotypes_scaled[gen, :].reshape(1, -1)
        #ic(pca_optimal, pca_optimal.shape)
        if pca_population.size > 0:
            frame_data = [
                Scatter(x=pca_population[:, 0], y=pca_population[:, 1], mode='markers', name='Population',
                        marker={'color': stats_stacked['fitnesses'][role][gen], 'size': 8, 'colorscale': 'Inferno',
                                'cmin': 0.0, 'cmid': 0.5, 'cmax': 1.0, 'opacity': 0.5,# 'Viridis
                                'colorbar': {'title': 'Fitness'}}),
                Scatter(x=[pca_optimal[0, 0]], y=[pca_optimal[0, 1]], mode='markers', name='Optimal',
                        marker={'color': 'green', 'size': 12, 'symbol': 'cross'})
            ]
        else:
            frame_data = [
                Scatter(x=[pca_optimal[0, 0]], y=[pca_optimal[0, 1]], mode='markers', name='Optimal (only)',
                        marker={'color': 'red', 'size': 12, 'symbol': 'star'})
            ]
            
        frame = Frame(
            data=frame_data,
            layout=Layout(title=f"{role.capitalize()} population genotypes evolution", annotations=[{
                'text': f"Generation {gen} (optimal only)", 'showarrow': False, 'xref': 'paper', 'yref': 'paper',
                'x': 0.5, 'y': 0.95,
                'xanchor': 'center', 'yanchor': 'bottom', 'font': {'size': 16}
            }] if pca_population.size == 0 else [{
                'text': f"Generation {gen}", 'showarrow': False, 'xref': 'paper', 'yref': 'paper', 'x': 0.5,
                'y': 0.95,
                'xanchor': 'center', 'yanchor': 'bottom', 'font': {'size': 16}
            }])
        )
        frames.append(frame)
        
    return frames



def pad_sizes(sizes: List[np.ndarray], max_size: int) -> List[np.ndarray]:
    padded_sizes = [np.pad(size, (0, max_size - len(size))) for size in sizes]
    return padded_sizes


def plot_population_sizes(sizes: List[int], roles: Dict[int, str]) -> None:
    max_size = max([len(size) for size in sizes])
    padded_sizes = pad_sizes(sizes, max_size)
    fig = go.Figure()
    for i, size in enumerate(padded_sizes):
        fig.add_trace(go.Scatter(x=list(range(max_size)), y=size, mode='lines', name=roles[i]))
    fig.update_layout(title='Population Sizes', xaxis_title='Generation', yaxis_title='Size')
    st.plotly_chart(fig)

def plot_fitnesses(fitnesses: List[np.ndarray], roles: Dict[int, str]) -> None:
    max_size = max([len(fitness) for fitness in fitnesses])
    padded_fitnesses = pad_sizes(fitnesses, max_size)
    fig = go.Figure()
    for i, fitness in enumerate(padded_fitnesses):
        fig.add_trace(go.Scattergl(x=list(range(max_size)), y=fitness, mode='lines', name=roles[i]))
    fig.update_layout(title='Fitnesses', xaxis_title='Generation', yaxis_title='Fitness')
    st.plotly_chart(fig)

@st.cache_data
def build_figure(frames, role, animation_speed):
    if frames:
        initial_frame = frames[0]
        fig = Figure(
            data=initial_frame.data,
            layout=Layout(
                autosize=True,
                xaxis=dict(range=[-1, 1], autorange=False, zeroline=True, showgrid=False),
                yaxis=dict(range=[-1, 1], autorange=False, zeroline=True, showgrid=False),
                title=f"{role.capitalize()} Population Genotypes Evolution",
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {'frame': {'duration': animation_speed, 'redraw': True},
                                               'fromcurrent': True, 'transition': {'duration': 0}}]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                              'mode': 'immediate', 'transition': {'duration': 0}}]
                        },
                        {
                            'label': 'Reset',
                            'method': 'animate',
                            'args': [{'frame': {'duration': 0, 'redraw': True}},
                                     {'mode': 'immediate', 'transition': {'duration': 0}}],
                        }
                    ],
                    'direction': 'right',
                    'x': 1,
                    'xanchor': 'right',
                    'y': -0.3,
                    'yanchor': 'bottom'
                }]
            ),
            frames=frames
        )
        return fig
    return None

def run_simulation(env, num_generations):
    progress_bar = st.progress(0)
    status_text = st.empty()

    for progress, stats_stacked in env.run(num_generations):
        progress_bar.progress(progress)
        status_text.text(f'Generation {int(progress * num_generations)}/{num_generations}')
    
    progress_bar.empty()
    status_text.empty()

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
        df.columns = pd.MultiIndex.from_arrays([df.columns, ['']*len(df.columns)])
        df[(col_name, 'prey')] = [p]*len(df)
        df[(col_name, 'predator')] = [q]*len(df)

        #ic(df.info())
        
        results.append(df)
        #progress_bar.progress((fitness_coefficients.index((p, q)) + 1) / len(fitness_coefficients))

    #progress_bar.empty()
    return pd.concat(results)


def setup_interface():
    st.set_page_config(
        page_title="Population Simulation",
        layout="wide"
    )
    st.title('Evolutionary Simulation')
    st.write('This is a simple simulation of evolution. The goal is to evolve a population of individuals to match a target genotype.')
    st.write('The population evolves through mutation and reproduction, with genotypes evolving towards a target through simulated genetic processes.')
    # ... rest of your setup code

def display_results(stats_stacked):
    st.subheader('Population Sizes')
    sizes = [stats_stacked['size'][role] for role in ['prey', 'predator']]
    roles = {0: 'Prey', 1: 'Predator'}
    plot_population_sizes(sizes, roles)

    st.subheader('Fitnesses')
    fitnesses = [stats_stacked['mean_fitness'][role] for role in ['prey', 'predator']]
    roles = {0: 'Prey', 1: 'Predator'}
    plot_fitnesses(fitnesses, roles)

def create_slider_with_dice(label: str, min_value: Union[int, float, List[Union[int, float]]], max_value: Union[int, float, List[Union[int, float]]], default_value: Union[int, float, List[Union[int, float]]], key: str) -> Union[Union[int, float], List[Union[int, float]]]:
    """
    Creates sliders and dice buttons for randomization. Supports both single values and lists.

    Args:
        label (str): The label for the slider.
        min_value (Union[int, float, List[Union[int, float]]]): The minimum value for the slider.
        max_value (Union[int, float, List[Union[int, float]]]): The maximum value for the slider.
        default_value (Union[int, float, List[Union[int, float]]]): The default value(s) for the slider.
        key (str): The key to identify the slider.

    Returns:
        Union[Union[int, float], List[Union[int, float]]]: The value(s) corresponding to the slider(s).
    """
    # Normalize inputs using match case
    match min_value:
        case list() as lst:
            min_values = lst
        case _:
            min_values = [min_value] * (len(default_value) if isinstance(default_value, list) else 1)
    
    match max_value:
        case list() as lst:
            max_values = lst
        case _:
            max_values = [max_value] * (len(default_value) if isinstance(default_value, list) else 1)
    
    match default_value:
        case list() as lst:
            default_values = lst
        case _:
            default_values = [default_value]

    values = []
    for i, (min_val, max_val, def_val) in enumerate(zip(min_values, max_values, default_values)):
        slider_label = f"{label} {i+1}" if len(default_values) > 1 else label
        slider_key = f"{key}_{i}"
        random_state = f"random_{slider_key}"
        if st.sidebar.button('🎲', key=f'dice_{slider_key}_button') or st.session_state.get('random_all', False) and slider_key != 'num_populations_0':
            rng = np.random.default_rng()
            st.session_state[f"dice_{slider_key}"] = rng.uniform(min_val, max_val) if isinstance(def_val, float) else rng.integers(min_val, max_val + 1)
            st.session_state[random_state] = True
        
        if random_state in st.session_state and st.session_state[random_state]:
            def_val = copy.deepcopy(st.session_state[f"dice_{slider_key}"])
    
        value = st.sidebar.slider(slider_label, min_val, max_val, def_val, key=slider_key)

        if value != st.session_state[slider_key]:
            st.session_state[random_state] = False

        values.append(value)

    return values if len(values) > 1 else values[0]
   
def setup_sidebar_controls():
    params = {
        'roles': ['prey', 'predator'],
        'seed': None,
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
        'meteor_impact_strategy': None,
        'global_warming_scale': None,
        'global_warming_var': None,
        'meteor_impact_every': None,
        'meteor_impact_at': None
    }
    roles = params['roles']
    
    params['seed'] = st.sidebar.number_input('Seed', 0, 1000, 42)
    params['num_populations'] = create_slider_with_dice('Number of populations', 1, 10, 2, 'num_populations')
    params['init_populations'] = create_slider_with_dice(f'Initial population', [1, 1], [1000, 1000], [200, 200], 'init_populations')
    params['num_genes'] = create_slider_with_dice(f'Number of genes', [2, 2], [10, 10], [5, 5], 'num_genes')
    params['fitness_coefficients'] = create_slider_with_dice(f'Fitness coefficient', [0.1, 0.1], [10.0, 10.0], [0.75, 0.75], 'fitness_coefficients')
    params['max_populations'] = create_slider_with_dice(f'Max population', [100, 100], [10000, 10000], [1000, 1000], 'max_populations')
    params['mutation_probabilities'] = create_slider_with_dice(f'Mutation probability', [0.0, 0.0], [1.0, 1.0], [0.15, 0.15], 'mutation_probabilities')
    params['mutation_effects'] = create_slider_with_dice(f'Mutation effect', [0.0, 0.0], [1.0, 1.0], [0.1, 0.1], 'mutation_effects')
    params['max_num_children'] = create_slider_with_dice(f'Max number of children', [1, 1], [10, 10], [2, 2], 'max_num_children')
    params['interaction_values'] = create_slider_with_dice(f'Interaction value', [-1.0, 0.0], [0.0, 1.0], [-0.6, 0.8], 'interaction_values')
    params['num_generations'] = create_slider_with_dice('Number of generations', 1, 1000, 300, 'num_generation')
    params['scenario'] = st.sidebar.selectbox('Scenario', ['global_warming', 'None'])
    params['meteor_impact_strategy'] = st.sidebar.selectbox('Meteor impact strategy', ['None', 'every', 'at'])
    if params['scenario'] == 'global_warming':
        params['global_warming_scale'] = create_slider_with_dice('Global warming scale', 0.0, 1.0, 1.0, 'global_warming_scale')
        params['global_warming_var'] = create_slider_with_dice('Global warming variance', 0.0, 1.0, 0.05, 'global_warming_var')    
    else:
        params['global_warming_scale'] = None
        params['global_warming_var'] = None
        
    if params['meteor_impact_strategy'] == 'every':
        params['meteor_impact_every'] = create_slider_with_dice( 'Meteor impact every', 1, 100, 20, 'meteor_impact_every')
        params['meteor_impact_at'] = None
    elif params['meteor_impact_strategy'] == 'at':
        params['meteor_impact_at'] = st.sidebar.multiselect('Meteor impact at', list(range(1, params['num_generations'])), [20, 40]) # type: ignore
        params['meteor_impact_every'] = None
    else:
        params['meteor_impact_every'] = None
        params['meteor_impact_at'] = None
    params['optimal_genotypes'] = [np.zeros(params['num_genes'][i]) for i, _ in enumerate(roles)] # type: ignore
    for key, value in params.items():
        st.session_state[key] = value
    
    return params

def preprocess_data(stats_stacked, roles):
    """Preprocess nested data into a flat structure suitable for a DataFrame."""
    data = []
    for role in stqdm(roles):
        for gen in range(len(stats_stacked['generation'][role])):
            entry = {
                'role': role,
                'generation': gen,
                'mean_fitness': stats_stacked['mean_fitness'][role][gen],
                'size': stats_stacked['size'][role][gen],
                'optimal_genotype': stats_stacked['optimal_genotype'][role][gen],
                # 
                'genotypes': stats_stacked['genotypes'][role][gen].flatten(),
                'fitnesses': stats_stacked['fitnesses'][role][gen].flatten()
            }
            #ic(entry['genotypes'])
            #ic(entry['fitnesses'])
            #ic(entry['optimal_genotype'])
            #ic(entry['role'])
            data.append(entry)
        
    return pd.DataFrame(data)


# Refactored code
def get_fitness_range(default_range, custom_range_input):
    """Get the fitness range from the user input."""
    if custom_range_input:
        try:
            custom_range = set(map(float, re.split(r',\s*', custom_range_input)))
        except ValueError:
            st.error("Invalid input for fitness range. Please use comma separated float values.")
            return default_range
        custom_range.update(default_range)
        return custom_range
    return default_range


def plot_population_contour(df, index_col, pivot_col, value_col, plot_title, xaxis_title, yaxis_title):
    """
    Plots a contour chart for population data across specified indices and columns.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the simulation results.
    index_col (str): Column to use as the index in the pivot table.
    pivot_col (str): Column to use as the columns in the pivot table.
    value_col (str): Column to use as the values in the pivot table.
    plot_title (str): Title for the plot.
    xaxis_title (str): Title for the X-axis.
    yaxis_title (str): Title for the Y-axis.
    """
    # Ensure the DataFrame is filled with zeros for NaN values
    df_filled = df.fillna(0)

    # Extract the X, Y and Z values for the contour plot
    X = df_filled[pivot_col].values
    Y = df_filled[index_col].values
    Z = df_filled[value_col].values

    # Create the contour plot
    fig = go.Figure(data=go.Contour(
        z=Z,
        x=X,
        y=Y,
        colorscale='Viridis'
    ))

    # Update the layout of the plot
    fig.update_layout(
        title=plot_title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=1000,
        width=1200
    )

    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

def run_simulation_grid(params):
    env = Environment(**params)
    stats_stacked = run_simulation(env, params['num_generations'])
    df = preprocess_data(stats_stacked, params['roles'])
    return df
@st.cache_data
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
        'fitness_coefficients': list(kwargs.get('fitness_coefficients', product([0.1, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 10.0, 100.0], repeat=2))),
        'mutation_effects': list(kwargs.get('mutation_effects', product([0.1, 0.3, 0.5, 0.75, 1.0], repeat=2))),
        'interaction_values': list(kwargs.get('interaction_values', product([-0.8, -0.6, -0.5, -0.4, -0.2, -0.1, 0.0], [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]))),
        'mutation_probabilities': list(kwargs.get('mutation_probabilities', product([0.1, 0.25, 0.5, 0.75, 0.9], repeat=2))),
    }
    
    #### Generate all combinations of parameters 
    param_combinations = ParameterGrid(param_grid)
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the number of workers as needed
        futures = []
        for param_tuned in stqdm(param_combinations):
        # Submit simulation tasks
            
            future = executor.submit(run_simulation_grid, {**fixed_params, **param_tuned})
            futures.append(future)
        # Collect results
        results = [future.result() for future in stqdm(futures)]
        results_df = pd.concat(results)
    #### Return results as a DataFrame   
    return results_df
 



        


if __name__ == '__main__':
    main()

