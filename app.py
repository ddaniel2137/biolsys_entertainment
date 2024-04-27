import numpy as np
import streamlit as st
from environment import Environment
from typing import List, Dict
from icecream import ic
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from plotly.graph_objs import Figure, Layout, Frame, Scatter, Scattergl
import plotly.express as px
import pandas as pd
import inspect


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
    ic(all_genotypes_scaled)
    ic(all_optimal_genotypes_scaled)
    for gen in range(len(stats_stacked['generation'][role])):
        pca_population = all_genotypes_scaled[lengths[gen]:lengths[gen + 1]]
        ic(pca_population, pca_population.shape)
        pca_optimal = all_optimal_genotypes_scaled[gen, :].reshape(1, -1)
        ic(pca_optimal, pca_optimal.shape)
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

    return stats_stacked


def run_simulations(params, fitness_coefficients):
    results = []
    roles = ['prey', 'predator']
    for p, q in fitness_coefficients:
        params['fitness_coefficients'] = [p, q]
        env = Environment(**params)
        stats_stacked = run_simulation(env, params['num_generations'])
        results.append(preprocess_data(stats_stacked, roles))
    return results


def display_grid(results, fitness_coefficients):
    for df, (p, q) in zip(results, fitness_coefficients):
        st.subheader(f'Fitness Coefficients: {p}, {q}')
        st.dataframe(df)

def setup_interface():
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
    
def setup_sidebar_controls():
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
    params['init_populations'] = [st.sidebar.slider(f'Initial population {rol}', 1, 1000, 30*(3*(2-i))) for i, rol in enumerate(roles)]
    params['num_genes'] = [st.sidebar.slider(f'Number of genes {rol}', 1, 10, 3) for i, rol in enumerate(roles)]
    params['optimal_genotypes'] = [np.zeros(params['num_genes'][i]) for i, _ in enumerate(roles)]
    params['fitness_coefficients'] = [st.sidebar.slider(f'Fitness coefficient {i}', 0.1, 10.0, 0.75) for i, rol in enumerate(roles)]
    params['max_populations'] = [st.sidebar.slider(f'Max population {rol}', 100, 100000, 1000) for i, rol in enumerate(roles)]
    params['mutation_probabilities'] = [st.sidebar.slider(f'Mutation probability {rol}', 0.0, 1.0, 0.2) for i, rol in enumerate(roles)]
    params['mutation_effects'] = [st.sidebar.slider(f'Mutation effect {rol}', 0.0, 1.0, 0.1) for i, rol in enumerate(roles)]
    params['max_num_children'] = [st.sidebar.slider(f'Max children {rol}', 1, 10, 2*(2-i)) for i, rol in enumerate(roles)]
    params['interaction_values'] = [st.sidebar.slider(f'Interaction value {role}', i-1.0, float(i), (-1)**(i+1)*0.4*(i+1)) for i, role in enumerate(roles)]
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
        params['meteor_impact_at'] = st.sidebar.multiselect('Meteor impact at', list(range(1, params['num_generations'])), [20, 40])
        params['meteor_impact_every'] = None
    else:
        params['meteor_impact_every'] = None
        params['meteor_impact_at'] = None

    for key, value in params.items():
        st.session_state[key] = value
    
    return params

import pandas as pd
import numpy as np

def preprocess_data(stats_stacked, roles):
    """Preprocess nested data into a flat structure suitable for a DataFrame."""
    data = []
    for role in roles:
        for gen in range(len(stats_stacked['generation'][role])):
            entry = {
                'role': role,
                'generation': gen,
                'mean_fitness': stats_stacked['mean_fitness'][role][gen],
                'size': stats_stacked['size'][role][gen],
                'optimal_genotype': stats_stacked['optimal_genotype'][role][gen],
                # Flatten genotypes and fitnesses if they are stored as lists or arrays
                'genotypes': ' '.join(map(str, stats_stacked['genotypes'][role][gen])),
                'fitnesses': ' '.join(map(str, stats_stacked['fitnesses'][role][gen]))
            }
            data.append(entry)
    return pd.DataFrame(data)

# Usage remains the same

def main():
    setup_interface()
    params = setup_sidebar_controls()
    params_required = list(inspect.signature(Environment.__init__).parameters.keys())[1:-1]
    params_required.extend(
        ['seed', 'meteor_impact_strategy', 'global_warming_scale', 'global_warming_var', 'meteor_impact_every',
         'meteor_impact_at'])
    params_provided = {k: st.session_state[k] for k in params_required}
    np.random.seed(st.session_state['seed'])
    animation_bool = st.checkbox('Show animation')

    if st.button('Run Simulation'):
        st.session_state['run_simulation'] = True

    if 'run_simulation' in st.session_state and st.session_state['run_simulation']:
        try:
            env = Environment(**params_provided)
            stats_stacked = run_simulation(env, st.session_state['num_generations'])
            # Assuming stats_stacked is structured as: { 'stat_name': [values_over_time], ... }
            # Direct conversion to DataFrame
            df = preprocess_data(stats_stacked, params['roles'])
            #df_new = expand_variable_length_columns(df, 'element')
            ic(df['genotypes'])
            #df.drop(columns=[('genotypes', 'prey'), ('genotypes', 'predator'), ('fitnesses', 'prey'), ('fitnesses', 'predator')], inplace=True)
            #ic(df['genotypes'])
            ic(df)
            ic(df.columns)
            ic(df.head())
            #ic(df.genotypes)
            #ic(df.element_genotypes)
            st.data_editor(df)
            display_results(stats_stacked)

            st.session_state['run_simulation'] = False

            if animation_bool:
                animation_speed_prey = st.sidebar.slider('Animation Speed for Prey (ms per frame)', min_value=0,
                                                         max_value=2000, value=800, step=100)
                animation_speed_predator = st.sidebar.slider('Animation Speed for Predator (ms per frame)', min_value=0,
                                                             max_value=2000, value=800, step=100)
                with st.container():
                    frames_prey = create_frames(stats_stacked, 'prey')
                    frames_predator = create_frames(stats_stacked, 'predator')
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.header('Prey Population')
                        fig_prey = build_figure(frames_prey, 'Prey', animation_speed_prey)
                        st.plotly_chart(fig_prey, use_container_width=True)
                    
                    with col2:
                        st.header('Predator Population')
                        fig_predator = build_figure(frames_predator, 'Predator', animation_speed_predator)
                        st.plotly_chart(fig_predator, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to run simulation: {str(e)}")
            st.session_state['run_simulation'] = False
    
    st.tabs(['Fitness Coefficients', 'Mutation Params', 'Interaction Values', 'Scenario', 'Meteor Impact', 'Global Warming'])
    
    with st.expander('Fitness Coefficients'):
        prey_fitness_range = st.multiselect('Prey fitness range', [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        st.write(prey_fitness_range)
        predator_fitness_range = st.multiselect('Predator fitness range', [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        if st.button('Run Simulations'):
            st.session_state['run_simulations'] = True
        
        if 'run_simulations' in st.session_state and st.session_state['run_simulations']:
        
            
            fitness_coefficients = [(p, q) for p in prey_fitness_range for q in predator_fitness_range]
            results = run_simulations(params_provided, fitness_coefficients)
            for df, (p, q) in zip(results, fitness_coefficients):
                df['prey_fitness'] = p
                df['predator_fitness'] = q
            
            display_grid(results, fitness_coefficients)
            st.session_state['run_simulations'] = False


if __name__ == '__main__':
    main()