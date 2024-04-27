import numpy as np
import streamlit as st
from environment import Environment
from typing import List, Dict
from icecream import ic
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from plotly.graph_objs import Figure, Layout, Frame, Scatter, Scattergl


# Helper function to create frames for the animation
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


def main():
    st.title('Evolutionary Simulation')
    st.write(
        'This is a simple simulation of evolution. The goal is to evolve a population of individuals to match a target genotype.')
    st.write(
        'The population evolves through mutation and reproduction, with genotypes evolving towards a target through simulated genetic processes.')
    roles = ['prey', 'predator']
    num_populations = st.slider('Number of populations', 1, 10, 2)
    init_populations = [st.slider(f'Initial population {rol}', 1, 1000, 30*(3*(2-i))) for i, rol in enumerate(roles)]
    num_genes = [st.slider(f'Number of genes {rol}', 1, 10, 4) for i, rol in enumerate(roles)]
    optimal_genotypes = [np.zeros(num_genes[i]) for i, _ in enumerate(roles)]
    fitness_coefficients = [st.slider(f'Fitness coefficient {i}', 0.1, 10.0, 0.75) for i, rol in enumerate(roles)]
    max_populations = [st.slider(f'Max population {rol}', 100, 100000, 1000) for i, rol in enumerate(roles)]
    mutation_probabilities = [st.slider(f'Mutation probability {rol}', 0.0, 1.0, 0.5) for i, rol in enumerate(roles)]
    mutation_effects = [st.slider(f'Mutation effect {rol}', 0.0, 1.0, 0.1) for i, rol in enumerate(roles)]
    max_num_children = [st.slider(f'Max children {rol}', 1, 10, 3*(2-i)) for i, rol in enumerate(roles)]
    interaction_values = [st.slider(f'Interaction value {role}', i-1.0, float(i), (-1)**(i+1)*0.2) for i, role in enumerate(roles)]
    num_generations = st.slider('Number of generations', 1, 1000, 10)
    scenario = st.selectbox('Scenario', ['global_warming', 'None'])
    if scenario == 'global_warming':
        global_warming_scale = st.slider('Global warming scale', 0.0, 1.0, 1.0)
        global_warming_var = st.slider('Global warming var', 0.0, 1.0, 0.1)
    else:
        global_warming_scale = None
        global_warming_var = None
    meteor_impact_strategy = st.selectbox('Meteor impact strategy', ['every', 'at', 'None'])
    if meteor_impact_strategy == 'every':
        meteor_impact_every = st.slider('Meteor impact every', 1, 100, 20)
        meteor_impact_at = None
    elif meteor_impact_strategy == 'at':
        meteor_impact_at = st.multiselect('Meteor impact at', list(range(1, num_generations)), [20, 40])
        meteor_impact_every = None
    else:
        meteor_impact_every = None
        meteor_impact_at = None
    animation_speed_prey = st.slider('Animation Speed for Prey (ms per frame)', min_value=0, max_value=2000, value=800,
                                     step=100)
    animation_speed_predator = st.slider('Animation Speed for Predator (ms per frame)', min_value=0, max_value=2000,
                                         value=800, step=100)
    animation_bool = st.checkbox('Show animation')
    if st.button('Run Simulation'):
        st.session_state['run_simulation'] = True
    if 'run_simulation' in st.session_state and st.session_state['run_simulation']:
        try:
            env = Environment(
                init_populations, num_genes, optimal_genotypes, fitness_coefficients, max_populations,
                mutation_probabilities, mutation_effects, max_num_children, interaction_values, scenario, meteor_impact_strategy,
                num_generations, global_warming_scale=global_warming_scale, global_warming_var=global_warming_var,
                meteor_impact_every=meteor_impact_every, meteor_impact_at=meteor_impact_at
            )
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for progress, stats_stacked in env.run(num_generations):
                progress_bar.progress(progress)
                status_text.text(f'Generation {int(progress * num_generations)}/{num_generations}')
            
            # st.write(stats_stacked)
            
            st.subheader('Population Sizes')
            sizes = [stats_stacked['size'][role] for role in ['prey', 'predator']]
            roles = {0: 'Prey', 1: 'Predator'}
            plot_population_sizes(sizes, roles)
            
            st.subheader('Fitnesses')
            fitnesses = [stats_stacked['mean_fitness'][role] for role in ['prey', 'predator']]
            roles = {0: 'Prey', 1: 'Predator'}
            plot_fitnesses(fitnesses, roles)
            
            
            st.session_state['run_simulation'] = False
            frames_prey = create_frames(stats_stacked, 'prey') if animation_bool else None
            frames_predator = create_frames(stats_stacked, 'predator') if animation_bool else None
            if animation_bool:
                # Initial frame setup
                col1, col2 = st.columns(2)
                with col1:
                    st.header('Prey Population')
                    fig_prey = build_figure(frames_prey, 'Prey', animation_speed_prey)
                    st.plotly_chart(fig_prey, use_container_width=True)
                
                with col2:
                    st.header('Predator Population')
                    fig_predator = build_figure(frames_predator, 'Predator', animation_speed_predator)
                    st.plotly_chart(fig_predator, use_container_width=True)
                  # Reset the flag after running the simulation
        
        except Exception as e:
            st.error(f"Failed to run simulation: {str(e)}")
            st.session_state['run_simulation'] = False  # Reset the flag in case of failure


if __name__ == '__main__':
    main()