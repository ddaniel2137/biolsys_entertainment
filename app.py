import numpy as np
import streamlit as st
from environment import Environment
from typing import List, Dict
import matplotlib.pyplot as plt
from icecream import ic
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from plotly.graph_objs import Figure, Layout, Frame, Scatter


# Helper function to perform PCA and format the data
def prepare_pca_data(population_genotypes, optimal_genotype):
    try:
        pca = PCA(n_components=2)
        pca_population = pca.fit_transform(population_genotypes)
        pca_optimal = pca.transform(optimal_genotype.reshape(1, -1))
        return pca_population, pca_optimal[0]
    except Exception:
        return None, None


# Helper function to create frames for the animation
def create_frames(stats_stacked, role):
    frames = []
    for gen in range(len(stats_stacked['generation'][role])):
        genotypes = np.array(stats_stacked['genotypes'][role][gen])
        optimal_genotype = np.array(stats_stacked['optimal_genotype'][role][gen])
        # Assuming PCA transformation function
        pca_population, pca_optimal = prepare_pca_data(genotypes, optimal_genotype)
        if pca_population is None:
            break
        else:
            frame = Frame(
                data=[
                    Scatter(x=pca_population[:, 0], y=pca_population[:, 1], mode='markers', name='Population',
                            marker={'color': 'blue'}),
                    Scatter(x=[pca_optimal[0]], y=[pca_optimal[1]], mode='markers', name='Optimal',
                            marker={'color': 'red', 'size': 12})
                ],
                layout=Layout(title=f"{role.capitalize()} population genotypes evolution", annotations=[{
                    'text': f"Generation {gen}", 'showarrow': False, 'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.95,
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
    fig, ax = plt.subplots()
    for i, size in enumerate(padded_sizes):
        ax.plot(size, label=roles[i])
    ax.set_xlabel('Generation')
    ax.set_ylabel('Population size')
    ax.legend()
    st.pyplot(fig)


def build_figure(frames, role, animation_speed):
    if frames:
        initial_frame = frames[0]
        fig = Figure(
            data=initial_frame.data,
            layout=Layout(
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
                                            'transition': {'duration': 0}}]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'transition': {'duration': 0}}]
                        }
                    ],
                    'direction': 'left',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
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
    
    num_populations = st.slider('Number of populations', 1, 10, 2)
    init_populations = [st.slider(f'Initial population {i}', 1, 1000, 30) for i in range(num_populations)]
    num_genes = [st.slider(f'Number of genes {i}', 1, 10, 3) for i in range(num_populations)]
    optimal_genotypes = [np.zeros(num_genes[i]) for i, _ in enumerate(num_genes)]
    fitness_coefficients = [st.slider(f'Fitness coefficient {i}', 0.1, 10.0, 1.0) for i in range(num_populations)]
    max_populations = [st.slider(f'Max population {i}', 100, 10000, 1000) for i in range(num_populations)]
    mutation_probabilities = [st.slider(f'Mutation probability {i}', 0.0, 1.0, 0.5) for i in range(num_populations)]
    mutation_effects = [st.slider(f'Mutation effect {i}', 0.0, 1.0, 0.1) for i in range(num_populations)]
    max_num_children = [st.slider(f'Max children {i}', 1, 5, 2) for i in range(num_populations)]
    interaction_values = [st.slider(f'Interaction value {i}', 0.0, 1.0, 0.1) for i in range(num_populations)]
    num_generations = st.slider('Number of generations', 1, 1000, 10)
    scenario = st.selectbox('Scenario', ['global_warming', 'None'])
    if scenario == 'global_warming':
        global_warming_scale = st.slider('Global warming scale', 0.0, 1.0, 0.01)
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
    if st.button('Run simulation'):
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
        
        except Exception as e:
            st.error(f"Failed to run simulation: {str(e)}")
            st.session_state['run_simulation'] = False  # Reset the flag in case of failure


if __name__ == '__main__':
    main()