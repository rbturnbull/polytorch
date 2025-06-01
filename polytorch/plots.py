from pathlib import Path
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from itertools import cycle
import torch

from .embedding import PolyEmbedding
from .data import CategoricalData, OrdinalData


def format_fig(fig) -> go.Figure:
    """Formats a plotly figure in a nicer way."""
    fig.update_layout(
        width=1000,
        height=800,
        plot_bgcolor="white",
        title_font_color="black",
        font=dict(
            family="Linux Libertine Display O",
            size=18,
            color="black",
        ),
    )
    gridcolor = "#dddddd"
    fig.update_xaxes(gridcolor=gridcolor)
    fig.update_yaxes(gridcolor=gridcolor)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside', zeroline=True, zerolinewidth=1, zerolinecolor='black')

    return fig


def plot_embedding(embedding:PolyEmbedding, n_components:int=2, show:bool=False, output_path:str|Path|None=None) -> go.Figure:
    """
    Plots the embedding in 2D or 3D.

    Args:
        embedding (PolyEmbedding): The embedding to plot.
        n_components (int, optional): The number of principal components to plot. Can be 2 or 3. Defaults to 2.
        show (bool, optional): Whether to show the plot. Defaults to False.
        output_path (str|Path|None, optional): The path to save the plot to. 
            Can be in HTML, PNG, JPEG, SVG or PDF. Defaults to None.

    Returns:
        go.Figure: The plotly figure
    """
    if n_components not in [2, 3]:
        raise ValueError(f"n_components must be 2 or 3, not {n_components}")
    
    # get embedding weights
    weights = []
    labels = []
    colors = []

    # The default colors are same as px.colors.qualitative.Plotly
    # I'm not using that directly because that requires plotly express
    # which requires pandas to be installed
    cmap = cycle(['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])

    for input_type, module in zip(embedding.input_types, embedding.embedding_modules):
        weight = module.weight
        if len(weight.shape) == 1:
            weight = weight.unsqueeze(0)
        
        weights.append(weight)
        
        if isinstance(input_type, CategoricalData) and not isinstance(input_type, OrdinalData):
            my_labels = (
                input_type.labels if input_type.labels is not None 
                else [f"{input_type.name}_{i}" for i in range(input_type.category_count)]
            )
            labels.extend(my_labels)

            my_colors = (
                input_type.colors if input_type.colors is not None 
                else [next(cmap) for _ in range(input_type.category_count)]
            )
            colors.extend(my_colors)
        else:
            labels.append(input_type.name)
            colors.append(getattr(input_type, "color", '') or next(cmap))

    weights = torch.cat(weights, dim=0).detach()

    # Perform a principal component analysis
    pca = PCA(n_components=n_components)
    weights_reduced = pca.fit_transform(weights)

    # plot
    fig = go.Figure()

    # This is done as a loop so that the legend has all the different categorical labels separate
    # This will be a large list in the legend potentially
    # This could be an option in the future
    for vector, label, color in zip(weights_reduced, labels, colors):
        if n_components == 2:
            fig.add_trace(go.Scatter(
                x=[vector[0]],
                y=[vector[1]],
                mode='markers',
                name=label,
                marker_color=color,
            ))
            fig.update_xaxes(title_text="Component 1")
            fig.update_yaxes(title_text="Component 2")
        elif n_components == 3:
            fig.add_trace(go.Scatter3d(
                x=[vector[0]],
                y=[vector[1]],
                z=[vector[2]],
                mode='markers',
                name=label,
                marker_color=color,
            ))
            fig.update_layout(scene = dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3',
            ))
        
    fig.update_layout(showlegend=True)
    format_fig(fig)

    # output image as file
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == '.html':
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path))

    if show:
        fig.show()

    return fig