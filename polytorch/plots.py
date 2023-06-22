# -*- coding: future_typing -*-

from pathlib import Path
from .embedding import PolyEmbedding
import plotly.graph_objects as go
from sklearn.decomposition import PCA

import torch

from .embedding import ContinuousEmbedding, OrdinalEmbedding
from .data import CategoricalData


def format_fig(fig):
    """Formats a plotly figure in a nicer way."""
    fig.update_layout(
        width=1200,
        height=550,
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

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')

    return fig


def plot_embedding(embedding:PolyEmbedding, n_components:int=2, show:bool=False, output_path:str|Path|None=None) -> go.Figure:
    # get embedding weights
    weights = []
    for input_type, module in zip(embedding.input_types, embedding.embedding_modules):
        weight = module.weight
        if len(weight.shape) == 1:
            weight = weight.unsqueeze(0)
        
        weights.append(weight)
        # if isinstance(module, (OrdinalEmbedding, ContinuousEmbedding)):
        #     weights.append(module.weight.detach())
        # elif isinstance(input_type, CategoricalData):
        #     weights.append(module.weight.detach())
        # else:
        #     raise ValueError("Unknown input type")
    weights = torch.cat(weights, dim=0).detach()

    # PCA
    pca = PCA(n_components=n_components)
    weights_reduced = pca.fit_transform(weights)

    # plot
    fig = go.Figure()

    if n_components == 2:
        fig.add_trace(go.Scatter(
            x=weights_reduced[:,0], 
            y=weights_reduced[:,1],
            mode='markers',
        ))
        fig.update_xaxes(title_text="Component 1")
        fig.update_yaxes(title_text="Component 2")
    elif n_components == 3:
        fig.add_trace(go.Scatter3d(
            x=weights_reduced[:,0], 
            y=weights_reduced[:,1],
            z=weights_reduced[:,2],
            mode='markers',
        ))
        fig.update_layout(scene = dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3',
        ))
    else:
        raise ValueError(f"n_components must be 2 or 3, not {n_components}")
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