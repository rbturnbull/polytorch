
from polytorch.plots import plot_embedding
from polytorch import PolyEmbedding, OrdinalData, ContinuousData, CategoricalData
import plotly.graph_objects as go

def test_plot_embedding():
    embedding_size = 8
    batch_size = 10
    timesteps = 3
    height = width = 128
    ordinal_count = 7
    category_count = 5

    embedding = PolyEmbedding(embedding_size=embedding_size, input_types=[
        OrdinalData(category_count=ordinal_count),
        ContinuousData(),
        CategoricalData(category_count=category_count),  
    ])

    fig = plot_embedding(embedding, show=False, n_components=2)
    assert fig is not None
    assert isinstance(fig, go.Figure)

