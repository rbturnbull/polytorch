import pytest
from pathlib import Path
from polytorch.plots import plot_embedding
from polytorch import PolyEmbedding, OrdinalData, ContinuousData, CategoricalData
import plotly.graph_objects as go
from tempfile import NamedTemporaryFile
from unittest.mock import patch


def test_plot_embedding_2d():
    embedding_size = 8
    ordinal_count = 7
    category_count = 5

    embedding = PolyEmbedding(embedding_size=embedding_size, input_types=[
        OrdinalData(category_count=ordinal_count),
        ContinuousData(),
        CategoricalData(
            category_count=category_count,
            labels=["Earth", "Fire", "Water", "Air", "Heart"],
            colors=["Brown", "Red", "Blue", "Cyan", "Orange"],
        ),  
    ])

    fig = plot_embedding(embedding, n_components=2)
    assert fig is not None
    assert isinstance(fig, go.Figure)

    fig_json = fig.to_json()
    assert '{"data":[{"marker":{"color":"#636EFA"},"mode":"markers","name":"OrdinalData"' in fig_json
    assert '"xaxis":{"title":{"text":"Component 1"},"gridcolor":"#dddddd","showline":true,"' in fig_json
    assert '"mode":"markers","name":"Earth","x":' in fig_json

    with NamedTemporaryFile(suffix=".html") as tmp:
        output_path = Path(tmp.name)
        plot_embedding(embedding, n_components=2, output_path=output_path)

        assert output_path.exists()
        assert "<html>" in output_path.read_text()

    # todo poetry add kaleido
    # with NamedTemporaryFile(suffix=".jpg") as tmp:
    #     plot_embedding(embedding, n_components=2, output_path=tmp.name)
    #     assert Path(tmp.name).exists()


def test_plot_embedding_3d():
    embedding_size = 8
    ordinal_count = 7
    category_count = 5

    embedding = PolyEmbedding(embedding_size=embedding_size, input_types=[
        OrdinalData(category_count=ordinal_count),
        ContinuousData(),
        CategoricalData(category_count=category_count),  
    ])

    fig = plot_embedding(embedding, n_components=3)
    assert fig is not None
    assert isinstance(fig, go.Figure)

    fig_json = fig.to_json()
    assert '{"data":[{"marker":{"color":"#636EFA"},"mode":"markers","name":"OrdinalData"' in fig_json
    assert '"scene":{"xaxis":{"title":{"text":"Component 1"}},"yaxis":{"title":{"text":"Component 2"}},"zaxis":{"title":{"text":"Component 3"}}}' in fig_json
    assert '"type":"scatter3d"},{"marker"' in fig_json


def test_plot_embedding_wrong_components():
    embedding_size = 8
    ordinal_count = 7
    category_count = 5

    embedding = PolyEmbedding(embedding_size=embedding_size, input_types=[
        OrdinalData(category_count=ordinal_count),
        ContinuousData(),
        CategoricalData(category_count=category_count),  
    ])

    with pytest.raises(ValueError):
        plot_embedding(embedding, n_components=4)


@patch.object(go.Figure, "show")
def test_plot_embedding_show(mock_show):
    embedding_size = 8
    ordinal_count = 7
    category_count = 5

    embedding = PolyEmbedding(embedding_size=embedding_size, input_types=[
        OrdinalData(category_count=ordinal_count),
        ContinuousData(),
        CategoricalData(category_count=category_count),  
    ])

    plot_embedding(embedding, show=True)
    mock_show.assert_called_once()
