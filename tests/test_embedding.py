import torch
from tempfile import NamedTemporaryFile
from pathlib import Path

from polytorch.data import OrdinalData, ContinuousData, CategoricalData, BinaryData
from polytorch.embedding import OrdinalEmbedding, PolyEmbedding, ContinuousEmbedding


def test_ordinal_embedding_simple():
    embedding_size = 8
    batch_size = 10
    category_count = 5
    
    embedding = OrdinalEmbedding(embedding_size=embedding_size, category_count=category_count)
    ordinal = torch.randint( low=0, high=category_count, size=(batch_size,) )

    embedded = embedding(ordinal)
    assert embedded.shape == (batch_size, embedding_size)
    for i in range(1, embedding_size):
        if ordinal[i-1] != 0:
            torch.testing.assert_close((embedded[i]/embedded[i-1]).min(), ordinal[i]/ordinal[i-1])
            torch.testing.assert_close((embedded[i]/embedded[i-1]).max(), ordinal[i]/ordinal[i-1])


def test_ordinal_embedding_complex():
    embedding_size = 8
    batch_size = 10
    timesteps = 3
    height = width = 128
    category_count = 5
    
    embedding = OrdinalEmbedding(embedding_size=embedding_size, category_count=category_count)
    ordinal = torch.randint( low=0, high=category_count, size=(batch_size, timesteps, height, width) )

    x = embedding(ordinal)
    assert x.shape == (batch_size, timesteps, height, width, embedding_size)


def test_polyembedding_ordinal():
    embedding_size = 8
    batch_size = 10
    timesteps = 3
    height = width = 128
    category_count = 5

    embedding = PolyEmbedding(embedding_size=embedding_size, input_types=[OrdinalData(category_count=category_count)])
    ordinal = torch.randint( low=0, high=category_count, size=(batch_size, timesteps, height, width) )

    x = embedding(ordinal)
    assert x.shape == (batch_size, timesteps, height, width, embedding_size)


def test_continuous_embedding_simple():
    embedding_size = 8
    batch_size = 10
    
    embedding = ContinuousEmbedding(embedding_size=embedding_size)
    continuous = torch.randn( (batch_size, ) )

    embedded = embedding(continuous)
    assert embedded.shape == (batch_size, embedding_size)
    assert embedding.bias.requires_grad == True

    for i in range(1, embedding_size):
        torch.testing.assert_close((embedded[i]/embedded[0]).min(), continuous[i]/continuous[0])
        torch.testing.assert_close((embedded[i]/embedded[0]).max(), continuous[i]/continuous[0])
    

def test_continuous_embedding_simple_no_bias():
    embedding_size = 8
    batch_size = 10
    
    embedding = ContinuousEmbedding(embedding_size=embedding_size, bias=False)
    continuous = torch.randn( (batch_size, ) )

    embedded = embedding(continuous)
    assert embedded.shape == (batch_size, embedding_size)
    assert embedding.bias.requires_grad == False

    for i in range(1, embedding_size):
        torch.testing.assert_close((embedded[i]/embedded[0]).min(), continuous[i]/continuous[0])
        torch.testing.assert_close((embedded[i]/embedded[0]).max(), continuous[i]/continuous[0])
    

def test_continuous_embedding_complex():
    embedding_size = 8
    batch_size = 10
    timesteps = 3
    height = width = 128
    
    embedding = ContinuousEmbedding(embedding_size=embedding_size)
    continuous = torch.randn( (batch_size, timesteps, height, width) )

    embedded = embedding(continuous)
    assert embedded.shape == (batch_size, timesteps, height, width, embedding_size)


def test_polyembedding_continuous():
    embedding_size = 8
    batch_size = 10

    embedding = PolyEmbedding(embedding_size=embedding_size, input_types=[ContinuousData()])
    continuous = torch.randn( (batch_size) )

    embedded = embedding(continuous)
    assert embedded.shape == (batch_size, embedding_size)

    for i in range(1, embedding_size):
        torch.testing.assert_close((embedded[i]/embedded[0]).min(), continuous[i]/continuous[0])
        torch.testing.assert_close((embedded[i]/embedded[0]).max(), continuous[i]/continuous[0])


def test_polyembedding_categorical_simple():
    embedding_size = 8
    batch_size = 10
    category_count = 10
    
    categorical = torch.randint( low=0, high=category_count, size=(batch_size,) )
    embedding = PolyEmbedding(embedding_size=embedding_size, input_types=[CategoricalData(category_count=category_count)])

    embedded = embedding(categorical)
    assert embedded.shape == (batch_size, embedding_size)
    

def test_polyembedding_binary_simple():
    embedding_size = 8
    batch_size = 10
    
    binary = torch.randint( low=0, high=1, size=(batch_size,) )
    embedding = PolyEmbedding(embedding_size=embedding_size, input_types=[BinaryData()])

    embedded = embedding(binary)
    assert embedded.shape == (batch_size, embedding_size)
    

def test_polyembedding_all():
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
    
    size = (batch_size, timesteps, height, width)
    ordinal = torch.randint( low=0, high=ordinal_count, size=size )
    categorical = torch.randint( low=0, high=category_count, size=size )    
    continuous = torch.randn( size )

    x = embedding(ordinal, continuous, categorical)
    assert x.shape == (batch_size, timesteps, height, width, embedding_size)



def test_polyembedding_feature_axis():
    embedding_size = 8
    batch_size = 10
    timesteps = 3
    height = width = 128
    ordinal_count = 7
    category_count = 5

    embedding = PolyEmbedding(embedding_size=embedding_size, feature_axis=2, input_types=[
        OrdinalData(category_count=ordinal_count),
        ContinuousData(),
        CategoricalData(category_count=category_count),  
    ])
    
    size = (batch_size, timesteps, height, width)
    ordinal = torch.randint( low=0, high=ordinal_count, size=size )
    categorical = torch.randint( low=0, high=category_count, size=size )    
    continuous = torch.randn( size )

    x = embedding(ordinal, continuous, categorical)
    assert x.shape == (batch_size, timesteps, embedding_size, height, width)


def test_plot_embedding_2d_from_object():
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


    with NamedTemporaryFile(suffix=".html") as tmp:
        output_path = Path(tmp.name)
        fig = embedding.plot(n_components=2, output_path=output_path)
        assert fig is not None

        fig_json = fig.to_json()
        assert '{"data":[{"marker":{"color":"#636EFA"},"mode":"markers","name":"OrdinalData"' in fig_json
        assert '"xaxis":{"title":{"text":"Component 1"},"gridcolor":"#dddddd","showline":true,"' in fig_json
        assert '"mode":"markers","name":"Earth","x":' in fig_json

        assert output_path.exists()
        assert "<html>" in output_path.read_text()
