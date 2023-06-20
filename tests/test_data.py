
from polytorch.data import PolyData
import pytest


def test_polydata_abstract():
    data = PolyData()
    with pytest.raises(NotImplementedError):
        data.create_module(embedding_size=1)

    with pytest.raises(NotImplementedError):
        data.size()        