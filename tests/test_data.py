import pandas as pd
import pytest
import torch

from polytorch.data import PolyData, BinaryData, CategoricalData, HierarchicalData


def test_polydata_abstract():
    with pytest.raises(TypeError):
        PolyData()


def test_binary_data_loss_junk():
    data = BinaryData(loss_type="junk")
    with pytest.raises(NotImplementedError):
        target = torch.tensor([False, True, False, True, False]).unsqueeze(1)
        data.calculate_loss(target, target)


def test_categorical_data_loss_junk():
    data = CategoricalData(loss_type="junk", category_count=4)
    with pytest.raises(NotImplementedError):
        target = torch.tensor([0, 2, 3, 2, 3]).unsqueeze(1)
        data.calculate_loss(target, target)        


def test_categorical_data_loss_label_smoothing_validator():
    CategoricalData(label_smoothing=0.0, category_count=4)
    CategoricalData(label_smoothing=1.0, category_count=4)
    with pytest.raises(ValueError):
        CategoricalData(label_smoothing=1.1, category_count=4)
    with pytest.raises(ValueError):
        CategoricalData(label_smoothing=-0.1, category_count=4)


def test_hierarchical_data_from_series_builds_tree_and_reuses_leaves():
    series = pd.Series(
        [
            "Animal/Mammal/Dog",
            "Animal/Mammal/Cat",
            "Animal/Bird/Eagle",
            " Animal / Mammal / Dog ",
        ]
    )

    data, node_ids = HierarchicalData.from_series("taxonomy", series)

    assert isinstance(data, HierarchicalData)
    assert len(node_ids) == len(series)
    assert len(set(node_ids)) == 3
    assert node_ids[0] == node_ids[3]

    root = data.root
    assert root.name == "Animal"
    child_names = {child.name for child in root.children}
    assert child_names == {"Mammal", "Bird"}

    mammal = next(child for child in root.children if child.name == "Mammal")
    assert {child.name for child in mammal.children} == {"Dog", "Cat"}


def test_hierarchical_data_from_series_creates_virtual_root_for_multiple_trees():
    series = pd.Series(
        [
            "Animal/Mammal/Dog",
            "Plant/Tree/Oak",
        ]
    )

    data, node_ids = HierarchicalData.from_series("taxonomy", series)

    assert isinstance(data, HierarchicalData)
    assert len(node_ids) == len(series)
    assert data.root.name == "__root__"
    assert {child.name for child in data.root.children} == {"Animal", "Plant"}
        
