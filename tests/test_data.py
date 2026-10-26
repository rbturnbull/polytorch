import pandas as pd
import pytest
import torch

from polytorch.data import PolyData, BinaryData, CategoricalData, HierarchicalData, OrdinalData, ContinuousData


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
        


@pytest.mark.parametrize("tuple_state", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("data_class", [
    BinaryData, CategoricalData,
    OrdinalData,
    ContinuousData,
    HierarchicalData,
])
def test_data_restores_current_and_legacy_serialized_state(data_class, legacy, tuple_state):
    from attrs import fields
    from hierarchicalsoftmax import SoftmaxNode

    kwargs = {"name": "restored", "loss_weighting": 0.25}
    if issubclass(data_class, CategoricalData):
        kwargs["category_count"] = 3
    if data_class is HierarchicalData:
        kwargs["root"] = SoftmaxNode("root")
    original = data_class(**kwargs)
    state = {attribute.name: getattr(original, attribute.name) for attribute in fields(data_class)}
    if legacy:
        del state["loss_weighting"]
    restored = data_class.__new__(data_class)
    restored.__setstate__((None, state) if tuple_state else state)

    for name, value in state.items():
        assert getattr(restored, name) == value
    assert restored.loss_weighting == (1.0 if legacy else 0.25)


@pytest.mark.parametrize("mean,stdev", [(10.0, None), (None, 2.0), (10.0, 2.0)])
def test_continuous_loss_normalizes_targets(mean, stdev):
    from polytorch import ContinuousData

    data = ContinuousData(mean=mean, stdev=stdev)
    target = torch.tensor([8.0, 12.0])
    original_target = target.clone()
    prediction = torch.tensor([[1.0], [3.0]])
    normalized = (target - (mean or 0.0)) / (stdev or 1.0)
    expected = torch.nn.functional.smooth_l1_loss(prediction.squeeze(-1), normalized, reduction="none")
    torch.testing.assert_close(data.calculate_loss(prediction, target), expected)
    torch.testing.assert_close(target, original_target)


@pytest.mark.parametrize("values,message", [
    ([], "No data found"),
    ([" / / "], "Invalid hierarchical entry"),
])
def test_hierarchical_data_rejects_empty_paths(values, message):
    with pytest.raises(ValueError, match=message):
        HierarchicalData.from_series("taxonomy", pd.Series(values, dtype=object))
