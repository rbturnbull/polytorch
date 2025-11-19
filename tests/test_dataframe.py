from pathlib import Path

import pandas as pd
import pytest

from polytorch.data import BinaryData, CategoricalData, ContinuousData, HierarchicalData
from polytorch.dataframe import PolyDataFrame


TEST_DATA_DIR = Path(__file__).parent / "test-data"


def test_poly_dataframe_hierarchical():
    frame = PolyDataFrame(TEST_DATA_DIR / "poly_dataframe_hierarchical.csv")

    assert len(frame) == 6
    assert isinstance(frame.data_types[0], HierarchicalData)
    assert isinstance(frame.data_types[1], ContinuousData)
    assert [dt.name for dt in frame.data_types] == ["tree", "score"]

    hierarchical = frame.data_types[0]
    assert isinstance(hierarchical, HierarchicalData)
    root = hierarchical.root
    assert root.render_equal("""
        __root__
        ├── Animal
        │   ├── Eagle
        │   └── Bird
        ├── Vegetable
        │   ├── Tree
        │   └── Wheat
        └── Mineral
            ├── Rock
            └── Gold
    """)
    assert root.name == "__root__"
    assert len(root.children) == 3  # Animal, Vegetable, Mineral

    assert frame["tree"] == [3, 4, 5, 6, 7, 8]
    assert frame[0] == (3, pytest.approx(0.25))
    assert frame[1] == (4, pytest.approx(0.3))
    assert list(frame["partition"]) == [0, 0, 1, 0, 0, 0]
    assert list(frame["score"]) == [0.25, 0.3, 0.3, 0.9, 0.9, 0.9]
    assert list(frame["score:continuous"]) == [0.25, 0.3, 0.3, 0.9, 0.9, 0.9]


def test_polydataframe_from_csv_parses_types_and_values():
    frame = PolyDataFrame(TEST_DATA_DIR / "poly_dataframe_basic.csv")

    assert len(frame) == 3
    assert isinstance(frame.data_types[0], BinaryData)
    assert isinstance(frame.data_types[1], CategoricalData)
    assert isinstance(frame.data_types[2], ContinuousData)
    assert [dt.name for dt in frame.data_types] == ["is_active", "category", "score"]

    categorical = frame.data_types[1]
    assert isinstance(categorical, CategoricalData)
    assert categorical.category_count == 2

    assert frame[0] == (1, "A", pytest.approx(0.15))
    assert list(frame["is_active"]) == [1, 0, 1]
    assert list(frame["is_active:binary"]) == [1, 0, 1]
    assert frame["category", 1] == "B"
    assert frame["score"][2] == pytest.approx(0.65)
    assert list(frame["score : continuous"]) == [0.15, 0.45, 0.65]
    assert list(frame["notes"]) == ["keep", "drop", "keep"]


def test_polydataframe_accepts_dataframe_and_tuple_indexing():
    df = pd.DataFrame(
        {
            "flag:BinaryData": [True, False, True],
            "group:Categorical": ["train", "test", "train"],
        }
    )
    frame = PolyDataFrame(df)

    assert frame[1] == (False, "test")
    assert frame["flag", 2] == True
    assert list(frame["group"]) == ["train", "test", "train"]


def test_polydataframe_unknown_type_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        PolyDataFrame(TEST_DATA_DIR / "poly_dataframe_unknown.csv")

    assert "Unknown data type 'MysteryData' for column 'value'" in str(excinfo.value)
