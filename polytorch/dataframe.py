"""Typed dataframe loading and indexing for PolyTorch features."""

import numpy as np
import pandas as pd
from pathlib import Path

from . import data as data_module


class PolyDataFrame:
    """Load columns annotated with PolyTorch data types.

    Args:
        csv: A pandas dataframe or a CSV filename supplied as a pathlib.Path.
        data_columns: Annotated column names to include, without type suffixes.
            None or an empty collection includes all annotated columns.

    Attributes:
        pandas_df: The original dataframe, or the dataframe read from CSV.
        data_types: Data type instances in source column order.
        getters: Mapping from selected column names to indexable values.

    Notes:
        Headers use ``name:type``, such as ``score:continuous`` or
        ``group:CategoricalData``. Whitespace around names and types is stripped.
        Unannotated columns remain accessible by their original names but are
        excluded from row tuples. Categorical and ordinal values are not encoded;
        supply integer category indices when using embeddings or cross-entropy.

    Raises:
        ValueError: A selected column has an unknown type, or a requested column
            is missing.
        AssertionError: An annotated header contains multiple colons.

    Examples:
        >>> import pandas as pd
        >>> frame = PolyDataFrame(pd.DataFrame({"score:continuous": [0.25, 0.75]}))
        >>> assert frame[0] == (0.25,)
        >>> assert frame["score", 1] == 0.75
    """

    def __init__(self, csv: Path | pd.DataFrame, data_columns: set[str] | None = None):
        """Parse annotated columns and construct their data types and value getters."""
        self.pandas_df = pd.read_csv(csv) if isinstance(csv, Path) else csv

        data_columns = set(data_columns) if data_columns else None

        self.data_types = []
        self.getters = {}
        for col in self.pandas_df.columns:
            if ":" in col:
                components = col.split(":")
                assert (
                    len(components) == 2
                ), f"Invalid column name with multiple ':' characters: {col}"
                name, type_str = components

                name = name.strip()

                if data_columns and name not in data_columns:
                    continue

                # Make first letter uppercase to match class names
                type_str = type_str.strip()
                type_str = type_str[0].upper() + type_str[1:]

                if not type_str.endswith("Data"):
                    type_str += "Data"

                data_type_class = getattr(data_module, type_str, None)
                if data_type_class is None:
                    raise ValueError(
                        f"Unknown data type '{type_str}' for column '{name}'"
                    )

                data_type, getter = data_type_class.from_series(
                    name, self.pandas_df[col]
                )
                self.data_types.append(data_type)
                self.getters[name] = getter

        if data_columns:
            missing_columns = data_columns - set(self.getters)
            if missing_columns:
                raise ValueError(
                    f"Missing columns from dataframe: {', '.join(missing_columns)}"
                )

    def __len__(self):
        """Return the number of rows in the underlying dataframe."""
        return len(self.pandas_df)

    def __getitem__(self, idx) -> tuple:
        """Return a row, column, or individual value.

        Args:
            idx: A Python or NumPy integer for a row tuple; a string for a column;
                or a ``(column_name, row_index)`` tuple for an individual value.
                Selected annotated columns use names without type suffixes.
                Original dataframe column names also provide access to raw values.

        Returns:
            A row tuple in selected column order, an indexable column, or a value.
            Hierarchical columns return node IDs through their parsed names and
            original path strings through their full annotated headers.

        Raises:
            KeyError: The key type or column name is unsupported.
            IndexError: A row index is out of bounds.
            AssertionError: A tuple key does not contain exactly two elements.
        """
        if isinstance(idx, (int, np.integer)):
            return tuple(getter[idx] for getter in self.getters.values())

        if isinstance(idx, str):
            if idx in self.getters:
                return self.getters[idx]
            elif idx in self.pandas_df.columns:
                return self.pandas_df[idx].values

        if isinstance(idx, tuple):
            assert (
                len(idx) == 2
            ), "Tuple indexing must be of the form (data_name, index)"
            return self[idx[0]][idx[1]]

        raise KeyError(f"Invalid index type: {type(idx)}")
