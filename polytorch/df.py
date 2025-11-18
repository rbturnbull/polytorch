import pandas as pd
from pathlib import Path

class PolyDataFrame:
    def __init__(self, csv:Path|pd.DataFrame):
        self.pandas_df = pd.read_csv(csv) if isinstance(csv, Path) else csv

        self.data_types = []
        self.getters = {}
        for col in self.pandas_df.columns:
            if ":" in col:
                components = col.split(":")
                assert len(components) == 2, f"Invalid column name with multiple ':' characters: {col}"
                name, type_str = components
                
                name = name.strip()
                type_str = type_str.title().strip()
                if not type_str.endswith("Data"):
                    type_str += "Data"

                try:
                    from .data import globals as data_globals
                    data_type_class = getattr(data_globals, type_str)
                except AttributeError:
                    raise ValueError(f"Unknown data type '{type_str}' for column '{name}'")
                
                data_type, getter = data_type_class.from_series(name, self.pandas_df[col])
                self.data_types.append(data_type)
                self.getters[name] = getter

    def __len__(self):
        return len(self.pandas_df)
    
    def __getitem__(self, idx) -> tuple:
        if isinstance(idx, int):
            return tuple(getter[idx] for getter in self.getters.values())
        
        if isinstance(idx, str):
            if idx in self.getters:
                return self.getters[idx]
            elif idx in self.pandas_df.columns:
                return self.pandas_df[idx].values
        
        if isinstance(idx, tuple):
            assert len(idx) == 2, "Tuple indexing must be of the form (data_name, index)"
            return self[idx[0]][idx[1]]
        
        raise KeyError(f"Invalid index type: {type(idx)}")
