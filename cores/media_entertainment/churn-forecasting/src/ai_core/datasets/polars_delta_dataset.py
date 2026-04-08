from kedro_datasets.polars import EagerPolarsDataset
import polars as pl
from typing import Any, Dict

class PolarsDeltaDataset(EagerPolarsDataset):
    """
    Custom Polars Dataset to handle PurePosixPath issue with Delta Lake.
    """
    def save(self, data: pl.DataFrame) -> None:
        # Convert filepath to string ensuring compatibility with deltalake
        file_path = str(self._filepath)
        
        # Call appropriate polars write method
        # We rely on configured save_args
        if self._file_format == "delta":
            data.write_delta(file_path, **self._save_args)
        else:
            # Fallback for other formats, assuming they also accept string paths
            method = f"write_{self._file_format}"
            getattr(data, method)(file_path, **self._save_args)

    def load(self) -> pl.DataFrame:
        # Convert filepath to string ensuring compatibility with deltalake
        file_path = str(self._filepath)
        
        if self._file_format == "delta":
            try:
                # Try standard read_delta first
                return pl.read_delta(file_path, **self._load_args)
            except Exception as e:
                # Fallback to direct deltalake usage if pl.read_delta fails (e.g. Schema iterable error)
                from deltalake import DeltaTable
                dt = DeltaTable(file_path)
                # Pass load_args filtering if needed? assuming load_args empty for now or basic
                # If load_args contains 'version', etc, handle:
                version = self._load_args.get("version")
                if version is not None:
                    dt = DeltaTable(file_path, version=version)
                
                return pl.from_arrow(dt.to_pyarrow_table())
        else:
             # Fallback to super if we ever use this for non-delta
             return super().load()
