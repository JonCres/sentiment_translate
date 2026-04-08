from kedro_datasets.polars import EagerPolarsDataset
import polars as pl
import pandas as pd
from typing import Union


class PolarsDeltaDataset(EagerPolarsDataset):
    """
    Custom Polars Dataset to handle PurePosixPath issue with Delta Lake.
    Supports both Polars and Pandas DataFrames.
    """

    def save(self, data: Union[pl.DataFrame, pd.DataFrame]) -> None:
        # Convert pandas to polars if necessary
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)

        # Convert filepath to string ensuring compatibility with deltalake
        file_path = str(self._filepath)

        # Call appropriate polars write method
        if self._file_format == "delta":
            save_args = self._save_args.copy()
            if "overwrite_schema" in save_args:
                overwrite_schema = save_args.pop("overwrite_schema")
                if overwrite_schema:
                    delta_options = save_args.get("delta_write_options", {})
                    delta_options["schema_mode"] = "overwrite"
                    save_args["delta_write_options"] = delta_options
            
            data.write_delta(file_path, **save_args)
        else:
            # Fallback for other formats
            method = f"write_{self._file_format}"
            getattr(data, method)(file_path, **self._save_args)

    def load(self) -> Union[pl.DataFrame, pd.DataFrame]:
        # Convert filepath to string ensuring compatibility with deltalake
        file_path = str(self._filepath)

        # Copy load args to avoid modifying the original dictionary permanently
        load_args = self._load_args.copy()
        return_pandas = load_args.pop("to_pandas", False)

        if self._file_format == "delta":
            try:
                # Try standard read_delta first
                df = pl.read_delta(file_path, **load_args)
            except Exception:
                # Fallback to direct deltalake usage if pl.read_delta fails
                from deltalake import DeltaTable

                dt = DeltaTable(file_path)

                version = load_args.get("version")
                if version is not None:
                    dt = DeltaTable(file_path, version=version)

                df = pl.from_arrow(dt.to_pyarrow_table())
        else:
            # Fallback to super for non-delta
            df = super().load()

        if return_pandas:
            return df.to_pandas()
        return df
