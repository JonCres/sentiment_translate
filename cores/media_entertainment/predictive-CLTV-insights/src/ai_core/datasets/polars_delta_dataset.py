from kedro_datasets.polars import EagerPolarsDataset
import polars as pl
from typing import Any, Dict

class PolarsDeltaDataset(EagerPolarsDataset):
    """Custom Kedro dataset for Polars + Delta Lake integration with path compatibility fixes.

    Extends Kedro's EagerPolarsDataset to properly handle Delta Lake file paths by
    converting PurePosixPath objects to strings. This resolves compatibility issues
    between Kedro's pathlib usage and Delta Lake's string-based path requirements.

    Delta Lake Benefits:
    - ACID transactions: Atomic writes prevent partial data corruption
    - Time travel: Query historical versions for point-in-time analysis
    - Schema evolution: Add columns without breaking existing pipelines
    - Efficient updates: Row-level updates without full rewrites
    - Scalability: Handles billions of rows efficiently

    Use Cases in CLTV Pipeline:
    - Feature store tables (RFM features, behavioral features)
    - Intermediate processed datasets (cleaned transactions)
    - Model input datasets (training/test splits with versioning)

    Path Handling:
    - Converts Kedro's PurePosixPath to string for Delta Lake compatibility
    - Handles both read_delta() and DeltaTable() APIs
    - Falls back gracefully if schema errors occur

    Attributes:
        _filepath: Path to Delta Lake directory (inherited from parent)
        _file_format: Always 'delta' for this dataset
        _save_args: Delta write options (mode, schema_mode, etc.)
        _load_args: Delta read options (version, columns, etc.)

    Example Usage in catalog.yml:
        processed_features_delta:
          type: aicore.datasets.polars_delta_dataset.PolarsDeltaDataset
          filepath: data/05_model_input/processed_features
          save_args:
            mode: overwrite  # or 'append', 'error', 'ignore'
            schema_mode: overwrite  # Allow schema changes
          load_args:
            version: 0  # Optional: Load specific version (time travel)

    Example in Pipeline:
        >>> import polars as pl
        >>> df = pl.DataFrame({"customer_id": ["C1", "C2"], "clv": [500, 300]})
        >>> # Save to Delta Lake
        >>> context.catalog.save("processed_features_delta", df)
        >>> # Load latest version
        >>> loaded_df = context.catalog.load("processed_features_delta")
        >>> # Time travel: Load version 5
        >>> catalog_entry = context.catalog._get_dataset("processed_features_delta")
        >>> catalog_entry._load_args = {"version": 5}
        >>> historical_df = catalog_entry.load()

    Fallback Mechanism:
        If pl.read_delta() fails (e.g., schema iteration errors), automatically
        falls back to DeltaTable().to_pyarrow_table() conversion.

    Note:
        - Delta Lake directory contains Parquet files + _delta_log/ transaction log
        - First save creates directory structure automatically
        - Requires deltalake Python package: `pip install deltalake`
        - Compatible with Feast feature store offline store backend
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
