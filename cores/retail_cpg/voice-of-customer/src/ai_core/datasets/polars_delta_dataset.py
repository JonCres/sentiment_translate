import logging
from pathlib import Path
from typing import Any, Dict

import deltalake
import polars as pl
from kedro_datasets.polars import EagerPolarsDataset

logger = logging.getLogger(__name__)

class PolarsDeltaDataset(EagerPolarsDataset):
    """
    Custom Polars Dataset to handle PurePosixPath issue with Delta Lake.
    """
    def _save(self, data: pl.DataFrame) -> None:
        # Convert filepath to string ensuring compatibility with deltalake
        file_path = str(Path(self._filepath).absolute())
        
        if self._file_format == "delta":
            import shutil
            import tempfile
            
            # For local FS on WSL/Windows (/mnt/c), deltalake/object_store often fails
            # with "Upload aborted" due to atomic rename issues on DrvFs.
            # Workaround: Write to a native Linux FS (/tmp) and copy over.
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / "delta_table"
                
                # Extract relevant args for write_deltalake
                mode = self._save_args.get("mode", "error")
                schema_mode = "overwrite" if self._save_args.get("overwrite_schema") else None
                storage_options = self._save_args.get("storage_options", None)
                
                deltalake.write_deltalake(
                    str(tmp_path),
                    data.to_arrow(),
                    mode=mode,
                    schema_mode=schema_mode,
                    storage_options=storage_options
                )
                
                # Now move/copy it to the final location
                target_path = Path(file_path)
                if target_path.exists():
                    shutil.rmtree(target_path)
                
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(tmp_path, target_path)
        else:
            # Fallback for other formats, assuming they also accept string paths
            method = f"write_{self._file_format}"
            getattr(data, method)(file_path, **self._save_args)

    def _load(self) -> pl.DataFrame:
        # Convert filepath to string ensuring compatibility with deltalake
        file_path = str(Path(self._filepath).absolute())
        
        if self._file_format == "delta":
            try:
                # Custom loading using deltalake and pyarrow to avoid Polars 1.x schema bug
                dt = deltalake.DeltaTable(file_path)
                return pl.from_arrow(dt.to_pyarrow_table())
            except Exception as e:
                logger.warning(f"Failed to read delta via pyarrow: {e}. Falling back to pl.read_delta.")
                return pl.read_delta(file_path, **self._load_args)
        else:
             # Fallback to super if we ever use this for non-delta
             return super()._load()
