from pathlib import Path
from typing import Any, Dict

import numpy as np
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class TensorDataset(AbstractDataset[np.ndarray, np.ndarray]):
    """
    A custom Kedro dataset for loading and saving numpy tensors (npy files).
    """

    def __init__(self, filepath: str, save_args: Dict[str, Any] = None):
        """
        Creates a new instance of TensorDataset.

        Args:
            filepath: The location of the file to load / save data.
            save_args: Optional arguments for saving the file (passed to np.save).
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = Path(path)
        self._save_args = save_args or {}

    def _load(self) -> np.ndarray:
        """
        Loads data from the file.

        Returns:
            Data structure containing the loaded data.
        """
        return np.load(self._filepath)

    def _save(self, data: np.ndarray) -> None:
        """
        Saves data to the file.

        Args:
            data: The data to save.
        """
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._filepath, data, **self._save_args)

    def _describe(self) -> Dict[str, Any]:
        """
        Returns a dict that describes the attributes of the dataset.
        """
        return dict(
            filepath=self._filepath, protocol=self._protocol, save_args=self._save_args
        )
