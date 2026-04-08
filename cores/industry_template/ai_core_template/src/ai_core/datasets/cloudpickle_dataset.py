from kedro.io import AbstractDataset
import cloudpickle
from pathlib import Path
from typing import Any, Dict

class CloudPickleDataset(AbstractDataset):
    """
    Custom Dataset to save/load objects using cloudpickle.
    Useful for objects containing lambdas (like lifetimes models).
    """
    def __init__(self, filepath: str):
        self._filepath = Path(filepath)

    def _load(self) -> Any:
        with open(self._filepath, "rb") as f:
            return cloudpickle.load(f)

    def _save(self, data: Any) -> None:
        # Ensure parent directory exists
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "wb") as f:
            cloudpickle.dump(data, f)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)
