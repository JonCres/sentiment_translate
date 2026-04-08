from kedro.io import AbstractDataset
import cloudpickle
from pathlib import Path
from typing import Any, Dict

class CloudPickleDataset(AbstractDataset):
    """Custom Kedro dataset for serializing complex Python objects using cloudpickle.

    CloudPickle extends Python's standard pickle to serialize objects that regular
    pickle cannot handle, including:
    - Lambda functions and nested functions
    - Objects defined in __main__ or interactive sessions
    - Classes with dynamic attributes
    - Lifetimes models (BetaGeoFitter, GammaGammaFitter) which use scipy optimizers

    This dataset is essential for saving probabilistic CLTV models that contain
    lambda functions in their internal optimization state.

    Why CloudPickle for Lifetimes Models?
    - BetaGeoFitter and GammaGammaFitter use scipy.optimize with lambda functions
    - Standard pickle.PickleError: "Can't pickle <lambda>"
    - CloudPickle serializes the function's bytecode and closure

    Attributes:
        _filepath: Path to pickle file (created automatically if doesn't exist)

    Example Usage in catalog.yml:
        bg_nbd_model:
          type: aicore.datasets.cloudpickle_dataset.CloudPickleDataset
          filepath: data/06_models/bg_nbd_model.pkl
          versioned: true  # Optional: Enable timestamped versioning

    Example in Pipeline:
        >>> from lifetimes import BetaGeoFitter
        >>> bgf = BetaGeoFitter(penalizer_coef=0.01)
        >>> bgf.fit(frequency, recency, T)
        >>> # Kedro saves automatically via catalog
        >>> context.catalog.save("bg_nbd_model", bgf)
        >>> # Later: Load model
        >>> loaded_bgf = context.catalog.load("bg_nbd_model")
        >>> predictions = loaded_bgf.predict(...)

    Note:
        - File extension .pkl recommended (not .pickle to avoid confusion with pickle)
        - Parent directories created automatically if missing
        - No compression applied (add gzip if file size is concern)
        - Compatible with Kedro versioning (creates timestamped subfolders)
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
