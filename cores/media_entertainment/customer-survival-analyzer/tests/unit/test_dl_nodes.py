import pytest
import pandas as pd
import numpy as np
from src.ai_core.pipelines.data_science.dl_nodes import (
    train_deepsurv_model,
    train_nmtlr_model,
)
from pycox.models import DeepSurv, NMTLR


@pytest.fixture
def sample_data():
    """Provides sample data for testing Deep Learning survival models."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Features
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features).astype("float32"),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Durations and Events
    duration = np.random.uniform(1, 100, size=n_samples).astype("float32")
    event = np.random.randint(0, 2, size=n_samples).astype("int32")
    y = pd.DataFrame({"duration": duration, "event": event})

    return {"X": X, "y": y, "n_features": n_features}


@pytest.fixture
def deepsurv_params(sample_data):
    """Provides sample parameters for DeepSurv model."""
    return {
        "num_features": sample_data["n_features"],
        "hidden_nodes": [32, 16],
        "batch_norm": True,
        "dropout": 0.1,
        "epochs": 5,  # Keep epochs low for quick testing
        "batch_size": 32,
        "learning_rate": 1e-3,
    }


@pytest.fixture
def nmtlr_params(sample_data):
    """Provides sample parameters for NMTLR model."""
    return {
        "num_features": sample_data["n_features"],
        "hidden_nodes": [32, 16],
        "num_durations": 10,  # Example: 10 discrete duration intervals
        "batch_norm": True,
        "dropout": 0.1,
        "epochs": 5,  # Keep epochs low for quick testing
        "batch_size": 32,
        "learning_rate": 1e-3,
    }


def test_train_deepsurv_model(sample_data, deepsurv_params):
    """
    Tests that train_deepsurv_model trains a DeepSurv model and returns the correct type.
    """
    model = train_deepsurv_model(sample_data, deepsurv_params)

    assert isinstance(model, DeepSurv)
    # Further assertions can be added here, e.g., checking if model has been fitted
    # For now, just checking type and that it runs without error is sufficient for a unit test.


def test_train_nmtlr_model(sample_data, nmtlr_params):
    """
    Tests that train_nmtlr_model trains an NMTLR model and returns the correct type.
    """
    model = train_nmtlr_model(sample_data, nmtlr_params)

    assert isinstance(model, NMTLR)
    # Further assertions can be added here.
