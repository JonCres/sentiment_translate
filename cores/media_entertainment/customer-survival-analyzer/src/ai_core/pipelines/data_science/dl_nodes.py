import torch
import gc
import numpy as np
import torchtuples as tt
from pycox.models import DeepSurv, NMTLR
from pycox.models.loss import NMTLRLoss
from pycox import models as pm
from typing import Any, Dict
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def clear_device_cache():
    """Release unoccupied cached memory for available devices (CUDA, XPU, MPS)."""
    # Force Python garbage collection first
    gc.collect()

    try:
        # CUDA (NVIDIA)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # XPU (Intel)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        # Log warning but don't crash processing
        logger.warning(f"Failed to clear device cache: {e}")


def train_deepsurv_model(train_data: pd.DataFrame, params: Dict[str, Any]) -> DeepSurv:
    """
    Trains a DeepSurv model using pycox.

    Args:
        train_data: DataFrame containing features and target columns 'duration', 'event'.
        params: Model hyperparameters.

    Returns:
        A trained pycox DeepSurv model.
    """
    logger.info("Training DeepSurv model...")
    clear_device_cache()

    # Split X and y
    # Assuming 'duration' and 'event' are the target columns
    # And all other numeric columns are features
    # Filter numeric features
    X_df = train_data.drop(columns=["duration", "event"]).select_dtypes(
        include=[np.number]
    )
    X = X_df.values.astype("float32")

    durations = train_data["duration"].values.astype("float32")
    events = train_data["event"].values.astype("int32")

    # Define the neural network
    net = tt.practical.MLPVanilla(
        in_features=params["num_features"],
        num_nodes=params["hidden_nodes"],
        out_features=1,
        batch_norm=params.get("batch_norm", True),
        dropout=params.get("dropout", 0.1),
    )

    # Initialize DeepSurv model
    model = DeepSurv(net, tt.optim.Adam)

    # Train the model
    log = model.fit(
        x=X,
        y=(durations, events),
        batch_size=params.get("batch_size", 256),
        epochs=params.get("epochs", 100),
        callbacks=[tt.callbacks.EarlyStopping()],
        verbose=False,
    )
    logger.info(f"DeepSurv model training finished with log: {log}.")

    return model


def train_nmtlr_model(data: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> NMTLR:
    """
    Trains a Neural Multi-Task Logistic Regression (N-MTLR) model using pycox.

    Args:
        data: A dictionary containing 'X' (features DataFrame) and 'y' (target DataFrame with 'event' and 'duration' columns).
        params: A dictionary of parameters for the N-MTLR model, including:
            - 'num_features': Number of input features.
            - 'hidden_nodes': List of integers for hidden layer sizes.
            - 'num_durations': Number of discrete duration intervals.
            - 'batch_norm': Boolean indicating whether to use batch normalization.
            - 'dropout': Float for dropout rate.
            - 'epochs': Number of training epochs.
            - 'batch_size': Training batch size.
            - 'learning_rate': Optimizer learning rate.

    Returns:
        A trained pycox NMTLR model.
    """
    logger.info("Training N-MTLR model...")
    clear_device_cache()

    X = data["X"].values.astype("float32")
    durations = data["y"]["duration"].values.astype("float32")
    events = data["y"]["event"].values.astype("int32")

    # Discretize durations for NMTLR
    # The `make_cuts` function can be used to determine `num_durations` and `labtrans`
    # For simplicity, we'll assume `num_durations` is provided in params
    # and `labtrans` is handled internally by NMTLR or pre-processed.
    # In a real scenario, this would involve a `label_transform` step.

    # Ensure `num_durations` is present in params
    if "num_durations" not in params:
        raise ValueError(
            "Parameter 'num_durations' is required for N-MTLR model training."
        )

    # Define the neural network
    net = tt.practical.MLPVanilla(
        in_features=params["num_features"],
        num_nodes=params["hidden_nodes"],
        out_features=params["num_durations"],  # NMTLR output is num_durations
        batch_norm=params.get("batch_norm", True),
        dropout=params.get("dropout", 0.1),
    )

    # Initialize NMTLR model
    # NMTLR uses NMTLRLoss
    model = NMTLR(net, tt.optim.Adam, loss=NMTLRLoss())

    # Train the model
    log = model.fit(
        x=X,
        y=(durations, events),
        batch_size=params.get("batch_size", 256),
        epochs=params.get("epochs", 100),
        callbacks=[tt.callbacks.EarlyStopping()],
        verbose=False,
    )
    logger.info(f"N-MTLR model training finished with log: {log}.")

    return model
