import torch
import torch.nn as nn
import torch.optim as optim
import logging
import gc
from typing import Any, Dict
import numpy as np
import pandas as pd

# Check if pycox is available
try:
    import torchtuples as tt
    from pycox.models import CoxPH, MTLR
except ImportError:
    tt = None
    CoxPH = None
    MTLR = None

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


class CNNBiLSTMWithAttention(nn.Module):
    """
    CNN-BiLSTM with Multi-Head Self-Attention for sequential consumption patterns.
    As per 2025 M&E standards in overview.md.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=n_heads, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Predict CLTV residual or propensity

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len] for CNN
        x = torch.relu(self.cnn(x))
        x = x.transpose(1, 2)  # [batch, seq_len, features]

        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling over time
        pooled = torch.mean(attn_out, dim=1)
        return self.fc(pooled)


def train_deepsurv_model(data: pd.DataFrame, params: Dict[str, Any]) -> Any:
    """
    Trains a DeepSurv model for survival-based monetisation.
    """
    logger.info("Training DeepSurv model using PyCox...")

    clear_device_cache()
    if CoxPH is None:
        logger.warning("pycox not installed. Returning None.")
        return None

    # Preprocessing for DeepSurv
    get_target = lambda df: (df["duration"].values, df["event"].values)
    durations, events = get_target(data)

    # Feature extraction
    x = (
        data.drop(columns=["duration", "event"])
        .select_dtypes(include=[np.number])
        .values.astype("float32")
    )
    y = get_target(data)

    # Network
    in_features = x.shape[1]
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(
        in_features,
        num_nodes,
        out_features,
        batch_norm,
        dropout,
        output_bias=output_bias,
    )

    model = CoxPH(net, tt.optim.Adam)

    batch_size = 256
    epochs = params.get("epochs", 10)
    lr_finder = model.lr_finder(x, y, batch_size, tolerance=10)
    model.optimizer.set_lr(lr_finder.get_best_lr())

    model.fit(x, y, batch_size, epochs=epochs, verbose=False)
    return model


def train_nmtlr_model(data: pd.DataFrame, params: Dict[str, Any]) -> Any:
    """
    Trains an N-MTLR model for multi-task survival.
    """
    logger.info("Training N-MTLR model...")
    clear_device_cache()
    if MTLR is None:
        return None
    # Scaffold implementation - similar to DeepSurv but with MTLR loss
    return None


def train_sequential_cltv_model(
    sequences: np.ndarray, labels: np.ndarray, params: Dict[str, Any]
) -> Any:
    """
    Trains the CNN-BiLSTM Attention model on engagement sequences.
    """
    logger.info("Training Sequential Attention Model...")
    clear_device_cache()

    if len(sequences) == 0:
        logger.warning("No sequences provided for training.")
        return None

    input_dim = sequences.shape[2]
    hidden_dim = (
        params.get("modeling", {}).get("deep_learning", {}).get("hidden_dim", 32)
    )
    n_heads = 4

    model = CNNBiLSTMWithAttention(input_dim, hidden_dim, n_heads)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Simple training loop
    X_train = torch.FloatTensor(sequences)
    y_train = torch.FloatTensor(labels).unsqueeze(1)

    epochs = params.get("modeling", {}).get("deep_learning", {}).get("epochs", 5)
    batch_size = 32

    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(loader):.4f}")

    return model
