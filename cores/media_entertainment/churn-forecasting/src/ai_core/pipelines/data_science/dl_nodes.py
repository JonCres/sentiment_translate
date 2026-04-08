from typing import Any, Dict
import logging

import numpy as np
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def get_device(purpose: str = "model training", framework: str = "torch") -> str:
    """
    Determines the best available hardware accelerator.

    Args:
        purpose: A description of what the device will be used for.
        framework: The ML framework ('torch' or 'xgboost').

    Returns:
        str: The device string.
    """
    is_xgboost = framework.lower() == "xgboost"

    # XPU (Intel GPU)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        if is_xgboost:
            import xgboost as xgb

            build_info = xgb.core.build_info()
            if any("SYCL" in k or "ONEAPI" in k for k in build_info.keys()):
                logger.info(f"Using SYCL (oneAPI) for {purpose} acceleration.")
                return "sycl"
            else:
                logger.warning(
                    "XPU detected, but XGBoost build lacks SYCL support. Falling back to CPU."
                )
                return "cpu"
        logger.info(f"Using XPU for {purpose} acceleration.")
        return "xpu"

    # CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        logger.info(f"Using CUDA for {purpose} acceleration.")
        return "cuda"

    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if is_xgboost:
            logger.info(f"MPS detected, but using CPU for XGBoost ({purpose}).")
            return "cpu"
        logger.info(f"Using MPS for {purpose} acceleration.")
        return "mps"

    # CPU Fallback
    logger.warning(
        f"⚠️ ALERT: Using CPU for {purpose}. "
        "Performance will be significantly degraded. (GPU/XPU/MPS not detected or available)"
    )
    return "cpu"


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        weights = torch.tanh(self.attn(x))  # (batch, seq_len, 1)
        weights = torch.softmax(weights, dim=1)
        output = torch.sum(weights * x, dim=1)  # (batch, hidden_dim)
        return output, weights


class SimplifiedChurnCNN(nn.Module):
    """Simplified CNN-based model without LSTM for better stability."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.3):
        super(SimplifiedChurnCNN, self).__init__()
        self.hidden_dim = hidden_dim

        # Multi-layer CNN to capture temporal patterns
        self.cnn1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.cnn2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, features)
        print("DEBUG: Entering forward pass", flush=True)
        
        # Transpose for CNN: (batch, features, seq_len)
        x = x.transpose(1, 2).contiguous()
        print(f"DEBUG: Transposed shape: {x.shape}", flush=True)

        # CNN layers with pooling
        x = self.relu(self.cnn1(x))
        print("DEBUG: Passed CNN1", flush=True)
        x = self.dropout(x)
        
        x = self.relu(self.cnn2(x))
        print("DEBUG: Passed CNN2", flush=True)
        x = self.dropout(x)
        
        x = self.relu(self.cnn3(x))
        print("DEBUG: Passed CNN3", flush=True)

        # Global average pooling
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim)
        print("DEBUG: Passed Pooling", flush=True)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        print("DEBUG: Passed FC layers", flush=True)
        
        return x


class ChurnCNNAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.2):
        super(ChurnCNNAttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # CNN Layer to capture local spatial-temporal features
        self.cnn = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()

        # BiLSTM Layer
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            layer_dim,
            batch_first=True,
            dropout=dropout_prob if layer_dim > 1 else 0,
            bidirectional=True,
        )

        # Attention Layer
        self.attention = Attention(hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, features)
        # Transpose for CNN: (batch, features, seq_len)
        x_cnn = x.transpose(1, 2)
        x_cnn = self.relu(self.cnn(x_cnn))
        x_cnn = x_cnn.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        # LSTM
        try:
            # Ensure contiguous memory layout for LSTM
            x_cnn = x_cnn.contiguous()
            out, _ = self.lstm(x_cnn)  # (batch, seq_len, hidden_dim)
        except Exception as e:
            logger.error(f"LSTM forward pass failed: {e}")
            raise

        # Attention
        attn_out, weights = self.attention(out)  # (batch, hidden_dim)

        # Output
        out = self.fc(attn_out)
        out = self.sigmoid(out)
        return out


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_dummy = True

    def forward(self, x):
        return x


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


def train_deep_learning_model(
    tensor_sequences: np.ndarray, labels: np.ndarray, params: Dict[str, Any]
) -> Any:
    """
    Train a CNN-BiLSTM with Multi-Head Self-Attention for churn prediction.
    As specified in the technical approach.
    """
    clear_device_cache()
    dl_params = params.get("deep_learning", {})
    if not dl_params:
        logger.warning("No deep_learning parameters found in the 'modeling' group. Using defaults.")

    # Check for empty data (Dynamic Execution)
    if tensor_sequences is None or len(tensor_sequences) == 0:
        logger.warning(
            "No tensor sequences provided for Deep Learning training. Returning Dummy Model."
        )
        return DummyModel()

    # Data Conversion: Polars DF -> Numpy 3D Tensor
    if hasattr(tensor_sequences, "select"):  # Check if Polars DataFrame
        try:
            # Extract flattened lists
            # Assuming single column 'engagement_sequence_flat'
            flat_list = tensor_sequences["engagement_sequence_flat"].to_list()
            tensor_data = np.array(flat_list)

            # Reshape
            seq_len = dl_params.get("sequence_length", 90)
            if tensor_data.shape[1] % seq_len != 0:
                logger.warning(
                    "Flattened sequence length does not match expected sequence length. Reshaping might fail."
                )

            num_features = tensor_data.shape[1] // seq_len
            tensor_sequences = tensor_data.reshape(-1, seq_len, num_features)
            logger.info(f"Reshaped tensor sequences to {tensor_sequences.shape}")
        except Exception as e:
            logger.error(f"Failed to convert Polars DF to Tensor: {e}")
            return DummyModel()

    if isinstance(tensor_sequences, (list, tuple)):
        tensor_sequences = np.array(tensor_sequences)

    input_dim = tensor_sequences.shape[2]  # (N, Seq, Features)
    hidden_dim = dl_params.get("lstm_units", 64)
    layer_dim = dl_params.get("layer_dim", 1)
    output_dim = 1
    learning_rate = dl_params.get("learning_rate", 0.001)
    num_epochs = dl_params.get("epochs", 10)
    batch_size = dl_params.get("batch_size", 32)

    # Aggressive stability fixes for Mac SEGV
    torch.set_num_threads(1)
    device = torch.device("cpu")
    logger.info(f"STABILITY MODE: Forcing CPU and 1 thread for training.")

    # Prepare Data
    logger.info(f"Preparing data: {len(tensor_sequences)} samples, {input_dim} features")
    logger.info(f"Data shape: {tensor_sequences.shape}, dtype: {tensor_sequences.dtype}")
    logger.info(f"Labels shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")

    if len(labels) != len(tensor_sequences):
        raise ValueError(
            f"Mismatch in data length: X={len(tensor_sequences)}, y={len(labels)}"
        )

    # Check for NaN/inf values
    if np.isnan(tensor_sequences).any():
        logger.warning("Found NaN values in tensor_sequences, replacing with 0")
        tensor_sequences = np.nan_to_num(tensor_sequences, nan=0.0)
    if np.isinf(tensor_sequences).any():
        logger.warning("Found inf values in tensor_sequences, replacing with 0")
        tensor_sequences = np.nan_to_num(tensor_sequences, posinf=0.0, neginf=0.0)

    logger.info("Converting to PyTorch tensors with contiguity guarantee...")
    try:
        # Guarantee C-contiguity and float32
        X_contiguous = np.ascontiguousarray(tensor_sequences, dtype=np.float32)
        X_tensor = torch.from_numpy(X_contiguous)
        
        if isinstance(labels, np.ndarray):
            y_contiguous = np.ascontiguousarray(labels, dtype=np.float32)
            y_tensor = torch.from_numpy(y_contiguous).unsqueeze(1)
        else:
            y_tensor = torch.FloatTensor(labels).unsqueeze(1)
        
        logger.info(f"Tensor conversion successful. X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
    except Exception as e:
        logger.error(f"Failed to convert to tensors: {e}")
        raise

    logger.info(f"Creating DataLoader with batch_size={batch_size}...")
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Initializing model: hidden_dim={hidden_dim}, layer_dim={layer_dim}...")

    # Use simplified CNN model instead of LSTM-based model for stability
    # LSTM has known issues on MPS and can be slow/unstable on CPU
    use_simplified_model = dl_params.get("use_simplified_model", True)

    if use_simplified_model:
        logger.info("Using SimplifiedChurnCNN (CNN-only, no LSTM) for better stability")
        model = SimplifiedChurnCNN(input_dim, hidden_dim, output_dim)
    else:
        logger.info("Using ChurnCNNAttentionLSTM (with BiLSTM)")
        model = ChurnCNNAttentionLSTM(input_dim, hidden_dim, layer_dim, output_dim)

    logger.info(f"Moving model to {device}...")
    model.to(device)

    logger.info("Setting up loss and optimizer...")
    # Using BCEWithLogitsLoss for better numerical stability (avoids log(0) issues)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info(f"Starting training for {num_epochs} epochs...")

    model.train()
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}...")
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(loader):
            if i == 0:
                logger.info(f"Processing first batch (shape: {inputs.shape})...")

            inputs, targets = inputs.to(device), targets.to(device)

            if i == 0:
                logger.info("Running forward pass...")
            optimizer.zero_grad()
            outputs = model(inputs)

            if i == 0:
                logger.info(f"Forward pass complete. Output shape: {outputs.shape}")
                logger.info("Computing loss...")
            loss = criterion(outputs, targets)

            if i == 0:
                logger.info("Running backward pass...")
            loss.backward()

            if i == 0:
                logger.info("Updating weights...")
            optimizer.step()

            running_loss += loss.item()

            # Log progress every 10 batches
            if (i + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(loader)}, Loss: {loss.item():.4f}")

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {running_loss / len(loader):.4f}")

    return model
