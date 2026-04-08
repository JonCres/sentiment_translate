import pytest
import torch
from utils.device import get_device


def test_get_device_torch():
    """Test device selection for torch."""
    device = get_device(purpose="test", framework="torch")
    assert isinstance(device, str)
    if torch.cuda.is_available():
        assert device == "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        assert device == "xpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        assert device == "mps"
    else:
        assert device == "cpu"


def test_get_device_xgboost():
    """Test device selection for xgboost."""
    device = get_device(purpose="test", framework="xgboost")
    assert isinstance(device, str)
    # XGBoost device mapping might be different, but get_device handles it.
    assert device in ["cpu", "sycl", "cuda"]
