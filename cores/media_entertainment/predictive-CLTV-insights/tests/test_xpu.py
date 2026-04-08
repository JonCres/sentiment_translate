import pytest
import torch


def test_xpu_available():
    """Verify that XPU is available and PyTorch version is correct."""
    print(f"PyTorch version: {torch.__version__}")

    # Check if XPU is available
    is_available = torch.xpu.is_available()
    print(f"XPU available: {is_available}")

    if is_available:
        print(f"Device count: {torch.xpu.device_count()}")
        print(f"Device name: {torch.xpu.get_device_name(0)}")

    assert is_available, "XPU device not found or not available"
    assert "2.6" in torch.__version__ or "2.7" in torch.__version__, (
        f"Expected PyTorch >= 2.6, got {torch.__version__}"
    )
