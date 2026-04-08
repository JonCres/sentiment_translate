import pytest
import torch


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_xpu_available():
    """Test if XPU is available and accessible via torch.xpu."""
    assert torch.xpu.is_available()
    assert torch.xpu.device_count() > 0
    device = torch.device("xpu")
    tensor = torch.tensor([1.0, 2.0], device=device)
    assert tensor.device.type == "xpu"
    print(f"XPU Device Name: {torch.xpu.get_device_name(0)}")
