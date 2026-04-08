import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def empty_torch_cache():
    """Empty the PyTorch cache for available hardware accelerators (CUDA, XPU, MPS)."""
    
    # 1. CUDA (NVIDIA)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ CUDA cache emptied successfully!")
        print(f"  Devices: {torch.cuda.device_count()} NVIDIA GPU(s) found.")
    
    # 2. XPU (Intel)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
        print("✓ XPU cache emptied successfully!")
        print(f"  Devices: {torch.xpu.device_count()} Intel GPU(s) found.")
    
    # 3. MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS doesn't have an explicit empty_cache like CUDA/XPU in all versions, 
        # but calling this is the standard way to clear the pool if supported.
        try:
            torch.mps.empty_cache()
            print("✓ MPS cache emptied successfully!")
        except AttributeError:
            # Older versions of torch might not have torch.mps.empty_cache()
            print("⚠ MPS detected, but torch.mps.empty_cache() is not available in this version.")

    if not any([
        torch.cuda.is_available(), 
        hasattr(torch, "xpu") and torch.xpu.is_available(),
        hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ]):
        print("⚠ No hardware accelerator (CUDA, XPU, MPS) available. No GPU cache to empty.")

if __name__ == "__main__":
    empty_torch_cache()
