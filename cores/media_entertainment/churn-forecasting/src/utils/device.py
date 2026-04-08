import logging
import torch
import gc


def get_device(purpose: str = "model inference", framework: str = "torch") -> str:
    """
    Determines the best available hardware accelerator.

    Args:
        purpose: A description of what the device will be used for.
        framework: The ML framework ('torch' or 'xgboost').

    Returns:
        str: The device string.
    """
    logger = logging.getLogger(__name__)

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
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to clear device cache: {e}")
