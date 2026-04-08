import logging
import torch
import gc
import os
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    CUDA = auto()
    XPU = auto()
    MPS = auto()
    CPU = auto()


class MultiDeviceManager:
    """Universal GPU architecture manager supporting CUDA, XPU, MPS, and CPU.

    Refactored to support native PyTorch XPU while maintaining backward compatibility
    with optional IPEX optimizations.
    """

    def __init__(self):
        self.device_type = self._detect_device()
        self.device_name = self._get_device_name()
        self.ipex = None

        if self.device_type == DeviceType.XPU:
            try:
                import intel_extension_for_pytorch as ipex

                self.ipex = ipex
            except ImportError:
                # In native XPU support (PyTorch 2.4+), IPEX is optional for basic XPU usage.
                logger.info("XPU detected. IPEX not found, using native torch.xpu.")

    def _detect_device(self) -> DeviceType:
        """Detect available compute device in priority order: CUDA > XPU > MPS > CPU."""
        if torch.cuda.is_available():
            return DeviceType.CUDA
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return DeviceType.XPU
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceType.MPS
        return DeviceType.CPU

    def _get_device_name(self) -> str:
        """Get string identifier for the detected device."""
        device_map = {
            DeviceType.CUDA: "cuda",
            DeviceType.XPU: "xpu",
            DeviceType.MPS: "mps",
            DeviceType.CPU: "cpu",
        }
        return device_map[self.device_type]

    def clear_cache(self):
        """Clear device cache before model loading."""
        # Force Python garbage collection first
        gc.collect()

        if self.device_type == DeviceType.CUDA:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif self.device_type == DeviceType.XPU:
            torch.xpu.empty_cache()
        elif self.device_type == DeviceType.MPS:
            torch.mps.empty_cache()

        if self.device_type != DeviceType.CPU:
            logger.info(f"✓ {self.device_type.name} cache cleared")

    def optimize_model(self, model, dtype=torch.float16):
        """Apply architecture-specific optimizations."""
        model = model.to(self.device_name)

        if self.device_type == DeviceType.XPU and self.ipex:
            try:
                model = self.ipex.optimize(model, dtype=dtype)
                logger.info(f"✓ IPEX optimization applied (dtype={dtype})")
            except Exception as e:
                logger.warning(f"Failed to apply IPEX optimization: {e}")
        elif self.device_type == DeviceType.CUDA:
            if dtype == torch.float16 and torch.cuda.is_bf16_supported():
                # Optional: Upgrade to bfloat16 if supported
                pass
            logger.info(f"✓ CUDA optimized (dtype={dtype})")
        elif self.device_type == DeviceType.MPS:
            logger.info(f"✓ MPS configured (dtype={dtype})")
        else:
            logger.info(f"✓ Model on CPU (dtype={dtype})")

        return model

    def enable_memory_efficiency(self, model, training: bool = False):
        """Enable memory-efficient attention and gradient checkpointing."""
        # Disable KV cache for inference memory savings
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
            logger.info("✓ Memory-efficient attention enabled (use_cache=False)")

        # Gradient checkpointing for training
        if training and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("✓ Gradient checkpointing enabled")

        # Device-specific memory optimizations
        if self.device_type == DeviceType.CUDA:
            torch.backends.cudnn.benchmark = True
        elif self.device_type == DeviceType.MPS:
            if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
                logger.info(
                    "ℹ️  Tip: Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable MPS memory limits"
                )

    def get_memory_stats(self) -> dict:
        """Get memory statistics for the current device."""
        stats = {"device": self.device_type.name, "allocated_gb": 0, "reserved_gb": 0}

        if self.device_type == DeviceType.CUDA:
            stats["allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        elif self.device_type == DeviceType.XPU:
            stats["allocated_gb"] = torch.xpu.memory_allocated() / 1024**3
            stats["reserved_gb"] = torch.xpu.memory_reserved() / 1024**3
        elif self.device_type == DeviceType.MPS:
            stats["allocated_gb"] = torch.mps.current_allocated_memory() / 1024**3
            if hasattr(torch.mps, "driver_allocated_memory"):
                stats["reserved_gb"] = torch.mps.driver_allocated_memory() / 1024**3

        return stats

    def print_memory_stats(self):
        """Print formatted memory statistics."""
        stats = self.get_memory_stats()

        logger.info(f"\n📊 Memory Stats ({stats['device']}):")
        logger.info(f"   Allocated: {stats['allocated_gb']:.2f} GB")

        if stats["reserved_gb"] > 0:
            logger.info(f"   Reserved:  {stats['reserved_gb']:.2f} GB")

        if "max_allocated_gb" in stats:
            logger.info(f"   Peak:      {stats['max_allocated_gb']:.2f} GB")

        if self.device_type == DeviceType.XPU and stats["allocated_gb"] > 30:
            logger.warning("   ⚠️  Approaching Intel Arc A770 32GB limit")
        elif self.device_type == DeviceType.CUDA:
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                utilization = (stats["allocated_gb"] / total_mem) * 100
                logger.info(
                    f"   Total VRAM: {total_mem:.2f} GB ({utilization:.1f}% used)"
                )
            except Exception:
                pass


def load_model(
    model_name: str = "microsoft/phi-2",
    torch_dtype: torch.dtype = torch.float16,
    training: bool = False,
    device_manager: Optional[MultiDeviceManager] = None,
) -> Tuple[Any, Any, MultiDeviceManager]:
    """
    Load and optimize model for the detected architecture.
    """
    manager = device_manager or MultiDeviceManager()

    logger.info(f"\n🚀 Loading {model_name} on {manager.device_type.name}...")

    manager.clear_cache()

    effective_dtype = torch_dtype
    if manager.device_type == DeviceType.CPU and torch_dtype == torch.float16:
        effective_dtype = torch.float32
        logger.info("ℹ️  Switched to float32 for CPU compatibility")
    elif manager.device_type == DeviceType.MPS and torch_dtype == torch.float16:
        if not torch.backends.mps.is_available():
            effective_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=effective_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = manager.optimize_model(model, dtype=effective_dtype)
    manager.enable_memory_efficiency(model, training=training)
    manager.print_memory_stats()

    return model, tokenizer, manager


def generate_text(
    model: Any,
    tokenizer: Any,
    device_manager: MultiDeviceManager,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> str:
    """Generate text with architecture-optimized settings."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device_manager.device_name) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if device_manager.device_type == DeviceType.CUDA:
        gen_kwargs["use_cache"] = True

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Backward compatibility wrappers and utility functions
def get_device(purpose: str = "model inference", framework: str = "torch") -> str:
    """Legacy wrapper for backward compatibility."""
    manager = MultiDeviceManager()

    if framework.lower() == "xgboost":
        if manager.device_type == DeviceType.XPU:
            import xgboost as xgb

            build_info = xgb.core.build_info()
            if any("SYCL" in k or "ONEAPI" in k for k in build_info.keys()):
                logger.info(f"Using SYCL (oneAPI) for {purpose} acceleration.")
                return "sycl"
            else:
                logger.warning(
                    "XPU detected but XGBoost lacks SYCL. Falling back to CPU."
                )
                return "cpu"
        elif manager.device_type == DeviceType.MPS:
            return "cpu"  # XGBoost doesn't support MPS

    return manager.device_name


def clear_device_cache():
    """Legacy wrapper for backward compatibility."""
    manager = MultiDeviceManager()
    manager.clear_cache()
