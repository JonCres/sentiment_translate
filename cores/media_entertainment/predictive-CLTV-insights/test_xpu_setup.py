#!/usr/bin/env python3
"""Test script to verify Intel XPU (Arc GPU) setup on Windows and Linux.

This script checks:
1. PyTorch installation and version
2. XPU device availability
3. XPU device properties
4. Basic tensor operations on XPU

Requirements:
- Linux: Intel compute runtime installed (Level Zero drivers)
- Windows: Intel Arc GPU drivers installed
- PyTorch 2.6+ with XPU support (via `uv sync --extra xpu`)

Author: Wizeline AI Cores Team
"""

import sys
from typing import Dict, Any


def check_pytorch_installation() -> Dict[str, Any]:
    """Check PyTorch installation and XPU support."""
    results = {
        "pytorch_installed": False,
        "pytorch_version": None,
        "has_xpu_module": False,
        "xpu_available": False,
        "xpu_device_count": 0,
        "xpu_device_name": None,
        "platform": sys.platform,
    }

    try:
        import torch

        results["pytorch_installed"] = True
        results["pytorch_version"] = torch.__version__

        # Check if torch has XPU module (PyTorch 2.4+)
        results["has_xpu_module"] = hasattr(torch, "xpu")

        if results["has_xpu_module"]:
            # Check if XPU is available
            results["xpu_available"] = torch.xpu.is_available()

            if results["xpu_available"]:
                results["xpu_device_count"] = torch.xpu.device_count()
                if results["xpu_device_count"] > 0:
                    # Get device properties
                    results["xpu_device_name"] = torch.xpu.get_device_name(0)

    except ImportError as e:
        results["error"] = f"PyTorch not installed: {e}"
    except Exception as e:
        results["error"] = f"Unexpected error: {e}"

    return results


def test_xpu_operations() -> Dict[str, Any]:
    """Test basic tensor operations on XPU."""
    test_results = {
        "tensor_creation": False,
        "tensor_operations": False,
        "device_transfer": False,
    }

    try:
        import torch

        if not torch.xpu.is_available():
            test_results["error"] = "XPU not available for testing"
            return test_results

        # Test 1: Create tensor on XPU
        x = torch.randn(100, 100, device="xpu")
        test_results["tensor_creation"] = True

        # Test 2: Basic operations
        y = torch.randn(100, 100, device="xpu")
        z = torch.matmul(x, y)
        test_results["tensor_operations"] = True

        # Test 3: Device transfer
        z_cpu = z.cpu()
        test_results["device_transfer"] = True

    except Exception as e:
        test_results["error"] = str(e)

    return test_results


def print_results(info: Dict[str, Any], tests: Dict[str, Any]) -> None:
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("Intel XPU (Arc GPU) Configuration Test")
    print("=" * 70)

    # System Info
    print(f"\n📋 System Information:")
    print(f"   Platform: {info['platform']}")
    print(f"   PyTorch Installed: {info['pytorch_installed']}")
    if info["pytorch_installed"]:
        print(f"   PyTorch Version: {info['pytorch_version']}")

    # XPU Status
    print(f"\n🔍 XPU Detection:")
    print(f"   Has torch.xpu module: {info['has_xpu_module']}")
    print(f"   XPU Available: {info['xpu_available']}")

    if info["xpu_available"]:
        print(f"\n✅ XPU Device Found!")
        print(f"   Device Count: {info['xpu_device_count']}")
        print(f"   Device Name: {info['xpu_device_name']}")

        # Test Results
        print(f"\n🧪 XPU Operations Tests:")
        print(f"   Tensor Creation: {'✅' if tests['tensor_creation'] else '❌'}")
        print(
            f"   Tensor Operations: {'✅' if tests['tensor_operations'] else '❌'}"
        )
        print(f"   Device Transfer: {'✅' if tests['device_transfer'] else '❌'}")

        if all(tests.values()):
            print(f"\n🎉 All tests passed! XPU is fully functional.")
        else:
            print(f"\n⚠️  Some tests failed. Check error messages above.")
            if "error" in tests:
                print(f"   Error: {tests['error']}")
    else:
        print(f"\n❌ XPU Not Available")

        # Diagnostics
        print(f"\n🔧 Troubleshooting:")
        if info["platform"] == "linux":
            print(f"   Linux XPU Setup Requirements:")
            print(f"   1. Install Intel compute runtime:")
            print(f"      sudo apt install intel-level-zero-gpu intel-opencl-icd")
            print(f"   2. Install PyTorch XPU:")
            print(f"      uv sync --extra xpu")
            print(f"   3. Verify drivers:")
            print(f"      ls /dev/dri/render* (should show renderD128 or similar)")
        elif info["platform"] == "win32":
            print(f"   Windows XPU Setup Requirements:")
            print(
                f"   1. Install Intel Arc GPU drivers from: https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html"
            )
            print(f"   2. Install PyTorch XPU:")
            print(f"      uv sync --extra xpu")
            print(f"   3. Restart your system after driver installation")
        else:
            print(f"   XPU is only supported on Windows and Linux")

        if "error" in info:
            print(f"\n   Error Details: {info['error']}")

    print("\n" + "=" * 70 + "\n")


def main() -> int:
    """Main test function."""
    info = check_pytorch_installation()

    if not info["pytorch_installed"]:
        print("❌ PyTorch is not installed!")
        print("\nInstall with: uv sync --extra xpu")
        return 1

    tests = {}
    if info["xpu_available"]:
        tests = test_xpu_operations()

    print_results(info, tests)

    # Return exit code
    if info["xpu_available"] and all(tests.values()):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
