import os
import time
import psutil


def get_memory_snapshot() -> dict:
    """Take a snapshot of current memory usage."""
    process = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()

    snapshot = {
        "process_ram_mb": round(
            process.memory_info().rss / (1024 * 1024), 2
        ),
        "system_ram_used_mb": round(vm.used / (1024 * 1024), 2),
        "system_ram_available_mb": round(vm.available / (1024 * 1024), 2),
        "system_ram_total_mb": round(vm.total / (1024 * 1024), 2),
        "system_ram_percent": vm.percent,
        "gpu_available": False,
        "gpu_ram_used_mb": None,
        "gpu_ram_total_mb": None,
        "gpu_name": "No GPU",
    }

    try:
        import torch
        if torch.cuda.is_available():
            snapshot["gpu_available"] = True
            snapshot["gpu_ram_used_mb"] = round(
                torch.cuda.memory_allocated() / (1024 * 1024), 2
            )
            snapshot["gpu_ram_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2
            )
            snapshot["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            snapshot["gpu_name"] = "CUDA not available — CPU only"
    except ImportError:
        snapshot["gpu_name"] = "torch not installed"

    return snapshot


def measure_memory_delta(before: dict, after: dict) -> dict:
    """Calculate memory difference between two snapshots."""
    return {
        "ram_delta_mb": round(
            after["process_ram_mb"] - before["process_ram_mb"], 2
        ),
        "system_ram_delta_mb": round(
            after["system_ram_used_mb"] - before["system_ram_used_mb"], 2
        ),
        "peak_process_ram_mb": after["process_ram_mb"],
        "gpu_available": after["gpu_available"],
        "gpu_ram_used_mb": after["gpu_ram_used_mb"],
        "gpu_name": after["gpu_name"],
    }