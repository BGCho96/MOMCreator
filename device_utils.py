from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessingDevice:
    device: str
    compute_type: str
    display_name: str
    gpu_available: bool


def detect_nvidia_gpu() -> tuple[bool, str]:
    """
    Detect an NVIDIA GPU through nvidia-smi.

    This checks whether Windows can see an NVIDIA GPU/driver.
    Actual CUDA inference compatibility is finally verified when a model loads.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False, ""

    if result.returncode != 0:
        return False, ""

    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names:
        return False, ""

    return True, names[0]


def resolve_processing_device(use_gpu_if_available: bool) -> ProcessingDevice:
    gpu_available, gpu_name = detect_nvidia_gpu()

    if use_gpu_if_available and gpu_available:
        return ProcessingDevice(
            device="cuda",
            compute_type="float16",
            display_name=gpu_name or "NVIDIA GPU",
            gpu_available=True,
        )

    return ProcessingDevice(
        device="cpu",
        compute_type="int8",
        display_name="CPU",
        gpu_available=gpu_available,
    )
