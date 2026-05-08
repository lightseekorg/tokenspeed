# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

__all__ = [
    "ArchVersion",
    "InterconnectInfo",
    "PlatformInfo",
    "CapabilityRequirement",
    "Platform",
    "current_platform",
]


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchVersion:
    """Hardware generation identifier. Supports comparison operators."""

    major: int
    minor: int

    def __ge__(self, other: ArchVersion) -> bool:
        return (self.major, self.minor) >= (other.major, other.minor)

    def __gt__(self, other: ArchVersion) -> bool:
        return (self.major, self.minor) > (other.major, other.minor)

    def __le__(self, other: ArchVersion) -> bool:
        return (self.major, self.minor) <= (other.major, other.minor)

    def __lt__(self, other: ArchVersion) -> bool:
        return (self.major, self.minor) < (other.major, other.minor)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True)
class InterconnectInfo:
    """Multi-GPU interconnect topology."""

    topology: str  # "single_gpu", "pcie", "nvlink_pairs", "nvlink_full", "nvswitch"
    bandwidth_matrix: tuple[tuple[float, ...], ...] | None = None
    nvlink_version: int | None = None
    nvswitch_present: bool = False


@dataclass(frozen=True)
class Fp8E4M3FnDType:
    dtype: torch.dtype
    max: float
    min: float


_fp8e4m3_dtype = None


@dataclass(frozen=True)
class PlatformInfo:
    """Complete description of a compute platform."""

    vendor: str  # "nvidia", "amd"
    arch_version: ArchVersion
    device_name: str
    device_count: int

    # Memory
    total_memory: int  # Bytes per device
    memory_bandwidth: float  # GB/s

    # Compute
    sm_count: int  # Streaming multiprocessors (or CUs)
    max_threads_per_sm: int
    max_shared_memory_per_sm: int  # Bytes

    # Features (string-based for extensibility)
    sm_features: frozenset[str] = frozenset()  # Determined by compute capability
    runtime_features: frozenset[str] = frozenset()  # Detected at runtime

    # Interconnect (for multi-GPU)
    interconnect: InterconnectInfo | None = None

    # NUMA-local CPU IDs per logical device. Empty means unavailable.
    numa_cpu_affinity: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def detect(cls) -> PlatformInfo:
        """Detect platform from current environment."""
        return _detect_platform()

    # Convenience properties
    @property
    def is_nvidia(self) -> bool:
        return self.vendor == "nvidia"

    @property
    def is_hopper(self) -> bool:
        return self.is_nvidia and self.arch_version.major == 9

    @property
    def is_blackwell(self) -> bool:
        return self.is_nvidia and self.arch_version.major == 10

    @property
    def is_ampere(self) -> bool:
        return self.is_nvidia and self.arch_version.major == 8

    @property
    def is_amd(self) -> bool:
        return self.vendor == "amd"

    @property
    def is_cdna3(self) -> bool:
        return self.is_amd and self.arch_version == ArchVersion(9, 4)

    @property
    def is_cdna4(self) -> bool:
        return self.is_amd and self.arch_version == ArchVersion(9, 5)

    @property
    def is_ampere_plus(self) -> bool:
        return self.is_nvidia and self.arch_version >= ArchVersion(8, 0)

    @property
    def is_hopper_plus(self) -> bool:
        return self.is_nvidia and self.arch_version >= ArchVersion(9, 0)

    @property
    def is_blackwell_plus(self) -> bool:
        return self.is_nvidia and self.arch_version >= ArchVersion(10, 0)

    @property
    def is_cdna3_plus(self) -> bool:
        return self.is_amd and self.arch_version >= ArchVersion(9, 4)

    @property
    def is_cdna4_plus(self) -> bool:
        return self.is_amd and self.arch_version >= ArchVersion(9, 5)

    @property
    def arch(self) -> str:
        """Short architecture string for cache keys."""
        return str(self.arch_version)

    @property
    def is_fp8e4m3fnuz(self) -> bool:
        return self.is_cdna3

    @property
    def fp8e4m3fn(self) -> Fp8E4M3FnDType:
        global _fp8e4m3_dtype
        if _fp8e4m3_dtype is None:
            if self.is_cdna3:
                dtype = torch.float8_e4m3fnuz
                fp8_max = 224.0
            else:
                dtype = torch.float8_e4m3fn
                fp8_max = torch.finfo(dtype).max
            fp8_min = -fp8_max
            _fp8e4m3_dtype = Fp8E4M3FnDType(dtype=dtype, max=fp8_max, min=fp8_min)
        return _fp8e4m3_dtype

    @property
    def generation_name(self) -> str:
        """Human-readable generation name."""
        arch_version = (self.arch_version.major, self.arch_version.minor)
        if self.is_nvidia:
            names = {
                (8, 0): "Ampere",
                (8, 6): "Ampere",
                (8, 9): "Ada Lovelace",
                (9, 0): "Hopper",
                (10, 0): "Blackwell",
            }
            return names.get(arch_version, f"SM{arch_version[0]}.{arch_version[1]}")
        if self.is_amd:
            names = {
                (9, 4): "CDNA3",  # MI300
                (9, 5): "CDNA4",  # MI350
            }
            return names.get(arch_version, f"GFX{arch_version[0]}.{arch_version[1]}")
        return f"{self.vendor}:{arch_version[0]}.{arch_version[1]}"


@dataclass(frozen=True)
class CapabilityRequirement:
    """Requirements a kernel has on platform capabilities."""

    min_arch_version: ArchVersion | None = None
    max_arch_version: ArchVersion | None = None
    required_features: frozenset[str] = frozenset()
    vendors: frozenset[str] | None = None  # None = any vendor

    def satisfied_by(self, platform: PlatformInfo) -> bool:
        """Check if platform satisfies these requirements."""
        if self.vendors and platform.vendor not in self.vendors:
            return False

        if self.min_arch_version:
            if not platform.arch_version >= self.min_arch_version:
                return False

        if self.max_arch_version:
            if platform.arch_version > self.max_arch_version:
                return False

        all_features = platform.sm_features | platform.runtime_features
        if not self.required_features.issubset(all_features):
            return False

        return True

    def missing_features(self, platform: PlatformInfo) -> set[str]:
        """Return features required but not available."""
        all_features = platform.sm_features | platform.runtime_features
        return self.required_features - all_features


# ---------------------------------------------------------------------------
# Platform singleton
# ---------------------------------------------------------------------------


class Platform:
    """Global platform singleton with lazy initialization."""

    _instance: PlatformInfo | None = None

    @classmethod
    def get(cls) -> PlatformInfo:
        """Get current platform info (detected once, cached)."""
        if cls._instance is None:
            cls._instance = PlatformInfo.detect()
        return cls._instance

    @classmethod
    def override(cls, platform: PlatformInfo) -> None:
        """Override platform detection (for testing/debugging)."""
        cls._instance = platform

    @classmethod
    def reset(cls) -> None:
        """Reset cached platform (for testing)."""
        cls._instance = None
        _detect_cuda_nvlink_topology.cache_clear()


def current_platform() -> PlatformInfo:
    """Get current platform."""
    return Platform.get()


# ---------------------------------------------------------------------------
# Detection implementation
# ---------------------------------------------------------------------------


def _torch_version() -> tuple[int, ...]:
    """Return PyTorch version as a comparable tuple, e.g. (2, 7, 0)."""
    try:
        import torch

        return tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
    except Exception:
        return (0, 0, 0)


def _detect_platform() -> PlatformInfo:
    """Detect current platform capabilities."""
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "tokenspeed-kernel requires PyTorch with NVIDIA CUDA or AMD ROCm support."
        ) from None

    if torch.cuda.is_available():
        if hasattr(torch.version, "hip") and torch.version.hip:
            return _detect_rocm_platform()
        return _detect_cuda_platform()

    raise RuntimeError("tokenspeed-kernel requires an NVIDIA CUDA or AMD ROCm GPU.")


def _detect_cuda_platform() -> PlatformInfo:
    """Detect NVIDIA CUDA platform."""
    import torch

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch_version = ArchVersion(props.major, props.minor)
    sm_features = _get_cuda_sm_features(arch_version)
    runtime_features = _get_cuda_runtime_features()
    interconnect = _detect_cuda_interconnect()
    numa_cpu_affinity = _detect_cuda_numa_cpu_affinity()

    return PlatformInfo(
        vendor="nvidia",
        arch_version=arch_version,
        device_name=props.name,
        device_count=torch.cuda.device_count(),
        total_memory=props.total_memory,
        memory_bandwidth=_estimate_bandwidth(props),
        sm_count=props.multi_processor_count,
        max_threads_per_sm=getattr(props, "max_threads_per_multi_processor", 0),
        max_shared_memory_per_sm=getattr(props, "max_shared_memory_per_block", 0),
        sm_features=sm_features,
        runtime_features=runtime_features,
        interconnect=interconnect,
        numa_cpu_affinity=numa_cpu_affinity,
    )


def _get_cuda_sm_features(arch_version: ArchVersion) -> frozenset[str]:
    """Determine CUDA SM features from arch version."""
    features: set[str] = set()

    if arch_version >= ArchVersion(7, 0):
        features |= {"tensor_core:f16"}

    if arch_version >= ArchVersion(8, 0):
        features |= {"tensor_core:int8", "memory:async_copy"}

    if arch_version >= ArchVersion(8, 9):
        features |= {"tensor_core:f8"}

    if arch_version >= ArchVersion(9, 0):
        features |= {"memory:tma", "compute:cluster"}

    if arch_version >= ArchVersion(10, 0):
        features |= {"tensor_core:f4"}

    return frozenset(features)


def _get_cuda_runtime_features() -> frozenset[str]:
    """Detect CUDA runtime features from environment."""
    features: set[str] = {"runtime:cuda_graph"}

    if _check_symmetric_memory_available():
        features.add("comms:symmetric_memory")
    if _check_nvlink_available():
        features.add("comms:nvlink")
    if _detect_cuda_nvlink_topology() == "nvlink_full":
        features.add("comms:nvlink_full")

    return frozenset(features)


def _detect_rocm_platform() -> PlatformInfo:
    """Detect AMD ROCm platform."""
    import torch

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = _extract_amd_arch(props.gcnArchName)

    # Map AMD architectures
    arch_map = {
        "gfx942": ArchVersion(9, 4),  # MI300
        "gfx950": ArchVersion(9, 5),  # MI350
    }
    arch_version = arch_map.get(arch, ArchVersion(9, 0))
    sm_features = _get_rocm_sm_features(arch)
    runtime_features = _get_rocm_runtime_features()

    return PlatformInfo(
        vendor="amd",
        arch_version=arch_version,
        device_name=props.name,
        device_count=torch.cuda.device_count(),
        total_memory=props.total_memory,
        memory_bandwidth=_estimate_amd_bandwidth(props),
        sm_count=props.multi_processor_count,
        max_threads_per_sm=getattr(props, "max_threads_per_multi_processor", 0),
        max_shared_memory_per_sm={"gfx942": 64 * 1024, "gfx950": 64 * 1024}.get(
            arch, getattr(props, "max_shared_memory_per_block", 0)
        ),
        sm_features=sm_features,
        runtime_features=runtime_features,
        interconnect=_detect_rocm_interconnect(),
    )


def _get_rocm_sm_features(arch: str) -> frozenset[str]:
    """Determine ROCm SM features from architecture."""
    features: set[str] = set()

    if arch in ("gfx942", "gfx950"):
        features |= {"tensor_core:f16", "tensor_core:f8"}

    if arch == "gfx950":
        features |= {"tensor_core:f4", "memory:async_copy"}

    return frozenset(features)


def _get_rocm_runtime_features() -> frozenset[str]:
    """Detect ROCm runtime features from environment."""
    features: set[str] = set()

    if _check_symmetric_memory_available():
        features.add("comms:symmetric_memory")

    return frozenset(features)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extract_amd_arch(gcn_arch_name: str) -> str:
    """Extract base architecture from GCN arch name.

    Example: 'gfx942:sramecc+:xnack-' -> 'gfx942'
    """
    return gcn_arch_name.split(":")[0]


def _estimate_bandwidth(props: object) -> float:
    """Estimate memory bandwidth in GB/s from CUDA device properties."""
    clock_rate = getattr(props, "memory_clock_rate", 0)
    bus_width = getattr(props, "memory_bus_width", 0)
    if clock_rate and bus_width:
        return (clock_rate * 1e3 * (bus_width / 8) * 2) / 1e9
    return 0.0


def _estimate_amd_bandwidth(props: object) -> float:
    """Estimate memory bandwidth for AMD devices."""
    clock_rate = getattr(props, "memory_clock_rate", 0)
    bus_width = getattr(props, "memory_bus_width", 0)
    if clock_rate and bus_width:
        return (clock_rate * 1e3 * (bus_width / 8) * 2) / 1e9
    return 0.0


def _detect_cuda_interconnect() -> InterconnectInfo | None:
    """Detect CUDA multi-GPU interconnect topology."""
    try:
        import torch

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return InterconnectInfo(topology="single_gpu")

        nvlink_topology = _detect_cuda_nvlink_topology()
        if nvlink_topology:
            return InterconnectInfo(topology=nvlink_topology)
        return InterconnectInfo(topology="pcie")
    except Exception:
        return None


def _detect_rocm_interconnect() -> InterconnectInfo | None:
    """Detect ROCm multi-GPU interconnect topology."""
    try:
        import torch

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return InterconnectInfo(topology="single_gpu")
        # Probe /sys/class/kfd for xGMI links (HSA_IOLINK_TYPE_XGMI = 11).
        try:
            import os as _os

            kfd_root = "/sys/class/kfd/kfd/topology/nodes"
            xgmi_count = 0
            for node in _os.listdir(kfd_root):
                links_dir = _os.path.join(kfd_root, node, "io_links")
                if not _os.path.isdir(links_dir):
                    continue
                for link in _os.listdir(links_dir):
                    pf = _os.path.join(links_dir, link, "properties")
                    try:
                        with open(pf) as f:
                            for line in f:
                                if line.startswith("type ") and line.split()[1] == "11":
                                    xgmi_count += 1
                    except OSError:
                        continue
            if xgmi_count > 0:
                full = device_count * (device_count - 1)
                topo = "xgmi_full" if xgmi_count >= full else "xgmi_pairs"
                return InterconnectInfo(topology=topo)
        except Exception:
            pass
        return InterconnectInfo(topology="pcie")
    except Exception:
        return None


def _detect_cuda_numa_cpu_affinity() -> tuple[tuple[int, ...], ...]:
    """Return NUMA-local CPU IDs per visible CUDA device using NVML."""
    nvml_initialized = False
    try:
        import pynvml

        device_count = torch.cuda.device_count()
        if device_count == 0:
            return ()

        pynvml.nvmlInit()
        nvml_initialized = True

        c_ulong_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        cpu_count = os.cpu_count()
        if not cpu_count:
            return ()

        affinities: list[tuple[int, ...]] = []
        for device_id in range(device_count):
            props = torch.cuda.get_device_properties(device_id)
            pci_bus_id = (
                f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:"
                f"{props.pci_device_id:02X}.0"
            )
            handle = pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id)
            masks = pynvml.nvmlDeviceGetCpuAffinity(
                handle, math.ceil(cpu_count / c_ulong_bits)
            )
            affinities.append(
                tuple(
                    cpu
                    for cpu in range(cpu_count)
                    if masks[cpu // c_ulong_bits] & (1 << (cpu % c_ulong_bits))
                )
            )
    except Exception as e:
        logger.warning("NVML failed to query NUMA affinity: %s", e)
        return ()
    finally:
        if nvml_initialized:
            pynvml.nvmlShutdown()

    return tuple(affinities)


@lru_cache(maxsize=1)
def _detect_cuda_nvlink_topology() -> str | None:
    """Return NVLink topology for visible CUDA devices using NVML."""
    nvml_initialized = False
    try:
        import pynvml

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return None

        pynvml.nvmlInit()
        nvml_initialized = True

        handles = []
        for device_id in range(device_count):
            props = torch.cuda.get_device_properties(device_id)
            pci_bus_id = (
                f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:"
                f"{props.pci_device_id:02X}.0"
            )
            handles.append(pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id))

        has_nvlink = False
        full_nvlink = True
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i >= j:
                    continue
                try:
                    p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                        handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                    )
                    if p2p_status == pynvml.NVML_P2P_STATUS_OK:
                        has_nvlink = True
                    else:
                        full_nvlink = False
                except pynvml.NVMLError:
                    full_nvlink = False
    except Exception as e:
        logger.warning("NVML failed to query NVLink topology: %s", e)
        return None
    finally:
        if nvml_initialized:
            pynvml.nvmlShutdown()

    if full_nvlink:
        return "nvlink_full"
    if has_nvlink:
        return "nvlink_pairs"
    return None


def _check_symmetric_memory_available() -> bool:
    """Check if PyTorch symmetric memory is available."""
    try:
        import torch.distributed._symmetric_memory  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


def _check_nvlink_available() -> bool:
    """Check if NVLink connectivity is available."""
    return _detect_cuda_nvlink_topology() is not None
