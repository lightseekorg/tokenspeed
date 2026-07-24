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

"""Check environment configurations and dependency versions."""

import importlib.metadata
import os
import resource
import subprocess
import sys
from collections import OrderedDict, defaultdict

import torch


def is_cuda_build() -> bool:
    return torch.version.cuda is not None


def is_rocm_build() -> bool:
    return getattr(torch.version, "hip", None) is not None


# List of packages to check versions
PACKAGE_LIST = [
    "tokenspeed",
    "aiohttp",
    "apache-tvm-ffi",
    "compressed-tensors",
    "dill",
    "einops",
    "fastapi",
    "flashinfer-cubin",
    "flashinfer-python",
    "hf_transfer",
    "huggingface_hub",
    "instanttensor",
    "modelscope",
    "msgspec",
    "ninja",
    "numpy",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-cu13",
    "nvidia-ml-py",
    "nvtx",
    "openai",
    "openai-harmony",
    "orjson",
    "packaging",
    "partial-json-parser",
    "peft",
    "pillow",
    "prometheus-client",
    "psutil",
    "pybase64",
    "pybind11",
    "pydantic",
    "py-spy",
    "PyYAML",
    "pytest-asyncio",
    "python-multipart",
    "pyzmq",
    "quack-kernels",
    "requests",
    "setproctitle",
    "tiktoken",
    "tokenspeed-deepep",
    "tokenspeed-deepgemm",
    "tokenspeed-fa3",
    "tokenspeed-fa4",
    "tokenspeed-fast-hadamard-transform",
    "tokenspeed-flashmla",
    "tokenspeed-iris",
    "tokenspeed-kernel",
    "tokenspeed-kernel-amd",
    "tokenspeed-mla",
    "tokenspeed-mooncake",
    "tokenspeed-proton",
    "tokenspeed-scheduler",
    "tokenspeed-smg",
    "tokenspeed-smg-grpc-proto",
    "tokenspeed-smg-grpc-servicer",
    "tokenspeed-triton",
    "tokenspeed-triton-kernels",
    "tokenspeed-trtllm-kernel",
    "torch",
    "torch_memory_saver",
    "torchvision",
    "tqdm",
    "transformers",
    "uv",
    "uvicorn",
    "uvloop",
    "viztracer",
    "xgrammar",
]


def get_package_versions(packages: list[str]) -> dict[str, str]:
    """Get versions of specified packages."""
    versions = {}
    for package in packages:
        package_name = package.split("==")[0].split(">=")[0].split("<=")[0]
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "Package Not Found"
    return versions


def get_cuda_info() -> dict[str, object]:
    """Get CUDA-related information if available."""
    if is_cuda_build():
        cuda_info = {"CUDA available": torch.cuda.is_available()}

        if cuda_info["CUDA available"]:
            cuda_info.update(_get_gpu_info())
            cuda_info.update(_get_cuda_version_info())

        return cuda_info
    elif is_rocm_build():
        cuda_info = {"ROCM available": torch.cuda.is_available()}

        if cuda_info["ROCM available"]:
            cuda_info.update(_get_gpu_info())
            cuda_info.update(_get_cuda_version_info())

        return cuda_info
    return {}


def _get_gpu_info() -> dict[str, str]:
    """Get information about available GPUs."""
    devices = defaultdict(list)
    capabilities = defaultdict(list)
    for device_index in range(torch.cuda.device_count()):
        devices[torch.cuda.get_device_name(device_index)].append(str(device_index))
        capability = torch.cuda.get_device_capability(device_index)
        capabilities[f"{capability[0]}.{capability[1]}"].append(str(device_index))

    gpu_info = {}
    for name, device_ids in devices.items():
        gpu_info[f"GPU {','.join(device_ids)}"] = name

    if len(capabilities) == 1:
        # All GPUs have the same compute capability
        cap, gpu_ids = next(iter(capabilities.items()))
        gpu_info[f"GPU {','.join(gpu_ids)} Compute Capability"] = cap
    else:
        # GPUs have different compute capabilities
        for cap, gpu_ids in capabilities.items():
            gpu_info[f"GPU {','.join(gpu_ids)} Compute Capability"] = cap

    return gpu_info


def _get_cuda_version_info() -> dict[str, str | None]:
    """Get CUDA version information."""
    if is_cuda_build():
        from torch.utils.cpp_extension import CUDA_HOME

        cuda_info = {"CUDA_HOME": CUDA_HOME}

        if CUDA_HOME and os.path.isdir(CUDA_HOME):
            cuda_info.update(_get_nvcc_info())
            cuda_info.update(_get_cuda_driver_version())

        return cuda_info
    if is_rocm_build():
        from torch.utils.cpp_extension import ROCM_HOME

        cuda_info = {"ROCM_HOME": ROCM_HOME}

        if ROCM_HOME and os.path.isdir(ROCM_HOME):
            cuda_info.update(_get_nvcc_info())
            cuda_info.update(_get_cuda_driver_version())

        return cuda_info
    return {"CUDA_HOME": ""}


def _get_nvcc_info() -> dict[str, str]:
    """Get NVCC version information."""
    if is_cuda_build():
        from torch.utils.cpp_extension import CUDA_HOME

        if not CUDA_HOME:
            return {"NVCC": "Not Available"}

        try:
            nvcc = os.path.join(CUDA_HOME, "bin/nvcc")
            nvcc_output = subprocess.check_output([nvcc, "-V"], text=True).strip()
            return {
                "NVCC": nvcc_output[
                    nvcc_output.rfind("Cuda compilation tools") : nvcc_output.rfind(
                        "Build"
                    )
                ].strip()
            }
        except (OSError, subprocess.SubprocessError):
            return {"NVCC": "Not Available"}
    elif is_rocm_build():
        from torch.utils.cpp_extension import ROCM_HOME

        if not ROCM_HOME:
            return {"HIPCC": "Not Available"}

        try:
            hipcc = os.path.join(ROCM_HOME, "bin/hipcc")
            hipcc_output = subprocess.check_output(
                [hipcc, "--version"], text=True
            ).strip()
            return {
                "HIPCC": hipcc_output[
                    hipcc_output.rfind("HIP version") : hipcc_output.rfind("AMD clang")
                ].strip()
            }
        except (OSError, subprocess.SubprocessError):
            return {"HIPCC": "Not Available"}
    else:
        return {"NVCC": "Not Available"}


def _get_cuda_driver_version() -> dict[str, str]:
    """Get CUDA driver version."""
    if is_cuda_build():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            versions = set(output.strip().splitlines())
            if len(versions) == 1:
                return {"CUDA Driver Version": versions.pop()}
            else:
                return {"CUDA Driver Versions": ", ".join(sorted(versions))}
        except (OSError, subprocess.SubprocessError):
            return {"CUDA Driver Version": "Not Available"}
    elif is_rocm_build():
        try:
            output = subprocess.check_output(
                [
                    "rocm-smi",
                    "--showdriverversion",
                    "--csv",
                ],
                text=True,
            )
            versions = set(output.strip().splitlines())
            versions.discard("name, value")
            if not versions:
                return {"ROCM Driver Version": "Not Available"}
            ver = versions.pop()
            ver = ver.replace('"Driver version", ', "").replace('"', "")

            return {"ROCM Driver Version": ver}
        except (OSError, subprocess.SubprocessError):
            return {"ROCM Driver Version": "Not Available"}
    else:
        return {"CUDA Driver Version": "Not Available"}


def get_gpu_topology() -> str | None:
    """Get GPU topology information."""
    if is_cuda_build():
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return "\n" + result.stdout
        except (OSError, subprocess.SubprocessError):
            return None
    elif is_rocm_build():
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtopotype"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return "\n" + result.stdout
        except (OSError, subprocess.SubprocessError):
            return None
    else:
        return None


def get_hypervisor_vendor() -> str | None:
    try:
        output = subprocess.check_output(["lscpu"], text=True)
        for line in output.splitlines():
            if "Hypervisor vendor:" in line:
                _, _, vendor = line.partition(":")
                return vendor.strip()
        return None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    """Check and print environment information."""
    env_info = OrderedDict()
    env_info["Python"] = sys.version.replace("\n", "")
    env_info.update(get_cuda_info())
    env_info["PyTorch"] = torch.__version__
    env_info.update(get_package_versions(PACKAGE_LIST))

    gpu_topo = get_gpu_topology()
    if gpu_topo:
        if is_cuda_build():
            env_info["NVIDIA Topology"] = gpu_topo
        elif is_rocm_build():
            env_info["AMD Topology"] = gpu_topo

    hypervisor_vendor = get_hypervisor_vendor()
    if hypervisor_vendor:
        env_info["Hypervisor vendor"] = hypervisor_vendor

    ulimit_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    env_info["ulimit soft"] = ulimit_soft

    for k, v in env_info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
