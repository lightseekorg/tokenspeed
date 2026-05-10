import os
import shutil

from setuptools import setup

# Detect AMD/ROCm platform by checking common ROCm indicators.
# We deliberately avoid importing torch here because it may not be installed
# yet when pip resolves dependencies for the first time.
_is_amd = (
    os.path.isdir("/opt/rocm")
    or shutil.which("rocm-smi") is not None
    or os.environ.get("ROCM_HOME") is not None
    or os.environ.get("ROCM_PATH") is not None
)

_BASE_DEPS = [
    "aiohttp",
    "compressed-tensors",
    "dill",
    "einops",
    "fastapi",
    "hf_transfer",
    "huggingface_hub",
    "modelscope",
    "msgspec",
    "ninja",
    "numpy",
    "openai==2.33.0",
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
    "python-multipart",
    "pyzmq",
    "requests",
    "setproctitle",
    "tiktoken",
    "torch==2.11.0",
    "torchvision",
    "tqdm",
    "transformers==5.6.2",
    "uv",
    "uvicorn",
    "uvloop",
    "xgrammar==0.1.33",
    "viztracer",
]

# nvidia-cutlass-dsl and nvtx are NVIDIA-only; skip them on AMD/ROCm hosts.
_NVIDIA_DEPS = [
    "nvidia-cutlass-dsl",
    "nvtx",
]

install_requires = _BASE_DEPS if _is_amd else _BASE_DEPS + _NVIDIA_DEPS

setup(install_requires=install_requires)
