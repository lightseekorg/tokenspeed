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

"""Common utilities."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import io
import ipaddress
import json
import logging
import os
import pickle
import random
import re
import resource
import shutil
import subprocess
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
)

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import pybase64  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
import torch.distributed  # noqa: E402
import torch.distributed as dist  # noqa: E402
import triton  # noqa: E402
import zmq  # noqa: E402
from fastapi.responses import ORJSONResponse  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.routing import Mount  # noqa: E402
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.metrics.func_timer import enable_func_timer

logger = logging.getLogger(__name__)

time_infos = {}


_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    # Runtime env helpers still read a few legacy keys directly until the
    # central env module owns all boolean parsing.
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            logger.warning(
                "get_bool_env_var(%s) see non-understandable value=%s and treat as false",
                name,
                value,
            )
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


@lru_cache(maxsize=1)
def get_device_module():
    """Get the device module (cuda, hip, etc.) based on the current device."""
    return torch.get_device_module()


def maybe_inference_mode():
    from tokenspeed.runtime.utils.env import envs

    if envs.TOKENSPEED_ENABLE_TORCH_INFERENCE_MODE.get():
        return torch.inference_mode()
    else:
        return torch.no_grad()


def maybe_set_numa_aware_cpu_affinity(device_id: int) -> None:
    """Pin the current process to ``device_id``'s NUMA-local CPU set.

    NVIDIA-only optimization. No-op if the env var is False, the platform is not
    NVIDIA, or the process already has a constrained affinity (e.g., taskset).
    """
    from tokenspeed.runtime.utils.env import envs

    if not envs.TOKENSPEED_NUMA_AWARE_WORKER_AFFINITY.get():
        return
    platform = current_platform()
    if not platform.is_nvidia:
        return

    proc = psutil.Process()
    if proc.cpu_affinity() != list(range(psutil.cpu_count())):
        return

    if device_id >= len(platform.numa_cpu_affinity):
        return

    cpu_affinity = platform.numa_cpu_affinity[device_id]
    if not cpu_affinity:
        return

    proc.cpu_affinity(list(cpu_affinity))
    logger.info(
        "Worker process %s pinned to %s NUMA-local CPUs for device %s.",
        proc.pid,
        len(cpu_affinity),
        device_id,
    )


def get_available_gpu_memory(
    device, gpu_id, distributed=False, empty_cache=True, cpu_group=None
):
    """
    Get available memory for cuda:gpu_id device.
    When distributed is True, the available memory is the minimum available memory of all GPUs.
    """
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        assert gpu_id < num_gpus

        if torch.cuda.current_device() != gpu_id:
            logger.debug(
                "Current device is not %s, but %s, which may cause useless "
                "memory allocation for torch CUDA context.",
                gpu_id,
                torch.cuda.current_device(),
            )

        if empty_cache:
            torch.cuda.empty_cache()
        free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32)
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
        )
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)


def is_pin_memory_available() -> bool:
    return torch.cuda.is_available()


class LayerFn(Protocol):
    def __call__(self, idx: int, prefix: str) -> torch.nn.Module: ...


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str = "",
) -> torch.nn.ModuleList:
    """Make a list of layers with the given layer function"""
    start_layer = 0
    end_layer = num_hidden_layers
    modules = torch.nn.ModuleList(
        [
            layer_fn(idx=idx, prefix=add_prefix(idx, prefix))
            for idx in range(start_layer, end_layer)
        ]
    )
    return modules


def set_random_seed(seed: int) -> None:
    """Set the random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ImageData:
    url: str
    detail: Literal["auto", "low", "high"] | None = "auto"


def load_image(
    image_file: Image.Image | str | ImageData | bytes,
) -> tuple[Image.Image, tuple[int, int]]:
    if isinstance(image_file, ImageData):
        image_file = image_file.url

    image = image_size = None
    if isinstance(image_file, Image.Image):
        image = image_file
        image_size = (image.width, image.height)
    elif isinstance(image_file, bytes):
        image = Image.open(BytesIO(image_file))
    elif image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, stream=True, timeout=timeout)
        try:
            response.raise_for_status()
            image = Image.open(response.raw)
            image.load()  # Force loading to avoid issues after closing the stream
        finally:
            response.close()
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        image = Image.open(image_file)
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        image = Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    elif isinstance(image_file, str):
        image = Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    else:
        raise ValueError(f"Invalid image: {image_file}")

    return image, image_size


def set_ulimit(target_soft_limit=65535):
    # number of open files
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning("Failed to set RLIMIT_NOFILE: %s", e)

    # stack size
    resource_type = resource.RLIMIT_STACK
    current_soft, current_hard = resource.getrlimit(resource_type)
    target_soft_limit_stack_size = 1024 * target_soft_limit
    if current_soft < target_soft_limit_stack_size:
        try:
            resource.setrlimit(
                resource_type, (target_soft_limit_stack_size, current_hard)
            )
        except ValueError as e:
            logger.warning("Failed to set RLIMIT_STACK: %s", e)


def add_api_key_middleware(app, api_key: str):
    @app.middleware("http")
    async def authentication(request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path.startswith("/health"):
            return await call_next(request)
        if request.url.path.startswith("/metrics"):
            return await call_next(request)
        if request.headers.get("Authorization") != "Bearer " + api_key:
            return ORJSONResponse(content={"error": "Unauthorized"}, status_code=401)
        return await call_next(request)


def prepare_model_and_tokenizer(model_path: str, tokenizer_path: str):
    from tokenspeed.runtime.utils.env import envs

    if envs.TOKENSPEED_USE_MODELSCOPE.get():
        if not os.path.exists(model_path):
            from modelscope import snapshot_download

            model_path = snapshot_download(model_path)
            tokenizer_path = snapshot_download(
                tokenizer_path, ignore_patterns=["*.bin", "*.safetensors"]
            )
    return model_path, tokenizer_path


def configure_logger(server_args, prefix: str = ""):
    global LOG_PREFIX
    LOG_PREFIX = prefix

    global LOG_LEVEL
    LOG_LEVEL = server_args.log_level.upper()

    from tokenspeed._logging import suppress_noisy_third_party_logs
    from tokenspeed.runtime.utils.env import envs

    suppress_noisy_third_party_logs()

    if TOKENSPEED_LOGGING_CONFIG_PATH := envs.TOKENSPEED_LOGGING_CONFIG_PATH.get():
        if not os.path.exists(TOKENSPEED_LOGGING_CONFIG_PATH):
            raise Exception(
                "Setting TOKENSPEED_LOGGING_CONFIG_PATH from env with "
                f"{TOKENSPEED_LOGGING_CONFIG_PATH} but it does not exist!"
            )
        with open(TOKENSPEED_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())
        logging.config.dictConfig(custom_config)
        suppress_noisy_third_party_logs()
        return
    format = f"[%(asctime)s{prefix}] %(message)s"
    log_level = getattr(logging, server_args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Only set specified log level for tokenspeed-related loggers
    for logger_name in logging.Logger.manager.loggerDict:
        if "tokenspeed" in logger_name or logger_name.startswith("tokenspeed"):
            logger_obj = logging.getLogger(logger_name)
            if isinstance(logger_obj, logging.Logger):
                logger_obj.setLevel(log_level)
                for handler in logger_obj.handlers:
                    handler.setLevel(log_level)

    suppress_noisy_third_party_logs()


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)


def broadcast_pyobj(
    data: list[Any],
    rank: int,
    dist_group: torch.distributed.ProcessGroup | None = None,
    src: int = 0,
    force_cpu_device: bool = True,
):
    """Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu_device else "cpu"
    )

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)

            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            ).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data


step_counter = 0


def get_zmq_socket(
    context: zmq.Context, socket_type: zmq.SocketType, endpoint: str, bind: bool
) -> zmq.Socket:
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1

    socket = context.socket(socket_type)
    if endpoint.find("[") != -1:
        socket.setsockopt(zmq.IPV6, 1)

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type == zmq.PUSH:
        set_send_opt()
    elif socket_type == zmq.PULL:
        set_recv_opt()
    elif socket_type == zmq.DEALER:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")

    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)

    return socket


def delete_directory(dirpath):
    try:
        # This will remove the directory and all its contents
        shutil.rmtree(dirpath)
    except OSError as e:
        logger.warning("Failed to delete directory %s: %s", dirpath, e.strerror)


# Temporary directory for prometheus multiprocess mode
# Cleaned up automatically when this object is garbage collected
prometheus_multiproc_dir: tempfile.TemporaryDirectory


def set_prometheus_multiproc_dir():
    # Set prometheus multiprocess directory
    # tokenspeed uses prometheus multiprocess mode
    # we need to set this before importing prometheus_client
    # https://prometheus.github.io/client_python/multiprocess/
    global prometheus_multiproc_dir

    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        logger.debug("User set PROMETHEUS_MULTIPROC_DIR detected.")
        prometheus_multiproc_dir = tempfile.TemporaryDirectory(
            dir=os.environ["PROMETHEUS_MULTIPROC_DIR"]
        )
    else:
        prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
    logger.debug("PROMETHEUS_MULTIPROC_DIR: %s", os.environ["PROMETHEUS_MULTIPROC_DIR"])


def add_prometheus_middleware(app):
    # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def get_amdgpu_memory_capacity():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No AMD GPU available. Ensure ROCm drivers and a ROCm-enabled "
            "PyTorch build are installed and accessible."
        )

    # Query each visible device's total memory (bytes) via the torch API
    # (torch.cuda is reused for ROCm/HIP), and return the minimum in MiB so
    # the value matches the previous rocminfo-based implementation.
    memory_values = [
        torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)
        for i in range(torch.cuda.device_count())
    ]

    if not memory_values:
        raise ValueError("No GPU memory values found.")

    return min(memory_values)


def get_nvgpu_memory_capacity():
    try:
        # Run nvidia-smi and capture the output
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi error: {result.stderr.strip()}")

        # Parse the output to extract memory values
        memory_values = [
            float(mem)
            for mem in result.stdout.strip().split("\n")
            if re.match(r"^\d+(\.\d+)?$", mem.strip())
        ]

        if not memory_values:
            # Fallback to torch.cuda.mem_get_info() when failed to get memory capacity from nvidia-smi,
            # typically in NVIDIA MIG mode.
            if torch.cuda.is_available():
                logger.warning(
                    "Failed to get GPU memory capacity from nvidia-smi, falling back to torch.cuda.mem_get_info()."
                )
                return torch.cuda.mem_get_info()[1] // 1024 // 1024  # unit: MB
            raise ValueError("No GPU memory values found.")

        # Return the minimum memory value
        return min(memory_values)

    except FileNotFoundError:
        raise RuntimeError(
            "nvidia-smi not found. Ensure NVIDIA drivers are installed and accessible."
        )


def crash_on_warnings():
    # Crash on warning if we are running CI tests
    return get_bool_env_var("CI") or get_bool_env_var("GITHUB_ACTIONS")


def print_warning_once(msg: str) -> None:
    # Set the stacklevel to 2 to print the caller's line info
    logger.warning(msg, stacklevel=2)


def get_device_name(device_id: int = 0) -> str:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(device_id)

    return ""


@lru_cache(maxsize=8)
def get_device(device_id: int | None = None) -> str:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if device_id is None:
            return "cuda"
        return f"cuda:{device_id}"

    raise RuntimeError("No accelerator (CUDA/ROCm) is available.")


def dataclass_to_string_truncated(
    data, max_length=2048, skip_names: set[str] | None = None
):
    if skip_names is None:
        skip_names = set()
    if isinstance(data, str):
        if len(data) > max_length:
            half_length = max_length // 2
            return f"{repr(data[:half_length])} ... {repr(data[-half_length:])}"
        else:
            return f"{repr(data)}"
    elif isinstance(data, (list, tuple)):
        if len(data) > max_length:
            half_length = max_length // 2
            return str(data[:half_length]) + " ... " + str(data[-half_length:])
        else:
            return str(data)
    elif isinstance(data, dict):
        return (
            "{"
            + ", ".join(
                f"'{k}': {dataclass_to_string_truncated(v, max_length)}"
                for k, v in data.items()
                if k not in skip_names
            )
            + "}"
        )
    elif dataclasses.is_dataclass(data):
        fields = dataclasses.fields(data)
        return (
            f"{data.__class__.__name__}("
            + ", ".join(
                f"{f.name}={dataclass_to_string_truncated(getattr(data, f.name), max_length)}"
                for f in fields
                if f.name not in skip_names
            )
            + ")"
        )
    else:
        return str(data)


class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        """
        Serialize a Python object using ForkingPickler.

        Args:
            obj: The object to serialize.
            output_str (bool): If True, return a base64-encoded string instead of raw bytes.

        Returns:
            bytes or str: The serialized object.
        """
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()

        if output_str:
            # Convert bytes to base64-encoded string
            output = pybase64.b64encode(output).decode("utf-8")

        return output

    @staticmethod
    def deserialize(data):
        """
        Deserialize a previously serialized object.

        Args:
            data (bytes or str): The serialized data, optionally base64-encoded.

        Returns:
            The deserialized Python object.
        """
        if isinstance(data, str):
            # Decode base64 string to bytes
            data = pybase64.b64decode(data, validate=True)

        return ForkingPickler.loads(data)


def debug_timing(func):
    def wrapper(*args, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)
            tic.record()
            result = func(*args, **kwargs)
            toc.record()
            toc.synchronize()  # Wait for the function to complete without synchronizing all ops on the GPU
            elapsed = tic.elapsed_time(toc)
            indices = kwargs.get("indices", args[1] if len(args) > 1 else None)
            num_tokens = len(indices) if indices is not None else 0
            throughput = num_tokens / elapsed * 1000 if elapsed > 0 else 0
            logger.debug(
                "Transfer time: %s ms, throughput: %s tokens/s", elapsed, throughput
            )
            return result
        else:
            return func(*args, **kwargs)

    return wrapper


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def launch_dummy_health_check_server(host, port, enable_metrics):

    import uvicorn
    from fastapi import FastAPI, Response

    app = FastAPI()

    @app.get("/health")
    async def health():
        """Check the health of the http server."""
        return Response(status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        """Check the health of the http server."""
        return Response(status_code=200)

    # Add prometheus middleware
    if enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        timeout_keep_alive=5,
        loop="auto",
        log_config=None,
        log_level="warning",
    )
    server = uvicorn.Server(config=config)

    try:
        loop = asyncio.get_running_loop()
        logger.info(
            "Dummy health check server scheduled on existing loop at %s:%s", host, port
        )
        loop.create_task(server.serve())

    except RuntimeError:
        logger.info("Starting dummy health check server at %s:%s", host, port)
        server.run()


def set_cuda_arch():
    platform = current_platform()
    if not platform.is_nvidia:
        return

    arch = f"{platform.arch_version.major}.{platform.arch_version.minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch}{'+PTX' if arch == '9.0' else ''}"


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


def round_up(x: int, y: int) -> int:
    return ((x - 1) // y + 1) * y


setattr(triton, "next_power_of_2", next_power_of_2)


def add_prefix(name: str, prefix: str) -> str:
    """Add a weight path prefix to a module name.

    Args:
        name: base module name.
        prefix: weight prefix str to added to the front of `name` concatenated with `.`.

    Returns:
        The string `prefix.name` if prefix is non-empty, otherwise just `name`.
    """
    return name if not prefix else f"{prefix}.{name}"


# Can be more general if it is used in multiple places (keep it simple and thus not general now)


def log_info_on_rank0(logger, msg):
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(msg)


T = TypeVar("T")


class Withable(Generic[T]):
    def __init__(self):
        self._value: T | None = None

    @property
    def value(self) -> T:
        return self._value

    @contextmanager
    def with_value(self, new_value: T):
        assert self._value is None
        self._value = new_value
        try:
            yield
        finally:
            assert self._value is new_value
            self._value = None


def find_local_repo_dir(repo_id: str, revision: str | None = None) -> str | None:
    import huggingface_hub as hf

    # Build cache path
    cache_path = os.path.join(
        hf.constants.HF_HUB_CACHE,
        hf.constants.REPO_ID_SEPARATOR.join(["models", *repo_id.split("/")]),
    )

    # Get revision from main ref if not specified
    if not revision:
        ref_path = os.path.join(cache_path, "refs", "main")
        if os.path.isfile(ref_path):
            with open(ref_path) as f:
                revision = f.read().strip()

    # List files from revision directory
    if revision:
        rev_dir = os.path.join(cache_path, "snapshots", revision)
        if os.path.isdir(rev_dir):
            return rev_dir

    return None


def read_system_prompt_from_file(model_name: str) -> str:
    """Read system prompt from a file in the HuggingFace cache directory.

    Args:
        model_name: The model name to construct the file path

    Returns:
        The system prompt content from the file, or empty string if file not found
    """
    try:
        local_repo_dir = find_local_repo_dir(model_name)
        if local_repo_dir:
            system_prompt_file = os.path.join(local_repo_dir, "SYSTEM_PROMPT.txt")
            if os.path.exists(system_prompt_file):
                with open(system_prompt_file, encoding="utf-8") as f:
                    return f.read()

        return ""
    except Exception:
        # If anything fails, return empty string
        return ""


class LazyValue:
    def __init__(self, creator: Callable):
        self._creator = creator
        self._value = None

    @property
    def value(self):
        if self._creator is not None:
            self._value = self._creator()
            self._creator = None
        return self._value


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# Only physical cores are used. Logical cores are excluded.


def lru_cache_frozenset(maxsize=128):
    def _to_hashable(o):
        try:
            hash(o)
            return o
        except TypeError:
            # Not hashable; convert based on type
            if isinstance(o, (dict)):
                return frozenset(
                    (_to_hashable(k), _to_hashable(v)) for k, v in o.items()
                )
            elif isinstance(o, set):
                return frozenset(_to_hashable(v) for v in o)
            elif isinstance(o, (list, tuple)) or (
                isinstance(o, Sequence) and not isinstance(o, (str, bytes))
            ):
                return tuple(_to_hashable(v) for v in o)
            else:
                raise TypeError(f"Cannot make hashable: {type(o)}")

    def decorator(func):
        cache = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            h_args = tuple(_to_hashable(a) for a in args)
            h_kwargs = frozenset(
                (_to_hashable(k), _to_hashable(v)) for k, v in kwargs.items()
            )
            key = (h_args, h_kwargs)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        wrapper.cache_clear = cache.clear  # For manual cache clearing
        return wrapper

    return decorator


LOG_PREFIX = None
LOG_LEVEL = "INFO"


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = None

    def format(self, record):
        if self.FORMATS is None:
            format = f"[%(asctime)s {LOG_PREFIX}] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
            self.FORMATS = {
                logging.DEBUG: self.grey + format + self.reset,
                logging.INFO: self.grey + format + self.reset,
                logging.WARNING: self.yellow + format + self.reset,
                logging.ERROR: self.red + format + self.reset,
                logging.CRITICAL: self.bold_red + format + self.reset,
            }

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_colorful_logger(name):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(LOG_LEVEL)

    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(CustomFormatter())
    # ch.flush = lambda: True

    logger.addHandler(ch)
    logger.propagate = False
    return logger


def _maybe_json_dict(path: str | os.PathLike) -> dict[str, str]:
    with open(path) as f:
        try:
            return json.loads(f.read())
        except Exception:
            return dict[str, str]()


def _maybe_space_split_dict(path: str | os.PathLike) -> dict[str, str]:
    parsed_dict = dict[str, str]()
    with open(path) as f:
        for line in f.readlines():
            try:
                model_name, redirect_name = line.strip().split()
                parsed_dict[model_name] = redirect_name
            except Exception:
                pass
    return parsed_dict


def maybe_model_redirect(model: str) -> str:
    """
    Use model_redirect to redirect the model name to a local folder.

    :param model: hf model name
    :return: maybe redirect to a local folder
    """

    from tokenspeed.runtime.utils.env import envs

    model_redirect_path = envs.TOKENSPEED_MODEL_REDIRECT_PATH.get()

    if not model_redirect_path:
        return model

    if not Path(model_redirect_path).exists():
        return model

    redirect_dict = _maybe_json_dict(model_redirect_path) or _maybe_space_split_dict(
        model_redirect_path
    )
    if redirect_model := redirect_dict.get(model):
        logger.info("model redirect: [ %s ] -> [ %s ]", model, redirect_model)
        return redirect_model

    return model


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def convert_json_schema_to_str(json_schema: dict | str | type[BaseModel]) -> str:
    """Convert a JSON schema to a string.
    Parameters
    ----------
    json_schema
        The JSON schema.
    Returns
    -------
    str
        The JSON schema converted to a string.
    Raises
    ------
    ValueError
        If the schema is not a dictionary, a string or a Pydantic class.
    """
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    return schema_str
