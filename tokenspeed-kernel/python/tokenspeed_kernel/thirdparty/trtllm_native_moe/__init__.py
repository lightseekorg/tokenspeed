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

"""Temporary source-vendored TensorRT-LLM native routed-MoE runner.

The C++/CUDA source is derived from NVIDIA/TensorRT-LLM commit
54efe57e41f9742d915da977e7701c5cff2b7d6c.  TODO: use the implementation
exported by tokenspeed-trtllm-kernel and delete this vendored module once that
package exposes ``MxE4m3MxE2m1BlockScaleMoERunner``.
"""

from __future__ import annotations

import ctypes
import os
import threading
import warnings
from functools import lru_cache
from pathlib import Path

import torch

_CLASS_NAME = "MxE4m3MxE2m1BlockScaleMoERunner"
_TORCH_NAMESPACE = "tokenspeed_trtllm_native_moe"
_LIBRARY = Path(__file__).resolve().parent / "objs/libtokenspeed_trtllm_native_moe.so"
_TACTIC_ABI = "trtllm-54efe57e-v4-mxfp8-mxfp4-b200-tactics-v2"
_TACTIC_ABI_SYMBOL = "tokenspeed_trtllm_native_moe_tactic_abi"
_ROUTING_METHOD_RENORMALIZE = 1
_LOAD_LOCK = threading.RLock()
_LIBRARY_HANDLE: ctypes.CDLL | None = None
_OFFLINE_TACTICS_COMPATIBLE = False
_RUNNER = None
_LOAD_ERROR: Exception | None = None

# Offline-tuned on an NVIDIA B200 against the vendored source revision above.
# TensorRT-LLM tactic indices are cubin/config-order dependent, so this table
# must never be used for a runner supplied by another shared library.  TODO:
# replace both this table and the vendored runner with the autotuned interface
# exported by tokenspeed-trtllm-kernel.
_V4_PRO_TP8_B200_TACTICS = {
    1: (8, 120),
    2: (8, 180),
    4: (16, 26),
    8: (8, 136),
    16: (16, 18),
    32: (16, 50),
    64: (16, 50),
    128: (16, 50),
    256: (8, 168),
    512: (16, 50),
    1024: (32, 38),
    2048: (64, 8),
    4096: (128, 16),
    8192: (128, 16),
}
_V4_PRO_EP8_B200_TACTICS = {
    1: (8, 165),
    2: (8, 104),
    4: (16, 54),
    8: (8, 132),
    16: (8, 200),
    32: (8, 196),
    64: (16, 50),
    128: (8, 136),
    256: (16, 54),
    512: (32, 20),
    1024: (64, 13),
    2048: (128, 18),
    4096: (128, 18),
    8192: (128, 18),
}


def _registered_runner_class():
    try:
        return getattr(getattr(torch.classes, _TORCH_NAMESPACE), _CLASS_NAME)
    except (AttributeError, RuntimeError):
        return None


def _preload_torch_libraries() -> None:
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    mode = os.RTLD_GLOBAL | os.RTLD_NOW
    for name in (
        "libc10.so",
        "libtorch.so",
        "libtorch_cpu.so",
        "libc10_cuda.so",
        "libtorch_cuda.so",
    ):
        path = torch_lib / name
        if path.exists():
            ctypes.CDLL(path, mode=mode)


def _matches_offline_tactic_abi(library: ctypes.CDLL) -> bool:
    """Check that config indices were tuned against this runner build ABI."""

    try:
        get_tactic_abi = getattr(library, _TACTIC_ABI_SYMBOL)
    except AttributeError:
        return False
    get_tactic_abi.argtypes = []
    get_tactic_abi.restype = ctypes.c_char_p
    value = get_tactic_abi()
    try:
        return value is not None and value.decode("ascii") == _TACTIC_ABI
    except UnicodeDecodeError:
        return False


def _check_source_library_freshness() -> None:
    """Reject a source-checkout DSO older than its native build inputs."""

    source_root = Path(__file__).resolve().parent
    source_paths = [
        source_root / "CMakeLists.txt",
        source_root / "tactic_abi.cpp",
        source_root / "cmake",
    ]
    csrc = source_root / "csrc"
    if csrc.is_dir():
        source_paths.append(csrc)
    else:
        # Installed wheels intentionally contain only the built DSO and license
        # files, so there is no source freshness check to perform there.
        return

    newest_input_ns = 0
    for path in source_paths:
        candidates = path.rglob("*") if path.is_dir() else (path,)
        newest_input_ns = max(
            newest_input_ns,
            *(
                candidate.stat().st_mtime_ns
                for candidate in candidates
                if candidate.is_file()
            ),
        )
    if _LIBRARY.stat().st_mtime_ns < newest_input_ns:
        raise RuntimeError(
            f"native MoE library is older than its source inputs; rebuild {_LIBRARY}"
        )


def _load_runner_class():
    global _LIBRARY_HANDLE, _LOAD_ERROR, _OFFLINE_TACTICS_COMPATIBLE

    runner_class = _registered_runner_class()
    if runner_class is not None:
        return runner_class

    with _LOAD_LOCK:
        runner_class = _registered_runner_class()
        if runner_class is not None:
            return runner_class
        if _LOAD_ERROR is not None:
            raise RuntimeError(
                "native TRT-LLM MoE runner failed to load"
            ) from _LOAD_ERROR

        try:
            if not _LIBRARY.is_file():
                raise FileNotFoundError(f"native MoE library not built: {_LIBRARY}")
            _check_source_library_freshness()
            _preload_torch_libraries()
            _LIBRARY_HANDLE = ctypes.CDLL(_LIBRARY, mode=os.RTLD_LOCAL | os.RTLD_NOW)
            _OFFLINE_TACTICS_COMPATIBLE = _matches_offline_tactic_abi(_LIBRARY_HANDLE)
            if not _OFFLINE_TACTICS_COMPATIBLE:
                warnings.warn(
                    "native TRT-LLM MoE tactic ABI does not match the offline "
                    "table; using the runner's default tactic",
                    RuntimeWarning,
                    stacklevel=2,
                )
            runner_class = _registered_runner_class()
            if runner_class is None:
                raise RuntimeError(f"{_CLASS_NAME} was not registered by {_LIBRARY}")
            return runner_class
        except Exception as exc:
            _LOAD_ERROR = exc
            raise


def has_native_mxfp4_moe() -> bool:
    """Return whether the native MXFP8-by-MXFP4 routed-MoE runner is loadable."""

    try:
        _load_runner_class()
    except (ImportError, OSError, RuntimeError):
        return False
    return True


def _runner():
    global _RUNNER
    if _RUNNER is None:
        with _LOAD_LOCK:
            if _RUNNER is None:
                # SwiGLU is ActType 0 in this TensorRT-LLM source revision.
                _RUNNER = _load_runner_class()(0, True)
    return _RUNNER


@lru_cache(maxsize=256)
def _is_valid_offline_tactic(
    *,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    valid_hidden_size: int,
    valid_intermediate_size: int,
    local_num_experts: int,
    top_k: int,
    tile_n: int,
    config_index: int,
) -> bool:
    """Validate a table entry against the loaded runner before dispatch."""

    try:
        valid_configs = _runner().get_valid_configs(
            top_k,
            hidden_size,
            intermediate_size,
            local_num_experts,
            num_tokens,
            valid_hidden_size,
            valid_intermediate_size,
        )
    except (RuntimeError, TypeError, ValueError):
        return False
    is_valid = any(
        tuple(int(value) for value in config) == (tile_n, config_index)
        for config in valid_configs
    )
    if not is_valid:
        warnings.warn(
            f"native TRT-LLM MoE tactic {(tile_n, config_index)} is invalid "
            f"for M={num_tokens}; using the runner's default tactic",
            RuntimeWarning,
            stacklevel=2,
        )
    return is_valid


def select_native_mxfp4_moe_tactic(
    *,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    valid_hidden_size: int,
    valid_intermediate_size: int,
    local_num_experts: int,
    top_k: int,
    device: torch.device,
) -> tuple[int, int]:
    """Select an offline-tuned tactic for the supported V4-Pro B200 shapes.

    TensorRT-LLM maps the dynamic token dimension to the preceding power-of-two
    bucket.
    Unknown hardware, shapes, source libraries, and token counts use the
    runner's ``(-1, -1)`` fallback instead of guessing a config index.
    """

    if (
        _LIBRARY_HANDLE is None
        or not _OFFLINE_TACTICS_COMPATIBLE
        or device.type != "cuda"
        or torch.cuda.get_device_capability(device) != (10, 0)
        or hidden_size != 7168
        or valid_hidden_size != 7168
        or top_k != 6
        or num_tokens <= 0
        or num_tokens > 8192
    ):
        return (-1, -1)

    token_bucket = _tactic_token_bucket(num_tokens)
    if (
        intermediate_size == 384
        and valid_intermediate_size == 384
        and local_num_experts == 384
    ):
        tactics = _V4_PRO_TP8_B200_TACTICS
    elif (
        intermediate_size == 3072
        and valid_intermediate_size == 3072
        and local_num_experts == 48
    ):
        tactics = _V4_PRO_EP8_B200_TACTICS
    else:
        return (-1, -1)
    tactic = tactics.get(token_bucket)
    if tactic is None or not _is_valid_offline_tactic(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        valid_hidden_size=valid_hidden_size,
        valid_intermediate_size=valid_intermediate_size,
        local_num_experts=local_num_experts,
        top_k=top_k,
        tile_n=tactic[0],
        config_index=tactic[1],
    ):
        return (-1, -1)
    return tactic


def _tactic_token_bucket(num_tokens: int, max_tokens: int = 8192) -> int:
    """Map M to the floor-power-of-two bucket used by TRT-LLM autotuning."""

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    return min(1 << (num_tokens.bit_length() - 1), max_tokens)


def run_native_mxfp4_moe(
    *,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: torch.Tensor | None,
    gemm1_alpha: torch.Tensor | None,
    gemm1_beta: torch.Tensor | None,
    gemm1_clamp_limit: torch.Tensor | None,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: torch.Tensor | None,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    valid_hidden_size: int,
    valid_intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    tactic: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """Run preprocessed MXFP8 activations and MXFP4 expert weights.

    Weight shuffle/interleave, activation quantization, and routing top-k are
    deliberately owned by the public TokenSpeed op rather than this loader.

    Args:
        hidden_states: Packed MXFP8 activations with shape ``[tokens, hidden]``.
        hidden_states_scale: Flattened uint8 MX scale factors.
        gemm1_weights: Shuffled uint8 MXFP4 gate/up expert weights.
        gemm1_weights_scale: Interleaved uint8 gate/up weight scales.
        gemm1_bias: Optional per-expert gate/up bias.
        gemm1_alpha: Optional per-expert SwiGLU alpha.
        gemm1_beta: Optional per-expert SwiGLU beta.
        gemm1_clamp_limit: Optional per-expert SwiGLU clamp limit.
        gemm2_weights: Shuffled uint8 MXFP4 down expert weights.
        gemm2_weights_scale: Interleaved uint8 down weight scales.
        gemm2_bias: Optional per-expert down bias.
        num_experts: Global logical expert count.
        top_k: Number of selected experts per token.
        intermediate_size: Physically padded intermediate size per partition.
        valid_hidden_size: Unpadded model hidden size.
        valid_intermediate_size: Unpadded intermediate size per partition.
        local_expert_offset: First global expert owned by this rank.
        local_num_experts: Number of experts owned by this rank.
        topk_weights: Precomputed BF16 routing weights with shape
            ``[tokens, top_k]``.
        topk_ids: Precomputed int32 global expert ids with shape
            ``[tokens, top_k]``.
        output: Preallocated BF16 output with shape ``[tokens, hidden]``.
        tactic: TensorRT-LLM ``(tile_n, config_index)`` pair. ``(-1, -1)``
            requests its default valid tactic.

    Returns:
        The BF16 MoE output tensor written into ``output``.
    """

    if topk_weights.dtype != torch.bfloat16:
        raise TypeError(f"topk_weights must be bfloat16, got {topk_weights.dtype}")
    if topk_ids.dtype != torch.int32:
        raise TypeError(f"topk_ids must be int32, got {topk_ids.dtype}")
    if output.dtype != torch.bfloat16:
        raise TypeError(f"output must be bfloat16, got {output.dtype}")
    if len(tactic) != 2:
        raise ValueError(f"tactic must contain (tile_n, config), got {tactic!r}")
    if tactic != (-1, -1) and not _is_valid_offline_tactic(
        num_tokens=hidden_states.shape[0],
        hidden_size=hidden_states.shape[1],
        intermediate_size=intermediate_size,
        valid_hidden_size=valid_hidden_size,
        valid_intermediate_size=valid_intermediate_size,
        local_num_experts=local_num_experts,
        top_k=top_k,
        tile_n=int(tactic[0]),
        config_index=int(tactic[1]),
    ):
        raise ValueError(f"invalid native TRT-LLM MoE tactic for this shape: {tactic}")

    return _runner().run_moe(
        None,  # routing_logits: top-k results are already available
        None,  # routing_bias
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        None,  # output1_scale_scalar: MXFP8 uses block scales
        None,  # output1_scale_gate_scalar
        None,  # output2_scale_scalar
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        valid_hidden_size,
        valid_intermediate_size,
        local_expert_offset,
        local_num_experts,
        None,  # routed_scaling_factor
        # Top-k ids and weights are already computed by DeepSeek V4 routing.
        # The source runner has no DeepSeekV4 dispatch yet; Renormalize is a
        # supported precomputed-top-k path and does not recompute the scores.
        _ROUTING_METHOD_RENORMALIZE,
        [int(tactic[0]), int(tactic[1])],
        topk_weights,
        topk_ids,
        output,
    )


__all__ = [
    "has_native_mxfp4_moe",
    "run_native_mxfp4_moe",
    "select_native_mxfp4_moe_tactic",
]
