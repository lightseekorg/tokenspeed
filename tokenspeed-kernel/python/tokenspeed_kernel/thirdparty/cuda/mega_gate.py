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

"""Optional DeepGEMM BF16 fused MoE router GEMM + sqrt-softplus top-k adapter."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from functools import lru_cache

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import pdl_enabled, prepare_cuda_toolkit_env

_MAX_TOKENS = 1 << 20
_WARMUP_BUCKET_CAP = 65536
_MAX_EXPERTS = 512
_MAX_TOPK = 32
_HIDDEN_ALIGNMENT = 256
_EXPERT_ALIGNMENT = 4


def bucket_mega_gate_tokens(num_tokens: int) -> int:
    """Return the canonical warmed batch size for ``num_tokens`` tokens.

    Keeps exact ``M=1`` latency and buckets larger batches to powers of two
    starting at 16 (and 64K multiples above 65,536, within ``_MAX_TOKENS``)
    so arbitrary uncaptured serving batch sizes hit warmed DeepGEMM kernels
    without runtime JIT compilations.
    """
    if num_tokens <= 1:
        return num_tokens
    if num_tokens <= _WARMUP_BUCKET_CAP:
        return max(16, 1 << (num_tokens - 1).bit_length())
    stride = _WARMUP_BUCKET_CAP
    rounded = ((num_tokens + stride - 1) // stride) * stride
    return min(rounded, _MAX_TOKENS)


def mega_gate_warmup_token_counts(max_tokens: int) -> list[int]:
    """Return every canonical ``bf16_mega_gate`` bucket up to ``max_tokens``."""
    if max_tokens <= 0:
        return []
    target_max = bucket_mega_gate_tokens(min(max_tokens, _MAX_TOKENS))
    counts = [1]
    bucket = 16
    while bucket <= min(target_max, _WARMUP_BUCKET_CAP):
        counts.append(bucket)
        bucket *= 2
    if target_max > _WARMUP_BUCKET_CAP:
        counts.extend(range(2 * _WARMUP_BUCKET_CAP, target_max + 1, _WARMUP_BUCKET_CAP))
    return counts


_INIT_LOCK = threading.Lock()
_INITIALIZED_STREAMS: set[tuple[int, int]] = set()
_WORKSPACE_REGION: Callable[[], AbstractContextManager[object]] = nullcontext


@lru_cache(maxsize=1)
def _api():
    prepare_cuda_toolkit_env()
    import deep_gemm

    if not callable(getattr(deep_gemm, "bf16_mega_gate", None)):
        raise ImportError("Installed DeepGEMM does not expose bf16_mega_gate")
    return deep_gemm


@lru_cache(maxsize=1)
def is_bf16_mega_gate_available() -> bool:
    try:
        _api()
    except (ImportError, OSError):
        return False
    return True


def set_mega_gate_workspace_region(
    region: Callable[[], AbstractContextManager[object]],
) -> None:
    """Wrap first-per-stream score-barrier allocation in the memory saver workspace pool.

    DeepGEMM caches one zeroed score-barrier buffer per CUDA stream on the first
    uncaptured ``bf16_mega_gate`` launch on that stream. Scoping that allocation
    inside the runtime's CPU-backed ``workspace`` region prevents it from pinning
    an ordinary Torch CUDA allocator segment after memory release.

    Args:
        region: Context-manager factory for the workspace named region.
    """
    global _WORKSPACE_REGION
    _WORKSPACE_REGION = region


def _call_with_workspace_region(device: torch.device, fn: Callable[[], object]):
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    stream_key = (device_index, int(torch.cuda.current_stream(device).cuda_stream))
    if stream_key in _INITIALIZED_STREAMS:
        return fn()
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "bf16_mega_gate requires an uncaptured launch on this CUDA stream "
            "before graph capture so its per-stream barrier allocates inside "
            "the workspace pool"
        )
    with _INIT_LOCK:
        if stream_key in _INITIALIZED_STREAMS:
            return fn()
        with _WORKSPACE_REGION():
            result = fn()
        _INITIALIZED_STREAMS.add(stream_key)
        return result


def bf16_mega_gate(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool,
    routed_scaling_factor: float,
    ep_rank: int | None,
    correction_bias: torch.Tensor | None = None,
    image_bias: torch.Tensor | None = None,
    image_token_mask: torch.Tensor | None = None,
    to_physical_map: torch.Tensor | None = None,
    logical_count: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    unmapped_topk_idx: torch.Tensor | None = None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fused BF16 router GEMM + normalized sqrt-softplus top-k on SM100.

    Returns ``(topk_weights, topk_ids)`` with ``topk_weights`` in FP32 and
    ``topk_ids`` in INT64.
    """
    if not renormalize:
        raise ValueError("deep_gemm bf16_mega_gate requires normalized top-k weights")
    if hidden_states.ndim != 2 or gate_weight.ndim != 2:
        raise ValueError("hidden_states and gate_weight must be rank-2 tensors")
    num_tokens, hidden = hidden_states.shape
    num_routed_experts, weight_hidden = gate_weight.shape
    if hidden != weight_hidden or hidden <= 0 or hidden % _HIDDEN_ALIGNMENT != 0:
        raise ValueError(
            "bf16_mega_gate requires matching hidden dims divisible by "
            f"{_HIDDEN_ALIGNMENT}, got {hidden} and {weight_hidden}"
        )
    if (
        num_routed_experts <= 0
        or num_routed_experts > _MAX_EXPERTS
        or num_routed_experts % _EXPERT_ALIGNMENT != 0
    ):
        raise ValueError(
            f"bf16_mega_gate requires 1..{_MAX_EXPERTS} experts divisible by "
            f"{_EXPERT_ALIGNMENT}, got {num_routed_experts}"
        )
    if not 0 < top_k <= min(num_routed_experts, _MAX_TOPK):
        raise ValueError(
            f"bf16_mega_gate requires top_k in [1, {min(num_routed_experts, _MAX_TOPK)}], got {top_k}"
        )
    if num_tokens > _MAX_TOKENS:
        raise ValueError(
            f"bf16_mega_gate supports at most {_MAX_TOKENS} tokens, got {num_tokens}"
        )
    if to_physical_map is not None and ep_rank is None:
        raise ValueError("ep_rank is required when to_physical_map is provided")
    if ep_rank is not None and ep_rank < 0:
        raise ValueError(f"ep_rank must be non-negative, got {ep_rank}")
    if not math.isfinite(routed_scaling_factor):
        raise ValueError("routed_scaling_factor must be finite")
    if (image_bias is None) != (image_token_mask is None):
        raise ValueError("image_bias and image_token_mask must be supplied together")
    if (to_physical_map is None) != (logical_count is None):
        raise ValueError("to_physical_map and logical_count must be supplied together")

    device = hidden_states.device
    for tensor in (hidden_states, gate_weight):
        if (
            tensor.dtype != torch.bfloat16
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "hidden_states and gate_weight must be contiguous BF16 tensors on one CUDA device"
            )

    if image_token_mask is not None and (
        image_token_mask.ndim != 1
        or image_token_mask.shape[0] != num_tokens
        or image_token_mask.device != device
    ):
        raise ValueError(
            f"image_token_mask must be 1D with length {num_tokens} on {device}"
        )
    if mask is not None and (
        mask.ndim != 1 or mask.shape[0] != num_tokens or mask.device != device
    ):
        raise ValueError(f"mask must be 1D with length {num_tokens} on {device}")
    if unmapped_topk_idx is not None and (
        unmapped_topk_idx.shape != (num_tokens, top_k)
        or unmapped_topk_idx.device != device
    ):
        raise ValueError(
            f"unmapped_topk_idx must have shape [{num_tokens}, {top_k}] on {device}"
        )

    out_arg: tuple[torch.Tensor, torch.Tensor] | None = None
    if out is not None:
        out_weights, out_idx = out
        if (
            out_weights.shape != (num_tokens, top_k)
            or out_weights.dtype != torch.float32
            or not out_weights.is_contiguous()
            or out_weights.device != device
        ):
            raise ValueError(
                f"out weights must be contiguous FP32 [{num_tokens}, {top_k}] on {device}"
            )
        if (
            out_idx.shape != (num_tokens, top_k)
            or out_idx.dtype != torch.int64
            or not out_idx.is_contiguous()
            or out_idx.device != device
        ):
            raise ValueError(
                f"out expert IDs must be contiguous INT64 [{num_tokens}, {top_k}] on {device}"
            )
        out_arg = (out_idx, out_weights)

    if num_tokens == 0:
        if out is not None:
            return out
        return (
            torch.empty((0, top_k), dtype=torch.float32, device=device),
            torch.empty((0, top_k), dtype=torch.int64, device=device),
        )

    padded_tokens = bucket_mega_gate_tokens(num_tokens)
    api = _api()
    if api.get_pdl() != pdl_enabled():
        api.set_pdl(pdl_enabled())

    def _launch():
        if padded_tokens == num_tokens:
            return api.bf16_mega_gate(
                x=hidden_states,
                weight=gate_weight,
                num_topk=int(top_k),
                use_shared_as_routed=False,
                num_shared_experts=0,
                routed_scaling_factor=float(routed_scaling_factor),
                ep_rank=0 if ep_rank is None else int(ep_rank),
                scoring_func="sqrtsoftplus",
                mask=mask,
                bias=correction_bias,
                image_bias=image_bias,
                image_token_mask=image_token_mask,
                fix_routing_mask=None,
                to_physical_map=to_physical_map,
                logical_count=logical_count,
                unmapped_topk_idx=unmapped_topk_idx,
                force_random=None,
                out=out_arg,
            )
        # Pad only to canonical warmed buckets; keep mask=None when the caller
        # passed mask=None because DeepGEMM specializes on mask presence.
        extra = padded_tokens - num_tokens
        hidden_in = F.pad(hidden_states, (0, 0, 0, extra))
        image_mask_in = (
            F.pad(image_token_mask, (0, extra), value=False)
            if image_token_mask is not None
            else None
        )
        mask_in = F.pad(mask, (0, extra), value=False) if mask is not None else None
        unmapped_in = (
            torch.empty(
                (padded_tokens, top_k),
                dtype=unmapped_topk_idx.dtype,
                device=device,
            )
            if unmapped_topk_idx is not None
            else None
        )
        padded_idx, padded_weights = api.bf16_mega_gate(
            x=hidden_in,
            weight=gate_weight,
            num_topk=int(top_k),
            use_shared_as_routed=False,
            num_shared_experts=0,
            routed_scaling_factor=float(routed_scaling_factor),
            ep_rank=0 if ep_rank is None else int(ep_rank),
            scoring_func="sqrtsoftplus",
            mask=mask_in,
            bias=correction_bias,
            image_bias=image_bias,
            image_token_mask=image_mask_in,
            fix_routing_mask=None,
            to_physical_map=to_physical_map,
            logical_count=logical_count,
            unmapped_topk_idx=unmapped_in,
            force_random=None,
            out=None,
        )
        sliced_idx = padded_idx[:num_tokens]
        sliced_weights = padded_weights[:num_tokens]
        if unmapped_topk_idx is not None and unmapped_in is not None:
            unmapped_topk_idx.copy_(unmapped_in[:num_tokens])
        if out is not None:
            out_weights, out_idx = out
            out_weights.copy_(sliced_weights)
            out_idx.copy_(sliced_idx)
            return out_idx, out_weights
        return sliced_idx, sliced_weights

    topk_idx, topk_weights = _call_with_workspace_region(device, _launch)
    return topk_weights, topk_idx
