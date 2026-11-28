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
from tokenspeed_kernel.platform import pdl_enabled, prepare_cuda_toolkit_env

_MAX_TOKENS = 1 << 20
_MAX_EXPERTS = 512
_MAX_TOPK = 32
_HIDDEN_ALIGNMENT = 256
_EXPERT_ALIGNMENT = 4
_KERNEL_EXPERT_ALIGNMENT = 128
_KERNEL_BLOCK_K = 64
_KERNEL_MIN_BLOCK_TOKENS = 16
_KERNEL_MAX_BLOCK_TOKENS = 256
_KERNEL_MAX_LOGICAL_CTAS = 64
_KERNEL_MAX_GATE_WARPGROUPS = 7
_KERNEL_TRANSPOSE_COLS_PER_WG = 8
_KERNEL_TOPK_TOKENS_PER_WG = 4


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align(a: int, b: int) -> int:
    return _cdiv(a, b) * b


def _select_wave_block_tokens(num_tokens: int, wg: int) -> int:
    waves = _cdiv(_cdiv(num_tokens, _KERNEL_MAX_BLOCK_TOKENS), wg)
    return min(
        _KERNEL_MAX_BLOCK_TOKENS,
        _align(_cdiv(num_tokens, waves * wg), _KERNEL_MIN_BLOCK_TOKENS),
    )


def _get_num_waves(num_tokens: int, wg: int) -> int:
    block_tokens = _select_wave_block_tokens(num_tokens, wg)
    return _cdiv(_cdiv(num_tokens, block_tokens), wg)


def _mega_gate_candidates(
    hidden: int, num_experts: int, num_sms: int, deterministic: bool
) -> list[tuple[int, int, int, int, int, int]]:
    num_aligned_experts = _align(num_experts, _KERNEL_EXPERT_ALIGNMENT)
    nsk_choices = (1,) if deterministic else (1, 2, 4, 8)
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for neg in range(1, num_aligned_experts // _KERNEL_EXPERT_ALIGNMENT + 1):
        if num_aligned_experts % (_KERNEL_EXPERT_ALIGNMENT * neg) != 0:
            continue
        nepg = num_aligned_experts // neg
        if nepg > 256:
            continue
        for nmma in (1, 2):
            if nepg > 128 and nmma != 2:
                continue
            for nsk in nsk_choices:
                if hidden % (nsk * _KERNEL_BLOCK_K) != 0 or (nmma == 2 and nsk == 8):
                    continue
                nlc = neg * nmma * nsk
                if nlc > num_sms or nlc >= _KERNEL_MAX_LOGICAL_CTAS:
                    continue
                candidates.append((neg, nepg, nmma, nsk, nlc, num_sms // nlc))
    return candidates


def _mega_gate_heuristic_key(
    num_tokens: int,
    candidates: list[tuple[int, int, int, int, int, int]],
    num_sms: int,
    deterministic: bool,
) -> tuple[int, int, int, int] | None:
    best_score: tuple[int, int, int, int, int] | None = None
    best: tuple[int, int, int, int, int, int, int] | None = None
    for neg, nepg, nmma, nsk, nlc, wg in candidates:
        block_tokens = _select_wave_block_tokens(num_tokens, wg)
        num_token_blocks = _cdiv(num_tokens, block_tokens)
        launch_sms = min(wg, num_token_blocks) * nlc
        waves = _cdiv(num_token_blocks, wg)
        token_cols = block_tokens // nmma if nepg == 128 else block_tokens
        cols = min(token_cols, 56)

        doubling_nlc = nlc * 2
        doubling_free = False
        if doubling_nlc <= num_sms and doubling_nlc < _KERNEL_MAX_LOGICAL_CTAS:
            doubling_wg = num_sms // doubling_nlc
            if nmma == 1 and nsk != 8:
                doubling_free = _get_num_waves(num_tokens, doubling_wg) <= waves + 2
            elif nmma == 2 and nsk == 1 and waves == 1 and not deterministic:
                doubling_free = (
                    _select_wave_block_tokens(num_tokens, doubling_wg) == block_tokens
                )
        if doubling_free:
            continue

        score = (cols, launch_sms, nmma, -nsk, -waves)
        if best_score is None or score > best_score:
            best_score = score
            best = (neg, nepg, nmma, nsk, nlc, block_tokens, waves)

    if best is None:
        return None
    neg, nepg, nmma, nsk, nlc, block_tokens, waves = best
    token_cols = block_tokens // nmma if nepg == 128 else block_tokens
    ntu = _cdiv(token_cols, _KERNEL_TRANSPOSE_COLS_PER_WG)
    tku = _cdiv(block_tokens, _KERNEL_TOPK_TOKENS_PER_WG * nlc)

    def _balanced_wg(units: int) -> int:
        return _cdiv(units, _cdiv(units, _KERNEL_MAX_GATE_WARPGROUPS))

    ngw = (
        _KERNEL_MAX_GATE_WARPGROUPS
        if waves > 1
        else max(_balanced_wg(ntu), _balanced_wg(tku))
    )
    return (ngw * 128, nmma, nsk, neg)


def _mega_gate_transition_points(
    max_tokens: int,
    candidates: list[tuple[int, int, int, int, int, int]],
    num_sms: int,
) -> list[int]:
    wave_groups = sorted(
        {wg for *_, wg in candidates}
        | {
            num_sms // (nlc * 2)
            for *_, nlc, _ in candidates
            if nlc * 2 <= num_sms and nlc * 2 < _KERNEL_MAX_LOGICAL_CTAS
        }
    )
    points: set[int] = {1, max_tokens}

    def _add_pair(token_count: int) -> None:
        if 1 <= token_count <= max_tokens:
            points.add(token_count)
        if 1 <= token_count - 1 <= max_tokens:
            points.add(token_count - 1)

    for wg in wave_groups:
        step = _KERNEL_MAX_BLOCK_TOKENS * wg
        max_waves = _cdiv(max_tokens, step)
        for wave_idx in range(max_waves):
            base = wave_idx * step
            _add_pair(base + 1)
            waves = wave_idx + 1
            inner = _KERNEL_MIN_BLOCK_TOKENS * waves * wg
            m = base // inner
            limit = min(max_tokens, base + step) + 1
            while True:
                t = m * inner + 1
                if t > limit:
                    break
                _add_pair(t)
                m += 1
            for block_step in range(
                _KERNEL_MIN_BLOCK_TOKENS,
                _KERNEL_MAX_BLOCK_TOKENS + 1,
                _KERNEL_MIN_BLOCK_TOKENS,
            ):
                if waves == 1:
                    for mult in range(1, wg + 2):
                        t = mult * block_step + 1
                        if t > min(max_tokens, step) + 1:
                            break
                        _add_pair(t)
                _add_pair(waves * wg * block_step + 1)
    return sorted(points)


def mega_gate_warmup_token_counts(
    max_tokens: int,
    *,
    hidden: int,
    num_experts: int,
    num_sms: int,
    deterministic: bool = False,
) -> list[int]:
    """Return representative token counts covering SM100 MegaGate JIT configs.

    ``sm100_bf16_mega_gate`` passes ``num_tokens`` as a runtime ``uint32`` and
    specializes only on the heuristic tuple ``(num_gate_threads, num_mma_ctas,
    num_split_k, num_expert_groups)``. Mirroring the C++ heuristic transition
    points lets warmup compile all distinct templates in ``<= 15`` launches so
    arbitrary serving batch sizes execute unpadded with zero runtime JITs.
    """
    if hidden <= 0 or hidden % _HIDDEN_ALIGNMENT != 0:
        raise ValueError(
            f"hidden must be a positive multiple of {_HIDDEN_ALIGNMENT}, got {hidden}"
        )
    if (
        num_experts <= 0
        or num_experts > _MAX_EXPERTS
        or num_experts % _EXPERT_ALIGNMENT != 0
    ):
        raise ValueError(
            f"num_experts must be in 1..{_MAX_EXPERTS} divisible by {_EXPERT_ALIGNMENT}, got {num_experts}"
        )
    if num_sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    if max_tokens <= 0:
        return []
    limit = min(max_tokens, _MAX_TOKENS)
    candidates = _mega_gate_candidates(hidden, num_experts, num_sms, deterministic)
    if not candidates:
        return [1]
    seen_configs: dict[tuple[int, int, int, int], int] = {}
    for token_count in _mega_gate_transition_points(limit, candidates, num_sms):
        key = _mega_gate_heuristic_key(token_count, candidates, num_sms, deterministic)
        if key is not None and key not in seen_configs:
            seen_configs[key] = token_count
    return list(seen_configs.values()) or [1]


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
        or unmapped_topk_idx.dtype != torch.int64
        or unmapped_topk_idx.stride(1) != 1
        or unmapped_topk_idx.device != device
    ):
        raise ValueError(
            f"unmapped_topk_idx must have INT64 shape [{num_tokens}, {top_k}] "
            f"with unit trailing stride on {device}"
        )
    if to_physical_map is not None and (
        to_physical_map.ndim != 2
        or to_physical_map.shape[0] < num_routed_experts
        or to_physical_map.shape[1] <= 0
        or to_physical_map.dtype != torch.int32
        or not to_physical_map.is_contiguous()
        or to_physical_map.device != device
    ):
        raise ValueError(
            f"to_physical_map must be contiguous INT32 [>={num_routed_experts}, >=1] "
            f"on {device}"
        )
    if logical_count is not None and (
        to_physical_map is None
        or logical_count.shape != (to_physical_map.shape[0],)
        or logical_count.dtype != torch.int32
        or not logical_count.is_contiguous()
        or logical_count.device != device
    ):
        raise ValueError(
            f"logical_count must be contiguous INT32 matching to_physical_map rows on {device}"
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

    api = _api()
    if api.get_pdl() != pdl_enabled():
        api.set_pdl(pdl_enabled())

    def _launch():
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

    topk_idx, topk_weights = _call_with_workspace_region(device, _launch)
    return topk_weights, topk_idx
