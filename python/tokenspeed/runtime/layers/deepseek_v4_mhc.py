# Copyright (c) 2026 LightSeek Foundation
#
# Portions copyright the vLLM project contributors under Apache-2.0.

from __future__ import annotations

import logging
from functools import cache

import torch
import triton
import triton.language as tl
from tokenspeed_kernel.ops.mhc import (
    deep_gemm_mhc_prenorm_gemm,
    has_deep_gemm_mhc,
    supports_trtllm_mhc,
    trtllm_mhc_big_fuse,
    trtllm_mhc_fused_hc,
    trtllm_mhc_post_mapping,
)

from tokenspeed.runtime.utils import ceil_div
from tokenspeed.runtime.utils.env import envs

_MHC_BACKEND_ENV = "TOKENSPEED_V4_MHC_BACKEND"
_MHC_BACKEND_SELECTIONS_LOGGED: set[tuple[str, bool, bool, torch.device, int, int]] = (
    set()
)
logger = logging.getLogger(__name__)


def _mhc_backend() -> str:
    backend = envs.TOKENSPEED_V4_MHC_BACKEND.get().strip().lower()
    if backend not in {"auto", "native", "trtllm"}:
        raise ValueError(
            f"{_MHC_BACKEND_ENV}={backend!r} must be one of " "'auto'|'native'|'trtllm'"
        )
    return backend


def _validate_mhc_backend_config(
    device: torch.device,
    hc_mult: int,
    hidden_size: int,
) -> None:
    """Validate static mHC backend configuration before model construction."""

    backend = _mhc_backend()
    if backend == "trtllm" and not supports_trtllm_mhc(device, hc_mult, hidden_size):
        raise RuntimeError(
            "TRT-LLM mHC was requested but is unavailable for "
            f"device={device}, hc_mult={hc_mult}, hidden_size={hidden_size}"
        )


def _use_trtllm_mhc(
    device: torch.device,
    hc_mult: int,
    hidden_size: int,
    allow_trtllm_auto: bool = False,
) -> bool:
    """Resolve the experimental V4 mHC backend for one supported tensor shape."""

    backend = _mhc_backend()
    supported: bool | None = None
    if backend == "native":
        selected = False
    elif backend == "auto" and not allow_trtllm_auto:
        selected = False
    else:
        supported = supports_trtllm_mhc(device, hc_mult, hidden_size)
        if backend == "trtllm" and not supported:
            raise RuntimeError(
                "TRT-LLM mHC was requested but is unavailable for "
                f"device={device}, hc_mult={hc_mult}, hidden_size={hidden_size}"
            )
        selected = supported

    selection = (
        backend,
        allow_trtllm_auto,
        selected,
        device,
        hc_mult,
        hidden_size,
    )
    if selection not in _MHC_BACKEND_SELECTIONS_LOGGED:
        _MHC_BACKEND_SELECTIONS_LOGGED.add(selection)
        logger.info(
            "V4 mHC backend selection: requested=%s selected=%s "
            "allow_trtllm_auto=%s supported=%s device=%s hc_mult=%d hidden_size=%d",
            backend,
            "trtllm" if selected else "native",
            allow_trtllm_auto,
            supported if supported is not None else "not-probed",
            device,
            hc_mult,
            hidden_size,
        )
    return selected


@cache
def _compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    split_k = device_props.multi_processor_count // grid_size
    if k is not None:
        num_block_k = ceil_div(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


@triton.jit
def _load_reduced_mix(
    gemm_out_mul,
    token_id,
    mix_id: tl.constexpr,
    num_tokens,
    hc_mult3: tl.constexpr,
    n_splits: tl.constexpr,
):
    value = tl.full((), 0.0, tl.float32)
    for split_id in tl.static_range(0, n_splits):
        offset = split_id * num_tokens * hc_mult3 + token_id * hc_mult3 + mix_id
        value += tl.load(gemm_out_mul + offset)
    return value


@triton.jit
def _mhc_pre_mix_triton_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    pre_mix,
    post_mix,
    comb_mix,
    hidden_size: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    n_splits: tl.constexpr,
    hc_mult: tl.constexpr,
    hc_mult2: tl.constexpr,
    hc_mult3: tl.constexpr,
    block_comb: tl.constexpr,
    num_tokens,
):
    token_id = tl.program_id(0)

    rms_sum = tl.full((), 0.0, tl.float32)
    for split_id in tl.static_range(0, n_splits):
        rms_sum += tl.load(gemm_out_sqrsum + split_id * num_tokens + token_id)
    rms = tl.rsqrt(rms_sum / (hc_mult * hidden_size) + rms_eps)

    pre_scale = tl.load(hc_scale)
    for hc_id in tl.static_range(0, hc_mult):
        mix = _load_reduced_mix(
            gemm_out_mul,
            token_id,
            hc_id,
            num_tokens,
            hc_mult3,
            n_splits,
        )
        pre = tl.sigmoid(mix * rms * pre_scale + tl.load(hc_base + hc_id)) + hc_eps
        tl.store(pre_mix + token_id * hc_mult + hc_id, pre)

    post_scale = tl.load(hc_scale + 1)
    for hc_id in tl.static_range(0, hc_mult):
        mix = _load_reduced_mix(
            gemm_out_mul,
            token_id,
            hc_mult + hc_id,
            num_tokens,
            hc_mult3,
            n_splits,
        )
        post = (
            tl.sigmoid(mix * rms * post_scale + tl.load(hc_base + hc_mult + hc_id))
            * 2.0
        )
        tl.store(post_mix + token_id * hc_mult + hc_id, post)

    comb_offsets = tl.arange(0, block_comb)
    comb_mask = comb_offsets < hc_mult2
    comb_scale = tl.load(hc_scale + 2)
    comb_mix_values = tl.zeros((block_comb,), tl.float32)
    for split_id in tl.static_range(0, n_splits):
        split_base = split_id * num_tokens * hc_mult3 + token_id * hc_mult3
        comb_mix_values += tl.load(
            gemm_out_mul + split_base + hc_mult * 2 + comb_offsets,
            mask=comb_mask,
            other=0.0,
        )
    comb_values = comb_mix_values * rms * comb_scale + tl.load(
        hc_base + hc_mult * 2 + comb_offsets, mask=comb_mask, other=0.0
    )
    rows = comb_offsets // hc_mult
    cols = comb_offsets - rows * hc_mult
    active = comb_mask

    for row_id in tl.static_range(0, hc_mult):
        row_values = tl.where((rows == row_id) & active, comb_values, -float("inf"))
        row_max = tl.max(row_values, axis=0)
        comb_values = tl.where(
            (rows == row_id) & active, tl.exp(comb_values - row_max), comb_values
        )
    for row_id in tl.static_range(0, hc_mult):
        row_sum = tl.sum(tl.where((rows == row_id) & active, comb_values, 0.0), axis=0)
        comb_values = tl.where(
            (rows == row_id) & active, comb_values / row_sum + hc_eps, comb_values
        )
    for col_id in tl.static_range(0, hc_mult):
        col_sum = tl.sum(tl.where((cols == col_id) & active, comb_values, 0.0), axis=0)
        comb_values = tl.where(
            (cols == col_id) & active,
            comb_values / (col_sum + hc_eps),
            comb_values,
        )

    for _ in tl.static_range(1, sinkhorn_iters):
        for row_id in tl.static_range(0, hc_mult):
            row_sum = tl.sum(
                tl.where((rows == row_id) & active, comb_values, 0.0), axis=0
            )
            comb_values = tl.where(
                (rows == row_id) & active,
                comb_values / (row_sum + hc_eps),
                comb_values,
            )
        for col_id in tl.static_range(0, hc_mult):
            col_sum = tl.sum(
                tl.where((cols == col_id) & active, comb_values, 0.0), axis=0
            )
            comb_values = tl.where(
                (cols == col_id) & active,
                comb_values / (col_sum + hc_eps),
                comb_values,
            )

    tl.store(
        comb_mix + token_id * hc_mult2 + comb_offsets,
        comb_values,
        mask=comb_mask,
    )


@triton.jit
def _mhc_pre_layer_triton_kernel(
    pre_mix,
    residual,
    layer_input,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    block_h: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_block_id = tl.program_id(1)

    hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size
    layer_acc = tl.zeros((block_h,), tl.float32)
    for hc_id in tl.static_range(0, hc_mult):
        pre = tl.load(pre_mix + token_id * hc_mult + hc_id).to(tl.float32)
        residual_offsets = (
            token_id * hc_mult * hidden_size + hc_id * hidden_size + hidden_offsets
        )
        residual_values = tl.load(
            residual + residual_offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32)
        layer_acc += pre * residual_values
    tl.store(
        layer_input + token_id * hidden_size + hidden_offsets,
        layer_acc,
        mask=hidden_mask,
    )


@triton.jit
def _mhc_post_triton_kernel(
    comb,
    residual,
    post,
    hidden_states,
    out,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    block_h: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_block_id = tl.program_id(1)
    hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size
    hidden_values = tl.load(
        hidden_states + token_id * hidden_size + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)

    for out_hc in tl.static_range(0, hc_mult):
        acc = tl.load(post + token_id * hc_mult + out_hc).to(tl.float32) * hidden_values
        for in_hc in tl.static_range(0, hc_mult):
            comb_value = tl.load(
                comb + token_id * hc_mult * hc_mult + in_hc * hc_mult + out_hc
            ).to(tl.float32)
            residual_values = tl.load(
                residual
                + token_id * hc_mult * hidden_size
                + in_hc * hidden_size
                + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            acc += comb_value * residual_values
        tl.store(
            out
            + token_id * hc_mult * hidden_size
            + out_hc * hidden_size
            + hidden_offsets,
            acc,
            mask=hidden_mask,
        )


@triton.jit
def _mhc_post_hc4_triton_kernel(
    comb,
    residual,
    post,
    hidden_states,
    out,
    hidden_size: tl.constexpr,
    block_h: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_block_id = tl.program_id(1)
    hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size
    token_hidden_offset = token_id * hidden_size
    token_residual_offset = token_id * 4 * hidden_size

    hidden_values = tl.load(
        hidden_states + token_hidden_offset + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)

    post_base = token_id * 4
    acc0 = tl.load(post + post_base + 0).to(tl.float32) * hidden_values
    acc1 = tl.load(post + post_base + 1).to(tl.float32) * hidden_values
    acc2 = tl.load(post + post_base + 2).to(tl.float32) * hidden_values
    acc3 = tl.load(post + post_base + 3).to(tl.float32) * hidden_values

    comb_base = token_id * 16
    for in_hc in tl.static_range(0, 4):
        residual_values = tl.load(
            residual + token_residual_offset + in_hc * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        comb_row = comb_base + in_hc * 4
        acc0 += tl.load(comb + comb_row + 0).to(tl.float32) * residual_values
        acc1 += tl.load(comb + comb_row + 1).to(tl.float32) * residual_values
        acc2 += tl.load(comb + comb_row + 2).to(tl.float32) * residual_values
        acc3 += tl.load(comb + comb_row + 3).to(tl.float32) * residual_values

    tl.store(
        out + token_residual_offset + hidden_offsets,
        acc0,
        mask=hidden_mask,
    )
    tl.store(
        out + token_residual_offset + hidden_size + hidden_offsets,
        acc1,
        mask=hidden_mask,
    )
    tl.store(
        out + token_residual_offset + hidden_size * 2 + hidden_offsets,
        acc2,
        mask=hidden_mask,
    )
    tl.store(
        out + token_residual_offset + hidden_size * 3 + hidden_offsets,
        acc3,
        mask=hidden_mask,
    )


class _FusedHcPingPong:
    """Double-buffered workspace for TRT-LLM fused mHC.

    Two buffer sets alternate each call so consecutive layers never alias
    their input (previous output) with the current output.
    """

    def __init__(
        self, max_bs: int, hc_mult: int, hidden_size: int, device: torch.device
    ):
        n2 = hc_mult * hc_mult
        shape_n = hc_mult * (2 + hc_mult)
        self.bufs = tuple(
            (
                torch.empty(
                    max_bs, hc_mult, hidden_size, dtype=torch.bfloat16, device=device
                ),
                torch.empty(max_bs, hc_mult, dtype=torch.float32, device=device),
                torch.empty(max_bs, n2, dtype=torch.float32, device=device),
                torch.empty(max_bs, hidden_size, dtype=torch.bfloat16, device=device),
                torch.empty(max_bs, shape_n, dtype=torch.float32, device=device),
                torch.empty(max_bs, dtype=torch.float32, device=device),
                torch.empty(max_bs, dtype=torch.int32, device=device),
            )
            for _ in range(2)
        )
        self.idx = 0

    def reset(self):
        self.idx = 0

    def get(self):
        buf = self.bufs[self.idx]
        self.idx ^= 1
        return buf


class MhcFusedWorkspace:
    """Model-owned fused-mHC buffers for serial CUDA execution.

    Captured and eager forwards use separate pools so a large eager prefill
    cannot replace storage referenced by a decode graph. Captured buffers grow
    by powers of two, and superseded allocations stay alive for the lifetime
    of this workspace because existing graphs retain their raw addresses.

    Returned fused-mHC tensors borrow one of two alternating buffer sets. A
    caller must consume each state before the same workspace wraps around and
    reuses that set. CUDA graph replays must be serialized on the model stream.
    """

    def __init__(self) -> None:
        self._captured: dict[tuple[torch.device, int, int], _FusedHcPingPong] = {}
        self._retired_captured: list[_FusedHcPingPong] = []
        self._eager: dict[tuple[torch.device, int, int], _FusedHcPingPong] = {}

    def reset(self) -> None:
        """Start a serial model forward at the first ping-pong buffer.

        Eager buffers from the previous forward are released back to PyTorch's
        stream-aware allocator. Captured buffers remain owned for graph replay.

        Returns:
            None.
        """

        for pp in self._captured.values():
            pp.reset()
        self._eager.clear()

    def get(
        self,
        num_tokens: int,
        hc_mult: int,
        hidden_size: int,
        device: torch.device,
    ) -> _FusedHcPingPong:
        """Return the ping-pong allocation for the current execution mode.

        Args:
            num_tokens: Flattened token count required by this forward.
            hc_mult: Number of hyper-connection lanes.
            hidden_size: Per-lane hidden dimension.
            device: CUDA device that owns the buffers.

        Returns:
            A model-owned double-buffer allocation with sufficient capacity.
        """

        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        key = (device, hc_mult, hidden_size)
        cache = self._captured if capturing else self._eager
        pp = cache.get(key)
        if pp is None or pp.bufs[0][0].shape[0] < num_tokens:
            if capturing and pp is not None:
                self._retired_captured.append(pp)
            capacity = num_tokens
            if capturing:
                capacity = 1 << (max(num_tokens, 1) - 1).bit_length()
            pp = _FusedHcPingPong(capacity, hc_mult, hidden_size, device)
            cache[key] = pp
        return pp


def mhc_fused_hc(
    x_prev: torch.Tensor,
    residual_prev: torch.Tensor,
    post_prev: torch.Tensor,
    comb_prev: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    workspace: MhcFusedWorkspace,
    *,
    allow_trtllm_auto: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused post_mapping(prev) + pre_mapping(curr).

    Args:
        x_prev: Previous sublayer output.
        residual_prev: Previous hyper-connected residual.
        post_prev: Previous post-mapping weights.
        comb_prev: Previous lane-combination matrices.
        fn: Current pre-mapping projection weight.
        hc_scale: Current pre/post/comb scales.
        hc_base: Current pre/post/comb biases.
        rms_eps: RMS normalization epsilon.
        hc_eps: Hyper-connection normalization epsilon.
        sinkhorn_iters: Number of Sinkhorn normalization iterations.
        workspace: Model-owned serial fused-mHC workspace.
        allow_trtllm_auto: Allow the experimental backend when mode is `auto`.

    Returns:
        ``(residual_cur, layer_input, post_cur, comb_cur)``. The TRT-LLM path
        returns borrowed workspace views that must be consumed before the same
        ping-pong set is reused.
    """
    hc_mult = residual_prev.shape[-2]
    hidden_size = residual_prev.shape[-1]
    if (
        x_prev.dtype != torch.bfloat16
        or residual_prev.dtype != torch.bfloat16
        or fn.dtype != torch.float32
        or hc_scale.dtype != torch.float32
        or hc_base.dtype != torch.float32
        or post_prev.dtype != torch.float32
        or comb_prev.dtype != torch.float32
    ):
        raise RuntimeError("fast mHC requires bf16 states and fp32 weights/mixes")
    if _use_trtllm_mhc(residual_prev.device, hc_mult, hidden_size, allow_trtllm_auto):
        return _trtllm_mhc_fused_hc(
            x_prev,
            residual_prev,
            post_prev,
            comb_prev,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            workspace,
        )
    residual_cur = mhc_post(
        x_prev,
        residual_prev,
        post_prev,
        comb_prev,
        allow_trtllm_auto=allow_trtllm_auto,
    )
    layer_input, post_cur, comb_cur = mhc_pre(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        allow_trtllm_auto=allow_trtllm_auto,
    )
    return residual_cur, layer_input, post_cur, comb_cur


def _trtllm_mhc_fused_hc(
    x_prev: torch.Tensor,
    residual_prev: torch.Tensor,
    post_prev: torch.Tensor,
    comb_prev: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    workspace: MhcFusedWorkspace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual_prev.shape[-2]
    hidden_size = residual_prev.shape[-1]
    outer_shape = residual_prev.shape[:-2]
    B = residual_prev[..., 0, 0].numel()
    if B == 0:
        return (
            torch.empty_like(residual_prev),
            residual_prev.new_empty(*outer_shape, hidden_size),
            torch.empty(
                *outer_shape,
                hc_mult,
                1,
                dtype=torch.float32,
                device=residual_prev.device,
            ),
            torch.empty(
                *outer_shape,
                hc_mult,
                hc_mult,
                dtype=torch.float32,
                device=residual_prev.device,
            ),
        )

    x_flat = x_prev.reshape(B, hidden_size).contiguous()
    res_flat = residual_prev.reshape(B, hc_mult, hidden_size).contiguous()
    post_flat = post_prev.reshape(B, hc_mult).float().contiguous()
    comb_flat = comb_prev.reshape(B, hc_mult, hc_mult).float().contiguous()

    pp = workspace.get(B, hc_mult, hidden_size, x_prev.device)
    residual_cur, post_cur, comb_cur, layer_input, y_acc, r_acc, dc = pp.get()

    trtllm_mhc_fused_hc(
        x_flat,
        res_flat,
        post_flat,
        comb_flat,
        fn.contiguous(),
        hc_scale.contiguous(),
        hc_base.contiguous(),
        residual_cur,
        post_cur,
        comb_cur,
        layer_input,
        y_acc,
        r_acc,
        dc,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )

    return (
        residual_cur[:B].view(*outer_shape, hc_mult, hidden_size),
        layer_input[:B].view(*outer_shape, hidden_size),
        post_cur[:B].view(*outer_shape, hc_mult, 1),
        comb_cur[:B].view(*outer_shape, hc_mult, hc_mult),
    )


def _trtllm_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_dim = hc_mult * hidden_size
    mix_hc = fn.shape[0]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return (
            residual.new_empty(*outer_shape, hidden_size),
            torch.empty(
                *outer_shape, hc_mult, 1, dtype=torch.float32, device=residual.device
            ),
            torch.empty(
                *outer_shape,
                hc_mult,
                hc_mult,
                dtype=torch.float32,
                device=residual.device,
            ),
        )
    block_k = 64
    block_m = 64
    # mhc_big_fuse only supports split-K counts in {1, 2, 4, 8, 16}; floor the
    # heuristic value to the nearest power of two (split-K is a reduction, so
    # fewer splits is numerically identical).
    raw_splits = min(
        _compute_num_split(block_k, hc_dim, ceil_div(num_tokens, block_m)), 16
    )
    n_splits = 1 << (raw_splits.bit_length() - 1)
    y_acc = torch.empty(
        n_splits, num_tokens, mix_hc, dtype=torch.float32, device=residual.device
    )
    r_acc = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    deep_gemm_mhc_prenorm_gemm(
        residual_flat.reshape(num_tokens, hc_dim),
        fn,
        y_acc,
        r_acc,
        n_splits,
    )
    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    trtllm_mhc_big_fuse(
        y_acc,
        r_acc,
        residual_flat.contiguous(),
        hc_scale.float().contiguous(),
        hc_base.float().contiguous(),
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )
    return (
        layer_input.view(*outer_shape, hidden_size),
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
    )


def _trtllm_mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size).contiguous()
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return torch.empty_like(residual)
    x_flat = hidden_states.view(num_tokens, hidden_size).contiguous()
    post_flat = post.view(num_tokens, hc_mult).float().contiguous()
    comb_flat = comb.view(num_tokens, hc_mult, hc_mult).float().contiguous()
    out = torch.empty(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    trtllm_mhc_post_mapping(
        residual_flat,
        x_flat,
        post_flat,
        comb_flat,
        out,
    )
    return out.view(*outer_shape, hc_mult, hidden_size)


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    *,
    allow_trtllm_auto: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if (
        residual.dtype != torch.bfloat16
        or fn.dtype != torch.float32
        or hc_scale.dtype != torch.float32
        or hc_base.dtype != torch.float32
    ):
        raise RuntimeError("fast mHC requires bf16 residual and fp32 weights")
    if _use_trtllm_mhc(residual.device, hc_mult, hidden_size, allow_trtllm_auto):
        return _trtllm_mhc_pre(
            residual, fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters
        )
    if not residual.is_cuda:
        raise RuntimeError("fast mHC requires CUDA tensors")

    if not has_deep_gemm_mhc():
        raise RuntimeError("deep_gemm.tf32_hc_prenorm_gemm is unavailable")

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return (
            residual.new_empty(*outer_shape, hidden_size),
            torch.empty(
                *outer_shape,
                hc_mult,
                1,
                dtype=torch.float32,
                device=residual.device,
            ),
            torch.empty(
                *outer_shape,
                hc_mult,
                hc_mult,
                dtype=torch.float32,
                device=residual.device,
            ),
        )

    block_k = 64
    block_m = 64
    n_splits = _compute_num_split(
        block_k, hc_hidden_size, ceil_div(num_tokens, block_m)
    )

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    pre_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    deep_gemm_mhc_prenorm_gemm(
        residual_flat.view(num_tokens, hc_hidden_size),
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )
    block_h = 1024
    block_comb = triton.next_power_of_2(hc_mult2)
    _mhc_pre_mix_triton_kernel[(num_tokens,)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        n_splits=n_splits,
        hc_mult=hc_mult,
        hc_mult2=hc_mult2,
        hc_mult3=hc_mult3,
        block_comb=block_comb,
        num_tokens=num_tokens,
        num_warps=1,
    )
    _mhc_pre_layer_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        pre_mix,
        residual_flat,
        layer_input,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        block_h=block_h,
        num_warps=4,
    )

    return (
        layer_input.view(*outer_shape, hidden_size),
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
    )


def mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    allow_trtllm_auto: bool = False,
) -> torch.Tensor:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    if (
        hidden_states.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or post.dtype != torch.float32
        or comb.dtype != torch.float32
    ):
        raise RuntimeError("fast mHC requires bf16 states and fp32 mixes")
    if _use_trtllm_mhc(residual.device, hc_mult, hidden_size, allow_trtllm_auto):
        return _trtllm_mhc_post(hidden_states, residual, post, comb)
    if not hidden_states.is_cuda:
        raise RuntimeError("fast mHC requires CUDA tensors")
    if residual.numel() == 0:
        return torch.empty_like(residual)
    out = torch.empty_like(residual)
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    hidden_states_flat = hidden_states.view(-1, hidden_size)
    post_flat = post.view(-1, hc_mult)
    comb_flat = comb.view(-1, hc_mult, hc_mult)
    num_tokens = residual_flat.shape[0]
    if hc_mult == 4:
        block_h = 256
        _mhc_post_hc4_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
            comb_flat,
            residual_flat,
            post_flat,
            hidden_states_flat,
            out,
            hidden_size=hidden_size,
            block_h=block_h,
            num_warps=4,
        )
        return out

    block_h = 1024
    _mhc_post_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        comb_flat,
        residual_flat,
        post_flat,
        hidden_states_flat,
        out,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        block_h=block_h,
        num_warps=4,
    )
    return out
