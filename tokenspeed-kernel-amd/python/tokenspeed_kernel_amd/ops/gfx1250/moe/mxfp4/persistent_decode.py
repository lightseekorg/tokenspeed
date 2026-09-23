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

"""Persistent gfx1250 A8W4 MoE decode combine kernel."""

from __future__ import annotations

import torch

from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx1250.moe._common import (
    FP4,
    FnSpecs,
    FusedActivation,
    Tensor,
    swiglu_fn,
    wrap_torch_tensor,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4 import fused
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4._common import (
    MoEConfig,
    MoEPipelinedProgram,
    _enforce_wave_uniform_i32,
    create_descriptor,
    get_blocked_layout,
    get_scaled_dot_format_string,
    get_tdm_gather_scatter_idx_layout,
    ragged_metadata_fields,
)

_BLOCK_M = 16
_BLOCK_N = 128
_BLOCK_K = 512
_NUM_WARPS = 8
_NUM_BUFFERS = 2
_WGS_PER_CU = 12
_WORK_MAPPING = "m_pinned"

_SUPPORTED_BLOCK_N = (128, 256)
_SUPPORTED_BLOCK_K = (256, 512)
_SUPPORTED_NUM_WARPS = (4, 8)
_SUPPORTED_NUM_BUFFERS = (2, 3)


def persistent_moe_num_wgs(available_cus: int, wgs_per_cu: int) -> int:
    """Return the persistent worker-grid size for a device."""
    if available_cus <= 0:
        raise ValueError("available_cus must be positive")
    if wgs_per_cu <= 0:
        raise ValueError("wgs_per_cu must be positive")
    return available_cus * wgs_per_cu


def _validate_persistent_config(
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_buffers: int,
    wgs_per_cu: int,
    work_mapping: str,
    partial_tdm: bool,
) -> None:
    if block_m != _BLOCK_M:
        raise ValueError(
            f"persistent decode requires block_m={_BLOCK_M}, got {block_m}"
        )
    if block_n not in _SUPPORTED_BLOCK_N:
        raise ValueError(
            f"persistent decode block_n must be one of {_SUPPORTED_BLOCK_N}, "
            f"got {block_n}"
        )
    if block_k not in _SUPPORTED_BLOCK_K:
        raise ValueError(
            f"persistent decode block_k must be one of {_SUPPORTED_BLOCK_K}, "
            f"got {block_k}"
        )
    if num_warps not in _SUPPORTED_NUM_WARPS:
        raise ValueError(
            "persistent decode num_warps must be one of "
            f"{_SUPPORTED_NUM_WARPS}, got {num_warps}"
        )
    if num_buffers not in _SUPPORTED_NUM_BUFFERS:
        raise ValueError(
            "persistent decode num_buffers must be one of "
            f"{_SUPPORTED_NUM_BUFFERS}, got {num_buffers}"
        )
    if wgs_per_cu <= 0:
        raise ValueError(f"wgs_per_cu must be positive, got {wgs_per_cu}")
    if work_mapping not in ("flat_grid", "m_pinned"):
        raise ValueError(
            f"work_mapping must be 'flat_grid' or 'm_pinned', got {work_mapping!r}"
        )
    if partial_tdm and num_warps not in (4, 8):
        raise ValueError(f"partial_tdm requires 4 or 8 warps, got {num_warps}")


def _persistent_combine_launch_metadata(grid, kernel, args):
    """Report capacity-level A8W4 work and traffic to Proton."""
    m = args["M"]
    n = args["N"]
    k = args["K"]
    output_bytes = m * n * args["Y"].element_size()
    activation_bytes = m * k
    weight_bytes = n * k // 2
    weight_scale_bytes = n * triton.cdiv(k, 32)
    return {
        "name": kernel.name,
        "flops8": 2 * m * n * k,
        "bytes": activation_bytes + weight_bytes + weight_scale_bytes + output_bytes,
    }


@gluon.jit(launch_metadata=_persistent_combine_launch_metadata)
def _persistent_a8w4_combine_kernel(
    Y,
    stride_y_m,
    stride_y_n,
    XGlobalScale,
    X,
    stride_x_m,
    stride_x_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    B,
    stride_b_e,
    M,
    N,
    K,
    WriteBackIndx,
    writeback_size,
    XSliceSizes,
    XSliceOffs,
    XBlockOffs,
    XBlockSchedule,
    X_SLICE_SIZES_DIVISIBILITY: gl.constexpr,
    grid_n,
    N_EXPTS_TOT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    INDEX_TYPE: gl.constexpr,
    UPCAST_INDICES: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PARTIAL_TDM: gl.constexpr,
    WORK_MAPPING: gl.constexpr,
):
    gl.static_assert(WriteBackIndx is not None)
    gl.static_assert(WMxScale is not None)
    gl.static_assert(BLOCK_M == 16)
    gl.static_assert(BLOCK_N == 128 or BLOCK_N == 256)
    gl.static_assert(BLOCK_K == 256 or BLOCK_K == 512)
    gl.static_assert(NUM_BUFFERS == 2 or NUM_BUFFERS == 3)
    gl.static_assert(NUM_WARPS == 4 or NUM_WARPS == 8)
    gl.static_assert(WORK_MAPPING == "flat_grid" or WORK_MAPPING == "m_pinned")

    DTYPE_X: gl.constexpr = get_scaled_dot_format_string(X.dtype.element_ty)
    DTYPE_W: gl.constexpr = get_scaled_dot_format_string(W.dtype.element_ty)
    gl.static_assert(DTYPE_X == "e4m3")
    gl.static_assert(DTYPE_W == "e2m1")

    address_index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
    cfg = MoEConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        DTYPE_X,
        DTYPE_W,
        SCALE_BLOCK=32,
        NUM_BUFFERS=NUM_BUFFERS,
        W_TRANSPOSE=True,
        WITH_X_MX_SCALE=False,
        WITH_W_MX_SCALE=True,
        SCALE_PRESHUFFLE=True,
        index_type=INDEX_TYPE,
        PARTIAL_TDM=PARTIAL_TDM,
        NUM_SUBTILES=(1, 1, 1),
        EVEN_K=False,
        USE_GATHER=False,
        NUM_WARPS=NUM_WARPS,
    )

    BLOCK_K_PACKED_X: gl.constexpr = BLOCK_K
    BLOCK_K_PACKED_W: gl.constexpr = BLOCK_K // 2
    x_buffer = gl.allocate_shared_memory(
        X.dtype.element_ty,
        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K_PACKED_X],
        layout=cfg.shared_layout_x,
    )
    w_buffer = gl.allocate_shared_memory(
        W.dtype.element_ty,
        shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K_PACKED_W],
        layout=cfg.shared_layout_w,
    )
    w_scale_buffer = gl.allocate_shared_memory(
        gl.uint8,
        shape=[
            NUM_BUFFERS,
            cfg.BLOCK_N_PRESHUFFLED,
            cfg.BLOCK_K_SCALE_PRESHUFFLED,
        ],
        layout=cfg.shared_layout_w_scale,
    )

    output_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 4]], [BLOCK_M, BLOCK_N], [1, 0]
    )
    output_buffer = gl.allocate_shared_memory(
        Y.dtype.element_ty,
        (BLOCK_M, BLOCK_N),
        output_layout,
    )
    output_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=Y,
        shape=(writeback_size, N),
        strides=(stride_y_m, stride_y_n),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=output_layout,
    )

    unpadded_m = gl.load(XBlockOffs + N_EXPTS_TOT)
    gl.assume(unpadded_m >= 0)
    if unpadded_m == 0:
        return

    worker_pid = gl.program_id(0)
    num_workers = gl.num_programs(0)
    off_k_x = (worker_pid - worker_pid).to(gl.int32)
    total_tiles = unpadded_m * grid_n
    if WORK_MAPPING == "flat_grid":
        if worker_pid >= total_tiles:
            return
        first_m = 0
        last_m = gl.cdiv(total_tiles - worker_pid, num_workers)
        step_m = 1
    else:
        if unpadded_m <= num_workers:
            first_m = worker_pid % unpadded_m
            step_m = unpadded_m
            first_n = worker_pid // unpadded_m
            step_n = gl.cdiv(num_workers - first_m, unpadded_m)
        else:
            first_m = worker_pid
            step_m = num_workers
            first_n = 0
            step_n = 1
        last_m = unpadded_m
        last_n = grid_n

    x_global_scale = 1.0
    if XGlobalScale is not None:
        x_global_scale = gl.load(XGlobalScale).to(gl.float32)

    for persistent_iter in tl.range(first_m, last_m, step_m, disable_licm=True):
        if WORK_MAPPING == "flat_grid":
            logical_tile = worker_pid + persistent_iter * num_workers
            pid_m = logical_tile // grid_n
            first_n = logical_tile % grid_n
            last_n = first_n + 1
            step_n = 1
        else:
            pid_m = persistent_iter

        schedule = gl.load(XBlockSchedule + pid_m)
        expert_id = schedule & 0xFFFF
        block_id_m = schedule >> 16
        start_m = gl.load(XSliceOffs + expert_id)
        off_m = block_id_m * BLOCK_M
        expert_m = gl.multiple_of(
            gl.load(XSliceSizes + expert_id), X_SLICE_SIZES_DIVISIBILITY
        )

        expert_id = expert_id.to(address_index_type)
        start_m = start_m.to(address_index_type)
        off_m = off_m.to(address_index_type)
        x_ptr = X + start_m * stride_x_m
        w_ptr = W + expert_id * stride_w_e
        w_scale_ptr = WMxScale + expert_id * stride_w_mx_e
        descriptor_m = (expert_m - off_m).to(gl.int32)

        if WORK_MAPPING == "m_pinned":
            x_desc, w_base_desc, _, w_scale_base_desc, gathered_m = create_descriptor(
                cfg,
                x_ptr,
                w_ptr,
                None,
                w_scale_ptr,
                off_m,
                0,
                0,
                0,
                descriptor_m,
                N,
                K,
                stride_x_m,
                stride_x_k,
                stride_w_k,
                stride_w_n,
                None,
                None,
                stride_w_mx_n,
                stride_w_mx_k,
                None,
                start_m,
            )

        for pid_n in range(first_n, last_n, step_n):
            pid_n = pid_n.to(address_index_type)
            if WORK_MAPPING == "m_pinned":
                w_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
                    w_base_desc,
                    add_offsets=[(pid_n * BLOCK_N).to(gl.int32), 0],
                )
                w_scale_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
                    w_scale_base_desc,
                    add_offsets=[
                        (pid_n * cfg.BLOCK_N_PRESHUFFLED).to(gl.int32),
                        0,
                    ],
                )
            else:
                w_offs = pid_n * BLOCK_N * stride_w_n
                w_scale_offs = pid_n * cfg.BLOCK_N_PRESHUFFLED * stride_w_mx_n
                x_desc, w_desc, _, w_scale_desc, gathered_m = create_descriptor(
                    cfg,
                    x_ptr,
                    w_ptr,
                    None,
                    w_scale_ptr,
                    off_m,
                    0,
                    w_offs,
                    w_scale_offs,
                    descriptor_m,
                    BLOCK_N,
                    K,
                    stride_x_m,
                    stride_x_k,
                    stride_w_k,
                    stride_w_n,
                    None,
                    None,
                    stride_w_mx_n,
                    stride_w_mx_k,
                    None,
                    start_m,
                )

            program = MoEPipelinedProgram(
                cfg,
                x_buffer,
                w_buffer,
                gl.constexpr(0),
                w_scale_buffer,
                x_desc,
                w_desc,
                gl.constexpr(0),
                w_scale_desc,
                gathered_m,
                off_k_x,
            )
            accumulator = program.pipeline(K)
            accumulator *= x_global_scale

            bias_layout: gl.constexpr = get_blocked_layout(
                [BLOCK_N], B.dtype if B is not None else gl.float32, NUM_WARPS, 1
            )
            bias_offsets = BLOCK_N * pid_n + gl.arange(0, BLOCK_N, bias_layout)
            if B is not None:
                bias = gl.load(
                    B + expert_id * stride_b_e + bias_offsets,
                    mask=bias_offsets < N,
                    other=0.0,
                )
            else:
                bias = gl.full([BLOCK_N], 0.0, dtype=gl.float32, layout=bias_layout)
            accumulator += gl.convert_layout(bias, gl.SliceLayout(0, cfg.acc_layout))[
                None, :
            ]

            output_layout_registers: gl.constexpr = get_blocked_layout(
                [BLOCK_M, BLOCK_N], Y.dtype, NUM_WARPS
            )
            output = gl.convert_layout(
                accumulator.to(Y.dtype.element_ty), output_layout_registers
            )
            output_buffer.store(output)

            index_base_layout: gl.constexpr = get_tdm_gather_scatter_idx_layout(
                BLOCK_M, NUM_WARPS
            )
            index_layout: gl.constexpr = gl.SliceLayout(0, index_base_layout)
            index_offsets = gl.arange(0, BLOCK_M, index_layout)
            index_mask = (off_m + index_offsets < expert_m) & (
                start_m + off_m + index_offsets < writeback_size
            )
            writeback_ptr = WriteBackIndx + start_m
            destination_rows = gl.load(
                writeback_ptr + off_m + index_offsets,
                mask=index_mask,
                other=writeback_size,
            )
            destination_rows = gl.where(
                destination_rows == -1, writeback_size, destination_rows
            ).to(INDEX_TYPE)
            column_offset = _enforce_wave_uniform_i32((pid_n * BLOCK_N).to(gl.int32))
            output_tile_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
                output_desc,
                add_offsets=[0, column_offset],
                clamp_bounds=True,
            )
            gl.amd.cdna5.tdm.async_scatter(
                output_tile_desc, destination_rows, output_buffer
            )
            gl.amd.cdna5.tdm.async_wait(0)


def _persistent_a8w4_combine(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_global_scale: torch.Tensor | float | None,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    scatter_indx: torch.Tensor,
    out_dtype: torch.dtype,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_buffers: int,
    wgs_per_cu: int,
    work_mapping: str,
    partial_tdm: bool,
) -> tuple[torch.Tensor, object]:
    """Launch the persistent down projection for routed decode rows."""
    if a_ragged_metadata is None:
        raise ValueError("persistent decode combine requires M-ragged metadata")
    if scatter_indx is None:
        raise ValueError("persistent decode combine requires scatter indices")
    _validate_persistent_config(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_buffers=num_buffers,
        wgs_per_cu=wgs_per_cu,
        work_mapping=work_mapping,
        partial_tdm=partial_tdm,
    )

    x_torch = x.storage.data if isinstance(x, Tensor) else x
    w_torch = w.storage.data if isinstance(w, Tensor) else w
    if x_torch.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "persistent decode combine requires FP8 E4M3 activations, "
            f"got {x_torch.dtype}"
        )
    if w_torch.dtype != torch.uint8:
        raise TypeError(
            "persistent decode combine requires packed MXFP4 weights, "
            f"got {w_torch.dtype}"
        )
    if x_torch.ndim != 2 or w_torch.ndim != 3:
        raise ValueError(
            "persistent decode combine requires rank-2 activations and rank-3 weights"
        )

    m = int(x_torch.shape[0])
    k = int(x_torch.shape[1])
    packed_k, n = map(int, w_torch.shape[-2:])
    if packed_k * 2 != k:
        raise ValueError(
            f"K mismatch: activation K={k} vs packed weight K={packed_k * 2}"
        )
    if n % block_n:
        raise ValueError(f"output width {n} must be divisible by block_n={block_n}")

    if not isinstance(x, Tensor):
        x = wrap_torch_tensor(x, dtype=x_torch.dtype)
    if not isinstance(w, Tensor):
        if w_torch.stride(-2) != 1:
            w_torch = w_torch.transpose(-1, -2).contiguous().transpose(-1, -2)
            w = w_torch
        w = wrap_torch_tensor(w, dtype=FP4)

    w_scale_tensor = fused._as_tensor(w_scale)
    if w_scale_tensor is None:
        raise ValueError("persistent decode combine requires MXFP4 weight scales")
    w_scale_tensor.storage.data = w_scale_tensor.storage.data.view(torch.uint8)
    w_scale_tensor.dtype = torch.uint8
    w_scale_tensor = fused._mark_scale_preshuffled(w_scale_tensor, True)

    scatter_tensor = fused._index_tensor(scatter_indx, "dst_indx")
    if scatter_tensor is None:
        raise ValueError("persistent decode combine requires scatter indices")
    index_type = fused.get_tdm_index_type(x, None, scatter_tensor)

    output = torch.empty(
        (int(scatter_tensor.shape[0]), n),
        device=x_torch.device,
        dtype=out_dtype,
    )
    x_storage = fused._canonicalize_storage(x.storage, 3)
    w_storage = fused._canonicalize_storage(w.storage, 3)

    if x_global_scale is not None:
        if isinstance(x_global_scale, torch.Tensor):
            if x_global_scale.numel() != 1:
                raise ValueError("x_global_scale must be scalar")
            x_global_scale = x_global_scale.to(
                device=x_torch.device, dtype=torch.float32
            ).contiguous()
        else:
            x_global_scale = torch.tensor(
                [float(x_global_scale)],
                device=x_torch.device,
                dtype=torch.float32,
            )

    metadata = ragged_metadata_fields(a_ragged_metadata, block_m)
    slice_sizes, slice_offsets, block_offsets, block_schedule = metadata[:4]
    expected_slice_size, slice_sizes_divisibility = metadata[4:]
    del expected_slice_size

    num_experts = int(a_ragged_metadata.n_slices)
    grid_n = triton.cdiv(n, block_n)
    available_cus = torch.cuda.get_device_properties(
        x_torch.device
    ).multi_processor_count
    grid = persistent_moe_num_wgs(available_cus, wgs_per_cu)
    a_strides = [0] * (3 - x_storage.data.ndim) + list(x_storage.data.stride())
    w_scale_strides = w_scale_tensor.stride()
    w_scale_strides = (0,) * (3 - len(w_scale_strides)) + tuple(w_scale_strides)
    bias_stride = None if bias is None else bias.stride(0)

    kernel = _persistent_a8w4_combine_kernel[(grid,)](
        output,
        *output.stride(),
        x_global_scale,
        x_storage.data,
        a_strides[-2],
        a_strides[-1],
        w_storage.data,
        *w_storage.data.stride(),
        w_scale_tensor,
        *w_scale_strides,
        bias,
        bias_stride,
        m,
        n,
        k,
        scatter_tensor,
        scatter_tensor.shape[0],
        slice_sizes,
        slice_offsets,
        block_offsets,
        block_schedule,
        slice_sizes_divisibility,
        grid_n,
        num_experts,
        block_m,
        block_n,
        block_k,
        INDEX_TYPE=index_type,
        UPCAST_INDICES=fused.should_upcast_indices(x, w, output),
        NUM_BUFFERS=num_buffers,
        NUM_WARPS=num_warps,
        PARTIAL_TDM=partial_tdm,
        WORK_MAPPING=work_mapping,
        num_warps=num_warps,
        waves_per_eu=1,
    )
    return output, kernel


def gluon_mxfp4_a8w4_persistent_decode(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    activation: str,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    swiglu_beta: float = 1.0,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run gfx1250 A8W4 MoE decode with a persistent down projection.

    Args:
        hidden_states: BF16, FP16, or E4M3 token activations shaped
            ``(num_tokens, hidden_size)``.
        topk_weights: FP32 route weights shaped ``(num_tokens, top_k)``.
        topk_ids: Expert ids shaped ``(num_tokens, top_k)``.
        w13_weight: Preprocessed packed MXFP4 gate/up weights.
        w2_weight: Preprocessed packed MXFP4 down-projection weights.
        w13_mx_scale: Preprocessed gate/up E8M0 scales.
        w2_mx_scale: Preprocessed down-projection E8M0 scales.
        activation: Fused gate activation: ``"silu"``, ``"swiglu"``, or
            ``"situ"``.
        w13_bias: Optional FP32 gate/up expert bias.
        w2_bias: Optional FP32 down-projection expert bias.
        out_dtype: Down-projection output dtype before route reduction.
        swiglu_alpha: SwiGLU gate scale.
        swiglu_limit: SwiGLU clamp limit; zero disables clamping.
        swiglu_beta: SwiGLU linear-branch offset.
        situ_beta: SiTU gate clamp.
        situ_linear_beta: SiTU linear-branch clamp.
        out: Optional destination for the finalized token output.

    Returns:
        Tensor shaped ``(num_tokens, hidden_size)``.
    """
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must be rank-2, got {tuple(hidden_states.shape)}"
        )
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )

    w13_raw = w13_weight.storage.data if isinstance(w13_weight, Tensor) else w13_weight
    if not isinstance(w13_raw, torch.Tensor) or w13_raw.ndim != 3:
        raise ValueError("w13_weight must expose a rank-3 expert weight tensor")
    num_experts = int(w13_raw.shape[0])
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")

    topk_ids = topk_ids.to(device=hidden_states.device, dtype=torch.int32).contiguous()
    topk_weights = topk_weights.to(
        device=hidden_states.device, dtype=torch.float32
    ).contiguous()
    ragged_metadata, gather_indx, scatter_indx, _ = fused._precomputed_topk_route(
        topk_weights,
        topk_ids,
        num_experts,
    )

    hidden_fp8 = fused._quantize_fp8_activation(
        hidden_states,
        w13_weight.act_scale,
    )
    if activation == "situ":
        fused_activation = FusedActivation(
            FnSpecs("situ", None, ("beta", "linear_beta"), reduction_n=2),
            (float(situ_beta), float(situ_linear_beta)),
        )
    elif activation == "silu":
        fused_activation = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit", "beta"), reduction_n=2),
            (1.0, 0.0, 0.0),
        )
    elif activation == "swiglu":
        fused_activation = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit", "beta"), reduction_n=2),
            (float(swiglu_alpha), float(swiglu_limit), float(swiglu_beta)),
        )
    else:
        raise ValueError(
            "gfx1250 persistent A8W4 MoE supports activation 'silu', "
            f"'swiglu', or 'situ', got {activation!r}"
        )

    intermediate = fused.gluon_mxfp_ragged_matmul(
        hidden_fp8,
        w13_weight,
        w13_bias,
        w_mx_scale=w13_mx_scale,
        x_format="e4m3",
        x_global_scale=w13_weight.act_scale,
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        out_dtype=out_dtype,
        fused_activation=fused_activation,
        scale_preshuffle=True,
        block_m=_BLOCK_M,
        block_n=256,
        block_k=256,
        num_warps=4,
        num_buffers=3,
        decode=True,
        partial_tdm=False,
    )
    intermediate_fp8 = fused._quantize_fp8_activation(
        intermediate,
        w2_weight.act_scale,
    )
    flat, _ = _persistent_a8w4_combine(
        intermediate_fp8,
        w2_weight,
        w2_mx_scale,
        x_global_scale=w2_weight.act_scale,
        bias=w2_bias,
        a_ragged_metadata=ragged_metadata,
        scatter_indx=scatter_indx,
        out_dtype=out_dtype,
        block_m=_BLOCK_M,
        block_n=_BLOCK_N,
        block_k=_BLOCK_K,
        num_warps=_NUM_WARPS,
        num_buffers=_NUM_BUFFERS,
        wgs_per_cu=_WGS_PER_CU,
        work_mapping=_WORK_MAPPING,
        partial_tdm=False,
    )
    return fused._weighted_topk_reduce_gfx1250(
        flat,
        topk_weights,
        out=out,
        out_dtype=out_dtype,
    )


__all__ = [
    "gluon_mxfp4_a8w4_persistent_decode",
    "persistent_moe_num_wgs",
]
