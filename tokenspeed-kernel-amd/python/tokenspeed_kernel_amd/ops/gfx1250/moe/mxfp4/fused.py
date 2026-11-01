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

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
from tokenspeed_kernel_amd._triton import (
    aggregate,
    gl,
    gluon,
    tl,
    triton,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe._common import (
    FP4,
    FnSpecs,
    FusedActivation,
    RaggedTensorMetadata,
    Storage,
    Tensor,
    make_ragged_tensor_metadata,
    swiglu_fn,
    wrap_torch_tensor,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4._common import (
    MoEConfig,
    MoEPipelinedProgram,
    MoEProgramBase,
    _enforce_wave_uniform_i32,
    _situ_gfx1250,
    _swiglu_gfx1250,
    composition,
    compute_offsets,
    compute_pids,
    create_descriptor,
    get_blocked_layout,
    get_scaled_dot_format_string,
    get_tdm_gather_scatter_idx_layout,
    ragged_metadata_fields,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.decode import (
    _matmul_decode,
    decode_tile_config,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.quantize import (
    _SCALE_ROW_ALIGN,
    MXFP4_BLOCK,
    empty_mxfp4_activation_scale,
    launch_triton_quantize_mxfp4_activation_gfx1250,
    quantize_mxfp4_rows_tile_gfx1250,
    store_mxfp4_tile_gfx1250,
)


@dataclass
class PrecisionConfig:
    """Minimal precision config accepted by the gfx1250 MoE wrapper."""

    a_mx_scale: torch.Tensor | Tensor | None = None
    b_mx_scale: torch.Tensor | Tensor | None = None
    out_dtype: torch.dtype | None = None


@dataclass
class _OptFlags:
    block_m: int
    block_n: int
    block_k: int
    group_m: int = 8
    xcd_swizzle: int = 1
    split_k: int = 1


@dataclass(frozen=True)
class _NamedScaleLayout:
    name: str


@composition
@aggregate
class MoESliceKProgram:
    base: MoEProgramBase

    cfg: MoEConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer: gl.shared_memory_descriptor
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr

    x_desc: gl.amd.cdna5.tdm.tensor_descriptor
    w_desc: gl.amd.cdna5.tdm.tensor_descriptor
    x_scale_desc: gl.amd.cdna5.tdm.tensor_descriptor | gl.constexpr
    w_scale_desc: gl.amd.cdna5.tdm.tensor_descriptor | gl.constexpr

    gathered_m: gl.tensor | gl.constexpr
    off_k_x: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        x_buffer,
        w_buffer,
        x_scale_buffer,
        w_scale_buffer,
        x_desc,
        w_desc,
        x_scale_desc,
        w_scale_desc,
        gathered_m,
        off_k_x,
    ):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer = w_buffer
        self.x_scale_buffer = x_scale_buffer if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.x_desc = x_desc
        self.w_desc = w_desc
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.gathered_m = gathered_m
        self.off_k_x = off_k_x

        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(
        cfg: MoEConfig, x_desc, w_desc, x_scale_desc, w_scale_desc, gathered_m, off_k_x
    ):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer = gl.allocate_shared_memory(
            x_desc.dtype,
            shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_PACKED_X],
            layout=cfg.shared_layout_x,
        )
        w_buffer = gl.allocate_shared_memory(
            w_desc.dtype,
            shape=(
                [NUM_BUFFERS, cfg.BLOCK_N, BLOCK_K_PACKED_W]
                if cfg.W_TRANSPOSE
                else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N]
            ),
            layout=cfg.shared_layout_w,
        )

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[NUM_BUFFERS, cfg.BLOCK_M, cfg.BLOCK_K // cfg.SCALE_BLOCK],
                layout=cfg.shared_layout_x_scale,
            )
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.WITH_W_MX_SCALE:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_w_scale,
            )
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoESliceKProgram(
            cfg,
            x_buffer,
            w_buffer,
            x_scale_buffer,
            w_scale_buffer,
            x_desc,
            w_desc,
            x_scale_desc,
            w_scale_desc,
            gathered_m,
            off_k_x,
        )

    @gluon.jit
    def issue_subtile_local_loads(self, wmma_idx, subtile_start_idx: gl.constexpr):
        cfg = self.cfg
        NUM_SUBTILES_K: gl.constexpr = cfg.NUM_SUBTILES[2]
        SUBTILE_LEN: gl.constexpr = cfg.BLOCK_K // NUM_SUBTILES_K
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        SUBTILE_LEN_SCALE: gl.constexpr = SUBTILE_LEN // cfg.SCALE_BLOCK
        subtile_start: gl.constexpr = subtile_start_idx * SUBTILE_LEN

        x = (
            self.x_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
            .slice(
                subtile_start // cfg.DIV_FACTOR_X, SUBTILE_LEN // cfg.DIV_FACTOR_X, 1
            )
            .load(layout=cfg.dot_layout_x)
        )

        if cfg.W_TRANSPOSE:
            w = (
                self.w_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
                .slice(
                    subtile_start // cfg.DIV_FACTOR_W,
                    SUBTILE_LEN // cfg.DIV_FACTOR_W,
                    1,
                )
                .permute([1, 0])
                .load(layout=cfg.dot_layout_w)
            )
        else:
            w = (
                self.w_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
                .slice(
                    subtile_start // cfg.DIV_FACTOR_W,
                    SUBTILE_LEN // cfg.DIV_FACTOR_W,
                    0,
                )
                .load(layout=cfg.dot_layout_w)
            )

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer_slice = self.x_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
            x_scale_buffer_slice = x_scale_buffer_slice.slice(
                subtile_start // cfg.SCALE_BLOCK, SUBTILE_LEN_SCALE, 1
            )
            scale_x = x_scale_buffer_slice.load(layout=cfg.layout_x_scale)
        else:
            scale_x = 0
            scale_x = scale_x.to(gl.uint8)

        if cfg.WITH_W_MX_SCALE:
            w_scale_buffer_slice = self.w_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
            if cfg.SCALE_PRESHUFFLE:
                w_scale_buffer_slice = (
                    w_scale_buffer_slice.reshape(
                        (
                            cfg.BLOCK_N_PRESHUFFLED,
                            BLOCK_K_SCALE // cfg.SCALE_KWIDTH,
                            cfg.PRESHUFFLE_FACTOR // 4,
                            4,
                            cfg.SCALE_KWIDTH,
                        )
                    )
                    .permute((0, 3, 2, 1, 4))
                    .reshape((cfg.BLOCK_N, BLOCK_K_SCALE))
                )
            w_scale_buffer_slice = w_scale_buffer_slice.slice(
                subtile_start // cfg.SCALE_BLOCK, SUBTILE_LEN_SCALE, 1
            )
            scale_w = w_scale_buffer_slice.load(layout=cfg.layout_w_scale)
        else:
            scale_w = 0
            scale_w = scale_w.to(gl.uint8)

        return x, w, scale_x, scale_w

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        load_idx = 0
        wmma_idx = 0

        # prologue
        # iter 0
        load_idx = self.issue_global_loads(load_idx)

        # iter 1
        load_idx = self.issue_global_loads(load_idx)
        self.async_wait(cfg.NUM_BUFFERS - 1)

        # iter 0
        x0, w0, scale_x0, scale_w0 = self.issue_subtile_local_loads(wmma_idx, 0)

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K) - 1
        for _ in range(0, loop_ub - 1):
            # iter i
            accumulator = self.wmma(x0, scale_x0, w0, scale_w0, accumulator)
            # iter i
            x1, w1, scale_x1, scale_w1 = self.issue_subtile_local_loads(wmma_idx, 1)
            wmma_idx += 1
            # iter i + 2
            load_idx = self.issue_global_loads(load_idx)
            # iter i
            accumulator = self.wmma(x1, scale_x1, w1, scale_w1, accumulator)
            # iter i + 1
            self.async_wait(cfg.NUM_BUFFERS - 1)
            x0, w0, scale_x0, scale_w0 = self.issue_subtile_local_loads(wmma_idx, 0)

        # epilogue
        accumulator = self.wmma(x0, scale_x0, w0, scale_w0, accumulator)
        x1, w1, scale_x1, scale_w1 = self.issue_subtile_local_loads(wmma_idx, 1)
        wmma_idx += 1
        accumulator = self.wmma(x1, scale_x1, w1, scale_w1, accumulator)

        self.async_wait(0)
        x0, w0, scale_x0, scale_w0 = self.issue_subtile_local_loads(wmma_idx, 0)
        accumulator = self.wmma(x0, scale_x0, w0, scale_w0, accumulator)
        x1, w1, scale_x1, scale_w1 = self.issue_subtile_local_loads(wmma_idx, 1)
        wmma_idx += 1
        accumulator = self.wmma(x1, scale_x1, w1, scale_w1, accumulator)

        return accumulator

    @gluon.jit
    def warp_pipeline(self, loop_k):
        cfg = self.cfg
        load_idx = 0
        wmma_idx = 0
        gl.static_assert(cfg.NUM_BUFFERS == 3)

        # prologue
        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_global_loads(load_idx)

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K) - (cfg.NUM_BUFFERS - 1)
        gl.assume(loop_ub >= 0)
        self.async_wait(cfg.NUM_BUFFERS - 2)
        for _ in range(0, loop_ub):
            with gl.amd.warp_pipeline_stage("lds0", priority=1):
                x0, w0, scale_x0, scale_w0 = self.issue_subtile_local_loads(wmma_idx, 0)

            self.async_wait(cfg.NUM_BUFFERS - 3)
            with gl.amd.warp_pipeline_stage("tdm+wmma+lds1", priority=0):
                load_idx = self.issue_global_loads(load_idx)
                accumulator = self.wmma(x0, scale_x0, w0, scale_w0, accumulator)
                x1, w1, scale_x1, scale_w1 = self.issue_subtile_local_loads(wmma_idx, 1)
                wmma_idx += 1
                accumulator = self.wmma(x1, scale_x1, w1, scale_w1, accumulator)

        # epilogue
        for i in gl.static_range(cfg.NUM_BUFFERS - 1):
            self.async_wait(cfg.NUM_BUFFERS - 1 - i)
            x0, w0, scale_x0, scale_w0 = self.issue_subtile_local_loads(wmma_idx, 0)
            accumulator = self.wmma(x0, scale_x0, w0, scale_w0, accumulator)
            x1, w1, scale_x1, scale_w1 = self.issue_subtile_local_loads(wmma_idx, 1)
            accumulator = self.wmma(x1, scale_x1, w1, scale_w1, accumulator)
            wmma_idx += 1

        return accumulator


@composition
@aggregate
class MoESliceNKProgram:
    base: MoEProgramBase

    cfg: MoEConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer: gl.shared_memory_descriptor
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr

    x_desc: gl.amd.cdna5.tdm.tensor_descriptor
    w_desc: gl.amd.cdna5.tdm.tensor_descriptor
    x_scale_desc: gl.amd.cdna5.tdm.tensor_descriptor | gl.constexpr
    w_scale_desc: gl.amd.cdna5.tdm.tensor_descriptor | gl.constexpr

    gathered_m: gl.tensor | gl.constexpr
    off_k_x: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        x_buffer,
        w_buffer,
        x_scale_buffer,
        w_scale_buffer,
        x_desc,
        w_desc,
        x_scale_desc,
        w_scale_desc,
        gathered_m,
        off_k_x,
    ):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer = w_buffer
        self.x_scale_buffer = x_scale_buffer if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.x_desc = x_desc
        self.w_desc = w_desc
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.gathered_m = gathered_m
        self.off_k_x = off_k_x

        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(
        cfg: MoEConfig, x_desc, w_desc, x_scale_desc, w_scale_desc, gathered_m, off_k_x
    ):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer = gl.allocate_shared_memory(
            x_desc.dtype,
            shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_PACKED_X],
            layout=cfg.shared_layout_x,
        )
        w_buffer = gl.allocate_shared_memory(
            w_desc.dtype,
            shape=(
                [NUM_BUFFERS, cfg.BLOCK_N, BLOCK_K_PACKED_W]
                if cfg.W_TRANSPOSE
                else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N]
            ),
            layout=cfg.shared_layout_w,
        )

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[NUM_BUFFERS, cfg.BLOCK_M, cfg.BLOCK_K // cfg.SCALE_BLOCK],
                layout=cfg.shared_layout_x_scale,
            )
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.WITH_W_MX_SCALE:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_w_scale,
            )
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoESliceNKProgram(
            cfg,
            x_buffer,
            w_buffer,
            x_scale_buffer,
            w_scale_buffer,
            x_desc,
            w_desc,
            x_scale_desc,
            w_scale_desc,
            gathered_m,
            off_k_x,
        )

    @gluon.jit
    def issue_global_load_x(self, load_idx, pred=1):
        cfg = self.cfg
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if cfg.USE_GATHER:
            col_offset_x = self.off_k_x + load_idx * BLOCK_K_PACKED_X
            x_desc_k = gl.amd.cdna5.tdm.update_tensor_descriptor(
                self.x_desc, add_offsets=[0, col_offset_x], pred=pred, clamp_bounds=True
            )
            gl.amd.cdna5.tdm.async_gather(
                x_desc_k,
                self.gathered_m,
                self.x_buffer.index(load_idx % cfg.NUM_BUFFERS),
            )
        else:
            gl.amd.cdna5.tdm.async_load(
                self.x_desc,
                [0, load_idx * BLOCK_K_PACKED_X],
                self.x_buffer.index(load_idx % cfg.NUM_BUFFERS),
                pred=pred,
            )

        if cfg.WITH_X_MX_SCALE:
            if cfg.USE_GATHER:
                col_offset_x_scale = (
                    self.off_k_x * cfg.DIV_FACTOR_X // cfg.SCALE_BLOCK
                    + load_idx * BLOCK_K_SCALE
                )
                x_scale_desc_k = gl.amd.cdna5.tdm.update_tensor_descriptor(
                    self.x_scale_desc,
                    add_offsets=[0, col_offset_x_scale],
                    pred=pred,
                    clamp_bounds=True,
                )
                gl.amd.cdna5.tdm.async_gather(
                    x_scale_desc_k,
                    self.gathered_m,
                    self.x_scale_buffer.index(load_idx % cfg.NUM_BUFFERS),
                )
            else:
                gl.amd.cdna5.tdm.async_load(
                    self.x_scale_desc,
                    [0, load_idx * BLOCK_K_SCALE],
                    self.x_scale_buffer.index(load_idx % cfg.NUM_BUFFERS),
                    pred=pred,
                )
        return load_idx + 1

    @gluon.jit
    def issue_global_load_w(self, load_idx, pred=1):
        cfg = self.cfg
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        if cfg.W_TRANSPOSE:
            gl.amd.cdna5.tdm.async_load(
                self.w_desc,
                [0, load_idx * BLOCK_K_PACKED_W],
                self.w_buffer.index(load_idx % cfg.NUM_BUFFERS),
                pred=pred,
            )
        else:
            gl.amd.cdna5.tdm.async_load(
                self.w_desc,
                [load_idx * BLOCK_K_PACKED_W, 0],
                self.w_buffer.index(load_idx % cfg.NUM_BUFFERS),
                pred=pred,
            )

        if cfg.WITH_W_MX_SCALE:
            gl.amd.cdna5.tdm.async_load(
                self.w_scale_desc,
                [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
                self.w_scale_buffer.index(load_idx % cfg.NUM_BUFFERS),
                pred=pred,
            )
        return load_idx + 1

    @gluon.jit
    def issue_local_load_x(self, wmma_idx, subtile_start_idx: gl.constexpr):
        cfg = self.cfg
        NUM_SUBTILES_K: gl.constexpr = cfg.NUM_SUBTILES[2]
        SUBTILE_LEN: gl.constexpr = cfg.BLOCK_K // NUM_SUBTILES_K
        subtile_start: gl.constexpr = subtile_start_idx * SUBTILE_LEN

        x = (
            self.x_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
            .slice(
                subtile_start // cfg.DIV_FACTOR_X, SUBTILE_LEN // cfg.DIV_FACTOR_X, 1
            )
            .load(layout=cfg.dot_layout_x)
        )

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer_slice = self.x_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
            x_scale_buffer_slice = x_scale_buffer_slice.slice(
                subtile_start // cfg.SCALE_BLOCK, SUBTILE_LEN // cfg.SCALE_BLOCK, 1
            )
            scale_x = x_scale_buffer_slice.load(layout=cfg.layout_x_scale)
        else:
            scale_x = 0
            scale_x = scale_x.to(gl.uint8)
        return x, scale_x

    @gluon.jit
    def issue_local_load_w(
        self,
        wmma_idx,
        subtile_start_idx_k: gl.constexpr,
        subtile_start_idx_n: gl.constexpr,
    ):
        cfg = self.cfg
        NUM_SUBTILES_N: gl.constexpr = cfg.NUM_SUBTILES[1]
        NUM_SUBTILES_K: gl.constexpr = cfg.NUM_SUBTILES[2]
        SUBTILE_LEN_K: gl.constexpr = cfg.BLOCK_K // NUM_SUBTILES_K
        SUBTILE_LEN_N: gl.constexpr = cfg.BLOCK_N // NUM_SUBTILES_N
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        subtile_start_k: gl.constexpr = subtile_start_idx_k * SUBTILE_LEN_K
        subtile_start_n: gl.constexpr = subtile_start_idx_n * SUBTILE_LEN_N

        if cfg.W_TRANSPOSE:
            w = (
                self.w_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
                .slice(subtile_start_n, SUBTILE_LEN_N, 0)
                .slice(
                    subtile_start_k // cfg.DIV_FACTOR_W,
                    SUBTILE_LEN_K // cfg.DIV_FACTOR_W,
                    1,
                )
                .permute([1, 0])
                .load(layout=cfg.dot_layout_w)
            )
        else:
            w = (
                self.w_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
                .slice(
                    subtile_start_k // cfg.DIV_FACTOR_W,
                    SUBTILE_LEN_K // cfg.DIV_FACTOR_W,
                    0,
                )
                .slice(subtile_start_n, SUBTILE_LEN_N, 1)
                .load(layout=cfg.dot_layout_w)
            )

        w_scale_buffer_slice = self.w_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        if cfg.SCALE_PRESHUFFLE:
            w_scale_buffer_slice = (
                w_scale_buffer_slice.reshape(
                    (
                        cfg.BLOCK_N_PRESHUFFLED,
                        BLOCK_K_SCALE // cfg.SCALE_KWIDTH,
                        cfg.PRESHUFFLE_FACTOR // 4,
                        4,
                        cfg.SCALE_KWIDTH,
                    )
                )
                .permute((0, 3, 2, 1, 4))
                .reshape((cfg.BLOCK_N, BLOCK_K_SCALE))
            )
        w_scale_buffer_slice = w_scale_buffer_slice.slice(
            subtile_start_n, SUBTILE_LEN_N, 0
        ).slice(subtile_start_k // cfg.SCALE_BLOCK, SUBTILE_LEN_K // cfg.SCALE_BLOCK, 1)
        scale_w = w_scale_buffer_slice.load(layout=cfg.layout_w_scale)
        return w, scale_w

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        load_x_idx = 0
        load_w_idx = 0
        wmma_idx = 0

        # prologue: iter 0
        load_x_idx = self.issue_global_load_x(load_x_idx)
        load_w_idx = self.issue_global_load_w(load_w_idx)

        self.async_wait(0)
        x0, scale_x0 = self.issue_local_load_x(wmma_idx, 0)
        w00, scale_w00 = self.issue_local_load_w(wmma_idx, 0, 0)

        NUM_SUBTILES_M: gl.constexpr = cfg.NUM_SUBTILES[0]
        NUM_SUBTILES_N: gl.constexpr = cfg.NUM_SUBTILES[1]
        c0 = gl.zeros(
            (cfg.BLOCK_M // NUM_SUBTILES_M, cfg.BLOCK_N // NUM_SUBTILES_N),
            dtype=gl.float32,
            layout=cfg.acc_layout,
        )
        c1 = gl.zeros(
            (cfg.BLOCK_M // NUM_SUBTILES_M, cfg.BLOCK_N // NUM_SUBTILES_N),
            dtype=gl.float32,
            layout=cfg.acc_layout,
        )

        loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K)
        epilogue_lb = loop_ub - (cfg.NUM_BUFFERS - 1)
        gl.assume(loop_ub > 0)

        for i in range(0, loop_ub):
            pred = i - epilogue_lb
            pred = (pred >> 31) & 1

            # iter i + 1
            load_x_idx = self.issue_global_load_x(load_x_idx, pred=pred)
            load_w_idx = self.issue_global_load_w(load_w_idx, pred=pred)

            # iter i
            c0 = self.wmma(x0, scale_x0, w00, scale_w00, c0)
            w01, scale_w01 = self.issue_local_load_w(wmma_idx, 0, 1)

            c1 = self.wmma(x0, scale_x0, w01, scale_w01, c1)
            x1, scale_x1 = self.issue_local_load_x(wmma_idx, 1)
            w10, scale_w10 = self.issue_local_load_w(wmma_idx, 1, 0)

            c0 = self.wmma(x1, scale_x1, w10, scale_w10, c0)
            w11, scale_w11 = self.issue_local_load_w(wmma_idx, 1, 1)

            wmma_idx += 1
            c1 = self.wmma(x1, scale_x1, w11, scale_w11, c1)

            # iter i + 1
            self.async_wait(0)
            x0, scale_x0 = self.issue_local_load_x(wmma_idx, 0)
            w00, scale_w00 = self.issue_local_load_w(wmma_idx, 0, 0)

        accumulator = gl.join(c0, c1)
        accumulator = accumulator.permute(0, 2, 1).reshape((cfg.BLOCK_M, cfg.BLOCK_N))
        accumulator = gl.convert_layout(
            accumulator, cfg.acc_layout, assert_trivial=True
        )

        return accumulator


def _matmul_launch_metadata(grid, kernel, args):
    """Report the routed GEMM's work and modeled traffic to Proton.

    ``bytes`` counts each operand once: the routed activation rows, every
    expert's packed weights and scales, and the output rows. Re-reads of an
    activation or weight tile by other tiles are left to the cache and not
    counted.
    """
    m, n, k = args["M"], args["N"], args["K"]
    x, w, y = args["X"], args["W"], args["Y"]
    # Packed MXFP4 holds two values per byte.
    x_bytes = m * k * x.element_size() // (2 if x.dtype == torch.uint8 else 1)
    w_bytes = w.numel() * w.element_size()
    if args["WMxScale"] is not None:
        w_values = w.numel() * (2 if w.dtype == torch.uint8 else 1)
        w_bytes += w_values // args["SCALE_BLOCK"]
    y_values = m * (n // args["ACTIVATION_REDUCTION_N"])
    if args["YMxScale"] is None:
        y_bytes = y_values * y.element_size()
    else:
        y_bytes = y_values // 2 + y_values // args["SCALE_BLOCK"]
    return {
        "name": f"{kernel.name} [M={m}, N={n}, K={k}]",
        "flops8": 2 * m * n * k,
        "bytes": x_bytes + w_bytes + y_bytes,
    }


@gluon.jit(launch_metadata=_matmul_launch_metadata)
def _matmul(
    Y,
    stride_y_k,
    stride_y_z,
    stride_y_m,
    stride_y_n,
    YGlobalScale,
    YMxScale,
    stride_y_mx_m,
    XGlobalScale,
    X,
    stride_x_z,
    stride_x_m,
    stride_x_k,
    XMxScale,
    stride_x_mx_z,
    stride_x_mx_m,
    stride_x_mx_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    W_TRANSPOSE: gl.constexpr,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    B,
    stride_b_e,  # Bias
    M,
    N,
    K,
    KW,  # shapes
    GatherIndx,
    WriteBackIndx,
    writeback_size,
    RAGGED_DIMENSION: gl.constexpr,  #
    XSliceSizes,
    XSliceOffs,
    XBlockOffs,
    XBlockSchedule,
    X_EXPECTED_SLICE_SIZE: gl.constexpr,
    X_SLICE_SIZES_DIVISIBILITY: gl.constexpr,  #
    WSliceSizes,
    WSliceOffs,
    WBlockOffs,
    WBlockSchedule,
    W_EXPECTED_SLICE_SIZE: gl.constexpr,
    _W_SLICE_SIZES_DIVISIBILITY: gl.constexpr,  #
    # true grid size
    batch_size,
    grid_m,
    grid_n,
    DO_SWIGLU: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    DO_SITU: gl.constexpr,
    SITU_BETA: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    # MoE config
    N_EXPTS_TOT: gl.constexpr,
    # optimization config
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    SWIZZLE_MX_SCALE: gl.constexpr,
    EVEN_K: gl.constexpr,
    INDEX_TYPE: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
    NUM_BUFFERS: gl.constexpr = 2,
    SCALE_BLOCK: gl.constexpr = 32,
    SCHEDULE: gl.constexpr = "baseline",
    PINGPONG: gl.constexpr = False,
    NUM_WARPS: gl.constexpr = 4,
    PARTIAL_TDM: gl.constexpr = False,
):
    gl.static_assert(RAGGED_DIMENSION is None or RAGGED_DIMENSION == "M")
    SPLIT_K: gl.constexpr = 1

    DTYPE_X: gl.constexpr = get_scaled_dot_format_string(X.dtype.element_ty)
    DTYPE_W: gl.constexpr = get_scaled_dot_format_string(W.dtype.element_ty)

    # Width of pointer arithmetic only; TDM row indices use INDEX_TYPE.
    address_index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
    USE_GATHER: gl.constexpr = GatherIndx is not None

    SCALE_PRESHUFFLE: gl.constexpr = (
        SWIZZLE_MX_SCALE is not None and SWIZZLE_MX_SCALE != "STRIDED"
    )

    WITH_X_MX_SCALE: gl.constexpr = XMxScale is not None
    WITH_W_MX_SCALE: gl.constexpr = WMxScale is not None

    if SCHEDULE == "sliceNK":
        NUM_SUBTILES: gl.constexpr = (1, 2, 2)
    elif SCHEDULE == "sliceK":
        NUM_SUBTILES: gl.constexpr = (1, 1, 2)
    else:
        gl.static_assert(SCHEDULE == "baseline")
        NUM_SUBTILES: gl.constexpr = (1, 1, 1)

    cfg = MoEConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        DTYPE_X,
        DTYPE_W,
        SCALE_BLOCK=SCALE_BLOCK,
        NUM_BUFFERS=NUM_BUFFERS,
        W_TRANSPOSE=W_TRANSPOSE,
        WITH_X_MX_SCALE=WITH_X_MX_SCALE,
        WITH_W_MX_SCALE=WITH_W_MX_SCALE,
        SCALE_PRESHUFFLE=SCALE_PRESHUFFLE,
        index_type=INDEX_TYPE,
        PARTIAL_TDM=PARTIAL_TDM,
        NUM_SUBTILES=NUM_SUBTILES,
        EVEN_K=EVEN_K,
        USE_GATHER=USE_GATHER,
        NUM_WARPS=NUM_WARPS,
    )

    PACKED_BLOCK_K_W: gl.constexpr = BLOCK_K // cfg.DIV_FACTOR_W

    if _W_SLICE_SIZES_DIVISIBILITY is None:
        W_SLICE_SIZES_DIVISIBILITY: gl.constexpr = 1
    else:
        if PACKED_BLOCK_K_W > BLOCK_K:
            W_SLICE_SIZES_DIVISIBILITY: gl.constexpr = _W_SLICE_SIZES_DIVISIBILITY * (
                PACKED_BLOCK_K_W // BLOCK_K
            )
        else:
            W_SLICE_SIZES_DIVISIBILITY: gl.constexpr = _W_SLICE_SIZES_DIVISIBILITY // (
                BLOCK_K // PACKED_BLOCK_K_W
            )

    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = gl.program_id(0)
    if RAGGED_DIMENSION == "M":
        padding_m = grid_m - gl.load(XBlockOffs + N_EXPTS_TOT)
    else:
        padding_m: gl.constexpr = 0

    unpadded_m = grid_m - padding_m
    gl.assume(unpadded_m >= 0)
    total_actual_tiles = batch_size * unpadded_m * grid_n * SPLIT_K

    if padding_m > 0 and pid >= total_actual_tiles:
        return

    pid_s, pid_m, pid_n, pid_k = compute_pids(
        pid, unpadded_m, grid_n, total_actual_tiles, XCD_SWIZZLE, GROUP_M, SPLIT_K
    )

    expt_id, start_z, start_z_out, start_m, _, off_m, off_k_x, off_k_w = (
        compute_offsets(
            pid_s,
            pid_m,
            pid_k,
            XBlockSchedule,
            XSliceOffs,
            XBlockOffs,
            X_SLICE_SIZES_DIVISIBILITY,
            WBlockSchedule,
            WSliceOffs,
            W_SLICE_SIZES_DIVISIBILITY,
            RAGGED_DIMENSION,
            BLOCK_M,
            BLOCK_K,
            PACKED_BLOCK_K_W,
            SPLIT_K,
        )
    )
    if X_SLICE_SIZES_DIVISIBILITY is not None:
        off_k_x = off_k_x // X_SLICE_SIZES_DIVISIBILITY * X_SLICE_SIZES_DIVISIBILITY
    if W_SLICE_SIZES_DIVISIBILITY is not None:
        off_k_w = off_k_w // W_SLICE_SIZES_DIVISIBILITY * W_SLICE_SIZES_DIVISIBILITY

    if RAGGED_DIMENSION == "M":
        eM = gl.multiple_of(gl.load(XSliceSizes + expt_id), X_SLICE_SIZES_DIVISIBILITY)
    else:
        eM = M

    expt_id, off_m = expt_id.to(address_index_type), off_m.to(address_index_type)
    start_m, start_z = start_m.to(address_index_type), start_z.to(address_index_type)
    pid_n, pid_k = pid_n.to(address_index_type), pid_k.to(address_index_type)

    X_ptr = X + start_z * stride_x_z
    if not cfg.USE_GATHER:
        X_ptr += start_m * stride_x_m

    W_ptr = W + expt_id * stride_w_e
    w_offs = pid_n * BLOCK_N * stride_w_n

    if cfg.WITH_X_MX_SCALE:
        XMxScale_ptr = XMxScale + start_z.to(address_index_type) * stride_x_mx_z
        if not cfg.USE_GATHER:
            XMxScale_ptr += start_m * stride_x_mx_m
    else:
        XMxScale_ptr = XMxScale

    if cfg.WITH_W_MX_SCALE:
        WMxScale_ptr = WMxScale + expt_id * stride_w_mx_e
        w_scale_offs = pid_n * cfg.BLOCK_N_PRESHUFFLED * stride_w_mx_n
    else:
        WMxScale_ptr = WMxScale
        w_scale_offs = 0

    descriptor_m = M
    if not cfg.USE_GATHER:
        # Rows left in this expert fit i32 even when the weight slab needs the
        # wide index type.
        descriptor_m = (eM - off_m).to(gl.int32)
    # The weight descriptors start at this N tile, so their bounds must too.
    descriptor_n = (N - pid_n * BLOCK_N).to(gl.int32)
    x_desc, w_desc, x_scale_desc, w_scale_desc, gathered_m = create_descriptor(
        cfg,
        X_ptr,
        W_ptr,
        XMxScale_ptr,
        WMxScale_ptr,
        off_m,
        off_k_x,
        w_offs,
        w_scale_offs,
        descriptor_m,
        descriptor_n,
        K,
        stride_x_m,
        stride_x_k,
        stride_w_k,
        stride_w_n,
        stride_x_mx_m,
        stride_x_mx_k,
        stride_w_mx_n,
        stride_w_mx_k,
        GatherIndx,
        start_m,
    )

    Y_ptr = Y + start_z_out.to(address_index_type) * stride_y_z

    if SCHEDULE == "sliceNK":
        pgm = MoESliceNKProgram.initialize(
            cfg,
            x_desc,
            w_desc,
            x_scale_desc,
            w_scale_desc,
            gathered_m,
            off_k_x // cfg.DIV_FACTOR_X,
        )
    elif SCHEDULE == "sliceK":
        pgm = MoESliceKProgram.initialize(
            cfg,
            x_desc,
            w_desc,
            x_scale_desc,
            w_scale_desc,
            gathered_m,
            off_k_x // cfg.DIV_FACTOR_X,
        )
    else:
        pgm = MoEPipelinedProgram.initialize(
            cfg,
            x_desc,
            w_desc,
            x_scale_desc,
            w_scale_desc,
            gathered_m,
            off_k_x // cfg.DIV_FACTOR_X,
        )

    loop_k = K - off_k_x
    if PINGPONG:
        acc = pgm.warp_pipeline(loop_k)
    else:
        acc = pgm.pipeline(loop_k)
    if XGlobalScale is not None and not cfg.WITH_X_MX_SCALE:
        acc *= gl.load(XGlobalScale).to(gl.float32)

    # bias
    b_dtype = B.dtype if B is not None else gl.float32
    BLOCKED_LAYOUT_BIAS: gl.constexpr = get_blocked_layout(
        [BLOCK_N], b_dtype, cfg.NUM_WARPS, 1
    )
    offs_bias_n = BLOCK_N * pid_n + gl.arange(0, BLOCK_N, BLOCKED_LAYOUT_BIAS)
    mask_bias_n = offs_bias_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_bias_n
        if pid_k == 0:
            bias = gl.load(BPtrs, mask=mask_bias_n, other=0)
        else:
            bias = gl.full([BLOCK_N], 0, dtype=gl.float32, layout=BLOCKED_LAYOUT_BIAS)
    else:
        bias = gl.full([BLOCK_N], 0, dtype=gl.float32, layout=BLOCKED_LAYOUT_BIAS)

    bias = gl.convert_layout(bias, gl.SliceLayout(0, cfg.acc_layout))
    acc += bias[None, :]

    gl.static_assert(
        not (DO_SWIGLU and DO_SITU),
        "SwiGLU and SiTU cannot both be enabled",
    )
    if DO_SITU:
        out = _situ_gfx1250(acc, SITU_BETA, SITU_LINEAR_BETA)
        gl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
    elif DO_SWIGLU:
        out = _swiglu_gfx1250(acc, SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA)
        gl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
    else:
        out = acc
        gl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )

    if YMxScale is not None:
        gl.static_assert(
            WriteBackIndx is None and YGlobalScale is None,
            "MXFP4 output supports neither scatter nor a global output scale",
        )
        store_mxfp4_tile_gfx1250(
            out,
            Y_ptr,
            stride_y_m,
            YMxScale,
            stride_y_mx_m,
            start_m + off_m,
            eM - off_m,
            OUT_BLOCK_N * pid_n,
            yN,
            cfg.NUM_WARPS,
        )
        return

    BLOCKED_LAYOUT_Y: gl.constexpr = get_blocked_layout(
        [BLOCK_M, OUT_BLOCK_N], Y.dtype, cfg.NUM_WARPS
    )
    if YGlobalScale is not None:
        out = out * (1.0 / gl.load(YGlobalScale).to(gl.float32))
    out = out.to(Y.dtype.element_ty)
    out = gl.convert_layout(out, BLOCKED_LAYOUT_Y)

    OUTPUT_SHARED_LAYOUT: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[OUT_BLOCK_N, 4]], [BLOCK_M, OUT_BLOCK_N], [1, 0]
    )
    out_smem = gl.allocate_shared_memory(
        Y.dtype.element_ty, (BLOCK_M, OUT_BLOCK_N), OUTPUT_SHARED_LAYOUT
    )
    out_smem.store(out)

    if WriteBackIndx is not None:
        WriteBackIndx += start_m

        IDX_BASE_LAYOUT: gl.constexpr = get_tdm_gather_scatter_idx_layout(
            BLOCK_M, cfg.NUM_WARPS
        )
        IDX_LAYOUT: gl.constexpr = gl.SliceLayout(0, IDX_BASE_LAYOUT)

        idx_offs = gl.arange(0, BLOCK_M, IDX_LAYOUT)
        idx_mask = (off_m + idx_offs < eM) & (
            start_m + off_m + idx_offs < writeback_size
        )
        dst_row_indices = gl.load(
            WriteBackIndx + off_m + idx_offs, mask=idx_mask, other=writeback_size
        )
        dst_row_indices = dst_row_indices.to(cfg.index_type)

        y_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=Y_ptr,
            shape=(writeback_size, yN),
            strides=(stride_y_m, stride_y_n),
            block_shape=(BLOCK_M, OUT_BLOCK_N),
            layout=OUTPUT_SHARED_LAYOUT,
        )

        # TDM descriptor offsets do not support i64
        col_offset = OUT_BLOCK_N * _enforce_wave_uniform_i32(pid_n.to(gl.int32))
        y_desc_s = gl.amd.cdna5.tdm.update_tensor_descriptor(
            y_desc, add_offsets=[0, col_offset], clamp_bounds=True
        )
        gl.amd.cdna5.tdm.async_scatter(y_desc_s, dst_row_indices, out_smem)
        gl.amd.cdna5.tdm.async_wait(0)
    else:
        y_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=Y_ptr + start_m * stride_y_m,
            shape=(eM, yN),
            strides=(stride_y_m, stride_y_n),
            block_shape=(BLOCK_M, OUT_BLOCK_N),
            layout=OUTPUT_SHARED_LAYOUT,
        )
        # TDM descriptor offsets do not support i64
        gl.amd.cdna5.tdm.async_store(
            y_desc,
            [off_m.to(gl.int32), (OUT_BLOCK_N * pid_n).to(gl.int32)],
            out_smem,
        )
        gl.amd.cdna5.tdm.async_wait(0)


def _can_overflow_int32(tensor: Any) -> bool:
    if tensor is None:
        return False
    data = tensor.storage.data if isinstance(tensor, Tensor) else tensor
    shape = data.shape
    strides = data.stride()
    offset = 0
    for dim, stride in zip(shape, strides):
        offset += (dim - 1) * stride
    return offset > (1 << 31) - 1


def should_upcast_indices(*args: Any) -> bool:
    return any(_can_overflow_int32(arg) for arg in args if arg is not None)


def _canonicalize_storage(storage: Storage, out_ndim: int):
    assert out_ndim >= storage.data.ndim
    new_shape = [1] * (out_ndim - storage.data.ndim) + list(storage.data.shape)
    new_stride = [0] * (out_ndim - storage.data.ndim) + list(storage.data.stride())
    data = storage.data.as_strided(new_shape, new_stride)
    return Storage(data, storage.layout)


def _as_tensor(
    obj: torch.Tensor | Tensor | None, *, dtype: Any | None = None
) -> Tensor | None:
    if obj is None or isinstance(obj, Tensor):
        return obj
    return wrap_torch_tensor(obj, dtype=dtype)


def _mark_scale_preshuffled(scale: Tensor | None, enabled: bool) -> Tensor | None:
    if scale is not None and enabled:
        scale.storage.layout = _NamedScaleLayout("GFX1250_SCALE")
    return scale


def _as_scalar_scale(
    scale: torch.Tensor | float | None,
    name: str,
    device: torch.device,
) -> torch.Tensor | None:
    """Normalize a scalar scale argument to a one-element fp32 device tensor."""
    if scale is None:
        return None
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        return scale.to(device=device, dtype=torch.float32).contiguous()
    return torch.tensor([float(scale)], device=device, dtype=torch.float32)


def _activation_config(fused_activation: FusedActivation | None):
    if fused_activation is None:
        return False, 0.0, 0.0, 0.0, False, 0.0, 0.0, 1
    specs = fused_activation.specs
    if specs.name == FnSpecs.default().name:
        return False, 0.0, 0.0, 0.0, False, 0.0, 0.0, 1
    if specs.name == "situ":
        if len(fused_activation.fn_args) < 2:
            raise ValueError("SiTU activation requires beta and linear_beta")
        situ_beta = float(fused_activation.fn_args[0])
        situ_linear_beta = float(fused_activation.fn_args[1])
        if situ_beta <= 0.0 or situ_linear_beta <= 0.0:
            raise ValueError("SiTU beta and linear_beta must be positive")
        return (
            False,
            0.0,
            0.0,
            0.0,
            True,
            situ_beta,
            situ_linear_beta,
            int(specs.reduction_n),
        )
    if specs.name != "swiglu":
        raise NotImplementedError(
            "gfx1250 MoE only supports no activation, SwiGLU, or SiTU, "
            f"got {specs.name!r}"
        )
    if len(fused_activation.fn_args) < 2:
        raise ValueError("SwiGLU activation requires at least alpha and limit")
    alpha = float(fused_activation.fn_args[0])
    limit = float(fused_activation.fn_args[1])
    beta = (
        float(fused_activation.fn_args[2])
        if len(fused_activation.fn_args) >= 3
        else 1.0
    )
    return True, alpha, limit, beta, False, 0.0, 0.0, int(specs.reduction_n)


def _validate_schedule(
    *,
    schedule: str,
    pingpong: bool,
    num_buffers: int,
    block_n: int,
    block_k: int,
    num_warps: int,
) -> None:
    if schedule not in ("baseline", "sliceK", "sliceNK"):
        raise ValueError(
            f"schedule must be 'baseline', 'sliceK', or 'sliceNK', got {schedule!r}"
        )
    if schedule == "sliceNK":
        if block_k < 256 or block_n < 256:
            raise ValueError("sliceNK requires block_k >= 256 and block_n >= 256")
        if pingpong or num_buffers != 2:
            raise ValueError("sliceNK supports only num_buffers=2 and pingpong=False")
    if schedule == "sliceK":
        if block_k < 256:
            raise ValueError("sliceK requires block_k >= 256")
        if num_buffers not in (2, 3):
            raise ValueError("sliceK supports only num_buffers 2 or 3")
    if pingpong:
        if num_warps != 8:
            raise ValueError("pingpong requires num_warps=8")
        if schedule not in ("baseline", "sliceK"):
            raise ValueError("pingpong supports only baseline and sliceK schedules")
        if num_buffers != 3:
            raise ValueError("pingpong requires num_buffers=3")


def _resolve_block_m(
    decode: bool,
    m: int,
    num_experts: int | None,
) -> int:
    """Use stage defaults for prefill and expert occupancy for decode."""
    if not decode:
        return 64
    rows_per_expert = max(1, m // num_experts)
    return max(16, min(triton.next_power_of_2(rows_per_expert), 128))


# TDM zero-extends 16-bit indices when packing them (TDMUtility.cpp), so gl.int16
# names the width only and the field spans the full unsigned range.
_UINT16_MAX = (1 << 16) - 1


def select_tdm_index_width_bits(
    *,
    gather_input_rows: int | None,
    scatter_writeback_rows: int | None,
) -> int:
    """Return the narrowest TDM index width, in bits, that fits every index.
    Both directions share one index field, so the width must hold the largest
    value either emits.

    The two largest values differ by one. A gather emits row ordinals, so its
    largest is ``gather_input_rows - 1``. A scatter's masked-off lanes emit
    ``scatter_writeback_rows`` itself, one row past the last.
    """
    if gather_input_rows is None and scatter_writeback_rows is None:
        return 32
    if gather_input_rows is not None and gather_input_rows - 1 > _UINT16_MAX:
        return 32
    if scatter_writeback_rows is not None and scatter_writeback_rows > _UINT16_MAX:
        return 32
    return 16


def get_tdm_index_type(
    a: Tensor,
    gather_indx: torch.Tensor | None,
    scatter_indx: torch.Tensor | None,
) -> gl.dtype:
    """Pick the TDM gather/scatter index type for this launch."""
    gather_input_rows = None if gather_indx is None else int(a.shape_max[-2])
    scatter_writeback_rows = (
        None if scatter_indx is None else int(scatter_indx.shape[0])
    )
    width_bits = select_tdm_index_width_bits(
        gather_input_rows=gather_input_rows,
        scatter_writeback_rows=scatter_writeback_rows,
    )
    return gl.int16 if width_bits == 16 else gl.int32


def matmul(
    a,
    b,
    bias,
    a_ragged_metadata: RaggedTensorMetadata | None = None,
    b_ragged_metadata: RaggedTensorMetadata | None = None,
    gather_indx: torch.Tensor | None = None,
    scatter_indx: torch.Tensor | None = None,
    precision_config: PrecisionConfig | None = None,
    fused_activation: FusedActivation | None = None,
    *,
    x_global_scale: torch.Tensor | float | None = None,
    y_global_scale: torch.Tensor | float | None = None,
    y_mx_scale: torch.Tensor | None = None,
    num_buffers: int = 2,
    scale_block: int = 32,
    block_m: int,
    block_n: int = 128,
    block_k: int = 256,
    group_m: int = 8,
    xcd_swizzle: int = 1,
    w_transpose: bool = True,
    scale_preshuffle: bool | None = None,
    schedule: str = "baseline",
    pingpong: bool = False,
    num_warps: int = 4,
    decode: bool = False,
    partial_tdm: bool,
):
    """Run the gfx1250 Gluon MoE matmul kernel.

    Args:
        a: Dense or expert-routed activation tensor. FP8 tensors should use a
            torch float8 dtype; MXFP4 tensors should use packed ``torch.uint8``.
        b: Expert weight tensor in ``(E, K_packed, N)`` or dense ``(K_packed, N)``
            layout. Packed MXFP4 weights use ``torch.uint8`` storage.
        bias: Optional expert bias tensor with shape ``(E, N)`` or ``(N,)``.
        a_ragged_metadata: Expert-row metadata for routed dispatch/combine.
        b_ragged_metadata: Reserved for K-ragged weights; currently unsupported.
        gather_indx: Optional source row indices for dispatch.
        scatter_indx: Optional destination row indices for combine writeback.
        precision_config: MX scale/output dtype configuration.
        x_global_scale: Optional scalar activation dequantization scale.
        y_global_scale: Optional scalar scale the result is divided by before
            it is cast to the output dtype. Combined with an FP8 ``out_dtype``
            this quantizes the result in the epilogue, so a chained matmul can
            consume it without a separate quantize pass.
        y_mx_scale: Optional uint8 ``(rows, N_out // 32)`` buffer. When given,
            the result is written as packed MXFP4 (``N_out // 2`` bytes per
            row) with its e8m0 block scales here, quantized from the FP32
            result with the math of
            ``launch_triton_quantize_mxfp4_activation_gfx1250``.
        fused_activation: Optional SwiGLU or SiTU activation descriptor.
        block_m: Concrete row tile resolved by the caller.
        decode: Select the small-M, M-ragged decode kernel.
        partial_tdm: Split each TDM descriptor load across half the warps so a
            pair of operand loads issues as one fused TDM operation. Requires
            4 or 8 warps. The decode kernel pairs each weight load with its
            scale load instead, and requires weight scales.

    Returns:
        ``(output, kernel)`` where ``kernel`` is the Triton/Gluon launch object.
    """
    if b_ragged_metadata is not None:
        raise NotImplementedError(
            "gfx1250 MoE matmul does not support K-ragged weights"
        )
    if decode:
        schedule = "baseline"
        pingpong = False
    _validate_schedule(
        schedule=schedule,
        pingpong=pingpong,
        num_buffers=num_buffers,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
    )
    if partial_tdm and num_warps not in (4, 8):
        raise ValueError(f"partial_tdm requires 4 or 8 warps, got {num_warps}")

    if precision_config is None:
        precision_config = PrecisionConfig()
    fused_activation = fused_activation or FusedActivation(FnSpecs.default(), tuple())
    (
        do_swiglu,
        swiglu_alpha,
        swiglu_limit,
        swiglu_beta,
        do_situ,
        situ_beta,
        situ_linear_beta,
        activation_reduction_n,
    ) = _activation_config(fused_activation)

    a_torch = a.storage.data if isinstance(a, Tensor) else a
    b_torch = b.storage.data if isinstance(b, Tensor) else b
    is_input_batched = a_torch.ndim == 3
    has_scatter = scatter_indx is not None
    is_a_ragged = a_ragged_metadata is not None
    ragged_dimension = "M" if is_a_ragged else None
    if decode and is_input_batched:
        raise ValueError("decode kernel does not support dense-batched matmul")

    M = int(a_torch.shape[-2] if gather_indx is None else gather_indx.shape[0])
    K = int(a_torch.shape[-1])
    K_W, N = map(int, b_torch.shape[-2:])
    if a_torch.dtype == torch.uint8:
        K *= 2
    if b_torch.dtype == torch.uint8:
        K_W *= 2
    if K != K_W:
        raise ValueError(f"K mismatch: activation K={K} vs weight K={K_W}")

    out_dtype = precision_config.out_dtype or (
        a_torch.dtype if a_torch.dtype.is_floating_point else torch.bfloat16
    )

    if not isinstance(a, Tensor):
        a = wrap_torch_tensor(a, dtype=a_torch.dtype)
    if not isinstance(b, Tensor):
        if b_torch.stride(-2) != 1:
            b_torch = b_torch.transpose(-1, -2).contiguous().transpose(-1, -2)
            b = b_torch
        b_dtype = FP4 if b_torch.dtype == torch.uint8 else b_torch.dtype
        b = wrap_torch_tensor(b, dtype=b_dtype)

    index_type = get_tdm_index_type(a, gather_indx, scatter_indx)

    a_scale = _as_tensor(precision_config.a_mx_scale)
    b_scale = _as_tensor(precision_config.b_mx_scale)
    if a_scale is not None:
        a_scale.storage.data = a_scale.storage.data.view(torch.uint8)
        a_scale.dtype = torch.uint8
    if b_scale is not None:
        b_scale.storage.data = b_scale.storage.data.view(torch.uint8)
        b_scale.dtype = torch.uint8
    if scale_preshuffle is None:
        scale_preshuffle = False
    a_scale = _mark_scale_preshuffled(a_scale, bool(scale_preshuffle))
    b_scale = _mark_scale_preshuffled(b_scale, bool(scale_preshuffle))

    batch_size = b.shape[0] if ragged_dimension is None and b.ndim == 3 else 1
    opt_flags = _OptFlags(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=group_m,
        xcd_swizzle=xcd_swizzle,
    )

    expt_data_w = (None,) * 6
    expt_data_x = (
        (None,) * 6
        if ragged_dimension is None
        else ragged_metadata_fields(a_ragged_metadata, block_m)
    )

    y_rows = int(scatter_indx.shape[0] if scatter_indx is not None else M)
    y_cols = N // activation_reduction_n
    if y_mx_scale is not None:
        out_block_n = block_n // activation_reduction_n
        if has_scatter or batch_size != 1 or y_global_scale is not None:
            raise ValueError(
                "MXFP4 output supports neither scatter, batching, nor y_global_scale"
            )
        if y_cols % MXFP4_BLOCK or out_block_n % MXFP4_BLOCK:
            raise ValueError(
                f"MXFP4 output needs {MXFP4_BLOCK}-aligned columns and tiles, got "
                f"{y_cols} columns in {out_block_n}-column tiles"
            )
        if (
            y_mx_scale.dtype != torch.uint8
            or tuple(y_mx_scale.shape) != (y_rows, y_cols // MXFP4_BLOCK)
            or y_mx_scale.stride(1) != 1
        ):
            raise ValueError(
                "y_mx_scale must be a uint8 (rows, N_out // 32) tensor with unit "
                f"column stride, got {y_mx_scale.dtype} {tuple(y_mx_scale.shape)}"
            )
        out_dtype = torch.uint8
        y_cols //= 2
    out_base = torch.empty(
        (batch_size, y_rows, y_cols), device=a.device, dtype=out_dtype
    )
    out_matmul = out_base[None, :, :, :]
    if has_scatter:
        c_view = out_matmul.view(math.prod(out_matmul.shape[:-1]), out_matmul.shape[-1])
    else:
        c_view = out_matmul.view(
            math.prod(out_matmul.shape[:-2]), *out_matmul.shape[-2:]
        )
    c = wrap_torch_tensor(c_view)

    grid_m = triton.cdiv(M, opt_flags.block_m)
    if ragged_dimension == "M":
        grid_m = a_ragged_metadata.n_blocks(
            a_ragged_metadata.n_slices, M, opt_flags.block_m
        )
    grid_n = triton.cdiv(N, opt_flags.block_n)
    grid = grid_m * grid_n * batch_size

    n_valid_slices = (
        a_ragged_metadata.n_slices if ragged_dimension == "M" else batch_size
    )

    a_storage = _canonicalize_storage(a.storage, 3)
    b_storage = _canonicalize_storage(b.storage, 3)
    c_storage = _canonicalize_storage(c.storage, 3)

    a_strides = [0] * (3 - a_storage.data.ndim) + list(a_storage.data.stride())
    a_scale_strides = a_scale.stride() if a_scale is not None else (None, None, None)
    a_scale_strides = (0,) * (3 - len(a_scale_strides)) + tuple(a_scale_strides)
    b_scale_strides = b_scale.stride() if b_scale is not None else (None, None, None)
    b_scale_strides = (0,) * (3 - len(b_scale_strides)) + tuple(b_scale_strides)
    bias_stride = None if bias is None else bias.stride(0)

    swizzle_mx_scale = None if b_scale is None else b_scale.storage.layout.name
    x_global_scale = _as_scalar_scale(x_global_scale, "x_global_scale", a.device)
    y_global_scale = _as_scalar_scale(y_global_scale, "y_global_scale", a.device)

    target_kernel = _matmul_decode if decode else _matmul
    kernel = target_kernel[(grid,)](
        c_storage.data,
        *out_matmul.stride(),
        y_global_scale,
        y_mx_scale,
        None if y_mx_scale is None else y_mx_scale.stride(0),
        x_global_scale,
        a_storage.data,
        *a_strides,
        a_scale,
        *a_scale_strides,
        b_storage.data,
        *b_storage.data.stride(),
        w_transpose,
        b_scale,
        *b_scale_strides,
        bias,
        bias_stride,
        M,
        N,
        K,
        K_W,
        gather_indx,
        scatter_indx,
        None if scatter_indx is None else scatter_indx.shape[0],
        ragged_dimension,
        *expt_data_x,
        *expt_data_w,
        batch_size,
        grid_m,
        grid_n,
        do_swiglu,
        swiglu_alpha,
        swiglu_limit,
        swiglu_beta,
        do_situ,
        situ_beta,
        situ_linear_beta,
        activation_reduction_n,
        n_valid_slices,
        opt_flags.block_m,
        opt_flags.block_n,
        opt_flags.block_k,
        opt_flags.group_m,
        opt_flags.xcd_swizzle,
        SWIZZLE_MX_SCALE=swizzle_mx_scale,
        EVEN_K=(K % opt_flags.block_k == 0),
        UPCAST_INDICES=should_upcast_indices(a, b, out_matmul),
        INDEX_TYPE=index_type,
        NUM_BUFFERS=num_buffers,
        SCALE_BLOCK=scale_block,
        SCHEDULE=schedule,
        PINGPONG=pingpong,
        NUM_WARPS=num_warps,
        PARTIAL_TDM=partial_tdm,
        num_warps=num_warps,
    )
    out_final = c_storage.data
    if not is_input_batched:
        out_final = out_final.squeeze(0)
    return out_final, kernel


def _adapt_index(obj: Any | None, attr: str) -> Any | None:
    if obj is None or hasattr(obj, attr):
        return obj
    if isinstance(obj, torch.Tensor):
        return type("IndxAdapter", (), {attr: obj})()
    return obj


def _index_tensor(obj: Any | None, attr: str) -> torch.Tensor | None:
    if obj is None:
        return None
    return getattr(obj, attr) if hasattr(obj, attr) else obj


def gluon_mxfp_combine(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    x_format: str = "e2m1",
    x_global_scale: torch.Tensor | float = 1.0,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    scatter_indx,
    gate_scal: torch.Tensor | None = None,
    n_tokens: int | None = None,
    n_expts_act: int | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_m: int | None = None,
    block_n: int = 256,
    block_k: int = 256,
    num_warps: int = 4,
    num_buffers: int = 3,
    xcd_swizzle: int = 1,
    use_warp_pipeline: bool | None = None,
    use_slice_mn: bool | None = None,
    use_slice_n: bool | None = None,
    scale_load_mode: str = "transpose",
    w_transpose: bool = True,
    persistent: bool | None = None,
    num_ctas: int | None = None,
    w_preshuffle: bool = False,
    x_scale_ragged_padded: bool = False,
    decode: bool = False,
    partial_tdm: bool,
) -> torch.Tensor:
    """Combine GEMM using the gfx1250 Gluon MoE kernel."""
    del use_warp_pipeline, use_slice_mn, use_slice_n
    del persistent, num_ctas, w_preshuffle, x_scale_ragged_padded
    if gate_scal is not None:
        raise NotImplementedError(
            "gfx1250 source kernel does not apply route gate scaling"
        )
    if x_format == "e2m1" and x_scale is None:
        raise ValueError("x_scale is required for e2m1/MXFP4 activation input")
    if x_format != "e2m1" and x_scale is not None:
        raise ValueError("x_scale is only supported for e2m1/MXFP4 activation input")
    scatter_tensor = _index_tensor(scatter_indx, "dst_indx")
    num_experts = None if a_ragged_metadata is None else a_ragged_metadata.n_slices
    if block_m is None:
        block_m = _resolve_block_m(decode, int(x.shape[-2]), num_experts)
    precision = PrecisionConfig(
        out_dtype=out_dtype,
        a_mx_scale=x_scale,
        b_mx_scale=w_scale,
    )
    out, _ = matmul(
        x,
        w,
        bias,
        a_ragged_metadata=a_ragged_metadata,
        scatter_indx=scatter_tensor,
        precision_config=precision,
        x_global_scale=x_global_scale,
        scale_preshuffle=(scale_load_mode == "swizzle"),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_buffers=num_buffers,
        xcd_swizzle=xcd_swizzle,
        w_transpose=w_transpose,
        decode=decode,
        partial_tdm=partial_tdm,
    )
    if n_expts_act is not None and int(n_expts_act) > 1:
        if n_tokens is None:
            if out.shape[0] % int(n_expts_act) != 0:
                raise ValueError(
                    "n_tokens is required when output rows are not divisible by n_expts_act"
                )
            n_tokens = out.shape[0] // int(n_expts_act)
        out = out.view(int(n_tokens), int(n_expts_act), out.shape[-1]).sum(dim=1)
    return out


@triton.jit
def _fp8_quantize_kernel(
    x_ptr,
    out_ptr,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SCALE_TENSOR: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    if HAS_SCALE:
        if HAS_SCALE_TENSOR:
            scale = tl.load(scale)
        x = x * (1.0 / scale)
    tl.store(out_ptr + offsets, x.to(tl.float8e4nv), mask=mask)


def _quantize_fp8_activation(
    x: torch.Tensor,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    if x.dtype is torch.float8_e4m3fn:
        return x.contiguous()
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"gfx1250 FP8 path expects bf16/fp16/fp8 input, got {x.dtype}")
    x = x.contiguous()
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("FP8 activation scale must be scalar")
        scale = scale.contiguous()
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    block_size = 4096
    _fp8_quantize_kernel[(triton.cdiv(x.numel(), block_size),)](
        x,
        out,
        1.0 if scale is None else scale,
        x.numel(),
        BLOCK_SIZE=block_size,
        HAS_SCALE=scale is not None,
        HAS_SCALE_TENSOR=isinstance(scale, torch.Tensor),
    )
    return out


_ROUTE_NB = len(RaggedTensorMetadata.block_sizes())
_ROUTE_GL_DTYPE = {
    torch.float16: gl.float16,
    torch.bfloat16: gl.bfloat16,
    torch.float32: gl.float32,
}


def _route_next_pow2(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


@gluon.jit
def _precomputed_topk_route_m1_canonical_gfx1250_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    slice_sizes_ptr,
    slice_offsets_ptr,
    block_offsets_ptr,
    block_schedule_ptr,
    gather_indices_ptr,
    scatter_indices_ptr,
    gate_scale_ptr,
    stride_ik,
    stride_wk,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    GP: gl.constexpr,
    EP: gl.constexpr,
    MAX_BLOCKS: gl.constexpr,
    MAX_BLOCKS_POW2: gl.constexpr,
    NB: gl.constexpr,
    OUTPUT_DTYPE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    stride_bo: gl.constexpr,
    stride_bs: gl.constexpr,
):
    """Build canonical metadata for valid unique M=1 routes and TOPK<=16."""
    expert_layout: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    gate_layout: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    block_layout: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])

    gate = gl.arange(0, GP, layout=gate_layout)
    gate_mask = gate < TOPK
    expert = gl.load(
        topk_ids_ptr + gate * stride_ik,
        mask=gate_mask,
        other=0,
    ).to(gl.int32)
    weight = gl.load(
        topk_weights_ptr + gate * stride_wk,
        mask=gate_mask,
        other=0.0,
    )

    expert_offset = gl.arange(0, EP, layout=expert_layout)
    expert_mask = expert_offset < E

    # Valid top-k output contains one row per selected expert. Counting those
    # rows directly avoids an expert-wide prefix scan and rank atomics.
    histogram = gl.zeros([EP], dtype=gl.int32, layout=expert_layout)
    exclusive = gl.zeros([EP], dtype=gl.int32, layout=expert_layout)
    for candidate in gl.static_range(TOPK):
        candidate_expert = gl.load(topk_ids_ptr + candidate * stride_ik).to(gl.int32)
        histogram += gl.where(
            candidate_expert == expert_offset,
            1,
            0,
        )
        exclusive += gl.where(
            candidate_expert < expert_offset,
            1,
            0,
        )

    inclusive = exclusive + histogram
    last_expert = expert_offset == (E - 1)
    gl.store(slice_sizes_ptr + expert_offset, histogram, mask=expert_mask)
    gl.store(slice_offsets_ptr + expert_offset, exclusive, mask=expert_mask)
    gl.store(
        slice_offsets_ptr + expert_offset + 1,
        inclusive,
        mask=expert_mask & last_expert,
    )

    expert_blocks = (histogram > 0).to(gl.int32)
    block_exclusive = exclusive
    block_inclusive = block_exclusive + expert_blocks
    active_blocks = gl.sum(expert_blocks, axis=0)
    block = gl.arange(0, MAX_BLOCKS_POW2, layout=block_layout)
    block_mask = block < MAX_BLOCKS
    for block_size_index in gl.static_range(NB):
        gl.store(
            block_offsets_ptr + block_size_index * stride_bo + expert_offset,
            block_exclusive,
            mask=expert_mask,
        )
        gl.store(
            block_offsets_ptr + block_size_index * stride_bo + expert_offset + 1,
            block_inclusive,
            mask=expert_mask & last_expert,
        )
        gl.store(
            block_schedule_ptr + block_size_index * stride_bs + block,
            -1,
            mask=block_mask & (block >= active_blocks),
        )
        gl.store(
            block_schedule_ptr + block_size_index * stride_bs + block_exclusive,
            expert_offset,
            mask=(histogram > 0) & expert_mask,
        )

    position = gl.gather(exclusive, expert, axis=0)
    gl.store(gather_indices_ptr + position, 0, mask=gate_mask)
    gl.store(scatter_indices_ptr + position, gate.to(gl.int32), mask=gate_mask)
    gl.store(
        gate_scale_ptr + position,
        weight.to(OUTPUT_DTYPE),
        mask=gate_mask,
    )


@gluon.jit
def _sum_and_max(sum_a, max_a, sum_b, max_b):
    return sum_a + sum_b, gl.maximum(max_a, max_b)


def _small_route_launch_metadata(grid, kernel, args):
    gates = args["M"] * args["TOPK"]
    programs = grid[0]
    experts = args["E"]
    nb = args["NB"]
    quant_bytes = 0
    if args["x_ptr"] is not None:
        values = args["M"] * args["QUANT_K"]
        programs -= triton.cdiv(args["M"], args["QUANT_BLOCK_M"]) * triton.cdiv(
            args["QUANT_K"], args["QUANT_BLOCK_K"]
        )
        quant_bytes = values * args["x_ptr"].element_size() + values // 2 + values // 32
    return {
        "name": f"{kernel.name} [G={gates}, programs={programs}]",
        # Every routing program reads all ids; outputs are the slice
        # sizes/offsets, block offsets and schedules, and the gather/scatter/
        # gate rows.
        "bytes": programs * gates * 4
        + gates * 4
        + (2 * experts + 1) * 4
        + nb * ((experts + 1) + args["MAX_BLOCKS"]) * 4
        + gates * (4 + 4 + args["gate_scale_ptr"].element_size())
        + quant_bytes,
    }


@gluon.jit(
    launch_metadata=_small_route_launch_metadata,
    do_not_specialize=["M", "MAX_BLOCKS", "stride_bs"],
)
def _precomputed_topk_route_small_m_gfx1250_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    slice_sizes_ptr,
    slice_offsets_ptr,
    block_offsets_ptr,
    block_schedule_ptr,
    gather_indices_ptr,
    scatter_indices_ptr,
    gate_scale_ptr,
    M,
    MAX_BLOCKS,
    stride_bs,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    GP: gl.constexpr,
    EP: gl.constexpr,
    CH: gl.constexpr,
    MAX_BLOCKS_POW2: gl.constexpr,
    NB: gl.constexpr,
    OUTPUT_DTYPE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    stride_bo: gl.constexpr,
    x_ptr=None,
    stride_x=0,
    q_ptr=None,
    stride_q=0,
    q_scale_ptr=None,
    stride_q_scale=0,
    QUANT_K: gl.constexpr = 0,
    QUANT_BLOCK_M: gl.constexpr = 8,
    QUANT_BLOCK_K: gl.constexpr = 512,
):
    """Stable counting sort of contiguous ``[M, TOPK]`` ids by expert.

    Program ``p < cdiv(G, CH)`` places rows ``[p * CH, (p + 1) * CH)`` at
    (rows of lower experts) + (same-expert rows in earlier chunks) +
    (same-expert rows earlier in this chunk). The ``NB`` trailing programs
    write the slice and block metadata, one block size each. Every program
    builds the full expert histogram itself, so no program waits on another.
    M, MAX_BLOCKS and the schedule stride follow the batch and stay runtime
    values; the power-of-two extents GP and MAX_BLOCKS_POW2 select a binary.

    With ``x_ptr``, programs past those quantize the ``[M, QUANT_K]`` MoE
    input to MXFP4, one ``QUANT_BLOCK_M x QUANT_BLOCK_K`` tile each, on CUs
    the routing programs leave idle.
    """
    # G is a multiple of TOPK, which keeps the masked id loads vectorized.
    G = M * TOPK
    pid = gl.program_id(0)
    n_chunks = gl.cdiv(G, CH)
    if x_ptr is not None:
        quant_tile = pid - n_chunks - NB
        if quant_tile >= 0:
            quantize_mxfp4_rows_tile_gfx1250(
                x_ptr,
                stride_x,
                q_ptr,
                stride_q,
                q_scale_ptr,
                stride_q_scale,
                M,
                quant_tile,
                QUANT_K,
                QUANT_BLOCK_M,
                QUANT_BLOCK_K,
                NUM_WARPS,
            )
            return
    start = pid * CH
    gate_layout: gl.constexpr = gl.BlockedLayout(
        [GP // (32 * NUM_WARPS)], [32], [NUM_WARPS], [0]
    )
    expert_layout: gl.constexpr = gl.BlockedLayout(
        [EP // (32 * NUM_WARPS)], [32], [NUM_WARPS], [0]
    )
    # [CH, CH] rank tile: rows across lanes, columns split across warps, so
    # each thread compares only CH * CH / (32 * NUM_WARPS) pairs.
    rank_layout: gl.constexpr = gl.BlockedLayout(
        [CH // 32, CH // NUM_WARPS], [32, 1], [1, NUM_WARPS], [1, 0]
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, rank_layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, rank_layout)
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    # Issue every global load up front; the chunk loads do not depend on the
    # histogram.
    is_chunk = pid < n_chunks
    gate = gl.arange(0, GP, layout=gate_layout)
    gate_mask = gate < G
    ids = gl.load(topk_ids_ptr + gate, mask=gate_mask, other=E)
    row = start + gl.arange(0, CH, layout=row_layout)
    row_mask = row < G
    row_ids = gl.load(topk_ids_ptr + row, mask=row_mask & is_chunk, other=E)
    row_weights = gl.load(topk_weights_ptr + row, mask=row_mask & is_chunk, other=0.0)

    # Invalid ids count in bin E, after every expert; padding lanes go to the
    # unused bin E + 1 so the atomics need no mask.
    key = gl.where((ids >= 0) & (ids < E), ids, E)
    key = gl.where(gate_mask, key, E + 1)
    # One LDS atomic per row builds both histograms: the low 16 bits count all
    # rows of an expert, the high 16 bits those before this chunk (G < 2^15).
    increment = gl.where(gate < start, 65537, 1)
    counts_smem = gl.allocate_shared_memory(
        gl.int32,
        [EP],
        shared_layout,
        gl.zeros([EP], gl.int32, layout=expert_layout),
    )
    counts_smem.atomic_scatter_add(increment, key, axis=0)
    counts = counts_smem.load(expert_layout)
    histogram = counts & 0xFFFF
    exclusive = gl.cumsum(histogram, 0) - histogram
    expert = gl.arange(0, EP, layout=expert_layout)

    if is_chunk:
        row_key = gl.where((row_ids >= 0) & (row_ids < E), row_ids, E)
        # Every lane needs every key of the chunk; an LDS broadcast avoids a
        # uniform global load per column.
        col_key = gl.convert_layout(row_key, col_layout)
        col = start + gl.arange(0, CH, layout=col_layout)
        # Rows past G carry key E and sort after every real row, so they need
        # no mask in the rank.
        earlier = gl.expand_dims(col, 0) < gl.expand_dims(row, 1)
        same = gl.expand_dims(col_key, 0) == gl.expand_dims(row_key, 1)
        rank = gl.sum((same & earlier).to(gl.int32), axis=1)
        # One LDS round trip looks up each row's base in its own layout.
        base_smem = gl.allocate_shared_memory(
            gl.int32, [EP], shared_layout, exclusive + (counts >> 16)
        )
        position = base_smem.gather(row_key, axis=0) + rank
        weight = gl.where(row_key < E, row_weights, 0.0)
        gl.store(gather_indices_ptr + position, row // TOPK, mask=row_mask)
        gl.store(scatter_indices_ptr + position, row, mask=row_mask)
        gl.store(gate_scale_ptr + position, weight.to(OUTPUT_DTYPE), mask=row_mask)
    else:
        block_size_index = pid - n_chunks
        if block_size_index == 0:
            gl.store(slice_sizes_ptr + expert, histogram, mask=expert < E)
            # Bin E is the invalid-id bin, so exclusive[E] is the valid total.
            # Storing it apart keeps the expert-range store vectorized.
            gl.store(slice_offsets_ptr + expert, exclusive, mask=expert < E)
            gl.store(slice_offsets_ptr + expert, exclusive, mask=expert == E)
        log2_block = 4 + block_size_index
        expert_blocks = gl.where(
            expert < E, (histogram + ((1 << log2_block) - 1)) >> log2_block, 0
        )
        block_exclusive = gl.cumsum(expert_blocks, 0) - expert_blocks
        active_blocks, max_expert_blocks = gl.reduce(
            (expert_blocks, expert_blocks), 0, _sum_and_max
        )
        block_offsets = block_offsets_ptr + block_size_index * stride_bo + expert
        gl.store(block_offsets, block_exclusive, mask=expert < E)
        gl.store(block_offsets, block_exclusive, mask=expert == E)
        block_layout: gl.constexpr = gl.BlockedLayout(
            [MAX_BLOCKS_POW2 // (32 * NUM_WARPS)], [32], [NUM_WARPS], [0]
        )
        block = gl.arange(0, MAX_BLOCKS_POW2, layout=block_layout)
        gl.store(
            block_schedule_ptr + block_size_index * stride_bs + block,
            gl.full([MAX_BLOCKS_POW2], -1, gl.int32, layout=block_layout),
            mask=(block < MAX_BLOCKS) & (block >= active_blocks),
        )
        # Schedule entry (block << 16) | expert for each block of each expert.
        for expert_block in range(0, max_expert_blocks):
            gl.store(
                block_schedule_ptr
                + block_size_index * stride_bs
                + block_exclusive
                + expert_block,
                (expert_block << 16) | expert,
                mask=expert_block < expert_blocks,
            )


_SMALL_ROUTE_MAX_TOKENS = 512
_SMALL_ROUTE_MAX_GATES = 2048
# A route that also quantizes the MoE input keeps the one-launch kernel up to
# its token limit: from 342 to 512 DSV4.1 tokens it beat the staged route, with
# the quantize on its stage 4, by 5-9 us per layer.
_MXFP4_SMALL_ROUTE_MAX_GATES = 3072
_SMALL_ROUTE_NUM_WARPS = 4
# Rows per placement program. Every program rebuilds the full histogram, while
# the in-chunk rank costs CHUNK^2 compares; 64 rows measured fastest from 4 to
# 64 tokens, and 32 rows only added programs.
_SMALL_ROUTE_CHUNK = 64
# MoE-input tile per quantize program riding on the small route.
_SMALL_ROUTE_QUANT_BLOCK_M = 8
_SMALL_ROUTE_QUANT_BLOCK_K = 512


def _precomputed_topk_route_small_m_gfx1250(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    quantize_into: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[
    RaggedTensorMetadata,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if (
        topk_ids.ndim != 2
        or topk_weights.shape != topk_ids.shape
        or topk_weights.dtype not in _ROUTE_GL_DTYPE
        or not topk_ids.is_cuda
        or topk_weights.device != topk_ids.device
    ):
        raise ValueError("unsupported gfx1250 small-M precomputed route")
    tokens, topk = map(int, topk_ids.shape)
    gates = tokens * topk
    if not (1 <= tokens <= _SMALL_ROUTE_MAX_TOKENS and 0 < topk <= num_experts <= 1024):
        raise ValueError("unsupported gfx1250 small-M route shape")
    if gates > _MXFP4_SMALL_ROUTE_MAX_GATES:
        raise ValueError(
            "gfx1250 small-M route supports at most "
            f"{_MXFP4_SMALL_ROUTE_MAX_GATES} rows"
        )
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()

    device = topk_ids.device
    slice_sizes = torch.empty(num_experts, dtype=torch.int32, device=device)
    slice_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    block_offsets = torch.empty(
        _ROUTE_NB,
        num_experts + 1,
        dtype=torch.int32,
        device=device,
    )
    max_blocks = RaggedTensorMetadata.max_n_blocks(num_experts, gates)
    block_schedule = torch.empty(
        _ROUTE_NB,
        max_blocks,
        dtype=torch.int32,
        device=device,
    )
    gather_indices = torch.empty(gates, dtype=torch.int32, device=device)
    scatter_indices = torch.empty(gates, dtype=torch.int32, device=device)
    gate_scale = torch.empty(gates, dtype=topk_weights.dtype, device=device)
    quant_args = {}
    quant_programs = 0
    if quantize_into is not None:
        x, packed, scale = quantize_into
        quant_k = int(x.shape[1])
        quant_programs = triton.cdiv(tokens, _SMALL_ROUTE_QUANT_BLOCK_M) * triton.cdiv(
            quant_k, _SMALL_ROUTE_QUANT_BLOCK_K
        )
        quant_args = dict(
            x_ptr=x,
            stride_x=x.stride(0),
            q_ptr=packed,
            stride_q=packed.stride(0),
            q_scale_ptr=scale,
            stride_q_scale=scale.stride(0),
            QUANT_K=quant_k,
            QUANT_BLOCK_M=_SMALL_ROUTE_QUANT_BLOCK_M,
            QUANT_BLOCK_K=_SMALL_ROUTE_QUANT_BLOCK_K,
        )
    if tokens == 1 and topk <= 16 and quantize_into is None:
        num_warps = 8
        _precomputed_topk_route_m1_canonical_gfx1250_kernel[(1,)](
            topk_ids,
            topk_weights,
            slice_sizes,
            slice_offsets,
            block_offsets,
            block_schedule,
            gather_indices,
            scatter_indices,
            gate_scale,
            topk_ids.stride(1),
            topk_weights.stride(1),
            E=num_experts,
            TOPK=topk,
            GP=_route_next_pow2(gates),
            EP=_route_next_pow2(num_experts),
            MAX_BLOCKS=max_blocks,
            MAX_BLOCKS_POW2=_route_next_pow2(max_blocks),
            NB=_ROUTE_NB,
            OUTPUT_DTYPE=_ROUTE_GL_DTYPE[topk_weights.dtype],
            NUM_WARPS=num_warps,
            stride_bo=block_offsets.stride(0),
            stride_bs=block_schedule.stride(0),
            num_warps=num_warps,
        )
    else:
        num_warps = _SMALL_ROUTE_NUM_WARPS
        lanes = 32 * num_warps
        _precomputed_topk_route_small_m_gfx1250_kernel[
            (triton.cdiv(gates, _SMALL_ROUTE_CHUNK) + _ROUTE_NB + quant_programs,)
        ](
            topk_ids,
            topk_weights,
            slice_sizes,
            slice_offsets,
            block_offsets,
            block_schedule,
            gather_indices,
            scatter_indices,
            gate_scale,
            M=tokens,
            MAX_BLOCKS=max_blocks,
            stride_bs=block_schedule.stride(0),
            E=num_experts,
            TOPK=topk,
            GP=max(_route_next_pow2(gates), lanes),
            # Bin E collects invalid ids and bin E + 1 padding rows.
            EP=max(_route_next_pow2(num_experts + 2), lanes),
            CH=_SMALL_ROUTE_CHUNK,
            MAX_BLOCKS_POW2=max(_route_next_pow2(max_blocks), lanes),
            NB=_ROUTE_NB,
            OUTPUT_DTYPE=_ROUTE_GL_DTYPE[topk_weights.dtype],
            NUM_WARPS=num_warps,
            stride_bo=block_offsets.stride(0),
            num_warps=num_warps,
            **quant_args,
        )
    ragged_metadata = RaggedTensorMetadata(
        slice_sizes,
        slice_offsets,
        block_offsets,
        block_schedule,
    )
    return ragged_metadata, gather_indices, scatter_indices, gate_scale


_LARGE_ROUTE_NUM_WARPS = 8
_LARGE_ROUTE_HISTOGRAM_NUM_WARPS = 4
_LARGE_ROUTE_SCAN_EXPERTS = 8
_LARGE_ROUTE_SCAN_NUM_WARPS = 4
# MoE-input tile per quantize program riding on the staged route's stage 4.
_LARGE_ROUTE_QUANT_BLOCK_M = 8
_LARGE_ROUTE_QUANT_BLOCK_K = 512


def _large_route_chunk(num_experts: int, gates: int) -> int:
    # Every chunk stores one count per expert, and stage 4 ranks a chunk's
    # gates pairwise. Longer chunks shrink that table but lengthen the ranking
    # loop, so the best length grows with sqrt(experts * gates).
    work = num_experts * gates
    if work < 1 << 22:
        return 32
    return 64 if work < 1 << 24 else 128


@gluon.jit(
    do_not_specialize=[
        "G",
        "NUM_PROGRAMS",
        "TOKENS_PER_PROGRAM",
        "MAX_BLOCKS",
        "stride_bs",
    ]
)
def _precomputed_topk_route_large_stage1_gfx1250_kernel(
    topk_ids_ptr,
    chunk_offsets_ptr,
    block_schedule_ptr,
    gather_indices_ptr,
    scatter_indices_ptr,
    gate_scale_ptr,
    E: gl.constexpr,
    G,
    EP: gl.constexpr,
    NUM_PROGRAMS,
    TOKENS_PER_PROGRAM,
    ROUTE_BLOCK: gl.constexpr,
    MAX_BLOCKS,
    NB: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    stride_bs,
):
    """Count one bounded route chunk and initialize all output capacity.

    The gate count, chunking and schedule extent follow the batch, so they stay
    runtime values; only the power-of-two chunk extent selects a binary.
    """
    pid = gl.program_id(0)
    route_layout: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    expert_layout: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    route = gl.arange(0, ROUTE_BLOCK, layout=route_layout)
    idx = pid * TOKENS_PER_PROGRAM + route
    route_mask = (route < TOKENS_PER_PROGRAM) & (idx < G)
    expert = gl.load(topk_ids_ptr + idx, mask=route_mask, other=0).to(gl.int32)
    valid = route_mask & (expert >= 0) & (expert < E)
    # The 1024-bin lowering gives atomic updates a 128-byte lane stride,
    # aliasing LDS atomic resources. Smaller histograms reduce that contention.
    HISTOGRAM_BINS: gl.constexpr = min(EP, 256)
    for first_expert in gl.static_range(0, EP, HISTOGRAM_BINS):
        local_expert = expert - first_expert
        local_valid = valid & (local_expert >= 0) & (local_expert < HISTOGRAM_BINS)
        histogram = gl.histogram(
            gl.where(local_valid, local_expert, 0),
            HISTOGRAM_BINS,
            mask=local_valid,
            layout=expert_layout,
        ).to(gl.int32)
        expert_offset = first_expert + gl.arange(
            0, HISTOGRAM_BINS, layout=expert_layout
        )
        gl.store(
            chunk_offsets_ptr + pid * E + expert_offset,
            histogram,
            mask=expert_offset < E,
        )

    # Stage 4 overwrites the compact valid prefix. The remaining capacity is
    # deliberately safe and zero weighted.
    gl.store(gather_indices_ptr + idx, 0, mask=route_mask)
    gl.store(scatter_indices_ptr + idx, idx.to(gl.int32), mask=route_mask)
    gl.store(gate_scale_ptr + idx, 0.0, mask=route_mask)

    # max_n_blocks(E, G) is at most G, so five passes cover all five rows.
    schedule_stride = NUM_PROGRAMS * ROUTE_BLOCK
    schedule_idx = pid * ROUTE_BLOCK + route
    for schedule_pass in gl.static_range(NB):
        schedule_offset = schedule_idx + schedule_pass * schedule_stride
        schedule_row = schedule_offset // MAX_BLOCKS
        schedule_col = schedule_offset % MAX_BLOCKS
        gl.store(
            block_schedule_ptr + schedule_row * stride_bs + schedule_col,
            -1,
            mask=(schedule_offset < NB * MAX_BLOCKS) & (schedule_row < NB),
        )


@gluon.jit(do_not_specialize=["NUM_PROGRAMS"])
def _precomputed_topk_route_large_stage2_gfx1250_kernel(
    chunk_offsets_ptr,
    expert_totals_ptr,
    E: gl.constexpr,
    NUM_PROGRAMS,
    SCAN_BLOCK: gl.constexpr,
    EXPERT_BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Turn per-chunk counts into exclusive per-expert chunk offsets."""
    # Adjacent lanes cover adjacent experts in the row-major count buffer.
    # Scan only the row axis so each expert retains its independent prefix.
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [32 // EXPERT_BLOCK, EXPERT_BLOCK], [NUM_WARPS, 1], [1, 0]
    )
    rows = gl.arange(0, SCAN_BLOCK, layout=gl.SliceLayout(1, layout))
    experts = gl.program_id(0) * EXPERT_BLOCK + gl.arange(
        0, EXPERT_BLOCK, layout=gl.SliceLayout(0, layout)
    )
    mask = (rows[:, None] < NUM_PROGRAMS) & (experts[None, :] < E)
    offsets = rows[:, None] * E + experts[None, :]
    counts = gl.load(chunk_offsets_ptr + offsets, mask=mask, other=0)
    inclusive = gl.cumsum(counts, 0)
    gl.store(chunk_offsets_ptr + offsets, inclusive - counts, mask=mask)
    gl.store(expert_totals_ptr + experts, gl.sum(counts, 0), mask=experts < E)


@gluon.jit
def _precomputed_topk_route_large_stage3_gfx1250_kernel(
    expert_totals_ptr,
    slice_sizes_ptr,
    slice_offsets_ptr,
    block_offsets_ptr,
    E: gl.constexpr,
    EP: gl.constexpr,
    NB: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    stride_bo: gl.constexpr,
):
    """Build compact slice and per-block-size expert prefix metadata."""
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    expert = gl.arange(0, EP, layout=layout)
    expert_mask = expert < E
    counts = gl.load(expert_totals_ptr + expert, mask=expert_mask, other=0)
    inclusive = gl.cumsum(counts, 0)
    exclusive = inclusive - counts
    last_expert = expert == (E - 1)
    gl.store(slice_sizes_ptr + expert, counts, mask=expert_mask)
    gl.store(slice_offsets_ptr + expert, exclusive, mask=expert_mask)
    gl.store(
        slice_offsets_ptr + expert + 1,
        inclusive,
        mask=expert_mask & last_expert,
    )

    for block_size_index in gl.static_range(NB):
        expert_blocks = (counts + (16 << block_size_index) - 1) // (
            16 << block_size_index
        )
        block_inclusive = gl.cumsum(expert_blocks, 0)
        block_exclusive = block_inclusive - expert_blocks
        gl.store(
            block_offsets_ptr + block_size_index * stride_bo + expert,
            block_exclusive,
            mask=expert_mask,
        )
        gl.store(
            block_offsets_ptr + block_size_index * stride_bo + expert + 1,
            block_inclusive,
            mask=expert_mask & last_expert,
        )


@gluon.jit(
    do_not_specialize=[
        "G",
        "NUM_PROGRAMS",
        "TOKENS_PER_PROGRAM",
        "stride_bs",
        "ROUTE_PROGRAMS",
    ]
)
def _precomputed_topk_route_large_stage4_gfx1250_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    chunk_offsets_ptr,
    expert_totals_ptr,
    slice_offsets_ptr,
    block_offsets_ptr,
    block_schedule_ptr,
    gather_indices_ptr,
    scatter_indices_ptr,
    gate_scale_ptr,
    E: gl.constexpr,
    G,
    TOPK: gl.constexpr,
    NUM_PROGRAMS,
    TOKENS_PER_PROGRAM,
    ROUTE_BLOCK: gl.constexpr,
    NB: gl.constexpr,
    OUTPUT_DTYPE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    stride_bo: gl.constexpr,
    stride_bs,
    ROUTE_PROGRAMS=0,
    x_ptr=None,
    stride_x=0,
    q_ptr=None,
    stride_q=0,
    q_scale_ptr=None,
    stride_q_scale=0,
    QUANT_K: gl.constexpr = 0,
    QUANT_BLOCK_M: gl.constexpr = 8,
    QUANT_BLOCK_K: gl.constexpr = 512,
):
    """Materialize schedules and scatter valid routes into compact slices.

    With ``x_ptr``, programs from ``ROUTE_PROGRAMS`` on quantize the MoE
    input, ``G // TOPK`` rows of ``QUANT_K`` values, to MXFP4, one
    ``QUANT_BLOCK_M x QUANT_BLOCK_K`` tile each, beside the latency-bound
    routing programs.
    """
    pid = gl.program_id(0)
    if x_ptr is not None:
        quant_tile = pid - ROUTE_PROGRAMS
        if quant_tile >= 0:
            quantize_mxfp4_rows_tile_gfx1250(
                x_ptr,
                stride_x,
                q_ptr,
                stride_q,
                q_scale_ptr,
                stride_q_scale,
                G // TOPK,
                quant_tile,
                QUANT_K,
                QUANT_BLOCK_M,
                QUANT_BLOCK_K,
                NUM_WARPS,
            )
            return
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    offset = gl.arange(0, ROUTE_BLOCK, layout=layout)

    if pid < E:
        count = gl.load(expert_totals_ptr + pid)
        for block_size_index in gl.static_range(NB):
            num_blocks = (count + (16 << block_size_index) - 1) // (
                16 << block_size_index
            )
            schedule_start = gl.load(
                block_offsets_ptr + block_size_index * stride_bo + pid
            )
            for block_start in range(0, num_blocks, ROUTE_BLOCK):
                block_id = block_start + offset
                packed = (block_id << 16) | pid
                gl.store(
                    block_schedule_ptr
                    + block_size_index * stride_bs
                    + schedule_start
                    + block_id,
                    packed,
                    mask=block_id < num_blocks,
                )

    if pid < NUM_PROGRAMS:
        idx = pid * TOKENS_PER_PROGRAM + offset
        route_mask = (offset < TOKENS_PER_PROGRAM) & (idx < G)
        expert = gl.load(topk_ids_ptr + idx, mask=route_mask, other=0).to(gl.int32)
        valid = route_mask & (expert >= 0) & (expert < E)
        safe_expert = gl.where(valid, expert, 0)
        # Vector atomics do not provide distinct return ranks when lanes alias
        # on every supported backend. Count predecessors within this bounded
        # chunk instead; stage 2 already assigned disjoint inter-chunk ranges.
        chunk_rank = gl.zeros([ROUTE_BLOCK], gl.int32, layout=layout)
        for candidate in gl.static_range(ROUTE_BLOCK):
            candidate_idx = pid * TOKENS_PER_PROGRAM + candidate
            candidate_valid = (candidate < TOKENS_PER_PROGRAM) & (candidate_idx < G)
            candidate_expert = gl.load(
                topk_ids_ptr + candidate_idx,
                mask=candidate_valid,
                other=-1,
            ).to(gl.int32)
            candidate_valid &= (candidate_expert >= 0) & (candidate_expert < E)
            chunk_rank += (
                valid
                & candidate_valid
                & (offset > candidate)
                & (expert == candidate_expert)
            ).to(gl.int32)
        position = (
            gl.load(slice_offsets_ptr + safe_expert, mask=valid, other=0)
            + gl.load(
                chunk_offsets_ptr + pid * E + safe_expert,
                mask=valid,
                other=0,
            )
            + chunk_rank
        )
        gl.store(
            gather_indices_ptr + position,
            (idx // TOPK).to(gl.int32),
            mask=valid,
        )
        gl.store(
            scatter_indices_ptr + position,
            idx.to(gl.int32),
            mask=valid,
        )
        weight = gl.load(topk_weights_ptr + idx, mask=valid, other=0.0)
        gl.store(
            gate_scale_ptr + position,
            weight.to(OUTPUT_DTYPE),
            mask=valid,
        )


def _precomputed_topk_route_large_m_gfx1250(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    quantize_into: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[
    RaggedTensorMetadata,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if (
        topk_ids.ndim != 2
        or topk_weights.shape != topk_ids.shape
        or topk_weights.dtype not in _ROUTE_GL_DTYPE
        or not topk_ids.is_cuda
        or topk_weights.device != topk_ids.device
    ):
        raise ValueError("unsupported gfx1250 large-M precomputed route")
    tokens, topk = map(int, topk_ids.shape)
    if not (tokens > 16 and 0 < topk <= 16 and 1 <= num_experts <= 1024):
        raise ValueError("unsupported gfx1250 large-M route shape")
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()

    gates = tokens * topk
    num_programs = triton.cdiv(gates, _large_route_chunk(num_experts, gates))
    tokens_per_program = triton.cdiv(gates, num_programs)
    route_block = _route_next_pow2(tokens_per_program)
    device = topk_ids.device
    slice_sizes = torch.empty(num_experts, dtype=torch.int32, device=device)
    slice_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    block_offsets = torch.empty(
        _ROUTE_NB,
        num_experts + 1,
        dtype=torch.int32,
        device=device,
    )
    max_blocks = RaggedTensorMetadata.max_n_blocks(num_experts, gates)
    block_schedule = torch.empty(
        _ROUTE_NB,
        max_blocks,
        dtype=torch.int32,
        device=device,
    )
    gather_indices = torch.empty(gates, dtype=torch.int32, device=device)
    scatter_indices = torch.empty(gates, dtype=torch.int32, device=device)
    gate_scale = torch.empty(gates, dtype=topk_weights.dtype, device=device)
    chunk_offsets = torch.empty(
        num_programs,
        num_experts,
        dtype=torch.int32,
        device=device,
    )
    expert_totals = torch.empty(num_experts, dtype=torch.int32, device=device)
    expert_pad = _route_next_pow2(num_experts)

    _precomputed_topk_route_large_stage1_gfx1250_kernel[(num_programs,)](
        topk_ids,
        chunk_offsets,
        block_schedule,
        gather_indices,
        scatter_indices,
        gate_scale,
        E=num_experts,
        G=gates,
        EP=expert_pad,
        NUM_PROGRAMS=num_programs,
        TOKENS_PER_PROGRAM=tokens_per_program,
        ROUTE_BLOCK=route_block,
        MAX_BLOCKS=max_blocks,
        NB=_ROUTE_NB,
        NUM_WARPS=_LARGE_ROUTE_HISTOGRAM_NUM_WARPS,
        stride_bs=block_schedule.stride(0),
        num_warps=_LARGE_ROUTE_HISTOGRAM_NUM_WARPS,
    )
    _precomputed_topk_route_large_stage2_gfx1250_kernel[
        (triton.cdiv(num_experts, _LARGE_ROUTE_SCAN_EXPERTS),)
    ](
        chunk_offsets,
        expert_totals,
        E=num_experts,
        NUM_PROGRAMS=num_programs,
        SCAN_BLOCK=_route_next_pow2(num_programs),
        EXPERT_BLOCK=_LARGE_ROUTE_SCAN_EXPERTS,
        NUM_WARPS=_LARGE_ROUTE_SCAN_NUM_WARPS,
        num_warps=_LARGE_ROUTE_SCAN_NUM_WARPS,
    )
    _precomputed_topk_route_large_stage3_gfx1250_kernel[(1,)](
        expert_totals,
        slice_sizes,
        slice_offsets,
        block_offsets,
        E=num_experts,
        EP=expert_pad,
        NB=_ROUTE_NB,
        NUM_WARPS=_LARGE_ROUTE_NUM_WARPS,
        stride_bo=block_offsets.stride(0),
        num_warps=_LARGE_ROUTE_NUM_WARPS,
    )
    route_programs = max(num_programs, num_experts)
    quant_args = {}
    quant_programs = 0
    if quantize_into is not None:
        x, packed, scale = quantize_into
        quant_k = int(x.shape[1])
        quant_programs = triton.cdiv(tokens, _LARGE_ROUTE_QUANT_BLOCK_M) * triton.cdiv(
            quant_k, _LARGE_ROUTE_QUANT_BLOCK_K
        )
        quant_args = dict(
            ROUTE_PROGRAMS=route_programs,
            x_ptr=x,
            stride_x=x.stride(0),
            q_ptr=packed,
            stride_q=packed.stride(0),
            q_scale_ptr=scale,
            stride_q_scale=scale.stride(0),
            QUANT_K=quant_k,
            QUANT_BLOCK_M=_LARGE_ROUTE_QUANT_BLOCK_M,
            QUANT_BLOCK_K=_LARGE_ROUTE_QUANT_BLOCK_K,
        )
    _precomputed_topk_route_large_stage4_gfx1250_kernel[
        (route_programs + quant_programs,)
    ](
        topk_ids,
        topk_weights,
        chunk_offsets,
        expert_totals,
        slice_offsets,
        block_offsets,
        block_schedule,
        gather_indices,
        scatter_indices,
        gate_scale,
        E=num_experts,
        G=gates,
        TOPK=topk,
        NUM_PROGRAMS=num_programs,
        TOKENS_PER_PROGRAM=tokens_per_program,
        ROUTE_BLOCK=route_block,
        NB=_ROUTE_NB,
        OUTPUT_DTYPE=_ROUTE_GL_DTYPE[topk_weights.dtype],
        NUM_WARPS=_LARGE_ROUTE_NUM_WARPS,
        stride_bo=block_offsets.stride(0),
        stride_bs=block_schedule.stride(0),
        num_warps=_LARGE_ROUTE_NUM_WARPS,
        **quant_args,
    )

    ragged_metadata = RaggedTensorMetadata(
        slice_sizes,
        slice_offsets,
        block_offsets,
        block_schedule,
    )
    return ragged_metadata, gather_indices, scatter_indices, gate_scale


@gluon.jit
def _topk_route_is_valid_gfx1250(
    topk_ids_ptr,
    pid_m,
    slot,
    stride_im,
    stride_ik,
    E: gl.constexpr,
    HAS_TOPK_IDS: gl.constexpr,
):
    if HAS_TOPK_IDS:
        expert = gl.load(
            topk_ids_ptr + pid_m.to(gl.int64) * stride_im + slot * stride_ik
        ).to(gl.int32)
        return (expert >= 0) & (expert < E)
    return pid_m >= 0


@gluon.jit
def _weighted_topk_reduce_gfx1250_kernel(
    flat_ptr,
    weights_ptr,
    topk_ids_ptr,
    output_ptr,
    M,
    N,
    stride_fm,
    stride_fn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    stride_om,
    stride_on,
    E: gl.constexpr,
    HAS_TOPK_IDS: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [1], [0])
    offsets_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
    mask = (pid_m < M) & (offsets_n < N)
    accumulator = gl.zeros([BLOCK_N], gl.float32, layout=layout)
    for slot in gl.static_range(TOPK):
        route_valid = _topk_route_is_valid_gfx1250(
            topk_ids_ptr,
            pid_m,
            slot,
            stride_im,
            stride_ik,
            E,
            HAS_TOPK_IDS,
        )
        weight = gl.load(
            weights_ptr + pid_m.to(gl.int64) * stride_wm + slot * stride_wk
        ).to(gl.float32)
        weight = gl.where(route_valid, weight, 0.0)
        values = gl.amd.cdna5.buffer_load(
            flat_ptr,
            (
                (pid_m.to(gl.int64) * TOPK + slot) * stride_fm
                + offsets_n.to(gl.int64) * stride_fn
            ).to(gl.int32),
            mask=mask & route_valid,
            other=0.0,
        ).to(gl.float32)
        accumulator += values * weight
    gl.amd.cdna5.buffer_store(
        accumulator.to(output_ptr.dtype.element_ty),
        output_ptr,
        (pid_m.to(gl.int64) * stride_om + offsets_n.to(gl.int64) * stride_on).to(
            gl.int32
        ),
        mask=mask,
    )


def _weighted_topk_reduce_gfx1250(
    flat: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    topk_ids: torch.Tensor | None = None,
    num_experts: int | None = None,
    out: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if flat.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("weighted top-k reduction requires rank-2 inputs")
    tokens, topk = topk_weights.shape
    if tokens <= 0 or topk <= 0:
        raise ValueError("weighted top-k reduction requires tokens and routes")
    if flat.shape[0] != tokens * topk:
        raise ValueError("flat expert rows must equal tokens * top-k")
    if topk_ids is not None:
        if (
            topk_ids.shape != topk_weights.shape
            or topk_ids.device != flat.device
            or topk_ids.dtype != torch.int32
            or topk_ids.stride(1) != 1
            or num_experts is None
            or num_experts <= 0
        ):
            raise ValueError(
                "topk_ids must be contiguous int32 weights-shaped routes with "
                "a positive num_experts"
            )
    if (
        not flat.is_cuda
        or topk_weights.device != flat.device
        or flat.stride(1) != 1
        or topk_weights.stride(1) != 1
    ):
        raise ValueError(
            "weighted top-k inputs must be colocated GPU tensors with contiguous rows"
        )
    output_shape = (tokens, flat.shape[1])
    if out is None:
        out = torch.empty(output_shape, device=flat.device, dtype=out_dtype)
    elif (
        out.shape != output_shape
        or out.dtype != out_dtype
        or out.device != flat.device
        or out.stride(-1) != 1
    ):
        raise ValueError(
            "gfx1250 weighted top-k output must be a row-contiguous view with the "
            "requested shape, dtype, and device"
        )

    block_n = 256
    grid = tokens * triton.cdiv(flat.shape[1], block_n)
    _weighted_topk_reduce_gfx1250_kernel[(grid,)](
        flat,
        topk_weights,
        topk_ids,
        out,
        tokens,
        flat.shape[1],
        flat.stride(0),
        flat.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        0 if topk_ids is None else topk_ids.stride(0),
        0 if topk_ids is None else topk_ids.stride(1),
        out.stride(0),
        out.stride(1),
        E=0 if num_experts is None else num_experts,
        HAS_TOPK_IDS=topk_ids is not None,
        TOPK=topk,
        BLOCK_N=block_n,
        num_warps=1,
    )
    return out


def _route_from_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype | None = None,
) -> tuple[
    RaggedTensorMetadata,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    valid = (flat_ids >= 0) & (flat_ids < num_experts)
    safe_ids = torch.where(valid, flat_ids, flat_ids.new_zeros(()))
    sort_order = torch.argsort(safe_ids, stable=True)

    top_k = topk_ids.shape[1]
    gather_indx = (sort_order // top_k).to(torch.int32)
    scatter_indx = sort_order.to(torch.int32)
    gate_scal = topk_weights.reshape(-1)[sort_order]
    gate_scal = torch.where(valid[sort_order], gate_scal, torch.zeros_like(gate_scal))
    if dtype is not None and gate_scal.dtype != dtype:
        gate_scal = gate_scal.to(dtype)

    col_sum = torch.zeros((num_experts,), dtype=torch.int32, device=safe_ids.device)
    col_sum.scatter_add_(0, safe_ids, valid.to(torch.int32))
    ragged_metadata = make_ragged_tensor_metadata(col_sum, int(sort_order.numel()))
    return ragged_metadata, gather_indx, scatter_indx, gate_scal


def _uses_small_route(
    topk_ids: torch.Tensor,
    num_experts: int,
    max_gates: int = _SMALL_ROUTE_MAX_GATES,
) -> bool:
    # The one-launch bounded route wins in four paired whole-model repeats and
    # preserves the canonical M1 path. Each of its programs re-reads every id,
    # so past 2K gates the staged route is cheaper.
    return (
        0 < topk_ids.shape[0] <= _SMALL_ROUTE_MAX_TOKENS
        and topk_ids.numel() <= max_gates
        and num_experts <= 1024
    )


def _uses_staged_route(topk_ids: torch.Tensor, num_experts: int) -> bool:
    return (
        topk_ids.shape[0] > 16
        and 0 < topk_ids.shape[1] <= 16
        and 1 <= num_experts <= 1024
    )


def _precomputed_topk_route_mxfp4(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    hidden_states: torch.Tensor,
):
    """Route and quantize the MoE input to MXFP4.

    Returns the ``_precomputed_topk_route`` tuple followed by the packed
    values and scales of ``launch_triton_quantize_mxfp4_activation_gfx1250``.
    Multi-token routes quantize in extra programs of a routing launch: the
    one-launch route up to its token limit, then stage 4 of the staged route.
    Other batches launch the standalone quantizer.
    """
    tokens, k = map(int, hidden_states.shape)
    if (
        tokens > 1
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and hidden_states.stride(1) == 1
        and k % MXFP4_BLOCK == 0
    ):
        if _uses_small_route(topk_ids, num_experts, _MXFP4_SMALL_ROUTE_MAX_GATES):
            route_fn = _precomputed_topk_route_small_m_gfx1250
        elif _uses_staged_route(topk_ids, num_experts):
            route_fn = _precomputed_topk_route_large_m_gfx1250
        else:
            route_fn = None
        if route_fn is not None:
            packed = torch.empty(
                (tokens, k // 2), dtype=torch.uint8, device=hidden_states.device
            )
            scale = empty_mxfp4_activation_scale(tokens, k, hidden_states.device)
            route = route_fn(
                topk_weights,
                topk_ids,
                num_experts,
                quantize_into=(hidden_states, packed, scale),
            )
            return (*route, packed, scale)
    route = _precomputed_topk_route(topk_weights, topk_ids, num_experts)
    return (*route, *launch_triton_quantize_mxfp4_activation_gfx1250(hidden_states))


def _precomputed_topk_route(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
):
    if _uses_small_route(topk_ids, num_experts):
        return _precomputed_topk_route_small_m_gfx1250(
            topk_weights,
            topk_ids,
            num_experts,
        )
    if _uses_staged_route(topk_ids, num_experts):
        return _precomputed_topk_route_large_m_gfx1250(
            topk_weights,
            topk_ids,
            num_experts,
        )
    return _route_from_topk(
        topk_weights,
        topk_ids,
        num_experts,
        dtype=topk_weights.dtype,
    )


# Prefill tiles for FP8 activations, measured at 8192-token chunks, 896
# experts, top-16, so ~146 rows per expert. Neither GEMM saturates WMMA,
# TDM, LDS or HBM; each program runs its load, LDS and WMMA phases mostly back
# to back, so the tiles are chosen for how many programs stay resident and
# how well they share L2.
# - gate/up (K=3584): with four warps each wave needs ~560 VGPRs, so a WGP
#   holds one program. Eight warps need ~170 VGPRs each, and two programs fit
#   in LDS, putting four waves on each SIMD.
# - down (K=384): 128-wide K steps cover K exactly, where 256-wide steps
#   spend a quarter of the WMMAs on zero padding, and halve the operand ring
#   so four programs fit in LDS.
# - Workgroups are dispatched round-robin over the XCDs. xcd_swizzle=8 gives
#   each XCD a contiguous run of tiles, so the M tiles reading one expert's
#   weights and the N tiles reading one activation tile share an L2.
_PREFILL_GATE_UP_TILES = {
    "block_n": 256,
    "block_k": 256,
    "num_warps": 8,
    "num_buffers": 3,
    "xcd_swizzle": 8,
}
_PREFILL_DOWN_TILES = {
    "block_n": 256,
    "block_k": 128,
    "num_warps": 4,
    "num_buffers": 3,
    "xcd_swizzle": 8,
}

# The tiles above were measured only with FP8 activations. MXFP4 activations
# (A4W4) keep the original prefill tile until they are measured on that path.
_PREFILL_MXFP4_ACTIVATION_TILES = {
    "block_n": 256,
    "block_k": 256,
    "num_warps": 4,
    "num_buffers": 3,
}

# Row pitch in bytes of the scales the decode GEMM epilogue writes. Every
# N tile stores a few bytes into each row; with packed 18-byte rows those
# partial writes pile onto the same lines and cost the HBM-bound decode GEMM
# a microsecond at 16 requests. Prefill measured best with packed rows.
_DECODE_EPILOGUE_SCALE_ROW_ALIGN = 256


def gluon_mxfp_precomputed_mxfp4_fused_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    activation_dtype: str,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    activation: str = "swiglu",
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    swiglu_beta: float = 1.0,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    decode: bool = False,
    block_m: int | None = None,
    out: torch.Tensor | None = None,
    partial_tdm: bool = False,
) -> torch.Tensor:
    """Dispatch + combine for gfx1250 MXFP4-weight MoE with precomputed top-k.

    Args:
        hidden_states: Token activations in bf16/fp16/fp8, shaped
            ``(n_tokens, hidden_size)``. The MXFP4 path takes bf16/fp16 only.
        topk_weights: Route weights, shaped ``(n_tokens, top_k)``.
        topk_ids: Expert ids, shaped ``(n_tokens, top_k)``.
        w13_weight: gfx1250-preprocessed interleaved gate/up expert weight.
        w2_weight: gfx1250-preprocessed down-projection expert weight.
        w13_mx_scale: gfx1250-swizzled MXFP4 scale for ``w13_weight``.
        w2_mx_scale: gfx1250-swizzled MXFP4 scale for ``w2_weight``.
        activation_dtype: ``"fp8"`` feeds both projections FP8 activations
            with the weights' static per-tensor scales. ``"mxfp4"`` quantizes
            the input and the BF16 intermediate to MXFP4 with dynamic
            per-32-value e8m0 scales.
        w13_bias: Optional expert bias for the gate/up projection.
        w2_bias: Optional expert bias for the down projection.
        out_dtype: Final output dtype.
        activation: Fused gate activation, either ``"swiglu"``/``"silu"`` or
            ``"situ"``.
        swiglu_alpha: SwiGLU gate scale.
        swiglu_limit: Optional SwiGLU clamp limit; ``0`` disables clamping.
        swiglu_beta: SwiGLU linear branch offset.
        situ_beta: SiTU gate clamp.
        situ_linear_beta: SiTU linear-branch clamp.
        decode: Select the small-M decode kernel for both MoE projections.
        block_m: Optional row-tile override; unset values resolve per projection.
        out: Optional destination tensor for the finalized expert output.
        partial_tdm: Fuse each projection's operand loads into one TDM
            operation issued by half the warps.

    Returns:
        Tensor shaped ``(n_tokens, hidden_size)``.
    """
    if activation_dtype not in ("fp8", "mxfp4"):
        raise ValueError(
            "gfx1250 Gluon MXFP4 MoE supports activation_dtype 'fp8' or "
            f"'mxfp4', got {activation_dtype!r}"
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

    if activation_dtype == "mxfp4":
        (
            ragged_metadata,
            gather_indx,
            scatter_indx,
            _gate_scal,
            x_mxfp4,
            x_mx_scale,
        ) = _precomputed_topk_route_mxfp4(
            topk_weights, topk_ids, num_experts, hidden_states
        )
    else:
        ragged_metadata, gather_indx, scatter_indx, _gate_scal = (
            _precomputed_topk_route(topk_weights, topk_ids, num_experts)
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
            "gfx1250 Gluon MXFP4 MoE supports activation 'silu', "
            f"'swiglu', or 'situ', got {activation!r}"
        )
    if decode:
        gate_up_tiles = decode_tile_config(int(hidden_states.shape[-1]))
        # The down projection reduces over the activation's half of W13's N.
        down_tiles = decode_tile_config(int(w13_raw.shape[-1]) // 2)
    else:
        if activation_dtype == "fp8":
            gate_up_tiles = _PREFILL_GATE_UP_TILES
            down_tiles = _PREFILL_DOWN_TILES
        else:
            gate_up_tiles = down_tiles = _PREFILL_MXFP4_ACTIVATION_TILES
    launch = dict(block_m=block_m, decode=decode, partial_tdm=partial_tdm)
    if activation_dtype == "mxfp4":
        intermediate, intermediate_mx_scale = gluon_mxfp_ragged_matmul(
            x_mxfp4,
            w13_weight,
            w13_bias,
            w_mx_scale=w13_mx_scale,
            x_mx_scale=x_mx_scale,
            x_format="e2m1",
            a_ragged_metadata=ragged_metadata,
            gather_indx=gather_indx,
            fused_activation=fused_activation,
            scale_preshuffle=True,
            mxfp4_output=True,
            **gate_up_tiles,
            **launch,
        )
        stage2_kwargs = dict(
            x_scale=intermediate_mx_scale, x_format="e2m1", x_global_scale=None
        )
    else:
        x_fp8 = _quantize_fp8_activation(hidden_states, w13_weight.act_scale)
        # The second matmul wants its activation in FP8, so the first one
        # divides by that scale and casts in its epilogue. Quantizing
        # separately would re-read and rewrite the whole intermediate for no
        # other reason.
        intermediate = gluon_mxfp_ragged_matmul(
            x_fp8,
            w13_weight,
            w13_bias,
            w_mx_scale=w13_mx_scale,
            x_format="e4m3",
            x_global_scale=w13_weight.act_scale,
            y_global_scale=w2_weight.act_scale,
            a_ragged_metadata=ragged_metadata,
            gather_indx=gather_indx,
            out_dtype=torch.float8_e4m3fn,
            fused_activation=fused_activation,
            scale_preshuffle=True,
            **gate_up_tiles,
            **launch,
        )
        stage2_kwargs = dict(x_format="e4m3", x_global_scale=w2_weight.act_scale)
    flat = gluon_mxfp_combine(
        intermediate,
        w2_weight,
        w2_mx_scale,
        **stage2_kwargs,
        bias=w2_bias,
        a_ragged_metadata=ragged_metadata,
        scatter_indx=scatter_indx,
        out_dtype=out_dtype,
        **down_tiles,
        scale_load_mode="swizzle",
        **launch,
    )
    return _weighted_topk_reduce_gfx1250(
        flat,
        topk_weights,
        # Serving top-k is valid by construction. Invalid-ID masking stays on
        # the public reduce helper; applying it to every prefill token made
        # C1/C16 prefill substantially slower than the last accepted gate.
        out=out,
        out_dtype=out_dtype,
    )


def gluon_mxfp_ragged_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    w_mx_scale: torch.Tensor,
    x_global_scale: torch.Tensor | float | None = None,
    y_global_scale: torch.Tensor | float | None = None,
    x_mx_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    x_format: str = "e4m3",
    a_ragged_metadata=None,
    gather_indx=None,
    scatter_indx=None,
    fused_activation=None,
    n_tokens=None,
    n_expts_act=None,
    mxfp4_output: bool = False,
    **extra_kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Tokenspeed-style wrapper around ``matmul`` for routed MoE calls.

    With ``mxfp4_output`` the result is returned as ``(packed, scale)``, laid
    out like ``launch_triton_quantize_mxfp4_activation_gfx1250`` output.
    """
    if out_dtype is None:
        out_dtype = x.dtype if x.dtype.is_floating_point else torch.bfloat16
    gather_indx = _adapt_index(gather_indx, "src_indx")
    scatter_indx = _adapt_index(scatter_indx, "dst_indx")
    scale_preshuffle = bool(extra_kwargs.pop("scale_preshuffle", False))
    scale_load_mode = "swizzle" if scale_preshuffle else "transpose"
    extra_kwargs.pop("scale_load_mode", None)
    w_transpose = bool(extra_kwargs.pop("w_transpose", True))
    extra_kwargs.pop("w_preshuffle", None)
    gate_scal = extra_kwargs.pop("gammas", None)
    gate_scal = extra_kwargs.pop("gate_scal", gate_scal)
    allowed = {
        "num_buffers",
        "scale_block",
        "block_m",
        "block_n",
        "block_k",
        "group_m",
        "xcd_swizzle",
        "schedule",
        "pingpong",
        "num_warps",
        "decode",
        "partial_tdm",
    }
    launch_kwargs = {k: extra_kwargs.pop(k) for k in list(extra_kwargs) if k in allowed}
    launch_kwargs.setdefault("partial_tdm", False)
    combine_launch_kwargs = {
        k: v
        for k, v in launch_kwargs.items()
        if k
        in {
            "block_m",
            "block_n",
            "block_k",
            "num_buffers",
            "num_warps",
            "xcd_swizzle",
            "decode",
            "partial_tdm",
        }
    }
    unsupported = sorted(extra_kwargs)
    if unsupported:
        raise TypeError(f"unsupported gfx1250 MoE keyword(s): {unsupported}")

    if scatter_indx is not None and gather_indx is None:
        if y_global_scale is not None:
            raise ValueError("y_global_scale is not supported on the combine path")
        if mxfp4_output:
            raise ValueError("mxfp4_output is not supported on the combine path")
        return gluon_mxfp_combine(
            x,
            w,
            w_mx_scale,
            x_scale=x_mx_scale,
            x_format=x_format,
            x_global_scale=x_global_scale,
            bias=bias,
            a_ragged_metadata=a_ragged_metadata,
            scatter_indx=scatter_indx,
            gate_scal=gate_scal,
            n_tokens=n_tokens,
            n_expts_act=n_expts_act,
            out_dtype=out_dtype,
            scale_load_mode=scale_load_mode,
            w_transpose=w_transpose,
            **combine_launch_kwargs,
        )
    if fused_activation is not None:
        if x_format == "e2m1" and x_mx_scale is None:
            raise ValueError("x_mx_scale is required for e2m1/MXFP4 activation input")
        if x_format != "e2m1" and x_mx_scale is not None:
            raise ValueError(
                "x_mx_scale is only supported for e2m1/MXFP4 activation input"
            )
        launch_kwargs.setdefault("block_n", 256)
        launch_kwargs.setdefault("block_k", 256)
        launch_kwargs.setdefault("num_warps", 4)
        launch_kwargs.setdefault("num_buffers", 3)
    precision = PrecisionConfig(
        out_dtype=out_dtype, a_mx_scale=x_mx_scale, b_mx_scale=w_mx_scale
    )
    gather_tensor = _index_tensor(gather_indx, "src_indx")
    decode = bool(launch_kwargs.pop("decode", False))
    partial_tdm = bool(launch_kwargs.pop("partial_tdm", False))
    m = int(x.shape[-2] if gather_tensor is None else gather_tensor.shape[0])
    num_experts = None if a_ragged_metadata is None else a_ragged_metadata.n_slices
    block_m = launch_kwargs.pop("block_m", None)
    if block_m is None:
        block_m = _resolve_block_m(decode, m, num_experts)
    y_mx_scale = None
    if mxfp4_output:
        w_torch = w.storage.data if isinstance(w, Tensor) else w
        reduction_n = (
            1 if fused_activation is None else _activation_config(fused_activation)[-1]
        )
        y_mx_scale = empty_mxfp4_activation_scale(
            m,
            int(w_torch.shape[-1]) // reduction_n,
            x.device,
            _DECODE_EPILOGUE_SCALE_ROW_ALIGN if decode else _SCALE_ROW_ALIGN,
        )
    out, _ = matmul(
        x,
        w,
        bias,
        a_ragged_metadata=a_ragged_metadata,
        gather_indx=gather_tensor,
        scatter_indx=_index_tensor(scatter_indx, "dst_indx"),
        precision_config=precision,
        fused_activation=fused_activation,
        block_m=block_m,
        x_global_scale=x_global_scale,
        y_global_scale=y_global_scale,
        y_mx_scale=y_mx_scale,
        scale_preshuffle=scale_preshuffle,
        w_transpose=w_transpose,
        decode=decode,
        partial_tdm=partial_tdm,
        **launch_kwargs,
    )
    if mxfp4_output:
        return out, y_mx_scale
    return out


__all__ = [
    "PrecisionConfig",
    "gluon_mxfp_combine",
    "gluon_mxfp_precomputed_mxfp4_fused_moe",
    "gluon_mxfp_ragged_matmul",
]
