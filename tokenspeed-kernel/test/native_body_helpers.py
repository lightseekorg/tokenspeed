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

"""Test kernels around the native decoder layer's device bodies (gfx950).

The bodies are device functions with no program ids of their own; these
kernels give them one role per program so the bodies can be compiled offline
(codegen tests) and run against today's kernels (numerics tests).
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention._common import InputStrides
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
    AttentionConfig,
    AttentionProgram,
    mha_decode_sliding_split_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent import (
    ERROR_ABORT,
    FIRST_KERNEL_SITE,
    Launch,
    fail_stop,
    poll_tags16,
)

_ERROR_ABORT = gl.constexpr(ERROR_ABORT)
_RECORD_SITE = gl.constexpr(FIRST_KERNEL_SITE)


def compile_gfx950(fn, signature: dict, constexprs: dict, num_warps: int, **options):
    """Compile ``fn`` for gfx950 without a GPU and return the compiled kernel.

    Every pointer argument is declared 16-byte aligned, as a launch with
    real tensors specializes it, so buffer and LDS-direct copies vectorize.
    """
    names = list(fn.arg_names)
    attrs = {
        (names.index(name),): [["tt.divisibility", 16]]
        for name, kind in signature.items()
        if kind.startswith("*")
    }
    full = dict(signature)
    full.update({name: "constexpr" for name in constexprs})
    source = gluon._runtime.GluonASTSource(fn, full, constexprs=constexprs, attrs=attrs)
    return triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": num_warps, **options},
    )


@gluon.jit
def test_launch(error_ptr, tag, deadline_ticks):
    """A ``Launch`` for a test grid that has no persistent entry: the
    workgroup's linear program id as its identity, ``error_ptr`` (int32,
    ``ERROR_WORDS`` long, zeroed) for the failure record and abort word."""
    pid = (gl.program_id(0) * gl.num_programs(1) + gl.program_id(1)) * gl.num_programs(
        2
    ) + gl.program_id(2)
    zero = pid - pid
    return Launch(
        pid,
        pid % 8,
        pid // 8,
        zero.to(gl.int64),
        tag,
        zero,
        zero,
        error_ptr,
        error_ptr + _ERROR_ABORT,
        deadline_ticks,
        error_ptr,
    )


@gluon.jit
def wait_records_tags16(record_ptr, first, count: gl.constexpr, tag, launch):
    """The layer's record poll: ``poll_tags16`` (scalar ``s_load_dwordx16
    glc``) over each 64-byte line of ``count`` int32 records; on expiry the
    launch's ``fail_stop``."""
    gl.static_assert(count % 16 == 0)
    for line in gl.static_range(count // 16):
        if not poll_tags16(record_ptr + first + 16 * line, tag, launch, True):
            fail_stop(launch, _RECORD_SITE, line)


@gluon.jit
def split_records_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr: tl.const,
    cache_seqlens_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    record_ptr,
    error_ptr,
    tag,
    deadline_ticks,
    Q_STRIDE_B: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    MAX_SEQLEN_Q: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    IS_FP8: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    """The sliding split decode with records: every program streams its split
    and publishes its record; split 0 merges once all records carry ``tag``.
    The grid is ``_mha_decode_sliding_split_atomic``'s; all programs must be
    co-resident."""
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        NUM_KV_SPLITS,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        True,
        WINDOW_LEFT,
        IS_FP8,
        k_cache_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_B, Q_STRIDE_H, Q_STRIDE_D),
    )
    program = AttentionProgram.create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        gl.program_id(0),
        gl.program_id(1),
        gl.program_id(2),
    )
    launch = test_launch(error_ptr, tag, deadline_ticks)
    mha_decode_sliding_split_gfx950(
        program, q_ptr, k_cache_ptr, v_cache_ptr, NUM_STAGES
    )
    program.publish_split_record(record_ptr, launch.tag)
    if program.split_id == 0:
        merged = program.merge_split_records(
            record_ptr, launch.tag, q_ptr, False, wait_records_tags16, launch
        )
        program.store_merged(out_ptr, merged)
