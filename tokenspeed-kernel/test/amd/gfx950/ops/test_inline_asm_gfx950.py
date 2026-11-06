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

"""Offline gfx950 compile of the kernels that contain inline asm: the gate.

The compiler places wait states and keeps values out of written registers
only for code it can see, and it does not see inside an inline asm block. So
inline asm is allowed only where no compiler-visible operation exists, and
every compiled variant here is checked for:

- asm blocks from a helper not listed in ``INLINE_ASM`` (each entry states
  why no compiler-visible operation serves and what would retire it), and a
  listed helper without an ``# asm:`` reason at its source;
- asm templates that write a register they do not declare (including the
  implicit SCC and VCC), or an output that is not early clobber before the
  template's last input read;
- asm instructions reading an MFMA result, a transcendental result or a
  VALU-written SGPR inside the producer's hazard window.

Compiles without a GPU, at 8 rows and at 16 where a 16-row launch compiles a
different variant (the few-row GEMV at 4, the largest row count it serves).
Inline asm main already had is listed apart (``INLINE_ASM_ON_MAIN``).
"""

import importlib
import inspect
import re
from functools import cache

import pytest
import torch
from asm_checks import (
    inline_asm_helpers,
    inline_asm_input_hazards,
    inline_asm_templates,
    undeclared_asm_writes,
)
from test_iris_muse_spark import _compile_gfx950
from tokenspeed_kernel.platform import ArchVersion


def _peer_signal_store():
    from tokenspeed_kernel.ops.communication._iris.all_reduce import iris_arch_policy

    return iris_arch_policy(ArchVersion(9, 5)).peer_signal_store


def _sched_options():
    from tokenspeed_kernel_amd._scheduling import sched_compile_options

    return sched_compile_options()


# ---------------------------------------------------------------------------
# The kernels, as their launches compile them
# ---------------------------------------------------------------------------


# The attention shapes of one TP8 rank: 16 query heads over 2 KV heads of 64,
# FP8 pages of 64 tokens, 8 verify positions per request.
_ATTENTION = dict(
    Q_STRIDE_B=1024,
    Q_STRIDE_H=64,
    Q_STRIDE_D=1,
    SM_SCALE=0.125 * 1.4426950408889634,
    PAGE_TABLE_STRIDE=4096,
    PAGE_SIZE=64,
    MAX_SEQLEN_Q=8,
    NUM_Q_HEADS=16,
    NUM_KV_HEADS=2,
    HEAD_DIM=64,
    BLOCK_M=64,
    BLOCK_N=64,
    NUM_WARPS=4,
    IS_FP8=True,
    NUM_STAGES=2,
)
_ATTENTION_POINTERS = {
    "q_ptr": "*fp8e4nv",
    "k_cache_ptr": "*fp8e4nv",
    "v_cache_ptr": "*fp8e4nv",
    "page_table_ptr": "*i32",
    "cache_seqlens_ptr": "*i32",
    "mid_o_ptr": "*fp32",
    "mid_lse_ptr": "*fp32",
}


def _mha_decode_split():
    # Full attention split over the KV pages; the reduce kernel merges.
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
        gluon_mha_decode_gfx950 as fn,
    )

    constexprs = dict(_ATTENTION, NUM_KV_SPLITS=8, IS_SLIDING=False, WINDOW_LEFT=-1)
    return _compile_gfx950(fn, _ATTENTION_POINTERS, constexprs, 4, (), _sched_options())


def _mha_decode_balanced():
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
        gluon_mha_decode_balanced_gfx950 as fn,
    )

    types = dict(_ATTENTION_POINTERS, num_requests="i32", schedule_ptr="*i32")
    return _compile_gfx950(
        fn, types, dict(_ATTENTION), 4, ("num_requests",), _sched_options()
    )


def _attntail_lamport():
    from tokenspeed_kernel.ops.communication._iris.muse_spark.attntail import (
        iris_fused_lamport_allreduce_muse_attntail as fn,
    )

    types = {
        "branch_ptr": "*bf16",
        "mailbox_ptr": "*bf16",
        "stream_ptr": "*fp32",
        "post_weight_ptr": "*bf16",
        "alpha_ptr": "*fp32",
        "beta_ptr": "*fp32",
        "pre_weight_ptr": "*bf16",
        "out_stream_ptr": "*fp32",
        "pre_fp32_ptr": "*fp32",
        "pre_bf16_ptr": "*bf16",
        "epochs": "*i32",
        "heaps": ("i64",) * 8,
    }
    constexprs = dict(
        RANK=5,
        WORLD_SIZE=8,
        WIDTH=8192,
        MAX_ROWS=26,
        POST_EPS=1e-6,
        POST_WEIGHT_OFFSET=1.0,
        PRE_EPS=1e-6,
        PRE_WEIGHT_OFFSET=1.0,
        POST_HAS_WEIGHT=True,
        SENDERS=8,
        ROTATE_PEERS=False,
        ELEMENTS_PER_THREAD=8,
        NUM_WARPS=16,
        PEER_SIGNAL_STORE_CACHE=_peer_signal_store(),
    )
    return _compile_gfx950(fn, types, constexprs, 16)


def _routed_lamport():
    from tokenspeed_kernel.ops.communication._iris.all_reduce import iris_arch_policy
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_fused_lamport_allreduce_routed_rmsnorm_sum as fn,
    )

    policy = iris_arch_policy(ArchVersion(9, 5))
    types = {
        "routes": "*bf16",
        "route_map": "*i32",
        "route_weights": "*fp32",
        "norm_weight": "*bf16",
        "output": "*bf16",
        "region": "*bf16",
        "epochs": "*i32",
        "regions": ("i64",) * 8,
        "row_stride": "i32",
    }
    constexprs = dict(
        RANK=5,
        WIDTH=4096,
        EPS=1e-5,
        MAX_ELEMENTS=43 * 4096,
        PORTABLE_MATH=False,
        PEER_SIGNAL_STORE_CACHE=policy.peer_signal_store,
        WAIT_STORES=policy.wait_stores,
    )
    return _compile_gfx950(fn, types, constexprs, 1)


def _rows_gemv(rows):
    # The pre-expert projection of one TP8 rank: 512 outputs over K = 8192.
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16 import decode_gemv

    fn = decode_gemv.gluon_rowcta_bf16_gemv_rows_gfx950
    block_k, waves, outputs_per_wave, num_warps = decode_gemv._rows_config(
        rows, 512, 8192
    )
    types = {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*bf16"}
    types.update(M="i32", N="i32")
    constexprs = dict(
        K=8192,
        BLOCK_M=rows,
        BLOCK_K=block_k,
        WAVES_PER_OUTPUT=waves,
        OUTPUTS_PER_WAVE=outputs_per_wave,
        NUM_WARPS=num_warps,
    )
    return _compile_gfx950(fn, types, constexprs, num_warps, ("M",), _sched_options())


_MOETAIL = dict(
    RANK=5,
    WORLD_SIZE=8,
    MAX_ELEMENTS=43 * 4096,
    POST_EPS=1e-5,
    POST_WEIGHT_OFFSET=1.0,
    PRE_EPS=1e-5,
    PRE_WEIGHT_OFFSET=1.0,
    HAS_PRE=True,
)
_MOETAIL_POINTERS = {
    "region": "*bf16",
    "epochs": "*i32",
    "regions": ("i64",) * 8,
    "stream_ptr": "*fp32",
    "post_weight_ptr": "*bf16",
    "alpha_ptr": "*fp32",
    "beta_ptr": "*fp32",
    "pre_weight_ptr": "*bf16",
    "out_stream_ptr": "*fp32",
    "pre_bf16_ptr": "*bf16",
    "M": "i32",
}


def _moetail_fused():
    from tokenspeed_kernel.ops.communication._iris.muse_spark.moetail import (
        iris_fused_gemv_lamport_allgather_muse_moetail as fn,
    )

    types = dict(_MOETAIL_POINTERS, x_ptr="*bf16", weight_ptr="*bf16")
    constexprs = dict(
        _MOETAIL,
        K=4096,
        LOCAL_WIDTH=1024,
        BLOCK_M=8,
        BLOCK_K=512,
        WAVES_PER_OUTPUT=4,
        PEER_SIGNAL_STORE_CACHE=_peer_signal_store(),
    )
    return _compile_gfx950(fn, types, constexprs, 16, ("region", "M"), _sched_options())


# The router of one rank: 256 experts over K = 8192, FP32.
def _router_gemv():
    # Up to 8 rows run the GEMV.
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp32 import router_gemv

    types = {"x_ptr": "*fp32", "weight_ptr": "*fp32", "out_ptr": "*fp32", "M": "i32"}
    constexprs = dict(
        K=8192, N=256, BLOCK_M=8, VECTOR=router_gemv._VECTOR, MULTIPLIER=1.0
    )  # fmt: skip
    return _compile_gfx950(
        router_gemv.gluon_fp32_router_gemv_gfx950,
        types,
        constexprs,
        router_gemv._NUM_WARPS,
        ("M",),
    )


def _mha_extend_warp_pipelined():
    # The 8-wave tile (16-bit K/V of head dim 64), which votes on its maxima.
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha import extend

    block_m, block_q, block_n, num_warps = extend._select_extend_tile(2048, 8, True)
    types = {
        "q_ptr": "*bf16",
        "k_cache_ptr": "*bf16",
        "v_cache_ptr": "*bf16",
        "page_table_ptr": "*i32",
        "output_ptr": "*bf16",
        "lse_ptr": "*bf16",
        "sink_ptr": "*bf16",
        "cu_seqlens_q_ptr": "*i32",
        "cache_seqlens_ptr": "*i32",
        "num_seqs": "i32",
    }
    constexprs = dict(
        Q_STRIDE_T=1024, Q_STRIDE_H=64, Q_STRIDE_D=1,
        SM_SCALE=0.125 * 1.4426950408889634, PAGE_TABLE_STRIDE=4096,
        PAGE_SIZE=64, N_HEADS=16, N_KV_HEADS=2, HEAD_DIM=64, BLOCK_M=block_m,
        BLOCK_Q=block_q, BLOCK_N=block_n, NUM_WARPS=num_warps, IS_CAUSAL=True,
        HAS_SINK=False, HAS_LSE=False, WINDOW_LEFT=-1, IS_FP8=False,
        RAGGED=False, NUM_STAGES=extend._EXTEND_STAGES,
    )  # fmt: skip
    return _compile_gfx950(
        extend.gluon_mha_extend_gfx950,
        types,
        constexprs,
        num_warps,
        ("num_seqs",),
        _sched_options(),
    )


def _mla_prefill_8wave():
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla import prefill_8wave

    config = prefill_8wave.get_config(
        q=torch.empty(1, 16, 192, dtype=torch.bfloat16),
        k=torch.empty(1, 16, 192, dtype=torch.bfloat16),
    )
    types = {
        "q_ptr": "*bf16",
        "k_ptr": "*bf16",
        "v_ptr": "*bf16",
        "output_ptr": "*bf16",
        "lse_ptr": "*fp32",
        "cu_seqlens_q_ptr": "*i32",
        "cu_seqlens_kv_ptr": "*i32",
        "batch_size": "i32",
        "max_seqlen_q": "i32",
    }
    constexprs = dict(
        Q_STRIDE_T=16 * 192, Q_STRIDE_H=192, K_STRIDE_T=16 * 192, K_STRIDE_H=192,
        V_STRIDE_T=16 * 128, V_STRIDE_H=128, O_STRIDE_T=16 * 128, O_STRIDE_H=128,
        LSE_STRIDE_T=16, LSE_STRIDE_H=1, N_HEADS=config.n_heads,
        N_KV_HEADS=config.n_kv_heads, HEAD_DIM=config.head_dim,
        ROPE_DIM=config.rope_dim, SM_SCALE=0.0721, IS_CAUSAL=True,
        HAS_LSE=False, BLOCK_M=config.block_m, BLOCK_N=config.block_n,
        NUM_WARPS=config.num_warps, IS_FP8=False,
    )  # fmt: skip
    return _compile_gfx950(
        prefill_8wave.gluon_mla_prefill_8wave_gfx950,
        types,
        constexprs,
        config.num_warps,
        ("batch_size", "max_seqlen_q"),
        dict(_sched_options(), llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),)),
    )


def _dsa_persistent_topk():
    # The persistent radix top-2048 of decode rows.
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import sparse_mla

    types = {
        "logits": "*fp32",
        "histograms": "*i32",
        "pass_arrivals": "*i32",
        "pass_done": "*i32",
        "reset_arrivals": "*i32",
        "output_counters": "*i32",
        "block_table": "*i32",
        "row_starts": "*i32",
        "row_ends": "*i32",
        "out": "*i32",
        "lens_out": "*i32",
        "logits_stride": "i32",
        "block_table_cols": "i32",
    }
    constexprs = dict(
        out_stride=2048, page_size=64, q_len_per_req=1, IS_DECODE=True,
        GROUPS_PER_ROW=4, TOPK=2048, BLOCK_N=sparse_mla._PERSISTENT_BLOCK_N,
    )  # fmt: skip
    return _compile_gfx950(
        sparse_mla._dsa_persistent_radix_topk_kernel,
        types,
        constexprs,
        sparse_mla._PERSISTENT_NUM_WARPS,
        options=_sched_options(),
    )


VARIANTS = {
    "mha_decode_split": _mha_decode_split,
    "mha_decode_balanced": _mha_decode_balanced,
    "attntail_lamport": _attntail_lamport,
    "routed_lamport": _routed_lamport,
    "rows_gemv-rows4": lambda: _rows_gemv(4),
    "moetail_fused-rows8": _moetail_fused,
    "router_gemv-rows8": _router_gemv,
    "mha_extend_warp_pipelined": _mha_extend_warp_pipelined,
    "mla_prefill_8wave": _mla_prefill_8wave,
    "dsa_persistent_topk": _dsa_persistent_topk,
}


@cache
def _compiled(name):
    kernel = VARIANTS[name]()
    return kernel.asm["amdgcn"], kernel.asm["llir"]


# ---------------------------------------------------------------------------
# The listed inline asm
# ---------------------------------------------------------------------------

# The inline asm these kernels may contain, by the helper it comes from (the
# innermost function at the asm block): (where it is defined, as
# "module:name"; why no compiler-visible operation serves; the compiler
# support that would retire it). A helper not listed here fails the gate.
_AMD = "tokenspeed_kernel_amd.ops.gfx950."
INLINE_ASM = {}
# Inline asm that main already has, which the kernels above compile in.
INLINE_ASM_ON_MAIN = {
    "_keep_in_cluster": (
        _AMD + "attention._common:_keep_in_cluster",
        "keeps the ops producing a value in its warp-pipeline cluster",
        "a Gluon op that pins a value to a program point",
    ),
    "gluon_fp32_router_gemv_gfx950": (
        _AMD + "gemm.fp32.router_gemv:gluon_fp32_router_gemv_gfx950",
        "keeps each K step's FMAs at their step (an empty template)",
        "a Gluon op that pins a value to a program point",
    ),
    "gluon_mm_a16w16_medium_gfx950": (
        _AMD + "gemm.fp16.mm:gluon_mm_a16w16_medium_gfx950",
        "waits for the wave's partial stores before the split-K arrival",
        "a release fence that waits for stores without an L2 write-back",
    ),
    "_iris_subgroup_any": (
        "tokenspeed_kernel.ops.communication._iris.all_reduce:_iris_subgroup_any",
        "a wave vote in Iris kernels, which link no library for"
        " llvm.amdgcn.ballot",
        "a Gluon ballot (wave vote) op",
    ),
    "wait_vmem": (
        "tokenspeed_kernel.ops.communication._iris.synchronization:wait_vmem",
        "s_waitcnt vmcnt(0) in Iris kernels, which link no library",
        "a release fence that waits for stores without an L2 write-back",
    ),
}  # fmt: skip
# Known input hazards, by helper, until their fix lands: the reason.
KNOWN_INPUT_HAZARDS: dict[str, str] = {}


@pytest.mark.parametrize("name", VARIANTS)
def test_inline_asm_is_listed(name):
    """Every inline asm block comes from a listed helper."""
    assembly, _ = _compiled(name)
    helpers = inline_asm_helpers(assembly)
    listed = set(INLINE_ASM) | set(INLINE_ASM_ON_MAIN)
    assert sorted(set(helpers) - listed) == [], helpers


@pytest.mark.parametrize("name", VARIANTS)
def test_inline_asm_declares_its_writes(name):
    """Inline asm writes only its outputs and declared clobbers (implicit
    SCC and VCC included), and writes no output before its last input read
    unless that output is early clobber: the compiler keeps values in every
    other register across the block."""
    _, llir = _compiled(name)
    problems = {
        template: undeclared_asm_writes(template, constraints)
        for template, constraints in inline_asm_templates(llir)
    }
    assert {t: p for t, p in problems.items() if p} == {}


@pytest.mark.parametrize("name", VARIANTS)
def test_inline_asm_inputs_are_past_their_hazards(name):
    """No inline asm instruction reads a register inside its producer's
    hazard window: the compiler pads the wait states for its own
    instructions only."""
    assembly, _ = _compiled(name)
    problems = inline_asm_input_hazards(assembly)
    unknown = [p for p in problems if p[0] not in KNOWN_INPUT_HAZARDS]
    assert unknown == []
    known = sorted({helper for helper, _ in problems})
    if known:
        pytest.xfail("; ".join(f"{h}: {KNOWN_INPUT_HAZARDS[h]}" for h in known))


@pytest.mark.parametrize("name", VARIANTS)
def test_library_calls_are_linked(name):
    """Every LLVM library function the kernel calls was linked and inlined:
    a launch without the library's ``extern_libs`` still compiles, into a
    call to an undefined symbol that fails only when the kernel loads."""
    assembly, _ = _compiled(name)
    assert re.findall(r"\b__tokenspeed_\w+@", assembly) == []


@pytest.mark.parametrize("helper", INLINE_ASM)
def test_listed_helpers_state_their_reason(helper):
    """Each listed helper's source says why it is inline asm (``# asm:``)."""
    module, _, name = INLINE_ASM[helper][0].partition(":")
    function = importlib.import_module(module)
    for part in name.split("."):
        function = getattr(function, part)
    assert "# asm:" in inspect.getsource(getattr(function, "fn", function))


def test_inline_asm_checks_catch_their_cases():
    """The inline asm checks on small cases: undeclared and input writes, an
    output that may share an input's register, and an MFMA result read by
    asm without (and with) its wait states."""
    assert undeclared_asm_writes("s_mov_b32 s80, $1\n\ts_mov_b32 $0, s80", "=s,s") == [
        "s_mov_b32 writes undeclared s80"
    ]
    assert undeclared_asm_writes("s_add_u32 $1, $1, 1", "=s,s") == [
        "s_add_u32 writes input $1",
        "s_add_u32 writes undeclared scc",
    ]
    assert undeclared_asm_writes(
        "v_add_u32 $0, $1, $2\nv_add_u32 $0, $0, $3", "=v,v,v,v"
    ) == ["output $0 written before the last input read"]
    assert (
        undeclared_asm_writes("v_add_u32 $0, $1, $2\nv_add_u32 $0, $0, $3", "=&v,v,v,v")
        == []
    )
    assert (
        undeclared_asm_writes(
            "s_xor_b32 $0, s64, $1\n\ts_or_b32 $0, $0, s80",
            "=&s,s,~{scc},~{s[64:79]},~{s80}",
        )
        == []
    )
    mfma = "\tv_mfma_f32_16x16x32_bf16 v[2:5], v[8:11], v[12:15], v[2:5]\n"
    asm = "\t;;#ASMSTART\n\tds_write_b32 v10, v2\n\t;;#ASMEND\n\ts_endpgm\n"
    assert len(inline_asm_input_hazards(mfma + asm)) == 1
    assert inline_asm_input_hazards(mfma + "\ts_nop 7\n" * 3 + asm) == []
    trans = "\tv_exp_f32_e32 v3, v1\n"
    read = "\t;;#ASMSTART\n\tv_mov_b32 v4, v3\n\t;;#ASMEND\n\ts_endpgm\n"
    assert len(inline_asm_input_hazards(trans + read)) == 1
    assert inline_asm_input_hazards(trans + "\ts_nop 0\n" + read) == []
