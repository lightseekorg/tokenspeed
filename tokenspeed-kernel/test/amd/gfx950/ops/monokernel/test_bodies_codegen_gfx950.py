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

"""Offline gfx950 codegen checks of the native decoder layer's device bodies.

These compile without a GPU, so they run on any host with the AMD backend;
the test tree's conftest needs a device, so on a GPU-less host run them with
``pytest --noconftest``.
Each test names the instruction-level contract its body promises
(the persistent layer's codegen rules): waits, cache bits, and what may not drain a stream.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("tokenspeed_kernel_amd")

from asm_checks import instruction_lines  # noqa: E402
from monokernel_body_helpers import (  # noqa: E402
    column_relay_kernel,
    compile_gfx950,
    full_records_kernel,
    head_task_kernel,
    kpart_pooler_kernel,
    kpart_pooler_steps_kernel,
    kpart_router_kernel,
    kpart_router_steps_kernel,
    qkv_task_kernel,
    qkv_task_steps_kernel,
    router_weights_paced_kernel,
    sender_kernel,
    sender_sums_kernel,
    split_records_kernel,
    topk_keys_kernel,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent import (  # noqa: E402
    persistent_compile_options,
)

# The Muse Spark sliding layer at 8 rows: FP8 q and KV, 16 q / 2 KV heads of
# 64, an 8-token verify block, 8 splits of a 2048-token window, 2 stages.
_MUSE_SLIDING = dict(
    Q_STRIDE_B=1024,
    Q_STRIDE_H=64,
    Q_STRIDE_D=1,
    SM_SCALE=0.18033688011112042,
    PAGE_TABLE_STRIDE=8192,
    PAGE_SIZE=64,
    NUM_KV_SPLITS=8,
    MAX_SEQLEN_Q=8,
    NUM_Q_HEADS=16,
    NUM_KV_HEADS=2,
    HEAD_DIM=64,
    BLOCK_M=64,
    BLOCK_N=64,
    NUM_WARPS=4,
    WINDOW_LEFT=2047,
    IS_FP8=True,
    NUM_STAGES=2,
)


# The sliding-only constants the full-attention kernels do not take.
_SPLIT_ONLY = ("NUM_KV_SPLITS", "WINDOW_LEFT")


def test_split_record_follows_the_wave_drain_of_its_partial():
    """Each wave's record (one ``buffer_store_dword`` of the tag) comes after a
    ``s_waitcnt vmcnt(0)`` that follows the wave's last partial store, with no
    store between them, and the merger reads the partials ``sc0 nt`` without
    any L2 writeback or invalidate, issuing the splits' partial loads back to
    back with no full drain among them."""
    kernel = compile_gfx950(
        split_records_kernel,
        {
            "q_ptr": "*fp8e4nv",
            "k_cache_ptr": "*fp8e4nv",
            "v_cache_ptr": "*fp8e4nv",
            "page_table_ptr": "*i32",
            "cache_seqlens_ptr": "*i32",
            "mid_o_ptr": "*fp32",
            "mid_lse_ptr": "*fp32",
            "out_ptr": "*bf16",
            "record_ptr": "*i32",
            "error_ptr": "*i32",
            "tag": "i32",
            "deadline_ticks": "i64",
            "num_blocks": "i32",
        },
        dict(_MUSE_SLIDING, MERGE="designated"),
        num_warps=_MUSE_SLIDING["NUM_WARPS"],
        **persistent_compile_options(),
    )
    code = instruction_lines(kernel.asm["amdgcn"])
    assert not [line for line in code if re.match(r"buffer_(wbl2|inv)", line)]
    # The record is the last store before the merger's scalar record poll.
    poll = next(i for i, line in enumerate(code) if line.startswith("s_load_dwordx16"))
    record = max(
        i
        for i, line in enumerate(code[:poll])
        if line.startswith("buffer_store_dword ")
    )
    stores = [
        i
        for i, line in enumerate(code[:record])
        if line.startswith(("buffer_store", "global_store")) and i != record
    ]
    assert stores, "the partial stores precede the record"
    drains = [
        i
        for i in range(stores[-1] + 1, record)
        if code[i].startswith("s_waitcnt") and "vmcnt(0)" in code[i]
    ]
    assert drains, code[stores[-1] : record + 1]
    loads = [
        i for i, line in enumerate(code) if i > poll and line.startswith("buffer_load")
    ]
    partial_loads = [code[i] for i in loads]
    assert partial_loads and all(
        line.endswith("sc0 nt") for line in partial_loads
    ), partial_loads
    # The splits' partials stream back to back: the fold waits for them one
    # at a time (counted waits) and never drains the loads still in flight.
    assert not [
        line
        for line in code[loads[0] : loads[-1]]
        if line.startswith("s_waitcnt") and "vmcnt(0)" in line
    ]
    first_wait = next(
        i for i in range(loads[0], len(code)) if code[i].startswith("s_waitcnt vmcnt")
    )
    assert (
        sum(1 for i in loads if i < first_wait) >= 16
    ), "every split's partial load is issued before the first wait"


def test_spread_merge_issues_its_rows_partials_before_the_fold():
    """In the spread merge each split reads only its share of the rows: after
    the record poll, every split's partial and LSE loads for those rows
    (``sc0 nt``) issue back to back before the first wait, with no full drain
    among them and no L2 writeback or invalidate."""
    kernel = compile_gfx950(
        split_records_kernel,
        {
            "q_ptr": "*fp8e4nv",
            "k_cache_ptr": "*fp8e4nv",
            "v_cache_ptr": "*fp8e4nv",
            "page_table_ptr": "*i32",
            "cache_seqlens_ptr": "*i32",
            "mid_o_ptr": "*fp32",
            "mid_lse_ptr": "*fp32",
            "out_ptr": "*bf16",
            "record_ptr": "*i32",
            "error_ptr": "*i32",
            "tag": "i32",
            "deadline_ticks": "i64",
            "num_blocks": "i32",
        },
        dict(_MUSE_SLIDING, MERGE="spread"),
        num_warps=_MUSE_SLIDING["NUM_WARPS"],
        **persistent_compile_options(),
    )
    code = instruction_lines(kernel.asm["amdgcn"])
    assert not [line for line in code if re.match(r"buffer_(wbl2|inv)", line)]
    poll = max(i for i, line in enumerate(code) if line.startswith("s_load_dwordx16"))
    loads = [
        i for i, line in enumerate(code) if i > poll and line.startswith("buffer_load")
    ]
    assert loads and all(code[i].endswith("sc0 nt") for i in loads), [
        code[i] for i in loads
    ]
    assert not [
        line
        for line in code[loads[0] : loads[-1]]
        if line.startswith("s_waitcnt") and "vmcnt(0)" in line
    ]
    first_wait = next(
        i for i in range(loads[0], len(code)) if code[i].startswith("s_waitcnt vmcnt")
    )
    issued = sum(1 for i in loads if i < first_wait)
    assert issued >= 2 * _MUSE_SLIDING["NUM_KV_SPLITS"], issued


def _metadata(assembly: str, key: str) -> int:
    return int(re.search(rf"\.{key}:\s+(\d+)", assembly).group(1))


def test_sender_sums_waves_with_wide_lds_reads():
    """The o_proj sender (64 columns x K 1024, 8 waves) multiplies resident
    weights with no cross-wave traffic before one LDS exchange, and reads the
    exchange as whole 8-byte runs per wave (the columns are stored fastest),
    not as scattered 4-byte reads."""
    kernel = compile_gfx950(
        sender_kernel,
        {"w_ptr": "*bf16", "x_ptr": "*bf16", "out_ptr": "*bf16", "M": "i32"},
        dict(K=1024, OUTPUTS=64, VECTOR=2),
        num_warps=8,
    )
    assembly = kernel.asm["amdgcn"]
    code = instruction_lines(assembly)
    reads = [line.split()[0] for line in code if line.startswith("ds_read")]
    assert reads and not [
        r for r in reads if r in ("ds_read_b32", "ds_read2_b32")
    ], reads
    assert len(reads) <= 4, reads
    assert sum(1 for line in code if line.startswith("v_mfma")) == 16
    assert sum(1 for line in code if line == "s_barrier") == 1
    assert _metadata(assembly, "private_segment_fixed_size") == 0


@pytest.mark.parametrize(
    "kernel,extra,constexprs,max_vgprs",
    [
        (
            kpart_router_kernel,
            {"hi_ptr": "*bf16", "lo_ptr": "*bf16"},
            dict(EXPERTS=256),
            192,
        ),
        (
            kpart_pooler_kernel,
            {"w_ptr": "*bf16"},
            dict(OUTPUTS=512, ROWS=64),
            224,
        ),
        # x built one K step at a time: one step of x and its inputs live.
        (
            kpart_router_steps_kernel,
            {"hi_ptr": "*bf16", "lo_ptr": "*bf16"},
            dict(EXPERTS=256),
            112,
        ),
        (
            kpart_pooler_steps_kernel,
            {"w_ptr": "*bf16"},
            dict(OUTPUTS=512, ROWS=64),
            144,
        ),
    ],
    ids=["router", "pooler", "router-steps", "pooler-steps"],
)
def test_kpart_consumers_fit_their_register_budget(
    kernel, extra, constexprs, max_vgprs
):
    """The router and pooler K-part tasks hold their weights and build x
    from the deferred norm within the layer's per-role register budget, with no
    scratch (a spill would drain the resident weights)."""
    signature = {
        "b_ptr": "*bf16",
        "sums_ptr": "*fp32",
        "h_ptr": "*fp32",
        "alpha_ptr": "*fp32",
        "beta_ptr": "*fp32",
        "post_ptr": "*bf16",
        "pre_ptr": "*bf16",
        **extra,
        "part_ptr": "*i64",
        "tag": "i32",
        "M": "i32",
    }
    constexprs = dict(
        WIDTH=8192, PRODUCERS=64, PARTS=8, POST_EPS=1e-6, PRE_EPS=1e-5, **constexprs
    )
    assembly = compile_gfx950(kernel, signature, constexprs, num_warps=8).asm["amdgcn"]
    assert _metadata(assembly, "private_segment_fixed_size") == 0
    assert _metadata(assembly, "vgpr_count") <= max_vgprs
    code = instruction_lines(assembly)
    stores = [line for line in code if line.startswith("buffer_store")]
    assert stores and all(line.endswith("sc0 sc1") for line in stores), stores


def test_topk_keys_stay_in_one_wave():
    """The top-8 of a row runs on its wave alone: no LDS, no barrier, no
    scratch, 64-bit key compares."""
    assembly = compile_gfx950(
        topk_keys_kernel,
        {
            "logits_ptr": "*fp32",
            "bias_ptr": "*fp32",
            "ids_ptr": "*i32",
            "weights_ptr": "*fp32",
            "M": "i32",
        },
        dict(E=256, TOPK=8, ROUTED_SCALING_FACTOR=2.5),
        num_warps=8,
    ).asm["amdgcn"]
    code = instruction_lines(assembly)
    assert not [line for line in code if line.startswith("ds_") or line == "s_barrier"]
    assert any(line.startswith("v_cmp_gt_i64") for line in code)
    assert _metadata(assembly, "private_segment_fixed_size") == 0


@pytest.mark.parametrize(
    "kernel,max_vgprs",
    [(qkv_task_kernel, 204), (qkv_task_steps_kernel, 136)],
    ids=["full-x", "x-steps"],
)
@pytest.mark.parametrize("window", [3, 9], ids=["paced", "all-at-once"])
def test_qkv_task_paces_its_weights_and_fits(window: int, kernel, max_vgprs: int):
    """The QKV task (144 rows x K/16, x built from the FFN boundary's
    deferred norm) keeps at most ``window`` weight tiles in flight with
    counted waits, never draining its stream while issuing, and fits the
    budget with no scratch: the one-task-per-workgroup form (204 VGPRs),
    less with x built one K step at a time."""
    assembly = compile_gfx950(
        kernel,
        {
            "b_ptr": "*bf16",
            "sums_ptr": "*fp32",
            "h_ptr": "*fp32",
            "alpha_ptr": "*fp32",
            "beta_ptr": "*fp32",
            "post_ptr": "*bf16",
            "pre_ptr": "*bf16",
            "w_ptr": "*bf16",
            "part_ptr": "*i64",
            "tag": "i32",
            "M": "i32",
        },
        dict(
            WIDTH=8192, N=2304, PRODUCERS=64, WINDOW=window, POST_EPS=1e-6, PRE_EPS=1e-5
        ),
        num_warps=8,
        **persistent_compile_options(),
    ).asm["amdgcn"]
    assert _metadata(assembly, "private_segment_fixed_size") == 0
    assert _metadata(assembly, "vgpr_count") <= max_vgprs
    code = instruction_lines(assembly)
    weights = [
        i
        for i, line in enumerate(code)
        if line.startswith("buffer_load_dwordx4") and line.endswith("sc0 nt")
    ]
    assert len(weights) == 18  # 9 tiles x 2 loads per wave
    waits = [
        code[i]
        for i in range(weights[0], weights[-1])
        if code[i].startswith("s_waitcnt") and "vmcnt" in code[i]
    ]
    if window >= 9:
        # Unpaced: other loads may be waited for in between, never the stream.
        assert "s_waitcnt vmcnt(0)" not in waits, waits
    else:
        # Before each tile past the window: wait until (window - 1) tiles remain.
        assert waits == [f"s_waitcnt vmcnt({2 * (window - 1)})"] * (9 - window), waits


def test_head_task_prologue_runs_on_summed_partials():
    """A PROLOGUE task sums its head's 16 K-part partials (tag-polled 64-bit
    words) and runs the GQA prologue on the rows in registers: no store and
    reload of the rows, no scratch."""
    assembly = compile_gfx950(
        head_task_kernel,
        {
            "part_ptr": "*i64",
            "q_out_ptr": "*fp8e4nv",
            "q_weight_ptr": "*bf16",
            "k_weight_ptr": "*bf16",
            "cos_sin_ptr": "*fp32",
            "positions_ptr": "*i64",
            "k_cache_ptr": "*fp8e4nv",
            "v_cache_ptr": "*fp8e4nv",
            "slots_ptr": "*i64",
            "error_ptr": "*i32",
            "tag": "i32",
            "deadline_ticks": "i64",
            "M": "i32",
        },
        dict(NUM_Q_HEADS=16, NUM_KV_HEADS=2, HEAD_DIM=64),
        num_warps=8,
        **persistent_compile_options(),
    ).asm["amdgcn"]
    assert _metadata(assembly, "private_segment_fixed_size") == 0
    code = instruction_lines(assembly)
    # The rows never go through memory: the only multi-dword stores would be
    # rows of the summed head; the prologue writes FP8 bytes.
    assert not [
        line
        for line in code
        if line.startswith(("global_store_dwordx", "buffer_store_dwordx"))
    ]
    polls = [line for line in code if line.startswith("buffer_load_dwordx2")]
    assert polls and all(line.endswith("sc0 sc1") for line in polls), polls[:4]


def test_column_relay_holds_its_k_tile_and_chains_one_accumulator():
    """Each wave of the post-expert sender holds one 512-deep K tile (16 KB,
    16 weight loads) and runs 16 MFMAs on one accumulator chain, between a
    spin for its predecessor and a hand-off (LDS writes and one
    ``ds_add_rtn``); no barrier inside the relay and no scratch."""
    assembly = compile_gfx950(
        column_relay_kernel,
        {"x_ptr": "*bf16", "w_ptr": "*bf16", "out_ptr": "*bf16", "M": "i32"},
        dict(K=4096),
        num_warps=8,
    ).asm["amdgcn"]
    assert _metadata(assembly, "private_segment_fixed_size") == 0
    code = instruction_lines(assembly)
    weights = [
        line
        for line in code
        if line.startswith("buffer_load_dwordx4") and line.endswith("sc0 nt")
    ]
    assert len(weights) == 16
    mfmas = [i for i, line in enumerate(code) if line.startswith("v_mfma")]
    assert len(mfmas) == 16
    spins = [i for i, line in enumerate(code) if line.startswith("s_cbranch_vccnz")]
    handoff = next(
        i for i, line in enumerate(code) if line.startswith("ds_add_rtn_u32")
    )
    assert spins[0] < mfmas[0] and mfmas[-1] < handoff < spins[1]
    assert not [line for line in code[spins[0] : spins[1]] if line == "s_barrier"]


def test_sender_sums_add_their_columns_in_each_thread():
    """After the relay's last hand-off the sender forms its four norm sums
    with no LDS traffic beyond reading the relayed tile and no cross-lane
    reduction, and publishes them as tagged words; no scratch."""
    assembly = compile_gfx950(
        sender_sums_kernel,
        {
            "x_ptr": "*bf16",
            "w_ptr": "*bf16",
            "h_ptr": "*fp32",
            "alpha_ptr": "*fp32",
            "beta_ptr": "*fp32",
            "post_ptr": "*bf16",
            "sums_ptr": "*i64",
            "tag": "i32",
            "M": "i32",
        },
        dict(K=4096, N=8192),
        num_warps=8,
    ).asm["amdgcn"]
    assert _metadata(assembly, "private_segment_fixed_size") == 0
    code = instruction_lines(assembly)
    handoff = max(i for i, line in enumerate(code) if line.startswith("ds_add_rtn_u32"))
    tail = code[handoff:]
    assert not [
        line
        for line in tail
        if line.startswith(("ds_write", "ds_swizzle", "ds_bpermute"))
    ]
    assert not [line for line in tail if "dpp" in line or line == "s_barrier"]
    assert [line for line in tail if line.startswith("buffer_store_dwordx4")]


def test_full_records_wait_only_for_the_partial_and_merge_issue_first():
    """Full attention's split publishes its record once its own partial
    stores (``sc0 sc1``, written through) are done while the next role's 8
    weight loads stay in flight (``vmcnt(8)``, no ``vmcnt(0)``); the merger
    reads both 64-split tiles' partials ``sc0 sc1`` back to back before its
    first wait, with no scratch."""
    assembly = compile_gfx950(
        full_records_kernel,
        {
            "q_ptr": "*fp8e4nv",
            "cache_seqlens_ptr": "*i32",
            "num_requests": "i32",
            "schedule_ptr": "*i32",
            "page_table_ptr": "*i32",
            "k_cache_ptr": "*fp8e4nv",
            "v_cache_ptr": "*fp8e4nv",
            "mid_o_ptr": "*fp32",
            "mid_lse_ptr": "*fp32",
            "out_ptr": "*bf16",
            "sink_ptr": "*fp32",
            "record_ptr": "*i32",
            "hook_ptr": "*i32",
            "error_ptr": "*i32",
            "tag": "i32",
            "deadline_ticks": "i64",
            "program_indices": "i32",
        },
        dict(
            {k: v for k, v in _MUSE_SLIDING.items() if k not in _SPLIT_ONLY},
            HAS_SINK=True,
            HOOK_LOADS=8,
        ),
        num_warps=4,
        **persistent_compile_options(),
    ).asm["amdgcn"]
    assert _metadata(assembly, "private_segment_fixed_size") == 0
    code = instruction_lines(assembly)
    partials = [
        i
        for i, line in enumerate(code)
        if line.startswith("buffer_store_dword") and line.endswith("sc0 sc1")
    ]
    assert len(partials) == 6  # 4 x 16 B of the 64 x 64 FP32 tile, 2 lse
    record = next(
        i
        for i, line in enumerate(code)
        if line.startswith("global_store_dword") and line.endswith("sc1")
    )
    waits = [line for line in code[partials[-1] : record] if "vmcnt" in line]
    assert waits == ["s_waitcnt vmcnt(8)"], waits
    reads = [
        i
        for i, line in enumerate(code)
        if line.startswith("buffer_load") and line.endswith("sc0 sc1") and i > record
    ]
    # The merge's two-tile branch: per lane 16 lse words and 16 x 16 B of
    # partial rows for each of the two tiles.
    run = reads[: 2 * (16 + 16)]
    assert sum(code[i].startswith("buffer_load_dwordx4") for i in run) == 32
    assert not [line for line in code[run[0] : run[-1]] if "vmcnt" in line]


@pytest.mark.parametrize("loads", [1, 4])
def test_router_weights_pace_by_k_step(loads: int):
    """The router task's 8 step loads per wave (hi then lo, one stream) keep
    at most ``loads`` in flight: before each load past the first ``loads``
    the wave waits until ``loads - 1`` remain (a window of one: none)."""
    assembly = compile_gfx950(
        router_weights_paced_kernel,
        {"hi_ptr": "*bf16", "lo_ptr": "*bf16", "out_ptr": "*fp32"},
        dict(LOADS=loads),
        num_warps=8,
        **persistent_compile_options(),
    ).asm["amdgcn"]
    code = instruction_lines(assembly)
    weights = [
        i
        for i, line in enumerate(code)
        if line.startswith("buffer_load_dwordx4") and line.endswith("sc0 nt")
    ]
    assert len(weights) == 8
    waits = [
        code[i]
        for i in range(weights[0], weights[-1])
        if code[i].startswith("s_waitcnt") and "vmcnt" in code[i]
    ]
    assert waits == [f"s_waitcnt vmcnt({loads - 1})"] * (8 - loads), waits
