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

These compile without a GPU, so they run on any host with the AMD backend.
Each test names the instruction-level contract its body promises
(NATIVE-DESIGN P-ISA): waits, cache bits, and what may not drain a stream.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("tokenspeed_kernel_amd")

from asm_checks import instruction_lines  # noqa: E402
from native_body_helpers import compile_gfx950, split_records_kernel  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.persistent import (  # noqa: E402
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


def test_split_record_follows_the_wave_drain_of_its_partial():
    """Each wave's record (one ``buffer_store_dword`` of the tag) comes after a
    ``s_waitcnt vmcnt(0)`` that follows the wave's last partial store, with no
    store between them, and the merger reads the partials ``sc0 nt`` without
    any L2 writeback or invalidate."""
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
        },
        _MUSE_SLIDING,
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
    partial_loads = [line for line in code[poll:] if line.startswith("buffer_load")]
    assert partial_loads and all(
        line.endswith("sc0 nt") for line in partial_loads
    ), partial_loads
