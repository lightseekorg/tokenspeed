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

"""Batch dimensions must not choose new AMD request-path binaries.

Only argument binding and compiler cache keys are exercised here. CPU tensors
supply types/alignment; no GPU allocation, driver handle or kernel is used.
"""

from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_triton.backends.compiler import GPUTarget
from tokenspeed_triton.compiler.compiler import ASTSource, make_backend
from tokenspeed_triton.experimental.gluon._runtime import GluonASTSource
from tokenspeed_triton.runtime.jit import create_function_from_signature

register_cuda_ci(est_time=5, suite="runtime-1gpu")
pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4, reason="gfx950 compilation contracts"
)


def _source_hash(kernel, args, kwargs, *, gluon):
    backend = make_backend(GPUTarget("hip", "gfx950", 64))
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    bound, specialization, options = binder(*args, **kwargs)
    _, signature, constants, attrs = kernel._pack_args(
        backend, kwargs, bound, specialization, options
    )
    source = GluonASTSource if gluon else ASTSource
    return source(kernel, signature, constants, attrs).hash()


@pytest.mark.parametrize("moe", [False, True])
def test_small_allreduce_payload_count_reuses_one_binary(moe):
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_lamport_allreduce,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.moe_reduce import (
        iris_k3moe_lamport_allreduce_topk,
    )

    data = torch.empty(512, dtype=torch.bfloat16)
    epochs = torch.empty(8, dtype=torch.int32)
    peers = tuple((1 << 40) + rank * (1 << 24) for rank in range(8))
    common = dict(
        RANK=0,
        WORLD_SIZE=8,
        MAX_ELEMENTS=7168 * 256,
        NUM_STAGES=3,
        PEER_SIGNAL_STORE_CACHE=".wt",
        num_warps=1,
    )
    hashes = []
    for rows in (1, 5, 16, 63):
        if moe:
            kernel = iris_k3moe_lamport_allreduce_topk
            args = (data, data, data, data, epochs, peers)
            kwargs = dict(
                common,
                TOTAL_ELEMENTS=rows * 10752,
                ROUTED_ELEMENTS=rows * 3584,
                ROUTED_WIDTH=3584,
                TOPK=8,
            )
        else:
            kernel = iris_lamport_allreduce
            args = (data, data, data, epochs, *peers)
            kwargs = dict(common, TOTAL_ELEMENTS=rows * 7168)
        hashes.append(_source_hash(kernel, args, kwargs, gluon=True))
    assert len(set(hashes)) == 1


def test_prefix_page_table_width_reuses_one_binary():
    from tokenspeed.runtime.layers.attention.chunk import (
        create_chunked_cache_kv_indices_paged,
    )

    table = torch.empty(32, dtype=torch.int32)
    lengths = torch.empty(4, dtype=torch.int32)
    cumulative = torch.empty(5, dtype=torch.int32)
    output = torch.empty(128, dtype=torch.int32)
    args = (table, lengths, lengths, cumulative, output)
    hashes = [
        _source_hash(
            create_chunked_cache_kv_indices_paged,
            args,
            dict(page_table_ptr_stride=stride, PAGE_SIZE=128),
            gluon=False,
        )
        for stride in (1, 16, 782)
    ]
    assert len(set(hashes)) == 1


def test_sharded_scorer_table_width_reuses_one_binary():
    from tokenspeed_kernel.ops.attention.dsa._triton.topk import (
        _dsa_decode_logits_fp8_kernel,
    )

    q = torch.empty(1, 32, 128, dtype=torch.bfloat16)
    cache = torch.empty(256, dtype=torch.uint8)
    weights = torch.empty(1, 32, dtype=torch.float32)
    lengths = torch.empty(1, dtype=torch.int32)
    table = torch.empty(16, dtype=torch.int32)
    logits = torch.empty(512, dtype=torch.float32)
    hashes = []
    for columns in (1, 16, 16386):
        args = (
            q,
            cache.view(torch.float8_e4m3fn),
            cache.view(torch.float32),
            weights,
            lengths,
            table,
            logits,
            columns,
            columns * 64,
            lengths,
        )
        kwargs = dict(
            EXPLICIT_ROWS=True,
            INITIAL_TOKENS=0,
            LOCAL_TOKENS=0,
            page_size=64,
            row_bytes=132,
            page_stride_bytes=8448,
            max_seq_len=columns * 64,
            num_heads=32,
            head_dim=128,
            num_groups=1,
            softmax_scale=1.0,
            q_len_per_req=1,
            BLOCK_N=64,
            BLOCK_D=64,
            num_warps=4,
            num_stages=1,
        )
        hashes.append(
            _source_hash(_dsa_decode_logits_fp8_kernel, args, kwargs, gluon=False)
        )
    assert len(set(hashes)) == 1


def test_prefix_slice_alignment_reuses_one_binary():
    from tokenspeed.runtime.layers.attention.chunk import (
        create_chunked_cache_kv_indices_paged,
    )

    table = torch.empty(16, 1024, dtype=torch.int32)
    lengths = torch.empty(128, dtype=torch.int32)
    cumulative = torch.empty(128, dtype=torch.int32)
    output = torch.empty(8192, dtype=torch.int32)
    hashes = []
    for offset, cumulative_offset in ((0, 0), (1, 1), (1, 4)):
        hashes.append(
            _source_hash(
                create_chunked_cache_kv_indices_paged,
                (
                    table,
                    lengths[offset : offset + 16],
                    lengths[offset : offset + 16],
                    cumulative[cumulative_offset : cumulative_offset + 17],
                    output,
                ),
                dict(page_table_ptr_stride=1024, PAGE_SIZE=64),
                gluon=False,
            )
        )
    assert len(set(hashes)) == 1
