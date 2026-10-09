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

"""GFX950 kernels launched with batch-shaped arguments must not recompile.

Same pattern as ``test/ops/test_batch_shape_recompiles.py``: warm each integer
specialization class, then sweep the batch-dependent value inside
``assert_no_triton_compile``.
"""

import importlib
import sys

import pytest
import torch
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    is_cdna4,
    warm_specialization_classes,
)

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

DEVICE = "cuda"


def test_mxfp4_activation_quantize_row_count():
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import quantize

    x = torch.randn(2000, 512, device=DEVICE, dtype=torch.bfloat16)
    full, _ = quantize._quantize_mxfp4_activation(x)

    def run(rows):
        # Rows quantize independently: a shorter batch is a prefix.
        out, _ = quantize._quantize_mxfp4_activation(x[:rows])
        torch.testing.assert_close(out, full[:rows], rtol=0, atol=0)

    # 128 rows and up take the tiled kernel, fewer the scalar one.
    for rows in (128, 129, 32, 33):
        run(rows)
    with assert_no_triton_compile(
        quantize._mxfp4_quantize_cdna4_scale_tiled_kernel,
        quantize._mxfp4_quantize_cdna4_scale_kernel,
    ):
        for rows in (130, 1483, 1800, 48, 97, 100):
            run(rows)


def test_mha_extend_split_count():
    from tokenspeed_kernel.ops.attention.mha import mha_extend_with_kvcache
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha import extend

    heads, kv_heads, dim, page, q_len = 8, 2, 128, 64, 4
    k_cache = torch.randn(65, page, kv_heads, dim, device=DEVICE, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    # A fixed table width keeps the table stride out of the sweep.
    table = torch.arange(1, 65, dtype=torch.int32, device=DEVICE)[None]
    q = torch.randn(q_len, heads, dim, device=DEVICE, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=DEVICE)

    def run(kv_len):
        # One request over two KV heads under-fills the GPU, so the split count
        # follows the page count: half the pages, clamped to [8, 32].
        out = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=torch.tensor([0, kv_len], dtype=torch.int32, device=DEVICE),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=table,
            cache_seqlens=torch.tensor([kv_len], dtype=torch.int32, device=DEVICE),
            max_seqlen_q=q_len,
            max_seqlen_k=kv_len,
            solution="gluon",
        )
        k = k_cache[1:].flatten(0, 1)[:kv_len].float().repeat_interleave(4, dim=1)
        v = v_cache[1:].flatten(0, 1)[:kv_len].float().repeat_interleave(4, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), k) * dim**-0.5
        expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)

    # 32 pages give 16 splits and 18 pages give 9: both integer classes.
    run(32 * page)
    run(18 * page)
    with assert_no_triton_compile(
        extend.gluon_mha_extend_split_gfx950, extend.gluon_mha_extend_reduce_gfx950
    ):
        # 10, 11, 13, 20 and 32 splits.
        for pages in (20, 22, 26, 40, 64):
            run(pages * page - 5)


def test_mxfp8_gemm_row_count():
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8 import mm

    n, k = 512, 1024
    a = (torch.randn(2560, k, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
    a_scales = torch.randint(124, 130, (2560, k // 32), dtype=torch.uint8)
    b_scales = torch.randint(124, 130, (n, k // 32), dtype=torch.uint8)
    a_scales, b_scales = a_scales.to(DEVICE), b_scales.to(DEVICE)

    def run(rows):
        return mm.launch_gluon_mm_mxfp8_gfx950(
            a[:rows],
            b,
            a_scales[:rows],
            b_scales,
            torch.bfloat16,
            alpha=None,
            block_size=[1, 32],
            out=None,
        )

    full = run(2560)
    # The tile count follows the row count: 16 tiles and 6 warm both classes.
    run(2048)
    run(768)
    with assert_no_triton_compile(mm.gluon_mm_mxfp8_gfx950):
        for rows in (256, 1280, 1792):
            # Rows are independent: a shorter batch is a prefix.
            torch.testing.assert_close(run(rows), full[:rows], rtol=0, atol=0)


def test_mxfp4_precomputed_route_row_count():
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import routing

    experts, topk = 256, 8

    def run(tokens):
        ids = torch.stack(
            [
                (torch.arange(topk, dtype=torch.int32, device=DEVICE) * 3 + row * 7)
                % experts
                for row in range(tokens)
            ]
        )
        weights = torch.randn(tokens, topk, device=DEVICE)
        metadata, gather, scatter, gate = routing.gluon_precomputed_topk_fused_route(
            weights, ids, experts
        )
        sizes = torch.bincount(ids.flatten(), minlength=experts).to(torch.int32)
        assert torch.equal(metadata.slice_sizes, sizes)
        assert torch.equal(metadata.slice_offs[1:], sizes.cumsum(0).to(torch.int32))
        assert torch.equal(gate, weights.flatten()[scatter.long()])
        assert torch.equal(gather, scatter // topk)

    # 5 to 8 tokens share the power-of-two route tiles; the block counts
    # alternate between the two integer classes.
    run(5)
    run(6)
    with assert_no_triton_compile(routing._fused_precomputed_topk_route_small_m):
        for tokens in (7, 8):
            run(tokens)


def test_mxfp4_moe_sorting_route_count():
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import moe_sorting

    experts, topk, block = 64, 8, 32
    generator = torch.Generator(device=DEVICE).manual_seed(3)

    def run(tokens):
        ids = torch.randint(
            0, experts, (tokens, topk), device=DEVICE, generator=generator
        ).int()
        weights = torch.rand(tokens, topk, device=DEVICE, generator=generator)
        sorted_ids, sorted_weights, block_experts, valid, _ = (
            moe_sorting.gluon_moe_sorting(
                ids,
                weights,
                experts,
                128,
                torch.bfloat16,
                block,
                compact_route_programs=False,
            )
        )
        assert int(valid[1]) == tokens
        slots = sorted_ids[: int(valid[0])]
        routed = slots != ((topk << 24) | tokens)
        position = torch.nonzero(routed).flatten()
        token, slot = slots[routed] & 0xFFFFFF, slots[routed] >> 24
        flat = token * topk + slot
        assert torch.equal(
            flat.sort().values, torch.arange(tokens * topk, device=DEVICE)
        )
        torch.testing.assert_close(sorted_weights[position], weights.flatten()[flat])
        assert torch.equal(block_experts[position // block], ids.flatten()[flat])

    def key(tokens):
        routes = tokens * topk
        programs = max(experts, -(-routes // 1024))
        per_program = -(-routes // programs)
        counts = (routes, programs, per_program)
        return (
            *map(int_specialization_class, counts),
            1 << (per_program - 1).bit_length(),
        )

    # Route counts from 8K to 16K share one power-of-two route block.
    sweep = (1033, 1100, 1500, 2000)
    warm_specialization_classes(run, key, sweep, range(1030, 2049))
    with assert_no_triton_compile(
        moe_sorting._moe_sorting_stage1_kernel, moe_sorting._moe_sorting_stage4_kernel
    ):
        for tokens in sweep:
            run(tokens)


def test_moe_topk_row_count():
    from tokenspeed_kernel_amd.ops.gfx950.moe import _common

    experts, topk = 128, 4
    # Each row is a permutation of 0..127, exact in BF16, so the top-k has no ties.
    logits = torch.rand(2400, experts, device=DEVICE).argsort(-1).bfloat16()

    def run(rows):
        indices = _common.topk(logits[:rows], topk).indx
        expected = torch.topk(logits[:rows].float(), topk).indices
        assert torch.equal(indices.long().sort(-1).values, expected.sort(-1).values)

    # The routing bitmatrix pads its rows to 32, so its stride follows the batch.
    run(64)
    run(65)
    with assert_no_triton_compile(_common._topk_forward):
        for rows in (97, 130, 1483, 2336):
            run(rows)


def test_iris_allreduce_sizes(monkeypatch):
    pytest.importorskip("iris")
    from tokenspeed_kernel._triton import gl

    # Other tests stub the Iris module through sys.modules, which only works
    # while it has not been imported; record its absence so teardown drops the
    # import this test makes.
    name = "tokenspeed_kernel.ops.communication.iris"
    if name not in sys.modules:
        package = importlib.import_module("tokenspeed_kernel.ops.communication")
        monkeypatch.setattr(package, "iris", None, raising=False)
        monkeypatch.setitem(sys.modules, name, None)
        del sys.modules[name]
    comm = importlib.import_module(name)

    buf = torch.empty(1 << 22, device=DEVICE, dtype=torch.bfloat16)
    flags = torch.zeros(1024, device=DEVICE, dtype=torch.int32)
    heaps = [1 << 40] * 8
    common = dict(RANK=0, WORLD_SIZE=8, SUBGROUP_SIZE=64, WORDS_PER_LANE=2)
    common.update(ELEMENT_DTYPE=gl.bfloat16, ELEMENTS_PER_WORD=4)

    # Compile only: the allreduce itself needs the symmetric heap of every rank.
    def two_stage(words):
        tiles = -(-words // 128)
        comm.iris_reduce_symmetric_two_stage_gluon_kernel.warmup(
            buf,
            buf,
            buf,
            flags,
            *heaps,
            PARTITION_WORDS=words,
            BLOCK_WORDS=128,
            NUM_PROGRAMS=min(tiles, 84),
            NUM_TILES=tiles,
            NUM_WARPS=8,
            EXIT_BARRIER=True,
            num_warps=8,
            grid=(1,),
            **common,
        )

    def one_stage(numel):
        tiles = -(-numel // 512)
        comm.iris_reduce_symmetric_gluon_kernel.warmup(
            buf,
            buf,
            flags,
            *heaps,
            TOTAL_NUMEL=numel,
            BLOCK_SIZE=512,
            NUM_PROGRAMS=min(tiles, 84),
            NUM_TILES=tiles,
            NUM_WARPS=1,
            PUBLISH_READY=False,
            num_warps=1,
            grid=(1,),
            **common,
        )

    # The reduced size follows the batch, and so do the tile and program counts.
    two_stage(60480)
    two_stage(60481)
    one_stage(7168 * 3)
    one_stage(7168 * 3 + 1)
    with assert_no_triton_compile(
        comm.iris_reduce_symmetric_two_stage_gluon_kernel,
        comm.iris_reduce_symmetric_gluon_kernel,
    ):
        for words in (157248, 996240, 236880, 79296):
            two_stage(words)
        for numel in (7168 * 5, 7168 * 11, 7168 * 29 + 3):
            one_stage(numel)


def test_kda_fused_replay_batch_size():
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda import decode

    layers = 4
    descriptors = torch.zeros(layers, 10, dtype=torch.uint64, device=DEVICE)
    groups = torch.zeros(layers, dtype=torch.int32, device=DEVICE)

    # Compile only: the descriptors would have to address real layer buffers.
    def compile_for(batch):
        pages = torch.zeros(2, batch, dtype=torch.int32, device=DEVICE)
        accepted = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
        decode.gluon_kda_fused_replay_gfx950.warmup(
            descriptors,
            groups,
            pages,
            pages,
            accepted,
            H=12,
            D=128,
            TOKENS_PER_SEQUENCE=4,
            MIXED_ROW_STRIDE=4608,
            CONV_WEIGHT_ROW_STRIDE=4,
            CONV_WEIGHT_COL_STRIDE=1,
            CONV_POOL_PAGE_STRIDE=13824,
            CONV_POOL_CHANNEL_STRIDE=3,
            CONV_POOL_HISTORY_STRIDE=1,
            GATE_ROW_STRIDE=1536,
            BETA_ROW_STRIDE=12,
            STATE_POOL_PAGE_STRIDE=196608,
            HAS_LOWER_BOUND=True,
            LOWER_BOUND=-5.0,
            BATCH_SIZE=batch,
            num_warps=4,
            num_stages=2,
            grid=(1,),
        )

    compile_for(16)
    compile_for(3)
    with assert_no_triton_compile(decode.gluon_kda_fused_replay_gfx950):
        for batch in (5, 7, 11, 13, 15):
            compile_for(batch)


def test_kda_fused_decode_batch_size():
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda import decode

    heads, head_dim, pages = 12, 128, 96
    width = heads * head_dim
    # Q/K/V, output gate, f_a, and beta are column slices of one projection.
    packed = torch.randn(48, 4 * width + head_dim + heads, device=DEVICE).bfloat16()
    conv_weights = torch.randn(3 * width, 4, device=DEVICE).bfloat16()
    conv_states = torch.zeros(pages, 3 * width, 3, device=DEVICE).bfloat16()
    f_b_weight = torch.randn(width, head_dim, device=DEVICE).bfloat16()
    state_pool = torch.zeros(pages, heads, head_dim, head_dim, device=DEVICE)

    def run(batch):
        rows = packed[:batch]
        decode.gluon_kda_fused_decode_gfx950(
            mixed_qkv=rows[:, : 3 * width],
            conv_weights=conv_weights,
            conv_states=conv_states,
            f_a_out=rows[:, 4 * width : 4 * width + head_dim],
            f_b_weight=f_b_weight,
            beta_logits=rows[:, 4 * width + head_dim :],
            A_log=torch.zeros(heads, device=DEVICE),
            dt_bias=torch.zeros(width, device=DEVICE),
            output_gate=rows[:, 3 * width : 4 * width],
            norm_weight=torch.ones(head_dim, device=DEVICE).bfloat16(),
            norm_eps=1e-6,
            state_pool=state_pool,
            read_indices=torch.arange(batch, device=DEVICE, dtype=torch.int32),
            write_indices=torch.arange(
                batch, 2 * batch, device=DEVICE, dtype=torch.int32
            ),
            num_heads=heads,
            head_dim=head_dim,
            cu_seqlens=torch.arange(batch + 1, device=DEVICE, dtype=torch.int32),
            lower_bound=-5.0,
        )

    # The state pipeline depth is bucketed by programs per CU.
    for batch in (16, 32, 48):
        run(batch)
    with assert_no_triton_compile(decode.gluon_kda_fused_paged_decode_vmajor_gfx950):
        for batch in (1, 3, 7, 21, 22, 33, 42, 43, 47):
            run(batch)
