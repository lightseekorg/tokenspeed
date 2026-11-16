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

"""FlatKV atom addressing and ordinary padded-cache regression checks."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.paged_attention import PagedAttention


@requires_cuda
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_atom_translation_gpu_and_graph(packed, graph):
    specs = (
        GroupTableSpec("full", 4, 4, 6, 4),
        GroupTableSpec("subpages", 4, 1, 13, 16),
    )
    storage = torch.tensor(
        [0, 1, 7, -1, 3, 2, 5, 5, 5, 2, 0, -1, 8, 1, 3, 4, 0, 9, 9, 9, 9],
        dtype=torch.int32,
        device="cuda",
    )
    sources = {"full": storage[:9].view(3, 3), "subpages": storage[9:].view(3, 4)}
    if not packed:
        sources = {gid: src.clone() for gid, src in sources.items()}
    stacks = GroupTableStacks(
        specs, max_bs=3, max_tokens_per_req=2, max_extend_tokens=8, device="cuda"
    )
    lengths = torch.tensor([5, 8, 1], dtype=torch.int32, device="cuda")

    def run():
        stacks.fill(3, 2, sources)
        stacks.compute_decode_locations(3, lengths, 2)

    run()
    if graph:
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        sources["full"][0, 1] = 4
        sources["subpages"][1, 1] = 6
        stacks.tables.fill_(99)
        capture.replay()
    expected = torch.zeros_like(stacks.tables, device="cpu")
    for group_index, spec in enumerate(specs):
        raw = sources[spec.group_id].cpu().tolist()
        ratio = spec.block_granularity // spec.kernel_page_size
        for row in range(2):
            expanded = [
                block * spec.atom_pages_per_block + j if block > 0 else 0
                for block in raw[row]
                for j in range(ratio)
            ][: spec.max_num_pages]
            expected[group_index, row, : len(expanded)] = torch.tensor(expanded)
    torch.testing.assert_close(stacks.tables.cpu(), expected, atol=0, rtol=0)
    expected_locations = []
    for group_index, spec in enumerate(specs):
        locations = []
        for row, length in enumerate(lengths.cpu().tolist()):
            for position in range(length - 2, length):
                if position < 0:
                    locations.append(0)
                    continue
                page = expected[
                    group_index, row, position // spec.kernel_page_size
                ].item()
                locations.append(
                    page * spec.kernel_page_size + position % spec.kernel_page_size
                    if page > 0
                    else 0
                )
        expected_locations.append(locations)
    torch.testing.assert_close(
        stacks.decode_locs.cpu(),
        torch.tensor(expected_locations, dtype=torch.int32),
        atol=0,
        rtol=0,
    )


@requires_cuda
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("kernel_page_size", [64, 256])
def test_flat_kv_gpu_store_reaches_original_field(dtype, kernel_page_size, graph):
    grain, heads, dim = 256, 2, 64
    spec = CacheGroupSpec("full_attention", "full_history", grain, 1, replayable=False)
    storage_dtype = "bfloat16" if dtype == torch.bfloat16 else "uint8"
    fields = tuple(
        CacheFieldSpec(
            f"layer.{layer}.{name}",
            "flatkv",
            (grain, heads, dim),
            storage_dtype,
            exact_page_stride=False,
        )
        for layer in range(2)
        for name in ("k", "v")
    )
    layout = pack(
        ((spec, fields),),
        prefix_granularity=grain,
        cache_blocks_per_lcm_block={spec.group_id: 1},
        max_padding_fraction=float("inf"),
    )
    arena = CacheArena(layout.bind(8), "cuda", cache_group_specs=(spec,))
    pool = MHATokenToKVPool(arena, dtype, heads, dim, 2, 0)
    multiplier = pool.atom_pages_per_block(spec.group_id, kernel_page_size)
    tables = GroupTableStacks(
        [
            GroupTableSpec(
                spec.group_id,
                grain,
                kernel_page_size,
                grain // kernel_page_size,
                multiplier,
            )
        ],
        max_bs=1,
        max_tokens_per_req=1,
        max_extend_tokens=0,
        device="cuda",
    )
    raw = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([8], dtype=torch.int32, device="cuda")
    locations = tables.decode_locations(spec.group_id, 1, 1)
    layers = [
        PagedAttention(
            8, dim, dim**-0.5, heads, layer_id, rotary_emb=None, qk_norm=None
        )
        for layer_id in range(2)
    ]
    keys = [
        torch.full((1, heads, dim), layer_id + 1, dtype=torch.bfloat16, device="cuda")
        for layer_id in range(2)
    ]
    values = [torch.full_like(key, layer_id + 5) for layer_id, key in enumerate(keys)]

    def run():
        tables.fill(1, 1, {spec.group_id: raw})
        tables.compute_decode_locations(1, lengths, 1)
        for layer, key, value in zip(layers, keys, values):
            pool.set_kv_buffer(layer, locations, key, value)

    run()
    block = 1
    increment = 0
    if graph:
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        block = arena.plan.group(spec.group_id).page_count - 1
        raw.fill_(block)
        increment = 2
        for data in (*keys, *values):
            data.add_(increment)
        capture.replay()

    for layer_id in range(2):
        for name, value in (
            ("k", layer_id + 1 + increment),
            ("v", layer_id + 5 + increment),
        ):
            field_id = f"layer.{layer_id}.{name}"
            field = arena.plan.field(field_id)
            row_bytes = field.payload_bytes // grain
            byte_offset = (
                arena.plan.field_page_byte_offset(field_id, block) + 7 * row_bytes
            )
            actual = (
                arena.buffer[byte_offset : byte_offset + row_bytes].view(dtype).float()
            )
            torch.testing.assert_close(
                actual, torch.full_like(actual, value), atol=0, rtol=0
            )


def test_flat_kv_rejects_each_misaligned_page_even_when_the_span_aligns():
    group = CacheGroupSpec("full_attention", "full_history", 4, 1, replayable=False)
    fields = (
        CacheFieldSpec(
            "layer.0.k", "flatkv", (4, 1, 4), "bfloat16", exact_page_stride=False
        ),
        CacheFieldSpec(
            "layer.0.extra", "flatkv", (2,), "bfloat16", exact_page_stride=False
        ),
    )
    layout = pack(
        ((group, fields),),
        prefix_granularity=4,
        cache_blocks_per_lcm_block={group.group_id: 1},
        alignment=1,
        max_padding_fraction=float("inf"),
    )
    # Three pages make the complete span row-aligned despite a half-row gap
    # between each page; checking only that span would miss the bad stride.
    with pytest.raises(ValueError, match="whole number of atom rows"):
        CacheArena(layout.bind(2), "cpu", cache_group_specs=(group,))


@pytest.mark.parametrize("target_packing", [4, 5])
@pytest.mark.parametrize("layer_kv_head_counts", [None, (2,)])
def test_ordinary_sink_router_keeps_raw_page_addresses(
    target_packing, layer_kv_head_counts
):
    specs = (
        CacheGroupSpec("target", "full_history", 128, 1, replayable=False),
        CacheGroupSpec(
            "draft_swa",
            "sliding_window",
            128,
            1,
            1024,
            sink_tokens=16,
            replayable=False,
        ),
    )
    layout = pack(
        (
            (
                specs[0],
                (
                    CacheFieldSpec(
                        "layer.0.latent_kv", "slot.0", (128, 1, 16), "bfloat16"
                    ),
                ),
            ),
            (
                specs[1],
                tuple(
                    CacheFieldSpec(
                        f"layer.1.{name}",
                        "slot.0",
                        (128, 2, 8),
                        "bfloat16",
                        exact_page_stride=False,
                    )
                    for name in ("k", "v")
                ),
            ),
        ),
        prefix_granularity=128,
        cache_blocks_per_lcm_block={"target": target_packing, "draft_swa": 2},
        max_padding_fraction=float("inf"),
    )
    arena = CacheArena(layout.bind(3), "cpu", cache_group_specs=specs)
    pool = MHATokenToKVPool(
        arena,
        torch.bfloat16,
        2,
        8,
        1,
        0,
        field_layer_offset=1,
        layer_kv_head_counts=layer_kv_head_counts,
    )
    component = MHAConfig(
        backend_name="triton",
        num_attention_heads=2,
        num_kv_heads=2,
        head_dim=8,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=512,
        max_bs=1,
        components=(component,),
        speculative_num_draft_tokens=1,
    )
    router = CacheGroupRouter(
        lambda gid, grain: MHAAttnBackend(config, component, kernel_page_size=grain),
        is_draft=False,
        spec_num_tokens=1,
        device="cpu",
        consumed_group_ids=("draft_swa",),
    )
    router.set_cache_pool(pool)
    router.init_cuda_graph_state(1)
    router.refresh_decode_metadata(
        1,
        1,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables={"draft_swa": torch.tensor([[1]], dtype=torch.int32)},
    )
    assert router.leaves["draft_swa"].page_table_buf[0, 0].item() == 1
    assert router.decode_window_locations().tolist() == [135]
