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

"""Host-loaded cache blocks are touched only after their layer's load.

A Host L2 prefix hit copies the hit's Device blocks back layer by layer on the
load stream. An accessor that reads a loaded block, or writes the free rows of
a loaded tail block (a hit that ends inside a block loads that block whole and
the forward appends its new tokens to it), must first wait for that layer's
copy: otherwise the forward reads what the block held before, or the late copy
overwrites the rows it wrote. Each test holds the copy back on the load stream,
runs one model's accessor sequence on a real pool and checks, once the copy has
landed, the bytes the model saw or kept.
"""

import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

# ``test/`` (for ``ci_system``) and the repo root (for ``test.runtime.*``).
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

_HOLD_BACK_CYCLES = 200_000_000


@pytest.fixture
def host_cache(request, io_backend):
    """A target pool (and draft view) of one cache family over one arena, with
    a Host L2 tier and the device handle that runs a plan's cache ops.

    The arena holds random bytes. Page 1 of the group a test reads or writes
    is stored to Host page 1, which the test loads back into page 2, as a
    prefix hit does. Only that group moves: the groups of one LCM block alias
    each other's pages, which a scheduler never grants at once. The ops stand
    in for ``Cache.WriteBackOp`` and ``Cache.LoadBackOp``, which have no Python
    constructor.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    ts = pytest.importorskip("tokenspeed_scheduler")
    from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
    from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest
    from tokenspeed.runtime.execution.device import DeviceHandle
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.kv_cache.factory import (
        create_cache_arena,
        create_cache_pool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        cache_recipe,
    )

    class WriteBackOp:
        # A stream-ordered store of page ``src`` of each group to ``dst``.
        def __init__(self, op_id, groups, src, dst):
            self.op_ids = [op_id]
            self.group_ids = [list(groups)]
            self.src_pages = [[src] * len(groups)]
            self.dst_pages = [[dst] * len(groups)]
            self.source_pinned = [False]

    class LoadBackOp:
        def __init__(self, op_id, groups, src, dst):
            self.op_ids = [op_id]
            self.group_ids = [list(groups)]
            self.src_pages = [[src] * len(groups)]
            self.dst_pages = [[dst] * len(groups)]

    # The model-wide settings and scheduler limits of the families built
    # here from an attention config.
    model_wide = dict(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=4096,
        max_bs=2,
    )
    recipe_args = dict(
        server_args=SimpleNamespace(
            kv_offload_config=None,
            max_total_tokens=None,
            chunked_prefill_size=1024,
            disaggregation_mode="null",
            enable_prefix_caching=True,
            speculative_num_steps=0,
            speculative_num_draft_tokens=1,
        ),
        cache_budget_bytes=0,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    family, group_id = request.param
    draft_family = None
    if family == "kimi_k3":
        from test.runtime.conftest import kimi_recipe

        # The K3 drafts (DSpark, DFLASH2) are plain MLA views over the arena.
        recipe = kimi_recipe(draft_layers=1)
        draft_family = "mla"
    elif family in ("mha", "mha_mxfp8"):
        # A GQA target with a one-layer GQA draft (DFLASH, DSpark).
        mxfp8 = family == "mha_mxfp8"
        attn_config = AttnConfig(
            kv_cache_dtype=torch.float8_e4m3fn if mxfp8 else torch.bfloat16,
            kv_cache_mxfp8=mxfp8,
            prefix_granularity=128,
            kernel_page_size=128,
            components=(
                MHAConfig(
                    backend_name="mha",
                    num_attention_heads=16,
                    num_kv_heads=4,
                    head_dim=128,
                    attn_tp_size=1,
                ),
            ),
            **model_wide,
        )
        recipe = cache_recipe(
            "mha",
            model_config=SimpleNamespace(num_attention_layers=2),
            attn_config=attn_config,
            draft_model_config=SimpleNamespace(num_attention_layers=1),
            draft_attn_config=replace(attn_config, is_draft=True),
            **recipe_args,
        )
    elif family == "dsa":
        # GLM-5 / DeepSeek-V3.2: MLA latent plus an FP8 index-K plane.
        attn_config = AttnConfig(
            kv_cache_dtype=torch.bfloat16,
            prefix_granularity=64,
            kernel_page_size=64,
            components=(
                DSAConfig(
                    backend_name="dsa",
                    num_attention_heads=64,
                    num_kv_heads=64,
                    head_dim=192,
                    attn_tp_size=1,
                    kv_lora_rank=512,
                    qk_nope_head_dim=128,
                    qk_rope_head_dim=64,
                    v_head_dim=128,
                    scaling=1.0,
                    kv_cache_dim=576,
                    index_topk=1,
                    index_head_dim=128,
                    index_n_heads=1,
                    index_k_format="fp8_scaled",
                ),
            ),
            **model_wide,
        )
        recipe = cache_recipe(
            "dsa",
            model_config=SimpleNamespace(num_attention_layers=3),
            attn_config=attn_config,
            draft_model_config=None,
            draft_attn_config=None,
            **recipe_args,
        )
    elif family == "glm53_flash":
        from test.runtime.test_glm53_flash_cache_spec import _recipe

        recipe = _recipe(tp_size=8, mla_cache_dtype=torch.bfloat16, draft_layers=1)
    elif family == "inkling":
        from tokenspeed.runtime.configs.inkling_config import InklingModelConfig

        text_config = InklingModelConfig(num_hidden_layers=2, use_sconv=True)
        attn_config = AttnConfig(
            kv_cache_dtype=torch.bfloat16,
            prefix_granularity=128,
            kernel_page_size=128,
            components=(
                MHAConfig(
                    backend_name="mha",
                    num_attention_heads=text_config.num_attention_heads,
                    num_kv_heads=text_config.num_key_value_heads,
                    head_dim=text_config.head_dim,
                    attn_tp_size=1,
                    cache_layer_types=tuple(text_config.cache_layer_types),
                ),
            ),
            **model_wide,
        )
        recipe = cache_recipe(
            "inkling",
            model_config=SimpleNamespace(num_attention_layers=2, hf_config=text_config),
            attn_config=attn_config,
            draft_model_config=None,
            draft_attn_config=None,
            **recipe_args,
        )
    # A probe-sized arena: a few blocks per group, pages 1-3 below.
    recipe.cache_budget_bytes = 0
    recipe.probe_batch_rows = 4
    setup = recipe.setup()
    arena = create_cache_arena(setup.spec, device="cuda", enable_memory_saver=False)
    num_target_layers = setup.num_target_layers
    pool = create_cache_pool(
        setup.spec.layer_view(first_layer=0, num_layers=num_target_layers),
        recipe.attn_config,
        arena,
        num_layers=num_target_layers,
        rank=0,
    )
    draft_pool = None
    if setup.num_draft_layers:
        draft_pool = create_cache_pool(
            setup.spec.layer_view(
                first_layer=num_target_layers,
                num_layers=setup.num_draft_layers,
                family=draft_family,
            ),
            recipe.draft_attn_config,
            arena,
            num_layers=setup.num_draft_layers,
            rank=0,
            field_layer_offset=num_target_layers,
        )
    executor = HostCacheExecutor(
        pool,
        draft_pool=draft_pool,
        l2_tier=True,
        host_ratio=1.0,
        host_size_gb=0,
        snapshot_pool=RetractionPoolRequest(
            host_gb=0.0,
            ratio=0.0,
            max_retracted_requests=0,
            tail_lcm_blocks_per_request=0,
        ),
        slot_state_exporters=None,
        io_backend=io_backend,
        attn_tp_rank=0,
        kvp_rank=0,
        nprocs_per_node=1,
        world_cpu_group=None,
    )
    # A model executor without a model: the fields the handle reads.
    model_executor = ModelExecutor.__new__(ModelExecutor)
    model_executor.device = "cuda"
    model_executor.device_module = torch.cuda
    model_executor.default_stream = torch.cuda.default_stream()
    model_executor.execution_stream = torch.cuda.Stream()
    model_executor.token_to_kv_pool = pool
    model_executor.draft_token_to_kv_pool = draft_pool
    model_executor._cache_runtime_contract = arena.runtime_contract
    model_executor._cache_kvp_rank = 0
    model_executor.drafter = None
    handle = DeviceHandle(model_executor, host_cache_executor=executor)
    handle.bind_forward_thread()
    group = [layout.group_id for layout in executor.layout.groups].index(group_id)
    try:
        with (
            patch.object(ts.Cache, "WriteBackOp", WriteBackOp),
            patch.object(ts.Cache, "LoadBackOp", LoadBackOp),
        ):
            # A first load and a layerwise wait for every layer of each view,
            # so a held-back load below sets up no transfer tables and compiles
            # no wait kernel: either would wait for the device, the held-back
            # copy included.
            handle.execute(
                ts.ExecutionPlan(),
                None,
                cache_ops=[LoadBackOp(1, [group], 1, 3)],
            )
            for view in (pool, draft_pool):
                if view is not None:
                    for layer_id in range(view.layer_num):
                        view.layerwise_load_tracker.wait_for_layer(layer_id)
            torch.manual_seed(37)
            arena.buffer.random_(0, 256)
            # The store below copies behind the execution stream, not behind
            # the default stream that filled the arena.
            torch.cuda.synchronize()
            # Its plan also points the layerwise waits at no load again.
            handle.execute(
                ts.ExecutionPlan(),
                None,
                cache_ops=[WriteBackOp(2, [group], 1, 1)],
            )
            torch.cuda.synchronize()
            acks = sorted(int(ack.op_id) for ack in handle.poll_cache_results())
            assert acks == [1, 2]
            yield SimpleNamespace(
                ts=ts,
                recipe=recipe,
                arena=arena,
                pool=pool,
                draft_pool=draft_pool,
                executor=executor,
                handle=handle,
                group=group,
                LoadBackOp=LoadBackOp,
            )
    finally:
        handle.shutdown_cache()


@pytest.mark.parametrize("io_backend", ["kernel", "direct"])
@pytest.mark.parametrize(
    "host_cache", [("kimi_k3", "full_attention")], ids=["kimi_k3"], indirect=True
)
def test_a_draft_context_write_lands_after_its_tail_page_load(host_cache):
    """K3 DSpark / DFLASH2 inject the target's context into the MLA draft's
    latent cache (``set_mla_kv_buffer``) before the draft forward's first
    layerwise wait. On a Host tail hit the round's rows land in the loaded
    tail block, whose draft copy lands after every target layer's."""
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

    cache = host_cache
    draft = cache.draft_pool
    page_size = cache.arena.prefix_granularity
    page, hit = 2, page_size // 2
    latent = draft.kv_buffer[0]
    slots = torch.arange(page * page_size + hit, (page + 1) * page_size, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(5)
    nope, rope = (
        torch.randn(
            page_size - hit,
            1,
            width,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        for width in (draft.kv_lora_rank, draft.qk_rope_head_dim)
    )
    layer = PagedAttention(
        1,
        draft.kv_lora_rank + draft.qk_rope_head_dim,
        1.0,
        1,
        0,
        rotary_emb=None,
        qk_norm=None,
    )
    # The rows the write leaves, from the same write into page 3.
    draft.set_mla_kv_buffer(layer, slots + page_size, nope, rope, write_mask=None)
    torch.cuda.synchronize()
    expected = latent[slots + page_size].view(torch.uint8)
    host_rows = latent[page_size : page_size + hit].view(torch.uint8).clone()

    with torch.cuda.stream(cache.executor.load_stream):
        torch.cuda._sleep(_HOLD_BACK_CYCLES)
    cache.handle.execute(
        cache.ts.ExecutionPlan(),
        None,
        cache_ops=[cache.LoadBackOp(3, [cache.group], 1, page)],
    )
    draft.set_mla_kv_buffer(layer, slots, nope, rope, write_mask=None)
    draft.get_key_buffer(0)
    assert not cache.executor.load_stream.query()
    torch.cuda.synchronize()
    assert [int(ack.op_id) for ack in cache.handle.poll_cache_results()] == [3]
    start = page * page_size
    assert torch.equal(latent[start : start + hit].view(torch.uint8), host_rows)
    assert torch.equal(latent[slots].view(torch.uint8), expected)


@pytest.mark.parametrize("io_backend", ["kernel", "direct"])
@pytest.mark.parametrize(
    "host_cache",
    [("mha", "full_attention"), ("mha_mxfp8", "full_attention")],
    ids=["mha", "mha_mxfp8"],
    indirect=True,
)
def test_a_gqa_draft_context_write_lands_after_its_tail_page_load(host_cache):
    """DFLASH / DSpark GQA drafts inject the target's context into the draft's
    K/V cache (``set_kv_buffer``) before the draft forward's first layerwise
    wait. On a Host tail hit the round's rows land in the loaded tail block,
    whose draft copy lands after every target layer's."""
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

    cache = host_cache
    draft = cache.draft_pool
    page_size = cache.arena.prefix_granularity
    page, hit = 2, page_size // 2
    slots = torch.arange(page * page_size + hit, (page + 1) * page_size, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(11)
    k, v = (
        torch.randn(
            page_size - hit,
            draft.head_num,
            draft.head_dim,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        for _ in range(2)
    )
    layer = PagedAttention(
        draft.head_num,
        draft.head_dim,
        1.0,
        draft.head_num,
        0,
        rotary_emb=None,
        qk_norm=None,
    )
    k_rows, v_rows = draft.get_kv_buffer(0)
    # The rows the write leaves, from the same write into page 3.
    draft.set_kv_buffer(layer, slots + page_size, k, v)
    torch.cuda.synchronize()
    expected = [rows[slots + page_size].view(torch.uint8) for rows in (k_rows, v_rows)]
    host_rows = [
        rows[page_size : page_size + hit].view(torch.uint8).clone()
        for rows in (k_rows, v_rows)
    ]

    with torch.cuda.stream(cache.executor.load_stream):
        torch.cuda._sleep(_HOLD_BACK_CYCLES)
    cache.handle.execute(
        cache.ts.ExecutionPlan(),
        None,
        cache_ops=[cache.LoadBackOp(3, [cache.group], 1, page)],
    )
    draft.set_kv_buffer(layer, slots, k, v)
    draft.get_key_buffer(0)
    assert not cache.executor.load_stream.query()
    torch.cuda.synchronize()
    assert [int(ack.op_id) for ack in cache.handle.poll_cache_results()] == [3]
    start = page * page_size
    for rows, host, written in zip((k_rows, v_rows), host_rows, expected):
        assert torch.equal(rows[start : start + hit].view(torch.uint8), host)
        assert torch.equal(rows[slots].view(torch.uint8), written)


@pytest.mark.parametrize("io_backend", ["kernel", "direct"])
@pytest.mark.parametrize(
    "host_cache", [("dsa", "full_attention")], ids=["dsa"], indirect=True
)
def test_an_index_k_write_lands_after_its_tail_page_load(host_cache):
    """GLM-5 / DeepSeek-V3.2 write the round's index keys
    (``set_index_k_buffer``) before the layer's first layerwise wait, the
    top-k's ``get_index_k_buffer``. On a Host tail hit they land in the loaded
    tail block."""
    cache = host_cache
    pool = cache.pool
    page_size = cache.arena.prefix_granularity
    page, hit, layer = 2, page_size // 2, 0
    slots = torch.arange(page * page_size + hit, (page + 1) * page_size, device="cuda")
    keys = torch.randn(
        page_size - hit, pool.index_head_dim, dtype=torch.bfloat16, device="cuda"
    )
    # The rows the write leaves, from the same write into page 3.
    pool.set_index_k_buffer(layer, slots + page_size, keys, write_mask=None)
    torch.cuda.synchronize()
    expected = pool.gather_index_k_rows(
        layer, slots + page_size, index_k_format="fp8_scaled"
    )
    hit_slots = torch.arange(hit, device="cuda")
    host_rows = pool.gather_index_k_rows(
        layer, page_size + hit_slots, index_k_format="fp8_scaled"
    )

    with torch.cuda.stream(cache.executor.load_stream):
        torch.cuda._sleep(_HOLD_BACK_CYCLES)
    cache.handle.execute(
        cache.ts.ExecutionPlan(),
        None,
        cache_ops=[cache.LoadBackOp(3, [cache.group], 1, page)],
    )
    pool.set_index_k_buffer(layer, slots, keys, write_mask=None)
    pool.get_index_k_buffer(layer)
    assert not cache.executor.load_stream.query()
    torch.cuda.synchronize()
    assert [int(ack.op_id) for ack in cache.handle.poll_cache_results()] == [3]
    assert torch.equal(
        pool.gather_index_k_rows(
            layer, page * page_size + hit_slots, index_k_format="fp8_scaled"
        ),
        host_rows,
    )
    assert torch.equal(
        pool.gather_index_k_rows(layer, slots, index_k_format="fp8_scaled"), expected
    )


@pytest.mark.parametrize("io_backend", ["kernel", "direct"])
@pytest.mark.parametrize("model", ["target", "draft"])
@pytest.mark.parametrize(
    "host_cache",
    [("glm53_flash", "full_attention")],
    ids=["glm53_flash"],
    indirect=True,
)
def test_the_kpool_top_k_reads_index_pages_after_their_load(host_cache, model):
    """GLM-5.3-Flash's KPool prefill top-k reads every prefix index page
    through ``get_kpool_buffers`` before the layer's first layerwise wait (the
    latent prologue). Any Host hit loads those pages; the read sees what the
    page held before unless it waits."""
    from test.runtime.conftest import mla_layer_id

    cache = host_cache
    view = cache.pool if model == "target" else cache.draft_pool
    layer = mla_layer_id(view)
    page_size = cache.arena.prefix_granularity
    page = 2
    slots = torch.arange(page * page_size, (page + 1) * page_size, device="cuda")
    host_index = view.get_kpool_buffers(layer)[0][1].clone()
    # Load destinations are never zeroed: give the page other bytes.
    cache.arena.zero_blocks({"full_attention": np.asarray([page])})

    with torch.cuda.stream(cache.executor.load_stream):
        torch.cuda._sleep(_HOLD_BACK_CYCLES)
    cache.handle.execute(
        cache.ts.ExecutionPlan(),
        None,
        cache_ops=[cache.LoadBackOp(3, [cache.group], 1, page)],
    )
    seen = view.get_kpool_buffers(layer)[0][page].clone()
    view.kv_write_target(layer, slots, None)
    assert not cache.executor.load_stream.query()
    torch.cuda.synchronize()
    assert [int(ack.op_id) for ack in cache.handle.poll_cache_results()] == [3]
    assert torch.equal(seen, host_index)


@pytest.mark.parametrize("io_backend", ["kernel", "direct"])
@pytest.mark.parametrize(
    "host_cache",
    [("glm53_flash", "full_attention")],
    ids=["glm53_flash"],
    indirect=True,
)
def test_a_kpool_write_lands_after_its_tail_page_load(host_cache):
    """GLM-5.3-Flash writes the chunk's completed pools into the index pages
    (``kpool_prefill_write`` through ``get_kpool_buffers``) before the layer's
    first layerwise wait. On a Host tail hit they land in the loaded tail
    block."""
    from test.runtime.conftest import mla_layer_id

    from tokenspeed_kernel.ops.attention.kpool import kpool_prefill_write

    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig

    cache = host_cache
    pool = cache.pool
    layer = mla_layer_id(pool)
    page_size = cache.arena.prefix_granularity
    page = 2
    index = pool.get_kpool_buffers(layer)[0]
    values, scales = pool.index_k_block_views(index)
    rows_per_page = index.shape[1]
    hit = rows_per_page // 2
    write_slots = torch.arange(
        page * rows_per_page + hit, (page + 1) * rows_per_page, device="cuda"
    )
    kpool = cache.recipe.attn_config.component(DSAConfig).index_kpool
    generator = torch.Generator(device="cuda").manual_seed(7)
    keys, gates = (
        torch.randn(
            rows_per_page - hit,
            kpool,
            pool.index_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        for _ in range(2)
    )
    ape = torch.randn(kpool, pool.index_head_dim, device="cuda", generator=generator)
    # The rows the write leaves, from the same write into page 3.
    kpool_prefill_write(keys, gates, write_slots + rows_per_page, values, scales, ape)
    torch.cuda.synchronize()
    expected = (
        values[page + 1, hit:].view(torch.uint8).clone(),
        scales[page + 1, hit:].view(torch.int32).clone(),
    )
    host_rows = (
        values[1, :hit].view(torch.uint8).clone(),
        scales[1, :hit].view(torch.int32).clone(),
    )

    with torch.cuda.stream(cache.executor.load_stream):
        torch.cuda._sleep(_HOLD_BACK_CYCLES)
    cache.handle.execute(
        cache.ts.ExecutionPlan(),
        None,
        cache_ops=[cache.LoadBackOp(3, [cache.group], 1, page)],
    )
    index = pool.get_kpool_buffers(layer)[0]
    kpool_prefill_write(keys, gates, write_slots, *pool.index_k_block_views(index), ape)
    pool.kv_write_target(
        layer,
        torch.arange(page * page_size, (page + 1) * page_size, device="cuda"),
        None,
    )
    assert not cache.executor.load_stream.query()
    torch.cuda.synchronize()
    assert [int(ack.op_id) for ack in cache.handle.poll_cache_results()] == [3]
    assert torch.equal(values[page, :hit].view(torch.uint8), host_rows[0])
    assert torch.equal(scales[page, :hit].view(torch.int32), host_rows[1])
    assert torch.equal(values[page, hit:].view(torch.uint8), expected[0])
    assert torch.equal(scales[page, hit:].view(torch.int32), expected[1])


@pytest.mark.parametrize("io_backend", ["kernel", "direct"])
@pytest.mark.parametrize(
    "host_cache", [("inkling", "kvconv")], ids=["inkling"], indirect=True
)
def test_the_kv_short_conv_reads_its_checkpoint_after_the_load(host_cache):
    """Inkling's K/V short conv reads the chunk's input checkpoint
    (``kvconv_checkpoint_buffers``) before the attention prologue, the layer's
    first layerwise wait. A Host hit loads that checkpoint; the read sees what
    the slot held before unless it waits."""
    cache = host_cache
    pool = cache.pool
    page_size = cache.arena.prefix_granularity
    page, layer = 2, 0
    slots = torch.arange(page * page_size, (page + 1) * page_size, device="cuda")
    host_rows = [
        buffer[1].clone().view(torch.uint8)
        for buffer in pool.kvconv_checkpoint_buffers(layer)
    ]
    # Load destinations are never zeroed: give the slot other bytes.
    cache.arena.zero_blocks({"kvconv": np.asarray([page])})

    with torch.cuda.stream(cache.executor.load_stream):
        torch.cuda._sleep(_HOLD_BACK_CYCLES)
    cache.handle.execute(
        cache.ts.ExecutionPlan(),
        None,
        cache_ops=[cache.LoadBackOp(3, [cache.group], 1, page)],
    )
    seen = [buffer[page].clone() for buffer in pool.kvconv_checkpoint_buffers(layer)]
    pool.kv_write_target(layer, slots, None)
    assert not cache.executor.load_stream.query()
    torch.cuda.synchronize()
    assert [int(ack.op_id) for ack in cache.handle.poll_cache_results()] == [3]
    for rows, host in zip(seen, host_rows):
        assert torch.equal(rows.view(torch.uint8), host)


@pytest.mark.parametrize("io_backend", ["direct"])
@pytest.mark.parametrize(
    "host_cache", [("inkling", "kvconv")], ids=["inkling"], indirect=True
)
def test_a_later_layer_reads_its_checkpoint_after_its_own_load(host_cache):
    """The read waits for the layer it reads, not for an earlier one. In a
    load whose earlier layers land at once and whose copy of the last layer's
    checkpoint is held back, a read of that checkpoint sees what the copy
    brings."""
    from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker

    cache = host_cache
    pool = cache.pool
    page, layer = 2, pool.layer_num - 1
    assert layer > 0
    tracker = LayerwiseLoadTracker(pool.layer_num)
    pool.register_layerwise_load_tracker(tracker)
    buffers = pool.kvconv_checkpoint_buffers(layer)
    host_rows = [buffer[1].clone().view(torch.uint8) for buffer in buffers]
    # Load destinations are never zeroed: give the slot other bytes.
    cache.arena.zero_blocks({"kvconv": np.asarray([page])})

    # The load, on a stream of its own: the layer events record in layer
    # order, as the load stream records them, and only the last layer's copy
    # is held back, so a wait for an earlier layer returns before it lands.
    load = tracker.begin_load()
    events = tracker.event_sets[load]
    load_stream = torch.cuda.Stream()
    load_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(load_stream):
        for earlier in range(layer):
            events.layer_done_events[earlier].record()
        torch.cuda._sleep(_HOLD_BACK_CYCLES)
        for buffer in buffers:
            buffer[page].copy_(buffer[1])
        events.layer_done_events[layer].record()
    tracker.set_consumers(load)
    seen = [buffer[page].clone() for buffer in pool.kvconv_checkpoint_buffers(layer)]
    assert not load_stream.query()
    torch.cuda.synchronize()
    for rows, host in zip(seen, host_rows):
        assert torch.equal(rows.view(torch.uint8), host)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
