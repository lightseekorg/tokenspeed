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

"""MHA fields sharing a plane preserve their page gaps."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
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


def test_mha_pool_and_leaf_preserve_consecutive_kv_fields_with_page_padding():
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
    groups = (
        (
            specs[0],
            (CacheFieldSpec("layer.0.latent_kv", "slot.0", (128, 1, 16), "bfloat16"),),
        ),
        (
            specs[1],
            tuple(
                CacheFieldSpec(
                    f"layer.1.{field}",
                    "slot.0",
                    (128, 2, 8),
                    "bfloat16",
                    exact_page_stride=False,
                )
                for field in ("k", "v")
            ),
        ),
    )
    layout = pack(
        groups,
        prefix_granularity=128,
        cache_blocks_per_lcm_block={"target": 5, "draft_swa": 2},
        max_padding_fraction=float("inf"),
    )
    arena = CacheArena(layout.bind(2), "cpu", cache_group_specs=specs)
    pool = MHATokenToKVPool(arena, torch.bfloat16, 2, 8, 1, 0, field_layer_offset=1)
    key, value = pool.get_kv_buffer(0)
    assert key.ndim == value.ndim == 4
    assert key.shape == (5, 128, 2, 8)
    assert key.stride(0) == value.stride(0) == 5120
    assert value.data_ptr() - key.data_ptr() == 4096
    key[2, 127].fill_(3)
    value[2, 127].fill_(7)
    assert torch.all(key[2, 127] == 3)
    assert torch.all(value[2, 127] == 7)
    assert key[3].data_ptr() - key[2].data_ptr() == 10240
    spec = MHAConfig(
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
        context_len=4096,
        max_bs=1,
        max_resident_requests=1,
        speculative_num_draft_tokens=8,
        components=(spec,),
    )
    leaf = MHAAttnBackend(config, spec, kernel_page_size=128)
    layer = PagedAttention(
        num_heads=2,
        num_kv_heads=2,
        head_dim=8,
        scaling=1.0,
        layer_id=0,
        sliding_window_size=1023,
        rotary_emb=None,
        qk_norm=None,
    )
    read_key, read_value = leaf._get_kv_cache(layer, pool)
    assert read_key.data_ptr() == key.data_ptr()
    assert read_value.data_ptr() == value.data_ptr()
    assert read_key.stride() == key.stride()
    target = pool.kv_write_target(0, torch.tensor([383]), None)
    assert target.k_cache.data_ptr() == key.data_ptr()
    assert target.k_cache.stride() == key.stride()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("graph", [False, True])
def test_dflash_fused_context_scatter_uses_each_layers_page_stride(graph):
    from tokenspeed.runtime.execution.drafter._dflash_fused_kv import (
        _fused_norm_rope_stacked_scatter,
        forget_kv_buffer_ptrs,
    )

    forget_kv_buffer_ptrs()
    heads, dim, rows, layers = 2, 64, 128, 2
    keys, values, backing = [], [], []
    for layer in range(layers):
        payload = rows * heads * dim
        storage = torch.full(
            (3, 2 * payload + 256 * (layer + 1)),
            -3,
            dtype=torch.bfloat16,
            device="cuda",
        )
        strides = (storage.stride(0), heads * dim, dim, 1)
        backing.append(storage)
        keys.append(storage.as_strided((3, rows, heads, dim), strides))
        values.append(storage.as_strided((3, rows, heads, dim), strides, payload))
    projected = torch.randn(
        4, layers, 2 * heads * dim, dtype=torch.bfloat16, device="cuda"
    )
    norm = torch.ones(layers, dim, dtype=torch.bfloat16, device="cuda")
    eps = torch.full((layers,), 1e-5, device="cuda")
    cosine = torch.empty((4, 0), dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(4, device="cuda")
    slots = torch.tensor([128, 255, 256, 383], device="cuda")

    def run():
        _fused_norm_rope_stacked_scatter(
            projected, norm, eps, cosine, positions, slots, keys, values, heads, dim, 0
        )

    run()
    if graph:
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        projected.add_(0.5)
        slots.copy_(torch.tensor([129, 254, 257, 382], device="cuda"))
        for storage in backing:
            storage.fill_(-3)
        capture.replay()
    for layer, (key, value) in enumerate(zip(keys, values, strict=True)):
        k = projected[:, layer, : heads * dim].view(4, heads, dim).float()
        expected = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-5)
        torch.testing.assert_close(
            key[slots // rows, slots % rows].float(), expected, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            value[slots // rows, slots % rows],
            projected[:, layer, heads * dim :].view(4, heads, dim),
            atol=0,
            rtol=0,
        )
        assert torch.all(backing[layer][:, -(layer + 1) * 256 :] == -3)


def test_context_write_uses_draft_router_before_proposal_publication():
    from dataclasses import dataclass

    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.drafter.base import BaseDrafter
    from tokenspeed.runtime.execution.drafter.dflash import DFlash
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (
        learn_cache_group_geometry,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )
    from tokenspeed.runtime.layers.logits_processor_output import LogitsProcessorOutput

    def router(group_id, *, draft):
        kind = "sliding_window" if draft else "full_history"
        group = CacheGroupSpec(
            group_id,
            kind,
            128,
            1,
            1024 if draft else None,
            sink_tokens=16 if draft else 0,
            replayable=False,
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
            max_resident_requests=1,
            speculative_num_draft_tokens=4,
            is_draft=draft,
            components=(component,),
        )
        leaf = MHAAttnBackend(config, component, kernel_page_size=128)
        result = CacheGroupRouter(
            None,
            is_draft=draft,
            spec_num_tokens=4,
            device="cpu",
            consumed_group_ids=None,
        )
        result.bind(learn_cache_group_geometry((group,)), {group_id: leaf})
        result.init_cuda_graph_state(1)
        return result

    @dataclass
    class Inputs:
        positions_buf: torch.Tensor
        input_lengths_buf: torch.Tensor
        req_pool_indices_buf: torch.Tensor

    @dataclass
    class States:
        valid_cache_lengths: torch.Tensor

    class CPUContextWriter(DFlash):
        """Replace the unavailable GPU projection/scatter with a CPU write.

        DFlash's prefix bookkeeping and both routers remain the real classes.
        """

        def __init__(self, backend):
            BaseDrafter.__init__(
                self,
                spec_num_tokens=4,
                attn_backend=backend,
                input_buffers=Inputs(
                    torch.arange(256, 260), torch.tensor([1]), torch.tensor([0])
                ),
                runtime_states=States(torch.tensor([256], dtype=torch.int32)),
            )
            self.draft_seq_lens_buf = torch.zeros(1, dtype=torch.int32)
            self.context_starts_buf = torch.zeros(1, dtype=torch.int64)
            self.context_locs_buf = torch.zeros(4, dtype=torch.int32)
            self.cache = torch.zeros(2048, 4)

        def _write_native_cache(self, hidden, positions, locations, *, decode_only):
            self.cache.index_copy_(0, locations.long(), hidden)

    target = router("full_attention", draft=False)
    draft = router("draft_swa", draft=True)
    requests = torch.tensor([0], dtype=torch.int32)
    lengths = torch.tensor([260], dtype=torch.int32)
    for backend, group_id, pages in (
        (target, "full_attention", [[1, 2, 3, 4]]),
        (draft, "draft_swa", [[7, 8, 9, 10]]),
    ):
        backend.refresh_decode_metadata(
            1,
            1,
            requests,
            lengths,
            forward_mode=ForwardMode.DECODE,
            block_tables={group_id: torch.tensor(pages, dtype=torch.int32)},
        )
    target_slots = target.decode_window_locations().clone()
    context_slots = draft.decode_window_locations().clone()
    assert target_slots.tolist() == [384, 385, 386, 387]
    assert context_slots.tolist() == [1152, 1153, 1154, 1155]
    drafter = CPUContextWriter(draft)
    ctx = ForwardContext(
        attn_backend=target,
        token_to_kv_pool=None,
        bs=1,
        num_extends=0,
        input_num_tokens=4,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 1, 4),
    )
    target_hidden = torch.arange(16, dtype=torch.float32).view(4, 4) + 1
    drafter._update_native_cache_from_target(
        ctx,
        LogitsProcessorOutput(next_token_logits=None, hidden_states=target_hidden),
        torch.tensor([2], dtype=torch.int32),
    )
    torch.testing.assert_close(drafter.cache[context_slots.long()], target_hidden)
    assert torch.count_nonzero(drafter.cache[target_slots.long()]) == 0
    assert drafter.draft_seq_lens_buf.tolist() == [258]
    draft.publish_draft_step_locations(drafter.draft_seq_lens_buf, 4)
    assert draft.decode_window_locations().tolist() == [1154, 1155, 1156, 1157]
    torch.testing.assert_close(
        drafter.cache[context_slots[:2].long()], target_hidden[:2]
    )
