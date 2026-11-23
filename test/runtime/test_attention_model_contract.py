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

"""Model operations retain their meaning through attention composition."""

from dataclasses import dataclass
from test.ci_system.ci_register import register_cuda_ci
from test.runtime import test_cache_group_router as router_tests

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
    PrefillRequestLayout,
)
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    Qwen4ExpBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention, bind_cache_groups

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@dataclass(frozen=True)
class InputTransform:
    factor: int

    def apply(self, x):
        return x * self.factor


class CPUAttention(AttentionBackend):
    """A small CPU extension used as a model-binding consumer; no device mocks."""

    def __init__(self, factor):
        spec = SoftmaxAttnConfig(
            num_attention_heads=1, num_kv_heads=1, head_dim=8, attn_tp_size=1
        )
        config = AttnConfig(
            device="cpu",
            dtype=torch.float32,
            kv_cache_dtype=torch.float32,
            kv_cache_quant_method="none",
            context_len=32,
            max_bs=4,
            max_resident_requests=4,
            components=(spec,),
        )
        super().__init__(config, spec)
        self.test_config = config
        self.input_transform = InputTransform(factor)

    def model_binding(self, layer_id, binding_type):
        if binding_type is InputTransform:
            return self.input_transform
        return super().model_binding(layer_id, binding_type)

    def forward_decode(self, q, k, v, layer, token_to_kv_pool, bs, **kwargs):
        return q + 1


@pytest.mark.parametrize("nested", [False, True])
def test_model_input_binding_uses_the_execution_owner(nested):
    full, state = CPUAttention(2), CPUAttention(7)
    backend = HybridLinearAttnBackend(full, state, [0])
    if nested:
        backend = Qwen4ExpBackend(full.test_config, backend, None, None)
    x = torch.arange(3, dtype=torch.float32)
    for layer_id, factor in [(0, 2), (1, 7)]:
        binding = backend.model_binding(layer_id, InputTransform)
        actual = backend.forward(
            q=binding.apply(x),
            k=None,
            v=None,
            layer=None,
            token_to_kv_pool=None,
            forward_mode=ForwardMode.DECODE,
            bs=3,
            layer_id=layer_id,
        )
        torch.testing.assert_close(actual, x * factor + 1)
    assert backend.model_binding(1, str) is None


@pytest.mark.parametrize("stride", [1, 2])
def test_request_layout_preserves_active_cpu_lengths_without_aliasing(stride):
    inputs = torch.full((3 * stride,), -9, dtype=torch.int64)[::stride]
    prefixes = torch.full((3 * stride,), -9, dtype=torch.int64)[::stride]
    inputs.copy_(torch.tensor([3, 5, -1]))
    prefixes.copy_(torch.tensor([7, 11, -1]))
    layout = PrefillRequestLayout.from_cpu_lengths(2, inputs, prefixes)
    inputs[0] = 99
    prefixes[0] = 88
    assert layout.input_lengths == (3, 5)
    assert layout.prefix_lengths == (7, 11)


@pytest.mark.parametrize(
    "bad",
    [
        torch.ones(2),
        torch.ones(1, dtype=torch.int32),
        torch.ones((2, 1), dtype=torch.int32),
    ],
)
def test_request_layout_rejects_invalid_mirrors(bad):
    with pytest.raises(ValueError, match="CPU integer"):
        PrefillRequestLayout.from_cpu_lengths(2, bad, torch.zeros(2, dtype=torch.int64))


def test_hybrid_prefill_layout_refreshes_and_clears_with_router_metadata():
    fixture = router_tests.TestCacheGroupRouterTest()
    full, _ = fixture._router()
    backend = HybridLinearAttnBackend(full, CPUAttention(7), [0])
    prefixes, lengths = torch.tensor([2, 0]), torch.tensor([3, 2])
    full.init_forward_metadata(
        2,
        2,
        torch.arange(2),
        prefixes + lengths,
        ForwardMode.EXTEND,
        block_tables=fixture._tables(),
        block_tables_cpu={},
        extend_seq_lens=lengths,
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens=prefixes,
        extend_prefix_lens_cpu=prefixes,
        extend_replay_lens_cpu=torch.zeros_like(lengths),
        extend_prompt_lens_cpu=prefixes + lengths,
        extend_with_prefix=True,
        query_shard=None,
    )
    view = backend.prefill_request_layout()
    assert view == PrefillRequestLayout((3, 2), (2, 0))
    # A query-shard/model view does not consult any recurrent child metadata.
    full.refresh_decode_metadata(
        2,
        2,
        torch.arange(2),
        torch.tensor([6, 3]),
        forward_mode=ForwardMode.DECODE,
        block_tables=fixture._tables(),
    )
    assert backend.prefill_request_layout() is None
    assert view == PrefillRequestLayout((3, 2), (2, 0))
    group = backend.group_view("full_attention", 2)
    assert group.page_table.shape[0] == 2
    layer = PagedAttention(
        8, 16, 16**-0.5, 8, layer_id=0, rotary_emb=None, qk_norm=None
    )
    bind_cache_groups(layer, full.cache_pool)
    positions = torch.tensor([5, 2])
    slots = backend.prepare_sparse_kv_access(
        layer, None, positions, forward_mode=ForwardMode.DECODE
    )
    # The outer sparse entry resolves the same scheduler pages for this layer.
    torch.testing.assert_close(slots.long(), torch.tensor([6 * 4 + 1, 9 * 4 + 2]))
