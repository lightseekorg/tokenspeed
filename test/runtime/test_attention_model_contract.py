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
from test.runtime.cache_pool_test_utils import make_mha_pool_for_groups
from test.runtime.test_gdn_state_paging import _mamba_config_pair

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
from tokenspeed.runtime.layers.attention.backends.paged.msa import MSAHybridAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    Qwen4ExpBackend,
)
from tokenspeed.runtime.layers.attention.backends.state.mamba import MambaAttnBackend
from tokenspeed.runtime.layers.attention.backends.support import (
    CudaGraphSupport,
    TreeSupport,
)
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.paged_attention import PagedAttention, bind_cache_groups

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class CPUAcceptedRows:
    """Small extension publishing a selected forward row through the router."""

    cuda_graph_support = CudaGraphSupport()

    def __init__(self):
        self.rows = None
        self.output = None

    def validate_cache_pool(self, pool):
        if pool.arena.buffer.device.type != "cpu":
            raise ValueError("CPU rows require CPU storage")

    def bind_cache_pool(self, pool):
        self.rows = torch.zeros((4, 3))
        self.output = torch.zeros(4)

    def reset_batch(self):
        self.rows.zero_()

    def fixed_workspace_bytes(self):
        return (self.rows.numel() + self.output.numel()) * self.rows.element_size()

    def preallocate_verify_workspace(self, max_bs, draft_token_num):
        assert (max_bs, draft_token_num) == self.rows.shape
        return self.fixed_workspace_bytes()

    def model_binding(self, layer_id, binding_type):
        return self if binding_type is CPUAcceptedRows else None

    def tree_support(self):
        return TreeSupport(verify_blocker=None, draft_blocker=None)

    def commit_state_after_acceptance(
        self, accepted_lengths, *, num_extends, accepted_path
    ):
        assert accepted_path is None
        accepted = accepted_lengths[num_extends:]
        for row, count in enumerate(accepted.tolist()):
            if count:
                self.output[row] = self.rows[row, count - 1]


@pytest.mark.parametrize("num_extends", [0, 1])
def test_router_verify_rows_publish_through_hybrid_and_reset(num_extends):
    owners = [CPUAcceptedRows(), CPUAcceptedRows()]
    routers = [
        CacheGroupRouter(
            lambda gid, grain: router_tests._StubLeaf(grain, spec=3),
            is_draft=False,
            spec_num_tokens=3,
            device="cpu",
            consumed_group_ids=None,
            verify_state=owner,
        )
        for owner in owners
    ]
    backend = HybridLinearAttnBackend(*routers, [0])
    backend.set_cache_pool(router_tests._router_pool())
    backend.init_cuda_graph_state(4)
    assert backend.preallocate_verify_workspace(4, 3) == 128
    assert backend.fixed_workspace_bytes() == 128
    for layer, offset in [(0, 0), (1, 20)]:
        binding = backend.model_binding(layer, CPUAcceptedRows)
        binding.rows.copy_(torch.arange(12).reshape(4, 3) + offset)
    backend.commit_state_after_acceptance(
        torch.tensor([3] * num_extends + [1, 3, 0]),
        num_extends=num_extends,
        accepted_path=None,
    )
    torch.testing.assert_close(owners[0].output, torch.tensor([0.0, 5.0, 0.0, 0.0]))
    torch.testing.assert_close(owners[1].output, torch.tensor([20.0, 25.0, 0.0, 0.0]))
    for router, owner in zip(routers, owners):
        router.refresh_decode_metadata(
            2,
            2,
            torch.arange(2),
            torch.tensor([6, 3]),
            forward_mode=ForwardMode.DECODE,
            block_tables=router_tests.TestCacheGroupRouterTest()._tables(),
            for_graph_replay=True,
        )
        torch.testing.assert_close(owner.rows, torch.zeros((4, 3)))


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


class CPUModelRouter(CacheGroupRouter):
    """A model-binding owner with CPU metadata; only attention kernels are absent."""

    def __init__(self, factor):
        super().__init__(
            lambda group_id, grain: router_tests._StubLeaf(grain),
            is_draft=False,
            spec_num_tokens=1,
            device="cpu",
            consumed_group_ids=None,
        )
        self.input_transform = InputTransform(factor)

    def model_binding(self, layer_id, binding_type):
        if binding_type is InputTransform:
            return self.input_transform
        return super().model_binding(layer_id, binding_type)


def _msa_backend():
    spec = MSAConfig(
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=8,
        attn_tp_size=1,
        full_attn_backend_name="mha",
        compute_layer_types=("full", "sparse"),
        sparse_layer_ids=frozenset({1}),
        index_head_dim=8,
        index_n_heads=1,
        index_topk_blocks=1,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.float32,
        kv_cache_dtype=torch.float32,
        kv_cache_quant_method="none",
        context_len=32,
        max_bs=2,
        max_resident_requests=2,
        components=(spec,),
    )
    backend = MSAHybridAttnBackend(config, spec)
    # Keep the real composite and routers; replace GPU compute with the CPU
    # metadata leaves so both owners can offer independent model transforms.
    backend.full_router = CPUModelRouter(2)
    backend.sparse_router = CPUModelRouter(7)
    pool = make_mha_pool_for_groups(
        tuple(
            CacheGroupSpec(group, "full_history", 4, 1, replayable=False)
            for group in ("full_attention", "history_1")
        ),
        device="cpu",
    )
    backend.set_cache_pool(pool)
    backend.init_cuda_graph_state(2)
    tables = {
        group: torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        for group in ("full_attention", "history_1")
    }
    return backend, tables


def _publish_prefill(backend, tables, *, num_extends):
    lengths, prefixes = torch.tensor([3, 2]), torch.tensor([2, 0])
    backend.init_forward_metadata(
        2,
        num_extends,
        torch.arange(2),
        prefixes + lengths,
        ForwardMode.EXTEND if num_extends == 2 else ForwardMode.MIXED,
        block_tables=tables,
        block_tables_cpu={},
        extend_seq_lens=lengths[:num_extends],
        extend_seq_lens_cpu=lengths[:num_extends],
        extend_prefix_lens=prefixes[:num_extends],
        extend_prefix_lens_cpu=prefixes[:num_extends],
        extend_replay_lens_cpu=torch.zeros(num_extends, dtype=torch.int64),
        extend_prompt_lens_cpu=(prefixes + lengths)[:num_extends],
        extend_with_prefix=True,
        query_shard=None,
    )


@pytest.mark.parametrize("nested", [False, True])
def test_msa_model_inputs_and_request_layout_follow_the_routed_owner(nested):
    backend, tables = _msa_backend()
    if nested:
        backend = Qwen4ExpBackend(CPUAttention(2).test_config, backend, None, None)
    _publish_prefill(backend, tables, num_extends=2)
    layout = backend.prefill_request_layout()
    assert layout == PrefillRequestLayout((3, 2), (2, 0))
    positions = torch.tensor(layout.prefix_lengths)
    for layer_id, expected in ((0, [4, 0]), (1, [14, 0])):
        binding = backend.model_binding(layer_id, InputTransform)
        torch.testing.assert_close(binding.apply(positions), torch.tensor(expected))
    assert backend.model_binding(1, str) is None


@pytest.mark.parametrize("kind", ["router", "mamba", "msa"])
@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.IDLE])
def test_capture_discards_a_previous_live_prefill_layout(kind, mode):
    if kind == "router":
        fixture = router_tests.TestCacheGroupRouterTest()
        backend, _ = fixture._router()
        tables = fixture._tables()
    elif kind == "mamba":
        backend = MambaAttnBackend(
            *_mamba_config_pair(torch, heads=1, head_dim=8, spec_tokens=1)
        )
        backend.init_cuda_graph_state(2)
        tables = {}
    else:
        backend, tables = _msa_backend()
    _publish_prefill(backend, tables, num_extends=2)
    snapshot = backend.prefill_request_layout()
    assert snapshot == PrefillRequestLayout((3, 2), (2, 0))
    backend.init_forward_metadata_capture_cuda_graph(
        2, torch.arange(2), torch.tensor([5, 2]), mode, block_tables=tables
    )
    assert backend.prefill_request_layout() is None
    # A retained view is a snapshot; clearing the owner must not mutate it.
    assert snapshot == PrefillRequestLayout((3, 2), (2, 0))
    _publish_prefill(backend, tables, num_extends=2)
    assert backend.prefill_request_layout() == snapshot


def test_mamba_mixed_keeps_prefill_layout_while_preparing_its_decode_tail():
    backend = MambaAttnBackend(
        *_mamba_config_pair(torch, heads=1, head_dim=8, spec_tokens=2)
    )
    backend.init_cuda_graph_state(2)
    _publish_prefill(backend, {}, num_extends=1)
    assert backend.prefill_request_layout() == PrefillRequestLayout((3,), (2,))
    torch.testing.assert_close(
        backend.forward_metadata.query_start_loc,
        torch.tensor([0, 3, 5], dtype=torch.int32),
    )
