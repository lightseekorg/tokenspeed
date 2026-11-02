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

"""Real DSA initialization and execution with a public reference provider."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import NoKernelFoundError
from tokenspeed_kernel.signature import format_signatures

from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.fixture
def registry(monkeypatch):
    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    monkeypatch.setitem(global_server_args_dict, "numerics", "rl-bitwise")
    monkeypatch.setitem(global_server_args_dict, "dsa_slot_order", "sorted")
    return registry


def _register(registry, invariant, *, graph):
    def reference(*, q, kv_cache, topk_slots, topk_lens, softmax_scale, **kwargs):
        result = []
        flat = kv_cache.reshape(-1, q.shape[-1]).float()
        for row in range(len(q)):
            keys = flat[topk_slots[row, : int(topk_lens[row])].long()]
            scores = (q[row].float()[:, None, :] * keys[None, :, :]).sum(-1)
            probability = torch.softmax(scores * softmax_scale, dim=-1)
            result.append(
                (probability[:, :, None] * keys[None, :, :512]).sum(1).to(q.dtype)
            )
        return torch.stack(result)

    for mode in ("dsa_prefill", "dsa_decode"):
        registry.register(
            KernelSpec(
                name=f"public_{mode}",
                family="attention",
                mode=mode,
                solution="public_reference",
                format_signatures=format_signatures("q", "dense", {torch.bfloat16}),
                features=(frozenset({"batch_invariant"}) if invariant else frozenset())
                | (frozenset({"cuda_graph"}) if graph else frozenset()),
                traits={
                    "slot_order": frozenset({"sorted"}),
                    "q_len": frozenset({1, 2}),
                },
            ),
            reference,
        )

    # The AMD dense delegate (MLAAttnBackend) plans its own batch-invariant
    # leaves at construction; these only satisfy planning and never run.
    def dense_unused(**kwargs):
        raise AssertionError("the sparse path must not call dense MLA")

    for mode in ("mla_extend_with_kvcache", "mla_decode_with_kvcache"):
        registry.register(
            KernelSpec(
                name=f"public_{mode}",
                family="attention",
                mode=mode,
                solution="public_reference",
                format_signatures=format_signatures(
                    ("q", "kv_cache"), "dense", {torch.bfloat16}
                ),
                features=frozenset({"batch_invariant"}),
            ),
            dense_unused,
        )


def _backend():
    spec = DSAConfig(
        backend_name="dsa",
        num_attention_heads=4,
        num_kv_heads=1,
        head_dim=576,
        attn_tp_size=1,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_cache_dim=576,
        scaling=192**-0.5,
        index_topk=4,
        index_head_dim=128,
        index_n_heads=16,
        index_k_format="bf16",
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=128,
        max_bs=2,
        components=(spec,),
        speculative_num_draft_tokens=2,
    )
    return DSABackend(config, spec, kernel_page_size=64)


def test_sparse_plans_keep_their_provider_across_batches_and_registry_changes(registry):
    _register(registry, True, graph=False)
    backend = _backend()
    for mode in ("dsa_prefill", "dsa_decode"):
        registry._unregister(f"public_{mode}")
    q = torch.randn(2, 4, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = torch.randn(64, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    slots = torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda", dtype=torch.int32)
    lengths = torch.full((2,), 3, device="cuda", dtype=torch.int32)
    common = dict(
        kv_cache=cache,
        sparse_kv_cache=None,
        max_seqlen_k=64,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=192**-0.5,
        page_size=64,
        slot_order="sorted",
    )
    expected = dsa_prefill(
        q,
        topk_slots=slots,
        topk_lens=lengths,
        kernel_plan=backend.prefill_kernel_plan,
        **common,
    )
    for width in (1, 2):
        actual = dsa_decode(
            q,
            topk_slots=slots,
            topk_lens=lengths,
            q_len_per_req=width,
            kernel_plan=backend.decode_kernel_plan,
            **common,
        )
        assert torch.equal(actual, expected)
    solo = dsa_decode(
        q[1:2],
        topk_slots=slots[1:2],
        topk_lens=lengths[1:2],
        kernel_plan=backend.decode_kernel_plan,
        **common,
    )
    assert torch.equal(solo, expected[1:2])


def test_sparse_backend_rejects_missing_invariance_at_initialization(registry):
    _register(registry, False, graph=False)
    with pytest.raises(NoKernelFoundError):
        _backend()


@pytest.mark.parametrize("graph", [False, True])
def test_sparse_plans_decide_decode_graph_capture(registry, graph):
    _register(registry, True, graph=graph)
    support = _backend().cuda_graph_support
    assert support.decode_graph is graph
    # DSA keeps prefills eager whatever the planned leaf declares.
    assert not support.prefill_graph
