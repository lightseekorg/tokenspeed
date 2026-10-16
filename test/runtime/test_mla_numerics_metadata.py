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

"""MLA numerical policy prepares the same absorbed inputs in every phase.

These tests exercise the real backend and metadata kernels; the numerical
attention-provider acceptance belongs to its registered implementation.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.backends.support import (
    resolve_cuda_graph_support,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.fixture(autouse=True)
def prefill_configuration(monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 1024)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 1)


@pytest.fixture
def reference_provider(monkeypatch):
    from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
    from tokenspeed_kernel.signature import format_signatures

    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)

    def attention_reference(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        softmax_scale,
        *,
        cu_seqlens_q=None,
        **kwargs,
    ):
        # Deliberately per-query: neither matrix tiling nor a split over tokens.
        decode = q.ndim == 4
        queries = q[:, 0] if decode else q
        results = []
        for request in range(len(cache_seqlens)):
            start = request if decode else int(cu_seqlens_q[request])
            stop = request + 1 if decode else int(cu_seqlens_q[request + 1])
            for index in range(start, stop):
                length = int(cache_seqlens[request]) - (stop - index - 1)
                logical = torch.arange(length, device=q.device)
                keys = kv_cache[
                    page_table[request, logical // kv_cache.shape[1]].long(),
                    logical % kv_cache.shape[1],
                    0,
                ].float()
                scores = (queries[index].float()[:, None, :] * keys[None, :, :]).sum(-1)
                probs = torch.softmax(scores * softmax_scale, dim=-1)
                results.append(
                    (probs[:, :, None] * keys[None, :, :512]).sum(1).to(q.dtype)
                )
        result = torch.stack(results)
        return result[:, None] if decode else result

    for mode in ("mla_extend_with_kvcache", "mla_decode_with_kvcache"):
        registry.register(
            KernelSpec(
                name=f"test_{mode}",
                family="attention",
                mode=mode,
                solution="test_reference",
                format_signatures=format_signatures(
                    ("q", "kv_cache"), "dense", {torch.bfloat16}
                ),
                features=frozenset({"batch_invariant"}),
            ),
            attention_reference,
        )
    return registry


def _backend():
    spec = MLAConfig(
        backend_name="mla",
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
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=64,
        context_len=1024,
        max_bs=2,
        components=(spec,),
    )
    return MLAAttnBackend(config, spec, kernel_page_size=64)


def _prepare(backend, prefixes):
    prefix = torch.tensor(prefixes, dtype=torch.int32)
    lengths = torch.tensor([3, 5], dtype=torch.int32)
    pages = torch.arange(1, 33, dtype=torch.int32, device="cuda").view(2, 16)
    backend.init_forward_metadata(
        bs=2,
        num_extends=2,
        seq_lens=(prefix + lengths).cuda(),
        page_table=pages,
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=lengths.cuda(),
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens=prefix.cuda(),
        extend_prefix_lens_cpu=prefix,
        extend_with_prefix=any(prefixes),
        query_shard=None,
        page_table_cpu=pages.cpu(),
    )
    return backend.forward_prefill_metadata, pages


@pytest.mark.parametrize("prefixes", [(0, 0), (0, 7), (130, 514)])
def test_strict_prefill_keeps_absorbed_paged_inputs(
    monkeypatch, prefixes, reference_provider
):
    monkeypatch.setitem(global_server_args_dict, "numerics", "rl-bitwise")
    backend = _backend()
    metadata, pages = _prepare(backend, prefixes)
    assert metadata.use_absorbed_cached_extend
    assert torch.equal(metadata.page_table, pages)
    assert metadata.cum_extend_seq_lens.cpu().tolist() == [0, 3, 8]
    assert metadata.cum_seq_lens_kv.cpu().tolist() == [
        0,
        prefixes[0] + 3,
        sum(prefixes) + 8,
    ]
    assert backend.extend_kernel_plan.kernel.name == "test_mla_extend_with_kvcache"
    assert not backend.supports_mla_projected_value_decode
    assert not resolve_cuda_graph_support(backend).prefill_graph


def test_auto_initial_prefill_keeps_expanded_inputs(monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    backend = _backend()
    metadata, _ = _prepare(backend, (0, 0))
    assert not metadata.use_absorbed_cached_extend
    assert metadata.page_table is None
    assert metadata.cum_seq_lens_kv is None
    assert backend.kernel_solution is None
    assert backend.supports_mla_projected_value_decode


def test_planned_attention_executes_the_same_query_in_both_phases(
    monkeypatch, reference_provider
):
    from tokenspeed_kernel.ops.attention.mla import (
        mla_decode_with_kvcache,
        mla_extend_with_kvcache,
    )

    monkeypatch.setitem(global_server_args_dict, "numerics", "rl-bitwise")
    backend = _backend()
    q = torch.randn(1, 4, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = torch.randn(1, 64, 1, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    pages = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([5], device="cuda", dtype=torch.int32)
    common = dict(
        kv_cache=cache,
        page_table=pages,
        cache_seqlens=lengths,
        max_seqlen_k=64,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=192**-0.5,
    )
    extended = mla_extend_with_kvcache(
        q=q,
        cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 5], device="cuda", dtype=torch.int32),
        max_seqlen_q=1,
        plan=backend.extend_kernel_plan,
        **common,
    )
    decoded = mla_decode_with_kvcache(
        q=q[:, None], plan=backend.decode_kernel_plan, **common
    )
    assert torch.equal(extended, decoded[:, 0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
