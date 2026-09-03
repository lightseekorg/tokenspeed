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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.cache.transfer.layout import select_layer_fields
from tokenspeed.runtime.configs.model_config import is_qwen4_exp
from tokenspeed.runtime.configs.qwen4_exp_config import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention import registry as attention_registry
from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
    CacheGroupGeometry,
)
from tokenspeed.runtime.layers.attention.backends.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.backends.qwen4_exp import (
    Qwen4ExpMambaAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
    prepare_cache_setup,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    compute_cache_group_page_counts,
)
from tokenspeed.runtime.layers.attention.qsa import (
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    QSAIndexer,
    qsa_compressed_field,
    qsa_raw_key_field,
    qsa_rope_position_field,
)
from tokenspeed.runtime.layers.hyperconnection import (
    GatedResidualSimple,
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from tokenspeed.runtime.layers.quantization.utils import should_exclude_quant_module
from tokenspeed.runtime.layers.qwen4_exp_ple import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
    _nth_prime_after,
    quantize_ple_embedding_rows,
    qwen4_exp_ple_context_field,
    qwen4_exp_ple_conv_field,
)
from tokenspeed.runtime.models import qwen4_exp_nextn
from tokenspeed.runtime.models.qwen4_exp import (
    Qwen4ExpAttentionDecoderLayer,
    Qwen4ExpModel,
    _qwen4_exp_uses_sigmoid_output_gate,
    _qwen4_exp_uses_sparse_moe,
    _Qwen4ExpRMSNormGated,
    load_qwen4_exp_weights,
)

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForCausalLMNextN",
    ],
)
def test_is_qwen4_exp_uses_resolved_architecture(architecture: str) -> None:
    assert is_qwen4_exp(SimpleNamespace(architectures=[architecture]))
    assert not is_qwen4_exp(SimpleNamespace(architectures=["Qwen3_5ForCausalLM"]))


def test_qwen4_exp_modelopt_exclusions_match_shared_expert_fusion() -> None:
    exclusions = [
        "model.layers.0.mlp.shared_expert.gate_proj",
        "model.layers.0.mlp.shared_expert.up_proj",
    ]

    assert should_exclude_quant_module(
        "model.layers.0.mlp.shared_expert.gate_up_proj", exclusions
    )


def test_qwen4_exp_gdn_norm_uses_sigmoid_output_gate() -> None:
    norm = _Qwen4ExpRMSNormGated(hidden_size=2, eps=1e-6)
    value = torch.tensor([[3.0, 4.0]])

    torch.testing.assert_close(norm(value, torch.zeros_like(value)), norm(value) * 0.5)


@_requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen4_exp_gdn_norm_fused_matches_eager(dtype: torch.dtype) -> None:
    torch.manual_seed(73)
    head_v_dim, eps, rows = 128, 1e-6, 257
    norm = _Qwen4ExpRMSNormGated(head_v_dim, eps).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.rand(head_v_dim, device="cuda") + 0.5)
    x = torch.randn(rows, head_v_dim, device="cuda", dtype=dtype)
    z = torch.randn(rows, head_v_dim, device="cuda", dtype=dtype)

    def reference(value, gate):
        out = value.float()
        variance = out.square().mean(dim=-1, keepdim=True)
        out = out * torch.rsqrt(variance + eps)
        if gate is not None:
            out = out * torch.sigmoid(gate.float())
        return (out * norm.weight.float()).to(dtype)

    torch.testing.assert_close(norm(x, z), reference(x, z), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(norm(x), reference(x, None), rtol=2e-2, atol=2e-2)
    # Strided z views (the reshape chain in the linear-attn path) stay exact.
    z_wide = torch.randn(rows, 2, head_v_dim, device="cuda", dtype=dtype)
    torch.testing.assert_close(
        norm(x, z_wide[:, 1]), reference(x, z_wide[:, 1]), rtol=2e-2, atol=2e-2
    )


def test_qwen4_exp_selects_checkpoint_output_gate_type() -> None:
    assert _qwen4_exp_uses_sigmoid_output_gate(
        SimpleNamespace(output_gate_type="sigmoid")
    )
    assert not _qwen4_exp_uses_sigmoid_output_gate(
        SimpleNamespace(output_gate_type=None)
    )
    assert not _qwen4_exp_uses_sigmoid_output_gate(
        SimpleNamespace(output_gate_type="silu")
    )


def test_qwen4_exp_decoder_policies_are_model_local() -> None:
    dense_qwen38 = SimpleNamespace(num_experts=None)
    moe_qwen38 = SimpleNamespace(
        model_type="qwen4_exp_text",
        num_experts=8,
        attn_output_gate=False,
    )

    assert not _qwen4_exp_uses_sparse_moe(dense_qwen38)
    assert _qwen4_exp_uses_sparse_moe(moe_qwen38)
    assert Qwen4ExpAttentionDecoderLayer._uses_sparse_moe(moe_qwen38)
    assert Qwen4ExpAttentionDecoderLayer._uses_attention_output_gate(moe_qwen38)
    assert moe_qwen38.model_type == "qwen4_exp_text"


def test_qwen4_exp_config_normalizes_layer_and_ple_geometry() -> None:
    config = Qwen4ExpTextConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        ple_layer_ids=[3, 1, 3],
        ple_conv_kernel_size=4,
        ngram_size=3,
        hc_count=4,
        num_experts=None,
    )

    assert config.layers_block_type == [
        "linear_attention",
        "attention",
        "linear_attention",
        "attention",
    ]
    assert config.layer_types == [
        LINEAR_ATTENTION,
        FULL_ATTENTION,
        LINEAR_ATTENTION,
        FULL_ATTENTION,
    ]
    assert config.short_conv_layer_ids == [0, 2]
    assert config.short_conv_state_shape == (64, 9)
    assert config.ngram_context_len == 2


def test_qwen4_exp_flat_config_preserves_text_rope_parameters() -> None:
    config = Qwen4ExpConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=None,
        tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
    )

    assert config.text_config.rope_parameters["rope_theta"] == 1_000_000.0
    assert config.text_config.tie_word_embeddings
    assert config.text_config.model_type == "qwen4_exp_text"


@_requires_cuda
def test_hyperconnection_mix_and_combine_shapes() -> None:
    mixer = GatedResidualSimple(
        HyperConnectionConfig(
            hc_count=4,
            hidden_size=8,
            hc_lowrank=4,
            params_dtype=torch.float32,
        )
    ).cuda()
    hyper_input = torch.randn(5, 32, device="cuda")
    mixed, residuals = mixer.mix(hyper_input)
    combined = mixer.combine(torch.randn(5, 8, device="cuda"), residuals)

    assert mixed.shape == (5, 8)
    assert combined.shape == hyper_input.shape
    assert torch.isfinite(mixed).all()
    assert torch.isfinite(combined).all()


@_requires_cuda
def test_hyperconnection_norm_for_reuses_the_mix_time_norm() -> None:
    mixer = GatedResidualSimple(
        HyperConnectionConfig(
            hc_count=4,
            hidden_size=8,
            hc_lowrank=4,
            params_dtype=torch.float32,
        )
    ).cuda()
    hyper_input = torch.randn(6, 32, device="cuda")
    _, residuals = mixer.mix(hyper_input)
    sliced = hyper_input[2:5]
    unrelated = torch.randn(3, 32, device="cuda")
    sliced_reference = mixer.hc_norm(sliced)
    unrelated_reference = mixer.hc_norm(unrelated)

    recomputes = []
    mixer.hc_norm.register_forward_hook(
        lambda module, args, output: recomputes.append(args[0].shape)
    )

    # All-reduce hands the residual back untouched: borrow the tensor as-is.
    value, normalized, inject = mixer.norm_for(hyper_input, residuals)
    assert value is hyper_input
    assert normalized is residuals[1]
    assert inject is residuals[2]

    # Reduce-scatter slices rows off it: the same rows of the norm still apply.
    value, normalized, inject = mixer.norm_for(sliced, residuals)
    assert value is sliced
    assert torch.equal(normalized, sliced_reference)
    assert torch.equal(inject, residuals[2][2:5])
    assert recomputes == []

    # An unrelated residual has to be normalized again.
    _, normalized, _ = mixer.norm_for(unrelated, residuals)
    assert torch.equal(normalized, unrelated_reference)
    assert recomputes == [unrelated.shape]


@_requires_cuda
def test_hyperconnection_fused_projection_matches_split_checkpoint_weights() -> None:
    hc_count, hidden_size, lowrank = 4, 8, 6
    mixer = GatedResidualSimple(
        HyperConnectionConfig(
            hc_count=hc_count,
            hidden_size=hidden_size,
            hc_lowrank=lowrank,
            params_dtype=torch.float32,
        )
    ).cuda()
    down_weight = torch.randn(lowrank, hc_count * hidden_size, device="cuda")
    inject_weight = torch.randn(hc_count, hc_count * hidden_size, device="cuda")
    param = mixer.mix_inject_proj.weight
    loader = param.weight_loader
    loader(param, down_weight, "mix")
    loader(param, inject_weight, "inject")

    # The shared 1 / hc_count scale is exactly folded for power-of-two HC.
    torch.testing.assert_close(param[:lowrank], down_weight / hc_count)
    torch.testing.assert_close(param[lowrank:], inject_weight / hc_count)

    hyper_input = torch.randn(5, hc_count * hidden_size, device="cuda")
    block_output = torch.randn(5, hidden_size, device="cuda")
    mixed, residuals = mixer.mix(hyper_input)
    combined = mixer.combine(block_output, residuals)

    normalized = residuals[1]
    branches = normalized.unflatten(-1, (hc_count, hidden_size))
    gate = torch.nn.functional.silu(
        torch.nn.functional.linear(normalized, down_weight) / hc_count
    )
    weights = torch.sigmoid(mixer.input_mix_weight_up(gate)).unflatten(
        -1, (hc_count, hidden_size)
    )
    torch.testing.assert_close(mixed, (weights * branches).mean(dim=-2))

    inject = 2 * torch.sigmoid(
        torch.nn.functional.linear(normalized, inject_weight) / hc_count
    )
    expected = hyper_input.unflatten(
        -1, (hc_count, hidden_size)
    ) + block_output.unsqueeze(-2) * inject.unsqueeze(-1)
    torch.testing.assert_close(combined, expected.flatten(-2))


def test_qwen4_exp_loads_fp8_scales_only_into_attention_layers(monkeypatch) -> None:
    paged_attention = SimpleNamespace()
    model = SimpleNamespace(
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_rank=0, tp_size=1)),
        config=SimpleNamespace(num_hidden_layers=2, model_type="qwen4_exp_text"),
        layers=[SimpleNamespace(attn=paged_attention), SimpleNamespace()],
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.models.qwen4_exp.kv_cache_scales_loader",
        lambda *args: [(0, 0.25), (1, 0.5)],
    )

    Qwen4ExpModel.load_kv_cache_scales(model, "scales.json")

    assert paged_attention.k_scale == 0.25
    assert paged_attention.v_scale == 0.25
    assert paged_attention.k_scale_float == 0.25
    assert paged_attention.v_scale_float == 0.25


def test_qwen4_exp_qsa_page_table_expansion() -> None:
    assert QSAIndexer._page_table_expansion(64, 256) == 4
    assert QSAIndexer._page_table_expansion(256, 256) == 1
    with pytest.raises(ValueError, match="divisible"):
        QSAIndexer._page_table_expansion(96, 256)


# Qwen4-Exp's three history groups as the recipe declares them: the
# full-attention KV at P, the compressed QSA keys (64 rows x ratio 4) and the
# recent raw-key window (64 rows x 1).
_QSA_GROUP_GRANULARITIES = {
    FULL_ATTENTION: 256,
    QWEN4_EXP_QSA_CACHE_GROUP: 256,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP: 64,
}


def _qsa_router(
    *, kernel_page_size: int | None, max_bs: int = 4, spec: int = 1
) -> CacheGroupRouter:
    """A real ``CacheGroupRouter`` over real MHA leaves for the Qwen4-Exp
    history groups, bound the way ``set_cache_pool`` binds it (CPU)."""
    component = MHAConfig(
        backend_name="mha",
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=256,
        kernel_page_size=kernel_page_size,
        context_len=1024,
        max_bs=max_bs,
        speculative_num_draft_tokens=spec,
        components=(component,),
    )

    def leaf_factory(group_id: str, block_granularity: int) -> MHAAttnBackend:
        del group_id
        return MHAAttnBackend(
            config,
            component,
            kernel_page_size=MHAAttnBackend.resolve_kernel_page_size(
                config, block_granularity
            ),
        )

    router = CacheGroupRouter(
        leaf_factory, is_draft=False, spec_num_tokens=spec, device="cpu"
    )
    router.bind(
        CacheGroupGeometry(
            granularities=dict(_QSA_GROUP_GRANULARITIES),
            families={gid: "history" for gid in _QSA_GROUP_GRANULARITIES},
            full_history_group_id=FULL_ATTENTION,
        ),
        {
            gid: leaf_factory(gid, granularity)
            for gid, granularity in _QSA_GROUP_GRANULARITIES.items()
        },
    )
    router.init_cuda_graph_state(max_bs)
    return router


def _qsa_extend_round(router: CacheGroupRouter, block_tables: dict, seq_lens) -> None:
    bs = len(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    ones = torch.ones(bs, dtype=torch.int32)
    router.init_forward_metadata(
        bs,
        bs,
        torch.arange(1, bs + 1, dtype=torch.int32),
        seq_lens,
        ForwardMode.EXTEND,
        block_tables=block_tables,
        extend_seq_lens=ones,
        extend_seq_lens_cpu=ones,
        extend_prefix_lens=seq_lens - 1,
        extend_prefix_lens_cpu=seq_lens - 1,
        extend_with_prefix=True,
    )


def test_qwen4_exp_qsa_reads_group_geometry_from_the_router() -> None:
    indexer = object.__new__(QSAIndexer)
    router = _qsa_router(kernel_page_size=64)
    raw = torch.tensor([[3, 5], [7, -1]], dtype=torch.int32)
    _qsa_extend_round(
        router, {gid: raw for gid in _QSA_GROUP_GRANULARITIES}, seq_lens=[300, 9]
    )

    # A 64-token kernel page splits the 256-token compressed block four ways
    # and leaves the 64-token recent block whole; each table is the router's
    # batch-ordered stack view for exactly the forward's rows.
    table, expansion = indexer._group_geometry(
        router, QWEN4_EXP_QSA_CACHE_GROUP, 256, bs=2
    )
    assert expansion == 4
    assert table.shape == (2, router.leaves[QWEN4_EXP_QSA_CACHE_GROUP].max_num_pages)
    assert table[:, :8].tolist() == [
        [12, 13, 14, 15, 20, 21, 22, 23],
        [28, 29, 30, 31, 0, 0, 0, 0],
    ]
    table, expansion = indexer._group_geometry(
        router, QWEN4_EXP_QSA_RECENT_CACHE_GROUP, 64, bs=2
    )
    assert expansion == 1
    assert table[:, :2].tolist() == [[3, 5], [7, 0]]
    assert router.stacks.group_kernel_page_size(FULL_ATTENTION) == 64

    # Without a config override every leaf runs at its group's own grain.
    native = _qsa_router(kernel_page_size=None)
    assert indexer._group_geometry(native, QWEN4_EXP_QSA_CACHE_GROUP, 256, bs=1)[1] == 1
    assert native.stacks.group_kernel_page_size(QWEN4_EXP_QSA_RECENT_CACHE_GROUP) == 64


def test_qwen4_exp_qsa_metadata_follows_the_full_attention_leaf_slot() -> None:
    indexer = object.__new__(QSAIndexer)
    router = _qsa_router(kernel_page_size=64, spec=3)
    raw = torch.tensor([[3, 5]], dtype=torch.int32)
    tables = {gid: raw for gid in _QSA_GROUP_GRANULARITIES}
    leaf = router.leaves[FULL_ATTENTION]
    hybrid = SimpleNamespace(full_attn_backend=router)

    _qsa_extend_round(router, tables, seq_lens=[300])
    ctx = SimpleNamespace(attn_backend=hybrid, forward_mode=ForwardMode.EXTEND)
    assert indexer._metadata(ctx) is leaf.forward_extend_metadata
    assert indexer._seq_lens(indexer._metadata(ctx)).tolist() == [300]

    router.refresh_decode_metadata(
        1,
        1,
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([301], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
    )
    ctx = SimpleNamespace(attn_backend=router, forward_mode=ForwardMode.DECODE)
    assert indexer._metadata(ctx) is leaf.forward_decode_metadata
    assert indexer._seq_lens(indexer._metadata(ctx)).tolist() == [301]


def test_qwen4_exp_qsa_topk_solution_reads_env(monkeypatch) -> None:
    indexer = object.__new__(QSAIndexer)
    small = torch.empty(1, 64)  # 1 x 4096 blocks x 4B fits any default budget
    large = torch.empty(1, 8192)  # 1 x 524288 blocks x 4B exceeds 1 MiB

    monkeypatch.delenv("TOKENSPEED_QWEN4_EXP_QSA_TOPK_PATH", raising=False)
    monkeypatch.delenv("TOKENSPEED_QWEN4_EXP_QSA_MAX_LOGITS_MB", raising=False)
    assert indexer._topk_solution(1, small, 1, 64) == "logits"
    assert indexer._topk_solution(1, large, 1, 64) == "logits"

    # A tighter budget flips only the oversized batch onto the stream path.
    monkeypatch.setenv("TOKENSPEED_QWEN4_EXP_QSA_MAX_LOGITS_MB", "1")
    assert indexer._topk_solution(1, small, 1, 64) == "logits"
    assert indexer._topk_solution(1, large, 1, 64) == "stream"

    # Explicit backends pin the routing regardless of shape or budget.
    for pinned in ("stream", "logits"):
        monkeypatch.setenv("TOKENSPEED_QWEN4_EXP_QSA_TOPK_PATH", pinned)
        assert indexer._topk_solution(1, large, 1, 64) == pinned

    monkeypatch.setenv("TOKENSPEED_QWEN4_EXP_QSA_TOPK_PATH", "bogus")
    with pytest.raises(ValueError, match="TOPK_PATH"):
        indexer._topk_solution(1, small, 1, 64)


def test_qwen4_exp_qsa_owns_nonpersistent_radix_workspace(monkeypatch) -> None:
    monkeypatch.setattr(
        "tokenspeed.runtime.layers.attention.qsa.indexer.ReplicatedLinear",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.layers.attention.qsa.indexer.GemmaRMSNorm",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    config = SimpleNamespace(
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        hidden_size=64,
        rms_norm_eps=1e-6,
    )
    indexer = QSAIndexer(
        config,
        mapping=SimpleNamespace(),
        layer_id=0,
        quant_config=None,
        prefix="model.layers.0.attn",
        rotary_emb=SimpleNamespace(rotary_dim=16),
    )

    assert indexer._persistent_topk_workspace.dtype == torch.uint8
    assert indexer._persistent_topk_workspace.numel() == 1024 * 1024
    assert "_persistent_topk_workspace" not in indexer.state_dict()


def test_qwen4_exp_qsa_publishes_and_reuses_context_topk() -> None:
    rows = torch.tensor([[3, 1, -1], [5, 2, 0]], dtype=torch.int32)
    indexer = QSAIndexer.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.layer_id = 3
    indexer.share_topk_for_mtp_iteration = True
    indexer.compressed_token_page_size = 256
    indexer.recent_page_size = 64
    indexer.compress_ratio = 4

    router = _qsa_router(kernel_page_size=64)
    raw = torch.tensor([[3], [5]], dtype=torch.int32)
    _qsa_extend_round(
        router, {gid: raw for gid in _QSA_GROUP_GRANULARITIES}, seq_lens=[8, 9]
    )
    logical = torch.tensor([7, 8])
    requests = torch.tensor([0, 1])
    cache_locs = torch.tensor([1, 2], dtype=torch.int32)
    updates = []
    cache_accesses = []
    group_locs_calls = []
    pool = SimpleNamespace(
        layerwise_load_tracker=SimpleNamespace(
            wait_for_layer=lambda layer_id: cache_accesses.append(("wait", layer_id))
        )
    )

    indexer._decode_query_lengths = lambda ctx, total_tokens: None
    indexer._logical_layout = lambda *args, **kwargs: (logical, requests, 1)
    indexer._project_qk = lambda hidden, positions: (
        torch.zeros((2, 1, 1)),
        torch.ones((2, 1, 1)),
    )

    def fields(actual_pool):
        assert actual_pool is pool
        cache_accesses.append(("fields", indexer.layer_id))
        return None, torch.empty(0), None

    indexer._fields = fields

    def group_cache_locs(*args):
        group_locs_calls.append(args)
        return cache_locs, cache_locs, torch.ones(2, dtype=torch.int32)

    indexer._group_cache_locs = group_cache_locs
    indexer._write_and_compress = lambda *args, **kwargs: updates.append((args, kwargs))

    selections = []
    indexer._select_tokens = lambda *args, **kwargs: (
        selections.append((args, kwargs)) or rows
    )
    ctx = SimpleNamespace(
        bs=2,
        num_extends=2,
        forward_mode=ForwardMode.EXTEND,
        accept_lengths=None,
        attn_backend=SimpleNamespace(full_attn_backend=router),
        token_to_kv_pool=pool,
        dsa_decode_topk=None,
    )

    actual = indexer(torch.zeros((2, 4)), torch.tensor([7, 8]), ctx)

    torch.testing.assert_close(actual, rows)
    torch.testing.assert_close(ctx.dsa_decode_topk, rows)
    assert cache_accesses == [("wait", 3), ("fields", 3)]
    assert len(updates) == 1
    assert len(selections) == 1
    # The cache-location and top-k kernels address the router's batch-ordered
    # kernel page tables for the two QSA groups, with the expansion each
    # leaf's kernel page size implies (256/64 for compressed, 64/64 recent).
    _, _, qsa_table, qsa_expansion, _, recent_table, recent_expansion, _, _ = (
        group_locs_calls[0]
    )
    stacks = router.stacks
    assert torch.equal(qsa_table, stacks.table(QWEN4_EXP_QSA_CACHE_GROUP, 2))
    assert qsa_table[:, :4].tolist() == [[12, 13, 14, 15], [20, 21, 22, 23]]
    assert qsa_expansion == 4
    assert torch.equal(recent_table, stacks.table(QWEN4_EXP_QSA_RECENT_CACHE_GROUP, 2))
    assert recent_table[:, :1].tolist() == [[3], [5]]
    assert recent_expansion == 1
    assert selections[0][0][3] is qsa_table
    assert selections[0][1]["qsa_page_expansion"] == 4

    def fail_selection(*args, **kwargs):
        raise AssertionError("top-k selection must be skipped")

    indexer._select_tokens = fail_selection
    actual = indexer(torch.zeros((2, 4)), torch.tensor([9, 10]), ctx)

    torch.testing.assert_close(actual, rows)
    assert cache_accesses == [
        ("wait", 3),
        ("fields", 3),
        ("wait", 3),
        ("fields", 3),
    ]
    assert len(updates) == 2


def test_qwen4_exp_nextn_compacts_context_topk_for_mtp_decode() -> None:
    rows = torch.arange(6 * 4, dtype=torch.int32).reshape(6, 4)

    prefill_topk, decode_topk = (
        qwen4_exp_nextn.Qwen4ExpForCausalLMNextN.prepare_dsa_topk_for_mtp_decode(
            (None, rows),
            torch.tensor([2, 5], dtype=torch.int32),
            num_prefill_rows=1,
        )
    )

    assert prefill_topk is None
    torch.testing.assert_close(decode_topk, rows[[2, 5]])


@_requires_cuda
def test_qwen4_exp_qsa_expands_mtp_layout_and_cache_locations() -> None:
    device = "cuda"
    indexer = object.__new__(QSAIndexer)
    metadata = SimpleNamespace(
        cache_seqlens_int32=torch.tensor([8, 12], dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
    )
    query_lengths = torch.full((2,), 4, dtype=torch.long, device=device)

    logical, requests, lengths = indexer._logical_layout(
        metadata,
        total_tokens=8,
        bs=2,
        query_lengths=query_lengths,
    )
    # Compressed pages arrive expanded 4x at consumer granularity (logical
    # pages 3 and 7 become entries 12 and 28); the recent table is already
    # at logical granularity. Complete-block counts ride along in the same
    # fused launch.
    qsa_locs, recent_locs, complete_blocks = indexer._group_cache_locs(
        logical,
        requests,
        torch.tensor([[12], [28]], dtype=torch.int32, device=device),
        4,
        256,
        torch.tensor([[3], [7]], dtype=torch.int32, device=device),
        1,
        256,
        4,
    )

    expected = torch.tensor(
        [772, 773, 774, 775, 1800, 1801, 1802, 1803], dtype=torch.int32
    )
    torch.testing.assert_close(lengths.cpu(), query_lengths.cpu())
    torch.testing.assert_close(logical.cpu(), torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]))
    torch.testing.assert_close(requests.cpu(), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    torch.testing.assert_close(qsa_locs.cpu(), expected)
    torch.testing.assert_close(recent_locs.cpu(), expected)
    torch.testing.assert_close(
        complete_blocks.cpu(),
        torch.tensor([1, 1, 1, 2, 2, 2, 2, 3], dtype=torch.int32),
    )


@_requires_cuda
def test_qwen4_exp_qsa_ignores_stale_prefill_lengths_during_draft_decode() -> None:
    device = "cuda"
    indexer = object.__new__(QSAIndexer)
    ctx = SimpleNamespace(bs=1, forward_mode=ForwardMode.DECODE)
    metadata = SimpleNamespace(
        cache_seqlens_int32=torch.tensor([24], dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor([0, 23], dtype=torch.int32, device=device),
    )

    query_lengths = indexer._decode_query_lengths(ctx, total_tokens=1)
    logical, requests, lengths = indexer._logical_layout(
        metadata,
        total_tokens=1,
        bs=1,
        query_lengths=query_lengths,
    )

    assert lengths == 1
    torch.testing.assert_close(logical.cpu(), torch.tensor([23]))
    torch.testing.assert_close(requests.cpu(), torch.tensor([0]))


@_requires_cuda
def test_qwen4_exp_qsa_rebuilds_layout_after_mtp_prefill_row_gather() -> None:
    device = "cuda"
    indexer = object.__new__(QSAIndexer)
    ctx = SimpleNamespace(bs=1, forward_mode=ForwardMode.EXTEND)
    metadata = SimpleNamespace(
        cache_seqlens_int32=torch.tensor([23], dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor([0, 23], dtype=torch.int32, device=device),
    )

    query_lengths = indexer._decode_query_lengths(
        ctx,
        total_tokens=1,
        force_uniform=True,
    )
    logical, requests, lengths = indexer._logical_layout(
        metadata,
        total_tokens=1,
        bs=1,
        query_lengths=query_lengths,
    )

    assert lengths == 1
    torch.testing.assert_close(logical.cpu(), torch.tensor([22]))
    torch.testing.assert_close(requests.cpu(), torch.tensor([0]))


def test_qwen4_exp_qsa_draft_write_mask_keeps_only_accepted_prefix() -> None:
    ctx = SimpleNamespace(
        bs=3,
        num_extends=1,
        accept_lengths=torch.tensor([0, 2, 1]),
    )
    seq_lens = torch.tensor([3, 14, 23])
    lengths = torch.tensor([3, 4, 4])
    logical = torch.tensor([0, 1, 2, 10, 11, 12, 13, 19, 20, 21, 22])
    requests = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    recent_locs = torch.ones_like(logical, dtype=torch.int32)

    actual = QSAIndexer._draft_accepted_write_mask(
        ctx,
        seq_lens,
        lengths,
        logical,
        requests,
        recent_locs,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [True, True, True, True, True, False, False, True, False, False, False]
        ),
    )


def _qsa_cache_test_indexer(device: str = "cuda"):
    indexer = object.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_head_dim = 2
    indexer.compress_ratio = 4
    indexer.compressed_token_page_size = 256
    indexer.recent_page_size = 64
    indexer.token_topk = 8
    indexer.block_topk = 2
    indexer.k_layernorm = SimpleNamespace(
        gemma_weight=torch.ones(2, device=device),
        variance_epsilon=0.0,
    )
    # Identity neox RoPE table: cos == 1, sin == 0 for every position.
    # Sized to cover the 1000 sentinel used by the request-isolation test.
    identity_cache = torch.zeros(1024, 2, device=device)
    identity_cache[:, 0] = 1.0
    indexer.rotary_emb = SimpleNamespace(
        cos_sin_cache=identity_cache,
        is_neox_style=True,
        rotary_dim=2,
        mrope_section=None,
    )
    raw = torch.zeros((3, 4, 1, 2), dtype=torch.float32, device=device)
    compressed = torch.zeros((3, 64, 1, 2), dtype=torch.float32, device=device)
    rope_positions = torch.zeros((3, 3), dtype=torch.int64, device=device)
    indexer._fields = lambda pool: (raw, compressed, rope_positions)
    indexer._draft_scratch = {}
    indexer.register_buffer(
        "_persistent_topk_workspace",
        torch.empty((1024 * 1024,), dtype=torch.uint8, device=device),
        persistent=False,
    )
    return indexer, SimpleNamespace(), raw, compressed, rope_positions


def _qsa_norm(values: torch.Tensor) -> torch.Tensor:
    """Reference Gemma RMSNorm with unit weight and zero epsilon."""

    return values * torch.rsqrt((values * values).mean())


@_requires_cuda
def test_qsa_project_norms_and_rotates_queries_without_rotating_raw_keys() -> None:
    device = "cuda"

    class Projection(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    indexer = QSAIndexer.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_n_heads = 2
    indexer.index_kv_heads = 1
    indexer.index_head_dim = 2
    indexer.index_qk_proj = Projection()
    indexer.q_layernorm = SimpleNamespace(
        gemma_weight=torch.ones(2, device=device), variance_epsilon=0.0
    )
    torch.manual_seed(3)
    cos_sin_cache = torch.randn(8, 2, device=device)
    indexer.rotary_emb = SimpleNamespace(
        cos_sin_cache=cos_sin_cache,
        is_neox_style=True,
        rotary_dim=2,
        mrope_section=None,
    )
    projected = torch.arange(12, dtype=torch.float32, device=device).reshape(2, 6)
    positions = torch.tensor([1, 5], device=device)

    query, raw_key = indexer._project_qk(projected, positions)

    normed = projected[:, :4].reshape(2, 2, 2)
    normed = normed * torch.rsqrt((normed * normed).mean(dim=-1, keepdim=True))
    cos = cos_sin_cache[positions][:, :1].unsqueeze(1)
    sin = cos_sin_cache[positions][:, 1:].unsqueeze(1)
    first, second = normed[..., :1], normed[..., 1:]
    expected = torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    )
    torch.testing.assert_close(query, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(raw_key, projected[:, 4:].reshape(2, 1, 2))


@_requires_cuda
def test_qwen4_exp_qsa_compresses_across_chunks_with_recent_raw_keys() -> None:
    device = "cuda"
    indexer, pool, raw, compressed, rope_cache = _qsa_cache_test_indexer(device)

    first_positions = torch.tensor([0, 1], dtype=torch.long, device=device)
    first_keys = torch.stack((first_positions, first_positions * 2), dim=1).view(
        -1, 1, 2
    )
    indexer._write_and_compress(
        first_keys,
        first_positions,
        first_positions,
        torch.zeros(2, dtype=torch.long, device=device),
        256 + first_positions.to(torch.int32),
        64 + first_positions.to(torch.int32),
        pool,
    )

    second_positions = torch.arange(2, 8, dtype=torch.long, device=device)
    second_keys = torch.stack((second_positions, second_positions * 2), dim=1).view(
        -1, 1, 2
    )
    indexer._write_and_compress(
        second_keys,
        second_positions,
        second_positions,
        torch.zeros(6, dtype=torch.long, device=device),
        256 + second_positions.to(torch.int32),
        64 + second_positions.to(torch.int32),
        pool,
    )

    expected_recent = torch.tensor([[4.0, 8.0], [5.0, 10.0], [6.0, 12.0], [7.0, 14.0]])
    torch.testing.assert_close(raw[1, :, 0].cpu(), expected_recent)
    torch.testing.assert_close(
        compressed[1, 0, 0].cpu(), _qsa_norm(torch.tensor([1.5, 3.0]))
    )
    torch.testing.assert_close(
        compressed[1, 1, 0].cpu(), _qsa_norm(torch.tensor([5.5, 11.0]))
    )
    torch.testing.assert_close(rope_cache[1].cpu(), torch.full((3,), 4))


@_requires_cuda
def test_qwen4_exp_qsa_draft_scratch_spans_compression_boundaries() -> None:
    device = "cuda"
    indexer, pool, raw, compressed, _ = _qsa_cache_test_indexer(device)

    committed_positions = torch.arange(3, dtype=torch.long, device=device)
    committed_keys = torch.stack(
        (
            committed_positions.to(torch.float32) + 1,
            (committed_positions.to(torch.float32) + 1).square(),
        ),
        dim=1,
    ).view(-1, 1, 2)
    indexer._write_and_compress(
        committed_keys,
        committed_positions,
        committed_positions,
        torch.zeros(3, dtype=torch.long, device=device),
        256 + committed_positions.to(torch.int32),
        64 + committed_positions.to(torch.int32),
        pool,
    )
    raw[1, 3, 0] = -99
    seed_position = torch.tensor([3], dtype=torch.long, device=device)
    scratch = indexer._draft_scratch_buffers(
        committed_keys[:1],
        indexer._position_values(seed_position),
        seed_position,
        1,
        reset=True,
    )

    for value in range(3, 8):
        logical = torch.tensor([value], dtype=torch.long, device=device)
        scalar = logical.to(torch.float32) + 1
        token_k = torch.stack((scalar, scalar.square()), dim=1).view(1, 1, 2)
        indexer._write_and_compress(
            token_k,
            logical,
            logical,
            torch.zeros(1, dtype=torch.long, device=device),
            256 + logical.to(torch.int32),
            64 + logical.to(torch.int32),
            pool,
            draft_scratch=scratch,
            stage_draft=True,
        )

    torch.testing.assert_close(
        raw[1, :, 0].cpu(),
        torch.tensor([[1.0, 1.0], [2.0, 4.0], [3.0, 9.0], [-99.0, -99.0]]),
    )
    first_group = torch.tensor([[1.0, 1.0], [2.0, 4.0], [3.0, 9.0], [4.0, 16.0]])
    second_group = torch.tensor([[5.0, 25.0], [6.0, 36.0], [7.0, 49.0], [8.0, 64.0]])
    torch.testing.assert_close(
        compressed[1, 0, 0].cpu(), _qsa_norm(first_group.mean(dim=0))
    )
    torch.testing.assert_close(
        compressed[1, 1, 0].cpu(), _qsa_norm(second_group.mean(dim=0))
    )


@_requires_cuda
def test_qwen4_exp_qsa_draft_mask_blocks_rejected_cache_writes() -> None:
    device = "cuda"
    indexer, pool, raw, compressed, _ = _qsa_cache_test_indexer(device)
    committed_positions = torch.arange(4, dtype=torch.long, device=device)
    committed_keys = (
        torch.stack(
            (committed_positions + 1, (committed_positions + 1).square()), dim=1
        )
        .to(torch.float32)
        .view(-1, 1, 2)
    )
    indexer._write_and_compress(
        committed_keys,
        committed_positions,
        committed_positions,
        torch.zeros(4, dtype=torch.long, device=device),
        256 + committed_positions.to(torch.int32),
        64 + committed_positions.to(torch.int32),
        pool,
    )
    compressed[1, 1, 0] = -77

    candidates = torch.arange(4, 8, dtype=torch.long, device=device)
    candidate_keys = (
        torch.stack((candidates + 1, (candidates + 1).square()), dim=1)
        .to(torch.float32)
        .view(-1, 1, 2)
    )
    indexer._write_and_compress(
        candidate_keys,
        candidates,
        candidates,
        torch.zeros(4, dtype=torch.long, device=device),
        256 + candidates.to(torch.int32),
        64 + candidates.to(torch.int32),
        pool,
        write_mask=torch.tensor([True, False, False, False], device=device),
    )

    torch.testing.assert_close(
        raw[1, :, 0].cpu(),
        torch.tensor([[5.0, 25.0], [2.0, 4.0], [3.0, 9.0], [4.0, 16.0]]),
    )
    torch.testing.assert_close(compressed[1, 1, 0].cpu(), torch.full((2,), -77.0))


@_requires_cuda
def test_qwen4_exp_qsa_does_not_mix_adjacent_requests() -> None:
    device = "cuda"
    indexer, pool, raw, compressed, rope_cache = _qsa_cache_test_indexer(device)
    raw[2, 0, 0] = 100
    raw[2, 1, 0] = 101
    rope_cache[2] = 1000
    logical = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    requests = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    keys = logical.to(torch.float32).view(-1, 1, 1).expand(-1, 1, 2).clone()
    qsa_locs = torch.tensor([256, 257, 514, 515], dtype=torch.int32, device=device)
    recent_locs = torch.tensor([64, 65, 130, 131], dtype=torch.int32, device=device)

    indexer._write_and_compress(
        keys,
        logical,
        logical,
        requests,
        qsa_locs,
        recent_locs,
        pool,
    )

    torch.testing.assert_close(
        compressed[2, 0, 0].cpu(), _qsa_norm(torch.tensor([51.5, 51.5]))
    )


@_requires_cuda
def test_qwen4_exp_qsa_commits_only_accepted_verify_raw_keys() -> None:
    device = "cuda"
    indexer, pool, raw, _, rope_cache = _qsa_cache_test_indexer(device)
    raw[1, :, 0] = torch.tensor(
        [[20.0, 40.0], [21.0, 42.0], [-2.0, -4.0], [-3.0, -6.0]], device=device
    )
    rope_cache[1] = 20
    logical = torch.arange(22, 26, dtype=torch.long, device=device).view(1, 4)
    token_k = torch.stack(
        (logical.to(torch.float32), logical.to(torch.float32) * 2), dim=-1
    ).view(1, 4, 1, 2)
    positions = logical.unsqueeze(-1).expand(-1, -1, 3).clone()
    recent_locs = (64 + logical).to(torch.int32)
    indexer._verify_scratch = {
        (1, 4): (token_k, positions, logical, recent_locs),
    }
    indexer._active_verify_width = 4
    indexer._last_pool = pool

    indexer.commit_verified(torch.tensor([1], dtype=torch.int32, device=device))

    torch.testing.assert_close(
        raw[1, :, 0].cpu(),
        torch.tensor([[20.0, 40.0], [21.0, 42.0], [22.0, 44.0], [-3.0, -6.0]]),
    )
    torch.testing.assert_close(rope_cache[1].cpu(), torch.full((3,), 20))

    indexer.commit_verified(torch.tensor([3], dtype=torch.int32, device=device))

    torch.testing.assert_close(
        raw[1, :, 0].cpu(),
        torch.tensor([[24.0, 48.0], [21.0, 42.0], [22.0, 44.0], [23.0, 46.0]]),
    )
    torch.testing.assert_close(rope_cache[1].cpu(), torch.full((3,), 24))


@_requires_cuda
def test_qwen4_exp_qsa_stage_verified_snapshots_strided_sources() -> None:
    device = "cuda"
    indexer = object.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_head_dim = 4
    indexer._verify_scratch = {}
    indexer._active_verify_width = None
    indexer._last_pool = None

    bs, width = 2, 3
    rows = bs * width
    token_k = torch.randn(rows, 1, indexer.index_head_dim, device=device)
    # 2-D mrope positions produce the transposed ``[rows, 3]`` view the
    # fused staging kernel must gather from.
    positions = torch.arange(3 * rows, device=device, dtype=torch.int64).view(3, rows)
    position_values = indexer._position_values(positions)
    assert not position_values.is_contiguous()
    logical = torch.arange(100, 100 + rows, device=device, dtype=torch.int64)
    recent_locs = torch.arange(7, 7 + rows, device=device, dtype=torch.int32)

    indexer._stage_verified(token_k, position_values, logical, recent_locs, bs, None)

    staged = indexer._verify_scratch[(bs, width)]
    torch.testing.assert_close(
        staged[0], token_k.reshape(bs, width, 1, indexer.index_head_dim)
    )
    torch.testing.assert_close(staged[1], positions.T.reshape(bs, width, 3))
    torch.testing.assert_close(staged[2], logical.reshape(bs, width))
    torch.testing.assert_close(staged[3], recent_locs.reshape(bs, width))
    assert indexer._active_verify_width == width
    assert indexer._last_pool is None


@_requires_cuda
def test_qwen4_exp_qsa_select_tokens_matches_reference() -> None:
    device = "cuda"
    indexer, pool, _, _, _ = _qsa_cache_test_indexer(device)
    # The scoring dot product needs head_dim >= 8, and the streaming block
    # top-k needs a power-of-two block_topk of at least 64.
    indexer.index_head_dim = 16
    indexer.block_topk = 64
    indexer.token_topk = 256
    compressed = torch.zeros((3, 64, 1, 16), dtype=torch.float32, device=device)
    torch.manual_seed(31)
    q = torch.randn(2, 3, 16, device=device)
    logical = torch.tensor([21, 10], dtype=torch.long, device=device)
    requests = torch.tensor([0, 1], dtype=torch.long, device=device)
    qsa_page_table = torch.tensor([[1], [2]], dtype=torch.int32, device=device)

    selected = indexer._select_tokens(q, logical, requests, qsa_page_table, compressed)

    # Only blocks before ``complete_blocks`` hold valid compressed keys, so
    # the selection is deterministic; compare selected token sets per row
    # because the streaming top-k does not preserve score order.
    ratio = indexer.compress_ratio
    assert selected.dtype == torch.int32
    assert selected.shape == (2, indexer.token_topk + ratio - 1)
    complete = (logical + 1) // ratio
    for row in range(2):
        blocks = torch.arange(int(complete[row]), device=device)
        block_tokens = (
            blocks.unsqueeze(-1) * ratio + torch.arange(ratio, device=device)
        ).reshape(-1)
        suffix_values = complete[row] * ratio + torch.arange(ratio - 1, device=device)
        suffix_values = suffix_values[suffix_values <= logical[row]]
        expected = (
            torch.cat((block_tokens, suffix_values)).sort().values.to(torch.int32)
        )
        got = selected[row][selected[row] >= 0].sort().values
        torch.testing.assert_close(got, expected)


def test_qwen4_exp_nextn_head_matches_attention_dp_layout(monkeypatch) -> None:
    calls = []

    def fake_replicated(*args, **kwargs):
        calls.append(("replicated", args, kwargs))
        return object()

    def fake_parallel(*args, **kwargs):
        calls.append(("parallel", args, kwargs))
        return object()

    monkeypatch.setattr(qwen4_exp_nextn, "ReplicatedLinear", fake_replicated)
    monkeypatch.setattr(qwen4_exp_nextn, "ParallelLMHead", fake_parallel)
    config = SimpleNamespace(hidden_size=16, vocab_size=128)
    attn = SimpleNamespace(has_dp=True, tp_rank=0, tp_size=2, tp_group=object())

    qwen4_exp_nextn._build_mtp_lm_head(
        config, SimpleNamespace(attn=attn), None, "draft"
    )
    attn.has_dp = False
    qwen4_exp_nextn._build_mtp_lm_head(
        config, SimpleNamespace(attn=attn), None, "draft"
    )

    assert [kind for kind, _, _ in calls] == ["replicated", "parallel"]
    assert calls[0][1] == (16, 128)
    assert calls[1][1] == (128, 16)


def test_qwen4_exp_nextn_reads_nested_mtp_index_sharing_config() -> None:
    nested = SimpleNamespace(index_share_for_mtp_iteration=True)

    assert qwen4_exp_nextn._mtp_index_sharing_enabled(
        SimpleNamespace(text_config=nested)
    )
    assert qwen4_exp_nextn._mtp_index_sharing_enabled(nested)
    assert not qwen4_exp_nextn._mtp_index_sharing_enabled(
        SimpleNamespace(
            index_share_for_mtp_iteration=True,
            text_config=SimpleNamespace(),
        )
    )


def test_qwen4_exp_ple_verify_workspace_shares_context_rows() -> None:
    context_field = qwen4_exp_ple_context_field(0)
    fields = (
        SimpleNamespace(
            group_id=QWEN4_EXP_PLE_CACHE_GROUP,
            field_id=context_field,
            shape=(2,),
        ),
        SimpleNamespace(
            group_id=QWEN4_EXP_PLE_CACHE_GROUP,
            field_id=qwen4_exp_ple_conv_field(0),
            shape=(16, 9),
        ),
        SimpleNamespace(
            group_id=QWEN4_EXP_PLE_CACHE_GROUP,
            field_id=qwen4_exp_ple_conv_field(2),
            shape=(16, 9),
        ),
    )
    cache_fields = {
        context_field: torch.empty((3, 2), dtype=torch.int64),
        qwen4_exp_ple_conv_field(0): torch.empty((3, 16, 9), dtype=torch.bfloat16),
        qwen4_exp_ple_conv_field(2): torch.empty((3, 16, 9), dtype=torch.bfloat16),
    }
    backend = object.__new__(Qwen4ExpMambaAttnBackend)
    backend.kv_pool = SimpleNamespace(
        arena=SimpleNamespace(
            plan=SimpleNamespace(fields=fields),
            field=cache_fields.__getitem__,
        )
    )
    backend.device = torch.device("cpu")
    backend._ple_verify_scratch = {}

    backend._ensure_ple_verify_scratch(max_bs=2, draft_token_num=3)
    first = backend.ple_verify_scratch(context_field, 0)
    second = backend.ple_verify_scratch(context_field, 2)

    assert first is not None and second is not None
    assert first[0] is second[0]
    assert first[0].shape == (8, 2)
    assert first[0].dtype == torch.int64
    assert first[1].shape == (8, 16, 9)
    assert first[1].dtype == torch.bfloat16
    assert second[1].shape == (8, 16, 9)


def test_qwen4_exp_backend_owns_ple_verify_commit() -> None:
    class PLELayer:
        def __init__(self) -> None:
            self.commits = []

        def commit_verified(self, accepted_lengths, pages) -> None:
            self.commits.append((accepted_lengths, pages))

    backend = object.__new__(Qwen4ExpMambaAttnBackend)
    backend._ple_layers = ()
    layers = (PLELayer(), PLELayer())
    backend.bind_ple_layers(layers)
    backend.bind_ple_layers(layers)
    accepted = torch.tensor([1, 3], dtype=torch.int32)
    pages = torch.tensor([4, 5], dtype=torch.int32)

    backend._commit_aux_verified_state(
        accepted,
        {QWEN4_EXP_PLE_CACHE_GROUP: pages},
    )

    assert [layer.commits for layer in layers] == [
        [(accepted, pages)],
        [(accepted, pages)],
    ]
    with pytest.raises(RuntimeError, match="cannot be rebound"):
        backend.bind_ple_layers((PLELayer(),))


def test_qwen4_exp_selects_model_specific_gdn_backend(monkeypatch) -> None:
    config = SimpleNamespace(
        device=torch.device("cpu"),
        num_attention_heads=2,
        num_kv_heads=1,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        head_dim=8,
        is_draft=False,
        speculative_num_draft_tokens=1,
        replay_ssm=False,
    )
    text_config = SimpleNamespace(
        model_type="qwen4_exp_text",
        full_attention_layer_ids=(1,),
        mamba2_cache_params=(None, None, None, None, (0,)),
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            text_config=text_config,
            architectures=["Qwen4ExpForConditionalGeneration"],
        ),
        attention_arch=object(),
    )
    state_group = SimpleNamespace(
        group_id=LINEAR_ATTENTION,
        family="state",
        checkpoint_granularity=128,
    )
    pool = SimpleNamespace(
        arena=SimpleNamespace(
            runtime_contract=SimpleNamespace(group_specs=(state_group,))
        ),
        state_group_by_layer={0: LINEAR_ATTENTION},
        get_component=lambda layer_id, name: None,
    )
    monkeypatch.setattr(
        attention_registry,
        "_create_attn_backend_with_name",
        lambda *args, **kwargs: SimpleNamespace(device=torch.device("cpu")),
    )

    backend = attention_registry._create_hybrid_linear_attn_backend(
        SimpleNamespace(speculative_algorithm=None, kda_backend="auto"),
        model_config,
        config,
        pool=pool,
    )

    assert isinstance(
        backend.linear_attn_backend,
        Qwen4ExpMambaAttnBackend,
    )


def _ple_layer_stub(hc_count: int = 1, hidden_size: int = 2, pages: int = 5):
    """A CPU-only PLE layer with the GEMMs / embedding stubbed out.

    Returns the layer, the arena fields keyed by name, and a dict the stubbed
    conv records its inputs into.
    """

    class ContextEmbedding(torch.nn.Module):
        eos_token_id = 0

        def forward(self, contexts):
            return contexts.to(torch.float32)

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.layer_id = 0
    layer.context_field_id = qwen4_exp_ple_context_field(0)
    layer.hidden_size = hidden_size
    layer.hc_count = hc_count
    layer.hc_hidden_size = hidden_size * hc_count
    layer.ngram_size = 2
    layer.context_len = 1
    layer.ple_embedding = ContextEmbedding()

    class KVProjection(torch.nn.Module):
        """Stand-in for the fused kv_proj GEMM (hc_hidden + hidden columns)."""

        def forward(self, values):
            kv = values.new_zeros(
                (*values.shape[:-1], layer.hc_hidden_size + layer.hidden_size)
            )
            return kv, None

    layer.kv_proj = KVProjection()
    layer.norm_key = torch.nn.Identity()
    layer.norm_query = torch.nn.Identity()
    layer.norm_conv = torch.nn.Identity()

    recorded = {}

    def conv_sequences(values, initial, lengths, index, *, add_terms=(), **kwargs):
        recorded["add_terms"] = add_terms
        recorded["lengths"] = lengths
        recorded["index"] = index
        output = torch.zeros_like(values)
        for term in add_terms:
            output = output + term
        return output, initial.clone(), initial.new_empty((0, *initial.shape[1:]))

    layer._conv_sequences = conv_sequences
    fields = {
        layer.context_field_id: torch.zeros((pages, 1), dtype=torch.int64),
        qwen4_exp_ple_conv_field(0): torch.zeros(
            (pages, layer.hc_hidden_size, 1), dtype=torch.bfloat16
        ),
    }
    return layer, fields, recorded


def test_qwen4_exp_ple_reads_state_block_metadata() -> None:
    layer, fields, folded = _ple_layer_stub(pages=3)
    context = fields[layer.context_field_id]
    context[1] = 5
    metadata = SimpleNamespace(
        state_in_blocks_by_group={
            QWEN4_EXP_PLE_CACHE_GROUP: torch.tensor([1], dtype=torch.int32)
        },
        state_out_blocks_by_group={
            QWEN4_EXP_PLE_CACHE_GROUP: torch.tensor([2], dtype=torch.int32)
        },
        extend_seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
        mamba_output_indices=None,
    )
    layer._metadata = lambda ctx: metadata
    layer._linear_backend = lambda ctx: SimpleNamespace()
    waited_layers = []
    pool = SimpleNamespace(
        arena=SimpleNamespace(field=fields.__getitem__),
        layerwise_load_tracker=SimpleNamespace(wait_for_layer=waited_layers.append),
    )
    ctx = SimpleNamespace(
        bs=1,
        forward_mode=ForwardMode.DECODE,
        token_to_kv_pool=pool,
    )
    hidden_states = torch.tensor([[1.0, 2.0]])

    output = layer(
        hidden_states,
        torch.tensor([7], dtype=torch.int64),
        ctx,
    )

    assert output.shape == (1, 2)
    assert waited_layers == [0]
    # The layer folds the residual into the conv itself, so it hands the conv
    # the incoming hidden states and returns updated states, not a delta.
    assert folded["add_terms"][1] is hidden_states
    torch.testing.assert_close(output, folded["add_terms"][0] + hidden_states)
    torch.testing.assert_close(context[2], torch.tensor([7], dtype=torch.int64))


def test_qwen4_exp_ple_forward_is_an_eager_break() -> None:
    # The prefill graph captures every bucket with a single dummy request, so
    # the PLE layer must stay eager or its per-request indices, page ids and
    # bs-shaped grids bake in at bs=1.
    assert hasattr(Qwen4ExpPLELayer.forward, "__wrapped__")


def test_qwen4_exp_qsa_entry_points_are_eager_breaks() -> None:
    # The QSA path replaces the attention backend's forward, so it carries the
    # layer's breaks itself: the indexer and the sparse-attention call both
    # address live page tables and per-request layouts that cannot be captured.
    assert hasattr(QSAIndexer.forward, "__wrapped__")
    assert hasattr(QSAIndexer.sparse_attention, "__wrapped__")


def test_qwen4_exp_ple_lengths_accept_a_padded_row_count() -> None:
    layer, _, _ = _ple_layer_stub()
    metadata = SimpleNamespace(
        extend_seq_lens_cpu=torch.tensor([3, 2], dtype=torch.int32)
    )

    # A padded-bucket replay passes the bucket size, not the real token count.
    assert layer._lengths(metadata, 8, 2) == [3, 2]
    assert layer._lengths(metadata, 5, 2) == [3, 2]

    uninferable = SimpleNamespace(extend_seq_lens_cpu=None)
    with pytest.raises(RuntimeError, match="cannot infer per-request token lengths"):
        layer._lengths(uninferable, 8, 3)


def test_qwen4_exp_ple_handles_a_ragged_padded_batch() -> None:
    layer, fields, recorded = _ple_layer_stub()
    metadata = SimpleNamespace(
        state_in_blocks_by_group={
            QWEN4_EXP_PLE_CACHE_GROUP: torch.tensor([1, 2], dtype=torch.int32)
        },
        state_out_blocks_by_group={
            QWEN4_EXP_PLE_CACHE_GROUP: torch.tensor([3, 4], dtype=torch.int32)
        },
        extend_seq_lens_cpu=torch.tensor([3, 2], dtype=torch.int32),
        mamba_output_indices=None,
    )
    layer._metadata = lambda ctx: metadata
    layer._linear_backend = lambda ctx: SimpleNamespace()
    pool = SimpleNamespace(arena=SimpleNamespace(field=fields.__getitem__))
    ctx = SimpleNamespace(
        bs=2,
        forward_mode=ForwardMode.EXTEND,
        token_to_kv_pool=pool,
    )
    # Eight rows for five real tokens: the tail is the padded bucket's filler.
    input_ids = torch.tensor([10, 11, 12, 20, 21, 1, 1, 1], dtype=torch.int64)
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(8, 2)

    output = layer(hidden_states, input_ids, ctx)

    assert output.shape == (5, 2)
    assert recorded["lengths"] == [3, 2]
    # Ragged lengths must take the general path, not the uniform arange one.
    req, _, _, starts, _, total, bs = recorded["index"]
    assert (bs, total) == (2, 5)
    torch.testing.assert_close(req, torch.tensor([0, 0, 0, 1, 1]))
    torch.testing.assert_close(starts, torch.tensor([0, 3]))
    # Each request carries its own last token into its own output page.
    context = fields[layer.context_field_id]
    torch.testing.assert_close(context[3], torch.tensor([12]))
    torch.testing.assert_close(context[4], torch.tensor([21]))


def test_qwen4_exp_cache_recipe_adds_ple_and_qsa_groups() -> None:
    text_config = SimpleNamespace(
        model_type="qwen4_exp_text",
        mamba2_cache_params=(
            (8, 3),
            (2, 4, 4),
            torch.bfloat16,
            torch.float32,
            (0,),
        ),
        ple_layer_ids=[1],
        ngram_context_len=2,
        short_conv_state_shape=(16, 9),
        short_conv_layer_ids=[0],
        indexer_n_heads=4,
        indexer_compress_ratio=4,
        indexer_head_dim=8,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=text_config),
        num_attention_layers=2,
    )
    attn_config = MHAConfig(
        device="cpu",
        backend_name="fa2",
        num_attention_heads=2,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        kv_cache_mxfp8=False,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=8,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=1024,
        max_bs=2,
        prefix_granularity=64,
        kernel_page_size=64,
        kv_cache_quant_method="none",
        max_scheduled_tokens=128,
    )
    server_args = SimpleNamespace(
        block_size=64,
        max_total_tokens=None,
        speculative_num_draft_tokens=0,
    )

    setup = prepare_cache_setup(
        family="qwen4_exp",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 20,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    fields = {field.field_id: field for field in setup.spec.memory_plan.fields}
    groups = {group.group_id: group for group in setup.spec.cache_group_specs}

    assert setup.spec.family == "qwen4_exp"
    context_field = qwen4_exp_ple_context_field(0)
    assert fields[context_field].shape == (2,)
    assert fields[qwen4_exp_ple_conv_field(0)].shape == (16, 9)
    assert fields[qsa_raw_key_field(1)].shape == (4, 1, 8)
    assert fields[qsa_compressed_field(1)].shape == (64, 1, 8)
    assert fields[qsa_rope_position_field(1)].shape == (3,)
    assert (
        fields[qsa_raw_key_field(1)].plane_id
        == fields[qsa_rope_position_field(1)].plane_id
    )
    assert groups[QWEN4_EXP_PLE_CACHE_GROUP].family == "state"
    assert groups[QWEN4_EXP_QSA_CACHE_GROUP].family == "history"
    assert fields[qsa_compressed_field(1)].group_id == QWEN4_EXP_QSA_CACHE_GROUP
    assert fields[qsa_raw_key_field(1)].group_id == QWEN4_EXP_QSA_RECENT_CACHE_GROUP
    assert (
        fields[qsa_rope_position_field(1)].group_id == QWEN4_EXP_QSA_RECENT_CACHE_GROUP
    )
    assert groups[QWEN4_EXP_QSA_CACHE_GROUP].retention == "full_history"
    assert groups[QWEN4_EXP_QSA_CACHE_GROUP].rows_per_page == 64
    assert groups[QWEN4_EXP_QSA_CACHE_GROUP].entry_stride_tokens == 4
    assert groups[QWEN4_EXP_QSA_CACHE_GROUP].block_granularity == 256
    assert groups[QWEN4_EXP_QSA_RECENT_CACHE_GROUP].family == "history"
    assert groups[QWEN4_EXP_QSA_RECENT_CACHE_GROUP].retention == "sliding_window"
    assert groups[QWEN4_EXP_QSA_RECENT_CACHE_GROUP].sliding_window_tokens == 4
    assert groups[QWEN4_EXP_QSA_RECENT_CACHE_GROUP].rows_per_page == 64
    assert groups[QWEN4_EXP_QSA_RECENT_CACHE_GROUP].entry_stride_tokens == 1
    assert groups[QWEN4_EXP_QSA_RECENT_CACHE_GROUP].block_granularity == 64
    assert setup.spec.memory_plan.prefix_granularity == 256
    assert all(
        setup.spec.memory_plan.prefix_granularity % group.block_granularity == 0
        for group in groups.values()
    )
    assert groups[QWEN4_EXP_PLE_CACHE_GROUP].block_granularity == 256
    assert fields[context_field].dtype == "int64"
    selected, consumers = select_layer_fields(
        setup.spec.memory_plan.fields,
        first_layer=0,
        num_layers=2,
    )
    assert selected == frozenset(fields)
    assert context_field in consumers[0]
    assert fields[qsa_raw_key_field(1)].dtype == "bfloat16"
    assert fields[qsa_compressed_field(1)].dtype == "bfloat16"
    assert fields[qsa_rope_position_field(1)].dtype == "int64"

    short_counts = compute_cache_group_page_counts(
        setup.spec.cache_group_specs,
        max_live_requests=2,
        max_scheduled_tokens=128,
        max_total_tokens=1024,
        max_context_len=1024,
    )
    long_counts = compute_cache_group_page_counts(
        setup.spec.cache_group_specs,
        max_live_requests=2,
        max_scheduled_tokens=128,
        max_total_tokens=8192,
        max_context_len=8192,
    )
    assert (
        short_counts[QWEN4_EXP_QSA_RECENT_CACHE_GROUP]
        == long_counts[QWEN4_EXP_QSA_RECENT_CACHE_GROUP]
    )
    assert (
        short_counts[QWEN4_EXP_QSA_CACHE_GROUP] < long_counts[QWEN4_EXP_QSA_CACHE_GROUP]
    )


# ---------------------------------------------------------------------------
# PLE batched-rewrite numerical-equivalence tests
# ---------------------------------------------------------------------------

_PLE_LENGTH_CASES = [
    [1, 1, 1, 1],  # decode
    [3, 1, 5],  # mixed prefill
    [0, 4, 2],  # request with no scheduled tokens
    [7],  # single long request
]

# The per-request reference conv cannot run on empty requests (its conv_input is
# narrower than the dilated receptive field), so the conv comparison skips those
# cases; _batch_indices bound checks still cover them.
_PLE_CONV_LENGTH_CASES = [
    [1, 1, 1, 1],
    [3, 1, 5],
    [7],
    [3, 3, 3],
]


def _ple_stub(ngram_size: int, conv_kernel_size: int, channels: int, eos: int = 0):
    """Bind the batched PLE methods to a lightweight attribute bag so they can
    be exercised without building the full VocabParallelEmbedding stack."""

    conv = torch.nn.Conv1d(
        channels,
        channels,
        conv_kernel_size,
        dilation=ngram_size,
        groups=channels,
        bias=False,
    )
    torch.nn.init.normal_(conv.weight)
    stub = SimpleNamespace(
        ngram_size=ngram_size,
        context_len=ngram_size - 1,
        hc_hidden_size=channels,
        conv_state_len=(conv_kernel_size - 1) * ngram_size,
        conv_kernel_size=conv_kernel_size,
        ple_embedding=SimpleNamespace(eos_token_id=eos),
        conv1d=conv,
    )
    stub._conv_sequences_torch = Qwen4ExpPLELayer._conv_sequences_torch.__get__(stub)
    stub._conv_sequences_cuda = Qwen4ExpPLELayer._conv_sequences_cuda.__get__(stub)
    token_contexts = Qwen4ExpPLELayer._token_contexts.__get__(stub)
    conv_sequences = Qwen4ExpPLELayer._conv_sequences.__get__(stub)
    return stub, token_contexts, conv_sequences


def _ref_token_contexts(input_ids, initial, lengths, ngram_size, context_len):
    contexts = []
    finals = []
    offset = 0
    for request, length in enumerate(lengths):
        tokens = input_ids[offset : offset + length].to(torch.long)
        prefix = initial[request].to(torch.long)
        sequence = torch.cat([prefix, tokens])
        for token_index in range(length):
            contexts.append(sequence[token_index : token_index + ngram_size])
        finals.append(sequence[-context_len:])
        offset += length
    if contexts:
        return torch.stack(contexts), torch.stack(finals)
    return (
        input_ids.new_empty((0, ngram_size), dtype=torch.long),
        initial.clone(),
    )


def _ref_conv_sequences(
    values, initial, lengths, weight, ngram_size, state_len, channels
):
    import torch.nn.functional as F

    outputs = []
    finals = []
    intermediate = []
    offset = 0
    for request, length in enumerate(lengths):
        sequence = values[offset : offset + length].transpose(0, 1).unsqueeze(0)
        conv_input = torch.cat([initial[request : request + 1], sequence], dim=-1)
        conv = (
            F.conv1d(conv_input, weight, dilation=ngram_size, groups=channels)
            .squeeze(0)
            .transpose(0, 1)
        )
        outputs.append(F.silu(conv))
        if state_len:
            windows = (
                conv_input.unfold(2, state_len, 1)[:, :, 1 : length + 1]
                .squeeze(0)
                .permute(1, 0, 2)
                .contiguous()
            )
            intermediate.append(windows)
            finals.append(windows[-1] if length else initial[request])
        else:
            empty = values.new_empty((length, channels, 0))
            intermediate.append(empty)
            finals.append(empty.new_empty((channels, 0)))
        offset += length
    if not outputs:
        return values, initial.clone(), initial.new_empty((0, *initial.shape[1:]))
    return torch.cat(outputs), torch.stack(finals), torch.cat(intermediate)


@pytest.mark.parametrize("lengths", _PLE_LENGTH_CASES)
def test_ple_batch_indices_stay_in_bounds(lengths) -> None:
    device = torch.device("cpu")
    req, col, lengths_t, starts, max_len, total, bs = Qwen4ExpPLELayer._batch_indices(
        lengths, device
    )

    assert bs == len(lengths)
    assert total == sum(lengths)
    assert max_len == (max(lengths) if lengths else 0)
    assert torch.equal(lengths_t, torch.tensor(lengths, dtype=torch.long))
    # Indices must never escape the packed [bs, max_len] grid; a stale index
    # bundle previously let req reach bs and tripped a device-side assert.
    if total:
        assert int(req.min()) >= 0 and int(req.max()) < bs
        assert int(col.min()) >= 0 and int(col.max()) < max_len
    # (req, col) must be a bijection onto the flat token order.
    expected_req = torch.repeat_interleave(
        torch.arange(bs), torch.tensor(lengths, dtype=torch.long)
    )
    assert torch.equal(req, expected_req)
    expected_col = torch.cat(
        [torch.arange(length) for length in lengths] or [torch.empty(0)]
    ).to(torch.long)
    assert torch.equal(col, expected_col)
    # starts rides along in the bundle so no consumer recomputes the scan; it
    # must be the exclusive prefix sum on both the uniform and ragged paths.
    expected_starts = torch.tensor(
        [sum(lengths[:i]) for i in range(bs)], dtype=torch.long
    )
    assert torch.equal(starts, expected_starts)
    if total:
        assert torch.equal(starts[req] + col, torch.arange(total))


def test_ple_batch_indices_uniform_matches_general_path() -> None:
    device = torch.device("cpu")
    lengths = [3, 3, 3]  # uniform -> arange fast path
    fast = Qwen4ExpPLELayer._batch_indices(lengths, device)
    # Same layout expressed non-uniformly so the searchsorted path is taken.
    general = Qwen4ExpPLELayer._batch_indices([3, 3, 3, 0], device)

    assert torch.equal(fast[0], general[0][: fast[5]])
    assert torch.equal(fast[1], general[1][: fast[5]])
    assert torch.equal(fast[3], general[3][: len(lengths)])


@pytest.mark.parametrize("lengths", _PLE_LENGTH_CASES)
def test_ple_token_contexts_matches_reference(lengths) -> None:
    ngram_size = 3
    context_len = ngram_size - 1
    _, token_contexts, _ = _ple_stub(ngram_size, conv_kernel_size=4, channels=4)
    bs = len(lengths)
    total = sum(lengths)
    torch.manual_seed(0)
    input_ids = torch.randint(1, 50, (total,), dtype=torch.long)
    initial = torch.randint(1, 50, (bs, context_len), dtype=torch.long)
    index = Qwen4ExpPLELayer._batch_indices(lengths, input_ids.device)

    contexts, finals = token_contexts(input_ids, initial, lengths, index)
    ref_contexts, ref_finals = _ref_token_contexts(
        input_ids, initial, lengths, ngram_size, context_len
    )

    assert torch.equal(contexts, ref_contexts)
    assert torch.equal(finals, ref_finals)


@pytest.mark.parametrize("lengths", _PLE_CONV_LENGTH_CASES)
def test_ple_conv_sequences_matches_reference(lengths) -> None:
    ngram_size = 3
    conv_kernel_size = 4
    channels = 4
    stub, _, conv_sequences = _ple_stub(ngram_size, conv_kernel_size, channels)
    state_len = stub.conv_state_len
    bs = len(lengths)
    total = sum(lengths)
    torch.manual_seed(1)
    values = torch.randn(total, channels, dtype=torch.float32)
    initial = torch.randn(bs, channels, state_len, dtype=torch.float32)
    weight = stub.conv1d.weight.to(values.dtype)
    index = Qwen4ExpPLELayer._batch_indices(lengths, values.device)

    conv_output, final_conv, intermediate = conv_sequences(
        values, initial, lengths, index
    )
    ref_output, ref_final, ref_intermediate = _ref_conv_sequences(
        values, initial, lengths, weight, ngram_size, state_len, channels
    )

    torch.testing.assert_close(conv_output, ref_output)
    torch.testing.assert_close(final_conv, ref_final)
    torch.testing.assert_close(intermediate, ref_intermediate)


@pytest.mark.parametrize("lengths", _PLE_LENGTH_CASES)
def test_ple_verify_scratch_fill_matches_reference(lengths) -> None:
    ngram_size = 3
    conv_kernel_size = 4
    channels = 4
    context_len = ngram_size - 1
    state_len = (conv_kernel_size - 1) * ngram_size
    bs = len(lengths)
    total = sum(lengths)
    width = max(lengths, default=0)
    torch.manual_seed(2)
    initial_context = torch.randint(1, 50, (bs, context_len), dtype=torch.long)
    initial_conv = torch.randn(bs, channels, state_len, dtype=torch.float32)
    contexts = torch.randint(1, 50, (total, ngram_size), dtype=torch.long)
    intermediate = torch.randn(total, channels, state_len, dtype=torch.float32)
    index = Qwen4ExpPLELayer._batch_indices(lengths, initial_context.device)
    req, col, _, _, _, idx_total, idx_bs = index

    rows = bs * (width + 1)
    ctx_new = torch.zeros(rows, context_len, dtype=torch.long)
    conv_new = torch.zeros(rows, channels, state_len, dtype=torch.float32)
    ctx_ref = torch.zeros_like(ctx_new)
    conv_ref = torch.zeros_like(conv_new)

    # Vectorized fill (mirrors Qwen4ExpPLELayer.forward verify branch).
    stride = width + 1
    init_rows = torch.arange(idx_bs) * stride
    ctx_new[init_rows] = initial_context
    conv_new[init_rows] = initial_conv
    if idx_total:
        token_rows = req * stride + 1 + col
        ctx_new[token_rows] = contexts[:, 1:]
        conv_new[token_rows] = intermediate

    # Reference per-request fill.
    token_start = 0
    for request, length in enumerate(lengths):
        base = request * (width + 1)
        ctx_ref[base] = initial_context[request]
        conv_ref[base] = initial_conv[request]
        token_end = token_start + length
        ctx_ref[base + 1 : base + length + 1] = contexts[token_start:token_end, 1:]
        conv_ref[base + 1 : base + length + 1] = intermediate[token_start:token_end]
        token_start = token_end

    assert torch.equal(ctx_new, ctx_ref)
    torch.testing.assert_close(conv_new, conv_ref)


@_requires_cuda
@pytest.mark.parametrize("group_size", [None, 4])
def test_grouped_gemma_rmsnorm_cuda_matches_reference(group_size) -> None:
    hidden = 12
    norm = GroupedGemmaRMSNorm(hidden, eps=1e-6, group_size=group_size).cuda()
    with torch.no_grad():
        norm.weight.normal_()
    x = torch.randn(5, hidden, device="cuda", dtype=torch.float32)
    effective_group_size = hidden if group_size is None else group_size
    grouped = x.float().unflatten(-1, (-1, effective_group_size))
    expected = (
        grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + 1e-6)
    ).flatten(-2) * (1.0 + norm.weight.float())

    torch.testing.assert_close(norm(x), expected, atol=1e-2, rtol=1e-2)


def _ngram_stub(ngram_size: int, heads_per_ngram: int = 4, eos: int = 7):
    """Attribute bag exposing the hash-id methods without building the full
    VocabParallelEmbedding stack."""

    cls = Qwen4ExpNGramEmbedding
    stub = SimpleNamespace(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_heads=(ngram_size - 1) * heads_per_ngram,
        ple_layer_index=2,
        unigram_vocab_size=50_000,
        eos_token_id=eos,
        _PRIME_1=cls._PRIME_1,
        _MASK64=cls._MASK64,
        _SPLITMIX_GAMMA=cls._SPLITMIX_GAMMA,
        _SPLITMIX_M1=cls._SPLITMIX_M1,
        _SPLITMIX_M2=cls._SPLITMIX_M2,
        _splitmix64=cls._splitmix64,
    )
    stub.layer_multipliers = cls._build_layer_multipliers.__get__(stub)(
        ngram_size, 1234
    )
    sizes = [
        _nth_prime_after(19_999, 2 * stub.ngram_heads + i + 1)
        for i in range(stub.ngram_heads)
    ]
    offsets, running = [], 0
    for size in sizes:
        offsets.append(running)
        running += size
    stub.ngram_heads_vocab_sizes = torch.tensor(sizes, dtype=torch.long)
    stub.ngram_heads_offsets = torch.tensor(offsets, dtype=torch.long)
    return stub


def _legacy_ngram_ids(stub, contexts: torch.Tensor) -> torch.Tensor:
    """Reference: the original full-window shift/hash algorithm."""

    eos = stub.eos_token_id

    def shift_right(values, shift):
        if shift == 0:
            return values
        bsz, seq = values.shape
        idx = torch.arange(seq, dtype=torch.long)
        eos_pos = torch.where(values == eos, idx, torch.tensor(-1))
        prev = torch.cat(
            [
                torch.full((bsz, 1), -1, dtype=torch.long),
                torch.cummax(eos_pos, dim=1).values[:, :-1],
            ],
            dim=1,
        )
        src = idx - shift
        gathered = values.gather(1, src.clamp_min(0).unsqueeze(0).expand(bsz, -1))
        valid = (idx.unsqueeze(0) - prev - 1 >= shift) & (src.unsqueeze(0) >= 0)
        return torch.where(valid, gathered, values.new_full((), eos))

    shifted = [shift_right(contexts, s) for s in range(stub.ngram_size)]
    rows = torch.arange(contexts.shape[0])
    column = torch.full_like(rows, contexts.shape[1] - 1)
    blocks = []
    for gram in range(2, stub.ngram_size + 1):
        head_start = (gram - 2) * stub.heads_per_ngram
        head_end = head_start + stub.heads_per_ngram
        mixed = shifted[0] * stub.layer_multipliers[0]
        for position in range(1, gram):
            mixed = torch.bitwise_xor(
                mixed, shifted[position] * stub.layer_multipliers[position]
            )
        ids = torch.remainder(
            mixed.unsqueeze(-1),
            stub.ngram_heads_vocab_sizes[head_start:head_end].view(1, 1, -1),
        ) + stub.ngram_heads_offsets[head_start:head_end].view(1, 1, -1)
        blocks.append(ids[rows, column])
    return torch.cat(blocks, dim=-1)


@pytest.mark.parametrize("ngram_size", [2, 3, 4])
def test_ngram_ids_anchor_rewrite_matches_legacy(ngram_size) -> None:
    stub = _ngram_stub(ngram_size)
    torch.manual_seed(0)
    total = 512
    contexts = torch.randint(1, 50_000, (total, ngram_size), dtype=torch.long)
    contexts[torch.rand(total, ngram_size) < 0.25] = stub.eos_token_id

    got = Qwen4ExpNGramEmbedding._ngram_ids_torch.__get__(stub)(contexts)

    assert torch.equal(got, _legacy_ngram_ids(stub, contexts))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton n-gram kernel requires CUDA"
)
@pytest.mark.parametrize("ngram_size", [2, 3, 4])
@pytest.mark.parametrize("lengths", [[1, 1, 1, 1], [3, 1, 5], [0, 4, 2], [0, 0]])
def test_ngram_ids_flat_kernel_matches_legacy(ngram_size, lengths) -> None:
    stub = _ngram_stub(ngram_size)
    context_len = ngram_size - 1
    bs, total = len(lengths), sum(lengths)
    torch.manual_seed(1)
    flat_ids = torch.randint(1, 50_000, (total,), dtype=torch.long)
    flat_ids[torch.rand(total) < 0.25] = stub.eos_token_id
    initial = torch.randint(1, 50_000, (bs, context_len), dtype=torch.long)
    initial[torch.rand(bs, context_len) < 0.25] = stub.eos_token_id

    # Ground-truth window matrix: per request [prefix | tokens] slices.
    contexts, offset = [], 0
    for request, length in enumerate(lengths):
        seq = torch.cat([initial[request], flat_ids[offset : offset + length]])
        for k in range(length):
            contexts.append(seq[k : k + ngram_size])
        offset += length
    contexts = (
        torch.stack(contexts)
        if contexts
        else torch.empty((0, ngram_size), dtype=torch.long)
    )
    reference = _legacy_ngram_ids(stub, contexts)

    # The bundle's starts feed the hash kernel's addressing, so a wrong scan
    # here would show up as mismatched ids rather than passing silently.
    req, col, _, starts, _, tot, _ = Qwen4ExpPLELayer._batch_indices(
        lengths, torch.device("cuda")
    )
    stub.layer_multipliers = stub.layer_multipliers.cuda()
    stub.ngram_heads_vocab_sizes = stub.ngram_heads_vocab_sizes.cuda()
    stub.ngram_heads_offsets = stub.ngram_heads_offsets.cuda()
    flat = Qwen4ExpNGramEmbedding._ngram_ids_flat_cuda.__get__(stub)

    ids, tail = flat(flat_ids.cuda(), initial.cuda(), req, col, starts, need_tail=True)
    assert torch.equal(ids.cpu(), reference)
    assert torch.equal(tail.cpu(), contexts[:, 1:])

    ids_only, no_tail = flat(
        flat_ids.cuda(), initial.cuda(), req, col, starts, need_tail=False
    )
    assert torch.equal(ids_only.cpu(), reference)
    assert no_tail is None

    stride = max(lengths, default=0) + 1
    scratch = torch.full(
        ((bs + 1) * stride, context_len), -1, dtype=torch.long, device="cuda"
    )
    direct_ids, direct_tail = flat(
        flat_ids.cuda(),
        initial.cuda(),
        req,
        col,
        starts,
        need_tail=False,
        tail_out=scratch,
        tail_block_rows=stride,
    )
    assert torch.equal(direct_ids.cpu(), reference)
    assert direct_tail is None
    initial_rows = torch.arange(bs, device="cuda") * stride
    assert torch.equal(scratch[initial_rows].cpu(), initial)
    token_rows = req * stride + 1 + col
    assert torch.equal(scratch[token_rows].cpu(), contexts[:, 1:])
    untouched = torch.ones(scratch.shape[0], dtype=torch.bool, device="cuda")
    untouched[initial_rows] = False
    untouched[token_rows] = False
    assert torch.all(scratch[untouched] == -1)


@pytest.mark.parametrize("lengths", _PLE_LENGTH_CASES)
def test_ple_final_context_matches_token_contexts(lengths) -> None:
    ngram_size = 3
    context_len = ngram_size - 1
    stub, token_contexts, _ = _ple_stub(ngram_size, conv_kernel_size=4, channels=4)
    stub.context_len = context_len
    bs, total = len(lengths), sum(lengths)
    torch.manual_seed(4)
    flat_ids = torch.randint(1, 50, (total,), dtype=torch.long)
    initial = torch.randint(1, 50, (bs, context_len), dtype=torch.long)
    index = Qwen4ExpPLELayer._batch_indices(lengths, flat_ids.device)
    _, _, lengths_t, starts, _, _, _ = index

    _, ref_final = token_contexts(flat_ids, initial, lengths, index)
    got_final = Qwen4ExpPLELayer._final_context.__get__(stub)(
        flat_ids, initial, lengths_t, starts
    )

    assert torch.equal(got_final, ref_final)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused PLE conv requires CUDA"
)
@pytest.mark.parametrize("lengths", [[1, 1, 1, 1], [3, 1, 5], [0, 4, 2], [7]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ple_conv_fused_matches_torch(lengths, dtype) -> None:
    stub, _, _ = _ple_stub(ngram_size=3, conv_kernel_size=4, channels=8)
    stub.conv1d = stub.conv1d.to("cuda", dtype)
    state_len = stub.conv_state_len
    bs, total = len(lengths), sum(lengths)
    torch.manual_seed(3)
    values = torch.randn(total, 8, dtype=dtype, device="cuda")
    initial = torch.randn(bs, 8, state_len, dtype=dtype, device="cuda")
    req, col, lengths_t, _, max_len, tot, _ = Qwen4ExpPLELayer._batch_indices(
        lengths, torch.device("cuda")
    )

    ref = stub._conv_sequences_torch(
        values, initial, req, col, lengths_t, max_len, tot, bs
    )
    got = stub._conv_sequences_cuda(values, initial, req, col, lengths_t, tot, bs, True)

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(got[0], ref[0], rtol=tol, atol=tol)
    assert torch.equal(got[1], ref[1])  # final windows are pure copies
    assert torch.equal(got[2], ref[2])  # intermediate windows likewise

    # need_intermediate=False keeps output/final and skips the big windows
    # materialization entirely.
    skipped = stub._conv_sequences_cuda(
        values, initial, req, col, lengths_t, tot, bs, False
    )
    torch.testing.assert_close(skipped[0], ref[0], rtol=tol, atol=tol)
    assert torch.equal(skipped[1], ref[1])
    assert skipped[2].shape[0] == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused PLE conv requires CUDA"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ple_conv_epilogue_folds_full_width_adds(dtype) -> None:
    lengths = [3, 1, 5]
    channels = 8
    stub, _, _ = _ple_stub(ngram_size=3, conv_kernel_size=4, channels=channels)
    stub.conv1d = stub.conv1d.to("cuda", dtype)
    state_len = stub.conv_state_len
    bs, total = len(lengths), sum(lengths)
    torch.manual_seed(4)
    values = torch.randn(total, channels, dtype=dtype, device="cuda")
    initial = torch.randn(bs, channels, state_len, dtype=dtype, device="cuda")
    gated = torch.randn(total, channels, dtype=dtype, device="cuda")
    # A row slice of a wider buffer: strided rows must feed the kernel directly.
    residual = torch.randn(total, 2 * channels, dtype=dtype, device="cuda")[
        :, :channels
    ]
    req, col, lengths_t, _, _, tot, _ = Qwen4ExpPLELayer._batch_indices(
        lengths, torch.device("cuda")
    )

    plain = stub._conv_sequences_cuda(
        values, initial, req, col, lengths_t, tot, bs, True
    )
    fused = stub._conv_sequences_cuda(
        values,
        initial,
        req,
        col,
        lengths_t,
        tot,
        bs,
        True,
        add_terms=(gated, residual),
    )

    # The epilogue stands in for separate tensor adds. Rounding each fold to a
    # store dtype narrower than the fp32 accumulator reproduces them exactly; a
    # pure fp32 stream has no such barrier after the SiLU, so it keeps one more
    # bit of the product and may land an ulp away (the more accurate way).
    def matches(got, want):
        if dtype == torch.float32:
            torch.testing.assert_close(got, want)
        else:
            assert torch.equal(got, want)

    # Folding addends must leave the carried state outputs untouched.
    assert torch.equal(fused[1], plain[1])
    assert torch.equal(fused[2], plain[2])
    matches(fused[0], residual + (gated + plain[0]))

    # One addend folds too, and the unused slot stays unread.
    single = stub._conv_sequences_cuda(
        values, initial, req, col, lengths_t, tot, bs, False, add_terms=(gated,)
    )
    matches(single[0], gated + plain[0])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused PLE conv requires CUDA"
)
@pytest.mark.parametrize("lengths", [[3, 1, 5], [0, 4, 2], [0, 0]])
def test_ple_conv_scatters_windows_into_verify_scratch(lengths) -> None:
    channels = 8
    dtype = torch.bfloat16
    stub, _, _ = _ple_stub(ngram_size=3, conv_kernel_size=4, channels=channels)
    stub.conv1d = stub.conv1d.to("cuda", dtype)
    state_len = stub.conv_state_len
    bs, total = len(lengths), sum(lengths)
    torch.manual_seed(5)
    values = torch.randn(total, channels, dtype=dtype, device="cuda")
    initial = torch.randn(bs, channels, state_len, dtype=dtype, device="cuda")
    req, col, lengths_t, _, _, tot, _ = Qwen4ExpPLELayer._batch_indices(
        lengths, torch.device("cuda")
    )

    packed = stub._conv_sequences_cuda(
        values, initial, req, col, lengths_t, tot, bs, True
    )
    # Scratch is one (width + 1) row block per request, plus a spare block to
    # prove the kernel stays inside the rows it owns.
    stride = max(lengths) + 1
    scratch = torch.zeros(
        (bs + 1) * stride, channels, state_len, dtype=dtype, device="cuda"
    )
    scattered = stub._conv_sequences_cuda(
        values,
        initial,
        req,
        col,
        lengths_t,
        tot,
        bs,
        True,
        windows_out=scratch,
        windows_block_rows=stride,
    )

    # Same conv results, with carried and token windows landing in their
    # rollback rows and nothing else in the scratch disturbed.
    assert torch.equal(scattered[0], packed[0])
    assert torch.equal(scattered[1], packed[1])
    assert scattered[2] is scratch
    initial_rows = torch.arange(bs, device="cuda") * stride
    assert torch.equal(scratch[initial_rows], initial)
    token_rows = req * stride + 1 + col
    assert torch.equal(scratch[token_rows], packed[2])
    untouched = torch.ones(scratch.shape[0], dtype=torch.bool, device="cuda")
    untouched[initial_rows] = False
    untouched[token_rows] = False
    assert not scratch[untouched].any()

    # A scratch the kernel cannot address by row must be refused, not repacked
    # into a copy whose writes would be dropped.
    with pytest.raises(ValueError):
        stub._conv_sequences_cuda(
            values,
            initial,
            req,
            col,
            lengths_t,
            tot,
            bs,
            False,
            windows_out=scratch,
            windows_block_rows=stride,
        )
    with pytest.raises(ValueError):
        stub._conv_sequences_cuda(
            values,
            initial,
            req,
            col,
            lengths_t,
            tot,
            bs,
            True,
            windows_out=scratch.transpose(1, 2),
            windows_block_rows=stride,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused PLE conv requires CUDA"
)
def test_ple_conv_accepts_precomputed_starts() -> None:
    """A handed-down scan must address exactly what the internal one did."""
    lengths = [3, 1, 5]  # ragged, so starts is not a plain multiple of anything
    channels = 8
    dtype = torch.bfloat16
    stub, _, _ = _ple_stub(ngram_size=3, conv_kernel_size=4, channels=channels)
    stub.conv1d = stub.conv1d.to("cuda", dtype)
    bs, total = len(lengths), sum(lengths)
    torch.manual_seed(7)
    values = torch.randn(total, channels, dtype=dtype, device="cuda")
    initial = torch.randn(bs, channels, stub.conv_state_len, dtype=dtype, device="cuda")
    req, col, lengths_t, starts, _, tot, _ = Qwen4ExpPLELayer._batch_indices(
        lengths, torch.device("cuda")
    )

    rescanned = stub._conv_sequences_cuda(
        values, initial, req, col, lengths_t, tot, bs, True
    )
    reused = stub._conv_sequences_cuda(
        values, initial, req, col, lengths_t, tot, bs, True, starts=starts
    )

    for handed, internal in zip(reused, rescanned):
        assert torch.equal(handed, internal)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused PLE page access requires CUDA"
)
def test_ple_page_access_skips_null_pages_and_page_padding() -> None:
    pages, channels, state = 6, 4, 3
    row_numel = channels * state
    # The plan may pad a page past the row it holds, so the field view is
    # strided rather than contiguous; a kernel assuming dense pages would read
    # and write the wrong rows here.
    pad = 7
    torch.manual_seed(6)
    base = torch.randn(pages * (row_numel + pad), dtype=torch.bfloat16, device="cuda")
    field = base.as_strided((pages, channels, state), (row_numel + pad, state, 1))
    page_ids = torch.tensor([2, 0, 4], dtype=torch.int32, device="cuda")

    read = Qwen4ExpPLELayer._read_pages(field, page_ids, 1.5)

    # Page id 0 is the null page and reads as the default instead.
    assert torch.equal(read[0], field[2])
    assert torch.equal(read[2], field[4])
    assert torch.equal(read[1], torch.full_like(read[1], 1.5))

    values = torch.randn(3, channels, state, dtype=torch.bfloat16, device="cuda")
    before = base.clone()
    Qwen4ExpPLELayer._write_pages(field, page_ids, values)

    assert torch.equal(field[2], values[0])
    assert torch.equal(field[4], values[2])
    # The null page's row, the untargeted pages and every pad element are left
    # exactly as they were.
    touched = torch.zeros_like(base, dtype=torch.bool)
    for page in (2, 4):
        start = page * (row_numel + pad)
        touched[start : start + row_numel] = True
    assert torch.equal(base[~touched], before[~touched])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused PLE page access requires CUDA"
)
def test_ple_page_read_keeps_int64_default_exact() -> None:
    context = torch.zeros(4, 2, dtype=torch.int64, device="cuda")
    context[3] = torch.tensor([11, 12], device="cuda")
    page_ids = torch.tensor([3, 0], dtype=torch.int32, device="cuda")
    # Past fp32's exactly-representable range: the fill must stay integral.
    eos = 2**33 + 1

    read = Qwen4ExpPLELayer._read_pages(context, page_ids, eos)

    assert read.dtype == torch.int64
    assert torch.equal(read[0], context[3])
    assert torch.equal(read[1], torch.full_like(read[1], eos))


def test_ple_kv_proj_shard_loader_routes_rows() -> None:
    stub = SimpleNamespace(hc_hidden_size=8, hidden_size=4)
    load = Qwen4ExpPLELayer._load_kv_proj_shard.__get__(stub)
    param = torch.zeros(12, 6)
    key_w = torch.randn(8, 6)
    value_w = torch.randn(4, 6)

    load(param, key_w, "key")
    load(param, value_w, "value")

    assert torch.equal(param[:8], key_w)
    assert torch.equal(param[8:], value_w)
    with pytest.raises(ValueError):
        load(param, torch.randn(5, 6), "key")


def _gate_norm_stub(hc_count: int, hidden_size: int, dtype, device):
    """PLE-layer attribute bag with real grouped norms for the gating chain."""

    stub = SimpleNamespace(
        hc_count=hc_count,
        hidden_size=hidden_size,
        hc_hidden_size=hc_count * hidden_size,
    )
    torch.manual_seed(42)
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = GroupedGemmaRMSNorm(
            hc_count * hidden_size, 1e-6, group_size=hidden_size
        ).to(device)
        with torch.no_grad():
            norm.weight.normal_(std=0.5)
        norm.gemma_weight = (norm.weight.data + 1.0).to(dtype)
        norm.weight.data = norm.weight.data.to(dtype)
        setattr(stub, name, norm)
    return stub


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused PLE gating requires CUDA"
)
@pytest.mark.parametrize(
    "dtype,hc_count,hidden_size,total",
    [
        (torch.float32, 4, 512, 333),
        (torch.bfloat16, 4, 2048, 257),
        (torch.bfloat16, 2, 1024, 1),
    ],
)
def test_ple_gate_norm_fused_matches_unfused(
    dtype, hc_count, hidden_size, total
) -> None:
    stub = _gate_norm_stub(hc_count, hidden_size, dtype, "cuda")
    torch.manual_seed(0)
    # Build key/value as kv_proj-style split views so the kernel's strided
    # addressing is exercised.
    kv = torch.randn(
        total, hc_count * hidden_size + hidden_size, dtype=dtype, device="cuda"
    )
    key, value = kv.split([hc_count * hidden_size, hidden_size], dim=-1)
    query = torch.randn(total, hc_count * hidden_size, dtype=dtype, device="cuda")

    ref_gated, ref_norm = Qwen4ExpPLELayer._gate_and_norm_torch.__get__(stub)(
        key, query, value
    )
    got_gated, got_norm = Qwen4ExpPLELayer._gate_and_norm_cuda.__get__(stub)(
        key, query, value
    )

    tol = 2e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(got_gated, ref_gated, rtol=tol, atol=tol)
    torch.testing.assert_close(got_norm, ref_norm, rtol=tol, atol=tol)

    empty_gated, empty_norm = Qwen4ExpPLELayer._gate_and_norm_cuda.__get__(stub)(
        key[:0], query[:0], value[:0]
    )
    assert empty_gated.shape == (0, hc_count * hidden_size)
    assert empty_norm.shape == (0, hc_count * hidden_size)


def test_ple_fp8_quantize_roundtrip() -> None:
    torch.manual_seed(0)
    # Rows spanning several orders of magnitude: per-row scales must adapt.
    rows = (
        torch.randn(1024, 64, dtype=torch.bfloat16)
        * torch.logspace(-3, 1, 1024).unsqueeze(1).bfloat16()
    )

    quantized, scale = quantize_ple_embedding_rows(rows)

    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    dequant = quantized.to(torch.float32) * scale.unsqueeze(1)
    reference = rows.to(torch.float32)
    relative = (dequant - reference).abs() / reference.abs().clamp_min(1e-6)
    assert relative.median() < 0.05  # e4m3 per-row quantization bound

    # All-zero rows must produce finite scales and exact-zero dequant.
    zero_q, zero_s = quantize_ple_embedding_rows(
        torch.zeros(3, 16, dtype=torch.bfloat16)
    )
    assert torch.isfinite(zero_s).all()
    assert (zero_q.to(torch.float32) * zero_s.unsqueeze(1) == 0).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 embedding gather requires CUDA"
)
def test_ple_fp8_dequant_gather_matches_bf16() -> None:
    total_rows, head_dim, tokens, heads = 4096, 64, 256, 8
    torch.manual_seed(1)
    table = torch.randn(total_rows, head_dim, dtype=torch.bfloat16, device="cuda")
    quantized, scale = quantize_ple_embedding_rows(table)
    ids = torch.randint(0, total_rows, (tokens, heads), device="cuda")

    stub = SimpleNamespace(
        ngram_embedding=SimpleNamespace(tp_size=1, num_embeddings_padded=total_rows),
        ngram_embedding_scale=scale,
        embed_store_dtype=torch.float8_e4m3fn,
        embed_output_dtype=torch.bfloat16,
    )
    raw = torch.nn.functional.embedding(ids, quantized)
    dequant = Qwen4ExpNGramEmbedding._dequant.__get__(stub)(raw, ids)
    reference = torch.nn.functional.embedding(ids, table)

    assert dequant.dtype == torch.bfloat16
    relative = (
        dequant.float() - reference.float()
    ).abs() / reference.float().abs().clamp_min(1e-3)
    assert relative.median() < 0.07


def _ple_checkpoint_loader_stub(
    store_dtype: torch.dtype,
) -> tuple[torch.nn.Module, Qwen4ExpNGramEmbedding]:
    root = torch.nn.Module()
    ple = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(ple)
    ple.embed_store_dtype = (
        torch.float8_e4m3fn if store_dtype == torch.float8_e4m3fn else None
    )
    ple._checkpoint_weight_scale = 1.0

    embedding = torch.nn.Module()
    embedding.org_vocab_size = 8
    embedding.shard_indices = SimpleNamespace(
        org_vocab_start_index=0,
        org_vocab_end_index=8,
    )
    embedding.register_parameter(
        "weight",
        torch.nn.Parameter(torch.empty(8, 4, dtype=store_dtype), requires_grad=False),
    )
    ple.ngram_embedding = embedding
    if ple.embed_store_dtype is not None:
        ple.register_buffer("ngram_embedding_scale", torch.ones(8))
    root.ple = ple
    return root, ple


@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize("store_dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_ple_prequantized_checkpoint_scale_is_applied(
    scale_first: bool,
    store_dtype: torch.dtype,
) -> None:
    root, ple = _ple_checkpoint_loader_stub(store_dtype)
    source = torch.tensor(
        [
            [-48.0, 72.0, -80.0, 64.0],
            [10.0, 20.0, 144.0, -88.0],
            [1.0, -2.0, 3.0, -4.0],
            [32.0, 40.0, -56.0, 8.0],
            [0.5, -0.5, 0.25, -0.25],
            [96.0, -112.0, 128.0, -144.0],
            [6.0, 7.0, 8.0, 9.0],
            [-16.0, -24.0, 48.0, 80.0],
        ],
        dtype=torch.float8_e4m3fn,
    )
    checkpoint_scale = torch.tensor([2.0e-4], dtype=torch.bfloat16)
    shard = ("ple.ngram_embedding.shard_0.weight", source)
    scale_weight = ("ple.ngram_embedding.weight_scale", checkpoint_scale)
    weights = [scale_weight, shard] if scale_first else [shard, scale_weight]

    loaded = load_qwen4_exp_weights(
        root,
        SimpleNamespace(num_experts=None, split_ngram_parts=1),
        SimpleNamespace(),
        weights,
        include_visual=False,
    )

    expected_scale = checkpoint_scale.float().item()
    restored = ple.ngram_embedding.weight.float()
    if store_dtype == torch.float8_e4m3fn:
        assert torch.equal(ple.ngram_embedding.weight, source)
        torch.testing.assert_close(
            ple.ngram_embedding_scale,
            torch.full((8,), expected_scale),
        )
        restored = restored * ple.ngram_embedding_scale.unsqueeze(1)
    torch.testing.assert_close(
        restored,
        source.float() * expected_scale,
        rtol=1e-2,
        atol=1e-5,
    )
    assert loaded == {"ple.ngram_embedding.weight"}


def test_should_exclude_quant_module_expands_fused_members() -> None:
    from tokenspeed.runtime.layers.quantization.utils import (
        should_exclude_quant_module,
    )

    ple = "model.language_model.layers.1.ple"
    both_members = [f"{ple}.key_proj", f"{ple}.value_proj"]

    # Both member projections excluded -> the fused kv_proj is excluded.
    assert should_exclude_quant_module(f"{ple}.kv_proj", both_members)
    # A single excluded member leaves the fused module quantized.
    assert not should_exclude_quant_module(f"{ple}.kv_proj", [f"{ple}.key_proj"])
    # An explicit fused-name entry still matches directly.
    assert should_exclude_quant_module(f"{ple}.kv_proj", [f"{ple}.kv_proj"])
    # Unrelated sibling modules stay quantized.
    assert not should_exclude_quant_module(f"{ple}.conv1d", both_members)

    # The gate_up_proj member expansion keeps its behavior.
    mlp = "model.language_model.layers.0.mlp"
    assert should_exclude_quant_module(
        f"{mlp}.gate_up_proj", [f"{mlp}.gate_proj", f"{mlp}.up_proj"]
    )
    assert not should_exclude_quant_module(f"{mlp}.gate_up_proj", [f"{mlp}.up_proj"])
