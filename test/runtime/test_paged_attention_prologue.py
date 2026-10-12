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

"""A layer states its prologue steps once; PagedAttention hands them to the
kernel entry."""

import ast
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed_kernel.ops.attention.prologue import (  # noqa: E402
    HeadKVCache,
    LatentKVCache,
    MRope,
    RopeStyle,
)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.layers import paged_attention  # noqa: E402
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm, RMSNorm  # noqa: E402
from tokenspeed.runtime.layers.rotary_embedding import (  # noqa: E402
    MRotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
)

# Draft models inject target-context rows into their own cache outside any attention forward.
_CONTEXT_KV_WRITERS = {
    "models/dflash.py:DFlashAttention.apply_k_norm",
    "models/dflash.py:DFlashDraftModel.write_context_kv",
    "models/dflash2.py:DFlash2DraftModel.write_context_kv",
    "models/kimi_k3_dspark.py:K3DSparkAttention.apply_latent_rope",
    "models/kimi_k3_dspark.py:K3DSparkModel.write_context_kv",
    "execution/drafter/dflash.py:DFlash._write_native_cache_fused",
    "execution/drafter/dflash.py:DFlash._write_native_cache_fused_mla",
    "models/deepseek_v41_dspark.py:DeepseekV41DSparkModel._main_kv",
}
# Keys the prologue does not own: sparse indexers' and DeepSeek-V4's own attention.
_OTHER_KEY_OWNERS = {
    "models/deepseek_v4.py:DeepseekV4Attention._project_q_kv",
    "models/glm5.py:GlmDsaIndexer.forward",
    "models/glm53_flash.py:Glm53FlashIndexer.forward",
}
# The norm, RoPE, quantize and KV-write steps the prologue owns: modules and kernel entries.
_PROLOGUE_STEPS = {
    "apply_k_rope",
    "apply_rope",
    "apply_rope_mla",
    "fp8_quantize",
    "fused_fp8_set_kv_buffer",
    "k_norm",
    "mla_latent_norm_rope_scatter",
    "q_norm",
    "qk_rmsnorm",
    "quantize_store_kv_mxfp8",
    "rotary_emb",
    "set_kv_buffer",
    "set_mla_kv_buffer",
    "set_mla_kv_buffer_triton",
    "store_kv_cache",
    "store_latent_per_token_head",
}


def _prologue(monkeypatch, *, qk_norm, mode, rows, slots):
    """Run ``PagedAttention.prologue`` and return what it handed the kernel entry."""
    handed = {}
    monkeypatch.setattr(
        paged_attention,
        "gqa_prologue",
        lambda q, k, v, **kw: handed.update(kw, q=q, k=k, v=v),
    )
    layer = paged_attention.PagedAttention(
        4, 64, 1.0, num_kv_heads=2, layer_id=0, rotary_emb=None, qk_norm=qk_norm
    )
    cache = torch.zeros(8, 2, 64, dtype=torch.bfloat16)
    ctx = SimpleNamespace(
        forward_mode=mode,
        attn_backend=SimpleNamespace(
            padded_write_locations=lambda layer, m, rows: handed.update(rows=rows)
            or slots,
            cache_placement=lambda layer: None,
        ),
        token_to_kv_pool=SimpleNamespace(
            kv_write_target=lambda layer_id, s, m: HeadKVCache(cache, cache, None, s)
        ),
    )
    kv = torch.zeros(rows, 2 * 64, dtype=torch.bfloat16)
    layer.prologue(
        torch.zeros(rows, 4 * 64, dtype=torch.bfloat16), kv, kv, torch.arange(rows), ctx
    )
    return handed


def test_the_prefill_prologue_runs_over_every_row_before_the_break(monkeypatch):
    """Outside a decode round the model runs the expanded prologue in the
    captured segment over the padded rows; a decode round and a narrowed draft
    step keep theirs in the break."""
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA

    handed = {}
    monkeypatch.setattr(
        paged_attention,
        "mla_prologue",
        lambda query, q_pe, latent, **kw: handed.update(kw, query=query) or "out",
    )
    layer = paged_attention.PagedAttention(
        2,
        192,
        1.0,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=128,
        rotary_emb=None,
        qk_norm=None,
    )
    cache = torch.zeros(8, 1, 576, dtype=torch.bfloat16)
    model = SimpleNamespace(
        forward_normal_chunked_kv_prepare=lambda *args: (
            DeepseekV3AttentionMLA.forward_normal_chunked_kv_prepare(model, *args)
        ),
        attn_mha=layer,
        num_local_heads=2,
        qk_head_dim=192,
        qk_nope_head_dim=128,
        v_head_dim=128,
        kv_lora_rank=512,
        kv_b_proj=lambda latent: (latent.new_zeros((latent.shape[0], 2 * 256)),),
    )

    def ctx(mode, narrowing=None):
        return SimpleNamespace(
            forward_mode=mode,
            draft_narrowing=narrowing,
            query_shard=None,
            attn_backend=SimpleNamespace(
                padded_write_locations=lambda layer, m, rows: torch.tensor(
                    [5, 6, 0, 0]
                )[:rows],
                cache_placement=lambda layer: None,
            ),
            token_to_kv_pool=SimpleNamespace(
                kv_write_target=lambda layer_id, s, m: LatentKVCache(cache, False, s, m)
            ),
        )

    q = torch.zeros(4, 2 * 192, dtype=torch.bfloat16)
    latent = torch.zeros(4, 576, dtype=torch.bfloat16)
    run = DeepseekV3AttentionMLA._prefill_prologue_before_break
    assert run(model, torch.arange(4), q, latent, ctx(ForwardMode.EXTEND)) == "out"
    assert handed["cache"].slots.tolist() == [5, 6, 0, 0]
    assert handed["cache"].write_mask is None
    assert handed["expanded"].k_nope.shape == (4, 2, 128)
    assert handed["expanded"].value.shape == (4, 2, 128)
    assert run(model, torch.arange(4), q, latent, ctx(ForwardMode.DECODE)) is None
    assert (
        run(model, torch.arange(4), q, latent, ctx(ForwardMode.EXTEND, object()))
        is None
    )


def test_the_write_lands_on_this_ranks_shard_under_kvp(monkeypatch):
    """Under KV parallelism the layer hands the pool this rank's
    local slots and the ownership mask; rows another rank owns, and padding
    rows, resolve to slot 0 with a False mask."""
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    handed = {}
    monkeypatch.setattr(
        paged_attention,
        "mla_prologue",
        lambda query, q_pe, latent, **kw: handed.update(cache=kw["cache"]),
    )
    layer = paged_attention.PagedAttention(
        4, 576, 1.0, num_kv_heads=1, layer_id=0, rotary_emb=None, qk_norm=None
    )
    placement = CachePlacement(
        block_granularity=4, virtual_block_count=8, group=(0, 1), rank=1
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_backend=SimpleNamespace(cache_placement=lambda layer: placement),
        token_to_kv_pool=SimpleNamespace(
            kv_write_target=lambda layer_id, s, m: LatentKVCache(None, False, s, m)
        ),
    )
    q = torch.zeros(4, 4, 576, dtype=torch.bfloat16)
    layer.latent_prologue(
        q,
        q[..., 512:],
        torch.zeros(4, 576, dtype=torch.bfloat16),
        torch.arange(4),
        ctx,
        slots=torch.tensor([4, 9, 8, 0]),
        expanded=None,
        key_rows=None,
    )
    assert handed["cache"].slots.tolist() == [0, 5, 4, 0]
    assert handed["cache"].write_mask.tolist() == [False, True, True, False]


def test_a_query_shard_gathers_the_rotated_latent_before_the_masked_store(
    monkeypatch,
):
    """Under query context parallelism the layer rotates its own rows without
    a cache, all-gathers the rotated latent to the whole span with the plan's
    row counts, and stores it through the owner-masked target: the same
    masked write as KVP alone, fed by every rank's rows."""
    from tokenspeed.runtime.execution.query_shard import QueryShardPlan
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    handed = {}
    rotated = torch.arange(3 * 576, dtype=torch.bfloat16).reshape(3, 576)

    def fake_mla_prologue(query, q_pe, latent, **kw):
        handed["prologue"] = kw
        return SimpleNamespace(query=query, key=None, value=None, latent=rotated)

    def fake_gather(tensor, group, scattered_num_tokens):
        handed["gather"] = (tensor, group, scattered_num_tokens)
        return torch.cat([tensor, tensor[:2]])  # the other rank's two rows

    monkeypatch.setattr(paged_attention, "mla_prologue", fake_mla_prologue)
    monkeypatch.setattr(paged_attention, "token_all_gather", fake_gather)
    monkeypatch.setattr(
        paged_attention,
        "latent_store",
        lambda latent, *, kv_lora_rank, cache: handed.update(
            store=(latent, kv_lora_rank, cache)
        ),
    )
    layer = paged_attention.PagedAttention(
        4, 576, 1.0, num_kv_heads=1, layer_id=0, rotary_emb=None, qk_norm=None
    )
    placement = CachePlacement(
        block_granularity=4, virtual_block_count=8, group=(0, 1), rank=1
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_backend=SimpleNamespace(cache_placement=lambda layer: placement),
        token_to_kv_pool=SimpleNamespace(
            kv_write_target=lambda layer_id, s, m: LatentKVCache(None, False, s, m)
        ),
    )
    plan = QueryShardPlan.from_forward(
        total_tokens=5, input_lengths=[5], size=2, rank=0
    )  # rows [3, 2]: this rank rotates three rows of a five-row span
    q = torch.zeros(3, 4, 576, dtype=torch.bfloat16)
    out = layer.latent_prologue(
        q,
        q[..., 512:],
        torch.zeros(3, 576, dtype=torch.bfloat16),
        torch.arange(3),
        ctx,
        slots=torch.tensor([4, 9, 8, 0, 5]),
        expanded=None,
        key_rows=paged_attention.QueryShardGather(plan, (0, 1)),
    )
    assert out.latent is rotated
    assert handed["prologue"]["cache"] is None
    tensor, group, counts = handed["gather"]
    assert tensor is rotated or torch.equal(tensor, rotated)
    assert group == (0, 1) and counts == [3, 2]
    stored, kv_lora_rank, cache = handed["store"]
    assert stored.shape == (5, 576) and kv_lora_rank == 512
    assert cache.slots.tolist() == [0, 5, 4, 0, 0]
    assert cache.write_mask.tolist() == [False, True, True, False, False]
    with pytest.raises(ValueError, match="whole span"):
        layer.latent_prologue(
            q,
            q[..., 512:],
            torch.zeros(3, 576, dtype=torch.bfloat16),
            torch.arange(3),
            ctx,
            slots=torch.tensor([4, 9, 8]),
            expanded=None,
            key_rows=paged_attention.QueryShardGather(plan, (0, 1)),
        )


def test_an_empty_shard_still_joins_the_gather_and_stores_its_pages(monkeypatch):
    """A rank whose shard has no rows rotates nothing (the prologue kernel is
    not run on zero rows) but joins the latent all-gather with an empty
    contribution and stores the rows it owns of what the others computed;
    skipping either would hang the group or leave its pages unwritten."""
    from tokenspeed.runtime.execution.query_shard import QueryShardPlan

    handed = {}
    monkeypatch.setattr(
        paged_attention,
        "mla_prologue",
        lambda *a, **kw: pytest.fail("the prologue kernel ran on zero rows"),
    )

    def fake_gather(tensor, group, scattered_num_tokens):
        handed["gather"] = (tensor.shape, group, scattered_num_tokens)
        return torch.arange(2 * 576, dtype=torch.bfloat16).reshape(2, 576)

    monkeypatch.setattr(paged_attention, "token_all_gather", fake_gather)
    monkeypatch.setattr(
        paged_attention,
        "latent_store",
        lambda latent, *, kv_lora_rank, cache: handed.update(
            store=(latent, kv_lora_rank, cache)
        ),
    )
    layer = paged_attention.PagedAttention(
        4, 576, 1.0, num_kv_heads=1, layer_id=0, rotary_emb=None, qk_norm=None
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_backend=SimpleNamespace(cache_placement=lambda layer: None),
        token_to_kv_pool=SimpleNamespace(
            kv_write_target=lambda layer_id, s, m: LatentKVCache(None, False, s, m)
        ),
    )
    # Two rows over four ranks: ranks 2 and 3 hold nothing.
    plan = QueryShardPlan.from_forward(
        total_tokens=2, input_lengths=[2], size=4, rank=3
    )
    assert plan.local_rows == 0
    q = torch.zeros(0, 4, 576, dtype=torch.bfloat16)
    out = layer.latent_prologue(
        q,
        q[..., 512:],
        torch.zeros(0, 576, dtype=torch.bfloat16),
        torch.arange(0),
        ctx,
        slots=torch.tensor([4, 9]),
        expanded=None,
        key_rows=paged_attention.QueryShardGather(plan, (0, 1, 2, 3)),
    )
    assert out.query is q and out.latent.shape == (0, 576)
    assert handed["gather"] == (torch.Size([0, 576]), (0, 1, 2, 3), [1, 1, 0, 0])
    stored, kv_lora_rank, cache = handed["store"]
    assert stored.shape == (2, 576) and kv_lora_rank == 512
    assert cache.slots.tolist() == [4, 9] and cache.write_mask is None


def test_the_dense_mla_paths_refuse_a_query_shard_before_any_early_return():
    """``DeepseekV3AttentionMLA.forward`` and the expanded prefill prologue
    attend every row of the span; a sharded forward is refused up front --
    also on a rank whose shard is empty, which must not return before the
    collectives the other ranks join -- and the absorbed projection runs the
    prologue on zero rows without the absorption GEMM."""
    from tokenspeed.runtime.execution.query_shard import QueryShardPlan
    from tokenspeed.runtime.models import deepseek_v3

    plan = QueryShardPlan.from_forward(
        total_tokens=2, input_lengths=[2], size=4, rank=3
    )
    sharded = SimpleNamespace(query_shard=plan)
    attention = deepseek_v3.DeepseekV3AttentionMLA.__new__(
        deepseek_v3.DeepseekV3AttentionMLA
    )
    # No head TP: the attention has no exchange to join on an idle forward.
    attention.has_head_tp = False
    attention.hidden_size = 8
    empty = torch.zeros(0, 8)
    with pytest.raises(RuntimeError, match="cannot take a query shard"):
        attention.forward(torch.arange(0), empty, sharded, None)
    with pytest.raises(RuntimeError, match="cannot take a query shard"):
        attention.forward_normal_chunked_kv_prepare(
            torch.arange(0), empty, empty, sharded, torch.arange(2)
        )
    # Unsharded: the empty-row return stands (the o_proj output shape).
    plain = SimpleNamespace(query_shard=None)
    assert attention.forward(torch.arange(0), empty, plain, None).shape == (0, 8)

    # The absorbed projection on an empty shard: no GEMM, the prologue runs.
    attention.num_local_heads = 2
    attention.qk_head_dim = 6
    attention.qk_nope_head_dim = 4
    attention.qk_rope_head_dim = 2
    attention.kv_lora_rank = 4
    attention.w_kc = None  # a GEMM would fail on it
    attention.mapping = SimpleNamespace(attn=SimpleNamespace(qcp_group=(0, 1, 2, 3)))
    seen = {}

    def prologue(Q, q_pe, latent, positions, ctx, *, slots, expanded, key_rows):
        seen.update(Q=Q, slots=slots, key_rows=key_rows)
        return SimpleNamespace(query=Q)

    attention.attn_mqa = SimpleNamespace(latent_prologue=prologue)
    Q = attention.forward_absorb_qkv_proj(
        torch.zeros(0, 12),
        torch.zeros(0, 6),
        torch.arange(0),
        sharded,
        torch.arange(2),
    )
    assert Q.shape == (0, 2, 6) and seen["slots"].tolist() == [0, 1]
    assert seen["key_rows"].plan is plan and seen["key_rows"].group == (0, 1, 2, 3)


def test_head_caches_take_no_write_mask():
    from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool

    pool = SimpleNamespace(get_kv_buffer=lambda layer_id: (None, None))
    with pytest.raises(ValueError, match="never sharded"):
        MHATokenToKVPool.kv_write_target(
            pool, 0, torch.arange(2), torch.ones(2, dtype=torch.bool)
        )


@pytest.mark.parametrize("norm_cls", [RMSNorm, GemmaRMSNorm])
def test_the_prologue_gets_the_stored_norm_weight(monkeypatch, norm_cls):
    """Gemma's ``1 + w`` is formed in fp32 by the kernel, never in the weight dtype."""
    q_norm, k_norm = norm_cls(64, eps=1e-5), norm_cls(64, eps=1e-5)
    handed = _prologue(
        monkeypatch,
        qk_norm=(q_norm, k_norm),
        mode=ForwardMode.EXTEND,
        rows=3,
        slots=torch.arange(3),
    )
    norm = handed["norm"]
    assert norm.q_weight is q_norm.weight and norm.k_weight is k_norm.weight
    assert norm.weight_offset == norm_cls.weight_offset and norm.eps == 1e-5
    assert handed["return_kv"]


def test_the_prologue_asks_for_a_slot_per_row_it_carries(monkeypatch):
    """A graph-padded forward writes every row: the backend pads the span with the
    dummy slot, so the captured prologue replays at any real token count."""
    handed = _prologue(
        monkeypatch,
        qk_norm=None,
        mode=ForwardMode.EXTEND,
        rows=4,
        slots=torch.tensor([5, 6, 7, 0]),
    )
    assert handed["rows"] == 4 and handed["cache"].slots.numel() == 4


@pytest.mark.parametrize("positions", [[0, 5, 4095], [0, 5, 4097]])
def test_longrope_moves_every_token_to_the_long_rows_past_the_original_context(
    positions,
):
    short, long = [1.0] * 32, [4.0] * 32
    rope = Phi3LongRoPEScaledRotaryEmbedding(
        64, 64, 8192, 4096, 10000, True, short, long
    )
    positions = torch.tensor(positions)
    rotary = rope.as_rotary(positions)
    if (positions > 4096).any():
        table = rope._compute_cos_sin_cache(8192, long, rope.long_mscale)
    else:
        table = rope._compute_cos_sin_cache(4096, short, rope.short_mscale)
    assert torch.equal(rotary.cos_sin_cache[rotary.positions], table[positions])


def test_mrope_hands_the_prologue_its_sections():
    rope = MRotaryEmbedding(
        128, 128, 4096, 10000, True, torch.bfloat16, [24, 20, 20], True
    )
    rotary = rope.as_rotary(torch.zeros(3, 5, dtype=torch.int64))
    assert rotary.mrope == MRope((24, 20, 20), True)
    assert rotary.style is RopeStyle.NEOX
    assert rotary.cos_sin_cache.dtype == torch.float32


@pytest.mark.parametrize("sparse", [True, False])
def test_msa_publishes_a_layer_after_every_field_it_writes(sparse):
    """The prologue wrote K/V; a sparse layer's kernel still writes its index keys."""
    from tokenspeed.runtime.layers.attention.backends.paged.msa import (
        MSAHybridAttnBackend,
    )

    order = []

    class _Leaf:
        def forward_extend(self, q, *args, **kwargs):
            order.append("attention")
            return q

    router = SimpleNamespace(
        write_locations=lambda layer, mode: torch.arange(4),
        _leaf_for=lambda layer: _Leaf(),
    )
    backend = object.__new__(MSAHybridAttnBackend)
    backend.step_counter = SimpleNamespace(record_cache=lambda: order.append("record"))
    backend.sparse_layer_ids = {0} if sparse else set()
    backend.sparse_router = backend.full_router = router
    layer = paged_attention.PagedAttention(
        4, 64, 1.0, num_kv_heads=2, layer_id=0, rotary_emb=None, qk_norm=None
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_backend=backend,
        token_to_kv_pool=None,
        bs=1,
    )
    layer.forward(torch.zeros(4, 4 * 64), None, None, None, ctx)
    assert order == (["attention", "record"] if sparse else ["record", "attention"])


def _functions(tree: ast.Module):
    """Module functions and class methods, by qualified name."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{node.name}.{item.name}", item
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def test_models_run_the_prologue_steps_only_through_the_prologue():
    """Model and drafter code normalizes, rotates, quantizes and writes
    attention K/V only through the attention prologue, apart from draft
    context injection and keys the prologue does not own."""
    runtime = pathlib.Path(__file__).resolve().parents[2] / "python/tokenspeed/runtime"
    callers = set()
    for path in [
        *runtime.glob("models/**/*.py"),
        *runtime.glob("execution/drafter/*.py"),
    ]:
        for name, fn in _functions(ast.parse(path.read_text())):
            if any(
                isinstance(node, ast.Call) and _called_name(node) in _PROLOGUE_STEPS
                for node in ast.walk(fn)
            ):
                callers.add(f"{path.relative_to(runtime)}:{name}")
    assert callers == _CONTEXT_KV_WRITERS | _OTHER_KEY_OWNERS
