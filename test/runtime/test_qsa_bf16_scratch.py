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

"""QSA BF16 prefill scratch: sizing, reservation, routing and serving."""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.qsa import torch_qsa_bf16_kv_scratch
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry
from torch.utils._python_dispatch import TorchDispatchMode

import tokenspeed.runtime.layers.attention.backends.paged.qsa as qsa_leaf
from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention import registry as attention_registry
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    Qwen4ExpBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen4_exp import (
    Qwen4ExpRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.qsa import bf16_scratch
from tokenspeed.runtime.layers.attention.qsa.bf16_scratch import (
    MIN_ROWS,
    TOKENS_PER_ROW,
    QSABf16PrefillScratch,
    QSABf16ScratchPlan,
    QSAExtendLengths,
    plan_qsa_bf16_scratch,
)
from tokenspeed.runtime.layers.attention.qsa.metadata import QSALayout, QSASelection
from tokenspeed.runtime.layers.paged_attention import PagedAttention

# Executed as a script by run_ci_suite: the test dir must be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-1gpu")

FP8 = torch.float8_e4m3fn
CONTEXT_LEN = 4096
MAX_BS = 8
PAGE = 256
_GROUPS = {
    FULL_ATTENTION: PAGE,
    QWEN4_EXP_QSA_CACHE_GROUP: PAGE,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP: 64,
}


# ---------------------------------------------------------------------------
# Sizing and reservation
# ---------------------------------------------------------------------------


def test_plan_holds_one_context_and_one_chunk_of_pages():
    plan = plan_qsa_bf16_scratch(
        context_len=262_156,
        max_rows=16_384,
        page_size=256,
        kv_heads=2,
        head_dim=256,
        q_heads=24,
    )
    # Reserved page 0, ceil((262156 + 16384) / 256) = 1089 pages and 64 partial
    # last pages.
    assert plan.capacity_pages == 1154
    kv_bytes = 2 * 1154 * 256 * 2 * 256 * 2
    slot_bytes = 16_384 * 2051 * 4
    output_bytes = 16_384 * 24 * 256 * 2
    assert plan.nbytes == kv_bytes + slot_bytes + output_bytes


def test_plan_never_holds_more_pages_than_a_chunk_may_copy():
    plan = plan_qsa_bf16_scratch(
        context_len=262_144,
        max_rows=MIN_ROWS,
        page_size=256,
        kv_heads=1,
        head_dim=256,
        q_heads=12,
    )
    assert plan.capacity_pages == 1 + MIN_ROWS * TOKENS_PER_ROW // 256


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"max_rows": MIN_ROWS - 1}, "chunked-prefill-size"),
        ({"head_dim": 128}, "256-wide heads"),
        ({"q_heads": 3}, "query groups"),
    ],
)
def test_plan_rejects_geometry_the_scratch_cannot_serve(overrides, message):
    kwargs = {
        "context_len": 4096,
        "max_rows": 4096,
        "page_size": 64,
        "kv_heads": 2,
        "head_dim": 256,
        "q_heads": 24,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        plan_qsa_bf16_scratch(**kwargs)


def _recipe(*, enabled, role="null", kv_cache_dtype=FP8, chunk=2048, budget=512 << 20):
    """A real Qwen4-Exp recipe: two QSA full-attention layers, 24/2 heads."""
    layer_types = (FULL_ATTENTION, FULL_ATTENTION)
    text_config = SimpleNamespace(
        ple_layer_ids=(),
        indexer_n_heads=1,
        indexer_head_dim=8,
        indexer_compress_ratio=4,
    )
    spec = MHAConfig(
        backend_name="qsa",
        num_attention_heads=24,
        num_kv_heads=2,
        head_dim=256,
        attn_tp_size=1,
        cache_layer_types=layer_types,
        sliding_window_tokens=None,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        kernel_page_size=None,
        context_len=CONTEXT_LEN,
        max_bs=2,
        pd_disaggregation_enabled=role != "null",
        speculative_num_steps=0,
        speculative_num_draft_tokens=1,
        is_draft=False,
        draft_block_decode=False,
        components=(spec,),
    )
    return Qwen4ExpRecipe(
        server_args=SimpleNamespace(
            kv_offload_config=None,
            block_size=64,
            max_total_tokens=None,
            speculative_num_draft_tokens=1,
            speculative_eagle_topk=1,
            speculative_draft_tree=False,
            enable_replay_ssm=False,
            mamba_ssm_dtype="float32",
            disaggregation_mode=role,
            enable_qsa_prefill_bf16_scratch=enabled,
            chunked_prefill_size=chunk,
            max_prefill_tokens=8192,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
            hf_text_config=text_config,
            num_attention_layers=len(layer_types),
        ),
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=budget,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


def test_recipe_reserves_the_scratch_before_sizing_the_pool():
    budget = 512 << 20
    plan = _recipe(enabled=True).qsa_bf16_scratch_plan
    assert plan == plan_qsa_bf16_scratch(
        context_len=CONTEXT_LEN,
        max_rows=2048,
        page_size=PAGE,
        kv_heads=2,
        head_dim=256,
        q_heads=24,
    )
    with_scratch = _recipe(enabled=True, budget=budget).setup()
    without = _recipe(enabled=False, budget=budget).setup()
    # The scratch is the whole fixed workspace of this engine, and the pool
    # is sized as if the budget were that much smaller.
    assert with_scratch.fixed_workspace_bytes == plan.nbytes
    assert without.fixed_workspace_bytes == 0
    smaller_budget = _recipe(enabled=False, budget=budget - plan.nbytes).setup()
    plan_with = with_scratch.spec.memory_plan
    assert plan_with.num_lcm_blocks == smaller_budget.spec.memory_plan.num_lcm_blocks
    assert plan_with.num_lcm_blocks < without.spec.memory_plan.num_lcm_blocks
    assert plan_with.arena_bytes + with_scratch.fixed_workspace_bytes <= budget


def test_recipe_reserves_nothing_on_the_decode_role_or_when_off():
    assert _recipe(enabled=True, role="decode").qsa_bf16_scratch_plan is None
    assert _recipe(enabled=False).qsa_bf16_scratch_plan is None
    assert _recipe(enabled=True, role="prefill").qsa_bf16_scratch_plan is not None


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"kv_cache_dtype": torch.bfloat16}, "fp8_e4m3"),
        ({"chunk": 1024}, "chunked-prefill-size"),
    ],
)
def test_recipe_rejects_a_scratch_that_cannot_serve(overrides, message):
    with pytest.raises(ValueError, match=message):
        _recipe(enabled=True, **overrides).workspace_bytes()


# ---------------------------------------------------------------------------
# The attention leaf, through the real router and composite backend
# ---------------------------------------------------------------------------


def _reference_attention(
    q,
    k_cache,
    v_cache,
    selected_slots,
    *,
    scale,
    max_seqlen_q,
    metadata_capacity_rows,
    k_scale,
    v_scale,
    out=None,
):
    """Stand-in for the GPU kernels: FP32 attention over the selected slots.

    Query heads form contiguous groups per KV head; non-positive slots are
    masked and a row without slots is zero, as in the kernels.
    """
    del max_seqlen_q, metadata_capacity_rows
    k_scale = 1.0 if k_scale is None else k_scale
    v_scale = 1.0 if v_scale is None else v_scale
    rows, heads, dim = q.shape
    kv_heads = k_cache.shape[1]
    result = torch.empty_like(q)
    for start in range(0, rows, 32):
        slots = selected_slots[start : start + 32].long()
        keys = k_cache[slots.clamp_min(0)].float() * k_scale
        values = v_cache[slots.clamp_min(0)].float() * v_scale
        query = q[start : start + 32].float().view(-1, kv_heads, heads // kv_heads, dim)
        scores = torch.einsum("ckgd,cnkd->ckgn", query, keys) * scale
        scores = scores.masked_fill(~(slots > 0)[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        chunk = torch.einsum("ckgn,cnkd->ckgd", weights, values)
        result[start : start + 32] = chunk.reshape(-1, heads, dim).to(q.dtype)
    if out is None:
        return result
    out.copy_(result)
    return out


class _OpRecorder(TorchDispatchMode):
    """Records every ATen op the code under test dispatches, except inside
    the GPU kernel stand-ins (on a GPU those are kernel launches)."""

    def __init__(self):
        super().__init__()
        self.ops = []
        self._paused = 0

    @contextmanager
    def paused(self):
        self._paused += 1
        try:
            yield
        finally:
            self._paused -= 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not self._paused:
            self.ops.append(func)
        return func(*args, **(kwargs or {}))


class _Calls:
    """Wrap a kernel stand-in, counting calls and keeping it unrecorded."""

    def __init__(self, fn):
        self.fn = fn
        self.recorder: _OpRecorder | None = None
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        if self.recorder is None:
            return self.fn(*args, **kwargs)
        with self.recorder.paused():
            return self.fn(*args, **kwargs)


def _config(*, is_draft=False, kv_cache_dtype=FP8):
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=256,
        attn_tp_size=1,
    )
    return AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        kernel_page_size=None,
        context_len=CONTEXT_LEN,
        max_bs=MAX_BS,
        pd_disaggregation_enabled=False,
        speculative_num_steps=0,
        speculative_num_draft_tokens=1,
        is_draft=is_draft,
        draft_block_decode=False,
        components=(spec,),
    )


def _plan(*, max_rows=4096, context_len=CONTEXT_LEN):
    return plan_qsa_bf16_scratch(
        context_len=context_len,
        max_rows=max_rows,
        page_size=PAGE,
        kv_heads=1,
        head_dim=256,
        q_heads=2,
    )


def _root(config):
    """Composite backend over a real QSA router; the pool is a stand-in."""
    router = attention_registry.create_paged_router(
        config, AttentionArch.MHA, backend_name="qsa"
    )
    pool = SimpleNamespace(
        field_layer_range=range(2),
        _field_layer_id=lambda layer_id: layer_id,
        paged_group_ids=tuple(_GROUPS),
        arena=SimpleNamespace(
            offload=None,
            plan=SimpleNamespace(fields=[]),
            cache_group_specs=tuple(
                SimpleNamespace(
                    group_id=group_id,
                    block_granularity=granularity,
                    attn_kind="full_history",
                    rows_per_page=granularity,
                    entry_stride_tokens=1,
                    sliding_window_tokens=None,
                )
                for group_id, granularity in _GROUPS.items()
            ),
        ),
    )
    root = Qwen4ExpBackend(config, router, None, None)
    root.set_cache_pool(pool)
    root.init_cuda_graph_state(MAX_BS)
    return root, router


def _reference_copy(*args, override, solution, **kwargs):
    """The copy op on CPU tensors: its PyTorch reference, which the GPU kernel
    test compares the Triton kernel against."""
    del override, solution
    return torch_qsa_bf16_kv_scratch(*args, **kwargs)


@pytest.fixture
def kernels(monkeypatch):
    """Stand-ins for the GPU kernels on CPU tensors: the BF16 copy, the CuTe
    attention it feeds and the FP8 path, each counting its calls."""
    fp8_path = _Calls(_reference_attention)
    monkeypatch.setattr(
        qsa_leaf,
        "qsa_sparse_attention",
        lambda *args, override, solution, sparse_metadata, **kwargs: fp8_path(
            *args, **kwargs
        ),
    )
    attention = _Calls(_reference_attention)
    monkeypatch.setattr(
        bf16_scratch,
        "select_qsa_sparse_attention_kernel",
        lambda *args, **kwargs: attention,
    )
    copy = _Calls(_reference_copy)
    monkeypatch.setattr(bf16_scratch, "qsa_bf16_kv_scratch", copy)
    return SimpleNamespace(fp8_path=fp8_path, attention=attention, copy=copy)


def test_leaf_allocates_the_reserved_plan_once(kernels):
    root, router = _root(_config())
    leaf = router.leaves[FULL_ATTENTION]
    plan = _plan()
    # Exactly the reserved bytes: the page map takes none of its own.
    assert root.preallocate_qsa_bf16_scratch(plan) == plan.nbytes
    scratch = leaf._bf16_scratch
    k_address = scratch.k.data_ptr()
    # Padded slots address scratch page 0, which the copy never writes.
    assert bool((scratch.k[:PAGE] == 0).all()) and bool((scratch.v[:PAGE] == 0).all())
    # The rebuilt pool after the startup probe hands the same plan to the same
    # leaf: nothing is allocated again.
    assert root.preallocate_qsa_bf16_scratch(plan) == plan.nbytes
    assert leaf._bf16_scratch is scratch and scratch.k.data_ptr() == k_address
    with pytest.raises(RuntimeError, match="planned differently"):
        root.preallocate_qsa_bf16_scratch(_plan(max_rows=2048))


def test_startup_checks_the_planned_bytes_against_the_allocation(kernels):
    root, _ = _root(_config())
    plan = _plan()
    kwargs = dict(
        server_args=SimpleNamespace(speculative_num_draft_tokens=1),
        config=SimpleNamespace(max_bs=4, qcp_size=1),
        backend=root,
        draft_backend=None,
        uses_paged_state_verify=False,
        is_inkling=False,
        qsa_bf16_scratch=plan,
    )
    attention_registry._prepare_fixed_workspaces(**kwargs, expected_bytes=plan.nbytes)
    with pytest.raises(RuntimeError, match="does not match allocated tensors"):
        attention_registry._prepare_fixed_workspaces(
            **kwargs, expected_bytes=plan.nbytes - 1
        )


@pytest.mark.parametrize(
    "config,plan_page,message",
    [
        (_config(is_draft=True), PAGE, "target only"),
        (_config(kv_cache_dtype=torch.bfloat16), PAGE, "FP8 E4M3"),
        (_config(), 64, "does not match"),
    ],
)
def test_leaf_refuses_a_plan_it_cannot_serve(kernels, config, plan_page, message):
    root, _ = _root(config)
    plan = plan_qsa_bf16_scratch(
        context_len=CONTEXT_LEN,
        max_rows=4096,
        page_size=plan_page,
        kv_heads=1,
        head_dim=256,
        q_heads=2,
    )
    with pytest.raises((RuntimeError, ValueError), match=message):
        root.preallocate_qsa_bf16_scratch(plan)


class _Forward:
    """One extend round through the real router and the QSA leaf."""

    def __init__(self, router, requests, *, pool_pages=96, seed=0, decode=False):
        generator = torch.Generator().manual_seed(seed)
        self.router = router
        self.leaf = router.leaves[FULL_ATTENTION]
        self.requests = requests
        bs = len(requests)
        lengths = [prefix + new for prefix, new in requests]
        self.rows = sum(new for _, new in requests)
        slots = pool_pages * PAGE
        values = torch.rand((2, slots, 1, 256), generator=generator) * 8 - 4
        self.k, self.v = values.to(FP8).unbind(0)
        widths = [-(-length // PAGE) for length in lengths]
        free = (torch.randperm(pool_pages - 1, generator=generator) + 1).tolist()
        blocks = torch.zeros((bs, CONTEXT_LEN // PAGE), dtype=torch.int32)
        for row, width in enumerate(widths):
            blocks[row, :width] = torch.tensor([free.pop() for _ in range(width)])
        self.seq_lens = torch.tensor(lengths, dtype=torch.int32)
        new_tokens = torch.tensor([new for _, new in requests], dtype=torch.int32)
        if decode:
            router.refresh_decode_metadata(
                bs,
                bs,
                torch.arange(1, bs + 1, dtype=torch.int32),
                self.seq_lens,
                forward_mode=ForwardMode.DECODE,
                block_tables={group: blocks for group in _GROUPS},
                num_extends=0,
                for_graph_replay=False,
            )
        else:
            router.init_forward_metadata(
                bs,
                bs,
                torch.arange(1, bs + 1, dtype=torch.int32),
                self.seq_lens,
                ForwardMode.EXTEND,
                block_tables={group: blocks for group in _GROUPS},
                extend_seq_lens=new_tokens,
                extend_seq_lens_cpu=new_tokens,
                extend_prefix_lens=self.seq_lens - new_tokens,
                extend_prefix_lens_cpu=self.seq_lens - new_tokens,
                extend_replay_lens_cpu=torch.zeros_like(self.seq_lens),
                extend_prompt_lens_cpu=self.seq_lens,
                extend_with_prefix=True,
                query_shard=None,
                block_tables_cpu={},
            )
        self.mode = ForwardMode.DECODE if decode else ForwardMode.EXTEND
        table = router.group_view(FULL_ATTENTION, bs).page_table
        positions = torch.cat(
            [torch.arange(prefix, prefix + new) for prefix, new in requests]
        )
        owners = torch.repeat_interleave(torch.arange(bs), new_tokens.long())
        self.topk = self._selection(positions, owners, table, generator)
        self.q = torch.randn((self.rows, 2 * 256), generator=generator).to(
            torch.bfloat16
        )
        self.layout = QSALayout(
            seq_lens=self.seq_lens,
            logical_positions=positions,
            request_indices=owners,
            qsa_locs=torch.zeros(self.rows, dtype=torch.int64),
            recent_locs=torch.zeros(self.rows, dtype=torch.int32),
            complete_blocks=((positions + 1) // 4).to(torch.int32),
            queries_per_request=None,
            qsa_page_table=table,
            full_page_table=table,
            full_kernel_page_size=PAGE,
            reset_draft_tags=None,
        )
        self.selection = QSASelection(
            self.topk, torch.full((self.rows, 512), -1, dtype=torch.int32), 4
        )
        self.layer = PagedAttention(
            num_heads=2,
            head_dim=256,
            scaling=0.0625,
            num_kv_heads=1,
            layer_id=1,
            logit_cap=0.0,
            v_head_dim=256,
            sliding_window_size=-1,
            rotary_emb=None,
            qk_norm=None,
        )
        self.layer.bind_cache_group(FULL_ATTENTION)

    @staticmethod
    def _selection(positions, owners, table, generator):
        """Up to 2048 distinct earlier slots per row, then its last three
        positions (the open tail); -1 pads rows near their request's start."""
        span = int(positions.max()) + 1
        scores = torch.rand((positions.numel(), span), generator=generator)
        tokens = torch.arange(span)
        scores[tokens[None, :] >= (positions - 2)[:, None]] = -1.0
        top = scores.topk(min(2048, span), dim=1)
        chosen = torch.where(top.values >= 0, top.indices, -1)
        tail = positions[:, None] + torch.arange(-2, 1)
        logical = torch.full((positions.numel(), 2051), -1, dtype=torch.int64)
        logical[:, : chosen.shape[1]] = chosen
        logical[:, 2048:] = tail
        pages = table[owners.long()].long()
        safe = logical.clamp_min(0)
        slots = pages.gather(1, safe // PAGE) * PAGE + safe % PAGE
        selected = torch.where(logical >= 0, slots, -1).to(torch.int32)
        # Every non-positive flavour of padding.
        selected[0, 2047] = 0
        selected[-1, 2046] = -12345
        return selected

    def run(self, root):
        root.sparse_topk.qsa_selection = self.selection
        root.sparse_topk.qsa_metadata = self.layout
        ctx = SimpleNamespace(
            attn_backend=root,
            bs=len(self.requests),
            forward_mode=self.mode,
            draft_narrowing=None,
        )
        pool = SimpleNamespace(get_kv_buffer=lambda layer_id: (self.k, self.v))
        forward = (
            self.leaf.forward_decode
            if self.mode.is_decode()
            else self.leaf.forward_extend
        )
        return forward(
            self.q,
            None,
            None,
            self.layer,
            None,
            pool,
            len(self.requests),
            topk_indices=self.topk,
            ctx=ctx,
        )

    def fp8_reference(self):
        return _reference_attention(
            self.q.view(self.rows, 2, 256),
            self.k,
            self.v,
            self.topk,
            scale=0.0625,
            max_seqlen_q=None,
            metadata_capacity_rows=None,
            k_scale=1.0,
            v_scale=1.0,
        ).reshape(self.rows, -1)


def test_large_extend_attends_the_bf16_copy_with_the_same_values(kernels):
    root, router = _root(_config())
    root.preallocate_qsa_bf16_scratch(_plan())
    scratch = router.leaves[FULL_ATTENTION]._bf16_scratch
    # Several requests in one chunk, rows at request starts and positions
    # below 2048 (mostly padding), a shared tail and padding flavours.
    forward = _Forward(router, [(1000, 1400), (0, 600), (37, 100)], seed=1)
    output = forward.run(root)
    assert kernels.copy.count == 1 and kernels.fp8_path.count == 0
    assert output.data_ptr() == scratch.output.data_ptr()
    # The copy is exact, so the scratch call sees the FP8 path's operands.
    assert torch.equal(output, forward.fp8_reference())


def test_serving_takes_views_of_the_reservation_only(kernels):
    root, router = _root(_config())
    root.preallocate_qsa_bf16_scratch(_plan())
    forward = _Forward(router, [(500, 2100)], seed=2)
    recorder = _OpRecorder()
    kernels.copy.recorder = recorder
    kernels.attention.recorder = recorder
    with recorder:
        output = forward.run(root)
    assert kernels.copy.count == 1 and kernels.attention.count == 1
    # Everything the path does around the two kernels is a view: no tensor
    # is constructed, copied or computed while serving.
    views = {
        torch.ops.aten.view.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.alias.default,
        torch.ops.aten._reshape_alias.default,
    }
    assert recorder.ops and set(recorder.ops) <= views, recorder.ops
    assert torch.equal(output, forward.fp8_reference())


def test_long_contexts_for_few_rows_keep_the_fp8_path(kernels):
    # A call may copy TOKENS_PER_ROW cache tokens per query row: 2048 rows
    # over 65 requests of 4096 tokens need 1041 pages, more than 1024 + 1.
    plan = QSABf16ScratchPlan(
        page_size=PAGE,
        kv_heads=1,
        head_dim=256,
        q_heads=2,
        capacity_pages=1100,
        max_rows=MIN_ROWS,
    )
    scratch = QSABf16PrefillScratch(plan, device="cpu")
    lengths = QSAExtendLengths(seq_lens=(CONTEXT_LEN,) * 65, rows=MIN_ROWS)
    assert (1041 - 1) * PAGE > MIN_ROWS * TOKENS_PER_ROW
    q = torch.zeros((MIN_ROWS, 2, 256), dtype=torch.bfloat16)
    unused = torch.zeros(1)
    output = scratch.attention(
        q,
        unused,
        unused,
        unused,
        page_table=unused,
        seq_lens=unused,
        lengths=lengths,
        scale=1.0,
    )
    assert output is None and kernels.copy.count == 0


@pytest.mark.parametrize(
    "requests,decode",
    [
        # Fewer rows than MIN_ROWS.
        ([(2000, MIN_ROWS - 1)], False),
        # More rows than the plan's chunk.
        ([(0, 4096), (0, 1)], False),
        # Decode keeps the FP8 path.
        ([(3000, 1)] * 4, True),
    ],
)
def test_other_calls_keep_the_fp8_path(kernels, requests, decode):
    root, router = _root(_config())
    root.preallocate_qsa_bf16_scratch(_plan())
    forward = _Forward(router, requests, seed=3, decode=decode)
    output = forward.run(root)
    assert kernels.copy.count == 0 and kernels.fp8_path.count == 1
    if not decode:
        assert torch.equal(output, forward.fp8_reference())


def test_contexts_at_and_past_the_capacity(kernels):
    root, router = _root(_config())
    # 1 + ceil((1024 + 4096) / 256) + 64 = 85 pages.
    plan = _plan(context_len=1024)
    root.preallocate_qsa_bf16_scratch(plan)
    assert plan.capacity_pages == 85
    # 84 context pages: four 4096-token requests (64 pages), then 16 and 4.
    at_capacity = [(4096 - 512, 512)] * 4 + [(16 * PAGE - 1, 1), (4 * PAGE - 1, 1)]
    forward = _Forward(router, at_capacity, seed=4)
    output = forward.run(root)
    assert kernels.copy.count == 1 and kernels.fp8_path.count == 0
    assert torch.equal(output, forward.fp8_reference())
    past = [(4096 - 512, 512)] * 4 + [(16 * PAGE - 1, 1), (4 * PAGE, 1)]
    forward = _Forward(router, past, seed=5)
    output = forward.run(root)
    assert kernels.copy.count == 1 and kernels.fp8_path.count == 1
    assert torch.equal(output, forward.fp8_reference())


def test_mixed_batches_have_no_host_lengths(kernels):
    root, router = _root(_config())
    leaf = router.leaves[FULL_ATTENTION]
    root.preallocate_qsa_bf16_scratch(_plan())
    seq_lens = torch.tensor([2600, 900], dtype=torch.int32)
    new = torch.tensor([2500], dtype=torch.int32)
    router.init_forward_metadata(
        2,
        1,
        torch.tensor([1, 2], dtype=torch.int32),
        seq_lens,
        ForwardMode.MIXED,
        block_tables={
            group: torch.tensor([[1] * 11, [12] * 11], dtype=torch.int32)
            for group in _GROUPS
        },
        extend_seq_lens=new,
        extend_seq_lens_cpu=new,
        extend_prefix_lens=seq_lens[:1] - new,
        extend_prefix_lens_cpu=seq_lens[:1] - new,
        extend_replay_lens_cpu=torch.zeros(1, dtype=torch.int32),
        extend_prompt_lens_cpu=seq_lens[:1],
        extend_with_prefix=True,
        query_shard=None,
        block_tables_cpu={},
    )
    assert leaf._extend_lengths is None
    _Forward(router, [(100, 2500), (0, 600)], seed=6)
    assert leaf._extend_lengths == QSAExtendLengths(seq_lens=(2600, 600), rows=3100)


def test_warmup_runs_both_paths_on_one_call(kernels):
    root, router = _root(_config())
    root.preallocate_qsa_bf16_scratch(_plan())
    forward = _Forward(router, [(0, 3000)], seed=7)
    with kernel_warmup(enabled=True):
        output = forward.run(root)
    # The scratch path compiles on the call; the FP8 path warms on it too.
    assert kernels.copy.count == 1 and kernels.fp8_path.count == 1
    assert torch.equal(output, forward.fp8_reference())


# ---------------------------------------------------------------------------
# GPU: the scratch against the BF16 kernel over the dequantized cache
# ---------------------------------------------------------------------------


def _cute_qsa_available() -> bool:
    if not torch.cuda.is_available():
        return False
    spec = KernelRegistry.get().get_by_name("cute_dsl_blackwell_qsa_sparse_attention")
    return spec is not None and spec.capability.satisfied_by(current_platform())


requires_cute_qsa = pytest.mark.skipif(
    not _cute_qsa_available(),
    reason="requires the CuTe DSL QSA kernel (SM100 to SM107)",
)


def _kernel_test_utils():
    kernel_test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tokenspeed-kernel",
        "test",
    )
    spec = importlib.util.spec_from_file_location(
        "qsa_bf16_scratch_kernel_test_utils", os.path.join(kernel_test_dir, "utils.py")
    )
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils


class _GpuCall:
    """A prefill call over an FP8 cache with real QSA selections."""

    def __init__(self, requests, *, page_size, pool_pages, seed):
        from tokenspeed_kernel.ops.attention.qsa.triton import (
            qwen4_exp_qsa_selected_slots,
        )

        device = "cuda"
        generator = torch.Generator(device=device).manual_seed(seed)
        lengths = [prefix + new for prefix, new in requests]
        widths = [-(-length // page_size) for length in lengths]
        values = torch.rand(
            (2, pool_pages * page_size, 2, 256), generator=generator, device=device
        )
        self.k, self.v = (values * 8 - 4).to(FP8).unbind(0)
        free = torch.randperm(pool_pages - 1, generator=generator, device=device) + 1
        table = torch.zeros(
            (len(requests), max(widths) + 2), dtype=torch.int32, device=device
        )
        start = 0
        for row, width in enumerate(widths):
            table[row, :width] = free[start : start + width].to(torch.int32)
            start += width
        self.table = table
        self.seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
        positions = torch.cat(
            [torch.arange(p, p + n, device=device) for p, n in requests]
        )
        owners = torch.repeat_interleave(
            torch.arange(len(requests), dtype=torch.int32, device=device),
            torch.tensor([n for _, n in requests], device=device),
        )
        complete = ((positions + 1) // 4).to(torch.int32)
        scores = torch.rand(
            (positions.numel(), int(complete.max()) + 1),
            generator=generator,
            device=device,
        )
        blocks = torch.arange(scores.shape[1], device=device)
        scores = torch.where(blocks[None, :] < complete[:, None], scores, -1.0)
        top = min(512, scores.shape[1])
        chosen = scores.topk(top, dim=1)
        selected_blocks = torch.full(
            (positions.numel(), 512), -1, dtype=torch.int32, device=device
        )
        selected_blocks[:, :top] = torch.where(
            chosen.values >= 0, chosen.indices, -1
        ).to(torch.int32)
        self.selected = qwen4_exp_qsa_selected_slots(
            selected_blocks,
            complete,
            positions,
            owners,
            table,
            page_size,
            4,
            2048,
            enable_pdl=False,
        )
        self.selected[0, 9] = 0
        self.selected[-1, 11] = -12345
        self.q = torch.randn(
            (positions.numel(), 24, 256), generator=generator, device=device
        ).to(torch.bfloat16)
        self.lengths = QSAExtendLengths(
            seq_lens=tuple(lengths), rows=sum(n for _, n in requests)
        )

    def scratch(self, scratch):
        return scratch.attention(
            self.q,
            self.k,
            self.v,
            self.selected,
            page_table=self.table,
            seq_lens=self.seq_lens,
            lengths=self.lengths,
            scale=0.0625,
        )

    def bf16_reference(self):
        from tokenspeed_kernel.ops.attention.qsa.cute_dsl import (
            cute_dsl_blackwell_qsa_sparse_attention,
        )

        return cute_dsl_blackwell_qsa_sparse_attention(
            self.q,
            self.k.to(torch.bfloat16),
            self.v.to(torch.bfloat16),
            self.selected,
            scale=0.0625,
            max_seqlen_q=1,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
        )


def _gpu_plan():
    # 1 + ceil((8192 + 4096) / 64) + 64 = 257 pages of 64 tokens.
    return plan_qsa_bf16_scratch(
        context_len=8192,
        max_rows=4096,
        page_size=64,
        kv_heads=2,
        head_dim=256,
        q_heads=24,
    )


@requires_cute_qsa
@pytest.mark.parametrize(
    "requests",
    [
        # Several requests per chunk, rows at request starts, positions
        # below 2048 and a long prefix.
        [(0, 700), (3000, 900), (17, 500), (1200, 300)],
        # One request whose rows all sit below 2048 (mostly padding).
        [(0, 2048)],
        # Exactly at the capacity: 256 context pages plus page 0.
        [(4096 - 512, 512)] * 4,
    ],
)
def test_scratch_matches_the_bf16_kernel_on_the_dequantized_cache(requests):
    scratch = QSABf16PrefillScratch(_gpu_plan(), device="cuda")
    call = _GpuCall(requests, page_size=64, pool_pages=320, seed=len(requests))
    output = call.scratch(scratch)
    assert output is not None
    assert torch.equal(output, call.bf16_reference())


@requires_cute_qsa
def test_scratch_returns_none_past_its_capacity():
    scratch = QSABf16PrefillScratch(_gpu_plan(), device="cuda")
    call = _GpuCall(
        [(4096 - 512, 512)] * 4 + [(0, 1)], page_size=64, pool_pages=320, seed=9
    )
    assert call.scratch(scratch) is None


@requires_cute_qsa
def test_scratch_serving_allocates_and_compiles_nothing_after_warmup():
    from tokenspeed_kernel.ops.attention.qsa.triton import (
        _qsa_bf16_kv_scratch_copy_kernel,
        _qsa_bf16_kv_scratch_remap_kernel,
    )

    utils = _kernel_test_utils()
    scratch = QSABf16PrefillScratch(_gpu_plan(), device="cuda")
    calls = [
        _GpuCall(requests, page_size=64, pool_pages=pool_pages, seed=seed)
        for seed, (requests, pool_pages) in enumerate(
            [
                ([(100, 2048)], 320),
                ([(0, 3000), (2000, 1000)], 400),
                ([(5, 600)] * 4 + [(0, 1600)], 333),
                ([(4000, 4096)], 320),
            ]
        )
    ]
    with kernel_warmup(enabled=True):
        assert calls[0].scratch(scratch) is None
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    with ExitStack() as checks:
        checks.enter_context(
            utils.assert_no_triton_compile(
                _qsa_bf16_kv_scratch_copy_kernel, _qsa_bf16_kv_scratch_remap_kernel
            )
        )
        checks.enter_context(utils.assert_no_cute_compile())
        outputs = [call.scratch(scratch) for call in calls]
        torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
    assert all(output is not None for output in outputs)
