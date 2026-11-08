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
from dataclasses import replace
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
from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import QwenGDNRecipe
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

register_cuda_ci(est_time=150, suite="runtime-1gpu")

FP8 = torch.float8_e4m3fn
CONTEXT_LEN = 4096
MAX_BS = 8
PAGE = 256
_GROUPS = {
    FULL_ATTENTION: PAGE,
    QWEN4_EXP_QSA_CACHE_GROUP: PAGE,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP: 64,
}


def _cute_qsa_available() -> bool:
    if not torch.cuda.is_available():
        return False
    spec = KernelRegistry.get().get_by_name("cute_dsl_blackwell_qsa_sparse_attention")
    return spec is not None and spec.capability.satisfied_by(current_platform())


requires_cute_qsa = pytest.mark.skipif(
    not _cute_qsa_available(),
    reason="requires the CuTe DSL QSA kernel (SM100 to SM107)",
)


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


def _recipe(
    *,
    enabled,
    role="null",
    speculative=False,
    kv_cache_dtype=FP8,
    chunk=2048,
    indexer_heads=1,
    budget=512 << 20,
    recipe_class=Qwen4ExpRecipe,
):
    """A real cache recipe: two QSA full-attention layers with 24/2 heads and,
    with ``speculative``, one QSA draft layer verifying three tokens."""
    layer_types = (FULL_ATTENTION, FULL_ATTENTION)
    width = 3 if speculative else 1
    text_config = SimpleNamespace(
        ple_layer_ids=(),
        indexer_n_heads=indexer_heads,
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
        speculative_num_steps=2 if speculative else 0,
        speculative_num_draft_tokens=width,
        is_draft=False,
        draft_block_decode=False,
        components=(spec,),
    )
    draft_config = (
        replace(
            config,
            is_draft=True,
            components=(replace(spec, cache_layer_types=(FULL_ATTENTION,)),),
        )
        if speculative
        else None
    )
    return recipe_class(
        server_args=SimpleNamespace(
            kv_offload_config=None,
            block_size=64,
            max_total_tokens=None,
            speculative_num_draft_tokens=width,
            speculative_eagle_topk=1,
            speculative_draft_tree=False,
            enable_replay_ssm=False,
            mamba_ssm_dtype="float32",
            disaggregation_mode=role,
            enable_qsa_prefill_bf16_scratch=enabled,
            chunked_prefill_size=chunk,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
            hf_text_config=text_config,
            num_attention_layers=len(layer_types),
        ),
        attn_config=config,
        draft_model_config=(
            SimpleNamespace(num_attention_layers=1, hf_text_config=text_config)
            if speculative
            else None
        ),
        draft_attn_config=draft_config,
        cache_budget_bytes=budget,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


@pytest.mark.parametrize(
    "role,speculative",
    [
        ("null", False),
        # Speculative decoding adds the QSA verify staging to the workspace.
        ("null", True),
        # The PD prefill role stages no verify, with or without a draft.
        ("prefill", False),
        ("prefill", True),
    ],
)
def test_recipe_reserves_the_scratch_before_sizing_the_pool(role, speculative):
    budget = 512 << 20

    def setup(*, enabled, budget):
        return _recipe(
            enabled=enabled, role=role, speculative=speculative, budget=budget
        ).setup()

    plan = _recipe(
        enabled=True, role=role, speculative=speculative
    ).qsa_bf16_scratch_plan
    assert plan == plan_qsa_bf16_scratch(
        context_len=CONTEXT_LEN,
        max_rows=2048,
        page_size=PAGE,
        kv_heads=2,
        head_dim=256,
        q_heads=24,
    )
    with_scratch = setup(enabled=True, budget=budget)
    without = setup(enabled=False, budget=budget)
    assert (without.fixed_workspace_bytes > 0) == (speculative and role == "null")
    # The scratch joins the engine's other fixed workspaces, and the pool is
    # sized as if the budget were that much smaller.
    assert with_scratch.fixed_workspace_bytes == (
        without.fixed_workspace_bytes + plan.nbytes
    )
    smaller_budget = setup(enabled=False, budget=budget - plan.nbytes)
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
        ({"chunk": -1}, "needs chunked prefill"),
        # A Qwen4-Exp checkpoint without an indexer has no QSA layers.
        ({"indexer_heads": None}, "needs a model with QSA layers"),
    ],
)
def test_recipe_rejects_a_scratch_that_cannot_serve(overrides, message):
    with pytest.raises(ValueError, match=message):
        _recipe(enabled=True, **overrides).workspace_bytes()


def test_families_without_qsa_layers_refuse_the_scratch():
    # Startup reads the plan of every family's recipe; a Qwen3.5 GDN model
    # has no QSA layers.
    with pytest.raises(ValueError, match="serves QSA sparse-attention models"):
        _recipe(enabled=True, recipe_class=QwenGDNRecipe).qsa_bf16_scratch_plan
    assert (
        _recipe(enabled=False, recipe_class=QwenGDNRecipe).qsa_bf16_scratch_plan is None
    )


# ---------------------------------------------------------------------------
# The attention leaf, through the real router and composite backend
# ---------------------------------------------------------------------------

# The attention stand-in computes on a GPU if present; the rest stays on CPU.
_STAND_IN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    # Dequantize the whole cache once, not every gathered chunk: converting the
    # gathered FP8 rows took over half of this helper's CPU time.
    k_cache = (k_cache.float() * k_scale).to(_STAND_IN_DEVICE)
    v_cache = (v_cache.float() * v_scale).to(_STAND_IN_DEVICE)
    selected_slots = selected_slots.to(_STAND_IN_DEVICE)
    result = torch.empty_like(q, device=_STAND_IN_DEVICE)
    for start in range(0, rows, 32):
        slots = selected_slots[start : start + 32].long()
        keys = k_cache[slots.clamp_min(0)]
        values = v_cache[slots.clamp_min(0)]
        query = q[start : start + 32].to(_STAND_IN_DEVICE).float()
        query = query.view(-1, kv_heads, heads // kv_heads, dim)
        scores = torch.einsum("ckgd,cnkd->ckgn", query, keys) * scale
        scores = scores.masked_fill(~(slots > 0)[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        chunk = torch.einsum("ckgn,cnkd->ckgd", weights, values)
        result[start : start + 32] = chunk.reshape(-1, heads, dim).to(q.dtype)
    if out is None:
        return result.to(q.device)
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


def _root(config, backend_name="qsa"):
    """Composite backend over a real router; the pool is a stand-in."""
    router = attention_registry.create_paged_router(
        config, AttentionArch.MHA, backend_name=backend_name
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


def test_full_attention_without_a_qsa_leaf_refuses_the_scratch(kernels):
    # Serving the QSA layers through another backend leaves no leaf to own it.
    root, router = _root(_config(), backend_name="mha")
    assert router.leaves and not any(
        isinstance(leaf, qsa_leaf.QSAAttnBackend) for leaf in router.leaves.values()
    )
    with pytest.raises(RuntimeError, match="needs one QSA attention leaf, found 0"):
        root.preallocate_qsa_bf16_scratch(_plan())


@pytest.mark.skipif(
    _cute_qsa_available(), reason="this platform has the CuTe DSL QSA kernel"
)
def test_a_platform_without_the_cute_kernel_refuses_the_scratch():
    with pytest.raises(ValueError, match="needs the CuTe DSL QSA sparse-attention"):
        QSABf16PrefillScratch(_plan(), device="cpu")


class _Forward:
    """One forward through the real router and the QSA leaf.

    ``requests`` are ``(cached prefix, new tokens)``; a MIXED forward's
    ``num_extends`` leading requests extend and the others decode one token.
    """

    def __init__(
        self,
        router,
        requests,
        *,
        pool_pages=96,
        seed=0,
        mode=ForwardMode.EXTEND,
        num_extends=None,
    ):
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
        if mode.is_decode():
            router.refresh_decode_metadata(
                bs,
                bs,
                torch.arange(1, bs + 1, dtype=torch.int32),
                self.seq_lens,
                forward_mode=mode,
                block_tables={group: blocks for group in _GROUPS},
                num_extends=0,
                for_graph_replay=False,
            )
        else:
            # Only extend requests have host lengths.
            extends = bs if mode.is_extend() else num_extends
            prefix_lens = self.seq_lens - new_tokens
            router.init_forward_metadata(
                bs,
                extends,
                torch.arange(1, bs + 1, dtype=torch.int32),
                self.seq_lens,
                mode,
                block_tables={group: blocks for group in _GROUPS},
                extend_seq_lens=new_tokens[:extends],
                extend_seq_lens_cpu=new_tokens[:extends],
                extend_prefix_lens=prefix_lens[:extends],
                extend_prefix_lens_cpu=prefix_lens[:extends],
                extend_replay_lens_cpu=torch.zeros(extends, dtype=torch.int32),
                extend_prompt_lens_cpu=self.seq_lens[:extends],
                extend_with_prefix=True,
                query_shard=None,
                block_tables_cpu={},
            )
        self.mode = mode
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
    "requests,mode",
    [
        # Fewer rows than MIN_ROWS.
        ([(2000, MIN_ROWS - 1)], ForwardMode.EXTEND),
        # More rows than the plan's chunk.
        ([(0, 4096), (0, 1)], ForwardMode.EXTEND),
        # Decode keeps the FP8 path.
        ([(3000, 1)] * 4, ForwardMode.DECODE),
    ],
)
def test_other_calls_keep_the_fp8_path(kernels, requests, mode):
    root, router = _root(_config())
    root.preallocate_qsa_bf16_scratch(_plan())
    forward = _Forward(router, requests, seed=3, mode=mode)
    output = forward.run(root)
    assert kernels.copy.count == 0 and kernels.fp8_path.count == 1
    if not mode.is_decode():
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


def test_mixed_batches_keep_the_fp8_path(kernels):
    root, router = _root(_config())
    root.preallocate_qsa_bf16_scratch(_plan())
    # The prefill's 2500 rows would qualify alone, but the decode request in
    # the same forward has no host lengths to size a copy by.
    mixed = _Forward(
        router, [(100, 2500), (900, 1)], seed=6, mode=ForwardMode.MIXED, num_extends=1
    )
    output = mixed.run(root)
    assert kernels.copy.count == 0 and kernels.fp8_path.count == 1
    assert torch.equal(output, mixed.fp8_reference())
    # The next pure extend sizes its copy by its own lengths again.
    forward = _Forward(router, [(100, 2500), (0, 600)], seed=7)
    output = forward.run(root)
    assert kernels.copy.count == 1 and kernels.fp8_path.count == 1
    assert torch.equal(output, forward.fp8_reference())


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
# GPU: the scratch against the FP8 prefill path it replaces and the BF16
# kernel over the dequantized cache
# ---------------------------------------------------------------------------


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
    """A prefill call over an FP8 cache with real QSA selections.

    ``heads`` are the query and KV heads per rank; request 1 starts with
    request 0's first ``shared`` pages, as after a prefix-cache hit.
    """

    def __init__(
        self, requests, *, page_size, pool_pages, seed, heads=(24, 2), shared=0
    ):
        from tokenspeed_kernel.ops.attention.qsa.triton import (
            qwen4_exp_qsa_selected_slots,
        )

        device = "cuda"
        q_heads, kv_heads = heads
        self.page_size = page_size
        generator = torch.Generator(device=device).manual_seed(seed)
        lengths = [prefix + new for prefix, new in requests]
        widths = [-(-length // page_size) for length in lengths]
        values = torch.rand(
            (2, pool_pages * page_size, kv_heads, 256),
            generator=generator,
            device=device,
        )
        self.k, self.v = (values * 8 - 4).to(FP8).unbind(0)
        free = torch.randperm(pool_pages - 1, generator=generator, device=device) + 1
        table = torch.zeros(
            (len(requests), max(widths) + 2), dtype=torch.int32, device=device
        )
        start = 0
        for row, width in enumerate(widths):
            share = min(shared, width, widths[0]) if row == 1 else 0
            table[row, :share] = table[0, :share]
            fresh = width - share
            table[row, share:width] = free[start : start + fresh].to(torch.int32)
            start += fresh
        self.table = table
        self.seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
        positions = torch.cat(
            [torch.arange(p, p + n, device=device) for p, n in requests]
        )
        owners = torch.repeat_interleave(
            torch.arange(len(requests), dtype=torch.int32, device=device),
            torch.tensor([n for _, n in requests], device=device),
        )
        self.positions, self.owners = positions, owners
        offsets = [0]
        for _, new in requests:
            offsets.append(offsets[-1] + new)
        self.offsets = tuple(offsets)
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
        self.selected_blocks = selected_blocks
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
        # Every non-positive flavour of padding, where the selection pads, so
        # the slots still match their logical blocks.
        for row, value in ((0, 0), (-1, -12345)):
            padded = (self.selected[row] == -1).nonzero()
            if padded.numel():
                self.selected[row, int(padded[0, 0])] = value
        self.q = torch.randn(
            (positions.numel(), q_heads, 256), generator=generator, device=device
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

    def fp8_prefill(self, override):
        """The call through the FP8 prefill path the scratch replaces, as the
        QSA leaf makes it: ragged rows, logical metadata, unit scales; a
        kernel ``override`` stands in for a deployment's override."""
        from tokenspeed_kernel.ops.attention.qsa import (
            QSASparseMetadata,
            QSAWorkspace,
            qsa_sparse_attention,
        )

        return qsa_sparse_attention(
            self.q,
            self.k,
            self.v,
            self.selected,
            scale=0.0625,
            max_seqlen_q=None,
            metadata_capacity_rows=None,
            k_scale=1.0,
            v_scale=1.0,
            override=override,
            solution=None,
            sparse_metadata=QSASparseMetadata(
                selected_blocks=self.selected_blocks,
                block_table=self.table,
                token_to_request=self.owners,
                query_positions=self.positions,
                query_start_loc=self.offsets,
                page_size=self.page_size,
                block_size=4,
                workspace=QSAWorkspace(),
                seq_lens=self.seq_lens,
            ),
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


def _gpu_plan(heads=(24, 2)):
    # 1 + ceil((8192 + 4096) / 64) + 64 = 257 pages of 64 tokens.
    q_heads, kv_heads = heads
    return plan_qsa_bf16_scratch(
        context_len=8192,
        max_rows=4096,
        page_size=64,
        kv_heads=kv_heads,
        head_dim=256,
        q_heads=q_heads,
    )


# Several requests per chunk, rows at request starts, positions below 2048 and
# a long prefix.
_SEVERAL_REQUESTS = [(0, 700), (3000, 900), (17, 500), (1200, 300)]
# Query and KV heads per rank at attention TP 1, 2 and 4: each runs its own
# CuTe kernel configuration.
_HEADS = [(24, 2), (12, 1), (6, 1)]


@requires_cute_qsa
# The default FP8 prefill kernel (FlashInfer FA2), and the CuTe kernel's FP8
# path a kernel override selects for prefill too.
@pytest.mark.parametrize("override", [None, "cute_dsl_blackwell_qsa_sparse_attention"])
@pytest.mark.parametrize("heads", _HEADS)
def test_scratch_matches_the_fp8_prefill_path(heads, override):
    scratch = QSABf16PrefillScratch(_gpu_plan(heads), device="cuda")
    call = _GpuCall(
        _SEVERAL_REQUESTS, page_size=64, pool_pages=320, seed=11, heads=heads
    )
    output = call.scratch(scratch)
    assert output is not None
    # The scratch replaces this path for the call; both kernels read the same
    # K/V values and differ only in rounding.
    torch.testing.assert_close(output, call.fp8_prefill(override), atol=1e-2, rtol=1e-2)


@requires_cute_qsa
@pytest.mark.parametrize(
    "requests,heads,shared",
    [
        *((_SEVERAL_REQUESTS, heads, 0) for heads in _HEADS),
        # One request whose rows all sit below 2048 (mostly padding).
        ([(0, 2048)], (24, 2), 0),
        # Exactly at the capacity: 256 context pages plus page 0.
        ([(4096 - 512, 512)] * 4, (24, 2), 0),
        # Two requests share their first 20 pages: the copy writes each of
        # them twice, and either copy may serve the slots.
        ([(3000, 900), (1500, 1300)], (24, 2), 20),
        ([(3000, 900), (1500, 1300)], (6, 1), 20),
    ],
)
def test_scratch_matches_the_bf16_kernel_on_the_dequantized_cache(
    requests, heads, shared
):
    scratch = QSABf16PrefillScratch(_gpu_plan(heads), device="cuda")
    call = _GpuCall(
        requests,
        page_size=64,
        pool_pages=320,
        seed=len(requests),
        heads=heads,
        shared=shared,
    )
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
