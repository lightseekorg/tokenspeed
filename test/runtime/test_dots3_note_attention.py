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

"""Scoped attention checks; no model weights or scheduler lifecycle claims.

Normally run with pytest. On a machine missing unrelated native dependencies,
run this file in a fresh process with --isolated-imports: it skips the listed
package initializers/registrations, not the runtime code or Triton kernels under
test. That mode does NOT validate normal package imports or native dispatch.
GPU selection belongs to the caller. Packed target verify tests replace only the
attention config; they do not enable serving speculation or establish native MTP
concat/tap/position/window semantics.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def runtime():
    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.paged import dsa, mla
    from tokenspeed.runtime.layers.attention.configs.base import SoftmaxAttnConfig
    from tokenspeed.runtime.layers.attention.configs.dots3_note import (
        Dots3NoteAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
    from tokenspeed.runtime.layers.attention.registry import (
        _create_attn_backend,
        _create_attn_config,
        _resolve_attn_side,
        _resolve_cache_family,
    )

    return SimpleNamespace(**locals())


def _inputs(*, device: str):
    full_layers = {0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45}
    hf = SimpleNamespace(
        model_type="dots3_note",
        num_hidden_layers=46,
        layer_types=[
            "full_attention" if i in full_layers else "sliding_attention"
            for i in range(46)
        ],
        num_attention_heads=128,
        num_key_value_heads=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        swa_num_attention_heads=64,
        swa_num_key_value_heads=64,
        swa_kv_lora_rank=1024,
        swa_qk_nope_head_dim=192,
        swa_qk_rope_head_dim=64,
        swa_v_head_dim=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
        sliding_window_size=513,
        swa_rope_theta=50000,
    )
    model = SimpleNamespace(
        model_profile=None,
        hf_config=SimpleNamespace(
            architectures=["Dot3NoteForCausalLM"], text_config=hf
        ),
        hf_text_config=hf,
        dtype=torch.bfloat16,
        context_len=640,
        num_attention_layers=46,
    )
    args = SimpleNamespace(
        device=device,
        attention_backend=None,
        drafter_attention_backend=None,
        model="dots3-test-checkpoint",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        speculative_eagle_topk=1,
        kv_cache_dtype="auto",
        kv_cache_quant_method="none",
        speculative_algorithm=None,
        speculative_draft_model_path=None,
        pipeline_parallel_size=1,
        attn_tp_size=8,
        data_parallel_size=1,
        max_num_seqs=4,
        prefix_granularity=64,
        spec_context_pad=0,
        disaggregation_mode="null",
        mapping=SimpleNamespace(
            attn=SimpleNamespace(
                tp_size=8,
                dp_size=1,
                dcp_size=1,
                dcp_rank=0,
                dcp_group=(0,),
                qcp_size=1,
                qcp_rank=0,
                qcp_group=(0,),
            )
        ),
    )
    return args, model


def _config(runtime, *, device: str):
    args, model = _inputs(device=device)
    model.attention_arch = runtime.AttentionArch.DSA
    return runtime._create_attn_config(args, model, is_draft=False)


def _pool(runtime):
    specs = tuple(
        runtime.CacheGroupSpec(
            group_id=gid,
            retention="full_history" if gid == "full" else "sliding_window",
            rows_per_page=64 if gid == "full" else 32,
            entry_stride_tokens=1,
            sliding_window_tokens=None if gid == "full" else 513,
            replayable=False,
        )
        for gid in ("full", "swa.0", "swa.1", "swa.2")
    )
    return SimpleNamespace(
        arena=SimpleNamespace(cache_group_specs=specs),
        paged_group_ids=tuple(spec.group_id for spec in specs),
    )


def _router(runtime, *, device: str, q_len_per_req: int):
    config = replace(
        _config(runtime, device=device), speculative_num_draft_tokens=q_len_per_req
    )
    router = runtime._create_attn_backend(runtime.AttentionArch.DSA, config)
    router.set_cache_pool(_pool(runtime))
    router.init_cuda_graph_state(4)
    return router


def _layer(spec, *, group_id: str, absorbed: bool):
    return SimpleNamespace(
        group_id=group_id,
        layer_id=0 if group_id == "full" else 2,
        tp_q_head_num=spec.num_attention_heads // spec.attn_tp_size,
        tp_k_head_num=spec.num_attention_heads // spec.attn_tp_size,
        tp_v_head_num=spec.num_attention_heads // spec.attn_tp_size,
        head_dim=spec.kv_cache_dim if absorbed else spec.head_dim,
        qk_head_dim=spec.head_dim,
        v_head_dim=spec.kv_lora_rank if absorbed else spec.v_head_dim,
        scaling=spec.scaling,
        sliding_window_size=-1 if group_id == "full" else 512,
        logit_cap=0.0,
        k_scale_float=None,
    )


def test_composite_config_and_family(runtime):
    args, model = _inputs(device="cpu")
    model.attention_arch = runtime.AttentionArch.DSA
    config = runtime._create_attn_config(args, model, is_draft=False)
    spec = config.component(runtime.SoftmaxAttnConfig)
    assert isinstance(spec, runtime.Dots3NoteAttnConfig)
    assert config.components == (spec,)
    assert spec.cache_layer_types == tuple(model.hf_text_config.layer_types)
    assert spec.sliding_window_tokens == 513
    assert (spec.full.kv_cache_dim, spec.swa.kv_cache_dim) == (576, 1088)
    assert (spec.full.v_head_dim, spec.swa.v_head_dim) == (128, 128)
    assert config.kv_cache_dtype == torch.bfloat16
    assert config.kernel_page_size is None
    assert config.cache_cell_size() == 2176
    profile = runtime._resolve_attn_side(model, None)
    assert runtime._resolve_cache_family(profile, config) == "dots3_note"
    with pytest.raises(ValueError, match="exactly one"):
        replace(config, components=(spec.full, spec.swa))
    # Dispatch by model_type also works without the architecture hint.
    model.hf_config = model.hf_text_config
    assert isinstance(
        runtime._create_attn_config(args, model, is_draft=False).components[0],
        runtime.Dots3NoteAttnConfig,
    )


@pytest.mark.parametrize(
    "change, error",
    [
        ({"kv_cache_dtype": "fp8"}, "BF16"),
        ({"kv_cache_quant_method": "per_token_head"}, "BF16"),
        ({"speculative_algorithm": "EAGLE3"}, "MTP"),
        ({"speculative_draft_model_path": "draft"}, "draft"),
        ({"pipeline_parallel_size": 2}, "pipeline"),
        ({"attention_backend": "mla"}, "attention backend"),
        ({"attn_tp_size": 3}, "TP size"),
    ],
)
def test_unsupported_serving_choices_fail(runtime, change, error):
    args, model = _inputs(device="cpu")
    args = SimpleNamespace(**(vars(args) | change))
    with pytest.raises((ValueError, NotImplementedError), match=error):
        runtime.Dots3NoteAttnConfig.generate(args, model, is_draft=False)


def test_unsupported_checkpoint_and_draft_fail(runtime):
    args, model = _inputs(device="cpu")
    with pytest.raises(NotImplementedError, match="draft"):
        runtime.Dots3NoteAttnConfig.generate(args, model, is_draft=True)
    with pytest.raises(NotImplementedError, match="draft"):
        runtime._create_attn_backend(
            runtime.AttentionArch.DSA,
            replace(_config(runtime, device="cpu"), is_draft=True),
        )
    model.hf_text_config.swa_kv_lora_rank = 512
    with pytest.raises(ValueError, match="swa_kv_lora_rank"):
        runtime.Dots3NoteAttnConfig.generate(args, model, is_draft=False)
    model.hf_text_config.layer_types.pop()
    with pytest.raises(ValueError, match="46 layers"):
        runtime.Dots3NoteAttnConfig.generate(args, model, is_draft=False)


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("backend", [None, "dots3_note", "triton"])
def test_native_mtp_resolves_to_ordinary_swa_mla(runtime, width, backend):
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.layers.attention.registry import (
        _resolve_heterogeneous_draft_family,
    )

    args, model = _inputs(device="cpu")
    args.speculative_algorithm = "MTP"
    args.speculative_num_steps = width - 1
    args.speculative_num_draft_tokens = width
    args.drafter_attention_backend = backend
    args.speculative_draft_model_path = args.model
    model.attention_arch = runtime.AttentionArch.DSA
    target = runtime._create_attn_config(args, model, is_draft=False)
    draft_model = SimpleNamespace(
        **(
            vars(model)
            | {"num_attention_layers": 1, "attention_arch": runtime.AttentionArch.MLA}
        )
    )
    draft = runtime._create_attn_config(args, draft_model, is_draft=True)
    assert draft.components == (
        replace(target.components[0].swa, cache_layer_types=("sliding_attention",)),
    )
    assert type(draft.components[0]) is MLAConfig
    assert draft.is_draft and not draft.draft_block_decode
    assert draft.kernel_page_size == 32
    assert (
        draft.speculative_num_draft_tokens
        == target.speculative_num_draft_tokens
        == width
    )
    assert draft.speculative_num_steps == width - 1
    assert draft_model.hf_text_config is model.hf_text_config
    assert (
        len(model.hf_text_config.layer_types)
        == model.hf_text_config.num_hidden_layers
        == 46
    )
    assert args.drafter_attention_backend == backend
    profile = runtime._resolve_attn_side(draft_model, backend)
    family = runtime._resolve_cache_family(profile, draft)
    assert (
        family
        == _resolve_heterogeneous_draft_family(
            "dots3_note", family, draft_family_declared=False
        )
        == "mla"
    )
    for unsupported in ("mha", "dsa", "msa"):
        with pytest.raises(RuntimeError, match="ordinary MLA draft"):
            _resolve_heterogeneous_draft_family(
                "dots3_note", unsupported, draft_family_declared=False
            )


@pytest.mark.parametrize(
    "changes",
    [
        {"speculative_algorithm": "EAGLE3"},
        {"speculative_algorithm": "DFLASH"},
        {"speculative_algorithm": "DSPARK"},
        {"speculative_num_steps": 0},
        {"speculative_num_draft_tokens": 2},
        {"speculative_eagle_topk": 2},
        {"drafter_attention_backend": "flashmla"},
        {"speculative_draft_model_path": "another-checkpoint"},
    ],
)
def test_mtp_rejects_unsupported_explicit_arguments(runtime, changes):
    args, model = _inputs(device="cpu")
    args = SimpleNamespace(**(vars(args) | {"speculative_algorithm": "MTP"} | changes))
    model.num_attention_layers = 1
    with pytest.raises((ValueError, NotImplementedError), match="dots3"):
        runtime.Dots3NoteAttnConfig.generate(args, model, is_draft=True)


@pytest.mark.parametrize("parallelism", ["qcp_size", "dcp_size"])
def test_dots3_rejects_context_parallelism(runtime, parallelism):
    args, model = _inputs(device="cpu")
    args.mapping.attn = SimpleNamespace(**(vars(args.mapping.attn) | {parallelism: 2}))
    with pytest.raises(NotImplementedError, match="CP"):
        runtime.Dots3NoteAttnConfig.generate(args, model, is_draft=False)


@pytest.mark.parametrize("q_len_per_req", [1, 2, 4])
def test_group_leaves_and_sparse_routing(runtime, q_len_per_req):
    router = _router(runtime, device="cpu", q_len_per_req=q_len_per_req)
    full = router.leaf_for(SimpleNamespace(group_id="full"))
    assert isinstance(full, runtime.dsa.DSABackend)
    assert isinstance(full._dense_backend, runtime.mla.MLAAttnBackend)
    assert full.kernel_page_size == 64
    assert full.kernel_solution == full._dense_backend.kernel_solution == "triton"
    assert full.num_local_heads == 16
    for gid in ("swa.0", "swa.1", "swa.2"):
        leaf = router.leaf_for(SimpleNamespace(group_id=gid))
        assert isinstance(leaf, runtime.mla.MLAAttnBackend)
        assert leaf.kernel_page_size == 32
        assert leaf.kv_lora_rank == 1024
        assert leaf.num_local_heads == 8
        assert leaf.kernel_solution == "triton"
    layer = SimpleNamespace(group_id="full", layer_id=0)
    full.forward_sparse_prefill = Mock(return_value="full-result")
    marker = object()
    assert router.forward_sparse_prefill(layer=layer, q=marker) == "full-result"
    full.forward_sparse_prefill.assert_called_once_with(layer=layer, q=marker)
    with pytest.raises(RuntimeError, match="single attention cache group"):
        _ = router.forward_decode_metadata
    with pytest.raises(KeyError, match="names cache group"):
        router.leaf_for(SimpleNamespace(group_id="unknown", layer_id=0))


@pytest.mark.parametrize("q_len_per_req", [1, 2, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_refresh_padding_write_locations_and_rebind(runtime, device, q_len_per_req):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    router = _router(runtime, device=device, q_len_per_req=q_len_per_req)
    tables = {
        gid: torch.full(
            (4, leaf.max_num_pages), i + 2, dtype=torch.int32, device=device
        )
        for i, (gid, leaf) in enumerate(router.leaves.items())
    }
    reqs = torch.arange(4, dtype=torch.int64, device=device)
    lengths = torch.tensor([65, 33, 1, 1], dtype=torch.int32, device=device)
    pointers = {}
    for actual_bs, replay in ((2, False), (1, True), (0, True), (2, False)):
        router.refresh_decode_metadata(
            4,
            actual_bs,
            reqs,
            lengths,
            forward_mode=runtime.ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            for_graph_replay=replay,
        )
        for gid, leaf in router.leaves.items():
            metadata = leaf.forward_decode_metadata
            assert metadata.q_len_per_req == q_len_per_req
            torch.testing.assert_close(
                metadata.seq_lens, lengths.clamp_min(q_len_per_req)
            )
            locations = router.write_locations(
                SimpleNamespace(group_id=gid), runtime.ForwardMode.DECODE
            )
            addresses = (
                metadata.page_table.data_ptr(),
                metadata.seq_lens.data_ptr(),
                locations.data_ptr(),
            )
            assert pointers.setdefault(gid, addresses) == addresses
            assert metadata.page_table[actual_bs:].count_nonzero() == 0
            assert locations.shape == (4 * q_len_per_req,)
            assert locations[actual_bs * q_len_per_req :].count_nonzero() == 0
            if actual_bs:
                positions = (
                    lengths[:actual_bs, None]
                    - q_len_per_req
                    + torch.arange(q_len_per_req, device=device)
                )
                pages = tables[gid][
                    torch.arange(actual_bs, device=device)[:, None],
                    positions.long() // leaf.kernel_page_size,
                ]
                expected = (
                    pages * leaf.kernel_page_size + positions % leaf.kernel_page_size
                )
                torch.testing.assert_close(
                    locations[: actual_bs * q_len_per_req],
                    expected.flatten().to(locations.dtype),
                )
    router.set_cache_pool(_pool(runtime))
    assert router.decode_write_locations is None
    for leaf in router.leaves.values():
        assert leaf.forward_decode_metadata is None
    router.init_cuda_graph_state(4)
    bad = _pool(runtime)
    bad.arena.cache_group_specs = (
        replace(bad.arena.cache_group_specs[0], rows_per_page=32),
        *bad.arena.cache_group_specs[1:],
    )
    old_pool = router.cache_pool
    with pytest.raises(RuntimeError, match="different geometry"):
        router.set_cache_pool(bad)
    assert router.cache_pool is old_pool


@pytest.mark.parametrize("q_len_per_req", [1, 2, 4])
@pytest.mark.parametrize(
    "group_id, page_stride", [("full", 534400), ("swa.0", 534400), ("swa.2", 106880)]
)
def test_paged_cache_reaches_kernel_without_copy(
    runtime, monkeypatch, group_id, page_stride, q_len_per_req
):
    config = _config(runtime, device="cpu")
    spec = config.components[0].full if group_id == "full" else config.components[0].swa
    leaf = _router(runtime, device="cpu", q_len_per_req=q_len_per_req).leaf_for(
        SimpleNamespace(group_id=group_id)
    )
    page_size = leaf.kernel_page_size
    cache = torch.empty_strided(
        (3, page_size, 1, spec.kv_cache_dim),
        (page_stride, spec.kv_cache_dim, spec.kv_cache_dim, 1),
        dtype=torch.bfloat16,
    )
    pool = SimpleNamespace(get_key_buffer=lambda layer_id: cache, quant_method="none")
    layer = _layer(spec, group_id=group_id, absorbed=True)
    leaf.refresh_decode_metadata(
        1,
        1,
        torch.tensor([1], dtype=torch.int32),
        torch.ones((1, leaf.max_num_pages), dtype=torch.int32),
        num_extends=0,
        for_graph_replay=False,
    )
    seen = []

    def kernel(**kwargs):
        assert kwargs["kv_cache"] is cache
        assert kwargs["solution"] == "triton"
        seen.append(kwargs)
        return torch.zeros(
            (kwargs["q"].shape[0], layer.tp_q_head_num, spec.kv_lora_rank),
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(runtime.mla, "mla_decode_with_kvcache", kernel)
    monkeypatch.setattr(runtime.dsa, "dsa_decode", kernel)
    kwargs = (
        {"topk_indices": torch.zeros((q_len_per_req, 2048), dtype=torch.int32)}
        if group_id == "full"
        else {}
    )
    leaf.forward_decode(
        torch.zeros(
            (q_len_per_req, layer.tp_q_head_num, spec.kv_cache_dim),
            dtype=torch.bfloat16,
        ),
        None,
        None,
        layer,
        torch.zeros(q_len_per_req, dtype=torch.int32),
        pool,
        1,
        save_kv_cache=False,
        **kwargs,
    )
    assert len(seen) == 1
    if group_id == "full":
        monkeypatch.setattr(runtime.dsa, "dsa_prefill", kernel)

        leaf.forward_sparse_prefill(
            q=torch.zeros(
                (1, layer.tp_q_head_num, spec.kv_cache_dim), dtype=torch.bfloat16
            ),
            layer=layer,
            token_to_kv_pool=pool,
            kv_seq_lens=torch.ones(1, dtype=torch.int32),
            topk_slots=torch.zeros((1, 2048), dtype=torch.int32),
            topk_lens=torch.ones(1, dtype=torch.int32),
            max_seq_len=1,
        )
        assert len(seen) == 2


def test_ordinary_mla_still_uses_automatic_solution(runtime):
    config = _config(runtime, device="cpu")
    spec = replace(
        config.components[0].swa, backend_name="mla", sliding_window_tokens=None
    )
    config = replace(config, components=(spec,))
    router = runtime._create_attn_backend(runtime.AttentionArch.MLA, config)
    pool = _pool(runtime)
    pool.arena.cache_group_specs = pool.arena.cache_group_specs[:1]
    pool.paged_group_ids = ("full",)
    router.set_cache_pool(pool)
    leaf = router.leaf_for(SimpleNamespace(group_id="full"))
    assert leaf.kernel_solution is None
    assert leaf.kernel_page_size == runtime.mla.MLAAttnBackend.default_kernel_page_size
    assert router.forward_prefill_metadata is leaf.forward_prefill_metadata
    assert router._leaf_for(SimpleNamespace(group_id="full")) is leaf


def test_mixed_prefill_metadata_keeps_each_groups_pages(runtime, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 128)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 1)
    router = _router(runtime, device="cuda", q_len_per_req=1)
    tables = {
        gid: torch.full(
            (3, leaf.max_num_pages), i + 2, dtype=torch.int32, device="cuda"
        )
        for i, (gid, leaf) in enumerate(router.leaves.items())
    }
    prefix_cpu = torch.tensor([65, 31], dtype=torch.int32)
    new_cpu = torch.tensor([2, 3], dtype=torch.int32)
    router.init_forward_metadata(
        3,
        2,
        torch.arange(3, device="cuda"),
        torch.tensor([67, 34, 33], dtype=torch.int32, device="cuda"),
        runtime.ForwardMode.MIXED,
        block_tables=tables,
        block_tables_cpu={gid: table.cpu() for gid, table in tables.items()},
        query_shard=None,
        extend_seq_lens=new_cpu.cuda(),
        extend_seq_lens_cpu=new_cpu,
        extend_prefix_lens=prefix_cpu.cuda(),
        extend_prefix_lens_cpu=prefix_cpu,
        extend_replay_lens_cpu=torch.zeros_like(new_cpu),
        extend_prompt_lens_cpu=prefix_cpu + new_cpu,
        extend_with_prefix=True,
    )
    for gid, leaf in router.leaves.items():
        meta = leaf.forward_prefill_metadata
        assert meta is leaf.chunked_prefill_metadata
        torch.testing.assert_close(meta.page_table, tables[gid][:2])
        assert leaf.forward_decode_metadata.num_extends == 2
        locations = router.write_locations(
            SimpleNamespace(group_id=gid), runtime.ForwardMode.EXTEND
        )
        positions = torch.tensor([65, 66, 31, 32, 33], device="cuda")
        expected = (
            tables[gid][0, 0] * leaf.kernel_page_size
            + positions % leaf.kernel_page_size
        )
        torch.testing.assert_close(locations, expected.to(locations.dtype))


@pytest.mark.parametrize("q_len_per_req", [1, 2, 4])
@pytest.mark.parametrize("group_id", ["full", "swa.0", "swa.2"])
def test_target_packed_verify_numeric(runtime, group_id, q_len_per_req):
    """Real target leaves/indexer, not a serving or native MTP implementation."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tokenspeed_kernel.ops.attention.dsa import dsa_decode_topk
    from tokenspeed_kernel.ops.attention.dsa.triton import _flatten_dense_kv_cache
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter

    from tokenspeed.runtime.models.dots3_note import quantize_index_rows

    torch.manual_seed(23)
    config = replace(
        _config(runtime, device="cuda"),
        speculative_num_draft_tokens=q_len_per_req,
        context_len=2176 if group_id == "full" else 640,
    )
    router = runtime._create_attn_backend(runtime.AttentionArch.DSA, config)
    router.set_cache_pool(_pool(runtime))
    router.init_cuda_graph_state(4)
    spec = config.components[0].full if group_id == "full" else config.components[0].swa
    layer = _layer(spec, group_id=group_id, absorbed=True)
    leaf = router.leaf_for(layer)
    page_size = leaf.kernel_page_size
    tables = {
        gid: torch.ones((2, child.max_num_pages), dtype=torch.int32, device="cuda")
        for gid, child in router.leaves.items()
    }
    num_pages = 2 * leaf.max_num_pages + 1
    tables[group_id] = (
        torch.randperm(num_pages - 1, device="cuda", dtype=torch.int32) + 1
    ).view(2, -1)
    lengths = (2113 if group_id == "full" else 546, 65)
    router.refresh_decode_metadata(
        2,
        2,
        torch.arange(2, device="cuda"),
        torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        forward_mode=runtime.ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
        for_graph_replay=False,
    )
    meta = leaf.forward_decode_metadata
    assert meta.q_len_per_req == q_len_per_req
    stride = 106880 if group_id == "swa.2" else 534400
    cache = torch.empty_strided(
        (num_pages, page_size, 1, spec.kv_cache_dim),
        (stride, spec.kv_cache_dim, spec.kv_cache_dim, 1),
        dtype=torch.bfloat16,
        device="cuda",
    ).normal_()
    assert not cache.is_contiguous()
    if group_id == "full":
        dense = _flatten_dense_kv_cache(cache)
        assert dense.data_ptr() == cache.data_ptr()
        assert dense.stride(0) == stride
    pool = SimpleNamespace(get_key_buffer=lambda layer_id: cache, quant_method="none")
    q = torch.randn(
        (2 * q_len_per_req, layer.tp_q_head_num, spec.kv_cache_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    # Poison later packed queries and unused capacity, not the first query's
    # causal span. Distinct values expose leakage at each packed query boundary.
    future_rows = []
    for req, length in enumerate(lengths):
        first_end = length - q_len_per_req + 1
        positions = torch.arange(
            first_end, leaf.max_num_pages * page_size, device="cuda"
        )
        slots = (
            tables[group_id][req, positions // page_size].long() * page_size
            + positions % page_size
        )
        poison = ((positions - first_end + 1).clamp_max(q_len_per_req) * 16).bfloat16()
        cache[slots // page_size, slots % page_size, 0] = poison[:, None]
        future_rows.append((slots, poison))
    kwargs = {}
    if group_id == "full":
        index_cache = torch.empty_strided(
            (num_pages, page_size * 132), (1068800, 1), dtype=torch.uint8, device="cuda"
        )
        index_rows = torch.randn((num_pages * page_size, 128), device="cuda").bfloat16()
        for slots, poison in future_rows:
            index_rows[slots] = poison[:, None]
        index_k, scales = quantize_index_rows(index_rows)
        index_k_block_split_scatter(
            index_cache,
            index_k,
            scales,
            torch.arange(num_pages * page_size, device="cuda"),
            page_size=page_size,
            head_dim=128,
            group_size=128,
            write_mask=None,
        )
        dequant = index_k.float() * scales[:, None]
        index_q = torch.randn((2 * q_len_per_req, 64, 128), device="cuda").bfloat16()
        weights = torch.rand((2 * q_len_per_req, 64), device="cuda")
        selected, topk_lens = dsa_decode_topk(
            index_q,
            weights,
            meta.seq_lens,
            meta.page_table,
            page_size=page_size,
            topk=2048,
            softmax_scale=128**-0.5,
            q_len_per_req=meta.q_len_per_req,
            batch_invariant=leaf.batch_invariant,
            slot_order=leaf.slot_order,
            index_k_cache=index_cache,
            solution="triton",
        )
        kwargs = {"topk_indices": selected, "topk_lens": topk_lens}
    actual = router.forward(
        q,
        None,
        None,
        layer,
        pool,
        runtime.ForwardMode.DECODE,
        2,
        save_kv_cache=False,
        **kwargs,
    ).view(2 * q_len_per_req, layer.tp_q_head_num, spec.kv_lora_rank)
    for req, length in enumerate(lengths):
        for j in range(q_len_per_req):
            row = req * q_len_per_req + j
            end = length - q_len_per_req + j + 1
            start = 0 if group_id == "full" else max(0, end - 513)
            positions = torch.arange(start, end, device="cuda")
            slots = (
                tables[group_id][req, positions // page_size].long() * page_size
                + positions % page_size
            )
            if group_id == "full":
                count = min(end, 2048)
                assert topk_lens[row].item() == count
                scores = (
                    (index_q[row].float() @ dequant[slots].T).relu()
                    * weights[row, :, None]
                ).sum(0) * 128**-0.5
                slots = slots[scores.topk(count).indices]
                torch.testing.assert_close(
                    selected[row, :count].long().sort().values,
                    slots.sort().values,
                    atol=0,
                    rtol=0,
                )
                assert torch.all(selected[row, count:] == -1)
            visible = cache[slots // page_size, slots % page_size, 0].float()
            expected = (q[row].float() @ visible.T * spec.scaling).softmax(
                -1
            ) @ visible[:, : spec.kv_lora_rank]
            torch.testing.assert_close(
                actual[row].float(), expected, atol=1e-3, rtol=5e-3
            )


@pytest.mark.parametrize("q_len_per_req", [1, 2, 4])
def test_decode_graph_uses_refreshed_group_tables(runtime, q_len_per_req):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tokenspeed_kernel.ops.attention.dsa import dsa_decode_topk

    config = _config(runtime, device="cuda")
    router = _router(runtime, device="cuda", q_len_per_req=q_len_per_req)
    tables = {
        gid: torch.ones((4, leaf.max_num_pages), dtype=torch.int32, device="cuda")
        for gid, leaf in router.leaves.items()
    }
    lengths = torch.tensor([65, 33, 1, 1], dtype=torch.int32, device="cuda")
    reqs = torch.arange(4, dtype=torch.int64, device="cuda")
    inputs = {}
    for gid, leaf in router.leaves.items():
        spec = config.components[0].full if gid == "full" else config.components[0].swa
        layer = _layer(spec, group_id=gid, absorbed=True)
        stride = 106880 if gid == "swa.2" else 534400
        cache = torch.empty_strided(
            (4, leaf.kernel_page_size, 1, spec.kv_cache_dim),
            (stride, spec.kv_cache_dim, spec.kv_cache_dim, 1),
            dtype=torch.bfloat16,
            device="cuda",
        )
        cache.copy_(
            torch.arange(4 * leaf.kernel_page_size, device="cuda").view(
                4, leaf.kernel_page_size, 1, 1
            )
            / 64
        )
        cache[0].zero_()
        pool = SimpleNamespace(
            get_key_buffer=lambda layer_id, cache=cache: cache, quant_method="none"
        )
        q = torch.zeros(
            (4 * q_len_per_req, layer.tp_q_head_num, spec.kv_cache_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        inputs[gid] = (layer, pool, q)
    index_q = torch.zeros(
        (4 * q_len_per_req, 64, 128), dtype=torch.bfloat16, device="cuda"
    )
    weights = torch.ones((4 * q_len_per_req, 64), device="cuda")
    index_cache = torch.empty_strided(
        (4, 64 * 132), (1068800, 1), dtype=torch.uint8, device="cuda"
    ).zero_()

    def refresh(actual_bs, replay):
        router.refresh_decode_metadata(
            4,
            actual_bs,
            reqs,
            lengths,
            forward_mode=runtime.ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            for_graph_replay=replay,
        )

    def forward():
        outputs = {}
        for gid, (layer, pool, q) in inputs.items():
            if gid == "full":
                meta = router.leaf_for(layer).forward_decode_metadata
                topk, topk_lens = dsa_decode_topk(
                    index_q,
                    weights,
                    meta.seq_lens,
                    meta.page_table,
                    page_size=64,
                    topk=2048,
                    softmax_scale=128**-0.5,
                    q_len_per_req=meta.q_len_per_req,
                    batch_invariant=router.leaf_for(layer).batch_invariant,
                    slot_order=router.leaf_for(layer).slot_order,
                    index_k_cache=index_cache,
                    solution="triton",
                )
            outputs[gid] = router.forward(
                q,
                None,
                None,
                layer,
                pool,
                runtime.ForwardMode.DECODE,
                4,
                save_kv_cache=False,
                **(
                    {"topk_indices": topk, "topk_lens": topk_lens}
                    if gid == "full"
                    else {}
                ),
            )
        return outputs

    refresh(2, False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    for step, (actual_bs, new_lengths) in enumerate(
        (
            (2, (65, 33, 1, 1)),
            (1, (546, 513, 65, 33)),
            (0, (1, 1, 1, 1)),
            (3, (129, 515, 97, 1)),
            (2, (33, 65, 1, 1)),
        )
    ):
        lengths.copy_(torch.tensor(new_lengths, dtype=torch.int32, device="cuda"))
        for i, table in enumerate(tables.values()):
            table.copy_(
                (torch.arange(table.numel(), device="cuda").view_as(table) + i + step)
                % 3
                + 1
            )
        refresh(actual_bs, True)
        graph.replay()
        for gid, output in captured.items():
            layer, pool, _ = inputs[gid]
            cache = pool.get_key_buffer(layer.layer_id)
            page_size = cache.shape[1]
            for req in range(actual_bs):
                for j in range(q_len_per_req):
                    end = new_lengths[req] - q_len_per_req + j + 1
                    start = 0 if gid == "full" else max(0, end - 513)
                    positions = torch.arange(start, end, device="cuda")
                    values = cache[
                        tables[gid][req, positions // page_size].long(),
                        positions % page_size,
                        0,
                        0,
                    ]
                    row = output[req * q_len_per_req + j]
                    torch.testing.assert_close(
                        row,
                        values.float().mean().bfloat16().expand_as(row),
                        atol=1e-3,
                        rtol=5e-3,
                    )
            assert output[actual_bs * q_len_per_req :].count_nonzero() == 0
        eager = forward()
        for gid in captured:
            torch.testing.assert_close(captured[gid], eager[gid], atol=0, rtol=0)


@pytest.mark.parametrize("window_left", [-1, 0, 512])
def test_prefill_forwards_explicit_window(runtime, monkeypatch, window_left):
    config = _config(runtime, device="cpu")
    spec = config.components[0].swa
    leaf = runtime.mla.MLAAttnBackend(config, spec, kernel_page_size=32)
    layer = _layer(spec, group_id="swa.0", absorbed=False)
    layer.sliding_window_size = window_left
    lengths = torch.tensor([2], dtype=torch.int32)
    cumulative = torch.tensor([0, 2], dtype=torch.int32)
    leaf.forward_prefill_metadata = SimpleNamespace(
        use_absorbed_cached_extend=False,
        max_extend_prefix_len=0,
        cum_extend_seq_lens=cumulative,
        max_extend_seq_len=2,
        extend_seq_lens=lengths,
    )
    q = torch.zeros((2, 8, 256), dtype=torch.bfloat16)
    v = torch.zeros((2, 8, 128), dtype=torch.bfloat16)
    kernel = Mock(return_value=(v, None))
    monkeypatch.setattr(runtime.mla, "mla_prefill", kernel)
    leaf.forward_extend(q, q, v, layer, None, None, 1, save_kv_cache=False)
    assert kernel.call_args.kwargs["window_left"] == window_left
    kwargs = dict(
        cum_seq_lens_q=cumulative,
        cum_seq_lens_kv=cumulative,
        max_q_len=2,
        max_kv_len=2,
        seq_lens=lengths,
        batch_size=1,
        causal=True,
    )
    # DeepSeek's chunked path stays full-history, without a new caller argument.
    leaf.forward_extend_chunked(q, q, v, spec.scaling, 0.0, **kwargs)
    assert kernel.call_args.kwargs["window_left"] == -1
    assert kernel.call_args.kwargs["is_causal"] is True
    leaf.forward_extend_chunked(
        q, q, v, spec.scaling, 0.0, **(kwargs | {"causal": False})
    )
    assert kernel.call_args.kwargs["window_left"] == -1
    assert kernel.call_args.kwargs["is_causal"] is False


@pytest.mark.parametrize("prefix, new_rows", [(0, 546), (545, 5)])
def test_swa_prefill_visible_prefix_and_current_rows(runtime, prefix, new_rows):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(7)
    config = _config(runtime, device="cuda")
    spec = config.components[0].swa
    leaf = runtime.mla.MLAAttnBackend(config, spec, kernel_page_size=32)
    q = torch.randn((new_rows, 8, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((prefix + new_rows, 8, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((prefix + new_rows, 8, 128), device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, new_rows], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, prefix + new_rows], device="cuda", dtype=torch.int32)
    output = runtime.mla.mla_prefill(
        q,
        k,
        v,
        cu_q,
        cu_k,
        new_rows,
        prefix + new_rows,
        spec.scaling,
        is_causal=True,
        window_left=512,
        solution=leaf.kernel_solution,
    )
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * spec.scaling
    positions = torch.arange(new_rows, device="cuda") + prefix
    keys = torch.arange(prefix + new_rows, device="cuda")
    visible = (keys[None, :] <= positions[:, None]) & (
        keys[None, :] >= positions[:, None] - 512
    )
    scores.masked_fill_(~visible[None, :, :], float("-inf"))
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float()).bfloat16()
    torch.testing.assert_close(output, reference, atol=1e-3, rtol=5e-3)


@pytest.mark.parametrize("group_id", ["full", "swa.0"])
@pytest.mark.parametrize("prefixes", [(0, 0), (545, 31)])
@torch.no_grad()
def test_model_prefill_does_not_synchronize(runtime, monkeypatch, group_id, prefixes):
    """Real model prefill methods and kernels, with synthetic projected inputs."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter

    from tokenspeed.runtime.models.dots3_note import (
        Dots3NoteAttention,
        quantize_index_rows,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    torch.manual_seed(17)
    lengths = (270, 5)  # Ragged requests, a full SWA tile and a partial tile.
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", sum(lengths))
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 1)
    config = replace(_config(runtime, device="cuda"), context_len=1024)
    router = runtime._create_attn_backend(runtime.AttentionArch.DSA, config)
    router.set_cache_pool(_pool(runtime))
    router.init_cuda_graph_state(4)
    spec = config.components[0].full if group_id == "full" else config.components[0].swa
    layer = _layer(spec, group_id=group_id, absorbed=True)
    leaf = router.leaf_for(layer)
    page_size = leaf.kernel_page_size
    tables = {
        gid: torch.ones((2, child.max_num_pages), dtype=torch.int32, device="cuda")
        for gid, child in router.leaves.items()
    }
    num_pages = 2 * leaf.max_num_pages + 1
    tables[group_id] = (
        torch.randperm(num_pages - 1, device="cuda", dtype=torch.int32) + 1
    ).view(2, -1)
    prefix_cpu = torch.tensor(prefixes, dtype=torch.int32)
    length_cpu = torch.tensor(lengths, dtype=torch.int32)
    router.init_forward_metadata(
        2,
        2,
        torch.arange(2, device="cuda"),
        (prefix_cpu + length_cpu).cuda(),
        runtime.ForwardMode.EXTEND,
        block_tables=tables,
        block_tables_cpu={gid: table.cpu() for gid, table in tables.items()},
        query_shard=None,
        extend_seq_lens=length_cpu.cuda(),
        extend_seq_lens_cpu=length_cpu,
        extend_prefix_lens=prefix_cpu.cuda(),
        extend_prefix_lens_cpu=prefix_cpu,
        extend_replay_lens_cpu=torch.zeros_like(length_cpu),
        extend_prompt_lens_cpu=prefix_cpu + length_cpu,
        extend_with_prefix=any(prefixes),
    )
    cache = torch.empty_strided(
        (num_pages, page_size, 1, spec.kv_cache_dim),
        (page_size * spec.kv_cache_dim + 64, spec.kv_cache_dim, spec.kv_cache_dim, 1),
        dtype=torch.bfloat16,
        device="cuda",
    )
    cache.normal_(std=0.2)
    index_cache = torch.empty_strided(
        (num_pages, page_size * 132),
        (page_size * 132 + 64, 1),
        dtype=torch.uint8,
        device="cuda",
    )
    if group_id == "full":
        index_k, scales = quantize_index_rows(
            torch.randn((num_pages * page_size, 128), device="cuda").bfloat16()
        )
        index_k_block_split_scatter(
            index_cache,
            index_k,
            scales,
            torch.arange(num_pages * page_size, device="cuda"),
            page_size=page_size,
            head_dim=128,
            group_size=128,
            write_mask=None,
        )
    heads = layer.tp_q_head_num
    kv_weight = (
        torch.randn(
            (heads * (spec.qk_nope_head_dim + spec.v_head_dim), spec.kv_lora_rank),
            device="cuda",
        )
        / spec.kv_lora_rank**0.5
    ).bfloat16()
    wk, wv = kv_weight.view(heads, -1, spec.kv_lora_rank).split(
        [spec.qk_nope_head_dim, spec.v_head_dim], dim=1
    )
    attn = SimpleNamespace(
        layer_id=layer.layer_id,
        attn_mqa=layer,
        num_local_heads=heads,
        kv_lora_rank=spec.kv_lora_rank,
        qk_nope_head_dim=spec.qk_nope_head_dim,
        v_head_dim=spec.v_head_dim,
        scaling=spec.scaling,
        window_left=layer.sliding_window_size,
        index_topk=2048,
        w_kc=wk.contiguous(),
        w_vc=wv.transpose(1, 2).contiguous(),
        kv_b_proj=lambda rows: (torch.nn.functional.linear(rows, kv_weight), None),
        absorb_query=lambda q: Dots3NoteAttention.absorb_query(attn, q),
        expand_values=lambda v: Dots3NoteAttention.expand_values(attn, v),
    )
    ctx = SimpleNamespace(
        num_extends=2,
        attn_backend=router,
        token_to_kv_pool=SimpleNamespace(
            get_key_buffer=lambda layer_id: cache,
            get_index_k_buffer=lambda layer_id: index_cache,
        ),
    )
    q = torch.randn((sum(lengths), heads, spec.head_dim), device="cuda").bfloat16()
    index_q = torch.randn((sum(lengths), 64, 128), device="cuda").bfloat16()
    weights = torch.randn((sum(lengths), 64), device="cuda")

    def run():
        if group_id == "full":
            return Dots3NoteAttention._sparse_prefill(
                attn, attn.absorb_query(q), index_q, weights, ctx, leaf
            )
        return Dots3NoteAttention._swa_prefill(attn, q, ctx, leaf)

    run()  # Compile/warm kernels before enforcing the serving-time contract.
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        actual = run()
    finally:
        torch.cuda.set_sync_debug_mode(previous)

    expected = []
    offset = 0
    for req, (prefix, length) in enumerate(zip(prefixes, lengths, strict=True)):
        positions = torch.arange(prefix + length, device="cuda")
        rows = cache[
            tables[group_id][req, positions // page_size].long(),
            positions % page_size,
            0,
        ]
        query = q[offset : offset + length]
        if group_id == "full":
            # Every causal span is below top-k, so sparse attention is dense here.
            query = attn.absorb_query(query)
            keys = rows[:, None, :].expand(-1, heads, -1)
            values = rows[:, None, : spec.kv_lora_rank].expand(-1, heads, -1)
        else:
            kv = torch.nn.functional.linear(rows[:, : spec.kv_lora_rank], kv_weight)
            kv = kv.view(-1, heads, spec.qk_nope_head_dim + spec.v_head_dim)
            keys = torch.cat(
                (
                    kv[..., : spec.qk_nope_head_dim],
                    rows[:, None, spec.kv_lora_rank :].expand(-1, heads, -1),
                ),
                dim=-1,
            )
            values = kv[..., spec.qk_nope_head_dim :]
        scores = (
            torch.einsum("qhd,khd->hqk", query.float(), keys.float()) * spec.scaling
        )
        query_positions = torch.arange(prefix, prefix + length, device="cuda")
        visible = positions[None, :] <= query_positions[:, None]
        if group_id != "full":
            visible &= positions[None, :] >= query_positions[:, None] - 512
        scores.masked_fill_(~visible[None, :, :], float("-inf"))
        result = torch.einsum(
            "hqk,khd->qhd", scores.softmax(-1), values.float()
        ).bfloat16()
        expected.append(attn.expand_values(result) if group_id == "full" else result)
        offset += length
    torch.testing.assert_close(actual, torch.cat(expected), atol=1e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("page_size,width", [(64, 576), (32, 1088)])
@pytest.mark.parametrize("masked", [False, True])
@torch.no_grad()
def test_latent_prologue_writes_only_descriptor_slots(page_size, width, masked):
    """Real packed pool: page gaps/index bytes and masked rows stay untouched."""
    from runtime.cache_pool_test_utils import make_arena, one_group
    from tokenspeed_kernel.ops.attention.prologue import mla_prologue

    from tokenspeed.runtime.layers.attention.kv_cache.dots3_note import (
        Dots3NoteCachePool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheFieldSpec,
        pack,
    )
    from tokenspeed.runtime.layers.rotary_embedding import get_rope

    fields = [
        CacheFieldSpec(
            "layer.0.latent_kv",
            "flatkv",
            (page_size, 1, width),
            "bfloat16",
            exact_page_stride=False,
        )
    ]
    fields.append(
        CacheFieldSpec(
            "layer.0.index_k" if page_size == 64 else "layer.0.guard",
            "flatkv",
            (page_size, 132),
            "uint8",
            exact_page_stride=False,
        )
    )
    group = one_group("cache", *fields, rows_per_page=page_size)
    plan = pack(
        (group,),
        prefix_granularity=page_size,
        cache_blocks_per_lcm_block={"cache": 1},
        alignment=1,
        max_padding_fraction=0.0,
    ).bind(3)
    arena = make_arena(plan, "cuda", cache_group_specs=(group[0],))
    pool = Dots3NoteCachePool(arena, layer_num=1, rank=0, field_layer_offset=0)
    arena.buffer.fill_(17)
    cache = pool.get_key_buffer(0)
    assert not cache.is_contiguous()
    slots = torch.tensor(
        [page_size + 3, 3 * page_size - 1, 3 * page_size + 1, 0], device="cuda"
    )
    positions = torch.tensor([7, 33, 511, 0], device="cuda")
    mask = torch.tensor([True, False, True, False], device="cuda") if masked else None
    target = pool.kv_write_target(0, slots, mask)
    assert (
        target.kv_cache is cache and target.slots is slots and target.write_mask is mask
    )
    assert not target.sanitize
    torch.manual_seed(29)
    q = torch.randn((4, 2, width), device="cuda", dtype=torch.bfloat16)
    latent = torch.randn((4, width), device="cuda", dtype=torch.bfloat16)
    rotary = get_rope(
        64,
        rotary_dim=64,
        max_position=1024,
        base=50000,
        rope_scaling=None,
        is_neox_style=False,
    ).cuda()
    q_ref, latent_ref = q.clone(), latent.clone()
    q_rope, k_rope = rotary(
        positions, q[..., -64:].clone(), latent[:, None, -64:].clone()
    )
    q_ref[..., -64:] = q_rope
    latent_ref[:, -64:] = k_rope.squeeze(1)
    expected = arena.buffer.clone()
    expected_cache = expected.view(cache.dtype).as_strided(
        cache.shape, cache.stride(), cache.storage_offset()
    )
    live = torch.ones_like(slots, dtype=torch.bool) if mask is None else mask
    selected = slots[live]
    expected_cache[selected // page_size, selected % page_size, 0] = latent_ref[live]
    result = mla_prologue(
        q,
        q[..., -64:],
        latent,
        expanded=None,
        rotary=rotary.as_rotary(positions),
        cache=target,
        solution=None,
        override=None,
    )
    torch.testing.assert_close(result.query, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(arena.buffer, expected, rtol=0, atol=0)


def _isolated_imports():
    """Load real APIs without unrelated eager native registrations; fresh process only."""
    import ast
    import importlib
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name, path in (
        ("tokenspeed_kernel", root / "tokenspeed-kernel/python/tokenspeed_kernel"),
        (
            "tokenspeed_kernel.ops.transform",
            root / "tokenspeed-kernel/python/tokenspeed_kernel/ops/transform",
        ),
        (
            "tokenspeed.runtime.layers.attention.backends",
            root / "python/tokenspeed/runtime/layers/attention/backends",
        ),
        (
            "tokenspeed.runtime.layers.quantization",
            root / "python/tokenspeed/runtime/layers/quantization",
        ),
        (
            "tokenspeed.runtime.distributed",
            root / "python/tokenspeed/runtime/distributed",
        ),
    ):
        spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
        module = importlib.util.module_from_spec(spec)
        module.__path__ = [str(path)]
        sys.modules[name] = module
        print(f"ISOLATED: skipped initializer {name}", flush=True)
    # No weight quantization is exercised by these attention-only fixtures.
    sys.modules["tokenspeed.runtime.layers.quantization"].QUANTIZATION_METHODS = {}
    for suffix in (
        "attention.mha",
        "attention.mla",
        "attention.dsa",
        "attention.dsv4",
        "attention.gdn",
        "attention.kpool",
    ):
        name = f"tokenspeed_kernel.ops.{suffix}"
        path = (
            root / "tokenspeed-kernel/python" / name.replace(".", "/") / "__init__.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        tree = ast.parse(path.read_text(), filename=str(path))
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Import)
                and all(
                    alias.name.startswith("tokenspeed_kernel.") for alias in node.names
                )
            )
        ]
        exec(compile(tree, str(path), "exec"), module.__dict__)
        print(f"ISOLATED: skipped side-effect registrations in {name}", flush=True)
    importlib.import_module("tokenspeed_kernel.ops.attention.mla.triton")
    importlib.import_module("tokenspeed_kernel.ops.attention.dsa.triton")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--isolated-imports" in args:
        args.remove("--isolated-imports")
        _isolated_imports()
    raise SystemExit(pytest.main([__file__, *args]))
