from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper
from tokenspeed.runtime.execution.drafter import eagle as eagle_module
from tokenspeed.runtime.execution.drafter.eagle import (
    Eagle,
    should_compute_dsa_topk_for_draft_first_step,
    should_reduce_draft_first_step,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.utils import workspace_indices_to_kv_slots
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.models import glm5 as glm5_module
from tokenspeed.runtime.models.extensible import ExtensibleLM
from tokenspeed.runtime.models.glm5 import (
    GlmDsaDecodeTopK,
    GlmDsaIndexerOutput,
    GlmDsaPrefillTopK,
    GlmMoeDsaAttention,
    GlmMoeDsaDecoderLayer,
)
from tokenspeed.runtime.models.glm5_nextn import (
    GlmMoeDsaDraftDecoderLayer,
    GlmMoeDsaForCausalLMNextN,
    GlmMoeDsaModelNextN,
)
from tokenspeed.runtime.moe import distribution_recorder as recorder_module
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeConfig,
    DpSamplingRuntimeLimits,
    DpSamplingSupport,
    DpSamplingTopology,
    resolve_dp_sampling_runtime,
    resolve_dp_sampling_support,
    validate_dp_sampling_lm_head_vocab,
)
from tokenspeed.runtime.sampling.logits_layout import LogitsLayoutPlan


def _graph_route(
    bs: int,
    ctx: ForwardContext,
    *,
    disable: bool = False,
    dp_size: int = 1,
    disable_padding: bool = False,
    max_bs: int,
    capture_bs: list[int],
    max_tokens_per_req: int = 1,
) -> tuple[bool, int]:
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.disable = disable
    wrapper.dp_size = dp_size
    wrapper.disable_padding = disable_padding
    wrapper.max_bs = max_bs
    wrapper.capture_bs = capture_bs
    wrapper.graphs = set(capture_bs)
    wrapper.max_tokens_per_req = max_tokens_per_req
    use_graph = wrapper.can_run(bs, ctx)
    return use_graph, wrapper.padded_bs(bs, ctx) if use_graph else bs


def _dp_runtime_config(
    *,
    tp_rank: int = 0,
    tp_size: int = 4,
    tp_group: tuple[int, ...] = (0, 1, 2, 3),
    num_tokens_per_req: int = 6,
    min_bs: int = 8,
    max_bucket_bs: int = 8,
    vocab_size: int = 8,
    device: torch.device | str = "cpu",
    skip_all_gather: bool = False,
) -> DpSamplingRuntimeConfig:
    return DpSamplingRuntimeConfig(
        enabled=True,
        vocab_size=vocab_size,
        max_bucket_bs=max_bucket_bs,
        min_bs=min_bs,
        num_tokens_per_req=num_tokens_per_req,
        topology=DpSamplingTopology(
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            skip_all_gather=skip_all_gather,
        ),
        device=device,
    )


def test_extensible_lm_exposes_base_sampling_setup_handles():
    base = SimpleNamespace(logits_processor=object(), lm_head=object())
    ext = ExtensibleLM.__new__(ExtensibleLM)
    torch.nn.Module.__init__(ext)
    ext.base_lm = base

    assert ext.logits_processor is base.logits_processor
    assert ext.lm_head is base.lm_head


def test_tokenspeed_kernel_moe_import_tolerates_optional_mxfp4_backends():
    import tokenspeed_kernel.ops.moe as moe

    assert callable(moe.moe_plan)


def test_logits_processor_dp_layout_threshold_and_modes():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    processor.configure_dp_logits_layout(_dp_runtime_config(min_bs=16))

    assert (
        processor._resolve_logits_layout_plan(
            torch.empty(15 * 6, 3),
            LogitsMetadata(forward_mode=ForwardMode.DECODE),
        )
        is None
    )

    decode_plan = processor._resolve_logits_layout_plan(
        torch.empty(16 * 6, 3),
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
    )
    assert decode_plan is not None

    verify_plan = processor._resolve_logits_layout_plan(
        torch.empty(16 * 6, 3),
        LogitsMetadata(forward_mode=ForwardMode.TARGET_VERIFY),
    )
    assert verify_plan is not None

    assert (
        processor._resolve_logits_layout_plan(
            torch.empty(32 * 6, 3),
            LogitsMetadata(forward_mode=ForwardMode.EXTEND),
        )
        is None
    )


def test_cuda_graph_wrapper_uses_existing_route_for_padding():
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.disable = False
    wrapper.dp_size = 1
    wrapper.disable_padding = False
    wrapper.max_bs = 32
    wrapper.capture_bs = [24, 32]
    wrapper.graphs = {24, 32}
    wrapper.max_tokens_per_req = 1
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=30,
        num_extends=0,
        input_num_tokens=30,
        forward_mode=ForwardMode.DECODE,
    )

    assert wrapper.can_run(30, ctx)
    assert wrapper.padded_bs(30, ctx) == 32


def test_cuda_graph_req_pool_padding_uses_reserved_sink_row():
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.config = SimpleNamespace(max_req_pool_size=99)
    active_indices = torch.tensor([7, 8], dtype=torch.int64)

    padded_indices = wrapper._pad_graph_req_pool_indices(active_indices, 4)

    assert padded_indices.tolist() == [7, 8, 99, 99]


def test_cuda_graph_state_write_padding_uses_reserved_sink_row():
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.config = SimpleNamespace(max_req_pool_size=99)
    wrapper.input_buffers = SimpleNamespace(
        state_write_req_pool_indices_buf=torch.full((4,), -1, dtype=torch.int64)
    )
    active_indices = torch.tensor([7, 8], dtype=torch.int64)

    wrapper._set_graph_state_write_indices(active_indices, 4)

    assert wrapper.input_buffers.state_write_req_pool_indices_buf.tolist() == [
        7,
        8,
        99,
        99,
    ]


def test_cuda_graph_replay_syncs_draft_seq_lens_before_draft_metadata():
    class Backend:
        uses_paged_cache_groups = False
        uses_padded_decode_token_mask = False

        def __init__(self):
            self.calls = []

        def init_forward_metadata_replay_cuda_graph(
            self,
            bs,
            req_pool_indices,
            seq_lens,
            *,
            req_to_page,
            forward_mode,
            **kwargs,
        ):
            self.calls.append(
                SimpleNamespace(
                    bs=bs,
                    seq_lens=seq_lens.clone(),
                    seq_lens_ptr=seq_lens.data_ptr(),
                    req_to_page=req_to_page,
                    forward_mode=forward_mode,
                )
            )

    target_backend = Backend()
    draft_backend = Backend()
    draft_seq_lens_buf = torch.full((4,), -1, dtype=torch.int32)
    draft_req_to_page = torch.zeros((4, 1), dtype=torch.int32)
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper.attn_backend = target_backend
    wrapper.draft_attn_backend = draft_backend
    wrapper.drafter = SimpleNamespace(
        draft_seq_lens_buf=draft_seq_lens_buf,
        req_to_page=draft_req_to_page,
    )
    wrapper.max_tokens_per_req = 6
    wrapper.use_target_verify_forward_mode = True

    seq_lens = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    wrapper._init_replay_metadata(
        padded_bs=4,
        actual_bs=4,
        req_pool_indices=torch.arange(4, dtype=torch.int64),
        seq_lens=seq_lens,
        req_to_page=torch.zeros((4, 1), dtype=torch.int32),
        forward_mode=ForwardMode.TARGET_VERIFY,
    )

    assert draft_seq_lens_buf.tolist() == [10, 11, 12, 13]
    assert target_backend.calls[0].seq_lens_ptr == seq_lens.data_ptr()
    assert draft_backend.calls[0].seq_lens_ptr == draft_seq_lens_buf.data_ptr()
    assert draft_backend.calls[0].seq_lens.tolist() == [10, 11, 12, 13]
    assert draft_backend.calls[0].req_to_page is draft_req_to_page
    assert draft_backend.calls[0].forward_mode is ForwardMode.DRAFT_EXTEND


def test_glm_nextn_draft_first_step_uses_reduced_collectives():
    model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)

    assert should_reduce_draft_first_step(model, ForwardMode.TARGET_VERIFY)
    assert should_compute_dsa_topk_for_draft_first_step(model)
    assert not should_reduce_draft_first_step(model, ForwardMode.IDLE)
    assert should_reduce_draft_first_step(object(), ForwardMode.DECODE)
    assert not should_compute_dsa_topk_for_draft_first_step(object())
    assert not should_reduce_draft_first_step(object(), ForwardMode.TARGET_VERIFY)


def test_glm_nextn_preserves_real_position_zero_embedding():
    model = GlmMoeDsaModelNextN.__new__(GlmMoeDsaModelNextN)
    torch.nn.Module.__init__(model)

    embeddings = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    captured_hidden_states = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ]
    )

    class Embed(torch.nn.Module):
        def forward(self, input_ids):
            return embeddings.index_select(0, input_ids.to(torch.int64))

    class IdentityNorm(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class CaptureEhProj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, hidden_states):
            self.seen = hidden_states.clone()
            return hidden_states[:, : embeddings.shape[1]].clone()

    class Decoder(torch.nn.Module):
        def forward(self, positions, hidden_states, ctx, out_cache_loc, residual):
            return hidden_states, residual

    eh_proj = CaptureEhProj()
    model.embed_tokens = Embed()
    model.enorm = IdentityNorm()
    model.hnorm = IdentityNorm()
    model.eh_proj = eh_proj
    model.decoder = Decoder()
    model.shared_head = torch.nn.Module()
    model.shared_head.norm = IdentityNorm()

    ctx = ForwardContext(
        attn_backend=SimpleNamespace(),
        token_to_kv_pool=None,
        bs=2,
        num_extends=0,
        input_num_tokens=2,
        forward_mode=ForwardMode.IDLE,
    )

    model.forward(
        input_ids=torch.tensor([0, 1], dtype=torch.int32),
        positions=torch.tensor([0, 1], dtype=torch.int64),
        ctx=ctx,
        out_cache_loc=torch.tensor([0, 1], dtype=torch.int32),
        captured_hidden_states=captured_hidden_states,
    )

    assert eh_proj.seen is not None
    assert torch.equal(eh_proj.seen[:, : embeddings.shape[1]], embeddings)
    assert torch.equal(eh_proj.seen[:, embeddings.shape[1] :], captured_hidden_states)


def test_expert_distribution_recorder_disable_region_suppresses_hooks():
    calls = []

    class Gatherer:
        def on_select_experts(self, **kwargs):
            calls.append(kwargs)

    class Accumulator:
        def get_single_pass_gatherer_key(self, debug_name):
            return "default"

    recorder = recorder_module._ExpertDistributionRecorderReal.__new__(
        recorder_module._ExpertDistributionRecorderReal
    )
    recorder._recording = True
    recorder._disable_all = False
    recorder._single_pass_gatherers = {"default": Gatherer()}
    recorder._accumulator = Accumulator()
    recorder._current_debug_name = SimpleNamespace(value=None)
    recorder._current_layer_idx = SimpleNamespace(value=7)

    with recorder.disable_this_region():
        recorder.on_select_experts(torch.tensor([1], dtype=torch.int64))

    assert calls == []

    recorder.on_select_experts(torch.tensor([2], dtype=torch.int64))

    assert len(calls) == 1
    assert calls[0]["layer_idx"] == 7
    assert calls[0]["topk_ids"].tolist() == [2]


def test_workspace_indices_to_kv_slots_preserves_padding():
    workspace_indices = torch.tensor(
        [[0, 2, -1], [3, -1, 1]],
        dtype=torch.int32,
    )
    kv_workspace_slots = torch.tensor([100, 101, 102, 103], dtype=torch.int32)

    mapped = workspace_indices_to_kv_slots(workspace_indices, kv_workspace_slots)

    assert mapped.dtype == torch.int32
    assert mapped.tolist() == [[100, 102, -1], [103, -1, 101]]


def test_glm_nextn_first_step_correction_refreshes_dsa_metadata():
    layer = GlmMoeDsaDraftDecoderLayer.__new__(GlmMoeDsaDraftDecoderLayer)
    seq_lens = torch.tensor([100, 100, 100], dtype=torch.int32)
    accept_lengths = torch.tensor([4, 2, 1], dtype=torch.int32)
    refreshed_seq_lens = []

    class Backend:
        spec_num_tokens = 4

        def advance_draft_forward_metadata(self, lens):
            refreshed_seq_lens.append(lens.clone())

    ctx = ForwardContext(
        attn_backend=Backend(),
        token_to_kv_pool=None,
        bs=3,
        num_extends=1,
        input_num_tokens=9,
        forward_mode=ForwardMode.DRAFT_EXTEND,
        draft_seq_lens_buf=seq_lens,
        accept_lengths=accept_lengths,
    )

    layer._apply_correction(ctx)

    assert seq_lens.tolist() == [100, 98, 97]
    assert len(refreshed_seq_lens) == 1
    assert refreshed_seq_lens[0].tolist() == [100, 98, 97]


def test_glm_nextn_draft_layer_corrects_after_full_forward(monkeypatch):
    layer = GlmMoeDsaDraftDecoderLayer.__new__(GlmMoeDsaDraftDecoderLayer)
    seq_lens = torch.tensor([100, 100], dtype=torch.int32)
    accept_lengths = torch.tensor([4, 2], dtype=torch.int32)
    events = []

    class Backend:
        spec_num_tokens = 6

        def advance_draft_forward_metadata(self, lens):
            events.append(("advance", lens.clone()))

    def fake_full_forward(
        self,
        positions,
        hidden_states,
        ctx,
        out_cache_loc,
        residual,
    ):
        events.append(("full_forward", ctx.draft_seq_lens_buf.clone()))
        return hidden_states[: ctx.bs], residual

    monkeypatch.setattr(GlmMoeDsaDecoderLayer, "forward", fake_full_forward)
    ctx = ForwardContext(
        attn_backend=Backend(),
        token_to_kv_pool=None,
        bs=2,
        num_extends=0,
        input_num_tokens=12,
        forward_mode=ForwardMode.DRAFT_EXTEND,
        draft_seq_lens_buf=seq_lens,
        accept_lengths=accept_lengths,
        draft_first_step_reduce=True,
    )

    out = layer.forward(
        positions=torch.arange(12, dtype=torch.int64),
        hidden_states=torch.empty((12, 3)),
        ctx=ctx,
        out_cache_loc=torch.arange(12, dtype=torch.int32),
        residual=None,
    )

    assert events[0][0] == "full_forward"
    assert events[0][1].tolist() == [100, 100]
    assert seq_lens.tolist() == [98, 96]
    assert events[1][0] == "advance"
    assert events[1][1].tolist() == [98, 96]
    assert isinstance(out, tuple)
    assert out[0].shape == (2, 3)
    assert out[1] is None


def test_glm_dsa_decode_seq_lens_distinguish_full_window_and_catchup():
    seq_lens = torch.tensor([106, 206], dtype=torch.int32)

    full_window = GlmMoeDsaAttention._expand_decode_seq_lens_per_token(
        seq_lens,
        q_len_per_req=6,
        draft_catchup=False,
    )
    catchup = GlmMoeDsaAttention._expand_decode_seq_lens_per_token(
        seq_lens,
        q_len_per_req=6,
        draft_catchup=True,
    )

    assert full_window.tolist() == [
        101,
        102,
        103,
        104,
        105,
        106,
        201,
        202,
        203,
        204,
        205,
        206,
    ]
    assert catchup.tolist() == [
        106,
        107,
        108,
        109,
        110,
        111,
        206,
        207,
        208,
        209,
        210,
        211,
    ]


def test_glm_dsa_decode_topk_schedule_refreshes_for_current_layout(monkeypatch):
    schedule_calls = []

    class FakeDeepGemm:
        @staticmethod
        def get_num_sms():
            return 1

        @staticmethod
        def get_paged_mqa_logits_metadata(seq_lens_2d, page_size, num_sms):
            schedule_calls.append(
                {
                    "shape": tuple(seq_lens_2d.shape),
                    "seq_lens": seq_lens_2d.clone(),
                    "page_size": page_size,
                    "num_sms": num_sms,
                }
            )
            return torch.full((1,), len(schedule_calls), dtype=torch.int32)

        @staticmethod
        def fp8_paged_mqa_logits(
            q_fp8,
            kv_cache,
            weights,
            seq_lens_2d,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        ):
            del kv_cache, weights, block_tables, schedule_metadata, max_seq_len
            del clean_logits
            return torch.zeros(
                (q_fp8.shape[0] * q_fp8.shape[1], int(seq_lens_2d.max().item())),
                dtype=torch.float32,
            )

    def fake_quantize_fp8_with_scale(q_2d, **kwargs):
        del kwargs
        return q_2d, torch.ones((q_2d.shape[0], 1), dtype=torch.float32)

    def fake_deterministic_decode_topk(logits, local_topk_offsets, topk):
        del topk
        local_topk_offsets.copy_(
            torch.arange(
                local_topk_offsets.shape[1],
                dtype=local_topk_offsets.dtype,
            ).expand_as(local_topk_offsets)
        )

    def fake_local_topk_to_global_slots(**kwargs):
        kwargs["out"].copy_(kwargs["local_topk_offsets"].to(torch.int32))
        kwargs["lens_out"].copy_(kwargs["seq_lens"].to(torch.int32))

    monkeypatch.setattr(glm5_module, "deep_gemm", FakeDeepGemm)
    monkeypatch.setattr(
        glm5_module,
        "quantize_fp8_with_scale",
        fake_quantize_fp8_with_scale,
    )
    monkeypatch.setattr(
        glm5_module,
        "deterministic_decode_topk",
        fake_deterministic_decode_topk,
    )
    monkeypatch.setattr(glm5_module, "has_deterministic_decode_topk", lambda: True)
    monkeypatch.setattr(
        glm5_module,
        "local_topk_to_global_slots",
        fake_local_topk_to_global_slots,
    )

    attn = GlmMoeDsaAttention.__new__(GlmMoeDsaAttention)
    attn.index_topk = 512
    attn.indexer = SimpleNamespace(
        index_head_dim=128,
        index_n_heads=1,
        softmax_scale=1.0,
    )
    attn.attn_mqa = SimpleNamespace(layer_id=0)
    attn._decode_topk_indices_buffer = None
    attn._decode_local_topk_offsets_buffer = None
    attn._decode_topk_lens_buffer = None
    attn._decode_topk_arange_buffer = None

    metadata = SimpleNamespace(
        num_extends=0,
        seq_lens_k=torch.tensor([64, 65], dtype=torch.int32),
        block_kv_indices=torch.zeros((2, 8), dtype=torch.int32),
    )

    class TokenPool:
        page_size = 64

        @staticmethod
        def has_index_k_with_scale_buffer():
            return True

        @staticmethod
        def get_index_k_with_scale_buffer(layer_id):
            del layer_id
            return torch.empty((512, 1, 128), dtype=torch.float32)

    ctx = ForwardContext(
        attn_backend=SimpleNamespace(
            forward_decode_metadata=metadata,
            spec_num_tokens=2,
        ),
        token_to_kv_pool=TokenPool(),
        bs=2,
        num_extends=0,
        input_num_tokens=4,
        forward_mode=ForwardMode.DECODE,
    )

    first_output = SimpleNamespace(
        query=torch.zeros((4, 1, 128), dtype=torch.float32),
        weights=torch.ones((4, 1), dtype=torch.float32),
    )
    attn._compute_decode_topk_indices_deepgemm(
        indexer_output=first_output,
        ctx=ctx,
        seq_lens_per_token=torch.tensor([63, 64, 64, 65], dtype=torch.int32),
        block_tables=metadata.block_kv_indices,
        block_tables_per_token=metadata.block_kv_indices.repeat_interleave(2, dim=0),
        q_len_per_req=2,
        decode_start=0,
        num_tokens=4,
        num_decode_tokens=4,
        topk=512,
    )

    second_output = SimpleNamespace(
        query=torch.zeros((2, 1, 128), dtype=torch.float32),
        weights=torch.ones((2, 1), dtype=torch.float32),
    )
    attn._compute_decode_topk_indices_deepgemm(
        indexer_output=second_output,
        ctx=ctx,
        seq_lens_per_token=torch.tensor([64, 65], dtype=torch.int32),
        block_tables=metadata.block_kv_indices,
        block_tables_per_token=metadata.block_kv_indices,
        q_len_per_req=1,
        decode_start=0,
        num_tokens=2,
        num_decode_tokens=2,
        topk=512,
    )
    cached_schedule = metadata._dsa_paged_mqa_schedule_metadata

    metadata.seq_lens_k.copy_(torch.tensor([70, 72], dtype=torch.int32))
    attn._compute_decode_topk_indices_deepgemm(
        indexer_output=second_output,
        ctx=ctx,
        seq_lens_per_token=torch.tensor([70, 72], dtype=torch.int32),
        block_tables=metadata.block_kv_indices,
        block_tables_per_token=metadata.block_kv_indices,
        q_len_per_req=1,
        decode_start=0,
        num_tokens=2,
        num_decode_tokens=2,
        topk=512,
    )

    assert [call["shape"] for call in schedule_calls] == [(2, 2), (2, 1), (2, 1)]
    assert schedule_calls[-1]["seq_lens"].tolist() == [[70], [72]]
    assert metadata._dsa_paged_mqa_schedule_metadata is cached_schedule
    assert metadata._dsa_paged_mqa_schedule_metadata.tolist() == [3]
    assert metadata._dsa_paged_mqa_schedule_q_len == 1
    assert metadata._dsa_paged_mqa_schedule_shape == (2, 1)


def test_glm_dsa_prefill_topk_uses_cpu_length_metadata(monkeypatch):
    captured = {}

    def fake_build_prefill_kv_workspace_slots(**kwargs):
        captured["workspace_max_seq_len"] = kwargs["max_seq_len"]
        captured["workspace_block_tables_shape"] = tuple(kwargs["block_tables"].shape)
        return (
            torch.arange(12, dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )

    def fake_compute_prefill_topk_indices_deepgemm(self, **kwargs):
        captured["deepgemm_max_seq_len"] = kwargs["max_seq_len"]
        captured["deepgemm_seq_len_sum"] = kwargs["seq_len_sum"]
        captured["deepgemm_num_prefill_tokens"] = kwargs["num_prefill_tokens"]
        return "prefill-topk"

    monkeypatch.setattr(
        glm5_module,
        "_build_prefill_kv_workspace_slots",
        fake_build_prefill_kv_workspace_slots,
    )
    monkeypatch.setattr(
        GlmMoeDsaAttention,
        "_compute_prefill_topk_indices_deepgemm",
        fake_compute_prefill_topk_indices_deepgemm,
    )

    # The live tensors deliberately disagree with the CPU mirror. The scalar
    # token-count, max-len, and chunk-budget decisions should use the CPU mirror
    # to avoid GPU syncs in the GLM DSA prefill top-k path.
    chunk_meta = SimpleNamespace(
        extend_prefix_lens=torch.tensor([100], dtype=torch.int32),
        extend_seq_lens=torch.tensor([100], dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor([7], dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
    )

    attn = GlmMoeDsaAttention.__new__(GlmMoeDsaAttention)
    attn.index_topk = 3
    ctx = ForwardContext(
        attn_backend=SimpleNamespace(chunked_prefill_metadata=chunk_meta),
        token_to_kv_pool=SimpleNamespace(page_size=64),
        req_to_page=torch.arange(4, dtype=torch.int32).view(1, 4),
        bs=1,
        num_extends=1,
        input_num_tokens=5,
        forward_mode=ForwardMode.EXTEND,
    )
    indexer_output = GlmDsaIndexerOutput(
        query=torch.empty((5, 1, 1), dtype=torch.float32),
        key=torch.empty((5, 1, 1), dtype=torch.float32),
        weights=torch.empty((5, 1), dtype=torch.float32),
    )

    result = attn._compute_prefill_topk_indices(
        indexer_output,
        ctx,
        num_prefill_tokens=5,
    )

    assert result == "prefill-topk"
    assert captured["workspace_max_seq_len"] == 12
    assert captured["workspace_block_tables_shape"] == (1, 1)
    assert captured["deepgemm_max_seq_len"] == 12
    assert captured["deepgemm_seq_len_sum"] == 12
    assert captured["deepgemm_num_prefill_tokens"] == 5


def test_glm_nextn_decode_first_step_keeps_full_verify_window():
    model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
    torch.nn.Module.__init__(model)
    seen = {}

    class DraftModel(torch.nn.Module):
        def forward(
            self,
            input_ids,
            positions,
            ctx,
            out_cache_loc,
            captured_hidden_states=None,
        ):
            seen["input_ids"] = input_ids
            seen["positions"] = positions
            seen["ctx"] = ctx
            seen["out_cache_loc"] = out_cache_loc
            seen["captured_hidden_states"] = captured_hidden_states
            # Simulate the decoder layer's draft_first_step_reduce output.
            return torch.empty((ctx.bs, 3)), None

    class Processor:
        def __call__(self, input_ids, hidden_states, lm_head, logits_metadata):
            seen["logits_input_ids"] = input_ids
            seen["logits_hidden_states"] = hidden_states
            seen["logits_metadata"] = logits_metadata
            return SimpleNamespace(
                hidden_states=hidden_states,
                next_token_logits=torch.empty((hidden_states.shape[0], 8)),
            )

    model.model = DraftModel()
    model.logits_processor = Processor()
    model.lm_head = object()

    input_ids = torch.arange(12, dtype=torch.int32)
    positions = torch.arange(100, 112, dtype=torch.int64)
    out_cache_loc = torch.arange(200, 212, dtype=torch.int32)
    hidden_states = torch.arange(12 * 3, dtype=torch.float32).view(12, 3)
    gather_ids = torch.tensor([3, 10], dtype=torch.int64)
    ctx = ForwardContext(
        attn_backend=SimpleNamespace(spec_num_tokens=6),
        token_to_kv_pool=None,
        bs=2,
        num_extends=0,
        input_num_tokens=12,
        forward_mode=ForwardMode.DRAFT_EXTEND,
        gather_ids=gather_ids,
        global_num_tokens=[12],
        global_bs=[2],
        draft_first_step_reduce=True,
        accept_lengths=torch.tensor([4, 5], dtype=torch.int32),
    )

    logits_output = model.forward(
        ctx,
        input_ids,
        positions,
        out_cache_loc,
        captured_hidden_states=hidden_states,
    )

    assert seen["ctx"] is ctx
    assert torch.equal(seen["input_ids"], input_ids)
    assert torch.equal(seen["positions"], positions)
    assert torch.equal(seen["out_cache_loc"], out_cache_loc)
    assert torch.equal(seen["captured_hidden_states"], hidden_states)
    assert seen["logits_metadata"].gather_ids is gather_ids
    assert seen["logits_hidden_states"].shape == (2, 3)
    assert logits_output.next_token_logits.shape == (2, 8)


def test_eagle_glm_first_step_computes_draft_dsa_decode_topk_before_reuse():
    eagle = Eagle.__new__(Eagle)
    eagle.spec_num_tokens = 6
    eagle.input_buffers = SimpleNamespace(
        positions_buf=torch.arange(12, dtype=torch.int64),
        out_cache_loc_buf=torch.arange(100, 112, dtype=torch.int32),
    )
    eagle.mm_pad_substitute_id = None
    eagle.padded_gather_ids_offsets_buf = torch.arange(2, dtype=torch.int64) * 6 - 1
    eagle.attn_backend = SimpleNamespace()
    eagle.token_to_kv_pool = None
    eagle.req_to_page = None
    eagle._dsa_reuse_mtp_topk = True
    eagle.draft_seq_lens_buf = torch.zeros((2,), dtype=torch.int32)
    draft_model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
    seen = {}
    draft_decode_topk = GlmDsaDecodeTopK(
        topk_indices=(1000 + torch.arange(12 * 3, dtype=torch.int32)).view(12, 3),
        topk_lens=100 + torch.arange(12, dtype=torch.int32),
    )

    class Runner:
        model = draft_model

        def forward(self, **kwargs):
            seen.update(kwargs)
            seen["initial_ctx_dsa_decode_topk"] = kwargs["ctx"].dsa_decode_topk
            kwargs["ctx"].dsa_decode_topk = draft_decode_topk
            return SimpleNamespace(
                hidden_states=torch.empty((2, 4)),
                next_token_logits=torch.empty((2, 8)),
            )

    eagle.draft_model_runner = Runner()
    full_decode_topk = GlmDsaDecodeTopK(
        topk_indices=torch.arange(12 * 3, dtype=torch.int32).view(12, 3),
        topk_lens=torch.arange(12, dtype=torch.int32),
    )
    draft_input = SimpleNamespace(
        input_num_tokens=12,
        num_extends=0,
        forward_mode=ForwardMode.TARGET_VERIFY,
        base_model_output=torch.arange(12, dtype=torch.int32),
        accept_lengths=torch.tensor([1, 4], dtype=torch.int64),
        base_out_hidden_states=torch.empty((12, 4)),
        global_num_tokens=[12],
        global_bs=[2],
        all_decode_or_idle=True,
        dsa_topk=(None, full_decode_topk),
    )

    logits_output, dsa_topk = eagle._run_first_step(2, draft_input)

    assert seen["initial_ctx_dsa_decode_topk"] is None
    assert seen["ctx"].gather_ids.tolist() == [0, 9]
    selected = dsa_topk[1]
    assert selected is not draft_decode_topk
    assert selected.topk_lens.tolist() == [100, 109]
    assert selected.topk_indices.tolist() == [
        draft_decode_topk.topk_indices[0].tolist(),
        draft_decode_topk.topk_indices[9].tolist(),
    ]
    assert logits_output.next_token_logits.shape == (2, 8)


def test_eagle_glm_mixed_first_step_selects_draft_dsa_decode_topk_rows():
    eagle = Eagle.__new__(Eagle)
    eagle.spec_num_tokens = 6
    eagle.input_buffers = SimpleNamespace(
        shifted_prefill_ids_buf=torch.arange(16, dtype=torch.int32),
        input_lengths_buf=torch.tensor([4, 1, 4], dtype=torch.int64),
        positions_buf=torch.arange(16, dtype=torch.int64),
        out_cache_loc_buf=torch.arange(100, 116, dtype=torch.int32),
    )
    eagle.mm_pad_substitute_id = None
    eagle.padded_gather_ids_offsets_buf = torch.arange(2, dtype=torch.int64) * 6 - 1
    eagle.attn_backend = SimpleNamespace()
    eagle.token_to_kv_pool = None
    eagle.req_to_page = None
    eagle._dsa_reuse_mtp_topk = True
    eagle.draft_seq_lens_buf = torch.zeros((3,), dtype=torch.int32)
    draft_model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
    seen = {}
    draft_prefill_topk = GlmDsaPrefillTopK(
        workspace_indices=torch.tensor(
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, -1],
            ],
            dtype=torch.int32,
        ),
        topk_lens=torch.tensor([200, 201, 202, 203], dtype=torch.int32),
        block_tables=torch.empty((1, 1), dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        max_seq_len=4,
        kv_workspace_slots=torch.tensor(
            [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510],
            dtype=torch.int32,
        ),
    )
    draft_decode_topk = GlmDsaDecodeTopK(
        topk_indices=torch.full((16, 3), -12345, dtype=torch.int32),
        topk_lens=torch.full((16,), -1, dtype=torch.int32),
    )
    draft_decode_topk.topk_indices[4:] = (
        1000 + torch.arange(12 * 3, dtype=torch.int32)
    ).view(12, 3)
    draft_decode_topk.topk_lens[4:] = 100 + torch.arange(12, dtype=torch.int32)

    class Runner:
        model = draft_model

        def forward(self, **kwargs):
            seen.update(kwargs)
            seen["initial_ctx_dsa_decode_topk"] = kwargs["ctx"].dsa_decode_topk
            kwargs["ctx"].dsa_prefill_topk = draft_prefill_topk
            kwargs["ctx"].dsa_decode_topk = draft_decode_topk
            return SimpleNamespace(
                hidden_states=torch.empty((3, 4)),
                next_token_logits=torch.empty((3, 8)),
            )

    eagle.draft_model_runner = Runner()
    full_decode_topk = GlmDsaDecodeTopK(
        topk_indices=torch.arange(16 * 3, dtype=torch.int32).view(16, 3),
        topk_lens=torch.arange(16, dtype=torch.int32),
    )
    draft_input = SimpleNamespace(
        input_num_tokens=16,
        num_extends=1,
        forward_mode=ForwardMode.TARGET_VERIFY,
        base_model_output=torch.arange(13, dtype=torch.int32),
        accept_lengths=torch.tensor([0, 1, 4], dtype=torch.int64),
        base_out_hidden_states=torch.empty((16, 4)),
        global_num_tokens=[16],
        global_bs=[3],
        all_decode_or_idle=False,
        dsa_topk=(None, full_decode_topk),
    )

    logits_output, dsa_topk = eagle._run_first_step(3, draft_input)

    assert seen["initial_ctx_dsa_decode_topk"] is None
    assert seen["ctx"].gather_ids.tolist() == [3, 4, 13]
    selected = dsa_topk[1]
    assert selected is not draft_decode_topk
    assert selected is not full_decode_topk
    assert selected.topk_lens.tolist() == [203, 100, 109]
    assert selected.topk_indices.tolist() == [
        [509, 510, -1],
        draft_decode_topk.topk_indices[4].tolist(),
        draft_decode_topk.topk_indices[13].tolist(),
    ]
    assert logits_output.next_token_logits.shape == (3, 8)


def test_eagle_multi_step_starts_from_accepted_prefix(monkeypatch):
    captured = {}

    def fake_compute_out_cache_loc_uniform(
        *,
        out_cache_loc_ptr,
        req_pool_indices,
        uniform_input_length,
        cache_start,
        req_to_pages,
        page_size,
    ):
        captured["req_pool_indices"] = req_pool_indices.clone()
        captured["uniform_input_length"] = uniform_input_length
        captured["cache_start"] = cache_start.clone()
        captured["page_size"] = page_size
        out_cache_loc_ptr.copy_(torch.tensor([11, 12, 21, 22], dtype=torch.int32))

    monkeypatch.setattr(
        eagle_module,
        "compute_out_cache_loc_uniform",
        fake_compute_out_cache_loc_uniform,
    )
    monkeypatch.setattr(
        eagle_module,
        "sampling_argmax",
        lambda logits: torch.argmax(logits, dim=-1),
    )

    class Backend:
        def __init__(self):
            self.advanced = []

        def advance_draft_forward_metadata(self, lens):
            self.advanced.append(lens.clone())

    backend = Backend()
    calls = []

    class Runner:
        def forward(self, **kwargs):
            calls.append(
                {
                    "ctx": kwargs["ctx"],
                    "input_ids": kwargs["input_ids"].clone(),
                    "positions": kwargs["positions"].clone(),
                    "out_cache_loc": kwargs["out_cache_loc"].clone(),
                    "spec_step_idx": kwargs["spec_step_idx"],
                }
            )
            return SimpleNamespace(
                hidden_states=torch.full((2, 3), kwargs["spec_step_idx"]),
                next_token_logits=torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
            )

    eagle = Eagle.__new__(Eagle)
    eagle.spec_num_steps = 3
    eagle.spec_num_tokens = 6
    eagle.page_size = 16
    eagle.req_to_page = torch.empty((8, 2), dtype=torch.int32)
    eagle.token_to_kv_pool = None
    eagle.attn_backend = backend
    eagle.draft_model_runner = Runner()
    eagle.hot_token_ids = None
    eagle.dp_size = 1
    eagle.world_size = 1
    eagle._dsa_reuse_mtp_topk = False
    eagle.draft_out_cache_loc_buf = torch.empty(4, dtype=torch.int32)
    eagle.draft_seq_lens_buf = torch.zeros(2, dtype=torch.int32)
    eagle.input_buffers = SimpleNamespace(
        req_pool_indices_buf=torch.tensor([4, 7], dtype=torch.int64),
        seq_lens_buf=torch.tensor([106, 206], dtype=torch.int32),
    )
    valid_cache_lengths = torch.zeros(8, dtype=torch.int32)
    valid_cache_lengths[4] = 100
    valid_cache_lengths[7] = 200
    eagle.runtime_states = SimpleNamespace(valid_cache_lengths=valid_cache_lengths)

    next_tokens = torch.full((2, 4), -1, dtype=torch.int32)
    draft_input = SimpleNamespace(
        num_extends=0,
        accept_lengths=torch.tensor([2, 5], dtype=torch.int32),
        global_num_tokens=[12],
        global_bs=[2],
        all_decode_or_idle=True,
    )
    logits_output = SimpleNamespace(
        hidden_states=torch.zeros((2, 3)),
        next_token_logits=torch.empty((2, 3)),
    )

    eagle._run_multi_step_decode(
        2,
        draft_ids=torch.tensor([8, 9], dtype=torch.int32),
        next_tokens=next_tokens,
        logits_output=logits_output,
        draft_input=draft_input,
        dsa_topk=(None, None),
    )

    assert captured["req_pool_indices"].tolist() == [4, 7]
    assert captured["uniform_input_length"] == 2
    assert captured["cache_start"].tolist() == [102, 205]
    assert captured["page_size"] == 16
    assert [x.tolist() for x in backend.advanced] == [[103, 206], [104, 207]]
    assert [call["spec_step_idx"] for call in calls] == [1, 2]
    assert [call["positions"].tolist() for call in calls] == [[102, 205], [103, 206]]
    assert [call["out_cache_loc"].tolist() for call in calls] == [[11, 21], [12, 22]]
    assert calls[0]["input_ids"].tolist() == [8, 9]
    assert calls[1]["input_ids"].tolist() == [1, 1]
    assert next_tokens[:, 2:].tolist() == [[1, 1], [1, 1]]


def test_eagle_mid_chunk_skip_is_disabled_during_cuda_graph_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        eagle_module,
        "sampling_argmax",
        lambda logits: torch.argmax(logits, dim=-1),
    )

    eagle = Eagle.__new__(Eagle)
    eagle.spec_num_steps = 3
    eagle.spec_num_tokens = 4
    eagle.hot_token_ids = None
    eagle.device = "cpu"
    eagle.dp_size = 1
    eagle._dsa_reuse_mtp_topk = False
    eagle.draft_seq_lens_buf = torch.zeros(1, dtype=torch.int32)
    eagle.padded_gather_ids_offsets_buf = torch.arange(1, dtype=torch.int64) * 4 - 1
    eagle.input_buffers = SimpleNamespace(
        all_extends_mid_chunk=True,
        seq_lens_buf=torch.tensor([10], dtype=torch.int32),
    )

    called = {}

    def fake_run_first_step(bs, draft_input):
        assert bs == 1
        return (
            SimpleNamespace(
                hidden_states=torch.zeros((1, 3)),
                next_token_logits=torch.tensor([[0.0, 1.0, 0.0]]),
            ),
            (None, None),
        )

    def fake_run_multi_step_decode(
        bs,
        draft_ids,
        next_tokens,
        logits_output,
        draft_input,
        dsa_topk,
    ):
        called["yes"] = True
        assert draft_ids.tolist() == [1]
        next_tokens[:, 2:] = torch.tensor([[2, 3]], dtype=torch.int32)

    eagle._run_first_step = fake_run_first_step
    eagle._run_multi_step_decode = fake_run_multi_step_decode
    eagle.attn_backend = SimpleNamespace(override_num_extends=lambda _: nullcontext())
    draft_input = SimpleNamespace(
        accept_lengths=torch.ones(1, dtype=torch.int32),
        num_extends=1,
        base_model_output=torch.tensor([7], dtype=torch.int32),
    )

    next_tokens = eagle.draft(draft_input)

    assert called == {"yes": True}
    assert next_tokens.tolist() == [[7, 1, 2, 3]]


def test_cuda_graph_route_uses_global_batch_for_dp_idle_rank():
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=0,
        num_extends=0,
        input_num_tokens=0,
        forward_mode=ForwardMode.DECODE,
        global_num_tokens=[0, 17],
        all_decode_or_idle=True,
    )

    assert _graph_route(
        0,
        ctx,
        dp_size=2,
        max_bs=32,
        capture_bs=[16, 32],
        max_tokens_per_req=1,
    ) == (True, 32)


def test_cuda_graph_route_respects_disable_padding_with_global_batch():
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=0,
        num_extends=0,
        input_num_tokens=0,
        forward_mode=ForwardMode.DECODE,
        global_num_tokens=[0, 17],
        all_decode_or_idle=True,
    )

    assert _graph_route(
        0,
        ctx,
        dp_size=2,
        disable_padding=True,
        max_bs=32,
        capture_bs=[16, 32],
        max_tokens_per_req=1,
    ) == (False, 0)


def test_configure_dp_sampling_sets_state():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )

    processor.configure_dp_logits_layout(_dp_runtime_config())
    assert processor.dp_sampling_enabled
    assert processor.dp_num_tokens_per_req == 6


def test_resolve_dp_sampling_runtime_uses_grouped_metadata():
    support = DpSamplingSupport(
        requested=True,
        enabled=True,
        infra_supports=True,
        drafter_available=True,
        backend_supports_verify=True,
        tp_size=4,
        tp_group_set=True,
    )

    runtime_config = resolve_dp_sampling_runtime(
        support=support,
        lm_head_rows=7,
        topology=DpSamplingTopology(
            tp_rank=0,
            tp_size=4,
            tp_group=(0, 1, 2, 3),
            skip_all_gather=False,
        ),
        limits=DpSamplingRuntimeLimits(
            runtime_vocab_size=7,
            max_num_seqs=17,
            data_parallel_size=1,
            num_tokens_per_req=6,
            configured_min_bs=None,
            device="cpu",
        ),
    )

    assert runtime_config.enabled
    assert runtime_config.vocab_size == 28
    assert runtime_config.max_bucket_bs == 20
    assert runtime_config.min_bs == 8
    assert runtime_config.num_tokens_per_req == 6


@pytest.mark.parametrize(
    "forward_mode",
    [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY],
)
def test_logits_processor_derives_dp_layout_from_effective_hidden_states(
    forward_mode,
):
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        tp_rank=0,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    processor.configure_dp_logits_layout(_dp_runtime_config(min_bs=5))

    plan = processor._resolve_logits_layout_plan(
        torch.empty(5 * 6, 3),
        LogitsMetadata(forward_mode=forward_mode),
    )

    assert plan is not None
    assert plan.effective_bs == 5
    assert plan.bucket_bs == 8


def test_dp_sampling_skip_all_gather_rejects_sharded_lm_head_vocab():
    with pytest.raises(RuntimeError, match="replicated/full-vocab LM head"):
        validate_dp_sampling_lm_head_vocab(
            lm_head_rows=4,
            vocab_size=7,
            tp_size=2,
            skip_all_gather=True,
            tie_word_embeddings=True,
        )


def test_resolve_dp_sampling_support_rejects_missing_preconditions():
    with pytest.raises(RuntimeError, match="backend_supports_dp_verify=False"):
        resolve_dp_sampling_support(
            requested=True,
            drafter_available=True,
            backend_supports_verify=False,
            topology=DpSamplingTopology(
                tp_rank=0,
                tp_size=4,
                tp_group=(0, 1, 2, 3),
                skip_all_gather=False,
            ),
        )


def test_skip_all_gather_dp_sampling_slices_hidden_states_before_lm_head():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        skip_all_gather=True,
        tp_rank=1,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    processor.configure_dp_logits_layout(
        _dp_runtime_config(tp_rank=1, skip_all_gather=True, device="cpu")
    )
    hidden_states = torch.arange(5 * 6 * 3, dtype=torch.float32).view(5 * 6, 3)
    lm_head = SimpleNamespace(weight=torch.ones(7, 3))
    plan = LogitsLayoutPlan(
        effective_bs=5,
        bucket_bs=8,
        tp_size=4,
        num_tokens_per_req=6,
    )

    logits = processor._get_logits(
        hidden_states,
        lm_head,
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
        plan=plan,
    )

    assert logits.shape == (12, 7)
    expected_rows = hidden_states[12:24].sum(dim=1)
    assert torch.equal(logits[:, 0], expected_rows)


def test_dp_sampling_slices_graph_effective_hidden_states_before_lm_head():
    processor = LogitsProcessor(
        SimpleNamespace(vocab_size=7, model_type="unit_test"),
        skip_all_gather=True,
        tp_rank=2,
        tp_size=4,
        tp_group=(0, 1, 2, 3),
    )
    processor.configure_dp_logits_layout(
        _dp_runtime_config(tp_rank=2, skip_all_gather=True, device="cpu")
    )
    hidden_states = torch.arange(5 * 6 * 3, dtype=torch.float32).view(5 * 6, 3)
    lm_head = SimpleNamespace(weight=torch.ones(7, 3))
    plan = LogitsLayoutPlan(
        effective_bs=5,
        bucket_bs=8,
        tp_size=4,
        num_tokens_per_req=6,
    )

    logits = processor._get_logits(
        hidden_states,
        lm_head,
        LogitsMetadata(forward_mode=ForwardMode.DECODE),
        plan=plan,
    )

    assert logits.shape == (12, 7)
    expected_rows = torch.cat(
        [hidden_states[24:30].sum(dim=1), torch.zeros(6, dtype=torch.float32)]
    )
    assert torch.equal(logits[:, 0], expected_rows)
