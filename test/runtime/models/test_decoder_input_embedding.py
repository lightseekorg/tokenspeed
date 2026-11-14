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

"""CPU checks for text lookup, media scatter, and graph input staging."""

from dataclasses import dataclass
from test.runtime.models.test_inkling2_model import _mapping, _model
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import LlamaConfig

from tokenspeed.runtime.configs.inkling_config import (
    InklingMMConfig,
    InklingModelConfig,
)
from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
from tokenspeed.runtime.models.base.input_embedding import DecoderInputEmbedding
from tokenspeed.runtime.models.inkling import InklingForConditionalGeneration
from tokenspeed.runtime.models.llama import LlamaForCausalLM
from tokenspeed.runtime.multimodal.embedder import VisionEmbedder
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalForwardContext,
    MultimodalInputs,
)


def _lookup() -> nn.Embedding:
    lookup = nn.Embedding(8, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        lookup.weight.copy_(torch.arange(1, 33).reshape(8, 4))
    return lookup


def _old_model(use_norm: bool) -> InklingForConditionalGeneration:
    text = InklingModelConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        d_rel=2,
        use_embed_norm=use_norm,
    )
    # This CPU test does not create the optional GPU stream.
    with patch("torch.cuda.is_available", return_value=False):
        model = InklingForConditionalGeneration(
            config=InklingMMConfig(text_config=text),
            mapping=_mapping(),
            quant_config=None,
            prefix="",
            is_multimodal_active=False,
            mm_attention_backend=None,
        )
    with torch.no_grad():
        model.model.embed_tokens.weight.normal_()
        if use_norm:
            model.model.embed_norm.weight.copy_(torch.arange(1, 9))
    return model


def _raw_model() -> LlamaForCausalLM:
    model = LlamaForCausalLM(
        config=LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=0,
            num_attention_heads=2,
            num_key_value_heads=1,
        ),
        mapping=_mapping(),
        quant_config=None,
        prefix="",
    )
    with torch.no_grad():
        model.model.embed_tokens.weight.normal_()
    return model


@pytest.fixture
def cpu_legacy_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only the old norm's GPU-only leaf with its CPU reference."""

    def rmsnorm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        residual: torch.Tensor | None,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        assert residual is None and out is None
        values = x.float()
        return (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
            * weight.float()
        ).to(x.dtype)

    monkeypatch.setattr("tokenspeed.runtime.layers.layernorm.rmsnorm", rmsnorm)


@dataclass
class _RunnerConfig:
    requires_request_token_history: bool


@dataclass
class _CPURunner:
    model: nn.Module
    model_config: _RunnerConfig
    is_generation: bool
    is_multimodal: bool


class _LanguageWrapper(nn.Module):
    def __init__(self, language_model: nn.Module) -> None:
        super().__init__()
        self.language_model = language_model


class _LegacyStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = _lookup()


class _LegacyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _LegacyStack()


class _CPUInputGraph(PrefillGraph):
    """Use the real graph owner and staging buffer without a GPU capture."""

    def __init__(self, model: nn.Module) -> None:
        config = ModelExecutorConfig(
            max_req_pool_size=2,
            output_length=1,
            enforce_eager=True,
            prefix_granularity=1,
            max_num_seqs=1,
            chunked_prefill_size=4,
            input_vocab_size=32,
            vocab_size=32,
            context_len=8,
            physical_context_len=8,
            device="cpu",
            gpu_id=0,
            global_rank=0,
            cudagraph_capture_sizes=None,
            disable_cuda_graph_padding=False,
            spec_topk=1,
            max_cudagraph_capture_size=1,
            model_is_mrope=False,
            autotune_cache_key=None,
            prefill_only=False,
            decode_only_attention=False,
            prefill_graph_capture_batch_sizes=None,
            disable_drafter_prefill_graph=True,
            input_logprob_chunk_tokens=4,
            enable_speculative_sampling=False,
            query_shard_size=1,
            query_shard_rank=0,
            enable_memory_saver=False,
            disable_prefill_graph=False,
            prefill_graph_max_tokens=4,
            prefill_graph_capture_sizes=[4],
        )
        runner = _CPURunner(model, _RunnerConfig(False), True, False)
        super().__init__(
            model_runner=runner,
            attn_backend=None,
            token_to_kv_pool=None,
            input_buffers=None,
            config=config,
            drafter=None,
            num_warmup=0,
            graph_supported=True,
        )
        weight = self.inner_model.embed_tokens.weight
        self._input_embeds_buf = torch.full(
            (4, weight.shape[1]), -99, dtype=weight.dtype
        )

    def stage(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self._compute_input_embeddings(
            input_ids, None, request_token_history=None
        )
        self._land_input_embeds(embeds, bucket=4)
        return self._input_embeds_buf


def test_adapter_uses_current_weights_and_one_learned_normalization() -> None:
    lookup = _lookup()
    norm = nn.RMSNorm(4, eps=1e-5, elementwise_affine=True, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 2, 3, 5]))
    embedding = DecoderInputEmbedding(lookup=lookup, transform=norm)
    ids = torch.tensor([1, 6])
    expected = norm(lookup(ids))
    torch.testing.assert_close(embedding(ids), expected, rtol=0, atol=0)
    assert not torch.equal(expected, norm(expected))
    with torch.no_grad():
        lookup.weight.add_(2)
        norm.weight.mul_(2)
    torch.testing.assert_close(embedding(ids), norm(lookup(ids)), rtol=0, atol=0)
    assert embedding.weight is lookup.weight
    assert (embedding.num_embeddings, embedding.embedding_dim) == (8, 4)


def test_inkling2_hook_normalizes_without_changing_supplied_embeddings() -> None:
    model = _model()
    with torch.no_grad():
        model.model.embed_tokens.weight.normal_()
        model.model.embed_norm.weight.copy_(torch.arange(1, 33))
    ids = torch.tensor([1, 3])
    raw = model.model.embed_tokens(ids)
    expected = model.model.embed_norm(raw)
    torch.testing.assert_close(
        model.get_decoder_input_embedding()(ids), expected, rtol=0, atol=0
    )
    assert not torch.equal(raw, expected)
    kwargs = model.prepare_model_kwargs(None, ids, {"input_embeds": raw})
    assert kwargs["input_embeds"] is raw


@pytest.mark.parametrize("use_norm", [False, True])
def test_old_inkling_hook_matches_its_multimodal_lookup(
    cpu_legacy_norm: None, use_norm: bool
) -> None:
    model = _old_model(use_norm)
    ids = torch.tensor([1, 3])
    raw = model.model.embed_tokens(ids)
    expected = model.model.embed_norm(raw) if use_norm else raw
    torch.testing.assert_close(
        model.get_decoder_input_embedding()(ids), expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.model.mm_text_embedding()(ids), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_prefill_staging_uses_outer_hook_and_preserves_buffer(
    cpu_legacy_norm: None, wrapped: bool
) -> None:
    model = _old_model(True)
    graph = _CPUInputGraph(_LanguageWrapper(model) if wrapped else model)
    address = graph._input_embeds_buf.data_ptr()
    ids = torch.tensor([1, 3, 5, 7])
    expected = model.model.embed_norm(model.model.embed_tokens(ids))
    torch.testing.assert_close(graph.stage(ids), expected, rtol=0, atol=0)
    shorter = torch.tensor([4, 2])
    staged = graph.stage(shorter)
    torch.testing.assert_close(
        staged[:2], model.get_decoder_input_embedding()(shorter), rtol=0, atol=0
    )
    assert torch.count_nonzero(staged[2:]) == 0
    assert staged.data_ptr() == address


def test_prefill_staging_keeps_raw_lookup_without_a_model_hook() -> None:
    model = _raw_model()
    graph = _CPUInputGraph(model)
    ids = torch.tensor([1, 3, 5, 7])
    torch.testing.assert_close(
        graph.stage(ids), model.model.embed_tokens(ids), rtol=0, atol=0
    )


def test_prefill_staging_preserves_legacy_wrapper_lookup() -> None:
    model = _LegacyLM()
    graph = _CPUInputGraph(model)
    ids = torch.tensor([1, 3])
    torch.testing.assert_close(
        graph.stage(ids)[:2], model.model.embed_tokens(ids), rtol=0, atol=0
    )


def test_media_scatter_does_not_normalize_encoder_rows() -> None:
    model = _model()
    with torch.no_grad():
        model.model.embed_tokens.weight.normal_()
        model.model.embed_norm.weight.fill_(3)
    media = torch.arange(32, dtype=torch.bfloat16).reshape(1, 32)
    item = MultimodalDataItem(
        modality=Modality.IMAGE, hash=7, offsets=[(1, 1)], encoded=media
    )
    context = MultimodalForwardContext(
        mm_inputs=[MultimodalInputs(mm_items=[item])],
        extend_prefix_lens=[0],
        extend_seq_lens=[3],
        max_encoder_tokens=4,
    )
    ids = torch.tensor([1, 10000, 3])
    actual, kwargs = VisionEmbedder(encoder_mapping=None).apply(
        input_ids=ids,
        text_embedding=model.get_decoder_input_embedding(),
        ctx=context,
        encoders={},
        multimodal_model=model,
    )
    expected = model.get_decoder_input_embedding()(torch.tensor([1, 3]))
    torch.testing.assert_close(actual[[0, 2]], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[1:2], media, rtol=0, atol=0)
    assert kwargs == {}
