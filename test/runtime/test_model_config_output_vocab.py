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

"""Output vocabulary controls sampling without shrinking input embeddings."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.model_config import ModelConfig  # noqa: E402
from tokenspeed.runtime.configs.model_init import ModelInitConfig
from tokenspeed.runtime.configs.vocabulary import VocabularySpec  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _load(tmp_path, *, input_vocab_size, output_fields):
    config = dict(
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        torch_dtype="bfloat16",
        vocab_size=input_vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.update(output_fields)
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = ServerArgs(
        model=str(tmp_path),
        device="cpu",
        dtype="bfloat16",
        trust_remote_code=False,
        max_model_len=256,
        world_size=1,
        max_num_seqs=2,
        chunked_prefill_size=128,
        enforce_eager=True,
        disable_prefill_graph=True,
    )
    return ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        model_override_args="{}",
        server_args=args,
        dtype=args.dtype,
        context_length=args.max_model_len,
    )


@pytest.mark.parametrize(
    "output_fields",
    [
        {},
        {"output_vocab_size": None},
        {"output_vocab_size": 64},
        {"output_vocab_size": 128},
    ],
)
def test_input_and_output_vocabularies_are_independent(tmp_path, output_fields):
    model = _load(tmp_path, input_vocab_size=128, output_fields=output_fields)
    assert model.vocabulary.input_size == 128
    assert model.vocabulary.output_size == (
        output_fields.get("output_vocab_size") or 128
    )
    assert model.hf_text_config.vocab_size == 128


@pytest.mark.parametrize("size", [0, -1, True, 96.5])
@pytest.mark.parametrize("domain", ["input", "output"])
def test_invalid_vocabulary_is_rejected(tmp_path, size, domain):
    with pytest.raises(ValueError, match=f"{domain} vocabulary"):
        _load(
            tmp_path,
            input_vocab_size=size if domain == "input" else 128,
            output_fields={"output_vocab_size": size if domain == "output" else 64},
        )


def test_executor_config_keeps_both_vocabularies(tmp_path):
    import torch

    from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig

    model = _load(
        tmp_path, input_vocab_size=128, output_fields={"output_vocab_size": 64}
    )
    args = ServerArgs(
        model=str(tmp_path),
        device="cpu",
        dtype="bfloat16",
        world_size=1,
        max_num_seqs=2,
        chunked_prefill_size=128,
        enforce_eager=True,
        disable_prefill_graph=True,
    )
    args.mapping.rank = 0
    executor = ModelExecutorConfig.from_server_args(
        args,
        model,
        torch.nn.Module(),
        max_req_pool_size=4,
        gpu_id=0,
        global_rank=0,
        prefix_granularity=1,
    )
    assert executor.vocabulary is model.vocabulary


@pytest.mark.parametrize("output_size", [64, 128])
def test_runtime_states_sizes_draft_probabilities_from_output_domain(output_size):
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates

    vocabulary = VocabularySpec(input_size=128, output_size=output_size)
    states = RuntimeStates(2, vocabulary, 1, "cpu")
    states.init_draft_probs(spec_num_tokens=2, reject_threshold=2.0)
    assert states.vocabulary is vocabulary
    assert states.future_input_map.shape == (3, 1)
    assert states.draft_probs.shape == (3, 2, output_size)


def test_draft_media_substitute_uses_input_vocabulary(tmp_path):
    import torch

    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.multimodal_runtime import MultimodalRuntime
    from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

    model = _load(
        tmp_path,
        input_vocab_size=128,
        output_fields={"output_vocab_size": 64, "image_token_id": 100},
    )
    buffers = InputBuffers(
        max_bs=1, max_num_tokens=4, state_write_padding_pool_index=2, device="cpu"
    )
    MultimodalRuntime.wire_drafter(buffers, model)
    item = MultimodalDataItem(modality=Modality.IMAGE, hash=123)
    item.set_pad_value()
    tokens = torch.tensor([10, item.pad_value], dtype=torch.int64)
    # The real substitution keeps the checkpoint's input-only token ID.
    from tokenspeed.runtime.multimodal.inputs import substitute_mm_pad_

    substitute_mm_pad_(tokens, buffers.mm_pad_substitute_ids)
    assert tokens.tolist() == [10, 100]


@pytest.mark.parametrize("tied", [False, True])
def test_common_model_builds_head_from_resolved_output_domain(tied):
    import torch
    from transformers import PretrainedConfig

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM

    class EmbeddingModel(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(
                config.vocab_size, config.hidden_size
            )

    class CausalModel(BaseCausalLM):
        model_cls = EmbeddingModel

    config = PretrainedConfig(vocab_size=128, hidden_size=8, tie_word_embeddings=tied)
    vocabulary = VocabularySpec(input_size=128, output_size=64)
    mapping = Mapping(rank=0, world_size=1)
    if tied:
        with pytest.raises(ValueError, match="equal vocabulary sizes"):
            CausalModel(
                ModelInitConfig(hf_config=config, vocabulary=vocabulary),
                mapping,
            )
        return
    model = CausalModel(
        ModelInitConfig(hf_config=config, vocabulary=vocabulary),
        mapping,
    )
    hidden = model.model.embed_tokens(torch.tensor([127]))
    model.logits_processor.prepare_communication(model.lm_head)
    logits = model.logits_processor._get_logits(
        hidden, model.lm_head, None, require_full_vocab=True
    )
    assert logits.shape == (1, 64)
    assert model.vocabulary is vocabulary
    assert config.vocab_size == 128


def test_target_outputs_must_be_valid_input_ids(tmp_path):
    with pytest.raises(ValueError, match="prefix of the input vocabulary"):
        _load(tmp_path, input_vocab_size=64, output_fields={"output_vocab_size": 128})


@pytest.mark.parametrize("output_size", [64, 65, 128])
def test_multimodal_wrapper_keeps_resolved_vocabulary(output_size):
    import torch

    from tokenspeed.runtime.configs.qwen3_5_config import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
    )
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        pytest.skip("Model construction requires CUDA")
    text_config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        layer_types=["full_attention"],
        full_attention_interval=1,
        tie_word_embeddings=False,
    )
    hf_config = Qwen3_5Config(text_config=text_config)
    vocabulary = VocabularySpec(input_size=128, output_size=output_size)
    with torch.device("cuda"):
        model = Qwen3_5ForConditionalGeneration(
            ModelInitConfig(hf_config=hf_config, vocabulary=vocabulary),
            mapping=Mapping(rank=0, world_size=1),
            is_multimodal_active=False,
        )
        with torch.no_grad():
            model.model.embed_tokens.weight.fill_(1)
            model.lm_head.weight.zero_()
            hidden = model.model.embed_tokens(torch.tensor([127]))
            logits = model.logits_processor._get_logits(
                hidden, model.lm_head, None, require_full_vocab=True
            )
    assert model.vocabulary is vocabulary
    assert model.config is text_config
    assert text_config.vocab_size == 128
    torch.testing.assert_close(
        logits, torch.zeros(1, output_size, dtype=logits.dtype, device=logits.device)
    )


def test_model_profile_owns_checkpoint_vocabulary_meaning(tmp_path, monkeypatch):
    import torch

    from tokenspeed.runtime.configs.load_config import LoadConfig
    from tokenspeed.runtime.configs.model_config import configure_mha_attention
    from tokenspeed.runtime.configs.model_profile import ModelProfile
    from tokenspeed.runtime.model_loader.loader import _initialize_model
    from tokenspeed.runtime.plugins import registry

    vocabulary = VocabularySpec(input_size=256, output_size=64)

    class ProfiledVocabularyModel(torch.nn.Module):
        def __init__(self, config: ModelInitConfig, mapping, quant_config):
            super().__init__()
            self.vocabulary = config.vocabulary
            self.embedding = torch.nn.Embedding(
                config.vocabulary.input_size, config.hf_config.hidden_size
            )
            self.head = torch.nn.Linear(
                config.hf_config.hidden_size, config.vocabulary.output_size
            )

        def forward(self, input_ids):
            return self.head(self.embedding(input_ids))

        @classmethod
        def model_profile(cls, hf_config):
            return ModelProfile(
                vocabulary=vocabulary,
                configure_attention=configure_mha_attention,
                cache_family="mha",
                linear_attention=None,
                default_attention_backend=None,
                request_token_history=False,
                is_multimodal=False,
                tokenizer_kwargs={},
                attention_instances_per_layer=1,
                numerics_envelopes=frozenset({"auto"}),
            )

    monkeypatch.setattr(registry, "_MODELS", dict(registry._MODELS))
    registry.register_model(ProfiledVocabularyModel)
    model = _load(
        tmp_path,
        input_vocab_size=128,
        output_fields={
            "architectures": ["ProfiledVocabularyModel"],
            "output_vocab_size": 32,
        },
    )
    assert model.vocabulary is vocabulary
    assert model.hf_text_config.vocab_size == 128
    assert model.hf_text_config.output_vocab_size == 32
    # Model construction must reuse the profile, not normalize the HF fields
    # (128 inputs / 32 outputs) again or infer output width from the input size.
    built = _initialize_model(model, LoadConfig())
    assert built.vocabulary is vocabulary
    assert built(torch.tensor([255])).shape == (1, 64)
