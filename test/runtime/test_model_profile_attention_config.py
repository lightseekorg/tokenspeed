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

"""Model-owned attention construction retains the runtime's hybrid composition."""

import dataclasses
import json
import os
import sys

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.model_config import (  # noqa: E402
    ModelConfig,
    configure_mha_attention,
)
from tokenspeed.runtime.configs.model_profile import ModelProfile  # noqa: E402
from tokenspeed.runtime.layers.attention.configs.base import (  # noqa: E402
    AttnConfig,
    model_wide_kwargs,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import (  # noqa: E402
    LinearAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig  # noqa: E402
from tokenspeed.runtime.layers.attention.registry import (  # noqa: E402
    _create_attn_config,
)
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _profile(factory, *, linear_attention):
    return ModelProfile(
        configure_attention=configure_mha_attention,
        create_attention_config=factory,
        cache_family="fixture",
        linear_attention=linear_attention,
        default_attention_backend=None,
        request_token_history=False,
        is_multimodal=False,
        tokenizer_kwargs={},
        attention_instances_per_layer=1,
        numerics_envelopes=frozenset({"auto"}),
    )


@pytest.fixture
def model_and_args(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                model_type="qwen3",
                architectures=["Qwen3ForCausalLM"],
                torch_dtype="bfloat16",
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=256,
                linear_layer_ids=[1],
                linear_attn_config=dict(
                    num_heads=2, head_dim=16, short_conv_kernel_size=4
                ),
            )
        )
    )
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
    args.mapping.rank = 0
    model = ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        model_override_args="{}",
        server_args=args,
        dtype=args.dtype,
        context_length=args.max_model_len,
    )
    return model, args


def _custom_attention(args, model, is_draft):
    """A model's storage policy differs from the ordinary MHA default."""
    return AttnConfig(
        components=(
            MHAConfig(
                num_attention_heads=model.num_attention_heads,
                num_kv_heads=model.num_key_value_heads,
                head_dim=model.head_dim,
                attn_tp_size=args.mapping.attn.tp_size,
            ),
        ),
        kernel_page_size=64 if is_draft else 128,
        **model_wide_kwargs(
            args,
            model,
            is_draft,
            kv_cache_dtype=torch.float8_e4m3fn,
            kv_cache_mxfp8=True,
        ),
    )


@pytest.mark.parametrize("is_draft", [False, True])
@pytest.mark.parametrize("linear", [False, True])
def test_profile_factory_preserves_storage_and_composes_linear(
    model_and_args, is_draft, linear
):
    model, args = model_and_args
    model.model_profile = _profile(
        _custom_attention, linear_attention="fixture_linear" if linear else None
    )
    config = _create_attn_config(args, model, is_draft=is_draft)
    assert config.kv_cache_dtype == torch.float8_e4m3fn
    assert config.kv_cache_mxfp8
    assert config.dtype == torch.bfloat16
    assert config.is_draft is is_draft
    assert config.context_len == model.context_len + args.spec_context_pad
    assert config.kernel_page_size == (64 if is_draft else 128)
    assert config.component(MHAConfig).num_attention_heads == model.num_attention_heads
    component = config.component(LinearAttnConfig)
    if linear:
        assert component.layer_ids == (1,)
        assert component.temporal_state_shape == (2, 16, 16)
    else:
        assert component is None


@pytest.mark.parametrize("profiled", [False, True])
@pytest.mark.parametrize("is_draft", [False, True])
def test_no_factory_preserves_architecture_generator(
    model_and_args, profiled, is_draft
):
    model, args = model_and_args
    if profiled:
        model.model_profile = _profile(None, linear_attention=None)
    assert _create_attn_config(args, model, is_draft=is_draft) == MHAConfig.generate(
        args, model, is_draft=is_draft
    )


def test_factory_result_must_be_attention_config(model_and_args):
    model, args = model_and_args
    model.model_profile = _profile(lambda *args: None, linear_attention=None)
    with pytest.raises(TypeError, match="must return AttnConfig"):
        _create_attn_config(args, model)


@pytest.mark.parametrize("is_draft", [False, True])
def test_factory_cannot_change_target_or_draft_role(model_and_args, is_draft):
    def wrong_role(args, model, is_draft):
        return _custom_attention(args, model, not is_draft)

    model, args = model_and_args
    model.model_profile = _profile(wrong_role, linear_attention=None)
    with pytest.raises(
        ValueError,
        match=f"must build the config for is_draft={is_draft}, got is_draft={not is_draft}",
    ):
        _create_attn_config(args, model, is_draft=is_draft)


def test_factory_error_does_not_fall_back(model_and_args):
    def reject(args, model, is_draft):
        raise ValueError("unsupported model storage format")

    model, args = model_and_args
    model.model_profile = _profile(reject, linear_attention=None)
    with pytest.raises(ValueError, match="unsupported model storage format"):
        _create_attn_config(args, model)


def test_factory_cannot_duplicate_linear_component(model_and_args):
    def with_linear(args, model, is_draft):
        config = _custom_attention(args, model, is_draft)
        return dataclasses.replace(
            config,
            components=config.components
            + (LinearAttnConfig.generate(args, model, is_draft),),
        )

    model, args = model_and_args
    model.model_profile = _profile(with_linear, linear_attention="fixture_linear")
    with pytest.raises(ValueError, match="leave linear attention composition"):
        _create_attn_config(args, model)


def test_profile_rejects_non_callable_factory():
    with pytest.raises(TypeError, match="must be callable or None"):
        _profile("factory", linear_attention=None)
