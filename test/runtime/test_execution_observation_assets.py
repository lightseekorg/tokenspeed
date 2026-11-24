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

"""CPU checks of local serving assets and real numerical-policy resolution."""

import json
from dataclasses import replace
from pathlib import Path
from test.execution_observation_assets import create_tiny_llama, resolve_qualification

import pytest
import torch


@pytest.fixture(scope="module")
def local_args(tmp_path_factory):
    return create_tiny_llama(tmp_path_factory.mktemp("observation-assets") / "target")


def test_synthetic_checkpoint_and_tokenizer_load_offline(local_args, monkeypatch):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer

    def no_cuda():
        raise AssertionError("asset checks must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    path = local_args[local_args.index("--model") + 1]
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    weights = load_file(f"{path}/model.safetensors", device="cpu")
    assert config.architectures == ["LlamaForCausalLM"]
    assert config.max_position_embeddings >= 1024
    assert tokenizer.encode("token17 token31", add_special_tokens=False) == [17, 31]
    assert weights["model.layers.0.self_attn.q_proj.weight"].shape == (256, 256)
    assert weights["lm_head.weight"].shape == (256, 256)
    assert all(
        value.device.type == "cpu" and value.dtype == torch.bfloat16
        for value in weights.values()
    )
    assert torch.count_nonzero(weights["lm_head.weight"]) > 0


@pytest.mark.parametrize(
    "speculation",
    [[], ["--speculative-algorithm", "DUMMY", "--speculative-num-draft-tokens", "8"]],
)
def test_auto_fixture_resolves_ordinary_w1_and_w8_without_rewriting_argv(
    local_args, speculation
):
    args = local_args + speculation
    original = list(args)
    report = resolve_qualification(args)
    assert args == original
    assert report["architectures"] == ["LlamaForCausalLM"]
    assert report["numerics"] == "auto"
    assert report["bitwise_contract"] is False


def test_unprofiled_dense_fixture_does_not_claim_rl_bitwise(local_args):
    args = list(local_args)
    args[args.index("--numerics") + 1] = "rl-bitwise"
    with pytest.raises(ValueError, match="has not been verified"):
        resolve_qualification(args)


def test_qualification_requires_an_explicit_policy(local_args):
    args = list(local_args)
    index = args.index("--numerics")
    del args[index : index + 2]
    with pytest.raises(ValueError, match="explicitly name"):
        resolve_qualification(args)


@pytest.mark.parametrize("algorithm", ["none", "DUMMY"])
def test_generated_recipe_resolves_the_real_serving_launch_matrix(
    local_args, monkeypatch, algorithm
):
    from test.test_execution_observations import serving_args

    monkeypatch.setenv("TOKENSPEED_EXECUTION_TEST_ARGS", json.dumps(local_args))
    for route in ("eager", "decode_graph", "prefill_breakable", "over_ladder"):
        report = resolve_qualification(
            serving_args(
                algorithm, route, base_variable="TOKENSPEED_EXECUTION_TEST_ARGS"
            )
        )
        assert report["architectures"] == ["LlamaForCausalLM"]
        assert report["numerics"] == "auto"


def test_registered_policy_uses_its_resolver_and_model_validator(local_args, tmp_path):
    from tokenspeed.runtime.configs.model_config import configure_mha_attention
    from tokenspeed.runtime.configs.model_profile import ModelProfile
    from tokenspeed.runtime.configs.numerics import register_numerics_envelope
    from tokenspeed.runtime.plugins import ensure_loaded
    from tokenspeed.runtime.plugins.registry import recording, register_model

    ensure_loaded()
    name = "observation-policy-unit-test"

    def resolve(args):
        args.sampling_stream = "per-request"

    def validate(*, quantization, vocab_size):
        if quantization is not None or vocab_size != 256:
            raise ValueError(
                "unit-test policy requires an unquantized 256-token vocabulary"
            )

    profile = ModelProfile(
        configure_attention=configure_mha_attention,
        cache_family="mha",
        linear_attention=None,
        default_attention_backend=None,
        request_token_history=False,
        is_multimodal=False,
        tokenizer_kwargs={},
        attention_instances_per_layer=1,
        numerics_envelopes=frozenset({"auto"}),
        create_attention_config=None,
    )

    class PolicyTestModel(torch.nn.Module):
        @classmethod
        def model_profile(cls, hf_config):
            return profile

    class Rollback(Exception):
        pass

    # This is a configuration-gate fixture, never a serving model declaration.
    target = Path(local_args[local_args.index("--model") + 1])
    config = json.loads((target / "config.json").read_text())
    config["architectures"] = ["PolicyTestModel"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = list(local_args)
    args[args.index("--model") + 1] = str(tmp_path)
    args[args.index("--numerics") + 1] = name
    with pytest.raises(Rollback):
        with recording():
            register_numerics_envelope(
                name, bitwise=True, resolve=resolve, validate=validate
            )
            profile = replace(profile, numerics_envelopes=frozenset({"auto", name}))
            register_model(PolicyTestModel)
            report = resolve_qualification(args)
            assert report["numerics"] == name
            assert report["bitwise_contract"] is True
            assert report["sampling_stream"] == "per-request"
            profile = replace(profile, numerics_envelopes=frozenset({"auto"}))
            with pytest.raises(ValueError, match="has not been verified"):
                resolve_qualification(args)
            profile = replace(profile, numerics_envelopes=frozenset({"auto", name}))
            config["vocab_size"] = 128
            (tmp_path / "config.json").write_text(json.dumps(config))
            with pytest.raises(ValueError, match="256-token vocabulary"):
                resolve_qualification(args)
            raise Rollback()
