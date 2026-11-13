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

"""Resolved HF sampling policy, including real NPU request-pool wiring."""

import json

import pytest

from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.registry import create_sampling_backend
from tokenspeed.runtime.sampling.sampling_params import (
    _TOP_K_DISABLED,
    SamplingParams,
)
from tokenspeed.runtime.utils.hf_transformers_utils import get_config
from tokenspeed.runtime.utils.server_args import prepare_server_args

CAP = "npu_sampling_top_p_top_k_cap"


def _load(tmp_path, checkpoint_policy, overrides):
    raw = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 8192,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        **checkpoint_policy,
    }
    path = tmp_path / "config.json"
    original = json.dumps(raw)
    path.write_text(original)
    args = prepare_server_args(
        ["--model", str(tmp_path), "--hf-overrides", json.dumps(overrides)]
    )
    config = get_config(
        str(tmp_path),
        trust_remote_code=False,
        model_override_args=json.loads(args.hf_overrides),
    )
    assert path.read_text() == original
    return args, config


@pytest.mark.parametrize(
    "checkpoint_policy,overrides,expected",
    [
        ({}, {}, False),
        ({}, {CAP: True}, True),
        ({}, {CAP: False}, False),
        ({CAP: True}, {}, True),
        ({CAP: True}, {CAP: False}, False),
        ({CAP: False}, {CAP: True}, True),
    ],
)
def test_resolved_hf_policy_reaches_sampling_config(
    tmp_path, checkpoint_policy, overrides, expected
):
    args, hf_config = _load(tmp_path, checkpoint_policy, overrides)
    config = SamplingBackendConfig.from_server_args(
        args,
        hf_config=hf_config,
        max_bs=32,
        max_draft_tokens_per_req=1,
        device="cpu",
    )
    assert config.npu_sampling_top_p_top_k_cap is expected


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
def test_hf_sampling_cap_rejects_non_boolean_values(tmp_path, value):
    args, hf_config = _load(tmp_path, {}, {CAP: value})
    with pytest.raises(ValueError, match=f"{CAP} must be a boolean"):
        SamplingBackendConfig.from_server_args(
            args,
            hf_config=hf_config,
            max_bs=1,
            max_draft_tokens_per_req=1,
            device="cpu",
        )


@pytest.mark.parametrize(
    "overrides,expected",
    [({}, _TOP_K_DISABLED), ({CAP: True}, 4096), ({CAP: False}, _TOP_K_DISABLED)],
)
def test_hf_override_reaches_real_npu_sampling_pool(tmp_path, overrides, expected):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("requires Ascend NPU")
    args, hf_config = _load(tmp_path, {}, overrides)
    backend = create_sampling_backend(
        args,
        hf_config=hf_config,
        max_bs=4,
        max_draft_tokens_per_req=1,
        device="npu",
        max_req_pool_size=4,
        vocab_size=8192,
    )
    params = [
        SamplingParams(top_p=0.95, seed=73),
        SamplingParams(top_p=1, seed=73),
        SamplingParams(temperature=0, top_p=0.95, seed=73),
        SamplingParams(top_k=127, top_p=0.95, seed=73),
    ]
    requested = [p.top_k for p in params]
    for generation, order in enumerate((params, params[::-1])):
        backend.prepare_step(
            [f"hf-cap-{generation}-{row}" for row in range(4)],
            [1, 2, 3, 4],
            order,
            num_tokens_per_req=1,
        )
        effective = [expected, _TOP_K_DISABLED, 1, 127]
        if generation:
            effective.reverse()
        assert backend._top_k_pool[1:].cpu().tolist() == effective
        assert [p.top_k for p in params] == requested


def test_generic_top_k_validation_is_unchanged():
    assert SamplingParams(top_p=0.95).top_k == _TOP_K_DISABLED
    with pytest.raises(ValueError, match="top_k must be < 128"):
        SamplingParams(top_p=0.95, top_k=4096).verify(vocab_size=163840)
