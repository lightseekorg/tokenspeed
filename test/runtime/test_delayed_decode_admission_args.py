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

"""The prefill role's decode allocation policy: early unless the launch
chooses independent, refused on the other roles, and independent only where
it runs."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.engine import rust_host  # noqa: E402
from tokenspeed.runtime.utils.server_args import (  # noqa: E402
    ServerArgs,
    prepare_server_args,
)

INDEPENDENT = dict(
    model="unused",
    disaggregation_mode="prefill",
    disaggregation_decode_allocation_policy="independent",
)


def test_prefill_admits_early_unless_the_launch_chooses():
    args = prepare_server_args(
        ["--model", "unused", "--disaggregation-mode", "prefill"]
    )
    assert args.disaggregation_decode_allocation_policy == "early"
    # Early keeps streaming the KV layer by layer, as before the flag.
    assert args.disaggregation_layerwise_interval == 1


@pytest.mark.parametrize("policy", ["early", "independent"])
def test_prefill_keeps_the_chosen_policy(policy):
    args = prepare_server_args(
        [
            "--model",
            "unused",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-decode-allocation-policy",
            policy,
            "--disaggregation-layerwise-interval",
            "0",
        ]
    )
    assert args.disaggregation_decode_allocation_policy == policy
    assert args.disaggregation_layerwise_interval == 0


def test_prefill_refuses_an_unknown_policy():
    with pytest.raises(ValueError, match=r"must be one of \['early', 'independent'\]"):
        ServerArgs(
            model="unused",
            disaggregation_mode="prefill",
            disaggregation_decode_allocation_policy="lazy",
        )


@pytest.mark.parametrize("role", ["null", "decode", "encode"])
@pytest.mark.parametrize("policy", ["early", "independent"])
def test_other_roles_refuse_any_policy(role, policy):
    assert (
        ServerArgs(
            model="unused", disaggregation_mode=role
        ).disaggregation_decode_allocation_policy
        is None
    )
    with pytest.raises(ValueError, match="requires --disaggregation-mode prefill"):
        ServerArgs(
            model="unused",
            disaggregation_mode=role,
            disaggregation_decode_allocation_policy=policy,
        )


def test_independent_admission_refuses_layerwise_transfer_without_overriding_it():
    with pytest.raises(
        ValueError, match="--disaggregation-layerwise-interval other than 0"
    ):
        ServerArgs(**INDEPENDENT)


@pytest.mark.parametrize(
    "layout, refused",
    [
        (dict(world_size=2, attn_tp_size=2), "attention TP > 1"),
        (dict(world_size=2, kv_parallel_size=2), "--kv-parallel-size > 1"),
        (
            dict(world_size=2, pipeline_parallel_size=2),
            "--pipeline-parallel-size > 1",
        ),
        (dict(enable_external_encoder=True), "--enable-external-encoder"),
    ],
)
def test_independent_admission_refuses_unsupported_layouts(layout, refused):
    with pytest.raises(ValueError, match=refused):
        ServerArgs(**INDEPENDENT, disaggregation_layerwise_interval=0, **layout)
    # The same layout admits early.
    args = ServerArgs(
        model="unused",
        disaggregation_mode="prefill",
        disaggregation_layerwise_interval=0,
        **layout,
    )
    assert args.disaggregation_decode_allocation_policy == "early"


def test_independent_admission_names_every_unsupported_setting():
    with pytest.raises(ValueError) as refusal:
        ServerArgs(**INDEPENDENT, world_size=2, attn_tp_size=2)
    assert "attention TP > 1" in str(refusal.value)
    assert "--disaggregation-layerwise-interval other than 0" in str(refusal.value)


@pytest.mark.parametrize(
    "policy, multimodal",
    [("independent", True), ("independent", False), ("early", True)],
)
def test_a_prefill_rank_refuses_independent_admission_for_a_multimodal_model(
    monkeypatch, policy, multimodal
):
    # Whether the model takes multimodal input is known once the rank read
    # its config, which stands in here for a checkpoint.
    monkeypatch.setattr(
        rust_host,
        "load_model_config",
        lambda *args, **kwargs: SimpleNamespace(is_multimodal_active=multimodal),
    )
    server_args = ServerArgs(
        model="unused",
        disaggregation_mode="prefill",
        disaggregation_decode_allocation_policy=policy,
        disaggregation_layerwise_interval=0,
    )
    server_args.mapping.rank = 0
    if policy == "independent" and multimodal:
        with pytest.raises(
            ValueError,
            match="independent does not support a multimodal model; serve it "
            "with --language-model-only",
        ):
            rust_host.HostBootstrap(server_args, None, None, None, None)
    else:
        # Past the refusal, the rank reads what the stand-in config lacks.
        with pytest.raises(AttributeError):
            rust_host.HostBootstrap(server_args, None, None, None, None)
