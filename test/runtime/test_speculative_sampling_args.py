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

"""``--enable-speculative-sampling`` resolution: what it needs, what it refuses."""

import pytest

from tokenspeed.runtime.utils.env import (
    global_server_args_dict,
    global_server_args_dict_update,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


@pytest.fixture(autouse=True)
def _restore_global_server_args():
    # global_server_args_dict_update mutates process-global state other tests
    # (the logits processor's argmax gate) read.
    saved = dict(global_server_args_dict)
    yield
    global_server_args_dict.clear()
    global_server_args_dict.update(saved)


def _args(**overrides) -> ServerArgs:
    kwargs = dict(
        model="x",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="draft",
        sampling_backend="flashinfer",
        enable_speculative_sampling=True,
    )
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


def test_default_is_off_and_published():
    args = ServerArgs(model="x")
    assert args.enable_speculative_sampling is False
    assert args.spec_reject_draft_prob_threshold == 2.0
    global_server_args_dict_update(args)
    assert global_server_args_dict["enable_speculative_sampling"] is False
    assert global_server_args_dict["spec_reject_draft_prob_threshold"] == 2.0


@pytest.mark.parametrize("algorithm", ["EAGLE3", "MTP"])
@pytest.mark.parametrize("backend", ["flashinfer", "flashinfer_full"])
def test_chain_drafters_with_a_draft_prob_verifier_resolve(algorithm, backend):
    args = _args(
        speculative_algorithm=algorithm,
        sampling_backend=backend,
        spec_reject_draft_prob_threshold=1.5,
    )
    assert args.enable_speculative_sampling
    global_server_args_dict_update(args)
    assert global_server_args_dict["enable_speculative_sampling"] is True
    assert global_server_args_dict["spec_reject_draft_prob_threshold"] == 1.5


def test_rl_bitwise_accepts_the_flag_with_a_flashinfer_verifier():
    args = _args(numerics="rl-bitwise")
    assert args.enable_speculative_sampling and args.sampling_backend == "flashinfer"


def test_refuses_without_speculative_decoding():
    with pytest.raises(ValueError, match="needs speculative decoding"):
        ServerArgs(model="x", enable_speculative_sampling=True)


@pytest.mark.parametrize("algorithm", ["DFLASH", "DSPARK"])
def test_refuses_block_drafters(algorithm):
    with pytest.raises(ValueError, match="whole block greedily"):
        _args(speculative_algorithm=algorithm)


def test_refuses_the_greedy_verifier():
    with pytest.raises(ValueError, match="greedy verifies by exact match"):
        _args(sampling_backend="greedy")


@pytest.mark.parametrize("backend", ["triton", "triton_full"])
def test_refuses_the_triton_verifiers(backend):
    with pytest.raises(ValueError, match="target-sampled exact match"):
        _args(sampling_backend=backend)


def test_refuses_a_threshold_a_real_probability_could_reach():
    with pytest.raises(ValueError, match="spec-reject-draft-prob-threshold"):
        _args(spec_reject_draft_prob_threshold=0.9)
    # Exactly 1.0 is the lowest safe value: no probability exceeds it.
    assert _args(spec_reject_draft_prob_threshold=1.0).spec_reject_draft_prob_threshold == 1.0


def test_cli_flags_parse():
    import argparse

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns = parser.parse_args(
        [
            "--model",
            "x",
            "--enable-speculative-sampling",
            "--spec-reject-draft-prob-threshold",
            "3.5",
        ]
    )
    assert ns.enable_speculative_sampling is True
    assert ns.spec_reject_draft_prob_threshold == 3.5
    ns = parser.parse_args(["--model", "x"])
    assert ns.enable_speculative_sampling is False
    assert ns.spec_reject_draft_prob_threshold == 2.0
