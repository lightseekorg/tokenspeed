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

"""The explicit prefill policy selector and its compatibility entry point."""

import pytest

from tokenspeed.runtime.utils.server_args import ServerArgs, prepare_server_args


@pytest.mark.parametrize("policy", ["fifo", "short-prefill-first"])
def test_prefill_policy_is_preserved_from_cli_and_programmatic_args(policy):
    cli_args = prepare_server_args(["--model", "x", "--prefill-policy", policy])
    direct_args = ServerArgs(model="x", prefill_policy=policy)
    assert cli_args.prefill_policy == direct_args.prefill_policy == policy
    assert not cli_args.enable_short_prefill_first
    assert not direct_args.enable_short_prefill_first


@pytest.mark.parametrize("enabled", [False, True])
def test_omitted_prefill_policy_keeps_the_legacy_selector(enabled):
    flags = ["--enable-short-prefill-first"] if enabled else []
    args = prepare_server_args(["--model", "x", *flags])
    assert args.prefill_policy is None
    assert args.enable_short_prefill_first is enabled


@pytest.mark.parametrize("policy", ["fifo", "short-prefill-first"])
def test_prefill_policy_and_legacy_flag_are_rejected_together(policy):
    with pytest.raises(
        ValueError, match="--prefill-policy.*--enable-short-prefill-first"
    ):
        prepare_server_args(
            ["--model", "x", "--prefill-policy", policy, "--enable-short-prefill-first"]
        )
    with pytest.raises(
        ValueError, match="--prefill-policy.*--enable-short-prefill-first"
    ):
        ServerArgs(model="x", prefill_policy=policy, enable_short_prefill_first=True)


@pytest.mark.parametrize(
    "policy", ["unknown", "fair-share", "FIFO", "ShortPrefillFirst"]
)
def test_prefill_policy_rejects_an_unsupported_selection(policy):
    with pytest.raises(ValueError, match="--prefill-policy"):
        ServerArgs(model="x", prefill_policy=policy)
    with pytest.raises(SystemExit):
        prepare_server_args(["--model", "x", "--prefill-policy", policy])


def test_prefill_policy_revalidates_after_configuration_changes():
    args = ServerArgs(model="x", enable_short_prefill_first=True)
    args.prefill_policy = "fifo"
    with pytest.raises(
        ValueError, match="--prefill-policy.*--enable-short-prefill-first"
    ):
        args.validate()


@pytest.mark.parametrize("policy", ["fifo", "short-prefill-first"])
def test_prefill_policy_preserves_the_legacy_chunk_yield(policy):
    args = prepare_server_args(
        [
            "--model",
            "x",
            "--disaggregation-mode",
            "prefill",
            "--chunked-prefill-size",
            "64",
            "--prefill-policy",
            policy,
            "--prefill-chunk-yield-tokens",
            "16",
        ]
    )
    assert args.prefill_policy == policy
    assert args.prefill_chunk_yield_tokens == 16
