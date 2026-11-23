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

"""CPU argument tests for bounded debug retraction injection."""

import argparse

import pytest


@pytest.fixture
def args_type(monkeypatch):
    # The parser is real; only unavailable accelerator discovery is supplied.
    from tokenspeed_kernel.platform import ArchVersion, Platform, PlatformInfo

    monkeypatch.setattr(
        Platform,
        "_instance",
        PlatformInfo(
            vendor="nvidia",
            arch_version=ArchVersion(9, 0),
            device_name="CPU fixture",
            device_count=0,
            total_memory=0,
            memory_bandwidth=0,
            sm_count=132,
            max_threads_per_sm=2048,
            max_shared_memory_per_sm=232448,
        ),
    )
    from tokenspeed.runtime.utils.server_args import ServerArgs

    return ServerArgs


@pytest.mark.parametrize("value", [None, 1, 64, 2**31 - 1])
@pytest.mark.parametrize("interval", [3, -2])
def test_optional_cap_preserves_valid_values_for_both_forcing_kinds(
    args_type, value, interval
):
    args = args_type(
        model="fixture",
        debug_force_retraction_interval=interval,
        debug_force_retraction_max_per_request=value,
    )
    assert args.debug_force_retraction_max_per_request == value
    assert args.debug_force_retraction_interval == interval


@pytest.mark.parametrize("value", [0, -1, -(2**31), 2**31, True, 1.5])
def test_explicit_invalid_caps_never_become_omission(args_type, value):
    with pytest.raises(ValueError, match="max-per-request.*integer"):
        args_type(
            model="fixture",
            debug_force_retraction_interval=3,
            debug_force_retraction_max_per_request=value,
        )


def test_cap_requires_forcing_and_a_role_with_recovery_images(args_type):
    assert args_type(model="fixture").debug_force_retraction_max_per_request is None
    with pytest.raises(ValueError, match="max-per-request.*nonzero"):
        args_type(model="fixture", debug_force_retraction_max_per_request=64)
    for role in ("prefill", "encode"):
        with pytest.raises(ValueError, match="max-per-request.*role"):
            args_type(
                model="fixture",
                disaggregation_mode=role,
                debug_force_retraction_interval=3,
                debug_force_retraction_max_per_request=64,
            )
        assert (
            args_type(
                model="fixture",
                disaggregation_mode=role,
                debug_force_retraction_interval=3,
            ).debug_force_retraction_interval
            == 0
        )
    with pytest.raises(ValueError, match="ratio 0"):
        args_type(
            model="fixture",
            retraction_snapshot_ratio=0,
            debug_force_retraction_interval=3,
            debug_force_retraction_max_per_request=64,
        )


def test_cli_distinguishes_omission_and_explicit_zero_before_validation(args_type):
    parser = argparse.ArgumentParser()
    args_type.add_cli_args(parser)
    missing = parser.parse_args(["--model", "fixture"])
    zero = parser.parse_args(
        [
            "--model",
            "fixture",
            "--debug-force-retraction-interval",
            "3",
            "--debug-force-retraction-max-per-request",
            "0",
        ]
    )
    assert missing.debug_force_retraction_max_per_request is None
    assert zero.debug_force_retraction_max_per_request == 0
    assert (
        args_type.from_cli_args(missing).debug_force_retraction_max_per_request is None
    )
    with pytest.raises(ValueError, match="max-per-request"):
        args_type.from_cli_args(zero)
    positive = parser.parse_args(
        [
            "--model",
            "fixture",
            "--debug-force-retraction-interval",
            "-2",
            "--debug-force-retraction-max-per-request",
            "64",
        ]
    )
    resolved = args_type.from_cli_args(positive)
    assert resolved.debug_force_retraction_interval == -2
    assert resolved.debug_force_retraction_max_per_request == 64
