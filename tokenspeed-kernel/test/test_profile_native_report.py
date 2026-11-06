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

"""Behavior of derived native labels; not an Nsight measurement test."""

import copy

import pytest
from tokenspeed_kernel.profile_native_report import _scope_message


@pytest.mark.parametrize("flops", [0, 48])
def test_native_communication_label_prioritizes_directional_utilization(flops):
    row = dict(
        kind="op",
        native_nvtx_label="communication.all_reduce",
        iter_id=100,
        layer_id=2,
        step_id="fixture:r0:s100",
        forward_id=100,
        gpu_duration_ns=3488,
        collective_count=1,
        useful_flops=flops,
        estimated_mbu=0.0014021634944543337,
        mbu_missing_reason=None,
        mfu=0.1 if flops else 0,
        mfu_total=0.1 if flops else 0,
        mfu_total_missing_reason=None,
        mfu_by_compute_type={},
        estimated_comm_utilization=0.08220183486238532,
        algbw_bytes_per_second=2348623853.211009,
        busbw_bytes_per_second=4110091743.1192656,
        communication_bottleneck_direction="rx",
        communication_payload_model={"id": "fixture_directional_payload"},
    )
    original = copy.deepcopy(row)
    message = _scope_message(row)
    assert message.startswith(
        "communication.all_reduce | Est.CommBWUtil=8.22% | GPU=3.488us | "
    )
    assert "Est.MBU(HBM)=0.14%" in message
    assert (
        message.index("Est.CommBWUtil=8.22%")
        < message.index("Est.MBU(HBM)=0.14%")
        < message.index("commAlgBW=")
    )
    assert "commBusBW=" in message and "bottleneck=rx" in message
    assert "whole Op GPU span" in message
    assert ("MFU_total=" in message) is bool(flops)
    assert row == original
    row["estimated_comm_utilization"] = None
    row["estimated_comm_utilization_missing_reason"] = "fixture missing mapping"
    unavailable = _scope_message(row)
    assert unavailable.startswith(
        "communication.all_reduce | Est.CommBWUtil=N/A | GPU=3.488us | "
    )
    assert "fixture missing mapping" in unavailable
    assert "Est.MBU(HBM)=0.14%" in unavailable


def test_native_label_reader_remains_independent_of_torch():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tokenspeed_kernel.profile_native_report; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
