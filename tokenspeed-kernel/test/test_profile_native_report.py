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
def test_native_communication_label_prioritizes_bandwidth(flops):
    row = dict(
        kind="op",
        native_nvtx_label=(
            "communication.all_reduce_residual_rmsnorm"
            if flops
            else "communication.all_reduce"
        ),
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
        estimated_comm_tx_payload_bytes_per_second=2348623853.211009,
        estimated_comm_rx_payload_bytes_per_second=16440366972.477064,
        estimated_comm_reference_efficiency=0.011,
        scheduler_status=dict(
            completion_status="token_committed",
            scheduled=True,
            gpu_observed=True,
            device_ready=True,
            authoritative_commit=True,
            token_commits=[],
        ),
        communication_bottleneck_direction="rx",
        communication_payload_model={"id": "fixture_directional_payload"},
    )
    original = copy.deepcopy(row)
    message = _scope_message(row)
    assert message.startswith(
        f"{row['native_nvtx_label']} | Est.CommBWUtil=8.22% | "
        "commAlgBW=2.35GB/s | commBusBW=4.11GB/s | "
        "Est.TXPayloadBW=2.35GB/s | Est.RemoteRXPayloadBW=16.4GB/s | GPU=3.488us | "
    )
    assert message.index("GPU=3.488us") < message.index("Est.MBU(HBM)=0.14%")
    assert message.index("Est.MBU(HBM)=0.14%") < message.index("status=token_committed")
    assert message.count(" | commAlgBW=") == message.count(" | commBusBW=") == 1
    assert "CommRefEff=1.1%" in message and "bottleneck=rx" in message
    assert "whole Op GPU span" in message
    assert "commAlgBW/commBusBW=NCCL-tests reference" in message
    if flops:
        assert message.index("GPU=3.488us") < message.index("MFU_total=")
    else:
        assert "MFU_total=" not in message
    assert row == original
    row["estimated_comm_utilization"] = None
    row["busbw_bytes_per_second"] = None
    row["estimated_comm_rx_payload_bytes_per_second"] = None
    row["estimated_comm_utilization_missing_reason"] = "fixture missing mapping"
    unavailable = _scope_message(row)
    assert unavailable.startswith(
        f"{row['native_nvtx_label']} | Est.CommBWUtil=N/A | "
        "commAlgBW=2.35GB/s | commBusBW=N/A | "
        "Est.TXPayloadBW=2.35GB/s | Est.RemoteRXPayloadBW=N/A | GPU=3.488us | "
    )
    assert "fixture missing mapping" in unavailable
    assert "Est.MBU(HBM)=0.14%" in unavailable

    row.pop("communication_payload_model")
    row.pop("estimated_comm_tx_payload_bytes_per_second")
    row.pop("estimated_comm_rx_payload_bytes_per_second")
    historical = _scope_message(row)
    assert "Est.TXPayloadBW=" not in historical
    assert "Est.RemoteRXPayloadBW=" not in historical
    assert historical.index("commAlgBW=") < historical.index("GPU=")


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
