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

"""Device-local NIC mapping for both explicit SM120 DeepEP transports."""

import pytest
from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.buffer import (
    configure_device_nic,
    nic_for_visible_device,
)


@pytest.mark.parametrize("device", range(4))
def test_nic_selection_follows_visible_device_order(device):
    assert (
        nic_for_visible_device("rdma_c, rdma_a,rdma_d,rdma_b", device, 4)
        == ("rdma_c", "rdma_a", "rdma_d", "rdma_b")[device]
    )


@pytest.mark.parametrize(
    "names,count", [("", 1), ("a,,b", 3), ("a,b", 3), ("a,b,c", 2)]
)
def test_supplied_map_requires_all_visible_devices(names, count):
    with pytest.raises(ValueError, match="every visible GPU"):
        nic_for_visible_device(names, 0, count)


@pytest.mark.parametrize("device", [-1, 2])
def test_device_ordinal_is_process_local(device):
    with pytest.raises(ValueError, match="visible CUDA device"):
        nic_for_visible_device("a,b", device, 2)


@pytest.mark.parametrize("wire", ["elastic_fp8_bf16", "nvfp4_fp8"])
def test_common_map_sets_both_vendor_variables(monkeypatch, wire):
    monkeypatch.setenv("TOKENSPEED_SM120_DEEPEP_NICS", "rdma_b,rdma_a")
    monkeypatch.setenv("TOKENSPEED_WIRE1_NICS", "old_a,old_b")
    monkeypatch.setattr("torch.cuda.current_device", lambda: 1)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.delenv("EP_NIC_NAME", raising=False)
    monkeypatch.delenv("NVSHMEM_HCA_LIST", raising=False)
    assert configure_device_nic(wire) == "rdma_a"
    import os

    assert os.environ["EP_NIC_NAME"] == "rdma_a"
    assert os.environ["NVSHMEM_HCA_LIST"] == "rdma_a:1"


def test_wire1_old_variable_remains_an_alias(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_SM120_DEEPEP_NICS", raising=False)
    monkeypatch.setenv("TOKENSPEED_WIRE1_NICS", "rdma_b,rdma_a")
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.delenv("EP_NIC_NAME", raising=False)
    monkeypatch.delenv("NVSHMEM_HCA_LIST", raising=False)
    assert configure_device_nic("nvfp4_fp8") == "rdma_b"


def test_absent_map_preserves_vendor_auto_selection(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_SM120_DEEPEP_NICS", raising=False)
    monkeypatch.delenv("TOKENSPEED_WIRE1_NICS", raising=False)
    monkeypatch.setenv("EP_NIC_NAME", "vendor_choice")
    monkeypatch.setenv("NVSHMEM_HCA_LIST", "vendor_choice:1")
    assert configure_device_nic("elastic_fp8_bf16") is None
    import os

    assert os.environ["EP_NIC_NAME"] == "vendor_choice"
    assert os.environ["NVSHMEM_HCA_LIST"] == "vendor_choice:1"
