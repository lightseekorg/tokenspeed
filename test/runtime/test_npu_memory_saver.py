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

"""Run with a built saver library on one exclusively assigned NPU."""

import ctypes
import os

import pytest


@pytest.mark.skipif(
    not os.environ.get("TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY"),
    reason="requires the native Ascend saver library and an NPU",
)
def test_graph_discard_preserves_persistent_buffers():
    import torch

    pytest.importorskip("torch_npu")
    from tokenspeed.runtime.utils.ascend_memory_saver_adapter import (
        AscendMemorySaverAdapter,
    )

    torch.npu.set_device(0)
    saver = AscendMemorySaverAdapter()
    graph = torch.npu.NPUGraph()
    with saver.named_region("attention_state"):
        constant = torch.full((128,), 7.0, device="npu")
    inputs = torch.arange(128, device="npu", dtype=torch.float32)
    for _ in range(3):
        inputs * inputs + constant
    torch.npu.synchronize()

    def blocks():
        stats = saver.stats()
        assert not stats["callback_fault"]
        return [block for block in stats["blocks"] if block["tag"] == "acl_graph"]

    try:
        with saver.npu_graph(
            graph,
            pool=None,
            stream=torch.npu.Stream(),
            tag="acl_graph",
            policy="discard",
            pool_scope="test_graph_discard",
            auto_dispatch_capture=True,
        ):
            output = inputs * inputs + constant
        graph.replay()
        torch.npu.synchronize()
        expected = output.cpu()
        addresses = {block["ptr"] for block in blocks()}
        assert addresses
        output_ptr = output.data_ptr()
        memset = ctypes.CDLL("libascendcl.so").aclrtMemset
        memset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        memset.restype = ctypes.c_int
        for _ in range(3):
            saver.pause()
            paused = blocks()
            assert saver.stats()["physical_bytes"] == 0
            assert saver.stats()["valid_backup_bytes"] > 0
            assert {block["ptr"] for block in paused} == addresses
            assert all(
                block["reserved"]
                and not block["physical_live"]
                and not block["mapped"]
                and not block["backup_valid"]
                for block in paused
            )
            saver.resume()
            resumed = blocks()
            assert {block["ptr"] for block in resumed} == addresses
            # Poison only owned discard storage: replay must overwrite all reads.
            for block in resumed:
                assert block["policy"] == "discard" and block["mapped"]
                size = block["aligned_bytes"]
                assert memset(int(block["ptr"], 16), size, 0xA5, size) == 0
            graph.replay()
            torch.npu.synchronize()
            assert output.data_ptr() == output_ptr
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(constant.cpu(), torch.full((128,), 7.0))
    finally:
        saver.resume()
        graph.reset()
        torch.npu.synchronize()
