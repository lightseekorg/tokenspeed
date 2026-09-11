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

"""GDN verify must not launch per-layer index arithmetic or buffer zero fills."""

from __future__ import annotations

import os
import sys

import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from test.runtime.test_gdn_state_paging import _ContractPool, _mamba_config_pair

from tokenspeed_kernel.ops.attention.gdn import flashinfer as gdn

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.state.mamba import MambaAttnBackend

pytestmark = pytest.mark.skipif(
    not gdn.is_decode_available(), reason="FlashInfer GDN MTP requires SM90+"
)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("replay_ssm", [False, True])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_verify_has_no_elementwise_launches(
    batch: int, replay_ssm: bool, cuda_graph: bool
) -> None:
    torch.manual_seed(17)
    steps, heads, dim, conv_width = 3, 32, 128, 4
    channels = 3 * heads * dim
    pool_size = 2 * batch + 1
    conv = torch.randn(
        pool_size, channels, conv_width - 1, dtype=torch.bfloat16, device="cuda"
    )
    ssm = torch.randn(pool_size, heads, dim, dim, dtype=torch.float32, device="cuda")
    conv[0].zero_()
    ssm[0].zero_()
    pool = _ContractPool(4, {0: ("linear_attention", conv, ssm)})
    backend = MambaAttnBackend(
        *_mamba_config_pair(
            torch,
            heads=heads,
            head_dim=dim,
            spec_tokens=steps,
            max_bs=batch,
            device="cuda",
            replay_ssm=replay_ssm,
        )
    )
    backend.set_kv_pool(pool)
    backend.init_cuda_graph_state(batch)
    inputs = dict(
        mixed_qkv=torch.randn(
            batch * steps, channels, dtype=torch.bfloat16, device="cuda"
        ),
        conv_weights=torch.randn(
            channels, conv_width, dtype=torch.bfloat16, device="cuda"
        ),
        bias=None,
        activation="silu",
        key_dim=heads * dim,
        value_dim=heads * dim,
        attention_tp_size=1,
        head_k_dim=dim,
        head_v_dim=dim,
        a=torch.randn(batch * steps, heads, dtype=torch.bfloat16, device="cuda"),
        b=torch.randn(batch * steps, heads, dtype=torch.bfloat16, device="cuda"),
        A_log=torch.randn(heads, dtype=torch.float32, device="cuda"),
        dt_bias=torch.randn(heads, dtype=torch.float32, device="cuda"),
        layer_id=0,
        seq_len=batch * steps,
    )
    req_rows = torch.arange(batch, dtype=torch.int32, device="cuda")
    seq_lens = torch.full((batch,), 7, dtype=torch.int32, device="cuda")
    tables = torch.arange(1, pool_size, dtype=torch.int32, device="cuda").view(batch, 2)

    def refresh(actual_bs: int) -> None:
        backend.refresh_decode_metadata(
            batch,
            actual_bs,
            req_rows,
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            for_graph_replay=cuda_graph,
            block_tables={"linear_attention": tables},
        )

    def forward() -> torch.Tensor:
        return backend.forward_decode(
            None,
            None,
            None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=pool,
            bs=batch,
            save_kv_cache=True,
            **inputs,
        )

    refresh(batch)
    for _ in range(2):
        forward()
    graph = None
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward()

    # Change both inputs and live batch size after capture, including idle.
    for actual_bs in dict.fromkeys((batch, max(1, batch - 1), 0)):
        inputs["mixed_qkv"].normal_()
        raw_qkv = inputs["mixed_qkv"].clone()
        refresh(actual_bs)
        expected = forward().clone()
        # causal_conv1d_update overwrites its packed input in place.
        inputs["mixed_qkv"].copy_(raw_qkv)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            if graph is not None:
                graph.replay()
            else:
                output = forward()
            torch.cuda.synchronize()
        events = profile.events()
        kernels = [
            event.name
            for event in events
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert any("gdn" in name.lower() for name in kernels), kernels
        assert not any("elementwise" in name.lower() for name in kernels), kernels
        assert not {"aten::sub", "aten::zeros", "aten::zero_"}.intersection(
            event.name for event in events
        )
        torch.testing.assert_close(
            output[:, : actual_bs * steps], expected[:, : actual_bs * steps]
        )
        assert ssm[0].count_nonzero() == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
