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

"""Device-level lifecycle checks for GLM's single-copy block-FP8 weight."""

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.base_config import (
    finalize_quantized_weights_after_loading,
)
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.utils.env import global_server_args_dict


def _make_packed_layer(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_EXPERIMENTAL_GLUON_FP8_BLOCKSCALE", "1")
    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", "triton")
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )
    layer = ReplicatedLinear(4096, 1024, bias=False, quant_config=config).cuda()
    canonical = (torch.randn_like(layer.weight.float()) * 0.05).to(layer.weight.dtype)
    layer.weight.data.copy_(canonical)
    layer.weight_scale_inv.data.fill_(1.0)
    layer.quant_method.packed_resident_requested = True
    layer.quant_method.process_weights_after_loading(layer)
    layer.quant_method.promote_weight_after_loading(layer)
    plan = layer._prepared_fp8_linear
    assert plan.packed_resident and plan.packed_weight_ptr == layer.weight.data_ptr()
    assert list(plan.parameters()) == [] and list(plan.buffers()) == []
    return layer, canonical


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_packed_weight_refresh_and_graph_replay(monkeypatch) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("requires gfx950")

    layer, canonical = _make_packed_layer(monkeypatch)
    plan = layer._prepared_fp8_linear

    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    original = layer(x)[0]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = layer(x)[0]
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replay_output, original, atol=0.1, rtol=0.1)

    pointer = layer.weight.data_ptr()
    torch.cuda.reset_peak_memory_stats()
    before_scale = torch.cuda.memory_allocated()
    layer.weight_scale_inv.weight_loader(
        layer.weight_scale_inv, torch.full_like(layer.weight_scale_inv, 1.5)
    )
    finalize_quantized_weights_after_loading(layer)
    torch.cuda.synchronize()
    scale_peak = torch.cuda.max_memory_allocated() - before_scale
    assert layer.weight.data_ptr() == pointer
    assert scale_peak < layer.weight.numel()

    updated = torch.zeros_like(canonical)
    torch.cuda.reset_peak_memory_stats()
    before_weight = torch.cuda.memory_allocated()
    layer.weight.weight_loader(layer.weight, updated)
    finalize_quantized_weights_after_loading(layer)
    torch.cuda.synchronize()
    weight_peak = torch.cuda.max_memory_allocated() - before_weight
    assert layer.weight.data_ptr() == pointer
    assert weight_peak < 4 * layer.weight.numel()
    assert torch.equal(layer.state_dict()["weight"], updated)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replay_output, torch.zeros_like(replay_output))

    layer.to("cpu").to("cuda")
    assert plan.packed_weight_ptr == layer.weight.data_ptr()
    assert torch.equal(layer.state_dict()["weight"], updated)
    assert torch.equal(layer(x)[0], torch.zeros_like(original))
    print(f"scale_only_peak_bytes={scale_peak} one_weight_peak_bytes={weight_peak}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_partial_packed_refresh_restores_captured_graph(monkeypatch) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("requires gfx950")

    layer, canonical = _make_packed_layer(monkeypatch)
    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = layer(x)[0]
    graph.replay()
    torch.cuda.synchronize()
    previous_output = replay_output.clone()
    previous_scale = layer.weight_scale_inv.detach().clone()
    packed_pointer = layer.weight.data_ptr()

    layer.weight.weight_loader(layer.weight, torch.zeros_like(canonical))
    layer.weight_scale_inv.weight_loader(
        layer.weight_scale_inv, torch.full_like(previous_scale, 1.5)
    )

    original_copy = torch.Tensor.copy_
    resident_copies = 0

    def fail_after_partial_copy(self, source, *args, **kwargs):
        nonlocal resident_copies
        if self.data_ptr() == packed_pointer:
            resident_copies += 1
            if resident_copies == 1:
                original_copy(self.flatten()[:64], source.flatten()[:64])
                raise RuntimeError("injected partial packed copy")
        return original_copy(self, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_after_partial_copy)
    with pytest.raises(RuntimeError, match="old weight restored"):
        finalize_quantized_weights_after_loading(layer)

    assert resident_copies == 2
    assert layer._prepared_fp8_linear.packed_valid
    assert layer.weight.data_ptr() == packed_pointer
    assert torch.equal(layer.state_dict()["weight"], canonical)
    assert torch.equal(layer.weight_scale_inv, previous_scale)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replay_output, previous_output, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_online_scale_update_waits_for_in_flight_graph(monkeypatch) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("requires gfx950")

    layer, _ = _make_packed_layer(monkeypatch)
    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    old_output = layer(x)[0].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = layer(x)[0]
    torch.cuda.synchronize()

    class ScaleOnlyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = layer

        def load_weights(self, weights):
            for name, value in weights:
                assert name == "linear.weight_scale_inv"
                self.linear.weight_scale_inv.weight_loader(
                    self.linear.weight_scale_inv, value
                )

    updated_scale = torch.full_like(layer.weight_scale_inv, 2.0)
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = ScaleOnlyModel()
    runner._weight_update_pg = object()
    runner._weight_update_device = torch.device("cuda")
    request = SimpleNamespace(
        names=["linear.weight_scale_inv"],
        dtype_names=[str(updated_scale.dtype).split(".")[-1]],
        shapes=[updated_scale.shape],
    )
    graph_finished_before_write = []

    def broadcast(buffer, *, src, group):
        assert src == 0 and group is runner._weight_update_pg
        graph_finished_before_write.append(graph_done.query())
        buffer.copy_(updated_scale)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    delay_stream = torch.cuda.Stream()
    execution_stream = torch.cuda.Stream()
    update_stream = torch.cuda.Stream()
    delay_done = torch.cuda.Event()
    graph_done = torch.cuda.Event()
    with torch.cuda.stream(delay_stream):
        torch.cuda._sleep(2_000_000_000)
        delay_done.record()
    with torch.cuda.stream(execution_stream):
        execution_stream.wait_event(delay_done)
        graph.replay()
        graph_done.record()
    assert not graph_done.query()

    with torch.cuda.stream(update_stream):
        success, _ = runner.update_weights_from_distributed(request)
    assert success and graph_done.query()
    assert graph_finished_before_write == [True]
    torch.testing.assert_close(replay_output, old_output, atol=0.1, rtol=0.1)

    new_output = layer(x)[0]
    assert not torch.allclose(new_output, old_output)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replay_output, new_output, atol=0.1, rtol=0.1)
