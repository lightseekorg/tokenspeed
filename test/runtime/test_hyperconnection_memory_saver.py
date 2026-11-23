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

"""GPU release/resume checks for the hyper-connection mix's persistent workspaces."""

import subprocess
import sys
from pathlib import Path
from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch

register_cuda_ci(est_time=30, suite="runtime-1gpu")


def _ordinary_bytes() -> int:
    """Bytes held by ordinary Torch CUDA allocator segments on this device."""
    return sum(
        segment["total_size"]
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == torch.cuda.current_device()
        and tuple(segment["segment_pool_id"]) == (0, 0)
    )


def _check_mix_workspaces_survive_release() -> None:
    """Release and resume warmed and later-grown fused mix workspaces on CUDA."""
    from tokenspeed_kernel.ops.residual import cute_fused
    from tokenspeed_kernel.ops.tuning import kernel_warmup

    from tokenspeed.runtime.layers.hyperconnection import (
        GatedResidualSimple,
        HyperConnectionConfig,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict
    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    global_server_args_dict["enable_memory_saver"] = True
    saver = TorchMemorySaverAdapter.create(enable=True)
    config = HyperConnectionConfig(hc_count=4, hidden_size=2560, hc_lowrank=320)
    # Backed-up weights let resumed forwards reuse the loaded parameters.
    with saver.region(tag="weights", enable_cpu_backup=True), torch.device("cuda"):
        mixer = GatedResidualSimple(config).to(torch.bfloat16)
        generator = torch.Generator(device="cuda").manual_seed(83)
        projection = (
            torch.randn(324, 10240, dtype=torch.bfloat16, generator=generator) * 0.01
        )
        up = torch.randn(10240, 320, dtype=torch.bfloat16, generator=generator) * 0.01
        param = mixer.mix_inject_proj.weight
        param.weight_loader(param, projection[:320], "mix")
        param.weight_loader(param, projection[320:], "inject")
        up_param = mixer.input_mix_weight_up.weight
        up_param.weight_loader(up_param, up)
        del projection, up
    with saver.named_region("executor_inputs"):
        hyper_input = torch.randn(
            256, 10240, dtype=torch.bfloat16, device="cuda", generator=generator
        )

    @torch.inference_mode()
    def forward(rows: int, warm: bool) -> tuple[torch.Tensor, torch.Tensor]:
        with kernel_warmup(enabled=warm):
            mixed, residuals = mixer.mix(hyper_input[:rows], normalized=None)
        return mixed.cpu(), residuals[2].cpu()

    expected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    # Eager calls reach new layouts lazily; a later kernel warmup walks the rest.
    for rows, warm in ((1, False), (200, False), (1, True)):
        layouts = len(cute_fused._WORKSPACES)
        expected[rows] = forward(rows, warm)
        assert len(cute_fused._WORKSPACES) > layouts
        for _ in range(3):
            torch.cuda.synchronize()
            retained = [
                (tensor, tensor.data_ptr(), tensor.cpu())
                for workspace in cute_fused._WORKSPACES.values()
                for tensor in workspace
            ]
            saver.pause(tag="weights")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ordinary_bytes = _ordinary_bytes()
            assert ordinary_bytes == 0, ordinary_bytes
            saver.resume(tag="weights")
            for tensor, pointer, reference in retained:
                assert tensor.data_ptr() == pointer
                torch.testing.assert_close(tensor.cpu(), reference, rtol=0, atol=0)
            for count, outputs in expected.items():
                torch.testing.assert_close(
                    forward(count, False), outputs, rtol=0, atol=0
                )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires native CUDA VMM memory-saver hooks",
)
def test_mix_workspaces_survive_release_and_later_growth() -> None:
    memory_saver = pytest.importorskip("torch_memory_saver")
    from tokenspeed_kernel.ops.residual import cute_fused

    if not cute_fused.supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires six resident Blackwell clusters and CuTe")
    # Preload hooks must be installed before CUDA initializes in the child.
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from test.runtime.test_hyperconnection_memory_saver import "
                "_check_mix_workspaces_survive_release; "
                "_check_mix_workspaces_survive_release()",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=600,
        )
