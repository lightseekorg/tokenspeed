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

"""GPU release/resume checks for Prims-TS DSA warmed and eager plan buffers."""

from __future__ import annotations

import subprocess
import sys
from functools import partial
from pathlib import Path
from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch

register_cuda_ci(est_time=90, suite="runtime-1gpu")


def _ordinary_bytes() -> int:
    """Bytes held by ordinary Torch CUDA allocator segments on this device."""
    return sum(
        segment["total_size"]
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == torch.cuda.current_device()
        and tuple(segment["segment_pool_id"]) == (0, 0)
    )


def _pool_bytes(pools: set[tuple[int, int]]) -> int:
    """Bytes held by Torch CUDA allocator segments in ``pools`` on this device."""
    return sum(
        segment["total_size"]
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == torch.cuda.current_device()
        and tuple(segment["segment_pool_id"]) in pools
    )


def _segment_pools(pointers: list[int]) -> set[tuple[int, int]]:
    """Allocator pool ids of the segments holding ``pointers`` on this device."""
    pools = set()
    for segment in torch.cuda.memory_snapshot():
        if segment["device"] != torch.cuda.current_device():
            continue
        first = segment["address"]
        last = first + segment["total_size"]
        if any(first <= pointer < last for pointer in pointers):
            pools.add(tuple(segment["segment_pool_id"]))
    return pools


def _check_dsa_prims_plans_survive_release() -> None:
    """Release and resume warmed and lazily-grown Prims-TS DSA plans on CUDA."""
    from tokenspeed_kernel.ops.attention.dsa import (
        dsa_prefill,
        set_dsa_prims_workspace_region,
    )
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel.platform import pdl_enabled
    from tokenspeed_kernel.thirdparty.flashinfer import dsa_prims

    from tokenspeed.runtime.utils.env import global_server_args_dict
    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    pdl_enabled(True)
    global_server_args_dict["enable_memory_saver"] = True
    saver = TorchMemorySaverAdapter.create(enable=True, device="cuda")
    set_dsa_prims_workspace_region(partial(saver.named_region, "workspace"))

    device = torch.device("cuda")
    heads = 16
    topk = 2048
    kv_slots = 4096
    max_rows = 16

    with saver.named_region("executor_inputs"):
        gen = torch.Generator(device=device).manual_seed(42)
        q = (
            torch.randn((max_rows, heads, 576), device=device, generator=gen) * 0.25
        ).to(torch.float8_e4m3fn)
        kv = (torch.randn((kv_slots, 576), device=device, generator=gen) * 0.25).to(
            torch.float8_e4m3fn
        )
        slots = torch.randint(
            0,
            kv_slots,
            (max_rows, topk),
            device=device,
            dtype=torch.int32,
            generator=gen,
        )
        lens = torch.full((max_rows,), 64, device=device, dtype=torch.int32)
        out = torch.empty((max_rows, heads, 512), device=device, dtype=torch.bfloat16)

    with saver.named_region("workspace"):
        probe = torch.empty(1, device=device)
    workspace_pools = _segment_pools([probe.data_ptr()])

    @torch.inference_mode()
    def run_prefill(rows: int, warm: bool) -> torch.Tensor:
        with kernel_warmup(enabled=warm):
            res = dsa_prefill(
                q=q[:rows],
                kv_cache=kv,
                sparse_kv_cache=None,
                topk_slots=slots[:rows],
                topk_lens=lens[:rows],
                max_seqlen_k=kv_slots,
                qk_nope_head_dim=128,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                softmax_scale=192**-0.5,
                page_size=64,
                logit_cap=0.0,
                k_scale=1.0,
                return_lse=False,
                out=out[:rows],
                solution="flashinfer_prims",
                slot_order="selection",
            )
        return res.cpu()

    # Phase 1: warmup ladder for rows=8 (buckets 1, 2, 4, 8 in _warmed_plans).
    # Phase 2: lazy row-count growth (rows=3 on warmed bucket 4).
    # Phase 3: lazy eager bucket growth (rows=16 in _plans LRU).
    expected: dict[int, torch.Tensor] = {}
    for rows, warm in ((8, True), (3, False), (16, False)):
        expected[rows] = run_prefill(rows, warm)
        all_plans = [*dsa_prims._warmed_plans.values(), *dsa_prims._plans.values()]
        assert all_plans
        plan_tensors = {
            tensor.data_ptr(): tensor
            for plan in all_plans
            for table in (plan.offsets, plan.full_lengths)
            for tensor in table.values()
        }
        assert plan_tensors
        assert _segment_pools(list(plan_tensors)) == workspace_pools

        plan_bytes = sum(t.numel() * t.element_size() for t in plan_tensors.values())
        torch.cuda.synchronize()
        steady_workspace_bytes = _pool_bytes(workspace_pools)
        for _ in range(3):
            torch.cuda.synchronize()
            retained = [(t, t.data_ptr(), t.cpu()) for t in plan_tensors.values()]
            torch.cuda.empty_cache()
            free_before = torch.cuda.mem_get_info()[0]
            saver.pause()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            assert torch.cuda.mem_get_info()[0] - free_before >= plan_bytes
            ordinary_bytes = _ordinary_bytes()
            assert ordinary_bytes == 0, ordinary_bytes
            saver.resume()
            for tensor, pointer, reference in retained:
                assert tensor.data_ptr() == pointer
                torch.testing.assert_close(tensor.cpu(), reference, rtol=0, atol=0)
            for count, ref in expected.items():
                torch.testing.assert_close(
                    run_prefill(count, False), ref, rtol=0, atol=0
                )
            torch.cuda.synchronize()
            assert _pool_bytes(workspace_pools) == steady_workspace_bytes


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires SM100 or SM103 with native CUDA VMM memory-saver hooks",
)
def test_dsa_prims_plans_survive_release_and_later_growth() -> None:
    memory_saver = pytest.importorskip("torch_memory_saver")
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from test.runtime.test_dsa_prims_memory_saver import "
                "_check_dsa_prims_plans_survive_release; "
                "_check_dsa_prims_plans_survive_release()",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=600,
        )
