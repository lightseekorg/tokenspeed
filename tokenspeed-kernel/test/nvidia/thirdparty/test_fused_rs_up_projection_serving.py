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

"""TP8 serving-adapter acceptance; execute directly under torchrun.

Tests the integrated dynamic-pointer adapter against a fixed-input binding
using the same continuous-range configuration. The two layers share raw storage,
own distinct outputs, and capture different latent/residual pointers for A/B
graphs at M4096 and M8192. These are test cases, not a dispatch whitelist.
This is kernel/graph acceptance, not a full-model quality or TTFT certificate.
"""

import argparse
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from fused_rs_up_ag_admission import _query_cluster_admission, cap_smem_precheck
from tokenspeed_kernel.ops.communication.fused_rs_workspace import (
    SharedRsWorkspace,
)
from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection import (
    prepare_medium_fused_rs_up_projection,
)
from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_serving import (
    IntegratedFusedRsUpProjectionServing,
)
from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_serving_config import (
    integrated_fused_rs_serving_config,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_symmetric_up_projection import (
    allocate_symmetric_up_projection_output,
)
from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_shared_down_projection
from up_projection_resources import collect_plan_resources


def admit(plan):
    resources = collect_plan_resources(plan)
    ptx = plan.compiled.__ptx__
    if isinstance(ptx, bytes):
        ptx = ptx.decode()
    check = cap_smem_precheck(resources, ptx, plan.max_active_clusters)
    occupancy = _query_cluster_admission(
        plan.compiled.__cubin__,
        check["ptx_entry"],
        plan.max_active_clusters,
        check["dynamic_smem_upper_bound_bytes"],
    )
    return {"resources": resources, "precheck": check, "occupancy": occupancy}


def check_all_ranks(condition, message):
    flag = torch.tensor(int(condition), device="cuda", dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise AssertionError(message)


def make_case(m, slot, adapters, weights, workspace, device):
    rank = dist.get_rank()
    torch.manual_seed(9231 + m + slot)
    latent = [
        torch.randn(m, 3584, device=device, dtype=torch.bfloat16) for _ in range(2)
    ]
    residual = torch.randn(m, 7168, device=device, dtype=torch.bfloat16)
    torch.manual_seed(825 + m + 7 * slot + 31 * rank)
    producers = [
        (
            torch.randn(m, 768, device=device, dtype=torch.bfloat16),
            torch.randn(7168, 768, device=device, dtype=torch.bfloat16)
            / math.sqrt(768),
        )
        for _ in range(2)
    ]
    references = []
    residual_ref = residual
    for layer in range(2):
        output = allocate_symmetric_up_projection_output(
            dist.group.WORLD, m, device=device
        )
        ref = prepare_medium_fused_rs_up_projection(
            latent[layer],
            weights[layer],
            residual_ref,
            workspace,
            output,
            residual_is_replicated=True,
            tuning=integrated_fused_rs_serving_config(m),
        )
        admit(ref)
        references.append(ref)
        residual_ref = output.tensor

    def run_reference():
        for layer, ref in enumerate(references):
            kimi3_shared_down_projection(
                *producers[layer], out=ref.input_view, solution="torch"
            )
            ref()
        return references[-1].output.tensor

    def run_candidate():
        prefix = residual
        for layer, adapter in enumerate(adapters):
            shared = kimi3_shared_down_projection(
                *producers[layer], out=adapter.input_view(m), solution="torch"
            )
            prefix = adapter(latent[layer], weights[layer], prefix, shared)
        return prefix

    run_reference()
    eager = run_candidate()
    torch.cuda.synchronize()
    check_all_ranks(
        torch.equal(eager, references[-1].output.tensor), "eager fixed/live mismatch"
    )
    check_all_ranks(
        adapters[0].output.tensor.data_ptr() != adapters[1].output.tensor.data_ptr(),
        "layer output alias",
    )

    # An independent old-addmm/AllReduce reference for layer zero, retaining the
    # already qualified absolute and relative-L2 bounds (no relaxed tolerance).
    shared = kimi3_shared_down_projection(*producers[0], out=None, solution="torch")
    owner = slice(rank * 896, (rank + 1) * 896)
    shared[:, owner].add_(residual[:, owner])
    shared[:, owner].addmm_(latent[0], weights[0].t())
    dist.all_reduce(shared)
    actual = adapters[0].output.tensor[:m].float()
    reference = shared.float()
    delta = actual - reference
    max_abs = delta.abs().max()
    relative_l2 = delta.norm() / reference.norm().clamp_min(1e-12)
    check_all_ranks(
        (max_abs <= 0.046875 + 0.01 * reference.abs().max()).item()
        and (relative_l2 <= 0.006).item(),
        "independent old-addmm reference bounds exceeded",
    )
    numerical = {"max_abs": max_abs.item(), "relative_l2": relative_l2.item()}

    for adapter in adapters:
        admit(adapter._plans[m])
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        run_candidate()
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph.replay()
    check_all_ranks(
        torch.equal(adapters[-1].output.tensor[:m], references[-1].output.tensor),
        "capture used stale warmup pointers",
    )
    # Preserve the expected result independently from any captured output.
    expected = references[-1].output.tensor.clone()
    return {
        "m": m,
        "slot": slot,
        "graph": graph,
        "expected": expected,
        "owners": (latent, residual, producers, references),
        "numerical": numerical,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    if dist.get_world_size() != 8:
        raise ValueError("this acceptance requires TP8")
    torch.backends.cuda.matmul.allow_tf32 = False
    workspace = SharedRsWorkspace.allocate(dist.group.WORLD, 8192, device)
    adapters = [
        IntegratedFusedRsUpProjectionServing(workspace, 8192, output=None)
        for _ in range(2)
    ]
    torch.manual_seed(5201 + dist.get_rank())
    weights = [
        torch.randn(896, 3584, device=device, dtype=torch.bfloat16) / math.sqrt(3584)
        for _ in range(2)
    ]
    cases = [
        make_case(m, slot, adapters, weights, workspace, device)
        for m in (4096, 8192)
        for slot in (0, 1)
    ]
    checks = []
    for generation in range(128):
        # A/B/A and cross-bucket alternation, with asymmetric rank arrival.
        for case_index in (0, 3, 1, 2, 0):
            case = cases[case_index]
            if dist.get_rank() == generation % 8:
                torch.cuda._sleep(10000)
            case["graph"].replay()
            actual = adapters[-1].output.tensor[: case["m"]]
            checks.append((actual == case["expected"]).all())
    flags = torch.stack(checks)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    torch.cuda.synchronize()
    check_all_ranks(bool(flags.all().item()), "changed-input multi-slot replay failed")
    record = {
        "passed": True,
        "world_size": 8,
        "generations": 128,
        "checks_per_rank": len(checks),
        "all_replay_flags": flags.cpu().tolist(),
        "capture_has_new_live_operand_pointers": True,
        "two_layers_share_raw_workspace": True,
        "per_layer_output_isolation": True,
        "fresh_serving_adapter_test": True,
        "full_model_or_ttft_qualified": False,
        "cases": [
            {"m": row["m"], "slot": row["slot"], "numerical": row["numerical"]}
            for row in cases
        ],
        "resources": {str(m): admit(adapters[0]._plans[m]) for m in (4096, 8192)},
    }
    records = [None] * 8
    dist.all_gather_object(records, record)
    if dist.get_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"ranks": records}, indent=2) + "\n")
        print("TP8_SERVING_ADAPTER_PASS_NOT_FULL_MODEL_ACCEPTANCE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
