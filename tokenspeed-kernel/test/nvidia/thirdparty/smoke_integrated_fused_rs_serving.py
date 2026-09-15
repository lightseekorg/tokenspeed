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


"""TP8 integrated back-half smoke, without non-tile-multiple stress cases."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from smoke_medium_fused_rs_serving import run
from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_serving import (
    IntegratedFusedRsUpProjectionServing,
)
from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_serving_config import (
    integrated_fused_rs_serving_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.tokens = [128, 256, 512, 1024, 2048, 4096, 8192]
    args.generations = 3
    args.rank_skew_cycles = 0
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    if dist.get_world_size() != 8:
        raise ValueError("integrated smoke requires TP8")
    torch.backends.cuda.matmul.allow_tf32 = False
    record = {
        "schema": "integrated-fused-serving-smoke-v1",
        "passed": False,
        "full_model_or_ttft_qualified": False,
        "non_tile_multiple_gpu_validation": False,
        "performance_measured": False,
    }
    try:
        local = run(
            args,
            torch.device("cuda", torch.cuda.current_device()),
            IntegratedFusedRsUpProjectionServing,
            integrated_fused_rs_serving_config,
        )
        ranks = [None] * 8
        dist.all_gather_object(ranks, local)
        record.update(passed=True, ranks=ranks)
    except Exception as exc:
        record["failure_type"] = type(exc).__name__
        raise
    finally:
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
