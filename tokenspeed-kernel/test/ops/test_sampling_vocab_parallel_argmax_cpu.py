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

"""Exercise real pack/decode kernels with Triton's CPU interpreter."""

import os
import subprocess
import sys
from pathlib import Path


def test_pack_decode_signed_zero_and_special_values_on_cpu():
    # Use a fresh process because other tests may already have imported JIT kernels.
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    kernel_python = str(Path(__file__).resolve().parents[2] / "python")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (kernel_python, env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_cpu_oracle():
    import torch
    from tokenspeed_kernel.platform import ArchVersion, Platform, PlatformInfo

    Platform.override(
        PlatformInfo(
            vendor="cpu",
            arch_version=ArchVersion(0, 0),
            device_name="host",
            device_count=0,
            total_memory=0,
            memory_bandwidth=0,
            sm_count=0,
            max_threads_per_sm=0,
            max_shared_memory_per_sm=0,
        )
    )
    from tokenspeed_kernel.ops.sampling.triton.vocab_parallel_argmax import (
        _pack_argmax_key_kernel,
        argmax_key_index,
        vocab_parallel_argmax_keys,
    )

    rows, width, shards = 9, 17, 8
    full = -torch.arange(1, rows * width * shards + 1).float().view(rows, -1)
    full[0, 5] = full[0, 87] = 3
    full[1, 1] = full[1, 91] = -0.5
    full[2].fill_(-1)
    full[2, 0], full[2, 23] = -0.0, 0.0
    full[3, 17] = full[3, 80] = float("nan")
    full[4].fill_(float("nan"))
    full[5, -1] = 99
    full[6].fill_(float("-inf"))
    full[7, 19] = full[7, 130] = float("inf")
    full[8].fill_(-1)
    full[8, 4], full[8, 98] = 0.0, -0.0
    for dtype in (torch.bfloat16, torch.float32):
        logits = full.to(dtype)
        keys = []
        for rank in range(shards):
            out = torch.empty(rows, dtype=torch.int64)
            vocab_parallel_argmax_keys(
                logits[:, rank * width : (rank + 1) * width],
                vocab_offset=rank * width,
                out=out,
            )
            keys.append(out)
        # Both zero signs must have exactly the same ordered value word.
        for row, left, right in ((2, 0, 1), (8, 0, 5)):
            assert keys[left][row] >> 32 == 0
            assert keys[right][row] >> 32 == 0
        reduced = torch.stack(keys).amax(dim=0)
        decoded = argmax_key_index(reduced, out=torch.empty_like(reduced))
        torch.testing.assert_close(decoded, logits.argmax(dim=-1), rtol=0, atol=0)
        # Preserve the invalid-local-index sentinel used by device argmax leaves.
        invalid = torch.full((rows,), -1, dtype=torch.int32)
        sentinel = torch.empty(rows, dtype=torch.int64)
        _pack_argmax_key_kernel[(rows,)](
            logits, logits.stride(0), invalid, sentinel, width
        )
        assert sentinel.tolist() == [torch.iinfo(torch.int64).min] * rows
        decoded = argmax_key_index(sentinel, out=torch.empty_like(sentinel))
        torch.testing.assert_close(decoded, invalid.long(), rtol=0, atol=0)


if __name__ == "__main__":
    _run_cpu_oracle()
