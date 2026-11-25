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

"""Fused FP8 index scoring and top-k for DSA prefill, vendored from vLLM.

One persistent kernel scores every visible key of a query row and selects its
top-k keys without materializing the ``[rows, keys]`` logits matrix
(``csrc/fused_mqa_topk``, SM100 family). The selection is exact up to ties at
the k-th score.
"""

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

import torch

HEADS = 32
HEAD_DIM = 128
TOPKS = (512, 1024, 2048)
# Candidates store the key index in 20 bits (kMaxKeys in constants.h).
MAX_KEYS = 1 << 20
# Rows per launch: the largest validated end to end.
MAX_ROWS = 16384
SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))

# Mirrors csrc/fused_mqa_topk/constants.h; the kernel checks the shapes.
_CANDIDATES_PER_ROW = 16384
_RETAINED_PER_ROW = 4096
_SLOTS_PER_SM = 2
_BLOCK_ROWS = 8


@functools.cache
def _device_facts(index: int) -> tuple[tuple[int, int], int]:
    properties = torch.cuda.get_device_properties(index)
    return (int(properties.major), int(properties.minor)), int(
        properties.multi_processor_count
    )


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def available(device: torch.device) -> bool:
    """Whether the kernel runs on ``device``: an SM100-family GPU with the
    DeepGEMM headers it is built against installed."""
    if device.type != "cuda" or importlib.util.find_spec("deep_gemm") is None:
        return False
    capability, _ = _device_facts(_device_index(device))
    return capability in SUPPORTED_COMPUTE_CAPABILITIES


def min_keys(topk: int) -> int:
    """Longest-row key count from which the kernel beats separate scoring and
    top-k on SM100 (measured upstream): 32K, or 64K for top-2048."""
    return 1 << 16 if topk > 1024 else 1 << 15


def beats_dense(
    device: torch.device,
    *,
    rows: int,
    dense_rows: int,
    keys: int,
    max_row_keys: int,
    topk: int,
) -> bool:
    """Whether a prefill chunk is faster on this kernel than on separate
    scoring and top-k.

    Each SM scans every key of one 8-row block at a time, so the blocks run in
    waves of one per SM. A last wave on less than half the SMs costs a whole
    wave, which loses unless the separate path needs more than one scoring
    launch anyway.

    Args:
        device: The chunk's CUDA device.
        rows: Query rows of the chunk.
        dense_rows: Rows the separate path scores per launch.
        keys: Gathered keys of the chunk, over all its requests.
        max_row_keys: Most keys one row ranks.
        topk: Keys selected per row.
    """
    _, sms = _device_facts(_device_index(device))
    last_rows = rows - (rows - 1) // MAX_ROWS * MAX_ROWS
    last_wave = (-(-last_rows // _BLOCK_ROWS) - 1) % sms + 1
    return (
        max_row_keys >= min_keys(topk)
        and keys <= MAX_KEYS
        and (rows > dense_rows or 2 * last_wave >= sms)
    )


@functools.cache
def _module():
    """The JIT-built kernel, compiled once per process for SM100a and SM103a."""
    import deep_gemm
    from flashinfer.jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

    csrc = Path(__file__).parent / "csrc"
    source = csrc / "fused_mqa_topk"
    return gen_jit_spec(
        "tokenspeed_fused_mqa_topk",
        [source / "fused_mqa_topk.cu", source / "fused_mqa_topk_binding.cu"],
        extra_cuda_cflags=[
            "-std=c++20",
            "--expt-relaxed-constexpr",
            *sm100a_nvcc_flags,
            *sm103a_nvcc_flags,
        ],
        extra_include_paths=[
            source,
            csrc,
            Path(deep_gemm.__file__).parent / "include",
        ],
        use_fast_math=False,
    ).build_and_load()


def build() -> None:
    """Compile and load the kernel (cached; a no-op once loaded)."""
    _module()


def key_scales(scales: torch.Tensor) -> torch.Tensor:
    """FP32 per-key scales from the index-key cache's scale words.

    Args:
        scales: ``[keys]`` int32, each word four equal UE8M0 bytes (the SM100
            ``fp8_scaled`` key format).

    Returns:
        ``[keys]`` FP32; ``2 ** (e - 127)`` has the bit pattern ``e << 23``.
    """
    return ((scales.view(torch.int32).reshape(-1) & 0xFF) << 23).view(torch.float32)


def _workspaces(device: torch.device, rows: int) -> list[torch.Tensor]:
    """Call-local scratch of one launch over ``rows`` rows, in the kernel's
    argument order: the zeroed block counter, then uninitialized scratch."""
    _, sms = _device_facts(_device_index(device))
    blocks = -(-rows // _BLOCK_ROWS)
    slot_rows = _BLOCK_ROWS * _SLOTS_PER_SM * min(blocks, sms)
    return [
        torch.zeros(2, dtype=torch.int32, device=device),
        torch.empty(
            (slot_rows, _CANDIDATES_PER_ROW), dtype=torch.uint16, device=device
        ),
        torch.empty((slot_rows, _CANDIDATES_PER_ROW), dtype=torch.int32, device=device),
        torch.empty((slot_rows, _RETAINED_PER_ROW), dtype=torch.float32, device=device),
        torch.empty((slot_rows, _RETAINED_PER_ROW), dtype=torch.int32, device=device),
        torch.empty(blocks * _BLOCK_ROWS, dtype=torch.int32, device=device),
        torch.empty(blocks, dtype=torch.int32, device=device),
    ]


def fused_mqa_topk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Write each row's top-k key indices into ``out``.

    Row ``i`` ranks the keys ``row_starts[i]:row_ends[i]``. The indices are
    relative to ``row_starts[i]`` and unordered; a row with at most k keys
    lists them in order and pads with -1.

    Args:
        q: FP8 E4M3 ``[rows, 32, 128]`` index queries.
        k: FP8 E4M3 ``[keys, 128]`` index keys, ``keys <= MAX_KEYS``.
        scales: FP32 ``[keys]`` per-key scales (:func:`key_scales`).
        weights: FP32 ``[rows, 32]`` head weights.
        row_starts: int32 ``[rows]`` first key of each row.
        row_ends: int32 ``[rows]`` end (exclusive) of each row's keys.
        out: int32 ``[rows, k]`` output, k in ``TOPKS``.

    All tensors are contiguous CUDA tensors on one device; ``q``, ``k``,
    ``scales``, ``weights`` and ``out`` are 16-byte aligned.
    """
    run = _module().fused_mqa_topk_prefill
    rows = q.shape[0]
    for start in range(0, rows, MAX_ROWS):
        end = min(start + MAX_ROWS, rows)
        run(
            q[start:end],
            k,
            scales,
            weights[start:end],
            row_starts[start:end],
            row_ends[start:end],
            out[start:end],
            *_workspaces(q.device, end - start),
        )
