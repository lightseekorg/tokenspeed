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

"""CuTe DSL single-pass multi-CTA (cluster) top-k for the DSA decode indexer.

Wraps the vendored TensorRT-LLM CuTe DSL radix runners as a length-aware per-row
top-k over a ``[num_rows, num_cols]`` indexer-logits matrix, matching the
``deterministic_decode_topk(..., lengths=seq_lens, q_len_per_req=next_n)`` contract
it replaces (returned int32 column indices are the "local offsets" that
``local_topk_to_global_slots`` maps to KV slots). Cluster-first: falls back to the
non-cluster runner when the problem exceeds cluster capacity. NVIDIA Blackwell
(sm_100+) only; gate on :func:`has_cute_dsl_decode_topk`.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import current_platform

__all__ = [
    "cute_dsl_decode_topk",
    "has_cute_dsl_decode_topk",
]


def _ts_supported_arch() -> bool:
    """Gate: the CuTe DSL multi-CTA / cluster radix top-k needs NVIDIA sm_100+.

    Returns False if platform detection raises (e.g. CPU-only host) so callers
    fall back transparently.
    """
    try:
        p = current_platform()
    except Exception:
        return False
    if not p.is_nvidia:
        return False
    sm = p.arch_version.major * 10 + p.arch_version.minor
    return sm >= 100


_CUTE_DSL_TOPK_AVAILABLE = False
_ClusterRunner = None
_BaseRunner = None

if _ts_supported_arch():
    try:
        import cutlass  # noqa: F401  (import probe: kernels JIT via cutlass-dsl)
        import cutlass.cute  # noqa: F401
        from tokenspeed_kernel.thirdparty.cute_dsl.topk import (
            CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner as _ClusterRunner,
        )
        from tokenspeed_kernel.thirdparty.cute_dsl.topk import (
            CuteDSLTopKDecodeSinglePassMultiCTARunner as _BaseRunner,
        )

        _CUTE_DSL_TOPK_AVAILABLE = True
    except ImportError:
        _CUTE_DSL_TOPK_AVAILABLE = False


def has_cute_dsl_decode_topk() -> bool:
    """Whether the CuTe DSL decode top-k kernel can run on this platform."""
    return _CUTE_DSL_TOPK_AVAILABLE


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def cute_dsl_decode_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    *,
    next_n: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Length-aware per-row top-k over a dense indexer-logits matrix.

    Args:
        logits: 2-D row-major contiguous CUDA tensor ``[num_rows, num_cols]``
            (``num_rows = num_reqs * next_n``), fp16/bf16/fp32. No pre-masking
            needed; the causal window is derived in-kernel from seq_lens/next_n.
        seq_lens: Per-request candidate length, int32 ``[num_reqs]``. Row ``r``'s
            window is ``seq_lens[r // next_n] - next_n + (r % next_n) + 1`` cols.
        topk: Candidates to select per row (1..2048).
        next_n: Query rows per request (speculative ``q_len_per_req``); 1 for decode.
        out: Optional int32 ``[num_rows, topk]`` buffer, written in place and returned.

    Returns:
        int32 ``[num_rows, topk]`` of per-row selected column indices (unsorted).
        Rows with window < topk have surplus slots filled by the radix collection;
        track valid count separately (``min(window, topk)``), as with
        ``deterministic_decode_topk``.

    Raises:
        RuntimeError: if the kernel is unavailable on this platform.
    """
    if not _CUTE_DSL_TOPK_AVAILABLE:
        raise RuntimeError(
            "cute_dsl_decode_topk is unavailable on this platform "
            "(requires NVIDIA Blackwell sm_100+ with nvidia-cutlass-dsl)."
        )
    if logits.dim() != 2:
        raise ValueError(
            f"logits must be 2-D [num_rows, num_cols], got {tuple(logits.shape)}"
        )
    if logits.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"logits dtype must be one of {_SUPPORTED_DTYPES}, got {logits.dtype}"
        )
    num_rows = logits.shape[0]
    topk = int(topk)
    next_n = int(next_n)

    logits = logits.contiguous()
    seq_lens = seq_lens.to(device=logits.device, dtype=torch.int32).contiguous()

    if out is not None:
        if out.shape != (num_rows, topk):
            raise ValueError(
                f"out must have shape {(num_rows, topk)}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.int32:
            raise ValueError(f"out dtype must be int32, got {out.dtype}")
        out = out.contiguous()

    # Cluster-first. Re-zero the shared ``row_states`` scratch every call: the
    # kernels self-clean it, but that reuse races under tight CUDA-graph scheduling.
    _ClusterRunner._row_states_initialized = False
    indices, _ = _ClusterRunner.forward(
        logits,
        seq_lens,
        topk,
        next_n,
        return_val=False,
        output_indices=out,
    )
    if indices is None:
        # Exceeds cluster capacity: fall back to the non-cluster runner.
        _BaseRunner._row_states_initialized = False
        indices, _ = _BaseRunner.forward(
            logits,
            seq_lens,
            topk,
            next_n,
            return_val=False,
            output_indices=out,
        )
    return indices
