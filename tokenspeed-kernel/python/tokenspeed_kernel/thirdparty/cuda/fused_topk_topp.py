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

"""Fused TopK + TopP renormalization kernel wrapper.

Matches the result of flashinfer's ``top_k_renorm_prob`` followed by
``top_p_renorm_prob(..., is_deterministic=True)`` in one launch sequence.
"""

import functools
from pathlib import Path

import torch


@functools.cache
def _load_fused_topk_topp_module():
    import tvm_ffi

    objs_dir = Path(__file__).parent / "objs" / "fused_topk_topp"
    so_path = objs_dir / "fused_topk_topp.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel fused_topk_topp library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


# Persistent per-device side streams used to overlap the top-p radix with the
# top-k radix on the main stream. Streams must be pre-created OUTSIDE CUDA
# graph capture (cudaStreamCreate is illegal inside capture), so callers that
# replay the kernel from a captured graph have to register them up front via
# ``prepare_for_device`` — typically from the sampling backend's ``__init__``.
_side_streams: dict[torch.device, torch.cuda.Stream] = {}


def prepare_for_device(device: torch.device | str | int) -> None:
    """Pre-create the side stream used by ``fused_topk_topp_renorm`` for the
    given device. Idempotent. Must be called before the first invocation that
    happens inside a CUDA graph capture region, so the stream handle is stable
    across the captured replays.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return
    if device not in _side_streams:
        _side_streams[device] = torch.cuda.Stream(device=device)


def _get_side_stream_handle(device: torch.device) -> int:
    """Return the raw ``cudaStream_t`` (as int64) for the device's side stream,
    creating it if we're not currently inside a capture region. Returns 0 to
    signal "no side stream" if we'd have to create one inside capture."""
    stream = _side_streams.get(device)
    if stream is not None:
        return int(stream.cuda_stream)
    if torch.cuda.is_current_stream_capturing():
        return 0
    prepare_for_device(device)
    return int(_side_streams[device].cuda_stream)


def fused_topk_topp_workspace_size(batch_size: int, vocab_size: int) -> int:
    """Workspace size (in bytes) required by ``fused_topk_topp_renorm``."""
    return int(
        _load_fused_topk_topp_module().fused_topk_topp_workspace_size(
            int(batch_size), int(vocab_size)
        )
    )


def fused_topk_topp_renorm(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    workspace: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused TopK + TopP renormalization.

    Args:
        probs:  ``[bs, V]`` float32, softmax'd probabilities.
        top_ks: ``[bs]`` int32. K in ``[1, V)`` plus the sentinel ``K = 1 << 30``
                which routes the row through the radix top-p path.
        top_ps: ``[bs]`` float32. P in ``(0, 1]``.
        workspace: optional pre-allocated uint8 scratch buffer; if omitted, a
                   fresh one is allocated via the CUDA caching allocator.
        out: optional pre-allocated ``[bs, V]`` float32 output; if omitted, a
             fresh one is allocated.

    Returns:
        ``[bs, V]`` float32. Non-kept positions are 0; kept positions are
        renormalized so each row sums to 1.
    """
    if out is None:
        out = torch.empty_like(probs)
    if workspace is None:
        ws_bytes = fused_topk_topp_workspace_size(probs.size(0), probs.size(1))
        workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=probs.device)
    side_handle = _get_side_stream_handle(probs.device)
    _load_fused_topk_topp_module().fused_topk_topp_renorm(
        probs, top_ks, top_ps, out, workspace, side_handle
    )
    return out
