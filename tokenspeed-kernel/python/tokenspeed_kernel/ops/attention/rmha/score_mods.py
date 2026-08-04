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

"""Reusable FA4 (flash_attn.cute) ``score_mod`` factories.

A ``score_mod`` is a ``cute.jit`` closure that FA4 fuses into the attention
kernel to modify pre-softmax logits. FA4 keys kernel compilation on the hash
of the closure object, so factories in this module MUST be memoized
(``functools.cache``): calling a factory twice with the same arguments must
return the same closure object, otherwise every call would trigger a fresh
kernel compilation.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

try:
    import cutlass.cute as cute
    from cutlass.cute import Float32
    from flash_attn.cute.seqlen_info import SeqlenInfoQK
except Exception as _import_error:  # pragma: no cover - environment dependent
    cute = None
    Float32 = None
    SeqlenInfoQK = None
    _cute_import_error = _import_error
else:
    _cute_import_error = None


__all__ = ["get_relative_bias_score_mod"]


@cache
def get_relative_bias_score_mod(rel_extent: int) -> Callable:
    """Build a memoized FA4 score_mod adding a learned relative-position bias.

    The returned closure adds ``rel_logits[global_q_idx, h_idx, q_pos - kv_pos]``
    to the pre-softmax attention logit when ``0 <= q_pos - kv_pos < rel_extent``
    and 0 otherwise, where ``q_pos = q_idx + (seqlen_k - seqlen_q)`` is the
    absolute query position within its sequence (so cached-prefix extend and
    decode line up with prefill).

    Callers must pass ``aux_tensors=[rel_logits]`` alongside the score_mod:

    - ``aux_tensors[0]``: relative bias logits with shape
      ``[total_q, num_q_heads, rel_extent]``. Rows are batch-flattened query
      rows in the same order as ``q``; inside the kernel a row is addressed as
      ``seqlen_info.offset_q + q_idx``, so varlen calls must supply
      ``cu_seqlens_q`` for the offsets to be correct (batch-mode decode would
      see ``offset_q == 0`` for every request and read the wrong rows).

    Args:
        rel_extent: Number of learned relative distances. Distances outside
            ``[0, rel_extent)`` contribute zero bias.

    Returns:
        A ``cute.jit`` closure with the FA4 score_mod signature
        ``(scores, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors)``
        suitable for the ``score_mod=`` argument of FA4 attention entry
        points. The result is cached per ``rel_extent`` because FA4 keys
        kernel compilation on the closure object.

    Raises:
        ImportError: If the FA4 CUTE interface (``flash_attn.cute`` on
            Blackwell) is not available.
    """
    if cute is None or Float32 is None or SeqlenInfoQK is None:
        raise ImportError(
            "get_relative_bias_score_mod requires the FA4 CUTE interface "
            "(cutlass.cute and flash_attn.cute, NVIDIA Blackwell only)."
        ) from _cute_import_error

    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        rel_logits = aux_tensors[0]

        seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + seqlen_local_offset) - kv_idx
        global_q_idx = seqlen_info.offset_q + q_idx

        rel_dist_0 = rel_dist[0]
        rel_idx = rel_dist_0 if rel_dist_0 >= 0 else 0
        rel_idx = rel_idx if rel_idx < rel_extent else (rel_extent - 1)

        rel_bias = rel_logits[global_q_idx[0], h_idx[0], rel_idx]
        rel_bias = Float32(rel_bias) if rel_dist_0 == rel_idx else Float32(0.0)
        return scores + rel_bias

    # Tag as the rel-bias score_mod so fa4 ops can route it to the fused rel_bias kernel path.
    score_mod_rel_bias.rel_extent = rel_extent
    return score_mod_rel_bias
