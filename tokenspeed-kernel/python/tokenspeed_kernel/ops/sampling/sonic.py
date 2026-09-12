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

"""sonic-sampler entry points for the runtime's ``sonic`` sampling backend.

sonic-sampler (optional dependency, v1.0.0) supplies everything;
``tokenspeed_kernel.thirdparty.sonic`` re-jits its fused kernels and wrappers
from their own source with two asserted patches (batch size as a runtime int,
in-kernel Philox noise) at import time. sonic binds ``triton`` at import time,
so it is imported under the ``tokenspeed_triton`` redirect: every kernel in
the chain compiles with the one vendored Triton.

``available`` is False when sonic-sampler is not installed; every other name
is then None. An installed sonic-sampler whose source no longer matches the
patch set fails this import loudly (``ImportError`` from the patch set).
"""

from __future__ import annotations

import importlib.util

from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton
from torch import Tensor

available = importlib.util.find_spec("sonic_sampler") is not None

if available:
    # sonic binds ``triton`` at import: route the whole chain to the vendored Triton.
    with redirect_triton_to_tokenspeed_triton():
        from sonic_sampler.base.sampler import Selection, Verification
        from sonic_sampler.core.buffer import SamplingBuffers
        from sonic_sampler.core.flags import ScopedIndicators
        from sonic_sampler.interface.base import TopKStrategy, TwoStageTiling
        from sonic_sampler.interface.dispatch import (
            BatchBucket,
            DispatchSummary,
            PriorityBucket,
            RuntimeConfig,
            Strategy,
            ThreeStageWarpConfig,
            TwoStageWarpConfig,
            VocabBucket,
        )
        from sonic_sampler.ops.base import MAX_K
        from tokenspeed_kernel.thirdparty.sonic import fused_multistep, fused_singular
else:
    SamplingBuffers = ScopedIndicators = None
    TopKStrategy = TwoStageTiling = None
    BatchBucket = DispatchSummary = PriorityBucket = RuntimeConfig = VocabBucket = None
    Strategy = None
    ThreeStageWarpConfig = TwoStageWarpConfig = None
    Selection = Verification = None
    fused_multistep = fused_singular = None
    MAX_K = None

# Measured (arch, packaged vocab bucket) -> batch buckets in sonic's TOML schema.
# An entry replaces the packaged bucket for every part of that arch, every
# vocabulary in the bucket and (one batch bucket) every batch size.
_MEASURED_DISPATCH: dict[tuple[int, int], list[tuple[int, int, Strategy, int, int]]] = {
    # sm100 131k-262k vocab: 2 second-stage warps, 82us -> 24us per row (GB200).
    (100, 262144): [(1 << 16, 4096, "bitonic", 8, 2)],
}


def _fallback_block_n(vocab_size: int) -> int:
    """Vocab tile off the packaged table: two or three blocks for sub-4096
    (test) vocabularies, sonic's reduction needing more than one."""
    if vocab_size > 4096:
        return 4096
    return 1 << (vocab_size.bit_length() - 2)


def make_tiling(
    arch: int,
    vocab_size: int,
    batch_size: int,
    lookahead: int,
    unpacked_buffers: bool,
) -> tuple[TwoStageTiling, Tensor | None, Tensor | None]:
    """A ``TwoStageTiling`` for this arch and vocabulary, with the measured
    dispatch swapped in where one exists.

    Args:
        arch: compute capability as ``major * 10 + minor``.
        vocab_size: the logits width (at least 512, sonic's reduction minimum).
        batch_size: the largest batch the scratchpad must hold.
        lookahead: draft tokens per request (0 without speculative decoding).
        unpacked_buffers: allocate the verify unpack buffers.

    Returns:
        ``(tiling, values, indices)`` as ``TwoStageTiling.factory`` returns them;
        off the packaged table (unknown arch, small vocabulary) sonic tiles by an
        explicit block size and ``tiling.tuning`` returns its defaults.
    """
    tuned = tuned_bucket(arch, vocab_size) is not None
    tiling, values, indices = TwoStageTiling.factory(
        vocab_size=vocab_size,
        batch_size=batch_size,
        lookahead=lookahead,
        block_size=None if tuned else _fallback_block_n(vocab_size),
        arch=arch,
        unpacked_buffers=unpacked_buffers,
    )
    if tuned:
        measured = measured_dispatch(arch, tiling.dispatch)
        if measured is not None:
            tiling.dispatch = measured
    return tiling, values, indices


def tuned_bucket(arch: int, vocab_size: int) -> VocabBucket | None:
    """The packaged vocab bucket ``TwoStageTiling.factory`` would resolve, or
    None when the factory would refuse: no packaged table for ``arch``, no
    bucket for ``vocab_size``, or a vocabulary the bucket's block tiles in a
    single block (sonic's reduction needs more than one).

    Args:
        arch: compute capability as ``major * 10 + minor``.
        vocab_size: the logits width.

    Returns:
        The matching ``VocabBucket``, or None (use the pre-tuning fallback).
    """
    summary = DispatchSummary.load(k=MAX_K, arch=arch)
    if summary is None:
        return None
    bucket = summary.vocab.get(size=vocab_size)
    if bucket is None or vocab_size <= bucket.block_n:
        return None
    return bucket


def measured_dispatch(arch: int, bucket: VocabBucket) -> VocabBucket | None:
    """The measured replacement for a packaged vocab bucket on this arch, if any.

    Args:
        arch: compute capability as ``major * 10 + minor``.
        bucket: the packaged ``VocabBucket`` a ``TwoStageTiling`` resolved.

    Returns:
        A ``VocabBucket`` of the same size whose batch buckets carry the measured
        configs, or None when nothing was measured for ``(arch, bucket.size)``.

    Raises:
        ValueError: a measured ``block_n`` is below the packaged bucket's
            smallest, which sized the tiling's scratchpad.
    """
    entries = _MEASURED_DISPATCH.get((arch, bucket.size))
    if entries is None:
        return None
    for _, block_n, _, _, _ in entries:
        if block_n < bucket.block_n:
            raise ValueError(
                f"measured block_n {block_n} < packaged minimum {bucket.block_n} "
                f"for vocab bucket {bucket.size}: scratchpad would be undersized"
            )
    batch = [
        BatchBucket(
            size=size,
            config=[
                RuntimeConfig(
                    priority=0,
                    block_n=block_n,
                    strategy=strategy,
                    first_warps=first,
                    second_warps=second,
                )
            ],
        )
        for size, block_n, strategy, first, second in entries
    ]
    return VocabBucket(size=bucket.size, batch=PriorityBucket(batch))


__all__ = [
    "MAX_K",
    "SamplingBuffers",
    "ScopedIndicators",
    "Selection",
    "ThreeStageWarpConfig",
    "TopKStrategy",
    "TwoStageTiling",
    "TwoStageWarpConfig",
    "Verification",
    "available",
    "fused_multistep",
    "fused_singular",
    "make_tiling",
    "measured_dispatch",
    "tuned_bucket",
]
