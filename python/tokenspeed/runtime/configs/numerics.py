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

"""The ``--numerics`` envelopes and the per-model verification gate.

An envelope is a contract verified end to end, not per switch (see
``docs/design/numerics.md``): a model serves an envelope other than ``auto``
only when its profile declares the model verified under it — the invariance
harness passes for its checkpoint and kernel selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_profile import ModelProfile

# Every envelope name, the default first. A model always serves ``auto``.
NUMERICS_ENVELOPES = ("auto", "rl-bitwise")

# Sampling backends whose greedy rows break exact logit ties toward the lowest
# token id in every batch shape: ``greedy`` is a canonical argmax, the
# FlashInfer backends overlay one on their pool route under rl-bitwise.
RL_BITWISE_SAMPLING_BACKENDS = frozenset({"flashinfer", "flashinfer_full", "greedy"})


def require_verified_numerics(
    numerics: str,
    *,
    model_profile: ModelProfile | None,
    architecture: str,
    quantization: str | None,
) -> None:
    """Refuse a model that the requested envelope has not been verified for.

    Args:
        numerics: The launch's ``--numerics`` envelope.
        model_profile: The model's registered profile, or None for an in-tree
            model (none of which is verified under an envelope beyond auto).
        architecture: The model's architecture name, for the error.
        quantization: The checkpoint's resolved quantization method, or None.

    Raises:
        ValueError: The envelope is not verified for this model, or the
            checkpoint is quantized (no batch-invariant quantized GEMM leaf
            exists, so quantized linears would select shape-dependent ones).
    """
    if numerics == "auto":
        return
    if model_profile is None or numerics not in model_profile.numerics_envelopes:
        raise ValueError(
            f"--numerics {numerics} is a contract verified per model, and "
            f"{architecture} has not been verified under it: its model profile "
            f"must list {numerics!r} in numerics_envelopes, which a model "
            "declares once the bitwise invariance harness passes for it"
        )
    if quantization is not None:
        raise ValueError(
            f"--numerics {numerics} serves unquantized checkpoints only: "
            f"{architecture} is {quantization}-quantized, and no "
            "batch-invariant quantized GEMM leaf exists"
        )


__all__ = [
    "NUMERICS_ENVELOPES",
    "RL_BITWISE_SAMPLING_BACKENDS",
    "require_verified_numerics",
]
