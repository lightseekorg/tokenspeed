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
# ``rl-bitwise`` is the one bitwise envelope: run and batch invariance plus
# the trainer's operation order (``docs/design/numerics.md``).
NUMERICS_ENVELOPES = ("auto", "rl-bitwise")

# Envelopes that promise the bitwise contract. Selection points that pin
# batch-invariant leaves test membership here rather than the one name, so an
# envelope added above it inherits every pin.
BITWISE_ENVELOPES = frozenset({"rl-bitwise"})

# Sampling backends whose greedy rows break exact logit ties toward the lowest
# token id in every batch shape: ``greedy`` is a canonical argmax, the
# FlashInfer backends overlay one on their pool route under rl-bitwise.
RL_BITWISE_SAMPLING_BACKENDS = frozenset({"flashinfer", "flashinfer_full", "greedy"})

# ``--sampling-stream``: which random stream the FlashInfer backends' sampled
# (non-greedy) rows draw from. ``batch`` is flashinfer's Philox stream keyed by
# the batch row; ``per-request`` is the Gumbel-max pool route keyed by
# (request seed, position), which the bitwise envelope requires.
SAMPLING_STREAMS = ("batch", "per-request")

# ``--yarn-ramp-mask-device``: where the deepseek_yarn RoPE inverse frequencies
# (position frequencies, both divisions and the YaRN linear ramp mask) are
# computed before moving to the model device. The trainer builds them on the
# host.
YARN_RAMP_MASK_DEVICES = ("cuda", "cpu")

# ``--mla-lora-scale``: where LongCat-style MLA applies its sqrt(hidden /
# lora_rank) norm scales — folded into the q_a/kv_a layernorm weights at load,
# or multiplied at runtime after q_b_proj / kv_a_layernorm as the trainer does.
MLA_LORA_SCALES = ("folded", "runtime")

# ``--layer-boundary-norm``: the norm at each physical layer boundary — the
# fused add+norm kernel, or a bf16 ``hidden + residual`` materialized first and
# a standalone RMSNorm, as the trainer does.
LAYER_BOUNDARY_NORMS = ("fused", "unfused")

# ``--router-topk``: correction-bias MoE routing — the fused CUDA kernel, or
# fp32 ``torch.softmax`` + ``torch.topk(probs + bias)`` in PyTorch tie order
# with ``-1`` zero-expert ids, as the trainer does.
ROUTER_TOPKS = ("fused", "torch")

# ``--logprob-order``: the selected-token log-softmax behind every returned
# logprob — ``torch.log_softmax``, or Megatron's vocab-parallel cross-entropy
# order over fixed ``MEGATRON_VOCAB_BLOCK``-wide vocab blocks. Logprobs only.
LOGPROB_ORDERS = ("torch", "megatron")
MEGATRON_VOCAB_BLOCK = 32768

# ``--moe-combine-order``: how a token's routed-expert contributions meet
# across the MoE TP-EP group — ``rank``: per-rank partials summed by the
# host's all-reduce / reduce-scatter, the identity zero-expert residual added
# around it; ``slot``: the MoE leaf folds the top-k slots in fp32 slot order
# across the EP group itself, residual included, as the trainer's grouped MLP
# does, and the host reduces nothing. Mirrors ``tokenspeed_kernel``'s
# ``moe.COMBINE_ORDERS``.
MOE_COMBINE_ORDERS = ("rank", "slot")

# ``--dsa-slot-order``: the order the sparse attention cores reduce a token's
# selected KV slots in — ``selection``: as the top-k leaf emitted them (what
# every core does); ``sorted``: ascending slot order, so the reduction is
# batch-invariant whenever the selected set is; served only by cores
# declaring the ``slot_order`` trait (the ``aok`` leaves). Mirrors
# ``tokenspeed_kernel``'s ``attention.dsa.SLOT_ORDERS``.
DSA_SLOT_ORDERS = ("selection", "sorted")


def require_verified_numerics(
    numerics: str,
    *,
    model_profile: ModelProfile | None,
    architecture: str,
    quantization: str | None,
    vocab_size: int,
) -> None:
    """Refuse a model that the requested envelope has not been verified for.

    Args:
        numerics: The launch's ``--numerics`` envelope.
        model_profile: The model's registered profile, or None for an in-tree
            model (none of which is verified under an envelope beyond auto).
        architecture: The model's architecture name, for the error.
        quantization: The checkpoint's resolved quantization method, or None.
        vocab_size: The model's vocabulary size; the envelope's Megatron-order
            logprobs fold fixed ``MEGATRON_VOCAB_BLOCK``-wide blocks of it.

    Raises:
        ValueError: The envelope is not verified for this model, the
            checkpoint is quantized (no batch-invariant quantized GEMM leaf
            exists, so quantized linears would select shape-dependent ones),
            or the vocabulary is not a whole number of Megatron blocks.
    """
    if numerics == "auto":
        return
    if model_profile is None or numerics not in model_profile.numerics_envelopes:
        raise ValueError(
            f"--numerics {numerics} is a contract verified per model, and "
            f"{architecture} has not been verified under it: its model "
            f"profile must list {numerics!r} in numerics_envelopes, which a "
            "model declares once the invariance harness and the teacher-forced "
            "logprob comparison against the trainer pass for it "
            "(docs/design/numerics.md, Acceptance)"
        )
    if quantization is not None:
        raise ValueError(
            f"--numerics {numerics} serves unquantized checkpoints only: "
            f"{architecture} is {quantization}-quantized, and no "
            "batch-invariant quantized GEMM leaf exists"
        )
    if vocab_size % MEGATRON_VOCAB_BLOCK != 0:
        raise ValueError(
            f"--numerics {numerics} folds --logprob-order megatron, whose "
            f"sum(exp) runs over fixed {MEGATRON_VOCAB_BLOCK}-wide vocabulary "
            f"blocks; {architecture} has vocab_size {vocab_size}, not a "
            "multiple of it"
        )


__all__ = [
    "BITWISE_ENVELOPES",
    "DSA_SLOT_ORDERS",
    "LAYER_BOUNDARY_NORMS",
    "LOGPROB_ORDERS",
    "MEGATRON_VOCAB_BLOCK",
    "MLA_LORA_SCALES",
    "MOE_COMBINE_ORDERS",
    "NUMERICS_ENVELOPES",
    "RL_BITWISE_SAMPLING_BACKENDS",
    "ROUTER_TOPKS",
    "SAMPLING_STREAMS",
    "YARN_RAMP_MASK_DEVICES",
    "require_verified_numerics",
]
