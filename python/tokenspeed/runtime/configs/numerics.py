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

from collections.abc import Callable
from typing import TYPE_CHECKING

from tokenspeed_cli.server_args import (
    BLOCKWISE_LOGPROB_ORDERS as BLOCKWISE_LOGPROB_ORDERS,
)
from tokenspeed_cli.server_args import BLOCKWISE_VOCAB_BLOCK as BLOCKWISE_VOCAB_BLOCK
from tokenspeed_cli.server_args import DSA_SLOT_ORDERS as DSA_SLOT_ORDERS
from tokenspeed_cli.server_args import LAYER_BOUNDARY_NORMS as LAYER_BOUNDARY_NORMS
from tokenspeed_cli.server_args import LOGPROB_ORDERS as LOGPROB_ORDERS
from tokenspeed_cli.server_args import MLA_LORA_SCALES as MLA_LORA_SCALES
from tokenspeed_cli.server_args import MOE_COMBINE_ORDERS as MOE_COMBINE_ORDERS
from tokenspeed_cli.server_args import ROUTER_TOPKS as ROUTER_TOPKS
from tokenspeed_cli.server_args import SAMPLING_STREAMS as SAMPLING_STREAMS
from tokenspeed_cli.server_args import YARN_FREQUENCY_DEVICES as YARN_FREQUENCY_DEVICES
from tokenspeed_cli.server_args import YARN_RAMP_MASK_DEVICES as YARN_RAMP_MASK_DEVICES

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_profile import ModelProfile
    from tokenspeed.runtime.utils.server_args import ServerArgs

# Every envelope name, the default first. A model always serves ``auto``.
# ``rl-bitwise`` is the built-in bitwise envelope: run and batch invariance plus
# the trainer's operation order (``docs/design/numerics.md``).
NUMERICS_ENVELOPES = ["auto", "rl-bitwise"]

# Envelopes that promise the bitwise contract. Selection points that pin
# batch-invariant leaves test membership here rather than the one name, so an
# envelope added above it inherits every pin.
# Mutate these shared containers at registration: modules importing them must
# see plugin additions even when they imported before plugin discovery.
BITWISE_ENVELOPES = {"rl-bitwise"}
_PLUGIN_ENVELOPES: dict[
    str, tuple[Callable[[ServerArgs], None], Callable[..., None]]
] = {}


def register_numerics_envelope(
    name: str,
    *,
    bitwise: bool,
    resolve: Callable[[ServerArgs], None],
    validate: Callable[..., None],
) -> None:
    """Register a plugin's numerical policy during its ``register()`` call.

    ``resolve(server_args)`` resolves launch settings. After the common model
    profile gate, ``validate(quantization=..., vocab_size=...)`` checks storage
    and vocabulary requirements. A bitwise policy opts into the existing
    batch-invariant selection points; it does not inherit RL operation order.
    Names cannot replace an existing policy. Failed plugin loading rolls back
    its resolver, CLI choice and bitwise membership together.
    """
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError("A numerics envelope requires a nonempty name")
    if name in NUMERICS_ENVELOPES:
        raise ValueError(f"Numerics envelope {name!r} is already registered")
    if type(bitwise) is not bool or not callable(resolve) or not callable(validate):
        raise TypeError("A numerics envelope requires bitwise, resolve and validate")
    _PLUGIN_ENVELOPES[name] = (resolve, validate)
    NUMERICS_ENVELOPES.append(name)
    if bitwise:
        BITWISE_ENVELOPES.add(name)

    def undo() -> None:
        del _PLUGIN_ENVELOPES[name]
        NUMERICS_ENVELOPES.remove(name)
        BITWISE_ENVELOPES.discard(name)

    from tokenspeed.runtime.plugins.registry import record_external

    record_external("numerics envelope", name, undo)


def resolve_numerics_envelope(server_args: ServerArgs) -> None:
    """Resolve the built-in or registered policy after plugin discovery."""
    if server_args.numerics == "rl-bitwise":
        server_args._resolve_rl_bitwise()
    elif server_args.numerics != "auto":
        _PLUGIN_ENVELOPES[server_args.numerics][0](server_args)


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
    vocab_size: int,
) -> None:
    """Refuse a model that the requested envelope has not been verified for.

    Args:
        numerics: The launch's ``--numerics`` envelope.
        model_profile: The model's registered profile, or None for an in-tree
            model (none of which is verified under an envelope beyond auto).
        architecture: The model's architecture name, for the error.
        quantization: The checkpoint's resolved quantization method, or None.
        vocab_size: The model's vocabulary size; the envelope's blockwise
            logprobs fold fixed ``BLOCKWISE_VOCAB_BLOCK``-wide blocks of it.

    Raises:
        ValueError: The envelope is not verified for this model, or its
            checkpoint validator rejects the storage or vocabulary. The RL
            policy requires unquantized weights and whole blockwise vocab blocks.
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
    if numerics != "rl-bitwise":
        _PLUGIN_ENVELOPES[numerics][1](quantization=quantization, vocab_size=vocab_size)
        return
    if quantization is not None:
        raise ValueError(
            f"--numerics {numerics} serves unquantized checkpoints only: "
            f"{architecture} is {quantization}-quantized, and no "
            "batch-invariant quantized GEMM leaf exists"
        )
    if vocab_size % BLOCKWISE_VOCAB_BLOCK != 0:
        raise ValueError(
            f"--numerics {numerics} folds --logprob-order blockwise, whose "
            f"sum(exp) runs over fixed {BLOCKWISE_VOCAB_BLOCK}-wide vocabulary "
            f"blocks; {architecture} has vocab_size {vocab_size}, not a "
            "multiple of it"
        )


__all__ = [
    "BITWISE_ENVELOPES",
    "DSA_SLOT_ORDERS",
    "LAYER_BOUNDARY_NORMS",
    "LOGPROB_ORDERS",
    "BLOCKWISE_VOCAB_BLOCK",
    "BLOCKWISE_LOGPROB_ORDERS",
    "MLA_LORA_SCALES",
    "MOE_COMBINE_ORDERS",
    "NUMERICS_ENVELOPES",
    "RL_BITWISE_SAMPLING_BACKENDS",
    "ROUTER_TOPKS",
    "SAMPLING_STREAMS",
    "YARN_RAMP_MASK_DEVICES",
    "YARN_FREQUENCY_DEVICES",
    "require_verified_numerics",
    "register_numerics_envelope",
    "resolve_numerics_envelope",
]
