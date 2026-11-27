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

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from transformers import PretrainedConfig

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


class RLCheckpointValidator(Protocol):
    """Check checkpoint storage; returning accepts it and raising rejects it."""

    def __call__(
        self, *, quantization: str | None, vocab_size: int, architecture: str
    ) -> None:
        """Validate checkpoint storage; return None to accept or raise to reject."""
        ...


@dataclass(frozen=True, kw_only=True)
class RLNumericsPolicy:
    """A model's implementation of the common ``rl-bitwise`` contract.

    Attributes:
        resolve: Called for the target model before workers and process groups
            start, with the shared ``ServerArgs``. A draft does not run its
            own resolver against that shared launch.
        validate: Called after profile admission for target and draft alike,
            with keyword arguments ``quantization`` (resolved storage format,
            or None), ``vocab_size`` (actual vocabulary size), and
            ``architecture`` (the loader's selected model name). Returning
            accepts storage; raising rejects it. Validation must stand alone:
            it cannot assume its paired resolver ran for a draft. A model's
            ``configure_attention`` hook owns launch/role-dependent limits.

    Registration is not evidence of numerical correctness or alignment;
    qualification covers each supported target/draft deployment together.
    """

    resolve: Callable[[ServerArgs], None]
    validate: RLCheckpointValidator

    def __post_init__(self) -> None:
        if not callable(self.resolve) or not callable(self.validate):
            raise TypeError("RLNumericsPolicy callbacks must be callable")


_MODEL_RL_POLICIES: dict[str, Callable[[PretrainedConfig], RLNumericsPolicy]] = {}


def register_model_rl_numerics(
    architecture: str,
    factory: Callable[[PretrainedConfig], RLNumericsPolicy],
) -> None:
    """Register checkpoint-aware arithmetic for the shared RL contract.

    Args:
        architecture: The nonempty, unpadded model registration name used by
            the loader.
        factory: A callable receiving that model's checkpoint configuration
            and returning an RLNumericsPolicy. It can return
            STANDARD_RL_NUMERICS for variants using the standard arithmetic.

    The model must still declare ``rl-bitwise`` in its verification profile.
    A failed plugin load rolls this registration back too.

    Raises:
        TypeError: The architecture is not a string or the factory is not callable.
        ValueError: The name is empty, padded, or already registered.
    """
    if not isinstance(architecture, str):
        raise TypeError("A numerical policy architecture must be a string")
    if not architecture or architecture != architecture.strip():
        raise ValueError("A numerical policy needs a nonempty, unpadded architecture")
    if not callable(factory):
        raise TypeError("A numerical policy factory must be callable")
    if architecture in _MODEL_RL_POLICIES:
        raise ValueError(f"Numerical policy for {architecture!r} is already registered")
    _MODEL_RL_POLICIES[architecture] = factory

    def undo() -> None:
        del _MODEL_RL_POLICIES[architecture]

    from tokenspeed.runtime.plugins.registry import record_external

    record_external("model numerical policy", architecture, undo)


def model_rl_numerics(
    architecture: str, hf_config: PretrainedConfig | None
) -> RLNumericsPolicy:
    """Select arithmetic for the architecture actually chosen by the loader.

    Args:
        architecture: The registered architecture selected for this model.
        hf_config: Its checkpoint configuration. None is accepted only when
            the architecture has no registered policy factory.

    Returns:
        STANDARD_RL_NUMERICS for an unregistered architecture, otherwise the
        RLNumericsPolicy returned by that architecture's factory.

    Raises:
        ValueError: A registered factory needs checkpoint context but none was given.
        TypeError: A factory returned something other than RLNumericsPolicy.
    """
    factory = _MODEL_RL_POLICIES.get(architecture)
    if factory is None:
        return STANDARD_RL_NUMERICS
    if hf_config is None:
        raise ValueError(
            f"{architecture}'s numerical policy needs its checkpoint config"
        )
    policy = factory(hf_config)
    if not isinstance(policy, RLNumericsPolicy):
        raise TypeError(f"{architecture}'s policy factory must return RLNumericsPolicy")
    return policy


def launch_rl_numerics(server_args: ServerArgs) -> RLNumericsPolicy:
    """Select the target policy without applying its resolver.

    Args:
        server_args: The launch arguments after plugin discovery. Model,
            revision, download directory, and explicit HF overrides identify
            the checkpoint metadata used to select its registered profile.

    Returns:
        The selected model policy. With no registered policy factories this
        returns STANDARD_RL_NUMERICS without checkpoint IO. Otherwise it loads
        checkpoint configuration and follows model-loader architecture order;
        a checkpoint with no registered profile also selects the standard policy.
    """
    if not _MODEL_RL_POLICIES:
        return STANDARD_RL_NUMERICS

    from tokenspeed.runtime.plugins import finish_headless_plugins
    from tokenspeed.runtime.plugins.registry import resolve_model_profile
    from tokenspeed.runtime.utils.hf_transformers_utils import (
        get_config,
        model_loader_architectures,
    )

    # Model policies need the real config and loader profile before launch
    # choices become fixed. CLI-only registration may have deferred both.
    finish_headless_plugins()
    hf_config = get_config(
        server_args.model,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        model_override_args=json.loads(server_args.hf_overrides),
        download_dir=server_args.download_dir,
    )
    resolved = resolve_model_profile(model_loader_architectures(hf_config), hf_config)
    if resolved is None:
        return STANDARD_RL_NUMERICS
    return model_rl_numerics(resolved[0], hf_config)


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
    """Resolve model-owned RL arithmetic or a registered legacy envelope."""
    if server_args.numerics == "rl-bitwise":
        launch_rl_numerics(server_args).resolve(server_args)
    elif server_args.numerics != "auto":
        _PLUGIN_ENVELOPES[server_args.numerics][0](server_args)


# Sampling backends whose greedy rows break exact logit ties toward the lowest
# token id in every batch shape: ``greedy`` is a canonical argmax, the
# FlashInfer backends overlay one on their pool route under rl-bitwise.
RL_BITWISE_SAMPLING_BACKENDS = frozenset({"flashinfer", "flashinfer_full", "greedy"})

# ``--sampling-stream``: which random stream the FlashInfer backends' sampled
# (non-greedy) rows draw from. ``batch`` is flashinfer's Philox stream keyed by
# the batch row; ``per-request`` is the Gumbel-max pool route keyed by
# (request seed, position) and position-keyed chain verify coins, which the
# bitwise envelope requires.
SAMPLING_STREAMS = ("batch", "per-request")

# ``--yarn-ramp-mask-device`` selects the deepseek_yarn linear ramp device;
# ``--yarn-frequency-device`` selects powers, divisions and interpolation.
# Under auto, both devices default independently to CUDA.
# rl-bitwise pins a CPU ramp and CUDA frequency construction.
YARN_RAMP_MASK_DEVICES = ("cuda", "cpu")
YARN_FREQUENCY_DEVICES = ("cuda", "cpu")

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
# logprob — ``torch.log_softmax``, or fixed-width blockwise reduction
# over ``BLOCKWISE_VOCAB_BLOCK``-wide vocab blocks. Logprobs only.
BLOCKWISE_VOCAB_BLOCK = 32768
# Block width describes the trainer's reduction, independently of serving TP.
BLOCKWISE_LOGPROB_ORDERS = {
    "blockwise": (BLOCKWISE_VOCAB_BLOCK, "negative-cross-entropy"),
    "blockwise-direct": (BLOCKWISE_VOCAB_BLOCK, "direct"),
    "blockwise-16384": (16384, "negative-cross-entropy"),
    "blockwise-direct-16384": (16384, "direct"),
}
LOGPROB_ORDERS = ("torch", *BLOCKWISE_LOGPROB_ORDERS)

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
# every core does); ``sorted``: ascending token position, so the reduction is
# batch-invariant whenever the selected set is; served only by cores
# declaring the ``slot_order`` trait (registered batch-invariant leaves). Mirrors
# ``tokenspeed_kernel``'s ``attention.dsa.SLOT_ORDERS``.
DSA_SLOT_ORDERS = ("selection", "sorted")


def require_verified_numerics(
    numerics: str,
    *,
    model_profile: ModelProfile | None,
    architecture: str,
    quantization: str | None,
    vocab_size: int,
    hf_config: PretrainedConfig | None = None,
) -> None:
    """Refuse a model that the requested envelope has not been verified for.

    Args:
        numerics: The launch's ``--numerics`` envelope.
        model_profile: The model's registered profile, or None for an in-tree
            model (none of which is verified under an envelope beyond auto).
        architecture: The exact registered architecture selected by the loader,
            used for policy selection and error context.
        hf_config: Checkpoint metadata for a registered model policy. It may
            be omitted when the supplied architecture has no registered policy.
        quantization: The checkpoint's resolved quantization method, or None.
        vocab_size: The actual vocabulary size checked by the selected policy.

    Raises:
        ValueError: The envelope is not verified for this model, or its
            checkpoint validator rejects the storage or vocabulary. The
            selected model policy owns checkpoint validation. The standard
            policy requires unquantized
            weights and whole blockwise vocabulary blocks.
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
    model_rl_numerics(architecture, hf_config).validate(
        quantization=quantization, vocab_size=vocab_size, architecture=architecture
    )


def _validate_standard_rl_numerics(*, quantization, vocab_size, architecture):
    if quantization is not None:
        raise ValueError(
            "--numerics rl-bitwise serves unquantized checkpoints only: "
            f"{architecture} is {quantization}-quantized, and no "
            "batch-invariant quantized GEMM leaf exists"
        )
    if vocab_size % BLOCKWISE_VOCAB_BLOCK != 0:
        raise ValueError(
            "--numerics rl-bitwise folds --logprob-order blockwise, whose "
            f"sum(exp) runs over fixed {BLOCKWISE_VOCAB_BLOCK}-wide vocabulary "
            f"blocks; {architecture} has vocab_size {vocab_size}, not a "
            "multiple of it"
        )


def _resolve_standard_rl_numerics(server_args: ServerArgs) -> None:
    server_args._resolve_rl_bitwise()


STANDARD_RL_NUMERICS = RLNumericsPolicy(
    resolve=_resolve_standard_rl_numerics,
    validate=_validate_standard_rl_numerics,
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
    "register_model_rl_numerics",
    "model_rl_numerics",
    "launch_rl_numerics",
    "RLNumericsPolicy",
    "RLCheckpointValidator",
    "STANDARD_RL_NUMERICS",
    "resolve_numerics_envelope",
]
