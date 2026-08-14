"""The fused FP8 decode gate must resolve hybrid sub-backends.

``data_type`` sits on the full-attention sub-backend for hybrid models. The gate
used to read it off the outer backend only, so a second ``elif`` existed to
catch NoPE + fp8 configurations the gate had missed. That branch was in fact
unreachable: reaching it required the sub-backend to be fp8 *and* the outer
backend not to be, while every supported wrapper either forwards ``data_type``
or has no sub-backend at all. Resolving the sub-backend in the gate itself makes
that coverage structural rather than incidental.
"""

from __future__ import annotations

import inspect

import torch


def _resolve(backend):
    """The gate's lookup, as the model performs it."""
    kv_backend = getattr(backend, "full_attn_backend", backend)
    return getattr(kv_backend, "data_type", None)


class _Plain:
    """Dense / DSA shape: holds data_type, no sub-backend."""

    data_type = torch.float8_e4m3fn


class _Forwarding:
    """hybrid_linear_attn shape: exposes a sub-backend and forwards data_type."""

    full_attn_backend = _Plain()

    @property
    def data_type(self):
        return self.full_attn_backend.data_type


class _NotForwarding:
    """msa shape: exposes a sub-backend but no data_type of its own."""

    full_attn_backend = _Plain()


def test_gate_sees_fp8_through_every_wrapper_shape():
    for backend in (_Plain(), _Forwarding(), _NotForwarding()):
        assert (
            _resolve(backend) == torch.float8_e4m3fn
        ), f"{type(backend).__name__} hides its fp8 kv dtype from the fused gate"


def test_resolution_is_what_makes_the_nope_branch_redundant():
    # The removed branch fired exactly when the sub-backend was fp8 and the
    # outer backend was not. _NotForwarding is the only shape where those two
    # differ, and the resolving gate now catches it, so nothing falls through.
    for backend in (_Plain(), _Forwarding(), _NotForwarding()):
        outer_is_fp8 = getattr(backend, "data_type", None) == torch.float8_e4m3fn
        gate_fires = _resolve(backend) == torch.float8_e4m3fn
        would_have_needed_the_branch = gate_fires and not outer_is_fp8
        assert gate_fires or not would_have_needed_the_branch, (
            f"{type(backend).__name__} would reach the removed NoPE branch "
            "without being caught by the fused gate"
        )

    assert not getattr(_NotForwarding(), "data_type", None), (
        "the shape that motivated the removed branch no longer exists; "
        "revisit whether this test still guards anything"
    )


def test_model_gate_resolves_the_sub_backend():
    from tokenspeed.runtime.models import deepseek_v3

    src = inspect.getsource(deepseek_v3.DeepseekV3AttentionMLA._mla_kv_is_fp8)
    assert (
        'getattr(ctx.attn_backend, "full_attn_backend", ctx.attn_backend)' in src
    ), "the fused fp8 gate no longer resolves the hybrid sub-backend"

    cls_src = inspect.getsource(deepseek_v3.DeepseekV3AttentionMLA)
    assert (
        'getattr(ctx.attn_backend, "data_type", None) == torch.float8_e4m3fn'
        not in cls_src
    ), "a gate reads data_type off the outer backend again; hybrids will miss it"


def test_decode_and_prefill_share_one_gate():
    from tokenspeed.runtime.models import deepseek_v3

    cls = deepseek_v3.DeepseekV3AttentionMLA
    decode = inspect.getsource(cls.forward_absorb_qkv_proj)
    prefill = inspect.getsource(cls.forward_normal_chunked_kv_prepare)

    for name, src in (("decode", decode), ("prefill", prefill)):
        assert (
            "self._mla_kv_is_fp8(ctx, k_scale)" in src
        ), f"the {name} path open-codes the fp8 gate instead of sharing it"


def test_every_hybrid_backend_forwards_data_type():
    """Subclasses must keep forwarding, or they silently lose the fused path.

    A hybrid wrapper that owns a sub-backend but no ``data_type`` reads as
    non-fp8 to the gate, so the model would fall back to the unfused path
    without any error. ``HybridKDABackend`` is the live example of a subclass
    that inherits this rather than restating it.
    """
    # Subclasses only register once their module is imported.
    import tokenspeed.runtime.layers.attention.backends.hybrid_kda  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
        HybridLinearAttnBackend,
    )

    def descendants(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from descendants(sub)

    for cls in (HybridLinearAttnBackend, *descendants(HybridLinearAttnBackend)):
        assert isinstance(
            getattr(cls, "data_type", None), property
        ), f"{cls.__name__} does not forward data_type; the fused fp8 path dies silently"


def test_dead_nope_query_branch_is_gone():
    from tokenspeed.runtime.models import deepseek_v3

    src = inspect.getsource(deepseek_v3)
    assert (
        "mla_nope_query_fp8" not in src
    ), "the unreachable NoPE query-assembly branch is back in the model"
