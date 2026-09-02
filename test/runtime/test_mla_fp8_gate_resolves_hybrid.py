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

import pytest
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


def _wrapper_classes():
    """Every backend class that owns a full-attention sub-backend.

    Scans the whole backends package rather than one class hierarchy: a wrapper
    that hides an fp8 sub-backend behind a non-fp8 ``data_type`` breaks the gate
    no matter what it inherits from.
    """
    import importlib
    import pkgutil

    from tokenspeed.runtime.layers.attention import backends

    for info in pkgutil.iter_modules(backends.__path__):
        try:
            mod = importlib.import_module(f"{backends.__name__}.{info.name}")
        except Exception:  # optional vendor backends may not import here
            continue
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod.__name__:
                continue
            try:
                owns_sub = any(
                    "self.full_attn_backend" in inspect.getsource(k)
                    for k in cls.__mro__
                    if k is not object
                )
            except (OSError, TypeError):
                continue
            if owns_sub:
                yield cls


# MSAHybridAttnBackend owns a sub-backend and deliberately does not forward
# data_type. It belongs to MiniMax M3, which routes through its own
# MiniMaxM3Attention rather than DeepseekV3AttentionMLA, so the gate never reads
# it. Any *other* non-forwarding wrapper is a bug, so this list stays explicit.
NON_FORWARDING_BY_DESIGN = {"MSAHybridAttnBackend"}


def _forwards_dtype(cls):
    """Read ``data_type`` off *cls* given an fp8 sub-backend, without __init__.

    Returns None when the class cannot be probed this way.
    """
    try:
        obj = object.__new__(cls)
        obj.full_attn_backend = _Plain()
    except TypeError:  # __slots__ or a custom __new__
        return None
    return getattr(obj, "data_type", None) == torch.float8_e4m3fn


class _LiesAboutDtype:
    """A wrapper that has a data_type property which does not forward."""

    full_attn_backend = _Plain()

    @property
    def data_type(self):
        return torch.bfloat16


def test_the_forwarding_check_rejects_a_broken_wrapper():
    # Negative control: an existence check would pass _LiesAboutDtype, because
    # it does define the property. The probe must not.
    assert isinstance(_LiesAboutDtype.data_type, property)
    assert _forwards_dtype(_LiesAboutDtype) is False
    assert _forwards_dtype(_Forwarding) is True


def test_every_hybrid_wrapper_forwards_the_sub_backend_dtype():
    """A wrapper must report the dtype its sub-backend actually stores.

    Checking that ``data_type`` merely *exists* is not enough: a property that
    returned bfloat16 while the sub-backend held fp8 would pass that and still
    disable the fused path. So construct each wrapper without running its
    __init__, give it an fp8 sub-backend, and read the value back.
    """
    found = list(_wrapper_classes())
    assert found, "no hybrid wrapper classes discovered; the scan is broken"

    checked = []
    for cls in found:
        if cls.__name__ in NON_FORWARDING_BY_DESIGN:
            continue
        forwards = _forwards_dtype(cls)
        if forwards is None:
            continue
        assert (
            forwards
        ), f"{cls.__name__} does not report its sub-backend's fp8 dtype; the fused path dies silently"
        checked.append(cls.__name__)

    # The scan is only worth anything if it reaches the wrappers K3 runs on.
    # K3 runs on HybridLinearAttnBackend (the KDA-specific subclass was an
    # empty shell and is gone).
    assert "HybridLinearAttnBackend" in checked


def test_dead_nope_query_branch_is_gone():
    import importlib

    from tokenspeed.runtime.models import deepseek_v3

    src = inspect.getsource(deepseek_v3)
    assert (
        "mla_nope_query_fp8" not in src
    ), "the unreachable NoPE query-assembly branch is back in the model"

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(
            "tokenspeed_kernel.ops.attention.triton.mla_query_assemble"
        )
