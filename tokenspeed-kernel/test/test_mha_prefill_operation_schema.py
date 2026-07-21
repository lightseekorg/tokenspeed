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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.contracts.ops.attention.mha_prefill import MHA_PREFILL
from tokenspeed_kernel.operation import OperationRegistry
from tokenspeed_kernel.registry import KernelRegistry, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    tensor_format,
)


def _signature(dtype=torch.bfloat16):
    tensor = dense_tensor_format(dtype)
    return format_signature(q=tensor, k=tensor, v=tensor)


def _register(*, signatures, traits=None, name="test_mha_prefill"):
    return register_kernel(
        "attention",
        "mha_prefill",
        name=name,
        solution="test",
        signatures=signatures,
        traits=traits,
    )


def _adapter(
    *,
    q,
    k,
    v,
    cu_seqlens,
    cu_seqlens_cpu,
    max_seqlen,
    window_left=-1,
    logit_cap=0.0,
    sinks=None,
    return_lse=False,
    softmax_scale=None,
):
    return MHA_PREFILL.reference(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=max_seqlen,
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
        softmax_scale=softmax_scale,
    )


def test_schema_is_published_and_reference_is_not_selectable() -> None:
    assert OperationRegistry.get().lookup("attention", "mha_prefill") is MHA_PREFILL
    names = {
        spec.name
        for spec in KernelRegistry.get().get_for_operator("attention", "mha_prefill")
    }
    assert "mha_prefill_reference" not in names


def test_reference_defines_causal_attention_and_sink_semantics() -> None:
    q = torch.zeros((2, 1, 1), dtype=torch.float32)
    k = torch.zeros_like(q)
    v = torch.tensor([[[2.0]], [[6.0]]])
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

    output, lse = MHA_PREFILL.reference(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=[0, 2],
        max_seqlen=2,
        sinks=torch.tensor([0.0]),
        return_lse=True,
    )

    torch.testing.assert_close(output[:, 0, 0], torch.tensor([1.0, 8.0 / 3.0]))
    torch.testing.assert_close(lse[:, 0], torch.log(torch.tensor([2.0, 3.0])))


def test_reference_requires_boundaries_to_cover_packed_inputs() -> None:
    q = torch.zeros((2, 1, 1), dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="inconsistent packed sequence metadata"):
        MHA_PREFILL.reference(
            q=q,
            k=q,
            v=q,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=[0, 1],
            max_seqlen=1,
        )


@pytest.mark.parametrize(
    "traits",
    [
        None,
        {
            "head_dim": frozenset({64, 128}),
            "sliding_window": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "support_sinks": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    ],
)
def test_valid_string_registration_is_accepted(fresh_registry, traits) -> None:
    _register(signatures={_signature()}, traits=traits)(_adapter)

    assert KernelRegistry.get().get_impl("test_mha_prefill") is _adapter


def test_signature_dtypes_are_kernel_claims_not_schema_policy(fresh_registry) -> None:
    signature = format_signature(
        q=dense_tensor_format(torch.float32),
        k=dense_tensor_format(torch.bfloat16),
        v=dense_tensor_format(torch.float16),
    )

    _register(signatures={signature})(_adapter)

    assert KernelRegistry.get().get_impl("test_mha_prefill") is _adapter


@pytest.mark.parametrize(
    "signature, error",
    [
        (
            format_signature(q=dense_tensor_format(torch.bfloat16)),
            "require roles",
        ),
        (
            format_signature(
                q=tensor_format("mxfp8", torch.uint8),
                k=tensor_format("mxfp8", torch.uint8),
                v=tensor_format("mxfp8", torch.uint8),
            ),
            "unscaled dense",
        ),
    ],
)
def test_invalid_signature_claim_is_rejected(
    fresh_registry, signature, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _register(signatures={signature})(_adapter)


@pytest.mark.parametrize(
    "traits, error",
    [
        ({"page_size": frozenset({64})}, "unknown.*page_size"),
        ({"head_dim": frozenset({0})}, "positive integers"),
        ({"head_dim": frozenset({True})}, "positive integers"),
        ({"return_lse": frozenset({1})}, "values must be bool"),
        ({"support_sinks": {True}}, "non-empty frozenset"),
    ],
)
def test_invalid_trait_claim_is_rejected(fresh_registry, traits, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _register(signatures={_signature()}, traits=traits)(_adapter)


def test_builtin_registrations_satisfy_schema() -> None:
    specs = KernelRegistry.get().get_for_operator("attention", "mha_prefill")
    assert specs
    for spec in specs:
        MHA_PREFILL.validate_registration(
            spec,
            KernelRegistry.get().get_impl(spec.name),
        )
