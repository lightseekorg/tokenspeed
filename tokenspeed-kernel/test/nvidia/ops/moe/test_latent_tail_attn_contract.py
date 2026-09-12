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


"""Contracts of the NVIDIA-only CuteDSL collective behind K3's attention reduce.

These live under test/nvidia because importing the collective pulls in
cuda.bindings, which ROCm images do not have.
"""

import pytest
import torch
from tokenspeed_kernel.ops.moe.latent_tail import (
    attn_reduce_shape_supported,
    build_attn_reduce_collective,
)


def test_the_attention_builder_asks_for_the_residual_epilogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The attention reduce emits reduced+residual; a RMSNorm here is silent.

    The flag is a compile-time variant, so the wrong value produces a kernel
    that runs, returns the right shape, and normalises the sum instead of
    adding the residual to it. Nothing downstream would raise.
    """
    import tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail as tail_pkg
    from tokenspeed_kernel.ops.moe.latent_tail import build_attn_reduce_collective

    seen = {}

    def fake_kernel(**kwargs):
        seen.update(kwargs)
        return "kernel"

    monkeypatch.setattr(tail_pkg, "CollectiveKernel", fake_kernel)
    built = build_attn_reduce_collective(
        group="group", rank=3, tp_size=8, hidden_size=7168, max_tokens=8
    )

    assert built == "kernel"
    assert seen["residual_from_shared"] is True
    # A latent narrower than hidden makes the residual read walk the wrong row.
    assert seen["latent_dim"] == 7168 and seen["hidden_dim"] == 7168
    # Capacity has to match the caller's window, and rank is not size.
    assert seen["max_m"] == 8 and seen["max_token_ctas"] == 8
    assert seen["rank"] == 3 and seen["tp_size"] == 8
    # __call__ refuses a split dispatch without it.
    assert seen["precompile_split"] is True


def test_the_attention_shape_probe_declines_instead_of_raising() -> None:
    """The constructor raises inside a rendezvous, so the probe answers first."""
    from tokenspeed_kernel.ops.moe.latent_tail import attn_reduce_shape_supported

    assert attn_reduce_shape_supported(tp_size=8, hidden_size=7168)
    # Cluster width is tp_size here, and the kernel takes powers of two up to 16.
    for tp in (7, 12, 24, 32):
        assert not attn_reduce_shape_supported(tp_size=tp, hidden_size=7168)


def test_the_epilogue_variant_is_part_of_the_compile_key() -> None:
    """Two epilogues sharing a key means one kernel is returned for the other.

    ``_COMPILED`` is module-global. K3's two live instances also differ in
    latent_dim and max_m, so they cannot collide today; the field still
    belongs in the key, because two instances of the same geometry differing
    only in the epilogue would otherwise share a kernel that runs and returns
    the right shape.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.allreduce_rmsnorm_reduce_scatter_early_exit import (  # noqa: E501
        _compile_key,
    )

    common = dict(
        rank=0,
        tp_size=8,
        latent_dim=7168,
        hidden_dim=7168,
        max_m=8,
        max_token_ctas=8,
        fp32_internal=True,
        include_reduce_scatter=False,
        include_routed=True,
    )
    residual = _compile_key(**common, residual_from_shared=True)
    rmsnorm = _compile_key(**common, residual_from_shared=False)
    assert residual != rmsnorm
    # Control: a key that separated on nothing would satisfy the line above too.
    assert (
        _compile_key(**{**common, "max_m": 64}, residual_from_shared=True) != residual
    )
