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

"""CuTe DSL output-only GDN WY verification for ReplaySSM."""

from __future__ import annotations

import importlib.util

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if (
    current_platform().is_hopper_plus
    and importlib.util.find_spec("cutlass") is not None
):

    @register_kernel(
        "attention",
        "gdn_decode_mtp",
        name="cute_dsl_gdn_decode_mtp",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            max_arch_version=ArchVersion(10, 7),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED + 1,
        traits={
            "head_dim": frozenset({128}),
            "value_head_dim": frozenset({128}),
            "verify_tokens": frozenset(range(1, 17)),
            "output_only": frozenset({True}),
            "tree": frozenset({False, True}),
            "separate_output_state": frozenset({False}),
            "strided_qkv": frozenset({False, True}),
        },
    )
    def cute_dsl_gdn_decode_mtp(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        b: torch.Tensor,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        scale: float | None,
        disable_state_update: bool,
        use_qk_l2norm: bool,
        intermediate_states_buffer: torch.Tensor | None,
        output_state_indices: torch.Tensor | None,
        output_state: torch.Tensor | None,
        tree_ancestors: torch.Tensor | None,
        replay_payload: torch.Tensor | None,
    ) -> torch.Tensor:
        """Verify a GDN chain or tree in WY form without writing recurrent states.

        Args:
            q: BF16/FP16 queries shaped ``[B, T, H, 128]``; strides may vary.
            k: Keys with the same shape and dtype as ``q``.
            v: Values shaped ``[B, T, HV, 128]`` with ``q.dtype``; ``HV % H == 0``.
            A_log: Contiguous floating-point log decay parameters shaped ``[HV]``.
            a: Floating-point decay inputs shaped ``[B, T, HV]``.
            dt_bias: Contiguous floating-point decay biases shaped ``[HV]``.
            b: Floating-point beta logits shaped ``[B, T, HV]``.
            initial_state: Read-only BF16/FP16/FP32 K-last pool shaped
                ``[rows, HV, 128, 128]``. Physical pool rows may have padding.
            initial_state_indices: Contiguous int32/int64 pool rows shaped ``[B]``.
                Negative rows start from zero; their outputs may be ignored.
            scale: Query scale; ``None`` means ``128**-0.5``.
            disable_state_update: Must be ``True``.
            use_qk_l2norm: Whether to normalize Q/K in FP32 with epsilon ``1e-6``.
            intermediate_states_buffer: Must be ``None``; no node states are saved.
            output_state_indices: Must be ``None``.
            output_state: Must be ``None``.
            tree_ancestors: Optional contiguous int64 ``[B, T]`` ancestor-or-self
                masks; ``None`` describes a chain. Ancestors precede descendants.
            replay_payload: Optional contiguous ``[B*T, H*128 + HV*128 + 2*HV]``
                destination for exact raw ``[k | v | a | b]`` rows. Its dtype and
                those of ``a``/``b`` must equal ``q.dtype``.

        Returns:
            Output shaped ``[B, T, HV, 128]`` with ``q.dtype``, for ``1 <= T <= 16``.
            Products use native TF32 MMA and FP32 accumulation. WY coefficients use
            FP32 triangular solves, or backward query propagation for four tokens.
            Warm the selected dtype/geometry/4/8/16-token bucket before capture.
        """
        if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
            raise ValueError(
                "q/k must have equal [B,T,H,128] shapes; v must be [B,T,HV,128]"
            )
        B, T, H, K = q.shape
        HV, V = v.shape[2:]
        if (
            K != 128
            or V != 128
            or not 1 <= T <= 16
            or H <= 0
            or HV % H
            or HV <= 0
            or v.shape[:2] != (B, T)
        ):
            raise ValueError("GDN WY requires K=V=128 and 1 <= T <= 16")
        if not disable_state_update or any(
            x is not None
            for x in (intermediate_states_buffer, output_state_indices, output_state)
        ):
            raise ValueError("GDN WY is output-only")
        if (
            q.dtype not in (torch.bfloat16, torch.float16)
            or k.dtype != q.dtype
            or v.dtype != q.dtype
        ):
            raise ValueError("q/k/v must share BF16 or FP16 dtype")
        floating = (torch.bfloat16, torch.float16, torch.float32)
        if (
            a.shape != (B, T, HV)
            or b.shape != (B, T, HV)
            or a.dtype not in floating
            or b.dtype not in floating
        ):
            raise ValueError("a/b must be floating [B,T,HV] tensors")
        for parameter in (A_log, dt_bias):
            if (
                parameter.shape != (HV,)
                or not parameter.is_contiguous()
                or parameter.dtype not in floating
            ):
                raise ValueError(
                    "A_log/dt_bias must be contiguous floating [HV] tensors"
                )
        if (
            initial_state.ndim != 4
            or initial_state.shape[1:] != (HV, 128, 128)
            or initial_state.stride()[1:] != (16384, 128, 1)
            or initial_state.stride(0) < HV * 16384
            or initial_state.dtype not in floating
        ):
            raise ValueError("initial_state must use K-last [pool,HV,128,128] storage")
        if (
            initial_state_indices.shape != (B,)
            or initial_state_indices.dtype not in (torch.int32, torch.int64)
            or not initial_state_indices.is_contiguous()
        ):
            raise ValueError("initial_state_indices must be contiguous int32/int64 [B]")
        if tree_ancestors is not None and (
            tree_ancestors.shape != (B, T)
            or tree_ancestors.dtype != torch.int64
            or not tree_ancestors.is_contiguous()
        ):
            raise ValueError("tree_ancestors must be contiguous int64 [B,T]")
        if replay_payload is not None:
            if (
                replay_payload.shape != (B * T, H * K + HV * V + 2 * HV)
                or not replay_payload.is_contiguous()
                or {replay_payload.dtype, a.dtype, b.dtype} != {q.dtype}
            ):
                raise ValueError(
                    "replay_payload must be contiguous [B*T,H*K+HV*V+2*HV], with the dtype of q/k/v/a/b"
                )
        inputs = (q, k, v, a, b, A_log, dt_bias, initial_state, initial_state_indices)
        if tree_ancestors is not None:
            inputs = inputs + (tree_ancestors,)
        if replay_payload is not None:
            inputs = inputs + (replay_payload,)
        if q.device.type != "cuda" or any(x.device != q.device for x in inputs):
            raise ValueError("all GDN WY inputs must be on the same CUDA device")

        from tokenspeed_kernel.ops.attention.gdn._cute_dsl.wy import launch_gdn_wy

        return launch_gdn_wy(
            q,
            k,
            v,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            tree_ancestors=tree_ancestors,
            replay_payload=replay_payload,
        )
