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

"""Registration shims for the AMD Gluon GDN prefill, decode and MTP-verify kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.attention.gdn import GdnStateCheckpoints
    from tokenspeed_kernel.ops.attention.kda import KdaOutputNorm

if current_platform().is_amd:
    from tokenspeed_kernel.ops.attention.gdn import GdnChunkPrefillResult
    from tokenspeed_kernel.ops.attention.gdn._triton.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from tokenspeed_kernel.ops.attention.gdn._triton.cumsum import chunk_local_cumsum
    from tokenspeed_kernel.ops.attention.gdn._triton.l2norm import l2norm_fwd
    from tokenspeed_kernel.ops.attention.gdn._triton.solve_tril import solve_tril
    from tokenspeed_kernel.ops.attention.gdn._triton.wy_fast import recompute_w_u_fwd
    from tokenspeed_kernel_amd.ops.gfx950.attention.gdn.chunk_prefill import (
        GDN_CHUNK_PREFILL_CHUNK,
        GDN_CHUNK_PREFILL_HEAD_DIM,
        launch_gluon_gdn_chunk_prefill_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.gdn.decode_mtp import (
        GDN_DECODE_MTP_KEY_DIM,
        GDN_DECODE_MTP_MAX_TOKENS,
        GDN_DECODE_MTP_STATE_DTYPES,
        GDN_DECODE_MTP_VALUE_DIMS,
        launch_gluon_gdn_decode_mtp_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.gdn.decode_step import (
        GDN_DECODE_STEP_KEY_DIM,
        GDN_DECODE_STEP_STATE_DTYPES,
        GDN_DECODE_STEP_VALUE_DIMS,
        launch_gluon_gdn_decode_step_gfx950,
    )

    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )

    @register_kernel(
        "attention",
        "gdn_chunk_prefill",
        name="gluon_gdn_chunk_prefill_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({GDN_CHUNK_PREFILL_HEAD_DIM}),
            "value_head_dim": frozenset({GDN_CHUNK_PREFILL_HEAD_DIM}),
            "num_v_gte_num_q": frozenset({True}),
            # Per-chunk states never leave the chip: no output_h.
            "output_h": frozenset({False}),
            "qk_l2norm": frozenset({False, True}),
            "state_checkpoints": frozenset({False, True}),
        },
    )
    def gluon_gdn_chunk_prefill_gfx950(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        scale: float | None,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        qk_l2norm: bool,
        output_final_state: bool,
        output_h: bool,
        state_checkpoints: GdnStateCheckpoints | None,
    ) -> GdnChunkPrefillResult:
        """gfx950 ``gdn_chunk_prefill``: FLA's Triton WY preparation, then the
        chunk recurrence and output as a two-level segment scan.

        The scan's states (final state, checkpoints) and the outputs after a
        sequence's first segment differ from the serial chunk scan by
        rounding; see the gfx950 kernel README for the tolerance.
        """
        if output_h:
            raise NotImplementedError("the gfx950 GDN chunk prefill has no output_h")
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if qk_l2norm:
            q, k = l2norm_fwd(q), l2norm_fwd(k)
        chunk = GDN_CHUNK_PREFILL_CHUNK
        g = chunk_local_cumsum(g, chunk_size=chunk, cu_seqlens=cu_seqlens)
        a = chunk_scaled_dot_kkt_fwd(
            k=k,
            beta=beta,
            g_cumsum=g,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
        )
        a = solve_tril(A=a, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(
            k=k, v=v, beta=beta, A=a, g_cumsum=g, cu_seqlens=cu_seqlens
        )
        out, final_state = launch_gluon_gdn_chunk_prefill_gfx950(
            q,
            k,
            w,
            u,
            g,
            initial_state=initial_state.contiguous(),
            cu_seqlens=cu_seqlens.to(torch.int32),
            scale=q.shape[-1] ** -0.5 if scale is None else scale,
            checkpoint_states=(
                None if state_checkpoints is None else state_checkpoints.states
            ),
            checkpoint_cu_starts=(
                None if state_checkpoints is None else state_checkpoints.cu_starts
            ),
            checkpoint_every_n_tokens=(
                None if state_checkpoints is None else state_checkpoints.every_n_tokens
            ),
        )
        return GdnChunkPrefillResult(
            out=out, final_state=final_state if output_final_state else None
        )

    @register_kernel(
        "attention",
        "gdn_decode_step",
        name="gluon_gdn_decode_step_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({GDN_DECODE_STEP_KEY_DIM}),
            "value_head_dim": GDN_DECODE_STEP_VALUE_DIMS,
            "state_dtype": GDN_DECODE_STEP_STATE_DTYPES,
            "padding_safe": frozenset({False, True}),
            "strided_qkv": frozenset({False, True}),
            "strided_state": frozenset({False, True}),
        },
    )
    def gluon_gdn_decode_step_gfx950(
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
        output_state_indices: torch.Tensor | None,
        use_qk_l2norm: bool,
    ) -> torch.Tensor:
        """gfx950 ``gdn_decode_step``: the state goes to ``output_state_indices``
        when given, else back to ``initial_state_indices``.

        One token needs no chunk algebra, so the step keeps its own per-token
        kernel rather than the verify's chunk form."""
        return launch_gluon_gdn_decode_step_gfx950(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state=initial_state,
            read_indices=initial_state_indices,
            write_indices=(
                initial_state_indices
                if output_state_indices is None
                else output_state_indices
            ),
            scale=q.shape[-1] ** -0.5 if scale is None else scale,
            use_qk_l2norm=use_qk_l2norm,
        )

    @register_kernel(
        "attention",
        "gdn_decode_mtp",
        name="gluon_gdn_decode_mtp_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({GDN_DECODE_MTP_KEY_DIM}),
            "value_head_dim": GDN_DECODE_MTP_VALUE_DIMS,
            "state_dtype": GDN_DECODE_MTP_STATE_DTYPES,
            "padding_safe": frozenset({False, True}),
            "tree": frozenset({False}),
            # The chunk (WY) form solves all tokens of a request at once.
            "verify_tokens": frozenset(range(1, GDN_DECODE_MTP_MAX_TOKENS + 1)),
            "output_norm": frozenset({False, True}),
            "separate_output_state": frozenset({False, True}),
            "strided_qkv": frozenset({False, True}),
            "strided_state": frozenset({False, True}),
        },
    )
    def gluon_gdn_decode_mtp_gfx950(
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
        output_norm: KdaOutputNorm | None = None,
    ) -> torch.Tensor:
        """gfx950 chain ``gdn_decode_mtp`` (draft trees run the Triton kernel).

        ``gdn_decode_mtp`` passes ``output_norm`` only when it is set; the
        kernel then applies the gated RMSNorm to each head's output.
        """
        if tree_ancestors is not None or replay_payload is not None:
            raise NotImplementedError("the gfx950 GDN verify kernel follows a chain")
        # Per-token rows (in output_state, default the pool itself) or the
        # dense intermediate buffer; the read row takes the final state only
        # when neither applies and the update is enabled.
        step_states = None
        if output_state_indices is not None:
            step_states = initial_state if output_state is None else output_state
        elif intermediate_states_buffer is not None:
            step_states = intermediate_states_buffer.view(-1, *initial_state.shape[1:])
        write_back = not disable_state_update and output_state_indices is None
        return launch_gluon_gdn_decode_mtp_gfx950(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state=initial_state,
            read_indices=initial_state_indices,
            final_indices=initial_state_indices if write_back else None,
            step_states=step_states,
            step_indices=output_state_indices,
            scale=q.shape[-1] ** -0.5 if scale is None else scale,
            use_qk_l2norm=use_qk_l2norm,
            norm_gate=None if output_norm is None else output_norm.gate,
            norm_weight=None if output_norm is None else output_norm.weight,
            norm_eps=None if output_norm is None else output_norm.eps,
        )
