# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""FlashInfer Blackwell (sm100) gated delta net chunked prefill.

Fast-path for GDN prefill on Blackwell (sm100/sm103) + CUDA 13 + bf16 +
head_dim==128. The caller gates on ``is_supported`` and must fail fast when the
fast-path is unavailable;
this module has no Triton fallback.

Convention vs the Triton FLA path (verified equal to bf16 on B200):
- q, k must be L2-normalized by the caller (the sm100 kernel ignores its own
  use_qk_l2norm flag).
- g is the FLA log-space forget gate; the sm100 kernel takes log internally, so
  we pass alpha = exp(g).
- beta is cast to float32 (flashinfer passes it through without casting).
- the recurrent state is stored transposed (K<->V) vs FLA, so we transpose the
  initial state in and the final state out.
- scale defaults to head_dim**-0.5 (FLA default).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.gdn_utils import (
    GdnCheckpointLayout,
    GdnChunkPrefillResult,
)
from tokenspeed_kernel.ops.attention.triton.linear.l2norm import l2norm_fwd
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()
SUPPORTED_HEAD_DIM = 128

_chunk_gated_delta_rule = error_fn
_has_blackwell_prefill = False

if platform.is_blackwell:
    try:
        from flashinfer.gdn_kernels import (
            _has_blackwell_prefill,
        )
        from flashinfer.gdn_prefill import chunk_gated_delta_rule as _fi_chunk

        _chunk_gated_delta_rule = _fi_chunk
    except ImportError:
        pass


def is_available() -> bool:
    """Whether the sm100 GDN kernel can run on this platform."""
    cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    # flashinfer's gdn_prefill treats compute-capability major 10 as the
    # Blackwell path (sm100 B200/GB200, sm103 B300), gated on CUDA>=13 and the
    # prefill kernel being present; it raises NotImplementedError otherwise.
    # Mirror that here so the caller does not commit to a crashing fast-path.
    return (
        _chunk_gated_delta_rule is not error_fn
        and platform.is_blackwell
        and cuda_major >= 13
        and _has_blackwell_prefill
    )


def is_supported(
    head_dim: int, dtype: torch.dtype, num_q_heads: int, num_v_heads: int
) -> bool:
    # bf16 is the verified path; fp16 is rejected (caller fails fast).
    # flashinfer reads g/beta/state with max(num_q, num_v) heads; the runtime
    # supplies them with num_v heads, so num_v < num_q (e.g. Hk=32, Hv=16) reads
    # out of bounds. Only num_v >= num_q is safe (GVA or equal heads).
    return (
        is_available()
        and head_dim == SUPPORTED_HEAD_DIM
        and dtype == torch.bfloat16
        and num_v_heads >= num_q_heads
    )


CHUNK_SIZE = 64

gdn_chunk_prefill = error_fn

if is_available():

    @register_kernel(
        "attention",
        "gdn_chunk_prefill",
        name="flashinfer_gdn_chunk_prefill",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({SUPPORTED_HEAD_DIM}),
            "head_v_dim": frozenset({SUPPORTED_HEAD_DIM}),
            "head_v_eq_head_k": frozenset({True}),
            "num_v_gte_num_q": frozenset({True}),
            "qk_l2norm": frozenset({False, True}),
            "output_h": frozenset({False, True}),
        },
        tags={"blackwell", "latency"},
    )
    def gdn_chunk_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        scale: float | None,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        qk_l2norm: bool = False,
        output_final_state: bool = True,
        output_h: bool = False,
    ):
        """Run one chunked GDN prefill on the sm100 kernel.

        q, k, v are [B, T, H, D] (B==1) or [T, H, D]; q/k already L2-normalized.
        g, beta are [B, T, H] or [T, H]; g in log space. initial_state is the FLA
        [N, H, K, V] recurrent state.

        Default returns ``(out, final_state)`` matching the FLA layout:
        out [B, T, Hv, D] in q.dtype, final_state [N, H, K, V].

        When ``output_h=True`` returns
        ``(out, final_state, fi_checkpoints, checkpoint_cu_starts)`` where:
        - ``fi_checkpoints``: raw flashinfer checkpoint buffer in FLA state layout
          ``[total_fi_ckpts, H, K, V]`` (float32). Checkpoint ``k`` within a
          sequence is the state *after* processing chunk ``k`` (= FLA ``h[k+1]``).
          Per-sequence count is ``L_i // CHUNK_SIZE``.
        - ``checkpoint_cu_starts``: int64 tensor of length ``N+1`` giving the
          cumulative start offset of each sequence's checkpoints in
          ``fi_checkpoints``.

        The caller indexes directly with flashinfer-native offsets rather than
        rebuilding the FLA h tensor.
        """
        batched = q.dim() == 4
        q3 = q.squeeze(0) if batched else q
        k3 = k.squeeze(0) if batched else k
        v3 = v.squeeze(0) if batched else v
        g2 = g.squeeze(0) if g.dim() == 3 else g
        beta2 = beta.squeeze(0) if beta.dim() == 3 else beta

        if qk_l2norm:
            q3 = l2norm_fwd(q3)
            k3 = l2norm_fwd(k3)

        head_dim = q3.shape[-1]
        if scale is None:
            scale = head_dim**-0.5

        # flashinfer's recurrent-state layout is [N, H, V, K]; FLA uses [N, H, K, V].
        fi_initial_state = initial_state.float().transpose(-1, -2).contiguous()

        state_checkpoints = None
        checkpoint_cu_starts = None
        if output_h:
            per_seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
            # flashinfer-native checkpoint count: only full chunks emit a checkpoint.
            per_seq_h_ckpts = per_seq_lens // CHUNK_SIZE
            total_h_ckpts = int(per_seq_h_ckpts.sum().item())
            H_state = fi_initial_state.shape[1]
            D_state = fi_initial_state.shape[-1]
            state_checkpoints = torch.empty(
                total_h_ckpts,
                H_state,
                D_state,
                D_state,
                device=fi_initial_state.device,
                dtype=torch.float32,
            )
            checkpoint_cu_starts = torch.zeros(
                per_seq_h_ckpts.numel() + 1,
                device=fi_initial_state.device,
                dtype=torch.int64,
            )
            checkpoint_cu_starts[1:] = torch.cumsum(per_seq_h_ckpts, dim=0)

        out, final_state = _chunk_gated_delta_rule(
            q3.contiguous(),
            k3.contiguous(),
            v3.contiguous(),
            g=torch.exp(g2).float().contiguous(),
            beta=beta2.float().contiguous(),
            scale=scale,
            initial_state=fi_initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=CHUNK_SIZE if output_h else 0,
        )

        out = out.to(q.dtype)
        if batched:
            out = out.unsqueeze(0)
        final_state_fla = (
            final_state.transpose(-1, -2) if final_state is not None else None
        )

        if not output_h:
            return GdnChunkPrefillResult(out=out, final_state=final_state_fla)

        # Return raw flashinfer checkpoints in FLA state layout [total_fi, H, K, V].
        # The caller indexes directly using flashinfer-native offsets
        h_ckpts_fla = state_checkpoints.transpose(-1, -2)
        return GdnChunkPrefillResult(
            out=out,
            final_state=final_state_fla,
            h=h_ckpts_fla,
            h_cu_starts=checkpoint_cu_starts,
            h_layout=GdnCheckpointLayout.FLASHINFER,
        )
