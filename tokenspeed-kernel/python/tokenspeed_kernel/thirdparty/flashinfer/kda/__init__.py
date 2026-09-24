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

"""FlashInfer KDA adapters for indexed TS state and frozen verification.

FlashInfer remains optional. TS prepares inputs and owns persistent state,
copy-on-write indices and accepted-prefix replay; the public recurrent API
performs the scan without any upstream source changes.
"""

from __future__ import annotations

from functools import cache
from inspect import signature

import torch
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.thirdparty.flashinfer.kda._epilogue import _gated_rmsnorm_bf16
from tokenspeed_kernel.thirdparty.flashinfer.kda._prepare import (
    prepare_kda_recurrent_inputs,
)

FLASHINFER_KDA_MAX_VERIFY_TOKENS = 16


@cache
def flashinfer_kda_recurrent_available() -> bool:
    """Require both public modes and their optional CuTe implementations."""
    try:
        from flashinfer import kda_decode
    except (ImportError, OSError, RuntimeError):
        return False
    return bool(
        getattr(kda_decode, "_RECURRENT_KDA_AVAILABLE", False)
        and getattr(kda_decode, "_KDA_OUTPUT_ONLY_AVAILABLE", False)
        and "disable_state_update" in signature(kda_decode.recurrent_kda).parameters
    )


def _recurrent_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run one recurrent step with independent source and destination indices."""
    from flashinfer.kda_decode import recurrent_kda

    # FI's grouped branch rebuilds dense token strides for Q/K/V and beta.
    # TS supplies slices of packed projection/conv tensors; compact them.
    if not all(tensor.is_contiguous() for tensor in (q, k, v)):
        q, k, v = torch.stack((q, k, v)).unbind(0)
    beta_logits = beta_logits.contiguous()
    out, _ = recurrent_kda(
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log=A_log.detach(),
        dt_bias=dt_bias.detach(),
        scale=q.shape[-1] ** -0.5,
        initial_state=state_pool,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=write_indices,
        num_spec_tokens=None,
        num_accepted_tokens=None,
        output=None,
        initial_state_source=state_pool,
        initial_state_indices=read_indices,
        beta_is_logit=True,
        disable_state_update=False,
        correction_cache=None,
        kg_cache=None,
        backend="cute-dsl",
    )
    return out


def flashinfer_kda_producer_decode(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
    output_gate: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
) -> torch.Tensor:
    """Prepare T=1 inputs, update indexed state, and optionally apply output norm."""
    batch = mixed_qkv.shape[0]
    qkv, g, beta, _, _ = prepare_kda_recurrent_inputs(
        mixed_qkv,
        f_a_out,
        f_b_weight,
        beta_logits,
        conv_weights,
        conv_states,
        state_pool,
        read_indices,
        write_indices,
        replay_payload=None,
        tokens=1,
        state_scratch=None,
    )
    q, k, v = [value.view(1, batch, num_heads, head_dim) for value in qkv.unbind()]
    out = _recurrent_decode(
        q,
        k,
        v,
        g.view(1, batch, num_heads, head_dim),
        beta.view(1, batch, num_heads),
        A_log,
        dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
    )
    if output_gate is not None:
        out = _gated_rmsnorm_bf16(
            out.reshape(batch, num_heads * head_dim),
            output_gate,
            norm_weight,
            norm_eps,
            num_heads,
            head_dim,
            enable_pdl=pdl_enabled(),
        )
    return out.view(1, batch, num_heads, head_dim)


def flashinfer_kda_producer_verify(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor | None,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    replay_payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    """Verify a fixed-width window without modifying committed state.

    ``replay_payload`` contains preallocated raw-QKV, low-rank gate-input
    and beta buffers for the existing accepted-prefix replay commit.
    """
    # Explicit registry overrides bypass trait filtering. Reject unsupported
    # windows before preparing inputs or writing any replay payload.
    if not 1 <= draft_token_num <= FLASHINFER_KDA_MAX_VERIFY_TOKENS:
        raise ValueError(
            "FlashInfer KDA verify requires 1 <= draft_token_num <= "
            f"{FLASHINFER_KDA_MAX_VERIFY_TOKENS}; got {draft_token_num}. "
            "Use automatic dispatch or native Triton for longer windows."
        )
    from flashinfer.kda_decode import recurrent_kda

    batch = read_indices.numel()
    rows = batch * draft_token_num
    qkv, g, beta, state, slots = prepare_kda_recurrent_inputs(
        mixed_qkv[:rows],
        f_a_out[:rows],
        f_b_weight,
        beta_logits[:rows],
        conv_weights,
        conv_states,
        state_pool,
        read_indices,
        write_indices,
        tokens=draft_token_num,
        state_scratch=state_scratch,
        replay_payload=replay_payload,
    )
    shape = (batch, draft_token_num, num_heads, head_dim)
    q, k, v = [value.view(shape) for value in qkv.unbind()]
    out, _ = recurrent_kda(
        q,
        k,
        v,
        g.view(shape),
        beta.view(shape[:-1]),
        A_log=A_log.detach(),
        dt_bias=dt_bias.detach(),
        scale=head_dim**-0.5,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_spec_tokens=None,
        num_accepted_tokens=None,
        output=None,
        initial_state_source=state,
        initial_state_indices=slots,
        beta_is_logit=True,
        disable_state_update=True,
        correction_cache=None,
        kg_cache=None,
        backend="cute-dsl",
    )
    return out.view(1, rows, num_heads, head_dim)
