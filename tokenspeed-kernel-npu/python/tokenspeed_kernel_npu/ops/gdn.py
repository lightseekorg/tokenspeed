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

"""Ascend Gated Delta Net over the host's K-last recurrent-state pool.

The device work is CANN's native Gated Delta Rule family:

* ``npu_chunk_gated_delta_rule`` scans a whole prefill batch, taking one
  ``actual_seq_lengths`` entry per request and one state row per request.
* ``npu_recurrent_gated_delta_rule`` scans token by token, taking one pool row
  per token through ``ssm_state_indices``. A request's tokens all name the same
  row, so the op carries the recurrence across the window; the final state
  lands on that row.

Both pools are K-last ``[rows, Hv, Dv, Dk]`` bfloat16, which is the runtime's
own SSM layout, so no transposes are needed at the call sites.

Four properties of these ops shape the code below:

* The recurrence reads row ``ssm_state_indices[0]`` and writes token ``t``'s
  resulting state to ``ssm_state_indices[t]``. A window that must leave the
  committed row alone therefore stages it in scratch rows and names those.
* ``BlockNum >= T``: the tiling rejects a pool with fewer rows than the call
  has tokens, so every entry checks that bound.
* ``beta`` is the activated update coefficient, not a logit. The portable
  Triton decode kernel applies its own sigmoid, so this module does it first.
* A negative ``ssm_state_indices`` entry faults the device instead of acting as
  a padded row. A padding row is therefore pointed at the reserved null page
  (row 0), whose zero state it reads, and that page is restored after the
  launch so a live row is never touched.

``num_accepted_tokens`` is not usable on Ascend 950PR: every value yields the
same output and final state as the full window. A verify that stops at a
shorter accepted prefix runs that prefix length instead.
"""

from __future__ import annotations

import torch
import torch_npu

# The native ops take bfloat16 activations; the recurrent pool may be bf16 or
# fp32 (Qwen3.6 selects fp32 through ``--mamba-ssm-dtype``).
_ACTIVATION_DTYPE = torch.bfloat16
_STATE_DTYPES = (torch.bfloat16, torch.float32)

# Softplus clamp: above this the portable kernel returns the input unchanged.
_SOFTPLUS_THRESHOLD = 20.0


def _require_bf16(tensor: torch.Tensor, what: str) -> None:
    if tensor.dtype != _ACTIVATION_DTYPE:
        raise NotImplementedError(
            f"Ascend GDN requires a bfloat16 {what}, got {tensor.dtype}"
        )


def _require_state_dtype(tensor: torch.Tensor) -> None:
    if tensor.dtype not in _STATE_DTYPES:
        raise NotImplementedError(
            "Ascend GDN requires a bfloat16 or float32 recurrent state, got "
            f"{tensor.dtype}"
        )


def _check_heads(k: torch.Tensor, v: torch.Tensor) -> None:
    num_k_heads, num_v_heads = k.shape[-2], v.shape[-2]
    if num_k_heads <= 0 or num_v_heads % num_k_heads:
        raise ValueError(
            f"num_v_heads ({num_v_heads}) must be a positive multiple of "
            f"num_k_heads ({num_k_heads})"
        )


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    """L2-normalize the last axis in fp32, then narrow to bf16, as the kernels do.

    The portable kernels normalize with ``sqrt(sum(x^2) + 1e-6)``; the epsilon
    keeps a zero row finite instead of producing NaNs.
    """
    flat = tensor.float()
    norm = torch.sqrt((flat * flat).sum(-1, keepdim=True) + 1e-6)
    return (flat / norm).to(_ACTIVATION_DTYPE)


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
    qk_l2norm: bool,
    output_final_state: bool = True,
    output_h: bool = False,
    state_checkpoints: object = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan a whole prefill batch with one native chunked GDR call.

    Args:
        q: ``[1, total_tokens, Hk, Dk]`` query.
        k: ``[1, total_tokens, Hk, Dk]`` key.
        v: ``[1, total_tokens, Hv, Dv]`` value.
        g: ``[1, total_tokens, Hv]`` fp32 log-space decay.
        beta: ``[1, total_tokens, Hv]`` activated update coefficient.
        scale: Softmax scale; ``None`` uses ``Dk ** -0.5``.
        initial_state: Per-request K-last state ``[B, Hv, Dv, Dk]``; the op
            overwrites it with the final state.
        cu_seqlens: int32 cumulative token offsets, ``[B + 1]``.
        qk_l2norm: Whether to L2-normalize q and k before the scan.
        output_final_state: Whether the caller consumes the final state; the
            native op returns it either way.
        output_h: Rejected; the native scan returns no intermediate
            checkpoints.
        state_checkpoints: Rejected, for the same reason as ``output_h``.

    Returns:
        ``(out, final_state)`` where ``out`` is
        ``[1, total_tokens, Hv, Dv]`` and ``final_state`` keeps the
        ``initial_state`` layout.
    """
    del output_final_state
    if output_h or state_checkpoints is not None:
        raise NotImplementedError(
            "Ascend chunk GDN returns the final state only, without intermediate "
            "checkpoints"
        )
    _require_bf16(q, "query")
    _require_state_dtype(initial_state)
    _check_heads(k[0], v[0])

    batch_size = cu_seqlens.shape[0] - 1
    total_tokens = q.shape[1]
    if initial_state.shape[0] < batch_size:
        raise ValueError(
            "Ascend chunk GDN takes one state row per request: "
            f"{initial_state.shape[0]} rows for {batch_size} requests"
        )

    query = q[0]
    key = k[0]
    if qk_l2norm:
        query = _normalize(query)
        key = _normalize(key)

    out, final_state = torch_npu.npu_chunk_gated_delta_rule(
        query.contiguous(),
        key.contiguous(),
        v[0].contiguous(),
        beta=beta[0].contiguous(),
        initial_state=initial_state,
        actual_seq_lengths=(cu_seqlens[1:] - cu_seqlens[:-1])[:batch_size].to(
            torch.int32
        ),
        scale=query.shape[-1] ** -0.5 if scale is None else scale,
        g=g[0].contiguous(),
    )
    if out.shape[0] != total_tokens:
        raise RuntimeError(
            f"Ascend chunk GDN returned {out.shape[0]} rows for {total_tokens} tokens"
        )
    return out.unsqueeze(0), final_state


def gdn_decode_step(
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
    scale: float | None = None,
    output_state_indices: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """One single-token GDN decode step, one pool row per request.

    The state is always updated: to ``output_state_indices`` when given, else
    back onto ``initial_state_indices`` in place.

    Args:
        q: ``[B, 1, Hk, Dk]`` query.
        k: ``[B, 1, Hk, Dk]`` key.
        v: ``[B, 1, Hv, Dv]`` value.
        A_log: ``[Hv]`` learned log decay.
        a: ``[B, 1, Hv]`` input-dependent decay.
        dt_bias: ``[Hv]`` decay bias.
        b: ``[B, 1, Hv]`` update-gate logits; the sigmoid is applied here.
        initial_state: K-last pool ``[rows, Hv, Dv, Dk]``.
        initial_state_indices: ``[B]`` row holding position n-1; ``-1`` marks
            graph padding.
        scale: Softmax scale; ``None`` uses ``Dk ** -0.5``.
        output_state_indices: ``[B]`` row receiving position n, or ``None``.
        use_qk_l2norm: Whether to L2-normalize q and k.

    Returns:
        ``[B, 1, Hv, Dv]`` layer output.
    """
    batch_size = q.shape[0]
    return _recurrent_window(
        q=q,
        k=k,
        v=v,
        decay=gdn_gating(A_log, a, dt_bias),
        beta=torch.sigmoid(b.float()).to(_ACTIVATION_DTYPE),
        initial_state=initial_state,
        read_rows=initial_state_indices,
        write_rows=output_state_indices,
        write_grid=None,
        write_pool=None,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
    ).reshape(batch_size, 1, v.shape[-2], v.shape[-1])


def gdn_decode_mtp(
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
    scale: float | None = None,
    disable_state_update: bool = True,
    use_qk_l2norm: bool = True,
    intermediate_states_buffer: torch.Tensor | None = None,
    output_state_indices: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    tree_ancestors: torch.Tensor | None = None,
    replay_payload: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-token GDN verify window, one state row per verified position.

    Token ``t`` of a request sees the keys of tokens ``0..t`` on top of the
    committed state, so the recurrence walks the whole window on one row while
    each position's resulting state lands on the row named for it.

    Args:
        q: ``[B, T, Hk, Dk]`` query.
        k: ``[B, T, Hk, Dk]`` key.
        v: ``[B, T, Hv, Dv]`` value.
        A_log: ``[Hv]`` learned log decay.
        a: ``[B, T, Hv]`` input-dependent decay.
        dt_bias: ``[Hv]`` decay bias.
        b: ``[B, T, Hv]`` update-gate logits.
        initial_state: K-last pool ``[rows, Hv, Dv, Dk]`` holding the
            committed snapshots.
        initial_state_indices: ``[B]`` committed row per request.
        scale: Softmax scale; ``None`` uses ``Dk ** -0.5``.
        disable_state_update: Leave the committed state untouched and stage the
            window instead.
        use_qk_l2norm: Whether to L2-normalize q and k.
        intermediate_states_buffer: Rejected; per-position states go to
            ``output_state``.
        output_state_indices: ``[B, T]`` row receiving each position's state,
            or ``None`` to keep the window on a single row per request.
        output_state: Pool the ``output_state_indices`` rows belong to;
            ``None`` writes into ``initial_state``.
        tree_ancestors: Rejected; draft trees need per-node parent walks.
        replay_payload: Rejected; the Ascend verify stages its own states.

    Returns:
        ``[B, T, Hv, Dv]`` verify output.
    """
    if intermediate_states_buffer is not None:
        raise NotImplementedError(
            "Ascend GDN verify does not support intermediate_states_buffer"
        )
    if tree_ancestors is not None:
        raise NotImplementedError("Ascend GDN verify does not support tree_ancestors")
    if replay_payload is not None:
        raise NotImplementedError("Ascend GDN verify does not support replay_payload")
    steps = q.shape[1]
    if steps < 1:
        raise ValueError("Ascend GDN verify needs a non-empty window")

    return _recurrent_window(
        q=q,
        k=k,
        v=v,
        decay=gdn_gating(A_log, a, dt_bias),
        beta=torch.sigmoid(b.float()).to(_ACTIVATION_DTYPE),
        initial_state=initial_state,
        read_rows=initial_state_indices,
        write_rows=None if disable_state_update else initial_state_indices,
        write_grid=output_state_indices,
        write_pool=output_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
    )


def _recurrent_window(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    read_rows: torch.Tensor,
    write_rows: torch.Tensor | None,
    write_grid: torch.Tensor | None,
    write_pool: torch.Tensor | None,
    scale: float | None,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    """Run the native recurrent op over a window.

    The op reads the state from ``ssm_state_indices[0]`` and writes token ``t``'s
    resulting state to ``ssm_state_indices[t]``, so one call serves three
    layouts:

    * one row per request (``write_grid is None``, ``write_rows`` given): the
      window evolves on the destination row, which is seeded from the read row
      when the two differ;
    * one row per token (``write_grid`` given): each position's state lands on
      its own row, and the window is seeded into the grid's first row;
    * the committed row in place (``write_grid is None``, ``write_rows`` is
      None): the window's final state stays where the read found it.

    A grid whose rows live in ``write_pool`` runs entirely there: the committed
    state is copied onto the grid's first row first, so the committed pool is
    left untouched. Padding rows (``-1``) are redirected to a spare row because
    the op reads the index directly and faults on a negative one.

    Args:
        q: ``[B, T, Hk, Dk]``.
        k: ``[B, T, Hk, Dk]``.
        v: ``[B, T, Hv, Dv]``.
        decay: ``[B, T, Hv]`` fp32 log-space decay.
        beta: ``[B, T, Hv]`` activated update coefficient.
        initial_state: K-last pool holding the committed snapshots.
        read_rows: ``[B]`` committed-state row per request.
        write_rows: ``[B]`` destination row per request, or ``None``.
        write_grid: ``[B, T]`` destination rows, one per verified position, or
            ``None``.
        write_pool: Pool the ``write_grid`` rows belong to, or ``None`` to use
            ``initial_state``.
        scale: Softmax scale.
        use_qk_l2norm: Whether to L2-normalize q and k.

    Returns:
        ``[B, T, Hv, Dv]`` output.
    """
    _require_bf16(q, "query")
    _require_state_dtype(initial_state)
    if write_pool is not None:
        _require_state_dtype(write_pool)
    _check_heads(k[0], v[0])

    batch_size, steps = q.shape[0], q.shape[1]
    total = batch_size * steps
    device = initial_state.device

    read_rows = read_rows.to(torch.int64)
    # A negative index is graph padding: it reads the zero state and writes
    # nothing. The op reads and writes the row it names, so padding rows are
    # pointed at the reserved null page (row 0) and that row is restored after
    # the launch, leaving every live row untouched.
    padding = read_rows < 0

    if write_grid is not None:
        state = initial_state if write_pool is None else write_pool
        grid = write_grid.to(torch.int64).reshape(batch_size, steps)
        grid = grid.masked_fill(padding[:, None], 0)
        # The op reads the window's starting state from the first named row, so
        # the committed snapshot has to sit there before the launch.
        _seed_rows(state, grid[:, 0], initial_state, read_rows.masked_fill(padding, 0))
        rows = grid.reshape(-1)
    elif write_rows is not None:
        state = initial_state
        targets = write_rows.to(torch.int64).reshape(batch_size)
        targets = targets.masked_fill(targets < 0, 0)
        _seed_rows(state, targets, state, read_rows.masked_fill(padding, 0))
        rows = targets.repeat_interleave(steps)
    else:
        # In place: a padding row writes the null page, which the restore below
        # puts back, so no live row moves. Staying on the device keeps a
        # captured call formable; the runtime always names an output row, so
        # this branch is outside the graph anyway.
        state = initial_state
        rows = read_rows.masked_fill(padding, 0).repeat_interleave(steps)

    if state.shape[0] < total:
        raise ValueError(
            "Ascend GDN needs at least one state row per token "
            f"(BlockNum >= T): {state.shape[0]} rows for {total} tokens"
        )

    # Padding rows wrote the null page; put it back so it stays the zero state.
    # The snapshot is unconditional because deciding on the host would need a
    # device sync, which an ACL graph capture forbids; one row copy per call is
    # far cheaper than the launch itself.
    null_page = state[0].clone()

    query = q.reshape(total, q.shape[2], q.shape[3])
    key = k.reshape(total, k.shape[2], k.shape[3])
    if use_qk_l2norm:
        query = _normalize(query)
        key = _normalize(key)

    # The op requires this tensor; a request's window is `steps` long, and the
    # op carries the recurrence over that window through the named rows.
    seq_lengths = torch.full((batch_size,), steps, dtype=torch.int32, device=device)
    out = torch_npu.npu_recurrent_gated_delta_rule(
        query.contiguous(),
        key.contiguous(),
        v.reshape(total, v.shape[2], v.shape[3]).contiguous(),
        state,
        beta=beta.reshape(total, -1).contiguous(),
        scale=query.shape[-1] ** -0.5 if scale is None else scale,
        actual_seq_lengths=seq_lengths,
        ssm_state_indices=rows.to(torch.int32),
        num_accepted_tokens=None,
        g=decay.reshape(total, -1).contiguous(),
        gk=None,
    )
    if null_page is not None:
        state[0] = null_page
    return out.reshape(batch_size, steps, v.shape[2], v.shape[3])


def _seed_rows(
    pool: torch.Tensor,
    target: torch.Tensor,
    source_pool: torch.Tensor,
    source: torch.Tensor,
) -> None:
    """Copy ``source_pool[source]`` onto ``pool[target]`` without a host sync.

    ``index_copy_`` is the direct spelling and stays stream-ordered on Ascend,
    which a decode graph can capture (unlike ``index_fill_``).
    """
    pool.index_copy_(0, target, source_pool.index_select(0, source))


def gdn_gating(
    A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor
) -> torch.Tensor:
    """Build the log-space decay ``g = -exp(A_log) * softplus(a + dt_bias)``.

    Ascend has no fused gating op, so the elementwise math is composed here.
    The softplus is clamped at 20 so ``exp`` stays finite, matching the
    portable fused kernel's threshold.

    Args:
        A_log: ``[Hv]`` learned log decay.
        a: ``[B, T, Hv]`` input-dependent decay.
        dt_bias: ``[Hv]`` decay bias.

    Returns:
        ``g`` with ``a``'s shape, in float32.
    """
    x = a.float() + dt_bias.float()
    softplus = torch.where(
        x > _SOFTPLUS_THRESHOLD,
        x,
        torch.log1p(torch.exp(torch.clamp(x, max=_SOFTPLUS_THRESHOLD))),
    )
    return -torch.exp(A_log.float()) * softplus


__all__ = [
    "gdn_chunk_prefill",
    "gdn_decode_mtp",
    "gdn_decode_step",
    "gdn_gating",
]
