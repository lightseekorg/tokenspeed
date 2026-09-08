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

"""FlashInfer FP32-state MTP adapter without per-call buffer initialization."""

from __future__ import annotations

import torch
from flashinfer.gdn_kernels.gdn_decode_mtp import (
    get_tile_v_mtp,
    get_vec_size_mtp,
    run_mtp_decode,
)


def gated_delta_rule_mtp(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    output: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: float | None,
    intermediate_states_buffer: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
    disable_state_update: bool,
    use_qk_l2norm: bool,
) -> None:
    """Run FlashInfer MTP into a caller-owned output without zero fills.

    Args:
        q: Query, ``[B, T, H, K]``, float16 or bfloat16.
        k: Key with the same shape and dtype as q.
        v: Value, ``[B, T, HV, V]``, same dtype as q.
        initial_state: FP32 K-last state pool, ``[pool_size, HV, V, K]``;
            page-strided pools are supported without copying.
        initial_state_indices: Int32/int64 read rows, ``[B]``. Negative rows
            skip all state and output accesses.
        output: BF16 ``[B, T, HV, V]`` output. Live rows are fully overwritten;
            padding retains its previous contents. An empty allocation is
            sufficient when the caller ignores padded outputs.
        A_log: FP32 log decay, ``[HV]``.
        a: Input-dependent decay, ``[B, T, HV]``, same dtype as q.
        dt_bias: FP32 decay bias, ``[HV]``.
        b: Update gate, ``[B, T, HV]``, same dtype as q.
        scale: Query scale, or None for ``K**-0.5``.
        intermediate_states_buffer: Optional contiguous FP32 batch-indexed
            cache, ``[>=B, >=T, HV, V, K]``. Every live step is overwritten.
        ssm_state_indices: Optional int32 ``[B, T]`` per-token pool write rows;
            mutually exclusive with the intermediate cache and requires
            state updates, T >= 2 and a contiguous pool (verify scratch).
        disable_state_update: Suppress pool writes. Otherwise the final state
            overwrites the read row unless per-token destinations are supplied.
        use_qk_l2norm: Normalize q/k inside the kernel.

    Returns:
        None. The output and enabled state destinations are written in place.
    """
    batch, steps, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    pool_size = initial_state.shape[0]
    tile_v = get_tile_v_mtp(batch, steps, num_v_heads=value_heads, v_dim=value_dim)
    assert initial_state.shape == (pool_size, value_heads, value_dim, key_dim)
    assert initial_state.dtype == torch.float32
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert A_log.dtype == dt_bias.dtype == torch.float32
    assert initial_state_indices.shape == (batch,)
    assert initial_state_indices.dtype in (torch.int32, torch.int64)
    assert output.shape == (batch, steps, value_heads, value_dim)
    assert output.dtype == torch.bfloat16
    assert key_dim >= 128 and value_dim >= 128 and value_dim % tile_v == 0

    use_pool_indexing = not initial_state.is_contiguous()
    state_source = (
        initial_state
        if use_pool_indexing
        else initial_state.view(pool_size * value_heads, value_dim, key_dim)
    )
    cache_intermediate_states = intermediate_states_buffer is not None
    cache_steps = steps
    if cache_intermediate_states:
        cache_batch, cache_steps = intermediate_states_buffer.shape[:2]
        assert cache_batch >= batch and cache_steps >= steps
        assert intermediate_states_buffer.shape[2:] == (
            value_heads,
            value_dim,
            key_dim,
        )
        assert intermediate_states_buffer.dtype == torch.float32
        assert intermediate_states_buffer.is_contiguous()
        intermediate_states = intermediate_states_buffer.view(
            cache_batch * cache_steps * value_heads, value_dim, key_dim
        )
    else:
        # All accesses are disabled by cache_intermediate_states=False. A
        # typed view suffices; no placeholder allocation or fill is needed.
        intermediate_states = initial_state[0]

    if ssm_state_indices is not None:
        assert not cache_intermediate_states and not disable_state_update
        assert steps >= 2
        assert ssm_state_indices.shape == (batch, steps)
        assert ssm_state_indices.dtype == torch.int32
        assert ssm_state_indices.device == q.device

    run_mtp_decode(
        h0_source=state_source,
        intermediate_states=intermediate_states,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        output=output,
        initial_state_indices=initial_state_indices,
        B=batch,
        T=steps,
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        pool_size=pool_size,
        cache_steps=cache_steps,
        tile_v=tile_v,
        vec_size=get_vec_size_mtp(batch, steps),
        scale=key_dim**-0.5 if scale is None else scale,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=disable_state_update,
        cache_intermediate_states=cache_intermediate_states,
        ssm_state_indices=ssm_state_indices,
        output_state_indices=None,
        use_pool_indexing=use_pool_indexing,
    )
