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

"""FlashInfer GDN adapters with optional PDL and uninitialized MTP buffers."""

from __future__ import annotations

import functools

import torch
from flashinfer.gdn_kernels.gdn_decode_mtp import (
    get_tile_v_mtp,
    get_vec_size_mtp,
)

# Preserve independent availability checks for optional FlashInfer entry points.
try:
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as _original_prefill
except ImportError:
    _original_prefill = None
try:
    from flashinfer.gdn_decode import (
        gated_delta_rule_decode_pretranspose as _original_decode,
    )
except ImportError:
    _original_decode = None
try:
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule_mtp as _original_bf16_mtp,
    )
except ImportError:
    _original_bf16_mtp = None

HAS_PREFILL = _original_prefill is not None
HAS_DECODE = _original_decode is not None
HAS_BF16_MTP = _original_bf16_mtp is not None


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
    enable_pdl: bool,
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
        enable_pdl: Enable dependent launch and input synchronization; selected
            before capture and retained by CUDA Graph replay.

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

    _mtp_runner(enable_pdl)(
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


@functools.cache
def _mtp_runner(enable_pdl: bool):
    from flashinfer.gdn_kernels import gdn_decode_mtp

    if not enable_pdl:
        return gdn_decode_mtp.run_mtp_decode

    from tokenspeed_kernel.thirdparty.flashinfer._pdl import _adapt_module

    return _adapt_module(
        gdn_decode_mtp,
        kernels=("gdn_verify_kernel_mtp", "gdn_verify_kernel_mtp_inline"),
        launchers=("run_gdn_verify_kernel_mtp", "run_gdn_verify_kernel_mtp_inline"),
        entrypoints=("run_mtp_decode",),
        caches=("_get_compiled_mtp_kernel", "_get_compiled_mtp_kernel_inline"),
        overrides={},
    )["run_mtp_decode"]


@functools.cache
def _bf16_runners(enable_pdl: bool):
    from flashinfer.gdn_kernels import gdn_decode_bf16_state

    if not enable_pdl:
        return vars(gdn_decode_bf16_state)

    from tokenspeed_kernel.thirdparty.flashinfer._pdl import _adapt_module

    return _adapt_module(
        gdn_decode_bf16_state,
        kernels=(
            "gdn_decode_bf16state_mtp_ilp4_kernel",
            "gdn_wide_vec_kernel",
            "gdn_wide_vec_kernel_t1",
        ),
        launchers=(
            "run_gdn_decode_bf16state_mtp_ilp4",
            "_run_wide_vec",
            "_run_wide_vec_t1",
        ),
        entrypoints=(
            "gated_delta_rule",
            "gated_delta_rule_mtp",
            "gated_delta_rule_mtp_wide_vec",
            "gated_delta_rule_t1_wide_vec",
        ),
        caches=("_compiled_kernels_mtp", "_compiled_kernels_wide_vec"),
        overrides={},
    )


@functools.cache
def _decode_runner(enable_pdl: bool):
    if not enable_pdl:
        return _original_decode

    from flashinfer import gdn_decode
    from flashinfer.gdn_kernels import gdn_decode_pretranspose
    from tokenspeed_kernel.thirdparty.flashinfer._pdl import (
        _adapt_module,
        _clone_function,
    )

    pretranspose = _adapt_module(
        gdn_decode_pretranspose,
        kernels=(
            "gdn_decode_kernel_small_batch_pretranspose",
            "gdn_decode_kernel_big_batch_pretranspose",
        ),
        launchers=(
            "run_gdn_decode_kernel_small_batch_pretranspose",
            "run_gdn_decode_kernel_big_batch_pretranspose",
        ),
        entrypoints=("run_pretranspose_decode",),
        caches=("_get_compiled_decode_kernel",),
        overrides={},
    )
    overrides = {"run_pretranspose_decode": pretranspose["run_pretranspose_decode"]}
    if gdn_decode._GDN_DECODE_BF16_STATE_AVAILABLE:
        bf16 = _bf16_runners(enable_pdl)
        overrides.update(
            _gated_delta_rule_bf16_state=bf16["gated_delta_rule"],
            _gated_delta_rule_bf16_state_mtp=bf16["gated_delta_rule_mtp"],
        )
    return _clone_function(_original_decode, {**vars(gdn_decode), **overrides})


@functools.cache
def _prefill_runner(enable_pdl: bool):
    if not enable_pdl:
        return _original_prefill

    from flashinfer import gdn_prefill
    from flashinfer.gdn_kernels.blackwell import gdn_prefill as sm100
    from tokenspeed_kernel.thirdparty.flashinfer._pdl import (
        _adapt_module,
        _clone_function,
        _PdlKernel,
    )

    class PdlGatedDeltaNetChunkedKernel(sm100.GatedDeltaNetChunkedKernel):
        kernel = _PdlKernel(sm100.GatedDeltaNetChunkedKernel.kernel)

    adapted = _adapt_module(
        sm100,
        kernels=(),
        launchers=(),
        entrypoints=("chunk_gated_delta_rule_sm100",),
        caches=("_get_compiled_cache",),
        overrides={"GatedDeltaNetChunkedKernel": PdlGatedDeltaNetChunkedKernel},
    )
    return _clone_function(
        _original_prefill,
        {
            **vars(gdn_prefill),
            "chunk_gated_delta_rule_sm100": adapted["chunk_gated_delta_rule_sm100"],
        },
    )


def gated_delta_rule_decode_pretranspose(*, enable_pdl: bool, **kwargs):
    """Run T=1 decode with FlashInfer's keyword arguments.

    ``enable_pdl`` selects dependent launch and synchronization. ``kwargs``
    contains the upstream decode inputs, state pool/indices and output options.
    Returns the upstream ``(output, state)`` pair with K-last state layout.
    """
    return _decode_runner(enable_pdl)(**kwargs)


def gated_delta_rule_bf16_mtp(*, enable_pdl: bool, **kwargs):
    """Run BF16-state MTP with FlashInfer's keyword arguments.

    ``enable_pdl`` selects an isolated PDL compilation/buffer cache; ``kwargs``
    contains the upstream MTP inputs and state destinations. Returns its
    ``[B, T, HV, V]`` output, updating enabled state destinations in place.
    """
    return _bf16_runners(enable_pdl)["gated_delta_rule_mtp"](**kwargs)


def chunk_gated_delta_rule(*args, enable_pdl: bool, **kwargs):
    """Run SM100 prefill with FlashInfer's positional and keyword arguments.

    ``args`` contains Q/K/V; ``kwargs`` contains upstream gates, state,
    sequence lengths and output options. The caller selects ``use_cp=False``.
    ``enable_pdl`` controls the persistent kernel. Returns the upstream output
    and optional final state.
    """
    return _prefill_runner(enable_pdl)(*args, **kwargs)
