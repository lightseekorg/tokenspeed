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
#
# Attention-Residual mixing op: RMSNorm + per-candidate softmax score + weighted
# sum over a set of residual-stream snapshots (the Kimi-K3 AttnRes block-mix).
# Dispatches to specialized GPU kernels, falling back to a portable torch path
# on other hardware or for shapes outside their supported ranges.
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["attn_res_fwd", "attn_res_fwd_available"]

# The Blackwell launcher instantiates aligned hidden sizes in [4096, 8192].
# Its block-residual source stride is passed to CUDA as a signed 32-bit int.
# AMD Gluon currently specializes Kimi K3's H=7168 fused-output-norm path.
_MAX_BLACKWELL_TOKENS = ((1 << 31) - 1) // 7168
_MAX_AMD_GLUON_TOKENS = 65536
_MAX_N = 12


def _select_attn_res_kernel(
    layer_residual,
    block_residual,
    res_weight,
    rms_weight,
    *,
    out_norm_weight,
    output_eps,
    eps,
    delta,
    num_valid_blocks,
    block_write_idx,
):
    tokens, hidden_size = layer_residual.shape
    valid_blocks = (
        block_residual.shape[0] if num_valid_blocks is None else int(num_valid_blocks)
    )
    if not 0 <= valid_blocks <= block_residual.shape[0]:
        raise ValueError("num_valid_blocks is outside block_residual")
    if block_write_idx != -1 and (
        block_write_idx != valid_blocks or block_write_idx >= block_residual.shape[0]
    ):
        raise ValueError("block_write_idx must append within block_residual")
    num_sources = valid_blocks + 1
    input_tensors = [
        layer_residual,
        block_residual,
        res_weight,
        rms_weight,
    ]
    if out_norm_weight is not None:
        input_tensors.append(out_norm_weight)
    if delta is not None:
        input_tensors.append(delta)
    inputs_on_same_gpu = layer_residual.is_cuda and all(
        tensor.is_cuda and tensor.device == layer_residual.device
        for tensor in input_tensors
    )
    hidden_dimension_contiguous = (
        layer_residual.stride(-1) == 1
        and block_residual.ndim == 3
        and block_residual.stride(-1) == 1
        and res_weight.stride(-1) == 1
        and rms_weight.stride(-1) == 1
        and (out_norm_weight is None or out_norm_weight.stride(-1) == 1)
    )
    delta_compatible = delta is None or (
        delta.shape == layer_residual.shape
        and delta.dtype == layer_residual.dtype
        and delta.device == layer_residual.device
        and delta.stride(-1) == 1
    )
    platform = Platform.get()
    if platform.is_cdna4 or platform.is_cdna5:
        eligible = (
            hidden_size == 7168
            and out_norm_weight is not None
            and 1 <= tokens <= _MAX_AMD_GLUON_TOKENS
        )
    else:
        eligible = (
            4096 <= hidden_size <= 8192
            and hidden_size % 1024 == 0
            and 1 <= tokens <= _MAX_BLACKWELL_TOKENS
        )
    eligible = eligible and 1 <= num_sources <= _MAX_N
    signature = format_signature(
        layer_residual=dense_tensor_format(layer_residual.dtype),
        block_residual=dense_tensor_format(block_residual.dtype),
    )
    kernel = select_kernel(
        "attn_res",
        "fwd",
        signature,
        traits={
            "fused_output_norm": out_norm_weight is not None,
            "has_delta": delta is not None,
            "hidden_dimension_contiguous": hidden_dimension_contiguous,
            "inputs_on_same_gpu": inputs_on_same_gpu,
            "large_prefill": tokens > 32,
            "hidden_size": hidden_size,
            "delta_compatible": delta_compatible,
            "partial_block_storage": valid_blocks != block_residual.shape[0],
            "separate_output_eps": out_norm_weight is not None and output_eps != eps,
            "writes_block": block_write_idx >= 0,
        },
        solution=None if eligible else "torch",
    )
    return kernel, valid_blocks


def attn_res_fwd(
    layer_residual,
    block_residual,
    res_weight,
    rms_weight,
    eps=1e-6,
    out_norm_weight=None,
    out_norm_eps=None,
    *,
    delta=None,
    num_valid_blocks=None,
    block_write_idx=-1,
):
    """Fused Attention-Residual forward.

    Candidates are ``block_residual[0..K-1]`` followed by ``layer_residual``
    (N = K + 1). Computes ``softmax_n(<RMSNorm(v_n), rms_weight * res_weight>)``
    over candidates, then the weighted sum of the raw candidates.

    Args:
        layer_residual: bf16 ``[T, H]`` current residual stream.
        block_residual: bf16 ``[K, T, H]`` periodic-snapshot storage.
        res_weight: bf16 ``[H]`` scorer projection weight.
        rms_weight: bf16 ``[H]`` RMSNorm weight.
        eps: RMSNorm epsilon.
        out_norm_weight: optional bf16 ``[H]``; when given, the following
            RMSNorm is fused into the epilogue and the return value is the
            normed mix.
        out_norm_eps: Optional output RMSNorm epsilon. Defaults to ``eps``.
        delta: Optional bf16 ``[T, H]`` update added to ``layer_residual``.
            The BF16-rounded sum is written back to ``layer_residual`` before
            it participates in the AttnRes mix.
        num_valid_blocks: Number of leading snapshots to include. Defaults to
            all rows in ``block_residual``.
        block_write_idx: Optional snapshot row receiving the updated layer
            residual. It must immediately follow the valid snapshots.

    Returns:
        bf16 ``[T, H]`` mixed residual (normed when ``out_norm_weight`` given).
    """
    output_eps = (
        eps if out_norm_weight is None or out_norm_eps is None else out_norm_eps
    )
    kernel, valid_blocks = _select_attn_res_kernel(
        layer_residual,
        block_residual,
        res_weight,
        rms_weight,
        out_norm_weight=out_norm_weight,
        output_eps=output_eps,
        eps=eps,
        delta=delta,
        num_valid_blocks=num_valid_blocks,
        block_write_idx=block_write_idx,
    )
    return kernel(
        layer_residual=layer_residual,
        block_residual=block_residual,
        res_weight=res_weight,
        rms_weight=rms_weight,
        eps=eps,
        out_norm_weight=out_norm_weight,
        out_norm_eps=output_eps,
        delta=delta,
        num_valid_blocks=valid_blocks,
        block_write_idx=block_write_idx,
    )


def attn_res_fwd_available(
    layer_residual,
    block_residual,
    res_weight,
    rms_weight,
    eps=1e-6,
    out_norm_weight=None,
    out_norm_eps=None,
    *,
    delta=None,
    num_valid_blocks=None,
    block_write_idx=-1,
):
    """Return whether a specialized kernel supports the exact AttnRes call.

    Args:
        layer_residual: Current residual stream shaped ``[tokens, hidden_size]``.
        block_residual: Periodic snapshots shaped ``[blocks, tokens, hidden_size]``.
        res_weight: AttnRes projection weight shaped ``[hidden_size]``.
        rms_weight: AttnRes RMSNorm weight shaped ``[hidden_size]``.
        eps: AttnRes RMSNorm epsilon.
        out_norm_weight: Optional following RMSNorm weight.
        out_norm_eps: Optional following RMSNorm epsilon. Defaults to ``eps``.
        delta: Optional update added in place to ``layer_residual``.
        num_valid_blocks: Number of leading snapshots included in the mix.
        block_write_idx: Optional row receiving the updated residual; it must
            immediately follow the valid snapshots.

    Returns:
        ``True`` when registry dispatch can run a specialized implementation.
    """
    output_eps = (
        eps if out_norm_weight is None or out_norm_eps is None else out_norm_eps
    )
    try:
        kernel, _ = _select_attn_res_kernel(
            layer_residual,
            block_residual,
            res_weight,
            rms_weight,
            out_norm_weight=out_norm_weight,
            output_eps=output_eps,
            eps=eps,
            delta=delta,
            num_valid_blocks=num_valid_blocks,
            block_write_idx=block_write_idx,
        )
    except (NoKernelFoundError, ValueError):
        return False
    from tokenspeed_kernel.ops.attn_res.torch import torch_attn_res_fwd

    return kernel.impl is not torch_attn_res_fwd


import tokenspeed_kernel.ops.attn_res.cuda  # noqa: E402,F401

# Registration side effects (must run so select_kernel can find the backends).
import tokenspeed_kernel.ops.attn_res.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.attn_res.torch  # noqa: E402,F401
