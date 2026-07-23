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

from collections.abc import Mapping
from typing import Any

import torch
from tokenspeed_kernel.operation import OperationSchema
from tokenspeed_kernel.signature import FormatSignature

__all__ = ["MHA_PREFILL", "mha_prefill_reference"]


_BOOL_TRAITS = frozenset(
    {"sliding_window", "support_logit_cap", "support_sinks", "return_lse"}
)


def _validate_signatures(signatures: frozenset[FormatSignature]) -> None:
    if not signatures:
        raise ValueError("attention.mha_prefill requires a format signature")
    for signature in signatures:
        if not isinstance(signature, FormatSignature):
            raise TypeError("MHA prefill signatures must be FormatSignature values")
        if {name for name, _ in signature.roles} != {"q", "k", "v"}:
            raise ValueError("MHA prefill signatures require roles 'q', 'k', and 'v'")
        formats = [signature.format_for(role) for role in ("q", "k", "v")]
        if any(
            tensor_format is None
            or tensor_format.format != "dense"
            or tensor_format.scale is not None
            for tensor_format in formats
        ):
            raise ValueError("MHA prefill roles must use unscaled dense formats")


def _validate_traits(traits: Mapping[str, frozenset[Any]]) -> None:
    unknown = set(traits) - ({"head_dim"} | _BOOL_TRAITS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown attention.mha_prefill trait(s): {names}")
    for name, values in traits.items():
        if not isinstance(values, frozenset) or not values:
            raise TypeError(f"MHA prefill trait {name!r} must be a non-empty frozenset")
        if name == "head_dim":
            if any(type(value) is not int or value <= 0 for value in values):
                raise ValueError(
                    "MHA prefill head_dim values must be positive integers"
                )
        elif any(type(value) is not bool for value in values):
            raise TypeError(f"MHA prefill trait {name!r} values must be bool")


def _sequence_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(groups, dim=1)
    v = v.float().repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->qhk", q.float(), k) * softmax_scale
    if logit_cap > 0:
        scores = logit_cap * torch.tanh(scores / logit_cap)

    positions = torch.arange(q.shape[0], device=q.device)
    visible = positions[None, :] <= positions[:, None]
    if window_left >= 0:
        visible &= positions[None, :] >= positions[:, None] - window_left
    scores = scores.masked_fill(~visible[:, None, :], float("-inf"))

    normalizer = scores
    if sinks is not None:
        sink_scores = sinks.float()[None, :, None].expand(q.shape[0], -1, -1)
        normalizer = torch.cat((scores, sink_scores), dim=-1)
    lse = torch.logsumexp(normalizer, dim=-1)
    weights = torch.exp(scores - lse[..., None])
    return torch.einsum("qhk,khd->qhd", weights, v), lse


def mha_prefill_reference(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Evaluate packed causal grouped-query MHA prefill in PyTorch.

    The batch is packed along the first tensor dimension. Query, key, and value
    tensors use identical sequence boundaries, and attention is computed
    independently within each sequence. Inputs are not modified.

    Args:
        q: Query tensor shaped
            ``[total_tokens, num_query_heads, head_dim]``.
        k: Key tensor shaped ``[total_tokens, num_kv_heads, head_dim]``.
        v: Value tensor with the same shape as ``k``. The tensors must be on
            the same device, and ``num_query_heads`` must be a positive integer
            multiple of ``num_kv_heads`` for grouped-query attention.
        cu_seqlens: One-dimensional cumulative sequence offsets shaped
            ``[batch_size + 1]``. Offsets must begin at zero, be
            nondecreasing, and end at ``total_tokens``.
        cu_seqlens_cpu: Host-side ``list[int]`` containing exactly the same
            offsets as ``cu_seqlens`` and at least two entries. This duplicate
            representation is part of the kernel ABI for backends that need
            host launch metadata.
        max_seqlen: Upper bound on every packed sequence length. It must be at
            least the largest difference between adjacent cumulative offsets.
        window_left: Number of preceding key positions visible to each query.
            A negative value enables full causal attention; otherwise query
            position ``i`` attends positions
            ``[max(0, i - window_left), i]`` within its sequence.
        logit_cap: Nonnegative soft cap for scaled query-key logits. A positive
            value ``c`` transforms each logit ``x`` to
            ``c * tanh(x / c)``; zero disables the cap.
        sinks: Optional uncapped, per-query-head sink logits shaped
            ``[num_query_heads]`` on the same device as ``q``. Each sink
            contributes to the softmax denominator but has a zero value, so it
            can absorb attention weight without contributing to the output.
        return_lse: Whether to return per-token, per-query-head log-sum-exp
            values in addition to the attention output.
        softmax_scale: Scale applied to query-key logits before the optional
            cap and softmax. ``None`` uses ``1 / sqrt(head_dim)``.

    Returns:
        The attention output has the same shape and dtype as ``q``, except
        ``torch.float8_e4m3fn`` and ``torch.float8_e5m2`` inputs produce BF16
        output. If ``return_lse`` is false, the output tensor is returned
        directly. Otherwise, returns ``(output, lse)``, where ``lse`` is a
        float32 tensor shaped ``[total_tokens, num_query_heads]`` using natural
        logarithms.

    Raises:
        ValueError: If Q/K/V shapes are incompatible or the packed sequence
            metadata is inconsistent.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("MHA prefill q, k, and v must have rank 3")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("MHA prefill q, k, and v have incompatible shapes")
    if k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError("MHA prefill query heads must be a multiple of KV heads")

    boundaries = cu_seqlens.detach().cpu().tolist()
    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if (
        boundaries != cu_seqlens_cpu
        or not lengths
        or boundaries[0] != 0
        or any(length < 0 for length in lengths)
        or boundaries[-1] != q.shape[0]
        or max(lengths) > max_seqlen
    ):
        raise ValueError("MHA prefill has inconsistent packed sequence metadata")

    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for start, end in zip(cu_seqlens_cpu, cu_seqlens_cpu[1:]):
        output, lse = _sequence_reference(
            q[start:end],
            k[start:end],
            v[start:end],
            window_left=window_left,
            logit_cap=float(logit_cap),
            sinks=sinks,
            softmax_scale=scale,
        )
        outputs.append(output)
        lses.append(lse)

    output_dtype = (
        torch.bfloat16
        if q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        else q.dtype
    )
    output = torch.cat(outputs).to(output_dtype)
    lse = torch.cat(lses)
    return (output, lse) if return_lse else output


MHA_PREFILL = OperationSchema(
    family="attention",
    mode="mha_prefill",
    reference=mha_prefill_reference,
    validate_signatures=_validate_signatures,
    validate_traits=_validate_traits,
).publish()
