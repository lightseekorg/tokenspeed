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

"""Ascend causal depthwise conv1d over the runtime's channel-major conv pool.

A pool row holds its request's last ``width - 1`` tokens, oldest first, so
token ``t`` of a ``width``-tap causal convolution reads
``X[t - (width - 1) .. t]``. ``npu_fused_causal_conv1d`` keeps exactly that
window, which makes the rolling pool the natural state: each row is the
``W = width - 1`` tokens a decode step needs. Two layout transposes bridge the
difference from the portable kernels:

* the runtime passes ``x`` as ``[channels, tokens]``; the native op takes
  ``[tokens, channels]``.
* the runtime pool is ``[..., channels, window]``; the native op is
  ``[..., window, channels]`` and updates it in place.

Four native limitations shape the code below:

* ``width`` is fixed at 3, while Qwen3.5/3.6 use 4 taps. The native call
  keeps the newest three columns and each older column ``c`` becomes a delayed
  tap on ``X[t - (width - 1) + c]``: window slot ``c`` for a decode, and a
  shifted row of the extended ``[window | chunk]`` buffer for a prefill.
* ``activation`` is accepted but never applied, and ``bias`` is unsupported;
  both are computed here.
* ``num_accepted_tokens`` changes neither output nor state, so a verify that
  must stop at a shorter accepted prefix has to run that prefix instead.
* every token names one pool row through ``cache_indices``, so a per-token
  window table (a chain verify's scratch rows, or a draft tree) cannot use the
  op and runs through the gathered PyTorch form instead.
"""

from __future__ import annotations

import torch
import torch_npu

# The native op takes bfloat16 activations and a bfloat16 window.
_ACTIVATION_DTYPE = torch.bfloat16

# The native op's kernel width is fixed.
_NATIVE_KERNEL_WIDTH = 3

# The native op requires channels % 128 == 0.
_NATIVE_CHANNEL_ALIGN = 128


def _check_layout(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.dtype != _ACTIVATION_DTYPE:
        raise NotImplementedError(
            f"Ascend causal conv1d requires bfloat16 activations, got {x.dtype}"
        )
    channels = weight.shape[0]
    if channels % _NATIVE_CHANNEL_ALIGN:
        raise NotImplementedError(
            f"Ascend causal conv1d requires a channel count that is a multiple "
            f"of {_NATIVE_CHANNEL_ALIGN}, got {channels}"
        )
    if weight.shape[1] < _NATIVE_KERNEL_WIDTH:
        raise NotImplementedError(
            "Ascend causal conv1d needs at least "
            f"{_NATIVE_KERNEL_WIDTH} taps, got {weight.shape[1]}"
        )


def _native_weight(weight: torch.Tensor) -> torch.Tensor:
    """``[channels, width]`` filters as the native ``[3, channels]`` newest taps."""
    return (
        weight[:, weight.shape[1] - _NATIVE_KERNEL_WIDTH :].transpose(0, 1).contiguous()
    )


def _native_states(conv_state: torch.Tensor) -> torch.Tensor:
    """``[..., channels, window]`` pool rows as contiguous ``[..., window, channels]``."""
    return conv_state.transpose(-1, -2).contiguous()


def _wants_silu(activation: bool | str | None) -> bool:
    return activation in (True, "silu", "swish")


def _apply_bias_and_activation(
    out: torch.Tensor,
    bias: torch.Tensor | None,
    activation: bool | str | None,
) -> torch.Tensor:
    """Apply the bias and activation the native op accepts but does not perform."""
    if bias is not None:
        out = out + bias.to(out.dtype)
    if _wants_silu(activation):
        out = out * torch.sigmoid(out.float()).to(out.dtype)
    return out


def _check_state(conv_state: torch.Tensor, weight: torch.Tensor) -> None:
    if conv_state.shape[-2] != weight.shape[0]:
        raise ValueError(
            f"conv state must be channel-major: got {tuple(conv_state.shape)} "
            f"for {weight.shape[0]} channels"
        )
    if conv_state.shape[-1] != weight.shape[1] - 1:
        raise ValueError(
            "Ascend causal conv1d keeps a window of width - 1 = "
            f"{weight.shape[1] - 1} tokens, got {conv_state.shape[-1]}"
        )


def causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    activation: bool | str | None,
    pad_slot_id: int,
    validate_data: bool = False,
    prefill_metadata: object = None,
    key_dim: int | None = None,
) -> torch.Tensor:
    """Variable-length prefill convolution over the rolling conv-state pool.

    Args:
        x: ``[channels, tokens]`` packed conv input; requests are concatenated
            left to right.
        weight: ``[channels, width]`` depthwise filters, oldest tap first.
        bias: ``[channels]`` bias, or ``None``.
        conv_states: ``[..., channels, window]`` pool; each request's window is
            shifted and updated in place.
        query_start_loc: int32 cumulative offsets, ``[batch + 1]``.
        cache_indices: int32 ``[batch]`` pool row per request, or ``None`` to
            use request order.
        has_initial_state: ``[batch]`` whether a request reads its window.
        activation: Activation applied to the output.
        pad_slot_id: Row marking a padded request.
        validate_data: Unused; the layout checks below always run.
        prefill_metadata: The portable kernel's chunk plan. The native op reads
            the request bounds itself, so this is only accepted for ABI
            compatibility.
        key_dim: The qkv fused split's key width; the packed-output entry that
            passes it is not registered, so it is only accepted for ABI
            compatibility.

    Returns:
        ``[channels, tokens]`` convolution output.
    """
    del validate_data, prefill_metadata, key_dim
    _check_layout(x, weight)
    _check_state(conv_states, weight)

    device = x.device
    batch_size = query_start_loc.numel() - 1
    x_tokens = x.transpose(0, 1).contiguous()
    native_states = _native_states(conv_states)
    native_cache = (
        torch.arange(batch_size, dtype=torch.int32, device=device)
        if cache_indices is None
        else cache_indices.to(torch.int32)
    )
    if has_initial_state is not None and not bool(has_initial_state.all().item()):
        # A request without history must not read its stale pool row. Zeroing
        # it is how the portable kernels express `has_initial_state`.
        native_states[native_cache[~has_initial_state.bool()].long()] = 0

    # Older taps read the window before the native call rolls it.
    delayed = _prefill_older_taps(
        x_tokens, native_states, native_cache, query_start_loc, weight.shape[1]
    )

    out = torch_npu.npu_fused_causal_conv1d(
        x_tokens,
        _native_weight(weight),
        native_states,
        query_start_loc=query_start_loc.to(torch.int32),
        cache_indices=native_cache,
        pad_slot_id=pad_slot_id,
        max_query_len=int((query_start_loc[1:] - query_start_loc[:-1]).max().item()),
    )
    # Only the request rows moved; writing the whole buffer back would also
    # overwrite rows this batch never named. Padded rows were skipped by the op.
    rows = native_cache[native_cache >= 0].long()
    if rows.numel():
        conv_states.index_copy_(0, rows, native_states[rows].transpose(1, 2))

    for column, source in enumerate(delayed):
        out = out + weight[:, column].to(out.dtype) * source
    out = _apply_bias_and_activation(out, bias, activation)
    return out.transpose(0, 1)


def _prefill_older_taps(
    x_tokens: torch.Tensor,
    native_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    width: int,
) -> list[torch.Tensor]:
    """``[tokens, channels]`` values for the taps older than the native span.

    Column ``c`` multiplies ``X[t - (width - 1) + c]``. For a token at
    request-local position ``p`` that is ``x[p - window + c]`` once
    ``p >= window - c``, and window slot ``p + c`` otherwise. ``native_states``
    must still hold the pre-chunk window.

    Args:
        x_tokens: ``[tokens, channels]`` packed conv input.
        native_states: ``[rows, window, channels]`` pool, before the update.
        cache_indices: ``[batch]`` pool row per request.
        query_start_loc: ``[batch + 1]`` cumulative offsets.
        width: Convolution width; columns ``0 .. width - 4`` are returned.

    Returns:
        One value tensor per older column, oldest first.
    """
    window = native_states.shape[1]
    if width <= _NATIVE_KERNEL_WIDTH:
        return []

    device = x_tokens.device
    channels = x_tokens.shape[-1]
    lengths = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int64)
    token_ids = torch.arange(x_tokens.shape[0], dtype=torch.int64, device=device)
    request = torch.repeat_interleave(
        torch.arange(lengths.numel(), dtype=torch.int64, device=device), lengths
    )
    local = token_ids - torch.repeat_interleave(
        query_start_loc[:-1].to(torch.int64), lengths
    )

    # [batch * window, channels], row `b * window + slot` is request b's slot.
    # A padded request (negative row) reads slot 0 of row 0; its output is dropped.
    windows = native_states.index_select(0, cache_indices.long().clamp(min=0)).reshape(
        -1, channels
    )
    padded = cache_indices < 0

    taps = []
    for column in range(width - _NATIVE_KERNEL_WIDTH):
        tap = torch.zeros_like(x_tokens)
        # Far enough into the request: an earlier packed token.
        lag = window - column
        far = local >= lag
        # A padded request has no history to read.
        far &= ~padded[request]
        tap[far] = x_tokens[token_ids[far] - lag]
        # Otherwise the request's own window, slot `p + column`.
        near = ~far & ~padded[request]
        row = request[near] * window + local[near] + column
        tap[near] = windows.index_select(0, row).to(x_tokens.dtype)
        taps.append(tap)
    return taps


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    cache_seqlens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    intermediate_conv_window: torch.Tensor | None = None,
    output_state_indices: torch.Tensor | None = None,
    pad_slot_id: int = -1,
    validate_data: bool = False,
    *,
    parent_indices: torch.Tensor | None,
    out: torch.Tensor | None = None,
    replay_payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Decode / verify convolution over a slot-indexed conv pool.

    Args:
        x: ``[batch, channels]`` or ``[batch, channels, steps]`` conv input,
            rooted at each request's committed window.
        conv_state: ``[..., channels, window]`` pool.
        weight: ``[channels, width]`` depthwise filters, oldest tap first.
        bias: ``[channels]`` bias, or ``None``.
        activation: Activation applied to the output.
        cache_seqlens: Rejected; the native op has no circular-buffer mode.
        conv_state_indices: int32 ``[batch]`` read row per request.
        num_accepted_tokens: Rejected; the native op ignores it.
        intermediate_conv_window: Rejected; per-token windows are written
            through ``output_state_indices`` instead.
        output_state_indices: ``[batch, 1]`` one row per request, or
            ``[batch, steps]`` one row per token; ``None`` writes back onto the
            read row.
        pad_slot_id: Row marking a padded request; negative rows are skipped.
        validate_data: Unused; the checks below always run.
        parent_indices: ``[batch, steps]`` parent per token for a draft tree,
            or ``None`` for a chain.
        out: Destination shaped like ``x``, or ``None`` to allocate one.
        replay_payload: Rejected; the Ascend verify stages its own states.

    Returns:
        Convolution output shaped like ``x``.
    """
    del validate_data
    _check_layout(x.reshape(-1, x.shape[-1]), weight)
    _check_state(conv_state, weight)
    if cache_seqlens is not None:
        raise NotImplementedError(
            "Ascend causal conv1d has no circular-buffer addressing"
        )
    if num_accepted_tokens is not None:
        raise NotImplementedError(
            "Ascend causal conv1d ignores num_accepted_tokens; run the accepted "
            "prefix at its own length instead"
        )
    if intermediate_conv_window is not None:
        raise NotImplementedError(
            "Ascend causal conv1d does not stage per-token conv windows"
        )
    if replay_payload is not None:
        raise NotImplementedError("Ascend causal conv1d does not fill a replay payload")

    batched = x.ndim == 3
    if batched:
        # [batch, channels, steps], the layout the portable kernels use.
        batch_size, channels, steps = x.shape
        flat = x.transpose(1, 2).reshape(batch_size * steps, channels)
    else:
        channels = x.shape[1]
        batch_size, steps = x.shape[0], 1
        flat = x

    read_rows = (
        torch.arange(batch_size, dtype=torch.int64, device=x.device)
        if conv_state_indices is None
        else conv_state_indices.reshape(-1).long()
    )
    # Padding rows name no pool row; they read row 0 and their output is ignored.
    live_read = read_rows.clamp(min=0)
    write_rows = _write_rows(output_state_indices, batch_size, steps, live_read)
    if parent_indices is None and steps == 1:
        result = _decode(
            flat,
            conv_state,
            weight,
            read_rows=live_read,
            write_rows=write_rows[:, 0],
        )
    else:
        result = _gathered(
            flat,
            conv_state,
            weight,
            batch_size=batch_size,
            steps=steps,
            read_rows=live_read,
            write_rows=write_rows,
            parent_indices=parent_indices,
        )

    result = _apply_bias_and_activation(result, bias, activation)
    if batched:
        result = result.reshape(batch_size, steps, channels).transpose(1, 2)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _write_rows(
    output_state_indices: torch.Tensor | None,
    batch_size: int,
    steps: int,
    read_rows: torch.Tensor,
) -> torch.Tensor:
    """``[batch, steps]`` destination row per token.

    ``None`` writes per-request rows back onto ``read_rows``. A ``[batch, 1]``
    table names one row per request and repeats it across the window.
    """
    if output_state_indices is None:
        return read_rows[:, None].expand(batch_size, steps).clone()
    rows = output_state_indices.reshape(batch_size, -1).long()
    if rows.shape[1] == 1:
        return rows.expand(batch_size, steps).clone()
    if rows.shape[1] != steps:
        raise ValueError(
            f"output_state_indices must have one or {steps} columns per request, "
            f"got {rows.shape[1]}"
        )
    return rows.contiguous()


def _decode(
    flat: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    read_rows: torch.Tensor,
    write_rows: torch.Tensor,
) -> torch.Tensor:
    """One token per request, one pool row each, through the native op."""
    device = flat.device
    width = weight.shape[1]

    # Older taps read the committed window: slot c holds X[t - window + c].
    delayed = []
    if width > _NATIVE_KERNEL_WIDTH:
        history = conv_state.index_select(0, read_rows)
        delayed = [
            history[:, :, column] for column in range(width - _NATIVE_KERNEL_WIDTH)
        ]

    native_states = _native_states(conv_state)
    live_write = write_rows.clamp(min=0)
    # The op writes back to the row it read, so a call landing on another row
    # first moves the committed window there. Copying unconditionally keeps the
    # decision on the device: a host comparison would have to sync, which an
    # ACL graph capture forbids, and copying a row onto itself is a no-op.
    native_states[live_write] = native_states[read_rows]
    out = torch_npu.npu_fused_causal_conv1d(
        flat,
        _native_weight(weight),
        native_states,
        query_start_loc=torch.arange(
            0, flat.shape[0] + 1, dtype=torch.int32, device=device
        ),
        cache_indices=live_write.to(torch.int32),
        max_query_len=1,
    )
    # Only the written rows moved, so only they go back to the pool.
    conv_state.index_copy_(0, live_write, native_states[live_write].transpose(1, 2))
    for column, source in enumerate(delayed):
        out = out + weight[:, column].to(out.dtype) * source
    return out


def _gathered(
    flat: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    batch_size: int,
    steps: int,
    read_rows: torch.Tensor,
    write_rows: torch.Tensor,
    parent_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Multi-token chains and draft trees, from per-request gathered windows.

    Extending the committed window with the chunk gives a sequence whose entry
    ``i`` is ``X[i - window]``, so a chain's tap ``c`` of token ``t`` is entry
    ``t + c``. A tree replaces that walk with a parent walk, as the portable
    kernel does: tap ``c`` uses the ``(width - 1 - c)``-th ancestor, or window
    slot ``(width - 1) - past_root`` when the walk runs past the root.
    """
    device = flat.device
    channels = flat.shape[-1]
    window = conv_state.shape[-1]
    width = weight.shape[1]

    tokens = flat.reshape(batch_size, steps, channels).float()
    history = conv_state.index_select(0, read_rows).float().transpose(1, 2)
    seq = torch.cat([history, tokens], 1)  # [batch, window + steps, channels]

    columns = []
    for column in range(width):
        if parent_indices is None:
            columns.append(seq[:, column : column + steps])
            continue
        node = torch.arange(steps, device=device).expand(batch_size, steps).clone()
        past = torch.zeros(batch_size, steps, dtype=torch.long, device=device)
        for _ in range(width - 1 - column):
            parent = parent_indices.long().gather(1, node.clamp(min=0))
            past += (node < 0).long() + ((node >= 0) & (parent < 0)).long()
            node = torch.where(node >= 0, parent, node)
        from_tokens = seq.gather(
            1,
            (node + window)
            .clamp(0, seq.shape[1] - 1)
            .unsqueeze(-1)
            .expand(batch_size, steps, channels),
        ).squeeze(-1)
        from_history = history.gather(
            1,
            (width - 1 - past)
            .clamp(0, window - 1)
            .unsqueeze(-1)
            .expand(batch_size, steps, channels),
        ).squeeze(-1)
        columns.append(
            torch.where((node >= 0).unsqueeze(-1), from_tokens, from_history)
        )

    out = torch.zeros(batch_size, steps, channels, dtype=torch.float32, device=device)
    for column, values in enumerate(columns):
        out += weight[:, column].float() * values

    # A token's window is columns 1..width-1; it goes to the row it names.
    windows = torch.stack(columns[1:], -1).to(conv_state.dtype)
    destinations = write_rows.reshape(-1).clamp(min=0)
    conv_state.index_copy_(
        0,
        destinations,
        windows.reshape(-1, channels, window),
    )
    return out.reshape(batch_size * steps, channels).to(flat.dtype)


__all__ = [
    "causal_conv1d_prefill",
    "causal_conv1d_update",
]
