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

from collections.abc import Callable
from functools import partial

import torch
from tokenspeed_kernel.ops.quantization import quantize_fp8

from tokenspeed.runtime.layers.moe.types import MoELayerSpec


def preserve_e8m0_bytes_for_uint8_param(
    dst: torch.Tensor,
    src: torch.Tensor,
) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and dst.dtype == torch.uint8 and src.dtype == e8m0_dtype:
        return src.view(torch.uint8)
    return src


def copy_expert_shard(
    dst: torch.Tensor,
    src: torch.Tensor,
    scale_dst: torch.Tensor | None = None,
    block_shape: tuple[int, int] | None = None,
    weight_copy: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
) -> None:
    """Write one loaded expert shard into its destination buffer.

    Args:
        dst: Destination view inside the expert weight buffer.
        src: Loaded (already sharded) checkpoint values.
        scale_dst: Destination view inside the block-scale buffer, or ``None``
            when the checkpoint values need no quantization.
        block_shape: ``(block_n, block_k)`` of one scale.
        weight_copy: Optional destination copy for a backend weight layout,
            applied after sharding and quantization.
    """
    if scale_dst is None or dst.dtype == src.dtype:
        src = preserve_e8m0_bytes_for_uint8_param(dst, src)
        if weight_copy is None:
            dst.copy_(src)
        else:
            weight_copy(dst, src)
        return

    quantized, scales = quantize_fp8(
        src.to(dst.device), granularity="block", block_size=block_shape
    )
    if weight_copy is None:
        dst.copy_(quantized)
    else:
        weight_copy(dst, quantized)
    scale_dst.copy_(scales)


def _narrow_block_scale(
    scale: torch.Tensor | None,
    block_shape: tuple[int, int] | None,
    shard_dim: int,
    block_start: int,
    length: int,
) -> torch.Tensor | None:
    """Narrow a per-expert block-scale view alongside its weight shard."""

    if scale is None or block_shape is None or shard_dim not in (0, 1):
        return None
    block = block_shape[shard_dim]
    num_blocks = (length + block - 1) // block
    return scale.narrow(shard_dim, block_start, num_blocks)


def load_w13(
    expert_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    shard_dim: int,
    tp_rank: int,
    is_bias: bool,
    use_presharded_weights: bool,
    do_transpose: bool,
    tp_size: int = 1,
    load_up_proj_weight_first: bool = False,
    expert_scale: torch.Tensor | None = None,
    block_shape: tuple[int, int] | None = None,
    weight_copy: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
) -> None:
    if shard_id not in {"w1", "w3", "w13"}:
        raise ValueError(f"Unexpected w13 shard_id: {shard_id}")

    if is_bias:
        shard_dim = -1

    if shard_id in {"w1", "w3"}:
        shard_size = expert_data.shape[shard_dim] // 2
    else:
        shard_size = expert_data.shape[shard_dim]

    switch_w13 = load_up_proj_weight_first
    if (switch_w13 and shard_id == "w1") or (not switch_w13 and shard_id == "w3"):
        start = shard_size
    else:
        start = 0

    if not use_presharded_weights:
        if not is_bias and do_transpose:
            loaded_weight = loaded_weight.transpose(-2, -1)
        load_start = min(shard_size * tp_rank, loaded_weight.shape[shard_dim])
        load_size = min(shard_size, loaded_weight.shape[shard_dim] - load_start)
        loaded_weight = loaded_weight.narrow(shard_dim, load_start, load_size)

    if loaded_weight.shape[shard_dim] == 0:
        return

    expert_data = expert_data.narrow(shard_dim, start, shard_size)
    dst = expert_data.narrow(shard_dim, 0, loaded_weight.shape[shard_dim])
    scale_dst = None
    if not is_bias:
        block_start = 0
        if start and block_shape is not None and shard_dim in (0, 1):
            block = block_shape[shard_dim]
            block_start = round_up(shard_size, block) // block
        scale_dst = _narrow_block_scale(
            expert_scale,
            block_shape,
            shard_dim,
            block_start,
            loaded_weight.shape[shard_dim],
        )
    copy_expert_shard(dst, loaded_weight, scale_dst, block_shape, weight_copy)


def load_w2(
    expert_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    shard_dim: int,
    tp_rank: int,
    is_bias: bool,
    use_presharded_weights: bool,
    do_transpose: bool,
    tp_size: int = 1,
    expert_scale: torch.Tensor | None = None,
    block_shape: tuple[int, int] | None = None,
    weight_copy: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
) -> None:
    if not isinstance(expert_data, torch.Tensor) or not isinstance(
        loaded_weight, torch.Tensor
    ):
        raise ValueError("expert_data and loaded_weight must be torch.Tensor")

    if shard_id != "w2":
        raise ValueError(f"shard_id must be 'w2', got {shard_id}")

    if is_bias:
        shard_dim = -1
        shard_size = expert_data.shape[-1]
    else:
        shard_size = expert_data.shape[shard_dim]

    if not use_presharded_weights:
        if not is_bias and do_transpose:
            loaded_weight = loaded_weight.transpose(-2, -1)
        if is_bias:
            loaded_weight = loaded_weight.narrow(shard_dim, 0, shard_size)
        else:
            load_start = min(shard_size * tp_rank, loaded_weight.shape[shard_dim])
            load_size = min(shard_size, loaded_weight.shape[shard_dim] - load_start)
            loaded_weight = loaded_weight.narrow(shard_dim, load_start, load_size)

    if loaded_weight.shape[shard_dim] == 0:
        return

    dst = expert_data.narrow(shard_dim, 0, loaded_weight.shape[shard_dim])
    scale_dst = None
    if not is_bias:
        scale_dst = _narrow_block_scale(
            expert_scale,
            block_shape,
            shard_dim,
            0,
            loaded_weight.shape[shard_dim],
        )
    copy_expert_shard(dst, loaded_weight, scale_dst, block_shape, weight_copy)


def get_shard_dim(param: torch.Tensor, shard_id: str, do_transpose: bool) -> int:
    is_transposed = getattr(param, "is_transposed", False)
    if do_transpose:
        is_transposed = True

    shard_dim = {"w1": 0, "w2": 1, "w3": 0, "w13": 0}[shard_id]
    if is_transposed:
        shard_dim = int(not shard_dim)
    return shard_dim


def load_model_weight(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
    tp_rank: int,
    is_bias: bool,
    use_presharded_weights: bool,
    do_transpose: bool,
    tp_size: int = 1,
    weight_copy: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
) -> None:
    expert_data = param.data[local_expert_id]
    shard_dim = get_shard_dim(param, shard_id, do_transpose)
    block_scale = getattr(param, "block_scale_inv", None)
    block_shape = getattr(param, "block_scale_shape", None)
    expert_scale = None if block_scale is None else block_scale.data[local_expert_id]
    if shard_id == "w2":
        load_w2(
            expert_data,
            loaded_weight,
            shard_id,
            shard_dim,
            tp_rank,
            is_bias,
            use_presharded_weights,
            do_transpose,
            tp_size=tp_size,
            expert_scale=expert_scale,
            block_shape=block_shape,
            weight_copy=weight_copy,
        )
    elif shard_id in {"w1", "w3", "w13"}:
        load_w13(
            expert_data,
            loaded_weight,
            shard_id,
            shard_dim,
            tp_rank,
            is_bias,
            use_presharded_weights,
            do_transpose,
            tp_size=tp_size,
            expert_scale=expert_scale,
            block_shape=block_shape,
            weight_copy=weight_copy,
        )
    else:
        raise ValueError(f"Unknown shard_id: {shard_id}")


def load_group_weight_scale(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    local_expert_id: int,
    shard_id: str,
    tp_rank: int,
    do_transpose: bool,
    tp_size: int = 1,
) -> None:
    load_model_weight(
        param,
        loaded_weight,
        shard_id,
        local_expert_id,
        tp_rank,
        False,
        False,
        do_transpose,
        tp_size=tp_size,
    )


def load_per_tensor_weight_scale(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
) -> None:
    if shard_id in {"w1", "w3"}:
        idx = 0 if shard_id == "w1" else 1
        param.data[local_expert_id][idx] = loaded_weight
    elif shard_id in {"w13", "w2"}:
        param.data[local_expert_id] = loaded_weight
    else:
        raise ValueError(f"Unknown shard_id: {shard_id}")


def load_channel_weight_scale(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str | None = None,
    local_expert_id: int | None = None,
    *,
    tp_rank: int,
    tp_size: int,
    param_name: str = "expert weight scale",
    use_presharded_weights: bool = False,
) -> None:
    """Load one projection's scales into ``[experts, 2 * ispp]`` (w13) or
    ``[experts, hidden]`` (w2) FP32 scales per output channel.

    ``loaded_weight`` holds one scale per output channel (``[N, 1]`` or
    ``[N]``) or one per tensor (one value, which fills the projection's
    channels on every rank). Gate (``w1``) and up (``w3``) fill the first and
    second half of w13's channels: ``tp_size * ispp`` of them, of which each
    rank takes its partition as ``load_w13`` does for the weight rows (``ispp``
    when presharded). ``w2``'s channels are the hidden size, which tensor
    parallelism does not split: every rank holds all of them. Any other count
    is refused. The values convert exactly to FP32 (from BF16, FP16, FP32 or
    FP8 E4M3).

    A tensor that holds every expert's scales at once (stacked or fused, such
    as ``[experts, N, 1]``) is refused: it comes without ``shard_id`` and
    ``local_expert_id``, and this loader does not split it.
    """
    stacked = loaded_weight.dim() > 2 or (
        loaded_weight.dim() == 2 and loaded_weight.shape[1] != 1
    )
    if shard_id is None or local_expert_id is None or stacked:
        raise ValueError(
            f"{param_name}: got one scale tensor of shape "
            f"{tuple(loaded_weight.shape)} for shard_id={shard_id!r}, "
            f"local_expert_id={local_expert_id!r}. FP8 per-channel experts load "
            f"one projection of one expert at a time ([N, 1], [N] or one value); "
            f"stacked or fused expert scales such as [experts, N, 1] are not "
            f"supported yet."
        )
    scale = loaded_weight.reshape(-1)
    expert = param.data[local_expert_id]
    if shard_id == "w2":
        dst, channels = expert, expert.shape[0]
    elif shard_id in {"w1", "w3"}:
        size = expert.shape[0] // 2
        dst = expert.narrow(0, 0 if shard_id == "w1" else size, size)
        channels = size if use_presharded_weights else size * tp_size
    else:
        raise ValueError(f"Unknown shard_id: {shard_id}")
    if scale.numel() == 1:
        dst.copy_(scale.expand(dst.shape[0]))
        return
    if scale.numel() != channels:
        raise ValueError(
            f"{shard_id} scales per output channel: expected {channels}, got "
            f"{tuple(loaded_weight.shape)}"
        )
    if channels != dst.shape[0]:
        scale = scale.narrow(0, dst.shape[0] * tp_rank, dst.shape[0])
    dst.copy_(scale)


def load_per_tensor_input_scale(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int | None,
) -> None:
    value = loaded_weight.detach().to(torch.float32).amax().to(param.device)
    torch.maximum(param.data, value, out=param.data)


def load_per_expert_input_scale(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int | None,
) -> None:
    """Keep one activation scale per expert, broadcasting a checkpoint that ships one.

    Checkpoints that ship a scalar scale per expert load it once per expert
    with ``local_expert_id=None``; accumulate the running maximum so every
    expert uses the global maximum, matching ``load_per_tensor_input_scale``.

    Args:
        param: The layer's ``[num_local_experts]`` activation scale.
        loaded_weight: A scalar, or one scale per expert.
        shard_id: Unused; the scale covers the whole projection.
        local_expert_id: Expert this scale belongs to, or None when the tensor
            already covers every local expert.
    """
    del shard_id
    value = loaded_weight.detach().to(dtype=torch.float32, device=param.device)
    if local_expert_id is not None:
        param.data[local_expert_id] = value.amax()
        return
    if value.numel() == 1:
        torch.maximum(param.data, value, out=param.data)
        return
    if value.numel() != param.data.numel():
        raise ValueError(
            f"expected one activation scale per expert, got {value.numel()} "
            f"for {param.data.numel()} experts"
        )
    param.data.copy_(value.reshape(param.data.shape))


def make_weight_loader(
    spec: MoELayerSpec,
    *,
    is_bias: bool = False,
    do_transpose: bool = False,
    use_presharded_weights: bool = False,
) -> Callable:
    return partial(
        load_model_weight,
        tp_rank=spec.tp_rank,
        is_bias=is_bias,
        use_presharded_weights=use_presharded_weights,
        do_transpose=do_transpose,
        tp_size=spec.tp_size,
    )


def make_channel_scale_loader(
    spec: MoELayerSpec, param_name: str = "expert weight scale"
) -> Callable:
    return partial(
        load_channel_weight_scale,
        tp_rank=spec.tp_rank,
        tp_size=spec.tp_size,
        param_name=f"{spec.prefix}.{param_name}",
    )


def make_group_scale_loader(
    spec: MoELayerSpec,
    *,
    do_transpose: bool = False,
) -> Callable:
    return partial(
        load_group_weight_scale,
        tp_rank=spec.tp_rank,
        do_transpose=do_transpose,
        tp_size=spec.tp_size,
    )


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


__all__ = [
    "load_channel_weight_scale",
    "load_per_tensor_input_scale",
    "load_per_tensor_weight_scale",
    "make_channel_scale_loader",
    "make_group_scale_loader",
    "make_weight_loader",
    "round_up",
]
