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
from tokenspeed_kernel.ops.gemm.fp8_utils import per_block_quant_fp8

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
) -> None:
    """Write one loaded expert shard into its destination buffer.

    Args:
        dst: Destination view inside the expert weight buffer.
        src: Loaded (already sharded) checkpoint values.
        scale_dst: Destination view inside the block-scale buffer, or ``None``
            when the checkpoint values need no quantization.
        block_shape: ``(block_n, block_k)`` of one scale.
    """
    if scale_dst is None or dst.dtype == src.dtype:
        src = preserve_e8m0_bytes_for_uint8_param(dst, src)
        dst.copy_(src)
        return

    quantized, scales = per_block_quant_fp8(src.to(dst.device), block_shape)
    dst.copy_(quantized)
    scale_dst.copy_(scales)


def _narrow_block_scale(
    scale: torch.Tensor | None,
    block_shape: tuple[int, int] | None,
    shard_dim: int,
    start: int,
    length: int,
) -> torch.Tensor | None:
    """Narrow a per-expert block-scale view alongside its weight shard."""

    if scale is None or block_shape is None or shard_dim not in (0, 1):
        return None
    block = block_shape[shard_dim]
    if start % block != 0:
        return None
    num_blocks = (length + block - 1) // block
    return scale.narrow(shard_dim, start // block, num_blocks)


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
        if tp_size > 1:
            # Derive the unpadded shard size from the checkpoint tensor.
            # expert_data may be Blackwell-padded; checkpoint is not.
            load_shard = loaded_weight.shape[shard_dim] // tp_size
            loaded_weight = loaded_weight.narrow(
                shard_dim, load_shard * tp_rank, load_shard
            )
        else:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )

    expert_data = expert_data.narrow(shard_dim, start, shard_size)
    dst = expert_data.narrow(shard_dim, 0, loaded_weight.shape[shard_dim])
    scale_dst = None
    if not is_bias:
        scale_dst = _narrow_block_scale(
            expert_scale,
            block_shape,
            shard_dim,
            start,
            loaded_weight.shape[shard_dim],
        )
    copy_expert_shard(dst, loaded_weight, scale_dst, block_shape)


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
            load_shard = shard_size
        elif tp_size > 1:
            load_shard = loaded_weight.shape[shard_dim] // tp_size
        else:
            load_shard = shard_size
        start = 0 if is_bias else load_shard * tp_rank
        loaded_weight = loaded_weight.narrow(shard_dim, start, load_shard)

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
    copy_expert_shard(dst, loaded_weight, scale_dst, block_shape)


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
    elif shard_id == "w2":
        param.data[local_expert_id] = loaded_weight
    else:
        raise ValueError(f"Unknown shard_id: {shard_id}")


def load_per_tensor_input_scale(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
) -> None:
    value = loaded_weight.detach().to(torch.float32).reshape(())
    if shard_id in {"w1", "w3"}:
        prev = param.data[local_expert_id]
        param.data[local_expert_id] = torch.maximum(prev, value)
    elif shard_id == "w2":
        param.data[local_expert_id] = value
    else:
        raise ValueError(f"Unknown shard_id for input scale: {shard_id}")


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


def per_tensor_scale_loader() -> Callable:
    return load_per_tensor_weight_scale


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


__all__ = [
    "load_per_tensor_input_scale",
    "make_group_scale_loader",
    "make_weight_loader",
    "per_tensor_scale_loader",
    "round_up",
]
