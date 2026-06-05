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

from functools import partial

import torch


def _preserve_e8m0_bytes_for_uint8_param(
    dst: torch.Tensor,
    src: torch.Tensor,
) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and dst.dtype == torch.uint8 and src.dtype == e8m0_dtype:
        return src.view(torch.uint8)
    return src


def _load_w13(
    expert_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    shard_dim: int,
    tp_rank: int,
    is_bias: bool,
    use_presharded_weights: bool,
    do_transpose: bool,
    load_up_proj_weight_first: bool,
    tp_size: int = 1,
):
    # Index the loaded weight for tp sharding.
    # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
    assert shard_id in {"w1", "w3", "w13"}

    if is_bias:
        # if this weight is a bias, the last dimension must be the sharded dimension
        shard_dim = -1

    if shard_id in {"w1", "w3"}:
        # non-fused version
        shard_size = expert_data.shape[shard_dim] // 2
    elif shard_id in {"w13"}:
        # fused version
        shard_size = expert_data.shape[shard_dim]
    else:
        raise NotImplementedError

    # Narrow parameter and load.
    # w1, gate_proj: Load into first logical weight of w13.
    # w3, up_proj: Load into second logical weight of w13.
    # The fused Cutlass kernel assumes a different layout.
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
    loaded_weight = _preserve_e8m0_bytes_for_uint8_param(dst, loaded_weight)
    dst.copy_(loaded_weight)


def _load_w2(
    expert_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    shard_dim: int,
    tp_rank: int,
    is_bias: bool,
    use_presharded_weights: bool,
    do_transpose: bool,
    tp_size: int = 1,
):
    if not isinstance(expert_data, torch.Tensor) or not isinstance(
        loaded_weight, torch.Tensor
    ):
        raise ValueError("expert_data and loaded_weight must be torch.Tensor")

    if shard_id != "w2":
        raise ValueError(f"shard_id must be 'w2', got {shard_id}")

    # Index the loaded weight for tp sharding.
    # down_proj: "RowParallel" so tp sharding on input_dim
    # Narrow parameter and load.
    if is_bias:
        # this expert_data is a bias, not weight,
        # for w2_weight_bias in TP, it does not need to be sharded
        shard_size = expert_data.shape[-1]
    else:
        # this parameter is a weight matrix
        # for w2 in TP, it shards the input_features, i.e., shard_dim=2
        shard_size = expert_data.shape[shard_dim]

    if not use_presharded_weights:
        if not is_bias and do_transpose:
            loaded_weight = loaded_weight.transpose(-2, -1)
        if tp_size > 1:
            load_shard = loaded_weight.shape[shard_dim] // tp_size
        else:
            load_shard = shard_size
        loaded_weight = loaded_weight.narrow(
            shard_dim, load_shard * tp_rank, load_shard
        )

    # w2, down_proj: Load into only logical weight of w2.
    dst = expert_data.narrow(shard_dim, 0, loaded_weight.shape[shard_dim])
    loaded_weight = _preserve_e8m0_bytes_for_uint8_param(dst, loaded_weight)
    dst.copy_(loaded_weight)


def get_shard_dim(param, shard_id, do_transpose):
    is_transposed = getattr(param, "is_transposed", False)
    if do_transpose:
        is_transposed = True

    SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}
    shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
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
):
    expert_data = param.data[local_expert_id]
    shard_dim = get_shard_dim(param, shard_id, do_transpose)

    if shard_id == "w2":
        load_func = _load_w2
    elif shard_id in ("w1", "w3", "w13"):
        load_func = partial(_load_w13, load_up_proj_weight_first=False)
    else:
        raise ValueError(f"Unknown shard_id: {shard_id}")

    load_func(
        expert_data,
        loaded_weight,
        shard_id,
        shard_dim,
        tp_rank,
        is_bias,
        use_presharded_weights,
        do_transpose,
        tp_size=tp_size,
    )


# for per tensor weight quantization
def load_per_tensor_weight_scale(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
):
    if shard_id in ("w1", "w3"):
        # We have to keep the weight scales of w1 and w3 because
        # we need to re-quantize w1/w3 weights after weight loading.
        idx = 0 if shard_id == "w1" else 1
        param.data[local_expert_id][idx] = loaded_weight
    # If we are in the row parallel case (down_proj)
    elif shard_id == "w2":
        param.data[local_expert_id] = loaded_weight
    else:
        raise RuntimeError(f"Unrecognized shard_id: {shard_id}")


def load_group_weight_scale(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    local_expert_id: int,
    shard_id: str,
    tp_rank: int,
    do_transpose: bool,
):
    expert_data = param.data[local_expert_id]
    shard_dim = get_shard_dim(param, shard_id, do_transpose)

    if shard_id == "w2":
        load_func = _load_w2
    elif shard_id in ("w1", "w3", "w13"):
        load_func = partial(_load_w13, load_up_proj_weight_first=False)
    else:
        raise ValueError(f"Unknown shard_id: {shard_id}")

    load_func(
        expert_data,
        loaded_weight,
        shard_id,
        shard_dim,
        tp_rank,
        False,
        False,
        do_transpose,
    )


def load_per_channel_weight_scale(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    local_expert_id: int,
    shard_id: str,
    tp_rank: int,
    do_transpose: bool,
):
    expert_data = param.data[local_expert_id]
    shard_dim = get_shard_dim(param, shard_id, do_transpose)

    # for per channel weight quantization
    if shard_id == "w2":
        loaded_weight = _preserve_e8m0_bytes_for_uint8_param(expert_data, loaded_weight)
        expert_data.copy_(loaded_weight)
    elif shard_id in ("w1", "w3"):
        _load_w13(
            expert_data,
            loaded_weight,
            shard_id,
            shard_dim,
            tp_rank,
            False,
            False,
            do_transpose,
            False,
        )
    else:
        raise ValueError(f"Unknown shard_id: {shard_id}")
