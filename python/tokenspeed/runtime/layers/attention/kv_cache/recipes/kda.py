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

"""KDA checkpoint fields and transient verify workspace."""

import torch

from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    cache_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupDeclaration,
)


def kda_recurrent_state_dtype(server_args) -> torch.dtype:
    """Return the dense-base dtype selected by ``--mamba-ssm-dtype``.

    Retained history and accumulation remain FP32. The selected dtype rounds
    the logical state only when materialization writes a new dense base.
    """
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[server_args.mamba_ssm_dtype]


def kda_state_fields(
    layer_id: int,
    plane_id: str,
    *,
    conv_shape: tuple[int, int],
    recurrent_shape: tuple[int, int, int],
    recurrent_state_dtype: torch.dtype,
    history_capacity: int,
) -> tuple[CacheFieldSpec, ...]:
    """Declare a layer's convolution state and recurrent checkpoint.

    Args:
        layer_id: Model layer owning every field.
        plane_id: Arena plane that packs this layer's state fields.
        conv_shape: Per-rank convolution history shape.
        recurrent_shape: Per-rank (heads, value dimension, key dimension).
        recurrent_state_dtype: Dense-base storage dtype (FP32 or BF16).
            History and logical-state accumulation remain FP32.
        history_capacity: Total slots shared by accepted history and the
            current verify candidates; capacity must be positive.
            A direct-write consumer must reserve its entire candidate window
            inside this capacity before verify; it must fit even at length zero.

    Returns:
        Fields belonging to one scheduler-owned state block. With history,
        the recurrent field is a dense base. FP32 corrections, normalized
        keys and multiplicative decays reconstruct its accepted endpoint in
        token order. The INT32 length publishes only the accepted prefix;
        candidate writes beyond it do not change the logical checkpoint.
        Verify writes into a private live page, consolidating the accepted
        history first when the candidate window would exceed capacity. Commit
        publishes the accepted length without copying these record fields.
        The next verify starts at that length, reusing rejected candidate slots.
        Copying or transferring a checkpoint must carry every field together.
    """
    if history_capacity <= 0:
        raise ValueError("KDA checkpoints require positive retained history capacity")
    fields = (
        CacheFieldSpec(
            f"layer.{layer_id}.conv_state",
            plane_id,
            conv_shape,
            cache_dtype_name(torch.bfloat16),
            exact_page_stride=False,
        ),
        CacheFieldSpec(
            f"layer.{layer_id}.recurrent_state",
            plane_id,
            recurrent_shape,
            cache_dtype_name(recurrent_state_dtype),
            exact_page_stride=False,
        ),
    )
    heads, value_dim, key_dim = recurrent_shape
    fields += tuple(
        CacheFieldSpec(
            f"layer.{layer_id}.{name}",
            plane_id,
            shape,
            cache_dtype_name(dtype),
            exact_page_stride=False,
        )
        for name, shape, dtype in (
            (
                "replay_correction",
                (heads, history_capacity, value_dim),
                torch.float32,
            ),
            ("replay_key", (heads, history_capacity, key_dim), torch.float32),
            ("replay_decay", (heads, history_capacity, key_dim), torch.float32),
            ("replay_length", (1,), torch.int32),
        )
    )
    return fields


def kda_retained_supported(config: AttnConfig) -> bool:
    """Probe the complete retained verify/commit geometry at startup."""
    from tokenspeed_kernel.ops.attention.kda import (
        kda_recurrent_layout,
    )
    from tokenspeed_kernel.ops.attention.kda import (
        kda_retained_supported as kernel_retained_supported,
    )

    linear = config.component(LinearAttnConfig)
    heads, head_dim, key_dim = linear.temporal_state_shape
    if head_dim != key_dim:
        return False
    return kernel_retained_supported(
        config.dtype,
        recurrent_layout=kda_recurrent_layout(),
        num_heads=heads,
        head_dim=head_dim,
        conv_kernel_size=linear.conv_kernel_size,
    )


def kda_verify_workspace_bytes(
    config: AttnConfig,
    groups: tuple[CacheGroupDeclaration, ...],
    *,
    draft_token_num: int,
) -> int:
    """Return retained KDA verify workspace bytes outside the cache arena.

    Args:
        config: Target attention configuration with model dimensions,
            activation dtype, and maximum serving batch size.
        groups: Cache declarations containing the target's KDA state fields.
        draft_token_num: Effective verify width per request. Use 1 for
            ordinary single-token decode.

    Returns:
        Required runtime-owned workspace in bytes for the maximum batch size.
        Includes original QKV, nine-column commit descriptors, page routing,
        and generation/position stamps. Cache-owned
        checkpoint fields, including dense recurrent state and retained
        history, are budgeted separately in the cache arena.
    """
    state_fields = tuple(
        field
        for spec, fields in groups
        if spec.attn_kind == "state"
        for field in fields
    )
    from tokenspeed_kernel.ops.attention.kda import (
        KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS,
    )

    linear = config.component(LinearAttnConfig)
    conv_fields = tuple(
        field for field in state_fields if field.field_id.endswith(".conv_state")
    )
    row_bytes = linear.conv_state_shape[0] * config.dtype.itemsize
    group_count = sum(spec.attn_kind == "state" for spec, _ in groups)
    descriptor_columns = KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS
    metadata_bytes = len(conv_fields) * (descriptor_columns * 8 + 4)
    metadata_bytes += group_count * 4 * 8
    metadata_bytes += (2 * group_count + 1) * config.max_bs * 4
    metadata_bytes += config.max_bs * 3 * 8
    metadata_bytes += len(conv_fields) * config.max_bs * draft_token_num * 2 * 8
    return (
        len(conv_fields) * config.max_bs * draft_token_num * row_bytes + metadata_bytes
    )
