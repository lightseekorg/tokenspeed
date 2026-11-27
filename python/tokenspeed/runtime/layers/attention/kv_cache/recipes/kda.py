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

"""KDA checkpoint fields and request-slot decode workspace."""

import torch

from tokenspeed.runtime.execution.request_slots import RequestSlotLayout
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    cache_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupDeclaration,
)

# Accepted records a request slot keeps per KDA layer, counting the verify
# window, when --kda-history-capacity is not set.
KDA_DEFAULT_HISTORY_CAPACITY = 16


def kda_recurrent_state_dtype(server_args) -> torch.dtype:
    """Return the dense-state dtype selected by ``--mamba-ssm-dtype``.

    Retained history and accumulation remain FP32. The selected dtype rounds
    the logical state only when a dense state is written to a cache page.
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
) -> tuple[CacheFieldSpec, ...]:
    """Declare a layer's convolution state and dense recurrent checkpoint.

    Args:
        layer_id: Model layer owning every field.
        plane_id: Arena plane that packs this layer's state fields.
        conv_shape: Per-rank convolution history shape.
        recurrent_shape: Per-rank (heads, value dimension, key dimension).
        recurrent_state_dtype: Dense-state storage dtype (FP32 or BF16).
            History and logical-state accumulation remain FP32.

    Returns:
        The BF16 convolution state and the dense recurrent state of one
        scheduler-owned state block. Accepted decode records and their count
        live in the request-slot workspace (``kda_verify_workspace_bytes``).
    """
    return (
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


def kda_history_capacity(config: AttnConfig, draft_token_num: int) -> int:
    """Retained updates a request slot keeps per layer, counting the verify window.

    ``--kda-history-capacity`` (``LinearAttnConfig.kda_history_capacity``)
    sets it; the default is ``KDA_DEFAULT_HISTORY_CAPACITY`` (16). It must be
    at least ``draft_token_num``, so a verify window wider than 16 needs the
    option. A smaller capacity shrinks the request-slot workspace but folds
    the accepted records into the dense state more often: at the window
    itself, every round that accepts any.

    Raises:
        ValueError: when the capacity, configured or default, is below the
            verify window.
    """
    # Like the state backends' other linear flags, a config without the
    # linear component (a backend built for tests) keeps the default.
    linear = config.component(LinearAttnConfig)
    configured = (
        KDA_DEFAULT_HISTORY_CAPACITY
        if linear is None or linear.kda_history_capacity is None
        else linear.kda_history_capacity
    )
    if configured < draft_token_num:
        raise ValueError(
            f"kda_history_capacity={configured} must be at least the verify "
            f"window of {draft_token_num} draft tokens"
        )
    return configured


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


def _kda_layer_count(groups: tuple[CacheGroupDeclaration, ...]) -> int:
    """KDA layers in the declarations: one convolution state field each."""
    return sum(
        field.field_id.endswith(".conv_state")
        for spec, fields in groups
        if spec.attn_kind == "state"
        for field in fields
    )


def kda_request_slot_bytes(
    config: AttnConfig,
    groups: tuple[CacheGroupDeclaration, ...],
    *,
    draft_token_num: int,
) -> int:
    """Return one request slot's retained KDA records over every KDA layer.

    A request keeps its records in its slot across rounds, scheduled or not,
    so every resident slot (``max_resident_requests``, including a decode
    role's extra slots) holds them.

    Args:
        config: Target attention configuration with model dimensions.
        groups: Cache declarations containing the target's KDA state fields.
        draft_token_num: Effective verify width per request; the history
            capacity (``kda_history_capacity``) must be at least this.

    Returns:
        Bytes of the slot's FP32 correction, key and decay records for each
        KDA layer, plus its INT32 accepted count, which every layer shares.
    """
    linear = config.component(LinearAttnConfig)
    heads, value_dim, key_dim = linear.temporal_state_shape
    return (
        _kda_layer_count(groups)
        * heads
        * kda_history_capacity(config, draft_token_num)
        * (value_dim + 2 * key_dim)
        * torch.float32.itemsize
        + torch.int32.itemsize
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
            activation dtype, maximum batch size and resident request limit.
        groups: Cache declarations containing the target's KDA state fields.
        draft_token_num: Effective verify width per request. Use 1 for
            ordinary single-token decode.

    Returns:
        Bytes for each request slot's retained FP32 correction, key and decay
        records and INT32 accepted count; plus per-batch QKV, commit
        descriptors, routing arrays and generation/position stamps used by
        verify and decode.
    """
    from tokenspeed_kernel.ops.attention.kda import (
        KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS,
    )

    linear = config.component(LinearAttnConfig)
    num_layers = _kda_layer_count(groups)
    num_slots = RequestSlotLayout(config.max_resident_requests).capacity
    retained_bytes = num_slots * kda_request_slot_bytes(
        config, groups, draft_token_num=draft_token_num
    )
    qkv_row_bytes = linear.conv_state_shape[0] * config.dtype.itemsize
    qkv_bytes = num_layers * config.max_bs * draft_token_num * qkv_row_bytes
    group_count = sum(spec.attn_kind == "state" for spec, _ in groups)
    descriptor_columns = KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS
    metadata_bytes = num_layers * (descriptor_columns * 8 + 4)
    metadata_bytes += group_count * 4 * 8
    metadata_bytes += (2 * group_count + 1) * config.max_bs * 4
    metadata_bytes += config.max_bs * 4 * 8
    metadata_bytes += num_layers * config.max_bs * draft_token_num * 2 * 8
    return retained_bytes + qkv_bytes + metadata_bytes
