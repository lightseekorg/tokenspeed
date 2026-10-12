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

"""KDA verify workspace shared by the Kimi-K3 and GLM cache recipes."""

import torch

from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupDeclaration,
)


def kda_replay_supported(config: AttnConfig) -> bool:
    """Return whether the target's dtype and recurrent geometry support replay."""
    from tokenspeed_kernel.ops.attention.kda import (
        kda_recurrent_layout,
        kda_replay_commit_supported,
    )

    linear = config.component(LinearAttnConfig)
    heads, head_dim, _ = linear.temporal_state_shape
    return bool(
        kda_replay_commit_supported(
            config.dtype,
            recurrent_layout=kda_recurrent_layout(),
            num_heads=heads,
            head_dim=head_dim,
        )
    )


def kda_verify_workspace_bytes(
    config: AttnConfig,
    groups: tuple[CacheGroupDeclaration, ...],
    *,
    draft_token_num: int,
    replay_kda: bool,
) -> int:
    """Return KDA verify staging bytes outside the cache arena.

    Args:
        config: The target's dimensions, dtype and serving batch capacity.
        groups: Cache declarations containing the target's KDA state fields;
            history fields, including draft history, are ignored.
        draft_token_num: The target's effective verify width.
        replay_kda: Replay capability cached by the recipe. Otherwise verify
            retains the full convolution and recurrent state for each position.

    Returns:
        Transient convolution rows and captured replay payloads, or the dense
        state tape. Raw-gate replay's convolution scratch aliases the arena
        and contributes zero. Callers decide whether this engine verifies.
    """
    state_fields = tuple(
        field for spec, fields in groups if spec.family == "state" for field in fields
    )
    if not replay_kda:
        return (
            config.max_bs
            * (draft_token_num + 1)
            * sum(field.payload_bytes for field in state_fields)
        )

    from tokenspeed_kernel.ops.attention.kda import kda_batched_replay_uses_raw_gate

    linear = config.component(LinearAttnConfig)
    heads, head_dim, _ = linear.temporal_state_shape
    raw_gate = kda_batched_replay_uses_raw_gate(
        config.dtype, num_heads=heads, head_dim=head_dim
    )
    conv_fields = tuple(
        field for field in state_fields if field.field_id.endswith(".conv_state")
    )
    conv_bytes = (
        0
        if raw_gate
        else config.max_bs * sum(field.payload_bytes for field in conv_fields)
    )
    row_bytes = (linear.conv_state_shape[0] + head_dim + heads) * config.dtype.itemsize
    row_bytes += (
        heads
        * head_dim
        * (torch.bfloat16.itemsize if raw_gate else torch.float32.itemsize)
    )
    return conv_bytes + len(conv_fields) * config.max_bs * draft_token_num * row_bytes
