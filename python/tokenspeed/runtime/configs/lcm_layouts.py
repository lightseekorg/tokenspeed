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

"""Model-specific field recipes for the shared LCM planner."""

from __future__ import annotations

from tokenspeed.runtime.configs.lcm_memory_plan import LcmFieldSpec


def draft_history_lcm_fields(
    *,
    layer_group_ids,
    enabled_layer_ids,
    logical_block_tokens,
    layer_kv_heads,
    head_dim,
    kv_element_size,
    kv_scale_block_size=0,
    kv_scale_element_size=0,
) -> tuple[LcmFieldSpec, ...]:
    """Describe the enabled draft model's attention history fields."""
    if len(layer_group_ids) != len(layer_kv_heads):
        raise ValueError(
            f"layer_group_ids has {len(layer_group_ids)} entries but "
            f"layer_kv_heads has {len(layer_kv_heads)}"
        )
    if logical_block_tokens <= 0 or head_dim <= 0 or kv_element_size <= 0:
        raise ValueError("draft history geometry must be positive")
    if bool(kv_scale_block_size) != bool(kv_scale_element_size):
        raise ValueError(
            "kv_scale_block_size and kv_scale_element_size must both be zero "
            "or both be positive"
        )
    if kv_scale_block_size and (
        logical_block_tokens % 128
        or head_dim % kv_scale_block_size
        or kv_scale_block_size <= 0
    ):
        raise ValueError("draft scale geometry is incompatible with the KV shape")

    enabled_layer_ids = tuple(enabled_layer_ids)
    if len(set(enabled_layer_ids)) != len(enabled_layer_ids) or any(
        isinstance(layer_id, bool)
        or not isinstance(layer_id, int)
        or layer_id < 0
        or layer_id >= len(layer_group_ids)
        for layer_id in enabled_layer_ids
    ):
        raise ValueError("enabled draft layer ids must be unique valid layer ids")

    occurrences: dict[str, int] = {}
    fields = []
    for layer_id in enabled_layer_ids:
        group_id = layer_group_ids[layer_id]
        kv_heads = layer_kv_heads[layer_id]
        if kv_heads <= 0:
            raise ValueError("draft layer KV heads must be positive")
        unit = occurrences.get(group_id, 0)
        occurrences[group_id] = unit + 1
        kv_shape = (logical_block_tokens, kv_heads, head_dim)
        fields.extend(
            (
                LcmFieldSpec(
                    group_id,
                    f"layer.{layer_id}.k",
                    f"unit.{unit}.k",
                    kv_shape,
                    kv_element_size,
                ),
                LcmFieldSpec(
                    group_id,
                    f"layer.{layer_id}.v",
                    f"unit.{unit}.v",
                    kv_shape,
                    kv_element_size,
                ),
            )
        )
        if kv_scale_block_size:
            scale_dim = head_dim // kv_scale_block_size
            scale_shape = (
                kv_heads,
                logical_block_tokens // 128,
                32,
                scale_dim,
                scale_dim,
            )
            fields.extend(
                (
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.k_scale",
                        f"unit.{unit}.k_scale",
                        scale_shape,
                        kv_scale_element_size,
                    ),
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.v_scale",
                        f"unit.{unit}.v_scale",
                        scale_shape,
                        kv_scale_element_size,
                    ),
                )
            )
    return tuple(fields)


def qwen_gdn_lcm_fields(
    *,
    layer_types,
    layer_group_ids,
    logical_block_tokens,
    kv_shape,
    kv_element_size,
    conv_shape,
    conv_element_size,
    ssm_shape,
    ssm_element_size,
) -> tuple[LcmFieldSpec, ...]:
    """Describe Qwen3.5 history pages and recurrent state checkpoints."""
    if len(layer_types) != len(layer_group_ids):
        raise ValueError(
            f"layer_types has {len(layer_types)} entries but layer_group_ids "
            f"has {len(layer_group_ids)}"
        )
    if tuple(kv_shape)[0] != logical_block_tokens:
        raise ValueError("kv_shape must start with logical_block_tokens")

    occurrences: dict[str, int] = {}
    fields = []
    for layer_id, (label, group_id) in enumerate(zip(layer_types, layer_group_ids)):
        unit = occurrences.get(group_id, 0)
        occurrences[group_id] = unit + 1
        if label == "linear_attention":
            fields.extend(
                (
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.ssm",
                        f"unit.{unit}.a",
                        tuple(ssm_shape),
                        ssm_element_size,
                    ),
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.conv",
                        f"unit.{unit}.b",
                        tuple(conv_shape),
                        conv_element_size,
                        # causal_conv1d reads stride(0) at launch, so conv can
                        # absorb the slack left by history V.
                        exact_page_stride=False,
                    ),
                )
            )
        else:
            fields.extend(
                (
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.k",
                        f"unit.{unit}.a",
                        tuple(kv_shape),
                        kv_element_size,
                    ),
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.v",
                        f"unit.{unit}.b",
                        tuple(kv_shape),
                        kv_element_size,
                    ),
                )
            )
    return tuple(fields)


def inkling_lcm_fields(
    *,
    layer_group_ids,
    logical_block_tokens,
    layer_kv_heads,
    head_dim,
    kv_element_size,
    hidden_size,
    checkpoint_rows,
    kvconv_element_size,
    hiddenconv_element_size,
    kv_scale_block_size=0,
    kv_scale_element_size=0,
) -> tuple[LcmFieldSpec, ...]:
    """Describe Inkling attention pages and ShortConv checkpoints."""
    if len(layer_group_ids) != len(layer_kv_heads):
        raise ValueError(
            f"layer_group_ids has {len(layer_group_ids)} entries but "
            f"layer_kv_heads has {len(layer_kv_heads)}"
        )
    if bool(kv_scale_block_size) != bool(kv_scale_element_size):
        raise ValueError(
            "kv_scale_block_size and kv_scale_element_size must both be zero "
            "or both be positive"
        )
    if kv_scale_block_size and head_dim % kv_scale_block_size:
        raise ValueError("head_dim must be divisible by kv_scale_block_size")
    if kv_scale_block_size and (
        logical_block_tokens % 128 or head_dim != 128 or kv_scale_block_size != 32
    ):
        raise ValueError(
            "Inkling MXFP8 scale fields require P divisible by 128, "
            "head_dim 128 and scale block size 32"
        )

    occurrences: dict[str, int] = {}
    fields = []
    for layer_id, (group_id, kv_heads) in enumerate(
        zip(layer_group_ids, layer_kv_heads)
    ):
        unit = occurrences.get(group_id, 0)
        occurrences[group_id] = unit + 1
        k_plane = f"unit.{unit}.k"
        v_plane = f"unit.{unit}.v"
        if hiddenconv_element_size > 1:
            hidden_k_plane = f"unit.{unit}.hidden_k"
            hidden_v_plane = f"unit.{unit}.hidden_v"
        else:
            hidden_k_plane = k_plane
            hidden_v_plane = v_plane
        kv_shape = (logical_block_tokens, kv_heads, head_dim)
        kvconv_shape = (checkpoint_rows, kv_heads * head_dim)
        hiddenconv_shape = (checkpoint_rows, hidden_size)
        fields.extend(
            (
                LcmFieldSpec(
                    group_id,
                    f"layer.{layer_id}.k",
                    k_plane,
                    kv_shape,
                    kv_element_size,
                ),
                LcmFieldSpec(
                    group_id,
                    f"layer.{layer_id}.v",
                    v_plane,
                    kv_shape,
                    kv_element_size,
                ),
                LcmFieldSpec(
                    "kvconv",
                    f"layer.{layer_id}.kvconv_k",
                    k_plane,
                    kvconv_shape,
                    kvconv_element_size,
                    exact_page_stride=False,
                ),
                LcmFieldSpec(
                    "kvconv",
                    f"layer.{layer_id}.kvconv_v",
                    v_plane,
                    kvconv_shape,
                    kvconv_element_size,
                    exact_page_stride=False,
                ),
                LcmFieldSpec(
                    "hiddenconv",
                    f"layer.{layer_id}.attnconv",
                    hidden_k_plane,
                    hiddenconv_shape,
                    hiddenconv_element_size,
                    exact_page_stride=False,
                ),
                LcmFieldSpec(
                    "hiddenconv",
                    f"layer.{layer_id}.mlpconv",
                    hidden_v_plane,
                    hiddenconv_shape,
                    hiddenconv_element_size,
                    exact_page_stride=False,
                ),
            )
        )
        if kv_scale_block_size:
            scale_dim = head_dim // kv_scale_block_size
            scale_shape = (
                kv_heads,
                logical_block_tokens // 128,
                32,
                scale_dim,
                scale_dim,
            )
            fields.extend(
                (
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.k_scale",
                        f"unit.{unit}.k_scale",
                        scale_shape,
                        kv_scale_element_size,
                    ),
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.v_scale",
                        f"unit.{unit}.v_scale",
                        scale_shape,
                        kv_scale_element_size,
                    ),
                )
            )
    return tuple(fields)
