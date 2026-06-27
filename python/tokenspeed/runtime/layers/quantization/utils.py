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

import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType

import numpy
import torch
from torch.nn import Module

from tokenspeed.runtime.layers.quantization.compressed_tensors.scalar_type import (
    ScalarType,
)


def should_exclude_quant_module(prefix: str, exclude_modules: list[str]) -> bool:
    """Whether ``prefix`` matches a ModelOpt-style glob in ``exclude_modules``."""
    if prefix is None or not exclude_modules:
        return False
    for pattern in exclude_modules:
        regex_str = pattern.replace(".", r"\.").replace("*", ".*")
        if re.fullmatch(regex_str, prefix):
            return True
    return False


def should_ignore_quant_layer(
    prefix: str,
    ignored_layers: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> bool:
    if prefix is None or ignored_layers is None:
        return False

    # layer_name = model.layers.0.self_attn.qkv_proj
    # proj_name = qkv_proj
    proj_name = prefix.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping and prefix not in ignored_layers:
        shard_proj_names = fused_mapping[proj_name]

        # Convert fused_name --> [shard_names]
        shard_names = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in shard_proj_names
        ]

        # Layer should be ignored if shards are ignored.
        should_ignore_layer = None
        for shard_name in shard_names:
            should_ignore_shard = check_equal_or_regex_match(
                layer_name=shard_name, targets=ignored_layers
            )

            # If shard_idx=0, set layer ignore to match shard.
            if should_ignore_layer is None:
                should_ignore_layer = should_ignore_shard

            # If shard_idx=1+ confirm scheme matches prior shards.
            elif should_ignore_shard != should_ignore_layer:
                raise ValueError(
                    f"Found a different quantization schemes for "
                    f"{shard_proj_names} in {prefix}. TokenSpeed "
                    "requires all to use the same scheme."
                )
    else:
        should_ignore_layer = check_equal_or_regex_match(
            layer_name=prefix, targets=ignored_layers
        )
        if not should_ignore_layer:
            if "gate_up_proj" in prefix:
                prefix_gate = prefix.replace("gate_up_proj", "gate_proj")
                prefix_up = prefix.replace("gate_up_proj", "up_proj")
                if prefix_gate in ignored_layers and prefix_up in ignored_layers:
                    should_ignore_layer = True
            elif "fused_qkv_a_proj_with_mqa" in prefix:
                prefix_q_a_proj = prefix.replace(
                    "fused_qkv_a_proj_with_mqa", "q_a_proj"
                )
                prefix_kv_a_proj_with_mqa = prefix.replace(
                    "fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa"
                )
                if (
                    prefix_q_a_proj in ignored_layers
                    and prefix_kv_a_proj_with_mqa in ignored_layers
                ):
                    should_ignore_layer = True
            elif "qkv_proj" in prefix:
                prefix_q_proj = prefix.replace("qkv_proj", "q_proj")
                prefix_k_proj = prefix.replace("qkv_proj", "k_proj")
                prefix_v_proj = prefix.replace("qkv_proj", "v_proj")
                if (
                    prefix_q_proj in ignored_layers
                    and prefix_k_proj in ignored_layers
                    and prefix_v_proj in ignored_layers
                ):
                    should_ignore_layer = True
            elif "experts" in prefix:
                should_ignore_layer = any(
                    [
                        prefix in layer_name
                        for layer_name in ignored_layers
                        if "experts" in layer_name
                    ]
                )

    assert should_ignore_layer is not None
    return should_ignore_layer


def check_equal_or_regex_match(layer_name: str, targets: Iterable[str]) -> bool:
    """
    Checks whether a layer_name is exactly equal or a regex match for
    if target starts with 're:' to any target in list.
    """
    for target in targets:
        if _is_equal_or_regex_match(layer_name, target):
            return True
    return False


def find_matched_target(
    layer_name: str | None,
    module: Module,
    targets: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> str:
    """
    Helper function to look up which "target" in the compressed-tensors
    config that a layer corresponds to.

    Recall that a compressed-tensors configs has a concept of
    config_groups, where each layer can be quantized with with a different
    scheme.

    targets in each config_group will be a list of either layer names
    (or regexes corresponding to layer names) or names of torch Modules.

    First, we try to match the layer_name with a target
    Second, we try to match the module's name with a target
    Third, we try to map the layer_name to a list of fused module names.
        *All* component module names must match in order for a match to be
        successful. A successful match returns the first component target

    :param layer_name: layer name
    :param module: torch.nn.Module
    :param targets: list of targets to match the layer against
    :param fused_mapping: map from fused layer names to its components
    :param fused_strategy: either "all" or "any". If using "all", fused
        layers match if "all" of its components match
    """

    if layer_name is None:
        layer_name = ""

    matched_target = (
        _find_first_match(layer_name, targets)
        or _find_first_match(module.__class__.__name__, targets, True)
        or _match_fused_layer(layer_name, targets, fused_mapping)
    )

    if matched_target is None:
        raise ValueError(
            f"Unable to find matching target for {layer_name} in the "
            "compressed-tensors config."
        )

    return matched_target


def _find_first_match(
    value: str, targets: Iterable[str], check_contains: bool = False
) -> str | None:
    """
    Returns first element of target that matches value either
    exactly or as a regex after 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.

    :param value: string to compare the list of targets against
    :param targets: list of targets to match the layer against
    :param check_contains: whether or not to do a substring match
    """

    for target in targets:
        if _is_equal_or_regex_match(value, target, check_contains=check_contains):
            return target
    return None


def _is_equal_or_regex_match(
    value: str, target: str, check_contains: bool = False
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    """

    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False


def _match_fused_layer(
    layer_name: str,
    target_layers: Iterable[str],
    fused_mapping: Mapping[str, list[str]],
) -> str | None:
    """
    Match a fused layer name to its corresponding individual layer in
    target_layers. Returns first value in fused_mapping which matches targets

    Implements an "all" matching strategy where a fused layer matches iff
    "all" of its components match

    :param layer_name: layer name
    :param target_layers: list of targets to match the layer against
    :param fused_mapping: map from fused layer names to its components

    Examples:
        layer_name = "model.layers.0.self_attn.qkv_proj"
        target_layers = ["model.layers.0.self_attn.q_proj",
                        "model.layers.0.self_attn.k_proj",
                        "model.layers.0.self_attn.v_proj"]
    """
    # find layer_name in mapping
    fused = next((key for key in fused_mapping if layer_name.endswith(key)), None)
    if fused is None:
        return None

    # expand path of unfused components
    unfused_paths = [
        layer_name.replace(fused, unfused) for unfused in fused_mapping[fused]
    ]

    # for each unfused component, find a match in targets
    unfused_matches: list[str | None] = []
    for unfused in unfused_paths:
        for target in target_layers:
            if _is_equal_or_regex_match(unfused, target):
                unfused_matches.append(target)
                break
        else:
            unfused_matches.append(None)

    return unfused_matches[0] if all(unfused_matches) else None


def convert_to_channelwise(
    weight_scale: torch.Tensor, logical_widths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Create channelwise buffer
    weight_scale_channel = torch.empty(
        (sum(logical_widths), 1), dtype=torch.float32, device=weight_scale.device
    )

    # Handle scalar tensor case: broadcast same scale to all channels
    if weight_scale.dim() == 0:
        weight_scale_channel.fill_(weight_scale.item())
        return weight_scale_channel

    # Expand each scale to match the size of each logical matrix.
    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        weight_scale_channel[start:end, :] = weight_scale[idx]
        start = end

    return weight_scale_channel


def update_tensor_inplace(old: torch.Tensor, new: torch.Tensor) -> None:
    old.copy_(new)


# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by TokenSpeed (and won't be freed)
def replace_parameter(
    mod: torch.nn.Module, name: str, new: torch.Tensor | torch.nn.Parameter
) -> None:

    old = getattr(mod, name)
    if (
        type(old) is type(new)
        and old.dtype == new.dtype
        and old.untyped_storage().nbytes() == new.untyped_storage().nbytes()
    ):
        # If we can just update in-place to avoid re-registering
        #   can be faster if the underlying storage is the same
        update_tensor_inplace(old, new)
    else:
        # Fallback re-register parameter, convert to Parameter if necessary
        # this not only ensures we don't register a tensor as a parameter, but
        # also ensures that all parameter subclasses get re-registered as
        # parameters for `torch.compile` compatibility
        if not isinstance(new, torch.nn.Parameter):
            new = torch.nn.Parameter(new, requires_grad=False)
        mod.register_parameter(name, torch.nn.Parameter(new, requires_grad=False))


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def unpack_cols(
    packed_q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    assert packed_q_w.shape == (
        size_k,
        size_n // pack_factor,
    ), f"packed_q_w.shape = {packed_q_w.shape} size_k = {size_k}, size_n = {size_n} pack_Factor = {pack_factor}"

    orig_device = packed_q_w.device

    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_cpu & mask
        packed_q_w_cpu >>= num_bits
        q_res[:, i::pack_factor] = vals

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def block_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    x_dq_block_tiles = [
        [
            x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block_tiles[j][i][:, :] = x_dq_block_tiles[j][i] * x_s[j][i]

    return x_dq_block
