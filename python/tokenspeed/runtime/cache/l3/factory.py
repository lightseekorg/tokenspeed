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

"""Resolve the L3 tier's settings from startup configuration and CPU facts.

The engine host runs the tier (``ts-l3``): this module only freezes what it
starts from -- the deployment identity, the Host pool's geometry and the
store client's settings (``L3Settings``).
"""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING, Any

from tokenspeed.runtime.cache.l3.config import (
    L3Settings,
    MooncakeStoreSettings,
    parse_extra_config,
)
from tokenspeed.runtime.utils.env import envs

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from tokenspeed.runtime.cache.host_storage import HostCacheResources
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.utils.server_args import ServerArgs


def _kv_producer_options(args: ServerArgs, *, draft_dtype: str) -> dict[str, Any]:
    """Snapshot numerical and partition choices for the deployment identity.

    Target dtype is already part of the packed-cache signature. An empty
    draft dtype means this stage has no draft. Storage admission knobs and
    output-only options do not change the KV produced for fixed tokens.
    """
    mapping = args.mapping
    assert mapping is not None
    return {
        "draft_dtype": draft_dtype,
        "numerics": args.numerics,
        "yarn_ramp_mask_device": args.yarn_ramp_mask_device,
        "mla_lora_scale": args.mla_lora_scale,
        "layer_boundary_norm": args.layer_boundary_norm,
        "router_topk": args.router_topk,
        "moe_combine_order": args.moe_combine_order,
        "dsa_slot_order": args.dsa_slot_order,
        "disable_autotune": args.disable_autotune,
        "disable_tf32": args.disable_tf32,
        "disable_pdl": args.disable_pdl,
        "force_deterministic_rsag": args.force_deterministic_rsag,
        "batch_invariant_collectives": args.batch_invariant_collectives,
        "tp_batch_invariant": args.tp_batch_invariant,
        "enable_allreduce_fusion": args.enable_allreduce_fusion,
        "comm_fusion_max_num_tokens": args.comm_fusion_max_num_tokens,
        "disable_nccl_nvls": args.disable_nccl_nvls,
        "disable_symm_mem": args.disable_symm_mem,
        "dense_gemm_backend": args.dense_gemm_backend,
        "moe_backend": args.moe_backend,
        "draft_moe_backend": (
            (args.draft_moe_backend or args.moe_backend) if draft_dtype else ""
        ),
        "moe_mxfp4_fp8_activation": args.moe_mxfp4_fp8_activation,
        "all2all_backend": args.all2all_backend,
        "deepep_mode": args.deepep_mode,
        "attn_head_tp_size": mapping.attn.head_tp_size,
        "attn_kvp_size": mapping.attn.kvp_size,
        "attn_qcp_size": mapping.attn.qcp_size,
        "linear_attn_tp_size": mapping.linear_attn.tp_size,
        "dense_tp_size": mapping.dense.tp_size,
        "moe_tp_size": mapping.moe.tp_size,
        "moe_ep_size": mapping.moe.ep_size,
    }


def mooncake_store_library() -> str:
    """The Mooncake store library the engine host loads:
    ``TOKENSPEED_MOONCAKE_STORE_LIBRARY``, else the store's C ABI the
    tokenspeed-mooncake wheel installs."""
    override = envs.TOKENSPEED_MOONCAKE_STORE_LIBRARY.get()
    if override:
        return override
    spec = importlib.util.find_spec("mooncake")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "Please install tokenspeed-mooncake to use --kvstore-storage-backend "
            "mooncake."
        )
    path = os.path.join(spec.submodule_search_locations[0], "libmooncake_store.so")
    if not os.path.exists(path):
        raise RuntimeError(
            f"{path} is missing: the L3 store needs a tokenspeed-mooncake build "
            "with the store's C ABI"
        )
    return path


def l3_settings(
    server_args: ServerArgs,
    model_config: ModelConfig,
    draft_model_config: ModelConfig | None,
    *,
    host_cache: HostCacheResources | None,
    world_cpu_group: ProcessGroup | None,
) -> L3Settings | None:
    """The settings the engine host starts this rank's L3 executor from.

    Args:
        server_args: Resolved launch settings; numerical choices are snapshotted.
        model_config: Target checkpoint configuration, without model weights.
        draft_model_config: Draft checkpoint configuration, or None.
        host_cache: Shared Host resources and resolved CPU cache facts.
            None when device construction produced no Host L2 allocation.
        world_cpu_group: The existing Gloo world group for checkpoint agreement;
            None in a single-process deployment.

    Returns:
        The settings, with the Host pool by address: the caller keeps
        ``host_cache.storage`` for the process. None without a storage
        backend; then no collective runs.
    """
    if server_args.kvstore_storage_backend is None:
        return None
    backend = server_args.kvstore_storage_backend.strip().lower()
    if backend not in ("memory", "mooncake"):
        raise ValueError(
            "unsupported KVStore storage backend "
            f"{server_args.kvstore_storage_backend!r}"
        )
    if host_cache is None:
        raise ValueError("L3 construction requires a Host L2 allocation")
    host_storage = host_cache.storage
    ownership = host_cache.ownership
    cache_fields = tuple(
        tuple(field.field_id for field in group.fields) for group in host_cache.groups
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.transfer import (
        build_cache_transfer_schema,
    )

    transfer_schema = build_cache_transfer_schema(
        host_cache.memory_plan,
        model_config=model_config,
        draft_model_config=draft_model_config,
    )
    if len(cache_fields) != len(ownership.shard_counts):
        raise ValueError("L3 field groups and Host ownership must agree")
    for fields, shards in zip(cache_fields, ownership.shard_counts, strict=True):
        if shards > 1:
            for field_id in fields:
                if transfer_schema.partition_for(field_id) is not None:
                    raise ValueError(
                        "L3 page sharding requires replicated field payloads; "
                        f"{field_id} is head-partitioned"
                    )

    from tokenspeed.runtime.utils.server_args import AUX_CAPTURE_SPEC_ALGORITHMS

    eagle3_layers_to_capture: tuple[int, ...] = ()
    if server_args.speculative_algorithm in AUX_CAPTURE_SPEC_ALGORITHMS:
        configured_layers = server_args.eagle3_layers_to_capture
        if configured_layers:
            eagle3_layers_to_capture = tuple(int(layer) for layer in configured_layers)
        elif draft_model_config is not None:
            from tokenspeed.runtime.execution.factory import _eagle_aux_layer_ids

            draft_layers = _eagle_aux_layer_ids(draft_model_config.hf_config)
            if draft_layers:
                eagle3_layers_to_capture = tuple(int(layer) for layer in draft_layers)
    attention_backend_name = host_cache.attention_backend_name
    draft_attention_backend_name = host_cache.draft_attention_backend_name
    from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig
    from tokenspeed.runtime.cache.l3.identity import cache_layout_signature

    cache_signature = cache_layout_signature(
        host_cache,
        cache_dtype=f"{server_args.kv_cache_dtype}:{model_config.dtype}",
    )
    config = StorageRuntimeConfig.from_extra_config(
        server_args.kvstore_storage_backend_extra_config,
        host_blocks=host_storage.num_host_lcm_blocks,
        prefix_granularity=host_cache.memory_plan.prefix_granularity,
    )
    import torch.distributed as dist

    from tokenspeed.runtime.cache.l3.identity import (
        L3_RUNTIME_COMPAT,
        l3_cache_quantization_id,
        l3_checkpoint_id,
        share_l3_checkpoint_ids,
        storage_deployment_id,
    )

    initialized = dist.is_available() and dist.is_initialized()
    if world_cpu_group is not None:
        if not initialized or dist.get_backend(world_cpu_group) != "gloo":
            raise ValueError("L3 checkpoint agreement requires a Gloo CPU group")
        world_size = dist.get_world_size(world_cpu_group)
        rank = dist.get_rank(world_cpu_group)
    else:
        if initialized and dist.get_world_size() > 1:
            raise ValueError("L3 checkpoint agreement requires an explicit CPU group")
        world_size, rank = 1, 0
    checkpoint_id = l3_checkpoint_id(
        model_config.model_path,
        hf_config=model_config.hf_config,
        revision=str(model_config.revision or ""),
        load_format=str(server_args.load_format),
        # LoadConfig currently gets the same empty extra-config;
        # both must stay aligned if a shard pattern is wired through.
        model_loader_extra_config={},
        ext_yaml=str(server_args.ext_yaml or ""),
    )
    if draft_model_config is not None:
        draft_revision = l3_checkpoint_id(
            draft_model_config.model_path,
            hf_config=draft_model_config.hf_config,
            revision=str(draft_model_config.revision or ""),
            load_format=str(server_args.load_format),
            model_loader_extra_config={},
            ext_yaml=str(server_args.ext_yaml or ""),
        )
    else:
        draft_revision = ""

    def gather_checkpoint_ids(payload: list) -> list:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload, group=world_cpu_group)
        return gathered

    checkpoint_id, draft_revision = share_l3_checkpoint_ids(
        [checkpoint_id, draft_revision],
        rank=rank,
        world_size=world_size,
        gather=gather_checkpoint_ids,
    )
    pipeline_rank = server_args.mapping.pp_rank if server_args.mapping.has_pp else 0
    if draft_model_config is not None:
        draft_model = str(draft_model_config.model_path)
        draft_quantization = str(draft_model_config.quantization or "")
    else:
        draft_model = ""
        draft_quantization = ""
    cache_quantization = l3_cache_quantization_id(
        quantization=str(model_config.quantization or ""),
        quantization_param_path=str(server_args.quantization_param_path or ""),
        draft_quantization=draft_quantization,
    )
    deployment_id = storage_deployment_id(
        str(server_args.model),
        revision=checkpoint_id,
        model_overrides=model_config.model_override_args,
        cache_signature=cache_signature,
        pipeline_rank=pipeline_rank,
        attn_tp_size=int(server_args.mapping.attn.tp_size),
        group_shard_counts=ownership.shard_counts,
        draft_model=draft_model,
        draft_revision=draft_revision,
        cache_quantization=cache_quantization,
        runtime_compat=L3_RUNTIME_COMPAT,
        attention_backend=attention_backend_name,
        draft_attention_backend=draft_attention_backend_name,
        skip_softmax_threshold=float(server_args.skip_softmax_threshold),
        eagle3_layers_to_capture=eagle3_layers_to_capture,
        kv_producer=_kv_producer_options(
            server_args,
            draft_dtype=(
                str(draft_model_config.dtype) if draft_model_config is not None else ""
            ),
        ),
    )

    pp_size = server_args.mapping.pp_size if server_args.mapping.has_pp else 1
    return L3Settings(
        runtime=config,
        deployment_id=deployment_id,
        has_draft=bool(draft_model),
        weight_version=str(server_args.weight_version),
        host_buffer=int(host_storage.host_buffer.data_ptr()),
        host_buffer_bytes=int(host_storage.host_buffer_bytes),
        groups=tuple(
            (int(packing), int(block_bytes), int(shards))
            for packing, block_bytes, shards in zip(
                host_storage.group_packing,
                host_storage.host_cache_block_bytes,
                ownership.shard_counts,
                strict=True,
            )
        ),
        lcm_block_bytes=int(host_storage.host_lcm_block_bytes),
        num_lcm_blocks=int(host_storage.num_host_lcm_blocks),
        owner_rank=int(ownership.rank),
        tp_rank=int(server_args.mapping.attn.tp_rank),
        backend=backend,
        mooncake=(
            MooncakeStoreSettings.resolve(
                parse_extra_config(server_args.kvstore_storage_backend_extra_config),
                library=mooncake_store_library(),
                tp_size=server_args.mapping.attn.tp_size,
                pp_size=pp_size,
            )
            if backend == "mooncake"
            else None
        ),
    )
