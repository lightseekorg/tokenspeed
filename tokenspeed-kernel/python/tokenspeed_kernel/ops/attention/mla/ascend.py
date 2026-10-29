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

"""Ascend MLA kernel registrations."""

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures


def ascend_mla_dcp_kernels():
    """Return the Ascend packed MLA partial/merge and graph-length adapter."""
    if not current_platform().is_npu:
        raise RuntimeError("Ascend MLA DCP kernels require an NPU platform")
    from tokenspeed_kernel_npu.ops import mla_dcp

    return mla_dcp


def prepare_ascend_mla_prefill(
    cache,
    *,
    ranks,
    bootstrap_group,
    max_current_rows,
    max_history_rows=0,
    history_null_rows=0,
    workspace=None,
):
    """Reuse the DSA AICPU gather/owner-commit transport with no Index-K plane."""
    from tokenspeed_kernel_npu.ops.dsa import AscendDSAPrefill

    return AscendDSAPrefill(
        cache,
        None,
        ranks=ranks,
        bootstrap_group=bootstrap_group,
        max_current_rows=max_current_rows,
        max_history_rows=max_history_rows,
        history_null_rows=history_null_rows,
        workspace=workspace,
    )


def pack_mla_panels(weight: torch.Tensor, *, width: int) -> torch.Tensor:
    """Bit-preserving CPU NK panels, carried by a genuine NZ16-row tensor."""
    if weight.device.type != "cpu" or weight.dtype != torch.bfloat16:
        raise ValueError("MLA panel packing requires CPU BF16 weights")
    if width <= 0 or width % 16:
        raise ValueError("MLA N tiles must be positive multiples of 16")
    panel = 256 if weight.ndim == 2 and weight.shape[1] == 768 else 512
    if weight.ndim != 2 or weight.shape[0] % width or weight.shape[1] % panel:
        raise ValueError(
            "MLA weights must cover complete N tiles and complete K panels"
        )
    panels = []
    for col in range(0, weight.shape[0], width):
        for start in range(0, weight.shape[1], panel):
            tile = weight[col : col + width, start : start + panel]
            panels.append(
                tile.reshape(width // 16, 16, panel // 16, 16)
                .permute(2, 0, 1, 3)
                .contiguous()
                .flatten()
            )
    packed = torch.cat(panels)
    return packed.reshape(-1, 16, 16).permute(1, 0, 2).contiguous().reshape(16, -1)


def _prepare_mla_prolog_weights(layer, *, group):
    """Pack fixed BF16 weights once, before any forward or graph capture."""
    if (
        (layer.num_heads, layer.num_local_heads, layer.q_lora_rank)
        not in ((64, 8, 1536), (32, 4, 768))
        or layer.use_output_gate
        or layer.q_lora_scale is not None
        or layer.kv_lora_scale is not None
    ):
        raise ValueError(
            "Fused MLA requires H64/TP8 or H32/TP8 without output gate/runtime LoRA scaling"
        )
    down = layer.fused_qkv_a_proj_with_mqa.weight.detach()
    qb = layer.q_b_proj.weight.detach()
    hidden = 4096 if layer.num_heads == 64 else 2048
    if down.shape != (layer.q_lora_rank + 576, hidden) or qb.shape != (
        layer.num_local_heads * 192,
        layer.q_lora_rank,
    ):
        raise ValueError("Unsupported MLA projection geometry")
    if down.dtype != torch.bfloat16 or qb.dtype != torch.bfloat16:
        raise ValueError("Fused MLA requires unquantized BF16 projections")

    def nz(weight, width):
        import torch_npu

        carrier = pack_mla_panels(weight.cpu(), width=width).to(down.device)
        try:
            torch.npu.config.allow_internal_format = True
            result = torch_npu.npu_format_cast(carrier, 29)
        finally:
            torch.npu.config.allow_internal_format = False
        if torch_npu.get_npu_format(result) != 29:
            raise RuntimeError("MLA panel carrier must have genuine NZ storage")
        return result

    pieces = [torch.empty_like(layer.w_kc) for _ in range(8)]
    dist.all_gather(pieces, layer.w_kc.contiguous(), group=group)
    if layer.rotary_emb is not None:
        if layer.rotary_emb.is_neox_style or layer.rotary_emb.rotary_dim != 64:
            raise ValueError("Fused MLA supports GPT-J rotary64 only")
        cos_sin = layer.rotary_emb.cos_sin_cache.float().contiguous()
    else:
        cos_sin = torch.zeros((1, 64), dtype=torch.float32, device=down.device)
    return dict(
        local_heads=layer.num_local_heads,
        down=nz(down, 96),
        qb=nz(qb, 128),
        uk=torch.cat(pieces, 0),
        gamma_q=layer.q_a_layernorm.weight.detach().contiguous(),
        gamma_kv=layer.kv_a_layernorm.weight.detach().contiguous(),
        eps_q=layer.q_a_layernorm.variance_epsilon,
        eps_kv=layer.kv_a_layernorm.variance_epsilon,
        rotary_mode=int(layer.rotary_emb is not None),
        cos_sin=cos_sin,
    )


def ascend_mla_prolog(
    x, *, weights, rank, window, query, cache, slots, qnorm, latent, positions
):
    """Project, exchange QB, absorb and store owner-local BF16 latent KV."""
    torch.ops.custom.npu_mla_prolog_fused(
        x,
        weights["down"],
        weights["gamma_q"],
        weights["gamma_kv"],
        weights["qb"],
        weights["uk"],
        window,
        query,
        cache,
        slots,
        qnorm,
        latent,
        positions,
        weights["cos_sin"],
        weights["eps_q"],
        weights["eps_kv"],
        rank * weights["local_heads"],
        weights["rotary_mode"],
    )


def ascend_mla_attention(query, *, cache, table, lengths, scale, out, lse):
    """Run packed FIA into FP32 storage using the existing graph length binding."""
    tokens, heads, _ = query.shape
    output, statistics = torch.ops.tokenspeed_npu.dcp_mla_query.out(
        query,
        cache.view(cache.shape[0], cache.shape[1], 576),
        table,
        lengths.key,
        lengths.values,
        num_heads=heads,
        scale=scale,
        out=out,
        lse=lse,
    )
    return output.view(tokens, heads, 512), statistics.view(tokens, heads)


def ascend_mla_merge(
    output, *, lse, visible, active, producer, maximum, total, peers, group
):
    """Pack normalized FIA partials once, then reuse DSA's direct LSE merge."""
    from tokenspeed_kernel_npu.ops.mla_dcp import prepare_mla_partials

    prepare_mla_partials(output, lse, visible, active, producer, maximum)
    return torch.ops.custom.npu_dsa_merge_lse_direct(
        producer, maximum, total, peers, group, 8, 0
    )


def ascend_prepare_mla_resources(*, group, rank, heads, rank_size):
    """Retain checked operator views of this MLA group's HCCL window."""
    from tokenspeed_kernel.ops.attention.dsa import DSAResources
    from tokenspeed_kernel.ops.attention.dsa.ascend import ascend_dsa_kernels

    ascend_dsa_kernels()
    window = torch.ops.custom.mla_prepare_prolog_window(group, rank, heads, rank_size)
    try:
        prepared = {
            rows: torch.ops.custom.mla_prepare_lse_buffers(
                group, rows, rank, heads, rank_size
            )
            for rows in (
                (32, 64, 128, 256, 512) if heads == 32 else (64, 128, 256, 512)
            )
        }
    except BaseException:
        torch.ops.custom.dsa_release_prolog_window(window)
        raise
    return DSAResources(
        window,
        {rows: values[:3] for rows, values in prepared.items()},
        {rows: values[3] for rows, values in prepared.items()},
        release=torch.ops.custom.dsa_release_prolog_window,
    )


if current_platform().is_npu:
    register_kernel(
        "attention",
        "mla_prepare_resources",
        name="ascend_prepare_mla_resources",
        solution="native",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=[format_signature()],
        priority=Priority.PERFORMANT,
    )(ascend_prepare_mla_resources)

    from tokenspeed_kernel_npu.ops.mla import (
        mla_decode_with_kvcache as _mla_decode_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mla import (
        mla_extend_with_kvcache as _mla_extend_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mla import (
        mla_normalize_project_query as _mla_normalize_project_query,
    )
    from tokenspeed_kernel_npu.ops.mla import mla_prefill as _mla_prefill
    from tokenspeed_kernel_npu.ops.mla import mla_project_value as _mla_project_value

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    _DTYPES = {torch.float16, torch.bfloat16}

    @register_kernel(
        "attention",
        "mla_normalize_project_query",
        name="ascend_mla_normalize_project_query",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(
            ("query", "kv", "projection_weight", "out"), "dense", _DTYPES
        ),
        priority=Priority.PERFORMANT,
        traits={"split_output": frozenset({False})},
        features={"portability"},
    )
    def mla_normalize_project_query(**kwargs):
        return _mla_normalize_project_query(**kwargs)

    @register_kernel(
        "attention",
        "mla_project_value",
        name="ascend_mla_project_value",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("attention", "weight", "out"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            "gate_kind": frozenset({"none", "sigmoid"}),
            "inputs_contiguous": frozenset({True}),
        },
        features={"portability"},
    )
    def mla_project_value(**kwargs):
        return _mla_project_value(**kwargs)

    @register_kernel(
        "attention",
        "mla_prefill",
        name="ascend_mla_prefill",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            "is_causal": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
        features={"portability"},
    )
    def mla_prefill(**kwargs):
        return _mla_prefill(**kwargs)

    @register_kernel(
        "attention",
        "mla_extend_with_kvcache",
        name="ascend_mla_extend_with_kvcache",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "kv_cache"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            "page_size": frozenset({64, 128}),
            "qk_nope_head_dim": frozenset({128}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "is_causal": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
        features={"portability"},
    )
    def mla_extend_with_kvcache(**kwargs):
        return _mla_extend_with_kvcache(**kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="ascend_mla_decode_with_kvcache",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "kv_cache"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            "page_size": frozenset({64, 128}),
            "q_len": frozenset({1}),
            "qk_nope_head_dim": frozenset({128}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
            "sliding_window": frozenset({False}),
            "block_on_query_axis": frozenset({True}),
        },
        features={"portability"},
    )
    def mla_decode_with_kvcache(**kwargs):
        return _mla_decode_with_kvcache(**kwargs)

    for operation, function, roles, dtype, preprocessor in (
        (
            "mla_fused_prolog",
            ascend_mla_prolog,
            "x",
            torch.bfloat16,
            _prepare_mla_prolog_weights,
        ),
        ("mla_fused_attention", ascend_mla_attention, "query", torch.bfloat16, None),
        ("mla_fused_merge", ascend_mla_merge, "output", torch.float32, None),
    ):
        register_kernel(
            "attention",
            operation,
            name=function.__name__,
            solution="native",
            capability=_CAPABILITY,
            signatures=format_signatures(roles, "dense", {dtype}),
            priority=Priority.PERFORMANT,
            weight_preprocessor=preprocessor,
            features={"native", "npu_graph"},
        )(function)


__all__ = [
    "mla_decode_with_kvcache",
    "mla_extend_with_kvcache",
    "mla_normalize_project_query",
    "mla_prefill",
    "mla_project_value",
]
