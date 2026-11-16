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

"""Versioned MLA/grouped expert algebra, independent of model/runtime code."""

from dataclasses import replace
from math import prod

from tokenspeed_kernel.op_profile import OpCost


def _require_contract(op, *, metadata, inputs):
    missing = [name for name in metadata if name not in op.scalar_args]
    missing += [name for name in inputs if name not in op.inputs]
    if missing:
        raise ValueError(
            "Required versioned cost metadata unavailable: " + ", ".join(missing)
        )


def _element_bytes(description):
    from math import prod

    count = prod(description.shape)
    if (
        count <= 0
        or description.logical_bytes <= 0
        or description.logical_bytes % count
    ):
        raise ValueError("Invalid logical tensor byte geometry")
    return description.logical_bytes // count


def _portable_mla_compute_type(op):
    metadata = op.scalar_args
    keys = (
        "selected_kernel_names",
        "selected_solutions",
        "selected_compute_types",
        "selected_engine_contract_ids",
    )
    if any(not isinstance(metadata.get(key, ()), (tuple, list)) for key in keys):
        return None
    if op.inputs["q"].shape[1] != 1:
        return None
    if (
        tuple(metadata.get("selected_kernel_names", ()))
        == ("triton_mla_decode_with_kvcache",)
        and tuple(metadata.get("selected_solutions", ())) == ("triton",)
        and tuple(metadata.get("selected_compute_types", ())) == ("fp32_simt",)
        and tuple(metadata.get("selected_engine_contract_ids", ()))
        == ("mla.triton_decode.fp32_simt.v1",)
        and metadata.get("selected_compute_type") == "fp32_simt"
        and metadata.get("selected_engine_contract_id")
        == "mla.triton_decode.fp32_simt.v1"
    ):
        return "fp32_simt"
    return None


def mla_decode_cost_v2(op):
    """Full causal DCP1 MLA with explicit logical-request/actual-row mapping."""
    _require_contract(
        op,
        metadata=(
            "mla_logical_batch",
            "mla_query_width",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "mla_dcp_size",
            "mla_attention_tp_size",
            "window_left",
            "noncausal_block_size",
            "mla_cache_policy",
            "mla_layout",
            "mla_row_request_ids",
            "mla_row_query_positions",
            "return_lse",
        ),
        inputs=("q", "kv_cache", "page_table", "cache_seqlens"),
    )
    q, cache, table, sequence = (
        op.inputs[name] for name in ("q", "kv_cache", "page_table", "cache_seqlens")
    )
    a = op.scalar_args
    lengths = op.tensor_values.get("cache_seqlens")
    if lengths is None:
        raise ValueError(
            "Actual per-call MLA visible lengths unavailable; warmup values cannot be reused"
        )
    if len(q.shape) != 4 or len(cache.shape) != 4 or len(table.shape) != 2:
        raise ValueError("MLA actual tensor geometry unavailable")
    rows, query_axis_width, heads, dim = q.shape
    batch, width, latent = (
        a["mla_logical_batch"],
        a["mla_query_width"],
        a["kv_lora_rank"],
    )
    if any(
        type(value) is not int or value <= 0
        for value in (batch, width, latent, rows, query_axis_width, heads)
    ):
        raise ValueError("MLA positive static geometry required")
    if (
        a["mla_dcp_size"] != 1
        or a["mla_attention_tp_size"] < 1
        or a["window_left"] != -1
        or a["noncausal_block_size"] != 1
    ):
        raise ValueError("Only rank-local full causal DCP1 MLA is modeled")
    if a["mla_cache_policy"] != "logical_request_once.v2":
        raise ValueError("Explicit logical_request_once.v2 cache policy required")
    if cache.shape[2:] != (1, dim) or dim != latent + a["qk_rope_head_dim"]:
        raise ValueError("MLA latent/tail/cache dimensions disagree")
    if len(lengths) != rows or sequence.shape != (rows,) or table.shape[0] != rows:
        raise ValueError("MLA actual metadata row count mismatch")
    if any(
        type(length) is not int
        or length < query_axis_width
        or length > table.shape[1] * cache.shape[1]
        for length in lengths
    ):
        raise ValueError("MLA visible lengths exceed the actual row/page geometry")
    row_ids = tuple(a["mla_row_request_ids"])
    positions = tuple(a["mla_row_query_positions"])
    if a["mla_layout"] == "query_axis":
        if (
            (rows, query_axis_width) != (batch, width)
            or row_ids != tuple(range(batch))
            or positions != (0,) * batch
        ):
            raise ValueError("MLA query-axis logical mapping mismatch")
        final_lengths = tuple(lengths)
        attended = sum(width * length - width * (width - 1) // 2 for length in lengths)
    elif a["mla_layout"] == "flattened":
        if (
            (rows, query_axis_width) != (batch * width, 1)
            or row_ids != tuple(b for b in range(batch) for _ in range(width))
            or positions != tuple(j for _ in range(batch) for j in range(width))
        ):
            raise ValueError("MLA flattened logical request/query mapping mismatch")
        final_lengths = tuple(lengths[(b + 1) * width - 1] for b in range(batch))
        expected = tuple(
            length - width + j + 1 for length in final_lengths for j in range(width)
        )
        if tuple(lengths) != expected:
            raise ValueError(
                "MLA flattened rows do not carry the actual causal per-query lengths"
            )
        attended = sum(lengths)
    else:
        raise ValueError(
            "Unsupported MLA layout; actual logical request mapping required"
        )
    if any(length < width for length in final_lengths):
        raise ValueError("MLA lengths must include the complete current query block")
    tokens = batch * width
    flops = 2 * heads * (dim + latent) * attended
    reads = q.logical_bytes + _element_bytes(cache) * dim * sum(final_lengths)
    reads += sequence.logical_bytes + _element_bytes(table) * sum(
        (length + cache.shape[1] - 1) // cache.shape[1] for length in lengths
    )
    projection = op.inputs.get("value_weight")
    output = op.outputs.get("return", op.outputs.get("return[0]"))
    if output is None:
        raise ValueError("MLA returned boundary output unavailable")
    if projection is not None:
        if projection.shape[:2] != (heads, latent) or len(projection.shape) != 3:
            raise ValueError("MLA projected-value geometry mismatch")
        value = projection.shape[2]
        if output.shape != (tokens, heads * value) or a["return_lse"]:
            raise ValueError("MLA projected-value output boundary mismatch")
        flops += 2 * tokens * heads * latent * value
        reads += projection.logical_bytes
        gate = op.inputs.get("gate")
        if gate is not None:
            if gate.shape != output.shape:
                raise ValueError("MLA gate boundary mismatch")
            reads += gate.logical_bytes
    elif output.shape != (rows, query_axis_width, heads, latent):
        raise ValueError("MLA latent output boundary mismatch")
    writes = output.logical_bytes
    if a["return_lse"]:
        lse = op.outputs.get("return[1]")
        if (
            lse is None
            or _element_bytes(lse) != 4
            or lse.logical_bytes != 4 * tokens * heads
        ):
            raise ValueError("MLA LSE output boundary unavailable")
        writes += lse.logical_bytes
    return OpCost(
        flops,
        None,
        reads,
        writes,
        reads + writes,
        "diagnostic_mla_decode.v2",
        (
            "full causal DCP1; rank-local heads; useful QK and latent AV products; softmax/exp excluded",
            "logical_request_once.v2: cache union per original request across heads and query rows; metadata per actual interface row",
            "KV prologue writes and absorbed query/value projections outside this boundary are not counted twice",
            "optional enclosed value projection uses boundary IO; internal latent intermediates excluded",
            "actual selected leaf contract only; storage dtype does not identify execution engine",
            "logical IO estimate, not physical HBM traffic",
        ),
        compute_type=_portable_mla_compute_type(op) if projection is None else None,
    )


def mla_project_value_cost_v2(op):
    _require_contract(op, metadata=(), inputs=("attention", "weight"))
    attention, weight = op.inputs["attention"], op.inputs["weight"]
    if len(attention.shape) != 3 or len(weight.shape) != 3:
        raise ValueError("MLA value projection requires actual headwise geometry")
    tokens, heads, latent = attention.shape
    if weight.shape[:2] != (heads, latent) or op.outputs["return"].shape != (
        tokens,
        heads * weight.shape[2],
    ):
        raise ValueError("MLA value projection boundary mismatch")
    reads = attention.logical_bytes + weight.logical_bytes
    if "gate" in op.inputs:
        if op.inputs["gate"].shape != op.outputs["return"].shape:
            raise ValueError("MLA projection gate geometry mismatch")
        reads += op.inputs["gate"].logical_bytes
    writes = op.outputs["return"].logical_bytes
    return OpCost(
        2 * tokens * heads * latent * weight.shape[2],
        None,
        reads,
        writes,
        reads + writes,
        "diagnostic_mla_project_value.v2",
        (
            "headwise useful value projection matmul; gate nonlinearities excluded",
            "boundary input/weight/gate read and returned output write once",
            "torch.bmm or unproven selected leaf has unknown engine; storage dtype is not instruction evidence",
        ),
        compute_type=None,
    )


def grouped_experts_cost_v2(op):
    """Z0 global-ID, contiguous-owner, TP1 finalized grouped expert boundary."""
    _require_contract(
        op,
        metadata=(
            "route_policy",
            "expert_placement",
            "enable_eplb",
            "redundant_experts",
            "zero_experts",
            "all_to_all",
            "tp_size",
            "dcp_size",
            "do_finalize",
            "shared_projection",
            "hidden_dim",
            "useful_intermediate",
            "allocated_intermediate",
            "local_experts",
            "num_experts",
            "ep_size",
            "ep_rank",
            "group_count",
            "logical_tokens",
            "weight_format",
            "scale_block_shape",
        ),
        inputs=("hidden_states", "routes", "weights", "w13", "w2"),
    )
    a = op.scalar_args
    routes = op.tensor_values.get("routes")
    if routes is None:
        raise ValueError(
            "Actual per-call routing IDs unavailable; warmup values cannot be reused"
        )
    if (
        a["route_policy"] != "global_expert_id"
        or a["expert_placement"] != "trivial"
        or a["enable_eplb"]
        or a["redundant_experts"] != 0
    ):
        raise ValueError(
            "Unsupported route/placement policy; require trivial global IDs without redundancy/EPLB"
        )
    if (
        a["zero_experts"] != 0
        or a["all_to_all"]
        or a["tp_size"] != 1
        or a["dcp_size"] != 1
        or not a["do_finalize"]
        or a["shared_projection"]
    ):
        raise ValueError(
            "Only Z0 TP1 DCP1 finalized experts without all-to-all/shared fusion are modeled"
        )
    hidden, useful, allocated = (
        a["hidden_dim"],
        a["useful_intermediate"],
        a["allocated_intermediate"],
    )
    if useful is None or type(useful) is not int or not 0 < useful <= allocated:
        raise ValueError(
            "Useful pre-padding intermediate geometry unavailable or invalid"
        )
    local_experts, total = a["local_experts"], a["num_experts"]
    if local_experts * a["ep_size"] != total or not 0 <= a["ep_rank"] < a["ep_size"]:
        raise ValueError("Invalid contiguous EP ownership")
    x, ids, weights, w13, w2 = (
        op.inputs[n] for n in ("hidden_states", "routes", "weights", "w13", "w2")
    )
    if len(x.shape) != 2 or x.shape[1] != hidden or len(weights.shape) != 2:
        raise ValueError(
            "Expected live grouped input rows and rectangular top-k weights"
        )
    rows, topk = weights.shape
    groups = a["group_count"]
    if (
        groups is None
        or rows != groups * a["logical_tokens"]
        or x.shape[0] != rows
        or ids.shape != (rows * topk,)
        or len(routes) != rows * topk
    ):
        raise ValueError(
            "Live group-major row geometry unavailable; padded/compact routes unsupported"
        )
    if w13.shape != (local_experts, 2 * allocated, hidden) or w2.shape != (
        local_experts,
        hidden,
        allocated,
    ):
        raise ValueError("Actual linear expert weight geometry mismatch")
    if any(type(expert) is not int or not 0 <= expert < total for expert in routes):
        raise ValueError(
            "Only real Z0 global expert IDs are supported; placeholders/zero experts unavailable"
        )
    if total % groups or any(
        expert // (total // groups) != index // (topk * a["logical_tokens"])
        for index, expert in enumerate(routes)
    ):
        raise ValueError(
            "Global expert IDs do not match the live group-major row mapping"
        )
    first = a["ep_rank"] * local_experts
    local = [expert for expert in routes if first <= expert < first + local_experts]
    unique = len(set(local))
    reads = x.logical_bytes + ids.logical_bytes + weights.logical_bytes
    reads += (w13.logical_bytes + w2.logical_bytes) // local_experts * unique
    complete_io = True
    scale_bytes = 0
    if a["weight_format"] == "fp8":
        block = a["scale_block_shape"]
        if not isinstance(block, (tuple, list)) or len(block) != 2 or min(block) <= 0:
            complete_io = False
        else:
            n, k = block
            shapes = (
                (local_experts, 2 * ((allocated + n - 1) // n), (hidden + k - 1) // k),
                (local_experts, (hidden + n - 1) // n, (allocated + k - 1) // k),
            )
            for name, shape in zip(("scale13", "scale2"), shapes):
                scale = op.inputs.get(name)
                if scale is None:
                    complete_io = False
                elif scale.shape != shape:
                    raise ValueError("Actual FP8 expert scale geometry mismatch")
                else:
                    _element_bytes(scale)
                    scale_bytes += scale.logical_bytes // local_experts * unique
    elif a["weight_format"] != "unquant":
        raise ValueError(
            "Only linear unquantized or block-scaled FP8 expert weights are modeled"
        )
    reads += scale_bytes
    output = op.outputs.get("return")
    if output is None or output.shape != x.shape:
        raise ValueError(
            "Finalized expert output required; deferred/borrowed return tensors unsupported"
        )
    writes = output.logical_bytes
    assumptions = (
        "gate/up/down useful matmul FLOPs count local route instances; selected weights/scales count unique local experts",
        "useful intermediate is saved from constructor before padding; allocated intermediate comes from actual weight geometry",
        "global-ID contiguous EP ownership, Z0, expert TP1, DCP1, no all-to-all, group-major live rows, finalized output",
        "logical boundary input/route metadata/output once; allocated selected weights and actual scale descriptions once",
        "activation quantization, permutation, nonlinearities, internal buffers and communication excluded; not full MoE Layer cost",
        "actual planned apply leaf recorded; private AOT/weight dtype does not prove an execution engine",
        (
            "scale descriptions complete"
            if complete_io
            else "scale descriptions unavailable; logical byte subtotal incomplete and Estimated MBU unavailable"
        ),
    )
    return OpCost(
        6 * len(local) * hidden * useful,
        None,
        reads,
        writes,
        reads + writes if complete_io else None,
        "diagnostic_grouped_experts.v2",
        assumptions,
        compute_type=None,
    )


def mla_decode_cost(op, *, hbm_policy):
    a = dict(op.scalar_args)
    _require_contract(
        op,
        metadata=("logical_batch", "query_width", "attention_tp_size", "dcp_size"),
        inputs=("q",),
    )
    batch, width = a["logical_batch"], a["query_width"]
    if any(
        type(a[name]) is not int or a[name] < 1
        for name in ("logical_batch", "query_width", "attention_tp_size", "dcp_size")
    ):
        raise ValueError("Positive actual logical and parallel geometry required")
    shape = op.inputs["q"].shape[:2]
    if shape == (batch, width):
        layout, ids, positions = "query_axis", tuple(range(batch)), (0,) * batch
    elif shape == (batch * width, 1):
        layout, ids, positions = (
            "flattened",
            tuple(b for b in range(batch) for _ in range(width)),
            tuple(j for _ in range(batch) for j in range(width)),
        )
    else:
        raise ValueError(
            "Actual query geometry does not match captured logical request mapping"
        )
    a.update(
        mla_logical_batch=batch,
        mla_query_width=width,
        mla_attention_tp_size=a["attention_tp_size"],
        mla_dcp_size=a["dcp_size"],
        mla_cache_policy="logical_request_once.v2",
        mla_layout=layout,
        mla_row_request_ids=ids,
        mla_row_query_positions=positions,
    )
    if a["mla_attention_tp_size"] is None or a["mla_dcp_size"] is None:
        raise ValueError("Actual parallel mapping unavailable")
    cost = mla_decode_cost_v2(replace(op, scalar_args=a))
    return replace(
        cost,
        rule="mla_decode.v2",
        estimated_hbm_bytes=(
            cost.estimated_hbm_bytes if hbm_policy == "logical_io_once" else None
        ),
    )


def mla_projection_cost(op, *, hbm_policy):
    cost = mla_project_value_cost_v2(op)
    return replace(
        cost,
        rule="mla_project_value.v2",
        estimated_hbm_bytes=(
            cost.estimated_hbm_bytes if hbm_policy == "logical_io_once" else None
        ),
    )


def grouped_expert_cost(op, *, hbm_policy):
    names = {
        "hidden_dim": "self.hidden_size",
        "useful_intermediate": "self.profile_useful_intermediate_size",
        "num_experts": "self.num_experts",
        "local_experts": "self.num_local_experts",
        "ep_rank": "self.ep_rank",
        "ep_size": "self.ep_size",
        "tp_size": "self.tp_size",
        "zero_experts": "self.zero_expert_num",
        "redundant_experts": "self.ep_num_redundant_experts",
        "weight_format": "self._quant_kind",
        "scale_block_shape": "self.profile_scale_block_shape",
        "all_to_all": "self.plan.supports_all_to_all_ep",
        "group_count": "moe_group_count",
        "route_policy": "moe_route_policy",
        "expert_placement": "moe_expert_placement",
        "enable_eplb": "moe_enable_eplb",
    }
    _require_contract(
        op,
        metadata=tuple(names.values()) + ("logical_batch", "dcp_size", "do_finalize"),
        inputs=(
            "self.w13_weight",
            "self.w2_weight",
            "topk_output.topk_ids",
            "topk_output.topk_weights",
            "hidden_states",
        ),
    )
    a = {
        **op.scalar_args,
        **{name: op.scalar_args[source] for name, source in names.items()},
    }
    batch, width = a["logical_batch"], a.get("query_width")
    logical_tokens = a.get("logical_tokens")
    if logical_tokens is None or width is not None:
        if type(batch) is not int or type(width) is not int or batch < 1 or width < 1:
            raise ValueError("Actual logical batch/query width unavailable")
        if logical_tokens is not None and logical_tokens != batch * width:
            raise ValueError("Live token count differs from logical batch/query width")
        logical_tokens = batch * width
    aliases = {
        "w13": "self.w13_weight",
        "w2": "self.w2_weight",
        "routes": "topk_output.topk_ids",
        "weights": "topk_output.topk_weights",
        "scale13": "self.w13_weight_scale_inv",
        "scale2": "self.w2_weight_scale_inv",
    }
    inputs = {
        **op.inputs,
        **{
            name: op.inputs[source]
            for name, source in aliases.items()
            if source in op.inputs
        },
    }
    if (
        any(
            type(a[name]) is not int or a[name] < 1
            for name in (
                "logical_batch",
                "hidden_dim",
                "num_experts",
                "local_experts",
                "tp_size",
                "ep_size",
                "dcp_size",
            )
        )
        or type(a["ep_rank"]) is not int
        or type(logical_tokens) is not int
        or logical_tokens < 1
    ):
        raise ValueError("Actual logical/expert parallel geometry unavailable")
    if len(inputs["w2"].shape) != 3:
        raise ValueError("Actual expert weight geometry unavailable")
    ids = inputs["routes"]
    if len(ids.shape) != 2 or ids.stride != (ids.shape[1], 1):
        raise ValueError("Only contiguous live route matrices are modeled")
    inputs["routes"] = replace(ids, shape=(prod(ids.shape),), stride=(1,))
    a.update(
        allocated_intermediate=inputs["w2"].shape[-1],
        logical_tokens=logical_tokens,
        shared_projection="shared_input" in op.inputs,
    )
    values = {"routes": op.tensor_values.get("topk_output.topk_ids")}
    cost = grouped_experts_cost_v2(
        replace(op, inputs=inputs, scalar_args=a, tensor_values=values)
    )
    return replace(
        cost,
        rule="grouped_experts.v2",
        estimated_hbm_bytes=(
            cost.estimated_hbm_bytes if hbm_policy == "logical_io_once" else None
        ),
    )


def register_graph_costs(registry, *, hbm_policy):
    registry.register(
        "mla_decode.v2", lambda op: mla_decode_cost(op, hbm_policy=hbm_policy)
    )
    registry.register(
        "mla_project_value.v2",
        lambda op: mla_projection_cost(op, hbm_policy=hbm_policy),
    )
    registry.register(
        "grouped_experts.v2", lambda op: grouped_expert_cost(op, hbm_policy=hbm_policy)
    )
