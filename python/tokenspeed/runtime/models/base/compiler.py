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

"""Layer Compiler: analyses ModuleSpec annotations and inserts CommOps.

The compiler inspects each decoder layer's sub-modules (in the order declared
by ``resolve_exec_plan``), examines adjacent Placement pairs, and inserts the
minimal set of communication operations to transition between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.base.comm_ops import (
    AllGatherOp,
    AllReduceOp,
    CommOp,
    DeferredReduceOp,
    FusedReduceNormOp,
    ReduceScatterOp,
    ResidualAllGatherOp,
    ResidualSliceOp,
    _scattered_num_tokens_all,
)
from tokenspeed.runtime.models.base.execution import (
    CompiledDecoderLayer,
    ExecutionNode,
    ExecutionState,
    ExecutionStep,
    StepRunner,
)
from tokenspeed.runtime.models.base.module_spec import (
    CallConvention,
    FusionCapability,
    ModuleKind,
    ModuleSpec,
)
from tokenspeed.runtime.models.base.placement import (
    ParallelGroup,
    Partial,
    Placement,
    PlacementType,
    Replicate,
    Shard,
    can_fuse_reduce_norm,
    group_has_parallel,
    use_all_reduce,
)


@dataclass
class _TrackedState:
    hidden: Placement | None = None
    residual: Placement | None = None


# ---------------------------------------------------------------------------
# Runner factories — dispatch on CallConvention
# ---------------------------------------------------------------------------


def _runner_from_node(node: ExecutionNode) -> StepRunner:
    module = node.module
    spec = node.spec

    if isinstance(module, FusedReduceNormOp):
        return lambda state, positions: _run_fused_reduce_norm(module, state)
    if spec.call == CallConvention.NORM_WITH_OPTIONAL_RESIDUAL:
        return lambda state, positions: _run_norm(module, state)
    if spec.call == CallConvention.ATTENTION:
        return lambda state, positions: _run_attention(module, state, positions)
    if spec.call == CallConvention.MOE:
        return lambda state, positions: _run_moe(module, state)
    return lambda state, positions: _run_hidden_states_only(module, state)


# ---------------------------------------------------------------------------
# Per-convention runner functions
# ---------------------------------------------------------------------------


def _run_fused_reduce_norm(
    module: FusedReduceNormOp,
    state: ExecutionState,
) -> ExecutionState:
    hidden_states, residual = module(state.hidden_states, state.residual, state.ctx)
    return ExecutionState(hidden_states, residual, state.ctx, state.out_cache_loc)


def _run_norm(module: nn.Module, state: ExecutionState) -> ExecutionState:
    if state.residual is not None:
        hidden_states, residual = module(state.hidden_states, state.residual)
    else:
        residual = state.hidden_states
        hidden_states = module(state.hidden_states)
    return ExecutionState(hidden_states, residual, state.ctx, state.out_cache_loc)


def _run_attention(
    module: nn.Module,
    state: ExecutionState,
    positions: torch.Tensor,
) -> ExecutionState:
    hidden_states = module(
        positions=positions,
        hidden_states=state.hidden_states,
        ctx=state.ctx,
        out_cache_loc=state.out_cache_loc,
    )
    return ExecutionState(hidden_states, state.residual, state.ctx, state.out_cache_loc)


def _run_moe(module: nn.Module, state: ExecutionState) -> ExecutionState:
    scattered = _scattered_num_tokens_all(state.ctx, module.mapping)
    num_global_tokens = sum(scattered)
    max_num_tokens_per_gpu = max(scattered) if scattered else 0
    hidden_states = module(
        state.hidden_states, num_global_tokens, max_num_tokens_per_gpu
    )
    return ExecutionState(hidden_states, state.residual, state.ctx, state.out_cache_loc)


def _run_hidden_states_only(module: nn.Module, state: ExecutionState) -> ExecutionState:
    hidden_states = module(state.hidden_states)
    return ExecutionState(hidden_states, state.residual, state.ctx, state.out_cache_loc)


# ---------------------------------------------------------------------------
# Placement helpers
# ---------------------------------------------------------------------------


def _input_group(spec: ModuleSpec) -> ParallelGroup | None:
    return spec.input_placement.group if spec.input_placement else None


def _output_group(spec: ModuleSpec) -> ParallelGroup | None:
    return spec.output_placement.group if spec.output_placement else None


def _find_last_compute_index(exec_plan: list[ExecutionNode]) -> int:
    """Find the index of the last compute (non-NORM) module in exec_plan."""
    last_idx = -1
    for i, mod in enumerate(exec_plan):
        spec = mod.spec
        if spec.kind != ModuleKind.NORM:
            last_idx = i
    return last_idx


# ---------------------------------------------------------------------------
# Main compilation entry point
# ---------------------------------------------------------------------------


def compile_decoder_layer(
    layer: nn.Module,
    exec_plan: list[ExecutionNode],
    mapping: Mapping,
    prev_layer_output_group: ParallelGroup | None = None,
    next_layer_input_group: ParallelGroup | None = None,
) -> CompiledDecoderLayer:
    """Analyse a decoder layer execution plan and produce a CompiledDecoderLayer."""

    last_compute_idx = _find_last_compute_index(exec_plan)

    steps: list[ExecutionStep] = []

    first_compute_input_group = find_first_compute_input_group(exec_plan)
    state = _TrackedState(
        hidden=_initial_hidden_placement(
            mapping=mapping,
            prev_layer_output_group=prev_layer_output_group,
            first_compute_input_group=first_compute_input_group,
        ),
        residual=_initial_residual_placement(
            mapping=mapping,
            prev_layer_output_group=prev_layer_output_group,
            first_compute_input_group=first_compute_input_group,
        ),
    )

    is_first_layer = prev_layer_output_group is None

    for mod_idx, node in enumerate(exec_plan):
        spec = node.spec
        is_last_compute = mod_idx == last_compute_idx

        if spec.kind == ModuleKind.NORM:
            step = _compile_norm_step(
                node=node,
                mapping=mapping,
                mod_idx=mod_idx,
                exec_plan=exec_plan,
                state=state,
            )
            steps.append(step)
        else:
            step = _compile_compute_step(
                node=node,
                mapping=mapping,
                mod_idx=mod_idx,
                exec_plan=exec_plan,
                is_last_compute=is_last_compute,
                state=state,
                next_layer_input_group=next_layer_input_group,
                is_first_layer=is_first_layer,
            )
            steps.append(step)

    # Determine the final placement after this layer.
    final_placement = _compute_final_placement(state, mapping)

    return CompiledDecoderLayer(
        steps=steps,
        final_placement=final_placement,
        mapping=mapping,
    )


# ---------------------------------------------------------------------------
# Per-step compilation
# ---------------------------------------------------------------------------


def _compile_norm_step(
    node: ExecutionNode,
    mapping: Mapping,
    mod_idx: int,
    exec_plan: list[ExecutionNode],
    state: _TrackedState,
) -> ExecutionStep:
    """Compile a NORM step."""
    module = node.module
    spec = node.spec
    next_compute_group = _find_next_compute_input_group(exec_plan, mod_idx)
    hidden = state.hidden
    src_group = hidden.group if hidden is not None else None
    prev_output_is_partial = hidden is not None and hidden.type == PlacementType.PARTIAL

    if (
        prev_output_is_partial
        and spec.fusion == FusionCapability.REDUCE_NORM
        and src_group is not None
        and next_compute_group is not None
        and can_fuse_reduce_norm(mapping, src_group, next_compute_group)
    ):
        fused_norm = FusedReduceNormOp(mapping, src_group, module)
        state.hidden = Replicate(src_group)
        fused_node = ExecutionNode(
            module=fused_norm,
            spec=spec,
            name=node.name,
        )
        return ExecutionStep(
            runner=_runner_from_node(fused_node),
            module=fused_norm,
            spec=spec,
            kind=spec.kind,
            captures_aux=spec.captures_aux,
            skip_on_idle=spec.skip_on_idle,
            name=node.name,
        )
    else:
        return ExecutionStep(
            runner=_runner_from_node(node),
            module=module,
            spec=spec,
            kind=spec.kind,
            captures_aux=spec.captures_aux,
            skip_on_idle=spec.skip_on_idle,
            name=node.name,
        )


def _compile_compute_step(
    node: ExecutionNode,
    mapping: Mapping,
    mod_idx: int,
    exec_plan: list[ExecutionNode],
    is_last_compute: bool,
    state: _TrackedState,
    next_layer_input_group: ParallelGroup | None,
    is_first_layer: bool = False,
) -> ExecutionStep:
    """Compile a compute step (ATTENTION / DENSE_MLP / MOE / GENERIC)."""
    spec = node.spec
    pre_comms: list[CommOp] = []
    post_comms: list[CommOp] = []

    input_group = _input_group(spec)
    output_group = _output_group(spec)

    hidden = state.hidden

    if (
        spec.input_placement is not None
        and spec.input_placement.type == PlacementType.REPLICATE
        and input_group is not None
        and group_has_parallel(mapping, input_group)
        and hidden is not None
        and hidden.type == PlacementType.SHARD
    ):
        gather_group = hidden.group
        pre_comms.append(AllGatherOp(mapping, gather_group))
        if state.residual is not None and state.residual.type == PlacementType.SHARD:
            pre_comms.append(ResidualAllGatherOp(mapping, gather_group))
            state.residual = Replicate(input_group)
        state.hidden = Replicate(input_group)
    elif hidden is None and not (is_first_layer and spec.kind == ModuleKind.ATTENTION):
        # Data is not tracked (no previous TP/EP), but the current module
        # expects Replicate on a group with compiler-managed parallelism
        # (e.g. Dense TP, MoE TP/EP).  All-gather on the input group.
        # The first layer's attention is exempt: data from embedding.
        pre_comms.append(AllGatherOp(mapping, input_group))
        state.hidden = Replicate(input_group)

    residual_before_post = state.residual

    state.hidden = (
        Partial(output_group)
        if output_group is not None and group_has_parallel(mapping, output_group)
        else None
    )

    if is_last_compute:
        _insert_last_compute_post_comms(
            post_comms=post_comms,
            spec=spec,
            mapping=mapping,
            next_layer_input_group=next_layer_input_group,
            exec_plan=exec_plan,
            state=state,
            hidden_before_input=hidden,
            residual_before_output=residual_before_post,
        )
    else:
        _insert_mid_layer_post_comms(
            post_comms=post_comms,
            spec=spec,
            mapping=mapping,
            mod_idx=mod_idx,
            exec_plan=exec_plan,
            state=state,
        )

    return ExecutionStep(
        runner=_runner_from_node(node),
        pre_comms=pre_comms,
        post_comms=post_comms,
        spec=spec,
        kind=spec.kind,
        captures_aux=spec.captures_aux,
        skip_on_idle=spec.skip_on_idle,
        name=node.name,
    )


# ---------------------------------------------------------------------------
# Post-communication insertion
# ---------------------------------------------------------------------------


def _insert_last_compute_post_comms(
    post_comms: list[CommOp],
    spec: ModuleSpec,
    mapping: Mapping,
    next_layer_input_group: ParallelGroup | None,
    exec_plan: list[ExecutionNode],
    state: _TrackedState,
    hidden_before_input: Placement | None,
    residual_before_output: Placement | None,
) -> None:
    """Insert post-communication for the last compute module in the layer."""
    output_group = _output_group(spec)
    if output_group is None or not group_has_parallel(mapping, output_group):
        state.hidden = None
        return

    if next_layer_input_group is None:
        next_layer_input_group = find_first_compute_input_group(exec_plan)
    # AR/RSAG decision must compare against ATTN_TP, not whatever
    # group attn_spec may have switched to (e.g. DENSE_TP).
    # This matches CommManager.use_all_reduce(is_moe) which checks
    # attn.tp_size against the output tp/ep size.
    use_ar = use_all_reduce(mapping, output_group, ParallelGroup.ATTN_TP)
    first_layer_dense_tp_from_dp_attention = (
        spec.kind == ModuleKind.DENSE_MLP
        and hidden_before_input is None
        and residual_before_output is None
        and not mapping.has_attn_tp
    )

    if not use_ar and first_layer_dense_tp_from_dp_attention:
        state.hidden = Shard(output_group)
        return

    if use_ar and mapping.has_attn_tp:
        post_comms.append(DeferredReduceOp(mapping, output_group))
        state.hidden = Partial(output_group)
    elif use_ar:
        post_comms.append(AllReduceOp(mapping, output_group))
        state.hidden = Replicate(output_group)
    else:
        post_comms.append(ReduceScatterOp(mapping, output_group))
        state.hidden = Shard(output_group)


def _insert_mid_layer_post_comms(
    post_comms: list[CommOp],
    spec: ModuleSpec,
    mapping: Mapping,
    mod_idx: int,
    exec_plan: list[ExecutionNode],
    state: _TrackedState,
) -> None:
    """Insert post-communication for a mid-layer compute module."""
    output_group = _output_group(spec)
    if output_group is None or not group_has_parallel(mapping, output_group):
        # No TP on this group → output is effectively Replicate, no comm.
        state.hidden = Replicate(output_group) if output_group is not None else None
        return

    next_compute_input_group = _find_next_compute_input_group(exec_plan, mod_idx)
    if next_compute_input_group is None:
        return

    next_norm_can_fuse = _intervening_norm_supports_fusion(exec_plan, mod_idx)
    use_ar = use_all_reduce(mapping, output_group, next_compute_input_group)

    if next_norm_can_fuse and can_fuse_reduce_norm(
        mapping, output_group, next_compute_input_group
    ):
        # Fused norm will absorb the reduce.
        # Data stays Partial until norm resolves it (scattered_on stays None).
        return

    if use_ar:
        post_comms.append(AllReduceOp(mapping, output_group))
        state.hidden = Replicate(output_group)
        if state.residual is not None and state.residual.type == PlacementType.SHARD:
            post_comms.append(ResidualAllGatherOp(mapping, output_group))
            state.residual = Replicate(output_group)
        return

    post_comms.append(ReduceScatterOp(mapping, output_group))
    state.hidden = Shard(output_group)
    if state.residual is None or state.residual.type == PlacementType.REPLICATE:
        post_comms.append(ResidualSliceOp(mapping, output_group))
    state.residual = Shard(output_group)


# ---------------------------------------------------------------------------
# Placement analysis helpers
# ---------------------------------------------------------------------------


def find_first_compute_input_group(exec_plan: list[ExecutionNode]) -> ParallelGroup:
    """Find the input group of the first compute (non-NORM) module in exec_plan."""
    for mod in exec_plan:
        spec = mod.spec
        if spec.kind != ModuleKind.NORM and spec.input_placement is not None:
            return spec.input_placement.group
    return ParallelGroup.ATTN_TP  # fallback


def _initial_hidden_placement(
    mapping: Mapping,
    prev_layer_output_group: ParallelGroup | None,
    first_compute_input_group: ParallelGroup,
) -> Placement | None:
    if prev_layer_output_group is None:
        return None
    if not group_has_parallel(mapping, prev_layer_output_group):
        return None
    # AR/RSAG decision must compare against ATTN_TP, matching
    # CommManager.use_all_reduce() which checks attn.tp_size.
    if use_all_reduce(mapping, prev_layer_output_group, ParallelGroup.ATTN_TP):
        return (
            Partial(prev_layer_output_group)
            if mapping.has_attn_tp
            else Replicate(prev_layer_output_group)
        )
    return Shard(prev_layer_output_group)


def _initial_residual_placement(
    mapping: Mapping,
    prev_layer_output_group: ParallelGroup | None,
    first_compute_input_group: ParallelGroup,
) -> Placement | None:
    if prev_layer_output_group is None:
        return None
    if not group_has_parallel(mapping, prev_layer_output_group):
        return None
    if use_all_reduce(mapping, prev_layer_output_group, ParallelGroup.ATTN_TP):
        return Replicate(prev_layer_output_group)
    return Shard(prev_layer_output_group)


def _find_next_compute_input_group(
    exec_plan: list[ExecutionNode],
    after_index: int,
) -> ParallelGroup | None:
    """Find the input group of the next compute module after *after_index*."""
    for i in range(after_index + 1, len(exec_plan)):
        spec = exec_plan[i].spec
        if spec.kind != ModuleKind.NORM and spec.input_placement is not None:
            return _input_group(spec)
    return None


def _intervening_norm_supports_fusion(
    exec_plan: list[ExecutionNode],
    compute_index: int,
) -> bool:
    """Check if there's a fusible norm between compute_index and the next compute module."""
    for i in range(compute_index + 1, len(exec_plan)):
        spec = exec_plan[i].spec
        if spec.kind == ModuleKind.NORM:
            return spec.fusion == FusionCapability.REDUCE_NORM
        else:
            return False
    return False


def _compute_final_placement(
    state: _TrackedState,
    mapping: Mapping,
) -> Placement | None:
    """Determine the final Placement based on tracked state."""
    hidden = state.hidden
    if hidden is None:
        return None
    return hidden if group_has_parallel(mapping, hidden.group) else hidden
