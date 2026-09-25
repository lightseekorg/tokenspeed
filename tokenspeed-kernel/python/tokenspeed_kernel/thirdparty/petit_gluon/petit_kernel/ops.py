"""Gluon-only host bindings; the VMM helper contains no GPU kernels."""

import torch
from lib.moe.rocm.fused_moe import (
    FusedMoE2Stage1Params,
    FusedMoE2Stage2Params,
    FusedMoE2StageCommonParams,
    FusedMoE2StageWorkspaceSize,
    FusedMoEMatmul2Stage1,
    FusedMoEMatmul2Stage2,
)
from lib.pybind.mega_moe import MegaMoe as mega_moe
from lib.pybind.mega_moe import MegaMoeQuantizeMxFp4 as mega_moe_quantize_mxfp4
from lib.pybind.mega_moe import (
    MegaMoeWorkspaceInputViews as mega_moe_workspace_input_views,
)
from lib.pybind.vmm_symmetric_heap import create_vmm_symmetric_heap as VmmSymmetricHeap


def fmoe_matmul_2stage_workspace_size(max_num_m_blocks, inter_dim, solution_id):
    return FusedMoE2StageWorkspaceSize(max_num_m_blocks, inter_dim, solution_id)


def _common(
    intermediate, ids, experts, valid, topk, m, dim, inter_dim, num_experts, persistent
):
    return FusedMoE2StageCommonParams(
        intermediate,
        intermediate.numel() * intermediate.element_size(),
        ids,
        experts,
        valid,
        topk,
        experts.numel(),
        m,
        dim,
        inter_dim,
        num_experts,
        torch.cuda.current_stream(intermediate.device),
        persistent,
    )


def fmoe_matmul_2stage_stage1(
    intermediate,
    input_q,
    w1_q,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    topk,
    input_scale,
    w1_scale,
    inter_dim,
    num_experts,
    solution_id,
    num_persistent_tgs=0,
    w13_bias=None,
):
    common = _common(
        intermediate,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_q.shape[0],
        input_q.shape[1] * 2,
        inter_dim,
        num_experts,
        num_persistent_tgs,
    )
    params = FusedMoE2Stage1Params(
        common, input_q, w1_q, input_scale, w1_scale, w13_bias
    )
    status = FusedMoEMatmul2Stage1(params, solution_id)
    if status:
        raise RuntimeError(f"two-stage stage one failed with code {status}")
    return intermediate


def fmoe_matmul_2stage_stage2(
    out,
    intermediate,
    w2_q,
    sorted_token_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    topk,
    w2_scale,
    inter_dim,
    num_experts,
    solution_id,
    num_persistent_tgs=0,
    w2_bias=None,
):
    common = _common(
        intermediate,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        out.shape[0],
        out.shape[1],
        inter_dim,
        num_experts,
        num_persistent_tgs,
    )
    params = FusedMoE2Stage2Params(common, out, w2_q, sorted_weights, w2_scale, w2_bias)
    status = FusedMoEMatmul2Stage2(params, solution_id)
    if status:
        raise RuntimeError(f"two-stage stage two failed with code {status}")
    return out
