"""Native Python binding contracts implemented with the Gluon MegaMoE kernels."""

import torch
from lib.moe.rocm.mega_moe import (
    GetMegaMoEWorkspaceInfo,
    MegaMoECompute,
    MegaMoEParams,
    MegaMoEQuantizeMxFp4,
    kMegaMoEProfileCounterCount,
    kMegaMoEProfileCtas,
)
from lib.pybind.vmm_symmetric_heap import runtime


def _check(condition, message):
    if not condition:
        raise RuntimeError(message)


def CheckKernelStatus(err, kernel_name):
    _check(err == 0, f"{kernel_name} failed with code {err}")


def CheckTensorDeviceAndType(tensor, dtype, device, name):
    _check(
        tensor.is_cuda and tensor.get_device() == device,
        f"{name} must be on the workspace HIP device",
    )
    _check(tensor.dtype == dtype, f"{name} has invalid dtype")


def CheckTensor(tensor, dtype, device, name):
    CheckTensorDeviceAndType(tensor, dtype, device, name)
    _check(tensor.is_contiguous(), f"{name} must be contiguous")


def WorkspaceLayout(info):
    layout = runtime().Layout()
    for field in (
        "barrier_record_bytes",
        "rank_sym_buffer_base",
        "rank_slot_bytes",
        "local_offset",
        "local_bytes",
    ):
        setattr(layout, field, getattr(info, field))
    return layout


def LookupSolution(heap, solution_id):
    info = GetMegaMoEWorkspaceInfo(heap.rank, solution_id)
    _check(
        heap.world_size == info.num_ranks,
        "MegaMoE solution rank count does not match the workspace",
    )
    return info


def AllocateInputViews(heap, max_tokens, info):
    _check(0 < max_tokens <= info.max_tokens_per_rank, "invalid MegaMoE token capacity")
    heap.allocate(WorkspaceLayout(info))
    base = heap.local_tensor().view(torch.uint8)
    rows = base[
        info.input_tokens_offset : info.input_tokens_offset
        + max_tokens * info.input_token_bytes
    ].view(max_tokens, info.input_token_bytes)
    tokens = rows[:, : info.hidden_size // 2]
    scales = rows[
        :, info.hidden_size // 2 : info.hidden_size // 2 + info.hidden_size // 32
    ]
    ids = (
        base[
            info.input_topk_expert_id_offset : info.input_topk_expert_id_offset
            + max_tokens * info.topk * 4
        ]
        .view(torch.int32)
        .view(max_tokens, info.topk)
    )
    weights = (
        base[
            info.input_topk_expert_weight_offset : info.input_topk_expert_weight_offset
            + max_tokens * info.topk * 4
        ]
        .view(torch.float32)
        .view(max_tokens, info.topk)
    )
    return tokens, scales, ids, weights


def MegaMoeWorkspaceInputViews(heap, max_tokens, solution_id):
    return AllocateInputViews(heap, max_tokens, LookupSolution(heap, solution_id))


def MegaMoeQuantizeMxFp4(input, output=None, output_scales=None):
    _check(input.is_cuda, "input must be on a HIP device")
    device = input.get_device()
    _check(input.dtype == torch.bfloat16, "input has invalid dtype")
    _check(
        input.ndim == 2 and input.shape[1] > 0 and input.shape[1] % 32 == 0,
        "input must be [num_tokens, hidden_size] with a 32-aligned hidden size",
    )
    _check(
        input.stride(1) == 1 and input.stride(0) % 8 == 0,
        "input rows must be contiguous and 16-byte aligned",
    )
    rows, cols = input.shape
    value_bytes, scale_cols = cols // 2, cols // 32
    row_bytes = value_bytes + ((scale_cols + 15) // 16 * 16)
    _check(
        (output is None) == (output_scales is None),
        "output and output_scales must be provided together",
    )
    if output is None:
        storage = torch.empty((rows, row_bytes), dtype=torch.uint8, device=input.device)
        output, output_scales = (
            storage[:, :value_bytes],
            storage[:, value_bytes : value_bytes + scale_cols],
        )
    else:
        CheckTensorDeviceAndType(output, torch.uint8, device, "output")
        CheckTensorDeviceAndType(output_scales, torch.uint8, device, "output_scales")
        _check(
            output.shape == (rows, value_bytes),
            "output must be [num_tokens, hidden_size / 2]",
        )
        _check(
            output_scales.shape == (rows, scale_cols),
            "output_scales must be [num_tokens, hidden_size / 32]",
        )
        _check(
            output.stride() == (row_bytes, 1)
            and output_scales.stride() == (row_bytes, 1),
            "output rows must include aligned scale storage",
        )
        _check(
            rows == 0 or output_scales.data_ptr() == output.data_ptr() + value_bytes,
            "output_scales must be a view into the output rows",
        )
    with torch.cuda.device(device):
        CheckKernelStatus(
            MegaMoEQuantizeMxFp4(
                input,
                output,
                rows,
                cols,
                input.stride(0),
                torch.cuda.current_stream(device),
            ),
            "MegaMoE MXFP4 quantizer",
        )
    return output, output_scales


def MegaMoe(
    heap,
    w13,
    w2,
    scales_w13,
    scales_w2,
    num_tokens,
    solution_id,
    w13_bias=None,
    w2_bias=None,
    out=None,
    input_tokens=None,
    input_topk_ids=None,
    input_topk_weights=None,
    profile=None,
):
    info = LookupSolution(heap, solution_id)
    device = heap.device_index
    for name, tensor in (
        ("w13", w13),
        ("w2", w2),
        ("fc1_scale", scales_w13),
        ("fc2_scale", scales_w2),
    ):
        CheckTensor(tensor, torch.uint8, device, name)
    _check(0 <= num_tokens <= info.max_tokens_per_rank, "invalid MegaMoE token count")
    local_experts = info.num_experts // info.num_ranks
    _check(
        w13.ndim == 3
        and w2.ndim == 3
        and w13.shape[0] == local_experts
        and w2.shape[0] == local_experts,
        "weights must contain the rank-local experts",
    )
    inter_dim = w2.shape[2] * 2
    _check(inter_dim % 512 == 0, "intermediate dimension must be 512-aligned")
    _check(
        w2.shape[1] == info.compute_hidden_size
        and w13.shape[1:] == (2 * inter_dim, info.compute_hidden_size // 2),
        "invalid MegaMoE weight shapes",
    )
    _check(
        not heap.allocate(WorkspaceLayout(info)),
        "request MegaMoE input views before launch",
    )
    with torch.cuda.device(device):
        if out is None:
            out = torch.empty(
                (num_tokens, info.compute_hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
        CheckTensor(out, torch.bfloat16, device, "out")
        _check(
            out.shape
            in ((num_tokens, info.hidden_size), (num_tokens, info.compute_hidden_size)),
            "out must be [num_tokens, hidden_size] or [num_tokens, compute_hidden_size]",
        )
        for name, tensor in (("w13_bias", w13_bias), ("w2_bias", w2_bias)):
            if tensor is not None:
                CheckTensor(tensor, torch.bfloat16, device, name)
        if profile is not None:
            CheckTensor(profile, torch.int64, device, "profile")
            _check(
                profile.shape == (kMegaMoEProfileCounterCount, kMegaMoEProfileCtas),
                "profile must have shape [35, 3072]",
            )
        external_count = sum(
            t is not None for t in (input_tokens, input_topk_ids, input_topk_weights)
        )
        _check(
            external_count in (0, 3),
            "external MegaMoE inputs must be provided together",
        )
        if external_count == 3:
            CheckTensorDeviceAndType(input_tokens, torch.uint8, device, "input_tokens")
            CheckTensor(input_topk_ids, torch.int32, device, "input_topk_ids")
            CheckTensor(input_topk_weights, torch.float32, device, "input_topk_weights")
            _check(
                input_tokens.shape == (num_tokens, info.hidden_size // 2),
                "input_tokens must be [num_tokens, hidden_size / 2]",
            )
            _check(
                input_tokens.stride() == (info.input_token_bytes, 1),
                "input_tokens rows must include aligned scale storage",
            )
            _check(
                input_topk_ids.shape == (num_tokens, info.topk),
                "input_topk_ids must be [num_tokens, topk]",
            )
            _check(
                input_topk_weights.shape == input_topk_ids.shape,
                "input_topk_weights must match input_topk_ids",
            )
        params = MegaMoEParams(
            out,
            out.stride(0),
            w13,
            w2,
            scales_w13,
            scales_w2,
            input_tokens,
            input_topk_ids,
            input_topk_weights,
            num_tokens,
            info.compute_hidden_size,
            inter_dim,
            w13_bias,
            w2_bias,
            heap.local_tensor().view(torch.uint8),
            heap.rank,
            torch.cuda.current_stream(device),
            profile,
        )
        CheckKernelStatus(MegaMoECompute(params, solution_id), "MegaMoE")
    return out
