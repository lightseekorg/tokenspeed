from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.flashinfer.dsa_topk import (
    deterministic_decode_topk,
    has_deterministic_decode_topk,
)
from tokenspeed_kernel.ops.attention.triton.dsa_sparse_layout import (
    local_topk_to_global_slots,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

platform = current_platform()


def _check_out(
    out: torch.Tensor | None,
    lens_out: torch.Tensor | None,
    *,
    tokens: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_out = (tokens, int(topk))
    if out is None:
        out = torch.empty(expected_out, dtype=torch.int32, device=device)
    elif out.shape != expected_out or out.dtype != torch.int32 or out.device != device:
        raise ValueError(
            "out must be int32 with shape "
            f"{expected_out} on {device}, got {tuple(out.shape)} {out.dtype} {out.device}"
        )
    expected_lens = (tokens,)
    if lens_out is None:
        lens_out = torch.empty(expected_lens, dtype=torch.int32, device=device)
    elif (
        lens_out.shape != expected_lens
        or lens_out.dtype != torch.int32
        or lens_out.device != device
    ):
        raise ValueError(
            "lens_out must be int32 with shape "
            f"{expected_lens} on {device}, got "
            f"{tuple(lens_out.shape)} {lens_out.dtype} {lens_out.device}"
        )
    return out, lens_out


if platform.is_nvidia:
    from tokenspeed_kernel.thirdparty import deep_gemm
    from tokenspeed_kernel.thirdparty import trtllm as _trtllm  # noqa: F401

    @register_kernel(
        "attention",
        "dsa_plan",
        name="deep_gemm_dsa_plan",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {format_signature(seq_lens=dense_tensor_format(torch.int32))}
        ),
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
        },
        priority=Priority.PERFORMANT,
    )
    def deep_gemm_dsa_plan(
        seq_lens: torch.Tensor,
        *,
        page_size: int,
        q_len_per_req: int = 1,
        out: object | None = None,
    ) -> torch.Tensor:
        q_len_per_req = int(q_len_per_req)
        seq_lens = seq_lens.to(dtype=torch.int32).contiguous()
        if seq_lens.dim() == 1:
            seq_lens = seq_lens.view(-1, q_len_per_req).contiguous()
        else:
            seq_lens = seq_lens.contiguous()
        refreshed = deep_gemm.get_paged_mqa_logits_metadata(
            seq_lens,
            int(page_size),
            deep_gemm.get_num_sms(),
        )
        if out is None:
            return refreshed
        if (
            not isinstance(out, torch.Tensor)
            or out.shape != refreshed.shape
            or out.device != refreshed.device
            or out.dtype != refreshed.dtype
        ):
            actual = (
                f"{tuple(out.shape)} {out.dtype} {out.device}"
                if isinstance(out, torch.Tensor)
                else type(out).__name__
            )
            raise RuntimeError(
                "DSA paged top-k plan changed shape during CUDA graph replay; "
                "recapture or use eager for this batch. "
                f"captured={actual}, refreshed={tuple(refreshed.shape)} "
                f"{refreshed.dtype} {refreshed.device}"
            )
        out.copy_(refreshed)
        return out

    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="deep_gemm_dsa_decode_topk",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.float32),
                )
            }
        ),
        traits={
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "page_size": frozenset({64}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
        },
        priority=Priority.PERFORMANT,
    )
    def deep_gemm_dsa_decode_topk(
        q: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        *,
        page_size: int,
        topk: int,
        softmax_scale: float,
        q_len_per_req: int = 1,
        index_k_cache: torch.Tensor | None = None,
        index_k_with_scale_cache: torch.Tensor | None = None,
        plan: object | None = None,
        out: torch.Tensor | None = None,
        lens_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = q.contiguous()
        weights = weights.float().contiguous()
        seq_lens = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
        block_table = block_table.to(device=q.device, dtype=torch.int32).contiguous()
        tokens = q.shape[0]
        out, lens_out = _check_out(
            out,
            lens_out,
            tokens=tokens,
            topk=topk,
            device=q.device,
        )

        q_2d = q.view(-1, q.shape[-1])
        q_fp8, q_scale = quantize_fp8_with_scale(
            q_2d,
            granularity="token_group",
            group_size=128,
            scale_encoding="float32",
        )
        q_fp8 = q_fp8.view_as(q)
        q_scale = q_scale.view(tokens, q.shape[1], 1)
        scaled_weights = (
            weights.unsqueeze(-1) * q_scale * float(softmax_scale)
        ).squeeze(-1)

        q_len_per_req = int(q_len_per_req)
        seq_lens_2d = seq_lens.view(-1, q_len_per_req).contiguous()
        request_block_table = block_table[::q_len_per_req].contiguous()
        max_seq_len = int(request_block_table.shape[1]) * int(page_size)
        if max_seq_len < int(topk):
            raise RuntimeError(
                "DeepGEMM DSA paged top-k requires block table capacity >= topk; "
                f"got capacity={max_seq_len}, topk={topk}"
            )
        schedule_metadata = plan
        if schedule_metadata is None:
            schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                seq_lens_2d,
                int(page_size),
                deep_gemm.get_num_sms(),
            )
        kv_cache = index_k_with_scale_cache.view(
            -1,
            int(page_size),
            1,
            index_k_with_scale_cache.shape[-1],
        )
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8.view(-1, q_len_per_req, q.shape[1], q.shape[-1]),
            kv_cache,
            scaled_weights.contiguous(),
            seq_lens_2d,
            request_block_table,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )
        logits.nan_to_num_(
            nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf")
        )
        col_ids = torch.arange(logits.shape[1], dtype=torch.int32, device=q.device)
        logits.masked_fill_(col_ids.view(1, -1) >= seq_lens.view(-1, 1), float("-inf"))
        local_topk_offsets = torch.empty_like(out)
        deterministic_decode_topk(logits, local_topk_offsets, int(topk))
        return local_topk_to_global_slots(
            local_topk_offsets=local_topk_offsets,
            block_table=block_table,
            block_size=int(page_size),
            seq_lens=seq_lens,
            out=out,
            lens_out=lens_out,
        )

    @register_kernel(
        "attention",
        "dsa_prefill_topk",
        name="deep_gemm_dsa_prefill_topk",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.float32),
                )
            }
        ),
        traits={
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "index_k_format": frozenset({"fp8_scaled"}),
        },
        priority=Priority.PERFORMANT,
    )
    def deep_gemm_dsa_prefill_topk(
        q: torch.Tensor,
        weights: torch.Tensor,
        kv_workspace_slots: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        *,
        topk: int,
        softmax_scale: float,
        index_k_cache: torch.Tensor | None = None,
        index_k_with_scale_cache: torch.Tensor | None = None,
        page_size: int | None = None,
        index_k_fp8: torch.Tensor | None = None,
        index_k_scale: torch.Tensor | None = None,
        max_logits_bytes: int | None = None,
        out: torch.Tensor | None = None,
        lens_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        q = q.contiguous()
        weights = weights.float().contiguous()
        row_starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
        row_ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
        tokens = q.shape[0]
        out, lens_out = _check_out(
            out,
            lens_out,
            tokens=tokens,
            topk=topk,
            device=q.device,
        )
        out.fill_(-1)

        q_2d = q.view(-1, q.shape[-1])
        q_fp8, q_scale = quantize_fp8_with_scale(
            q_2d,
            granularity="token_group",
            group_size=128,
            scale_encoding="float32",
        )
        q_fp8 = q_fp8.view_as(q)
        q_scale = q_scale.view(tokens, q.shape[1], 1)
        scaled_weights = (
            weights.unsqueeze(-1) * q_scale * float(softmax_scale)
        ).squeeze(-1)
        if index_k_fp8 is None or index_k_scale is None:
            hd = q.shape[-1]
            num_groups = hd // 128
            row_bytes = hd + num_groups * 4
            flat = index_k_with_scale_cache.reshape(-1)
            fp8_view = torch.as_strided(
                flat.view(q_fp8.dtype),
                (
                    index_k_with_scale_cache.shape[0] // int(page_size),
                    int(page_size),
                    hd,
                ),
                (int(page_size) * row_bytes, hd, 1),
            )
            scale_view = torch.as_strided(
                flat.view(torch.float32),
                (
                    index_k_with_scale_cache.shape[0] // int(page_size),
                    int(page_size),
                    num_groups,
                ),
                ((int(page_size) * row_bytes) // 4, num_groups, 1),
                (int(page_size) * hd) // 4,
            )
            slots = kv_workspace_slots.to(device=q.device, dtype=torch.long)
            index_k_fp8 = fp8_view[slots // int(page_size), slots % int(page_size)]
            index_k_scale = scale_view[slots // int(page_size), slots % int(page_size)]
        k_fp8 = (
            index_k_fp8.view(q_fp8.dtype)
            if index_k_fp8.dtype == torch.uint8
            else index_k_fp8
        )
        kv_fp8 = (k_fp8.contiguous(), index_k_scale.squeeze(-1).contiguous())
        candidate_lens = (row_ends - row_starts).clamp_min(0)
        lens_out.copy_(
            torch.minimum(candidate_lens, torch.full_like(candidate_lens, int(topk)))
        )
        if tokens == 0:
            return out, lens_out

        seq_len_sum = max(int(kv_workspace_slots.numel()), 1)
        if max_logits_bytes is None:
            max_query_rows = tokens
        else:
            max_query_rows = max(1, int(max_logits_bytes) // (seq_len_sum * 4))
        local_starts_i32 = torch.zeros_like(row_starts)
        for start in range(0, tokens, max_query_rows):
            end = min(start + max_query_rows, tokens)
            max_seqlen_k = int(candidate_lens[start:end].max().item())
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8[start:end].contiguous(),
                kv_fp8,
                scaled_weights[start:end].contiguous(),
                row_starts[start:end],
                row_ends[start:end],
                clean_logits=False,
                max_seqlen_k=max(max_seqlen_k, 1),
            )
            logits.nan_to_num_(
                nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf")
            )
            torch.ops.trtllm.indexer_topk_prefill(
                logits.contiguous(),
                local_starts_i32[start:end],
                candidate_lens[start:end].to(torch.int32).contiguous(),
                out[start:end],
                int(topk),
            )
        valid = out >= 0
        out.copy_(torch.where(valid, out + row_starts.unsqueeze(1), out))
        return out, lens_out
