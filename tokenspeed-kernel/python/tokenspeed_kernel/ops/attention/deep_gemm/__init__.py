from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.flashinfer.dsa_topk import (
    deterministic_decode_topk,
    has_ragged_decode_topk,
)
from tokenspeed_kernel.ops.attention.triton.dsa_topk import (
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
_PERSISTENT_TOPK_WORKSPACE_BYTES = 1024 * 1024


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
        signatures=frozenset({format_signature()}),
        traits={
            "page_size": frozenset({64}),
        },
        priority=Priority.PERFORMANT,
    )
    def deep_gemm_dsa_plan(
        *,
        page_size: int,
        seq_lens_2d: torch.Tensor,
        out: object | None = None,
    ) -> torch.Tensor:
        refreshed = deep_gemm.get_paged_mqa_logits_metadata(
            seq_lens_2d,
            page_size,
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
        seq_lens_2d: torch.Tensor | None = None,
        plan: object | None = None,
        out: torch.Tensor | None = None,
        lens_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert weights.dtype == torch.float32
        assert weights.is_contiguous()
        assert seq_lens.dtype == torch.int32
        assert seq_lens.is_contiguous()
        assert block_table.dtype == torch.int32
        assert block_table.is_contiguous()

        out, lens_out = _check_out(
            out,
            lens_out,
            tokens=q.shape[0],
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
        q_scale = q_scale.view(q.shape[0], q.shape[1], 1)
        scaled_weights = (
            weights.unsqueeze(-1) * q_scale * float(softmax_scale)
        ).squeeze(-1)

        if seq_lens_2d is None or plan is None:
            raise RuntimeError(
                "DeepGEMM DSA decode top-k requires precomputed plan and "
                "seq_lens_2d (built once per forward via dsa_plan)."
            )

        max_seq_len = block_table.shape[1] * page_size
        if max_seq_len < int(topk):
            raise RuntimeError(
                "DeepGEMM DSA paged top-k requires block table capacity >= topk; "
                f"got capacity={max_seq_len}, topk={topk}"
            )

        kv_cache = index_k_cache.view(
            -1,
            int(page_size),
            1,
            index_k_cache.shape[-1],
        )
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8.view(-1, q_len_per_req, q.shape[1], q.shape[-1]),
            kv_cache,
            scaled_weights.contiguous(),
            seq_lens_2d,
            block_table,
            plan,
            max_seq_len,
            clean_logits=False,
        )
        logits.nan_to_num_(
            nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf")
        )
        local_topk_offsets = torch.empty_like(out)
        if has_ragged_decode_topk():
            deterministic_decode_topk(
                logits,
                local_topk_offsets,
                topk,
                lengths=seq_lens,
                q_len_per_req=q_len_per_req,
                workspace=torch.empty(
                    (_PERSISTENT_TOPK_WORKSPACE_BYTES,),
                    dtype=torch.uint8,
                    device=q.device,
                ),
                max_seq_len=max_seq_len,
            )
        else:
            # No ragged CUDA top-k: mask each row to its causal window first.
            # seq_lens_2d is a full-length broadcast (only its last column is
            # read on the hot path), so derive the per-token bound from the
            # per-request seq_lens: seq_lens[req] - (q_len_per_req - 1) + j.
            offsets = torch.arange(
                1 - q_len_per_req, 1, device=seq_lens.device, dtype=torch.int32
            )
            seq_lens_per_token = (seq_lens.unsqueeze(1) + offsets).reshape(-1)
            col_ids = torch.arange(logits.shape[1], dtype=torch.int32, device=q.device)
            logits.masked_fill_(
                col_ids.view(1, -1) >= seq_lens_per_token.view(-1, 1), float("-inf")
            )
            deterministic_decode_topk(logits, local_topk_offsets, topk)

        return local_topk_to_global_slots(
            local_topk_offsets=local_topk_offsets,
            block_table=block_table,
            block_size=int(page_size),
            seq_lens=seq_lens,
            q_len_per_req=q_len_per_req,
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
            flat = index_k_cache.reshape(-1)
            fp8_view = torch.as_strided(
                flat.view(q_fp8.dtype),
                (
                    index_k_cache.shape[0] // int(page_size),
                    int(page_size),
                    hd,
                ),
                (int(page_size) * row_bytes, hd, 1),
            )
            scale_view = torch.as_strided(
                flat.view(torch.float32),
                (
                    index_k_cache.shape[0] // int(page_size),
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
