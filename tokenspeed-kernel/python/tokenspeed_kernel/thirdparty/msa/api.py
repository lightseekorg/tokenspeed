# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""FMHA varlen attention API: plan + run for SM100.

fmha_sm100_plan and fmha_sm100.

Doc is REPO/docs/fmha_sm100_api.md
"""

import math
from typing import Optional, Tuple, Union

__all__ = [
    "fmha_sm100_plan",
    "fmha_sm100",
    "sparse_topk_select",
    "_fmha_sm100_plan",
    "_fmha_sm100",
]

import numpy as np
import torch

from .jit import (
    _PACK_FACTORS,
    _dlpack_dtype_code,
    get_fmha_variant,
    get_plan_fn,
    get_reduction_module,
    get_sparse_topk_module,
)
from .sparse_fmha_adapter import sparse_fmha, sparse_fmha_plan

_np_staging = np.empty(4096 * 1024, dtype=np.int32)
_np_staging_offset = 0


def _reset_np_staging():
    global _np_staging_offset
    _np_staging_offset = 0


def _plan_buf_from_list(data, device):
    global _np_staging, _np_staging_offset
    n = len(data)
    end = _np_staging_offset + n
    if end > _np_staging.shape[0]:
        _np_staging = np.empty(max(end, _np_staging.shape[0] * 2), dtype=np.int32)
        _np_staging_offset = 0
        end = n
    _np_staging[_np_staging_offset:end] = data
    buf = torch.empty(n, dtype=torch.int32, device=device)
    buf.copy_(torch.from_numpy(_np_staging[_np_staging_offset:end]), non_blocking=True)
    _np_staging_offset = end
    return buf


from enum import IntEnum, auto


class _BuffTag(IntEnum):
    fmha_sm100_cutlass_workspace = auto()

    packed_work_range = auto()
    packed_work_info = auto()
    kv_tile_begin_indices = auto()
    kv_tile_end_indices = auto()
    kv_split_indices = auto()
    plan_cost = auto()
    num_kv_splits_per_row = auto()
    workspace_lse = auto()

    workspace_o = auto()

    sparse_topk_workspace = auto()

    Total = auto()


_workspace_cache = [[None] * _BuffTag.Total for _ in range(16)]
# _workspace_cache_per_plan = []


def _new_ws_cache():
    pass
    # global _workspace_cache
    # if len(_workspace_cache) == 0:
    #     print("new")
    #     _workspace_cache_per_plan.append([[None] * _BuffTag.Total for _ in range(16)])


def _alloc_workspace_buf(tag, size, device, dtype):
    global _workspace_cache
    device_id = torch.device(device).index
    buf = _workspace_cache[device_id][tag]
    if buf is not None and buf.shape[0] >= size:
        return buf
    buf = torch.empty(size, dtype=dtype, device=device)
    _workspace_cache[device_id][tag] = buf
    return buf


def _get_workspace_buf(tag, device):
    global _workspace_cache
    device_id = torch.device(device).index
    return _workspace_cache[device_id][tag]


def _alloc_perplan_buf(tag, size, device, dtype):
    # global _workspace_cache
    # device_id = torch.device(device).index
    # buf = _workspace_cache[-1][device_id][tag]
    # if buf is not None and buf.shape[0] >= size:
    #     return buf
    buf = torch.empty(size, dtype=dtype, device=device)
    # _workspace_cache[-1][device_id][tag] = buf
    return buf


_NUM_CTA = None


def _get_num_cta(device):
    global _NUM_CTA
    if _NUM_CTA is None:
        _NUM_CTA = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_CTA


def _compute_pack_factor(max_qo_len, num_qo_heads, num_kv_heads):
    if num_kv_heads == -1:
        return 1
    h_r = num_qo_heads // num_kv_heads
    if h_r <= 1 or max_qo_len <= 0 or max_qo_len > 32:
        return 1
    max_pf = 128 // max_qo_len
    for pf in reversed(_PACK_FACTORS):
        if pf <= max_pf and pf <= h_r and h_r % pf == 0:
            return pf
    return 1


def _to_device(t, d):
    return t.to(d, non_blocking=True) if t is not None else None


def _prefill_qlen_threshold(sparse):
    return 32 if sparse else 128


def _validate_fmha_inputs(
    q,
    k,
    v,
    qo_segment_lens,
    kv_segment_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    batch_size,
    qo_total_len,
    is_paged,
    kv_indices,
    kv_block_indexes,
    qo_offset,
    page_size,
    pack_factor=1,
    packed_work_range=None,
    packed_work_info=None,
    num_kv_splits=1,
    kv_tile_begin_indices=None,
    kv_tile_end_indices=None,
    kv_split_indices=None,
    qo_segment_offsets=None,
    kv_page_indptr=None,
):
    assert (
        q.dim() == 3
    ), f"q must be [total_qo_len, num_qo_heads, head_dim], got {q.shape}"
    assert (
        num_qo_heads % num_kv_heads == 0
    ), f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    assert (
        qo_segment_lens.dim() == 1
    ), f"qo_segment_lens must be 1D, got {qo_segment_lens.shape}"
    assert (
        kv_segment_lens.dim() == 1
    ), f"kv_segment_lens must be 1D, got {kv_segment_lens.shape}"
    assert qo_segment_lens.shape[0] == batch_size
    assert kv_segment_lens.shape[0] == batch_size

    qo_lens_cpu = qo_segment_lens.cpu()
    kv_lens_cpu = kv_segment_lens.cpu()

    expected_qo_total = qo_total_len * pack_factor if pack_factor > 1 else qo_total_len
    assert qo_lens_cpu.sum().item() == expected_qo_total, (
        f"sum(qo_segment_lens)={qo_lens_cpu.sum().item()} != expected={expected_qo_total} "
        f"(q.shape[0]={qo_total_len}, pack_factor={pack_factor})"
    )

    if is_paged:
        assert (
            k.dim() == 4
        ), f"paged K must be [total_pages, H_kv, page_size, D], got {k.shape}"
        assert (
            v.dim() == 4
        ), f"paged V must be [total_pages, H_kv, page_size, D], got {v.shape}"
        total_pages = k.shape[0]
        assert k.shape[1] == num_kv_heads
        assert k.shape[2] == page_size
        assert v.shape[0] == total_pages
        assert v.shape[1] == num_kv_heads
        assert v.shape[2] == page_size
    else:
        total_kv_len = k.shape[0]
        assert (
            kv_lens_cpu.sum().item() == total_kv_len
        ), f"sum(kv_segment_lens)={kv_lens_cpu.sum().item()} != k.shape[0]={total_kv_len}"

    if qo_offset is not None:
        assert (
            qo_offset.dim() == 1 and qo_offset.shape[0] == batch_size
        ), f"qo_offset must be [batch_size={batch_size}], got {qo_offset.shape}"
        # off_cpu = qo_offset.cpu()
        # assert (off_cpu >= 0).all(), (
        #     f"qo_offset must be non-negative, got min={off_cpu.min().item()}"
        # )
        # unpacked_qo_lens = qo_lens_cpu // pack_factor if pack_factor > 1 else qo_lens_cpu
        # max_offsets = kv_lens_cpu - unpacked_qo_lens
        # violations = off_cpu > max_offsets
        # if violations.any():
        #     b = violations.nonzero()[0].item()
        #     assert False, (
        #         f"qo_offset[{b}]={off_cpu[b].item()} > kv_len-qo_len="
        #         f"{kv_lens_cpu[b].item()}-{qo_lens_cpu[b].item()}={max_offsets[b].item()}"
        #     )

    if kv_indices is not None and kv_block_indexes is None:
        assert is_paged, "kv_indices provided but K/V are not paged (4D)"
        kvi_cpu = kv_indices.cpu()
        total_pages_in_table = kvi_cpu.shape[0]
        total_pages_needed = sum(
            (kv_lens_cpu[b].item() + page_size - 1) // page_size
            for b in range(batch_size)
        )
        assert total_pages_in_table == total_pages_needed, (
            f"kv_indices length {total_pages_in_table} != "
            f"sum(ceil(kv_segment_lens/page_size))={total_pages_needed}"
        )
        total_pages = k.shape[0]
        assert (kvi_cpu >= 0).all() and (kvi_cpu < total_pages).all(), (
            f"kv_indices has values outside [0, {total_pages}): "
            f"min={kvi_cpu.min().item()}, max={kvi_cpu.max().item()}"
        )

    if kv_block_indexes is not None:
        assert is_paged, "sparse mode requires paged KV (kv_indices)"
        assert kv_indices is not None, "sparse mode requires kv_indices"
        assert (
            kv_block_indexes.dim() == 3
        ), f"kv_block_indexes must be [total_qo_len, H_kv, KVBlockNum], got {kv_block_indexes.shape}"
        assert (
            kv_block_indexes.shape[0] == qo_total_len
        ), f"kv_block_indexes must be [total_qo_len={qo_total_len}, H_kv, KVBlockNum], got {kv_block_indexes.shape}"
        assert kv_block_indexes.shape[1] == num_kv_heads

        bi_cpu = kv_block_indexes.cpu()
        valid_mask = bi_cpu != -1
        assert (
            (bi_cpu >= 0) | (bi_cpu == -1)
        ).all(), "kv_block_indexes must contain only non-negative indices or -1"
        assert (
            valid_mask[:, :, :-1] >= valid_mask[:, :, 1:]
        ).all(), "kv_block_indexes: -1 padding must be at the end"
        unpacked_qo_lens = (
            qo_lens_cpu // pack_factor if pack_factor > 1 else qo_lens_cpu
        )
        batch_of_qtoken = torch.repeat_interleave(
            torch.arange(batch_size), unpacked_qo_lens
        )
        kv_lens_per_qtoken = kv_lens_cpu[batch_of_qtoken]
        num_pages = ((kv_lens_per_qtoken + page_size - 1) // page_size).view(-1, 1, 1)
        assert (
            bi_cpu[valid_mask] < num_pages.expand_as(bi_cpu)[valid_mask]
        ).all(), "kv_block_indexes: index out of range for batch page count"
        both_valid = valid_mask[:, :, :-1] & valid_mask[:, :, 1:]
        if both_valid.any():
            assert (
                bi_cpu[:, :, 1:][both_valid] > bi_cpu[:, :, :-1][both_valid]
            ).all(), "kv_block_indexes: must be strictly ascending"

    if qo_segment_offsets is not None:
        off_cpu = qo_segment_offsets.cpu()
        assert (
            off_cpu.shape[0] == batch_size + 1
        ), f"qo_segment_offsets must have {batch_size + 1} elements, got {off_cpu.shape[0]}"
        assert off_cpu[0].item() == 0, f"qo_segment_offsets[0]={off_cpu[0].item()} != 0"
        assert (
            off_cpu[-1].item() == expected_qo_total
        ), f"qo_segment_offsets[-1]={off_cpu[-1].item()} != expected={expected_qo_total}"
        diffs = off_cpu[1:] - off_cpu[:-1]
        assert (diffs >= 0).all(), "qo_segment_offsets must be non-decreasing"
        assert (
            diffs == qo_lens_cpu
        ).all(), "qo_segment_offsets must be cumsum of qo_segment_lens"

    if kv_page_indptr is not None:
        ip_cpu = kv_page_indptr.cpu()
        assert (
            ip_cpu.shape[0] == batch_size + 1
        ), f"kv_page_indptr must have {batch_size + 1} elements, got {ip_cpu.shape[0]}"
        assert ip_cpu[0].item() == 0, f"kv_page_indptr[0]={ip_cpu[0].item()} != 0"
        assert (
            ip_cpu[1:] >= ip_cpu[:-1]
        ).all(), "kv_page_indptr must be non-decreasing"
        if kv_indices is not None:
            assert (
                ip_cpu[-1].item() <= kv_indices.shape[0]
            ), f"kv_page_indptr[-1]={ip_cpu[-1].item()} > kv_indices.size={kv_indices.shape[0]}"

    if packed_work_range is not None and packed_work_info is not None:
        import math

        pwr_cpu = packed_work_range.cpu()
        pwi_cpu = packed_work_info.cpu()
        num_ctas = pwr_cpu.shape[0]
        max_work_idx = pwi_cpu.shape[0]

        packed_num_heads = (
            num_qo_heads // pack_factor if pack_factor > 1 else num_qo_heads
        )
        qo_tile_size = 128 if int(qo_lens_cpu.max()) <= 128 else 256
        max_tiles_per_batch = [
            (int(l) + qo_tile_size - 1) // qo_tile_size for l in qo_lens_cpu
        ]

        for cta in range(num_ctas):
            r = int(pwr_cpu[cta].item())
            start = r & 0xFFFFFFFF
            end = (r >> 32) & 0xFFFFFFFF
            assert start <= end, f"packed_work_range[{cta}]: start={start} > end={end}"
            assert (
                end <= max_work_idx
            ), f"packed_work_range[{cta}]: end={end} > packed_work_info.size={max_work_idx}"
            for wi in range(start, end):
                packed = int(pwi_cpu[wi].item())
                bi = packed & 0xFFFF
                hi = (packed >> 16) & 0xFFFF
                qo_tile = (packed >> 32) & 0xFFFFFFFF
                assert (
                    bi < batch_size
                ), f"packed_work_info[{wi}]: batch_idx={bi} >= batch_size={batch_size}"
                assert (
                    hi < packed_num_heads
                ), f"packed_work_info[{wi}]: head_idx={hi} >= packed_num_heads={packed_num_heads}"
                assert qo_tile < max_tiles_per_batch[bi], (
                    f"packed_work_info[{wi}]: qo_tile={qo_tile} >= max_tiles={max_tiles_per_batch[bi]} "
                    f"for batch {bi} (qo_len={int(qo_lens_cpu[bi])})"
                )

        # Verify completeness: every (batch, head, qo_tile) must appear exactly once
        # (for num_kv_splits=1) or exactly num_splits times (for split-KV, each with
        # a different split index covering the full KV range).
        expected = set()
        for bi in range(batch_size):
            for hi in range(packed_num_heads):
                for qt in range(max_tiles_per_batch[bi]):
                    expected.add((bi, hi, qt))

        seen = {}
        for cta in range(num_ctas):
            r = int(pwr_cpu[cta].item())
            start = r & 0xFFFFFFFF
            end = (r >> 32) & 0xFFFFFFFF
            for wi in range(start, end):
                packed = int(pwi_cpu[wi].item())
                bi = packed & 0xFFFF
                hi = (packed >> 16) & 0xFFFF
                qt = (packed >> 32) & 0xFFFFFFFF
                key = (bi, hi, qt)
                seen[key] = seen.get(key, 0) + 1

        missing = expected - set(seen.keys())
        assert (
            not missing
        ), f"plan missing {len(missing)} work items, e.g. {list(missing)[:5]}"

        if num_kv_splits <= 1:
            duplicates = {k: v for k, v in seen.items() if v > 1}
            assert (
                not duplicates
            ), f"plan has {len(duplicates)} duplicated work items (nosplit), e.g. {list(duplicates.items())[:5]}"

        if num_kv_splits > 1 and kv_tile_begin_indices is not None:
            tb_cpu = kv_tile_begin_indices.cpu()
            te_cpu = kv_tile_end_indices.cpu()
            sp_cpu = kv_split_indices.cpu()

            from collections import defaultdict

            tile_splits = defaultdict(list)

            for cta in range(num_ctas):
                r = int(pwr_cpu[cta].item())
                start = r & 0xFFFFFFFF
                end = (r >> 32) & 0xFFFFFFFF
                for wi in range(start, end):
                    tb, te = int(tb_cpu[wi].item()), int(te_cpu[wi].item())
                    assert (
                        tb <= te
                    ), f"kv_tile_begin[{wi}]={tb} > kv_tile_end[{wi}]={te}"
                    sp = int(sp_cpu[wi].item())
                    assert (
                        0 <= sp < num_kv_splits
                    ), f"kv_split_indices[{wi}]={sp} out of range [0, {num_kv_splits})"
                    packed = int(pwi_cpu[wi].item())
                    key = (
                        packed & 0xFFFF,
                        (packed >> 16) & 0xFFFF,
                        (packed >> 32) & 0xFFFFFFFF,
                    )
                    tile_splits[key].append((tb, te, sp))

            is_sparse = kv_block_indexes is not None
            kv_tile_size = 256 if qo_tile_size == 128 else 128
            for key, splits in tile_splits.items():
                bi, hi, qt = key
                if is_sparse:
                    kv_block_num = kv_block_indexes.shape[2]
                    kl = kv_block_num * page_size
                    off_q = kl - int(qo_lens_cpu[bi])
                else:
                    kl = int(kv_lens_cpu[bi])
                    off_q = (
                        int(qo_offset[bi].cpu())
                        if qo_offset is not None
                        else (kl - int(qo_lens_cpu[bi]))
                    )
                packed_q_end = (qt + 1) * qo_tile_size
                q_end = (
                    (packed_q_end - 1) // pack_factor + 1
                    if pack_factor > 1
                    else packed_q_end
                )
                eff_kv = min(q_end + off_q, kl)
                expected_iters = max(0, (eff_kv + kv_tile_size - 1) // kv_tile_size)
                splits_sorted = sorted(splits, key=lambda x: x[0])
                if expected_iters > 0:
                    assert (
                        splits_sorted[0][0] == 0
                    ), f"tile {key}: first split begins at {splits_sorted[0][0]}, expected 0"
                    assert (
                        splits_sorted[-1][1] == expected_iters
                    ), f"tile {key}: last split ends at {splits_sorted[-1][1]}, expected {expected_iters}"
                for i in range(1, len(splits_sorted)):
                    prev_end = splits_sorted[i - 1][1]
                    curr_begin = splits_sorted[i][0]
                    assert prev_end == curr_begin, (
                        f"tile {key}: gap/overlap at split boundary: "
                        f"prev_end={prev_end}, curr_begin={curr_begin}"
                    )
                sub_ids = sorted(s[2] for s in splits_sorted)
                expected_ids = list(range(len(splits_sorted)))
                assert (
                    sub_ids == expected_ids
                ), f"tile {key}: sub_ids {sub_ids} not contiguous 0..{len(splits_sorted)-1}"


def _expand_for_per_token_sparse(qo_lens, kv_lens, qo_offset, page_size, pack_factor=1):
    B = len(qo_lens)
    total_q = sum(qo_lens)

    if qo_offset is None:
        qo_offset = [kv_lens[i] - qo_lens[i] for i in range(B)]

    kv_page_indptr = [0]
    acc = 0
    for k in kv_lens:
        acc += (k + page_size - 1) // page_size
        kv_page_indptr.append(acc)

    expanded_qo_lens = [pack_factor] * total_q
    expanded_kv_lens = [0] * total_q
    expanded_qo_offset = [0] * total_q
    expanded_kv_page_indptr = [0] * (total_q + 1)

    idx = 0
    for b in range(B):
        kv_len_b = kv_lens[b]
        qo_offset_b = qo_offset[b]
        kv_page_indptr_b = kv_page_indptr[b]
        for j in range(qo_lens[b]):
            expanded_kv_lens[idx] = kv_len_b
            expanded_qo_offset[idx] = qo_offset_b + j
            expanded_kv_page_indptr[idx] = kv_page_indptr_b
            idx += 1

    expanded_kv_page_indptr[total_q] = kv_page_indptr[B]

    return (
        expanded_qo_lens,
        expanded_kv_lens,
        expanded_qo_offset,
        expanded_kv_page_indptr,
    )


class PlanInfo(dict):
    """Execution plan returned by the internal dense FMHA planner.

    Users normally receive the public tuple returned by ``fmha_sm100_plan`` and
    pass it unchanged to ``fmha_sm100``.  The dictionary stores CUDA worklists,
    sequence metadata, split-KV workspaces, and cached buffers owned by the
    plan.
    """

    def __del__(self):
        pass
        # print("del")
        # _workspace_cache_per_plan.append(self["_ws_cache"])


def _make_plan_info(
    packed_work_range,
    packed_work_info,
    kv_tile_begin_indices,
    kv_tile_end_indices,
    kv_split_indices,
    num_kv_splits,
    workspace_o,
    workspace_lse,
    max_qo_len,
    predicted_speedup,
    num_kv_splits_per_row,
    qo_segment_offsets,
    kv_segment_offsets,
    kv_page_indptr,
    max_k_tiles,
    qo_segment_lens,
    kv_segment_lens,
    qo_offset,
    pack_factor,
    orig_num_qo_heads,
    qo_len_uniform,
    cute_workspace_buffer,
):
    # ws = _workspace_cache_per_plan.pop()
    # print("pop")
    return PlanInfo(
        {
            # "_ws_cache" : ws,
            "packed_work_range": packed_work_range,
            "packed_work_info": packed_work_info,
            "kv_tile_begin_indices": kv_tile_begin_indices,
            "kv_tile_end_indices": kv_tile_end_indices,
            "kv_split_indices": kv_split_indices,
            "num_kv_splits": num_kv_splits,
            "workspace_o": workspace_o,
            "workspace_lse": workspace_lse,
            "max_qo_len": max_qo_len,
            "predicted_speedup": predicted_speedup,
            "num_kv_splits_per_row": num_kv_splits_per_row,
            "qo_segment_offsets": qo_segment_offsets,
            "kv_segment_offsets": kv_segment_offsets,
            "kv_page_indptr": kv_page_indptr,
            "max_k_tiles": max_k_tiles,
            "qo_segment_lens": qo_segment_lens,
            "kv_segment_lens": kv_segment_lens,
            "qo_offset": qo_offset,
            "pack_factor": pack_factor,
            "orig_num_qo_heads": orig_num_qo_heads,
            "qo_len_uniform": qo_len_uniform,
            "cute_workspace_buffer": cute_workspace_buffer,
            "MM-SA-Nv": False,
        }
    )


def _call_plan(
    qo_segment_offsets,
    qo_segment_lens,
    kv_segment_lens,
    packed_work_range,
    packed_work_info,
    qo_tile_size,
    kv_tile_size,
    num_qo_heads,
    num_ctas,
    causal,
    qo_offset,
    num_kv_splits,
    kv_tile_begin_indices,
    kv_tile_end_indices,
    kv_split_indices,
    chunk_size,
    out_max_sm_cost,
    num_kv_splits_per_row,
    workspace_lse,
    lse_total_size,
    pack_factor,
    cuda_stream=None,
):

    plan_module = get_plan_fn()

    if cuda_stream is None:
        cuda_stream = torch.cuda.current_stream().cuda_stream
    plan_module.plan(
        qo_segment_offsets,
        qo_segment_lens,
        kv_segment_lens,
        packed_work_range,
        packed_work_info,
        qo_tile_size,
        kv_tile_size,
        num_qo_heads,
        num_ctas,
        causal,
        qo_offset,
        num_kv_splits,
        kv_tile_begin_indices,
        kv_tile_end_indices,
        kv_split_indices,
        chunk_size,
        out_max_sm_cost,
        num_kv_splits_per_row,
        cuda_stream,
        workspace_lse,
        lse_total_size,
        pack_factor,
    )


def _fmha_sm100_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int = -1,
    qo_offset: Optional[Union[int, torch.Tensor]] = None,
    num_kv_splits: int = -1,
    page_size: int = -1,
    output_maxscore: bool = False,
    kv_block_num: int = -1,
    usable_SM_count: int = -1,
    causal: bool = True,
    sparse_kernel_mode: str = "auto",
    use_fp8_kvcache: bool = False,
    device=None,
    stream=None,
):
    device = torch.cuda.current_device() if device is None else device
    _reset_np_staging()

    qo_lens = qo_segment_lens.tolist()
    max_qo_len_orig = max(qo_lens) if qo_lens else 0

    if kv_block_num > 0 and (
        sparse_kernel_mode == "prefill"
        or (
            sparse_kernel_mode == "auto"
            and max_qo_len_orig > _prefill_qlen_threshold(True)
        )
    ):
        # print("Nv-Prefill")
        qo_segment_lens = qo_segment_lens.to(device)
        kv_segment_lens = kv_segment_lens.to(device)
        qo_offset = qo_offset.to(device)
        return sparse_fmha_plan(
            qo_segment_lens=qo_segment_lens,
            kv_segment_lens=kv_segment_lens,
            num_qo_heads=num_qo_heads,
            causal=causal,
            qo_offset=qo_offset,
            num_kv_splits=num_kv_splits,
            page_size=page_size,
            output_maxscore=output_maxscore,
            kv_block_num=kv_block_num,
            num_kv_heads=num_kv_heads,
            usable_SM_count=usable_SM_count,
            use_fp8_kvcache=use_fp8_kvcache,
        )

    _new_ws_cache()

    cute_workspace_buffer = _alloc_workspace_buf(
        _BuffTag.fmha_sm100_cutlass_workspace, 32 * 1024 * 1024, device, torch.uint8
    )

    cuda_stream = stream

    num_ctas = _get_num_cta(device)
    if usable_SM_count > 0:
        num_ctas = min(usable_SM_count, num_ctas)

    orig_num_qo_heads = num_qo_heads
    pack_factor = _compute_pack_factor(max_qo_len_orig, num_qo_heads, num_kv_heads)
    qo_len_uniform = len(qo_lens) > 0 and min(qo_lens) == max_qo_len_orig
    if pack_factor > 1:
        num_qo_heads = num_qo_heads // pack_factor

    kv_lens = kv_segment_lens.tolist()
    kv_page_indptr_list = None
    if kv_block_num > 0 and page_size > 0:
        total_q = sum(qo_lens)
        total_q_packed = total_q * pack_factor if pack_factor > 1 else total_q
        assert total_q_packed * num_qo_heads <= 65536
        qo_offset_in = qo_offset.tolist() if qo_offset is not None else None
        qo_lens, kv_lens, qo_offset, kv_page_indptr_list = _expand_for_per_token_sparse(
            qo_lens, kv_lens, qo_offset_in, page_size, pack_factor
        )
    elif kv_block_num > 0 and page_size <= 0:
        print("[Error] Sparse mode must be used together with paged kv!")
    else:
        if pack_factor > 1:
            qo_lens = [q * pack_factor for q in qo_lens]
        if page_size > 0:
            acc = 0
            kv_page_indptr_list = [0]
            for k in kv_lens:
                acc += (k + page_size - 1) // page_size
                kv_page_indptr_list.append(acc)

    acc = 0
    qo_offsets = [0]
    for q in qo_lens:
        acc += q
        qo_offsets.append(acc)
    acc = 0
    kv_offsets = [0]
    for k in kv_lens:
        acc += k
        kv_offsets.append(acc)

    max_kv_len = max(kv_lens)
    qo_offset_list = qo_offset if isinstance(qo_offset, list) else qo_offset.tolist()
    if not causal:
        qo_offset_list = [max_kv_len] * len(kv_lens)

    qo_segment_offsets = _plan_buf_from_list(qo_offsets, device)
    kv_segment_offsets = _plan_buf_from_list(kv_offsets, device)
    qo_offset = _plan_buf_from_list(qo_offset_list, device)
    kv_segment_lens = _plan_buf_from_list(kv_lens, device)
    kv_page_indptr = (
        _plan_buf_from_list(kv_page_indptr_list, device)
        if kv_page_indptr_list is not None
        else None
    )

    plan_kv_lens_list = kv_lens
    plan_qo_offset = qo_offset
    if kv_block_num > 0 and page_size > 0:
        plan_kv_lens_list = [kv_block_num * page_size] * len(kv_lens)
        plan_qo_offset = None

    max_qo_len = max(qo_lens)
    qo_tile_size = 128 if max_qo_len <= 128 else 256
    kv_tile_size = 256 if qo_tile_size == 128 else 128

    total_qo_len = qo_offsets[-1]

    max_k_tiles = (
        math.ceil(math.ceil(max_kv_len / 128) / 128) * 128 if output_maxscore else -1
    )
    maxscore_elems = num_qo_heads * max_k_tiles * total_qo_len
    if output_maxscore and maxscore_elems > (1 << 31):
        print(f"Too huge setting to output maxscore!")
        max_k_tiles = -1

    if num_kv_splits < 1 and qo_tile_size == 128:
        group_iters = []
        batch_size = len(qo_lens)
        for b_idx in range(batch_size):
            ql_b, kl_b = qo_lens[b_idx], plan_kv_lens_list[b_idx]
            off_q = kl_b - ql_b
            for t in range(math.ceil(ql_b / qo_tile_size)):
                packed_q_end = (t + 1) * qo_tile_size
                q_end = (
                    (packed_q_end - 1) // pack_factor + 1
                    if pack_factor > 1
                    else packed_q_end
                )
                if causal and q_end + off_q <= 0:
                    continue
                eff_kv = min(q_end + off_q, kl_b) if causal else kl_b
                group_iters.append(math.ceil(max(eff_kv, 0) / kv_tile_size))

        if not group_iters:
            num_kv_splits = 1
        elif len(group_iters) * num_qo_heads > 4096:
            # Too many tiles for split-KV smem — fall back to nosplit greedy
            num_kv_splits = 1
        else:
            total_iters = sum(group_iters) * num_qo_heads
            avg_iters = max(2, (total_iters + num_ctas - 1) // num_ctas + 3)
            chunk_size = avg_iters
            max_pieces = max(math.ceil(g / chunk_size) for g in group_iters)
            max_kv_splits = min(2 * max(max_pieces, 1), 64)

            max_work_items = 131072 * max_kv_splits

            packed_work_range = _alloc_perplan_buf(
                _BuffTag.packed_work_range, num_ctas, device, torch.int64
            )
            packed_work_info = _alloc_perplan_buf(
                _BuffTag.packed_work_info, max_work_items, device, torch.int64
            )
            kv_tile_begin_indices = _alloc_perplan_buf(
                _BuffTag.kv_tile_begin_indices, max_work_items, device, torch.int32
            )
            kv_tile_end_indices = _alloc_perplan_buf(
                _BuffTag.kv_tile_end_indices, max_work_items, device, torch.int32
            )
            kv_split_indices = _alloc_perplan_buf(
                _BuffTag.kv_split_indices, max_work_items, device, torch.int32
            )
            num_kv_splits_per_row = _alloc_perplan_buf(
                _BuffTag.num_kv_splits_per_row, total_qo_len, device, torch.int32
            )
            plan_cost = _alloc_workspace_buf(
                _BuffTag.plan_cost, 2, device, dtype=torch.float32
            )

            lse_total_size = max_kv_splits * total_qo_len * num_qo_heads
            workspace_lse = _alloc_perplan_buf(
                _BuffTag.workspace_lse, lse_total_size, device, torch.float32
            )

            qo_segment_lens_gpu = _plan_buf_from_list(qo_lens, device)
            plan_kv_lens_gpu = _plan_buf_from_list(plan_kv_lens_list, device)
            _call_plan(
                qo_segment_offsets,
                qo_segment_lens_gpu,
                plan_kv_lens_gpu,
                packed_work_range,
                packed_work_info,
                qo_tile_size,
                kv_tile_size,
                num_qo_heads,
                num_ctas,
                causal,
                plan_qo_offset,
                -max_kv_splits,
                kv_tile_begin_indices,
                kv_tile_end_indices,
                kv_split_indices,
                chunk_size,
                plan_cost,
                num_kv_splits_per_row,
                workspace_lse,
                lse_total_size,
                pack_factor,
                cuda_stream,
            )

            _cost = plan_cost.tolist()
            do_split = _cost[0] > 0
            predicted_speedup = _cost[1]

            if do_split:
                workspace_o = _alloc_workspace_buf(
                    _BuffTag.workspace_o,
                    total_qo_len * max_kv_splits * num_qo_heads * 128,
                    device,
                    dtype=torch.bfloat16,
                )
            else:
                kv_tile_begin_indices = None
                kv_tile_end_indices = None
                kv_split_indices = None
                num_kv_splits_per_row = None
                workspace_o = None
                workspace_lse = None
                max_kv_splits = 1

            return _make_plan_info(
                packed_work_range=packed_work_range,
                packed_work_info=packed_work_info,
                kv_tile_begin_indices=kv_tile_begin_indices,
                kv_tile_end_indices=kv_tile_end_indices,
                kv_split_indices=kv_split_indices,
                num_kv_splits=max_kv_splits,
                workspace_o=workspace_o,
                workspace_lse=workspace_lse,
                max_qo_len=max_qo_len,
                predicted_speedup=predicted_speedup,
                num_kv_splits_per_row=num_kv_splits_per_row,
                qo_segment_offsets=qo_segment_offsets,
                kv_segment_offsets=kv_segment_offsets,
                kv_page_indptr=kv_page_indptr,
                max_k_tiles=max_k_tiles,
                qo_segment_lens=qo_segment_lens_gpu,
                kv_segment_lens=kv_segment_lens,
                qo_offset=qo_offset,
                pack_factor=pack_factor,
                orig_num_qo_heads=orig_num_qo_heads,
                qo_len_uniform=qo_len_uniform,
                cute_workspace_buffer=cute_workspace_buffer,
            )

        num_kv_splits = 1
    elif num_kv_splits < 1:
        num_kv_splits = 1
    elif qo_tile_size == 256:
        num_kv_splits = 1

    packed_work_range = _alloc_perplan_buf(
        _BuffTag.packed_work_range, num_ctas, device, torch.int64
    )
    max_work_items = 131072 * max(num_kv_splits, 1)
    packed_work_info = _alloc_perplan_buf(
        _BuffTag.packed_work_info, max_work_items, device, torch.int64
    )
    if num_kv_splits > 1:
        kv_tile_begin_indices = _alloc_perplan_buf(
            _BuffTag.kv_tile_begin_indices, max_work_items, device, torch.int32
        )
        kv_tile_end_indices = _alloc_perplan_buf(
            _BuffTag.kv_tile_end_indices, max_work_items, device, torch.int32
        )
        kv_split_indices = _alloc_perplan_buf(
            _BuffTag.kv_split_indices, max_work_items, device, torch.int32
        )
        num_kv_splits_per_row = _alloc_perplan_buf(
            _BuffTag.num_kv_splits_per_row, total_qo_len, device, torch.int32
        )
        workspace_o = _alloc_workspace_buf(
            _BuffTag.workspace_o,
            total_qo_len * num_kv_splits * num_qo_heads * 128,
            device,
            dtype=torch.bfloat16,
        )

        lse_total_size = num_kv_splits * total_qo_len * num_qo_heads
        workspace_lse = _alloc_perplan_buf(
            _BuffTag.workspace_lse, lse_total_size, device, torch.float32
        )
    else:
        kv_tile_begin_indices = None
        kv_tile_end_indices = None
        kv_split_indices = None
        num_kv_splits_per_row = None
        workspace_o = None

        lse_total_size = 0
        workspace_lse = None

    qo_segment_lens_gpu = _plan_buf_from_list(qo_lens, device)
    plan_kv_lens_gpu = _plan_buf_from_list(plan_kv_lens_list, device)
    _call_plan(
        qo_segment_offsets,
        qo_segment_lens_gpu,
        plan_kv_lens_gpu,
        packed_work_range,
        packed_work_info,
        qo_tile_size,
        kv_tile_size,
        num_qo_heads,
        num_ctas,
        causal,
        plan_qo_offset,
        num_kv_splits,
        kv_tile_begin_indices,
        kv_tile_end_indices,
        kv_split_indices,
        0,
        None,
        num_kv_splits_per_row,
        workspace_lse,
        lse_total_size,
        pack_factor,
        cuda_stream,
    )

    return _make_plan_info(
        packed_work_range=packed_work_range,
        packed_work_info=packed_work_info,
        kv_tile_begin_indices=kv_tile_begin_indices,
        kv_tile_end_indices=kv_tile_end_indices,
        kv_split_indices=kv_split_indices,
        num_kv_splits=num_kv_splits,
        workspace_o=workspace_o,
        workspace_lse=workspace_lse,
        max_qo_len=max_qo_len,
        predicted_speedup=1.0,
        num_kv_splits_per_row=num_kv_splits_per_row,
        qo_segment_offsets=qo_segment_offsets,
        kv_segment_offsets=kv_segment_offsets,
        kv_page_indptr=kv_page_indptr,
        max_k_tiles=max_k_tiles,
        qo_segment_lens=qo_segment_lens_gpu,
        kv_segment_lens=kv_segment_lens,
        qo_offset=qo_offset,
        pack_factor=pack_factor,
        orig_num_qo_heads=orig_num_qo_heads,
        qo_len_uniform=qo_len_uniform,
        cute_workspace_buffer=cute_workspace_buffer,
    )


def _fmha_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info,
    kv_indices: Optional[torch.Tensor] = None,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override: Optional[Union[int, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    o_scale: Optional[float] = None,
    output_maxscore: bool = True,
    output_o: bool = True,
    check_input_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    if plan_info["MM-SA-Nv"]:
        return sparse_fmha(
            q=q,
            k=k,
            v=v,
            plan_info=plan_info,
            out=out,
            max_score=max_score,
            sm_scale=sm_scale,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            o_scale=o_scale,
            kv_indices=kv_indices,
            output_maxscore=output_maxscore,
            output_o=output_o,
            q_offset_override=q_offset_override,
            kv_block_indexes=kv_block_indexes,
            check_input_valid=check_input_valid,
        )

    nnz_qo, num_qo_heads, head_dim_qk = q.shape
    if kv_indices is None:
        nnz_kv, num_kv_heads, head_dim_vo = v.shape
        is_paged = False
    else:
        nnz_kv, num_kv_heads, page_size, head_dim_vo = v.shape
        is_paged = True

    qo_total_len = nnz_qo

    packed_work_range = plan_info["packed_work_range"]
    packed_work_info = plan_info["packed_work_info"]
    kv_tile_begin_indices = plan_info["kv_tile_begin_indices"]
    kv_tile_end_indices = plan_info["kv_tile_end_indices"]
    kv_split_indices = plan_info["kv_split_indices"]
    num_kv_splits = plan_info["num_kv_splits"]
    workspace_o = plan_info["workspace_o"]
    workspace_lse = plan_info["workspace_lse"]
    plan_max_qo_len = plan_info["max_qo_len"]
    num_kv_splits_per_row = plan_info["num_kv_splits_per_row"]
    qo_segment_offsets = plan_info["qo_segment_offsets"]
    kv_segment_offsets = plan_info["kv_segment_offsets"]
    kv_page_indptr = plan_info["kv_page_indptr"]
    max_k_tiles = plan_info["max_k_tiles"]
    qo_segment_lens = plan_info["qo_segment_lens"]
    kv_segment_lens = plan_info["kv_segment_lens"]
    qo_offset = (
        plan_info["qo_offset"] if q_offset_override is None else q_offset_override
    )
    pack_factor = plan_info["pack_factor"]
    orig_num_qo_heads = plan_info["orig_num_qo_heads"]
    qo_len_uniform = plan_info["qo_len_uniform"]
    workspace_buffer = plan_info["cute_workspace_buffer"]

    batch_size = qo_segment_lens.shape[0]

    if isinstance(qo_offset, int):
        qo_offset = torch.full_like(qo_segment_lens, qo_offset)
    elif qo_offset is not None:
        assert qo_offset.device == qo_segment_lens.device

    if check_input_valid:
        _validate_fmha_inputs(
            q,
            k,
            v,
            qo_segment_lens,
            kv_segment_lens,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            head_dim_vo,
            batch_size,
            qo_total_len,
            is_paged,
            kv_indices,
            kv_block_indexes,
            qo_offset,
            page_size if is_paged else 0,
            pack_factor=pack_factor,
            packed_work_range=packed_work_range,
            packed_work_info=packed_work_info,
            num_kv_splits=num_kv_splits,
            kv_tile_begin_indices=kv_tile_begin_indices,
            kv_tile_end_indices=kv_tile_end_indices,
            kv_split_indices=kv_split_indices,
            qo_segment_offsets=qo_segment_offsets,
            kv_page_indptr=kv_page_indptr,
        )

    if pack_factor > 1 and orig_num_qo_heads is not None:
        num_qo_heads = orig_num_qo_heads // pack_factor
        qo_total_len = nnz_qo * pack_factor

    max_qo_len = plan_max_qo_len
    qo_tile_size = 128 if max_qo_len <= 128 else 256

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim_qk)
    if q_scale is None:
        q_scale = 1.0
    if k_scale is None:
        k_scale = 1.0
    if v_scale is None:
        v_scale = 1.0
    if o_scale is None:
        o_scale = 1.0

    assert output_o or output_maxscore

    use_split_kv = num_kv_splits > 1 and workspace_o is not None

    if not output_maxscore or max_k_tiles == -1:
        max_score = None
        output_o = True
    elif max_score is None and max_k_tiles > 0:
        max_score = torch.full(
            (orig_num_qo_heads, max_k_tiles, nnz_qo),
            -float("inf"),
            dtype=torch.float32,
            device=q.device,
        )
    elif max_score is not None and max_k_tiles > 0:
        unpacked_t = nnz_qo
        unpacked_h = orig_num_qo_heads
        packed_t = qo_total_len
        packed_h = num_qo_heads
        valid_max_score_shapes = {
            (unpacked_h, max_k_tiles, unpacked_t),  # legacy [H, K, T]
            (unpacked_t, unpacked_h, max_k_tiles),  # row-contiguous [T, H, K]
            (packed_h, max_k_tiles, packed_t),
            (packed_t, packed_h, max_k_tiles),
        }
        assert (
            max_score.dtype == torch.float32
        ), f"max_score must be float32, got {max_score.dtype}"
        assert (
            max_score.device == q.device
        ), f"max_score must be on {q.device}, got {max_score.device}"
        assert tuple(max_score.shape) in valid_max_score_shapes, (
            "max_score must have shape [H,K,T] or [T,H,K]; "
            f"got {tuple(max_score.shape)}, expected one of "
            f"{sorted(valid_max_score_shapes)}"
        )

    if not output_o:
        out = None
    elif out is None:
        out_dtype = torch.bfloat16 if q.dtype.itemsize == 1 else q.dtype
        out = torch.empty(
            nnz_qo,
            orig_num_qo_heads,
            head_dim_vo,
            device=q.device,
            dtype=out_dtype,
        )

    # Determine variant dispatch parameters
    dtype_code = _dlpack_dtype_code(q.dtype)

    kv_block_indexes_ptr_exists = kv_block_indexes is not None
    max_score_exists = max_score is not None
    o_exists = out is not None
    if kv_block_indexes_ptr_exists:
        sparse_mode = 0
    elif max_score_exists and o_exists:
        sparse_mode = 1
    elif max_score_exists:
        sparse_mode = 2
    else:
        sparse_mode = 3

    variant_page_size = k.shape[2] if is_paged else -1

    variant_module = get_fmha_variant(
        dtype_code,
        qo_tile_size,
        (max_qo_len <= 64),
        sparse_mode,
        variant_page_size,
        use_split_kv,
        pack_factor,
    )

    variant_module.run(
        workspace_buffer,
        q,
        k,
        v,
        qo_segment_lens,
        kv_segment_lens,
        qo_segment_offsets,
        kv_segment_offsets,
        packed_work_range,
        packed_work_info,
        out,
        sm_scale,
        q_scale,
        k_scale,
        v_scale,
        o_scale,
        max_qo_len,
        qo_offset,
        num_kv_splits,
        kv_tile_begin_indices,
        kv_tile_end_indices,
        kv_split_indices,
        workspace_o,
        workspace_lse,
        num_kv_splits_per_row,
        qo_tile_size,
        kv_indices,
        kv_page_indptr,
        max_score,
        max_k_tiles,
        kv_block_indexes,
        pack_factor,
        bool(qo_len_uniform),
        torch.cuda.current_stream().cuda_stream,
    )

    # Split-KV reduction
    if use_split_kv and out is not None:
        log2_e = math.log2(math.exp(1.0))
        scale_softmax_log2 = float(q_scale * k_scale * sm_scale) * log2_e
        inv_scale_o = float(o_scale)

        reduction_module = get_reduction_module()
        reduction_module.reduction(
            workspace_o,
            out,
            workspace_lse,
            num_kv_splits_per_row,
            scale_softmax_log2,
            inv_scale_o,
            num_kv_splits,
            qo_total_len,
            num_qo_heads,
            head_dim_vo,
            num_qo_heads * head_dim_vo,
            head_dim_vo,
            num_qo_heads * head_dim_vo,
            head_dim_vo,
            orig_num_qo_heads,
            num_kv_heads,
            pack_factor,
            torch.cuda.current_stream().cuda_stream,
        )

    return out, max_score


def fmha_sm100_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    *args,
    qo_offset: Optional[Union[int, torch.Tensor]] = None,
    split_prefill_decode=True,
    **kwargs,
):
    """Build a reusable execution plan for ``fmha_sm100``.

    The plan is shape-dependent and can be reused across layers or repeated
    calls that share the same sequence lengths, head counts, page size, sparse
    mode, and output mode.  Planning may run CUDA kernels and allocates
    workspaces, so it should be done outside tight per-layer loops when
    possible.

    Parameters
    ----------
    qo_segment_lens : torch.Tensor
        Shape ``[batch_size]``, dtype int32/int64.  Per-request Q/O lengths.
        CPU tensors are accepted for the dense planner; sparse prefill planning
        moves them to CUDA internally.
    kv_segment_lens : torch.Tensor
        Shape ``[batch_size]``, dtype int32/int64.  Per-request KV lengths.
    *args
        Positional arguments forwarded to the internal planner.  In normal
        usage this is ``num_qo_heads`` and optionally ``num_kv_heads``.
    qo_offset : int or torch.Tensor, optional
        Per-request causal offset.  If omitted, defaults to
        ``kv_segment_lens - qo_segment_lens`` for bottom-right causal masking.
        A tensor must have shape ``[batch_size]``.
    split_prefill_decode : bool, optional
        If True, a mixed batch ordered as decode requests followed by prefill
        requests is split into two sub-plans.  The original order must already
        group short decode sequences before long prefill sequences.
    **kwargs
        Planner options forwarded to ``_fmha_sm100_plan``.  Common options are
        ``num_kv_heads``, ``num_kv_splits``, ``page_size``,
        ``output_maxscore``, ``kv_block_num``, ``usable_SM_count``, ``causal``,
        ``sparse_kernel_mode``, ``use_fp8_kvcache``, ``device``, and ``stream``.

    Returns
    -------
    tuple
        ``(has_mixed_prefill, split, batch_size, decode_plan, prefill_plan)``.
        Pass this tuple unchanged as ``plan_info`` to ``fmha_sm100``.
    """

    # assert qo_segment_lens.device.type == 'cpu' \
    #         and kv_segment_lens.device.type == 'cpu'
    # assert qo_offset is None or isinstance(qo_offset, int) or qo_offset.device.type == 'cpu'

    if qo_offset is None:
        qo_offset = kv_segment_lens - qo_segment_lens
    elif isinstance(qo_offset, int):
        qo_offset = torch.full_like(qo_segment_lens, qo_offset)

    batch_size = qo_segment_lens.shape[0]
    has_mixed_prefill = False
    qmax = qo_segment_lens.max().item()
    sparse = kwargs.get("kv_block_num", -1) > 0
    split_threshold = _prefill_qlen_threshold(sparse)
    if split_prefill_decode and qmax > split_threshold:
        split = (qo_segment_lens > split_threshold).nonzero(as_tuple=False)[0, 0].item()
        has_mixed_prefill = split > 0
    if has_mixed_prefill:
        # print(f"Split into 2 parts at index {split}")
        decode_qo_segment_lens = qo_segment_lens[:split]
        decode_kv_segment_lens = kv_segment_lens[:split]
        decode_qo_offset = qo_offset[:split]
        decode = _fmha_sm100_plan(
            decode_qo_segment_lens,
            decode_kv_segment_lens,
            *args,
            qo_offset=decode_qo_offset,
            **kwargs,
        )
        decode = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in decode.items()
        }
        prefill_qo_segment_lens = qo_segment_lens[split:]
        prefill_kv_segment_lens = kv_segment_lens[split:]
        prefill_qo_offset = qo_offset[split:]
        prefill = _fmha_sm100_plan(
            prefill_qo_segment_lens,
            prefill_kv_segment_lens,
            *args,
            qo_offset=prefill_qo_offset,
            **kwargs,
        )
        return (True, split, batch_size, decode, prefill)
    else:
        plan = _fmha_sm100_plan(
            qo_segment_lens, kv_segment_lens, *args, qo_offset=qo_offset, **kwargs
        )
        return (False, 0, batch_size, plan, None)


def fmha_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info,
    kv_indices: Optional[torch.Tensor] = None,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override: Optional[Union[int, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run dense, paged, or sparse SM100 FMHA using a precomputed plan.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``[total_qo_len, num_qo_heads, head_dim]``.  Supported dtypes are
        ``torch.bfloat16`` and ``torch.float8_e4m3fn``.  ``head_dim`` must be
        128.
    k : torch.Tensor
        Dense layout ``[total_kv_len, num_kv_heads, head_dim]`` or paged layout
        ``[total_pages, num_kv_heads, page_size, head_dim]``.
    v : torch.Tensor
        Same layout as ``k``.  The output head dimension follows ``v.shape[-1]``.
    plan_info : tuple
        Return value from ``fmha_sm100_plan`` for the same lengths, head layout,
        page size, and sparse/output mode.
    kv_indices : torch.Tensor, optional
        Paged-KV physical page table, flattened across the batch.  Required when
        ``k`` and ``v`` use paged layout.  Shape is ``[sum_pages]`` and dtype is
        int32.
    kv_block_indexes : torch.Tensor, optional
        Sparse KV block indices from ``sparse_topk_select``.  Shape
        ``[total_qo_len, num_kv_heads or num_qo_heads, kv_block_num]``, dtype
        int32, ascending per row with ``-1`` padding at the tail.
    q_offset_override : int or torch.Tensor, optional
        Runtime causal-offset override.  Tensor form must have shape
        ``[batch_size]`` on the same CUDA device as the plan metadata.  The
        override must stay within the causal range visible to the original
        plan.
    out : torch.Tensor, optional
        Preallocated output buffer with shape
        ``[total_qo_len, num_qo_heads, head_dim_v]``.
    max_score : torch.Tensor, optional
        Preallocated per-KV-tile score buffer with dtype float32.  Accepted
        layouts are legacy ``[num_qo_heads, max_k_tiles, total_qo_len]`` and
        row-contiguous ``[total_qo_len, num_qo_heads, max_k_tiles]``.
    **kwargs
        Runtime options forwarded to the kernel runner.  Common options are
        ``sm_scale``, ``q_scale``, ``k_scale``, ``v_scale``, ``o_scale``,
        ``output_maxscore``, ``output_o``, and ``check_input_valid``.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        ``(out, max_score)``.  Either item may be ``None`` if the corresponding
        output was disabled.  When both decode and prefill sub-plans are used,
        outputs are concatenated back into the original batch order.
    """
    has_mixed_prefill, split, batch_size, decode, prefill = plan_info
    if not has_mixed_prefill:
        return _fmha_sm100(
            q,
            k,
            v,
            decode,
            out=out,
            max_score=max_score,
            kv_indices=kv_indices,
            kv_block_indexes=kv_block_indexes,
            q_offset_override=q_offset_override,
            **kwargs,
        )
    else:

        decode_pack = decode.get("pack_factor", 1)
        decode_nnz = decode["qo_segment_offsets"][-1].item() // decode_pack
        is_paged = kv_indices is not None
        nnz_qo = q.shape[0]
        num_qo_heads = q.shape[1]

        q_decode = q[:decode_nnz]
        q_prefill = q[decode_nnz:]

        if is_paged:
            k_decode, v_decode = k, v
            k_prefill, v_prefill = k, v
            if "kv_page_indptr" in decode:
                kv_page_split = decode["kv_page_indptr"][-1].item()
            else:
                kv_page_split = decode["total_rows"]
            decode_kv_indices = kv_indices[:kv_page_split]
            prefill_kv_indices = kv_indices[kv_page_split:]
        else:
            if "kv_segment_offsets" in decode:
                decode_kv_nnz = decode["kv_segment_offsets"][-1].item()
            else:
                decode_kv_nnz = decode["cu_seqlens_k"][-1].item()
            k_decode, k_prefill = k[:decode_kv_nnz], k[decode_kv_nnz:]
            v_decode, v_prefill = v[:decode_kv_nnz], v[decode_kv_nnz:]
            decode_kv_indices = None
            prefill_kv_indices = None

        decode_block_idx = (
            kv_block_indexes[:decode_nnz] if kv_block_indexes is not None else None
        )
        prefill_block_idx = (
            kv_block_indexes[decode_nnz:] if kv_block_indexes is not None else None
        )
        if isinstance(q_offset_override, int):
            decode_qo_offset = torch.full(
                (split,), q_offset_override, dtype=torch.int32, device=q.device
            )
            prefill_qo_offset = torch.full(
                (batch_size - split,),
                q_offset_override,
                dtype=torch.int32,
                device=q.device,
            )
        elif q_offset_override is not None:
            decode_qo_offset = q_offset_override[:split]
            prefill_qo_offset = q_offset_override[split:]
        else:
            decode_qo_offset = None
            prefill_qo_offset = None

        # ---- Run kernels ----
        decode_out, decode_ms = _fmha_sm100(
            q_decode,
            k_decode,
            v_decode,
            decode,
            out=None,
            max_score=None,
            kv_indices=decode_kv_indices,
            kv_block_indexes=decode_block_idx,
            q_offset_override=decode_qo_offset,
            **kwargs,
        )
        prefill_out, prefill_ms = _fmha_sm100(
            q_prefill,
            k_prefill,
            v_prefill,
            prefill,
            out=None,
            max_score=None,
            kv_indices=prefill_kv_indices,
            kv_block_indexes=prefill_block_idx,
            q_offset_override=prefill_qo_offset,
            **kwargs,
        )

        # ---- Merge out ----
        if decode_out is not None and prefill_out is not None:
            combined_out = torch.cat([decode_out, prefill_out], dim=0)
        else:
            combined_out = None

        if out is not None and combined_out is not None:
            out.copy_(combined_out)

        # ---- Merge max_score ----
        if decode_ms is not None and prefill_ms is not None:
            d_kt, p_kt = decode_ms.shape[1], prefill_ms.shape[1]
            max_kt = max(d_kt, p_kt)
            combined_ms = torch.full(
                (num_qo_heads, max_kt, nnz_qo),
                -float("inf"),
                dtype=torch.float32,
                device=q.device,
            )
            combined_ms[:, :d_kt, :decode_nnz] = decode_ms
            combined_ms[:, :p_kt, decode_nnz:] = prefill_ms
        else:
            combined_ms = decode_ms if decode_ms is not None else prefill_ms

        if max_score is not None and combined_ms is not None:
            if max_score.shape == combined_ms.shape:
                max_score.copy_(combined_ms)
            elif max_score.shape == (nnz_qo, num_qo_heads, combined_ms.shape[1]):
                max_score.copy_(combined_ms.permute(2, 0, 1).contiguous())
            else:
                raise ValueError(
                    f"max_score shape {tuple(max_score.shape)} is incompatible "
                    f"with combined max_score shape {tuple(combined_ms.shape)}"
                )

        return (
            out if out is not None else combined_out,
            max_score if max_score is not None else combined_ms,
        )


# ============================================================================
# Sparse TopK Select
# ============================================================================


def sparse_topk_select(
    max_score: torch.Tensor,
    topk: int,
    num_valid_pages: Optional[Union[int, torch.Tensor]] = None,
    output: Optional[torch.Tensor] = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
    max_score_layout: str = "HKT",
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Select top-k KV-tile indices per (qo_head, token) row from the FMHA max-score tensor.

    Designed for the MQA proxy-KV sparse attention path where the dense pass uses
    ``num_kv_heads_dense=1``, so ``max_score.shape[0] == num_kv_heads_real`` and
    each head row is processed independently (no GQA reduction inside this call).

    Parameters
    ----------
    max_score : torch.Tensor
        Contiguous float32 max-score tensor.  ``max_score_layout="HKT"`` expects
        shape ``(num_qo_heads, max_k_tiles, total_qo_len)``.  ``"THK"`` expects
        shape ``(total_qo_len, num_qo_heads, max_k_tiles)`` and skips the
        internal transpose before top-k.  Slots beyond the actual KV tile count
        must be pre-filled with ``-inf`` (fmha_sm100 does this automatically via
        ``torch.full``).
    topk : int
        Must be exactly 16.
    num_valid_pages : int or torch.Tensor, optional
        Actual number of KV pages in the page table, i.e. ``ceil(kv_len / page_size)``.
        ``max_k_tiles`` is round-up-aligned and always >= ``num_valid_pages``.
        The kernel may select tile indices in ``[num_valid_pages, max_k_tiles-1]``
        (all-``-inf`` padding tiles). Passing ``num_valid_pages`` replaces those
        out-of-range indices with ``-1`` and sorts them to the tail, matching the
        sparse FMHA kernel's kv_block_indexes contract.
        Tensor form must be CUDA int32/int64 with shape ``[total_qo_len]`` and
        provides a per-query-token page count for mixed-length batches.
        **Strongly recommended**: omitting this allows OOB page-table accesses in
        the sparse attention pass.
    force_begin_blocks : int
        Number of KV blocks at the beginning of the sequence (indices 0..N-1) to
        always include in the top-k result, regardless of their scores.  Useful
        for sink tokens.  Default 0.
    force_end_blocks : int
        Number of KV blocks at the end of the valid sequence (indices
        nvp-N..nvp-1, closest to the current query) to always include.  Useful
        for local-window attention.  Default 0.
    block_table : torch.Tensor, optional
        Optional int32 tensor with shape ``(total_qo_len, num_qo_heads, max_k_tiles)``.
        The kernel still selects and sorts logical tile indices, but after sorting
        each logical index ``idx`` is replaced with ``block_table[t, h, idx]`` in
        the output.  Use this for per-token/per-head physical page-table gathers.

    Returns
    -------
    torch.Tensor
        Shape ``(total_qo_len, num_qo_heads, topk)``, int32.  Without
        ``block_table``, values are logical tile indices in ascending tile order.
        With ``block_table``, values are gathered block-table entries after that
        logical ascending sort.  Out-of-range entries (if any) are ``-1`` at the tail.
    """

    assert (
        max_score.dtype == torch.float32
    ), f"max_score must be float32, got {max_score.dtype}"
    assert max_score.dim() == 3, f"max_score must be 3D, got {max_score.shape}"
    assert max_score.is_contiguous(), "max_score must be contiguous"
    assert topk == 16, f"topk must be 16, got {topk}"

    layout = max_score_layout.upper()
    assert layout in {
        "HKT",
        "THK",
    }, f"max_score_layout must be 'HKT' or 'THK', got {max_score_layout!r}"
    if layout == "HKT":
        num_qo_heads, max_k_tiles, total_qo_len = max_score.shape
        layout_arg = 0
    else:
        total_qo_len, num_qo_heads, max_k_tiles = max_score.shape
        layout_arg = 1

    if block_table is not None:
        assert (
            block_table.dtype == torch.int32
        ), f"block_table must be int32, got {block_table.dtype}"
        assert (
            block_table.device == max_score.device
        ), f"block_table must be on {max_score.device}, got {block_table.device}"
        assert block_table.dim() == 3, (
            f"block_table must be 3D [total_qo_len, num_qo_heads, max_k_tiles], "
            f"got {tuple(block_table.shape)}"
        )
        assert tuple(block_table.shape) == (total_qo_len, num_qo_heads, max_k_tiles), (
            f"block_table shape must be {(total_qo_len, num_qo_heads, max_k_tiles)}, "
            f"got {tuple(block_table.shape)}"
        )
        assert all(
            s >= 0 for s in block_table.stride()
        ), f"block_table must have non-negative strides, got {block_table.stride()}"

    # v2.3 kernel only supports the insertion-sort path (K < 12288).
    assert max_k_tiles < 12288, (
        f"max_k_tiles={max_k_tiles} >= 12288: v2.3 kernel only supports K < 12288 "
        f"(radix-sort path not yet implemented). kv_len must be < {12288 * 128} tokens."
    )

    nvp_tensor = None
    if isinstance(num_valid_pages, torch.Tensor):
        assert (
            num_valid_pages.dim() == 1
        ), f"num_valid_pages tensor must be 1D [total_qo_len], got {tuple(num_valid_pages.shape)}"
        assert num_valid_pages.shape[0] == total_qo_len, (
            f"num_valid_pages tensor length {num_valid_pages.shape[0]} must match "
            f"total_qo_len={total_qo_len}"
        )
        assert (
            num_valid_pages.device == max_score.device
        ), f"num_valid_pages tensor must be on {max_score.device}, got {num_valid_pages.device}"
        assert num_valid_pages.dtype in (
            torch.int32,
            torch.int64,
        ), f"num_valid_pages tensor must be int32 or int64, got {num_valid_pages.dtype}"
        nvp_tensor = num_valid_pages.to(dtype=torch.int32).contiguous()
        nvp_arg = int(max_k_tiles)
    elif num_valid_pages is not None:
        nvp_arg = int(num_valid_pages)
        assert (
            0 < nvp_arg <= max_k_tiles
        ), f"num_valid_pages={nvp_arg} must be in (0, max_k_tiles={max_k_tiles}]"
    else:
        # v2.5_oob_clamp_in_kernel: kernel takes a unified num_valid_pages arg.
        # When the caller doesn't supply one, pass max_k_tiles so the in-kernel
        # `idx >= num_valid_pages` check never triggers (idx is always in
        # [0, max_k_tiles)).
        nvp_arg = int(max_k_tiles)

    assert force_begin_blocks >= 0 and force_end_blocks >= 0, (
        f"force_begin_blocks={force_begin_blocks} and force_end_blocks={force_end_blocks} "
        f"must be non-negative"
    )
    assert force_begin_blocks + force_end_blocks <= topk, (
        f"force_begin_blocks({force_begin_blocks}) + force_end_blocks({force_end_blocks}) "
        f"= {force_begin_blocks + force_end_blocks} exceeds topk={topk}"
    )

    # HKT needs a transpose buffer; THK is already row-contiguous over K and
    # should not allocate or pass a dummy workspace, especially under CUDA graph
    # capture.
    workspace_size = 0 if layout == "THK" else num_qo_heads * max_k_tiles * total_qo_len
    workspace_buffer = None
    if workspace_size:
        workspace_buffer = _alloc_workspace_buf(
            _BuffTag.sparse_topk_workspace,
            workspace_size,
            max_score.device,
            torch.int32,
        )

    if output is not None:
        assert output.dtype == torch.int32, f"output must be int32, got {output.dtype}"
        assert (
            output.device == max_score.device
        ), f"output must be on {max_score.device}, got {output.device}"
        assert output.dim() == 3, f"output must be 3D, got {tuple(output.shape)}"
        assert tuple(output.shape) == (total_qo_len, num_qo_heads, topk), (
            f"output shape must be {(total_qo_len, num_qo_heads, topk)}, "
            f"got {tuple(output.shape)}"
        )
        output_indices = output
    else:
        output_indices = torch.empty(
            total_qo_len,
            num_qo_heads,
            topk,
            dtype=torch.int32,
            device=max_score.device,
        )

    module = get_sparse_topk_module()
    # MQA dense pass: num_kv_heads_dense=1, so h_r=num_qo_heads.
    # The kernel only uses num_qo_heads = h_r * num_kv_heads as a product;
    # passing num_kv_heads=1 is equivalent to any other valid factorisation.
    #
    # v2.5_oob_clamp_in_kernel: OOB clamp is folded into the kernel — the prior
    # post-process torch.where + sort + torch.where chain (~84-101 us / call)
    # is replaced by passing num_valid_pages directly to the kernel.
    module.sparse_topk_select(
        max_score,
        output_indices,
        workspace_buffer,
        block_table,
        topk,
        nvp_arg,
        nvp_tensor,
        int(force_begin_blocks),
        int(force_end_blocks),
        layout_arg,
        torch.cuda.current_stream().cuda_stream,
    )

    return output_indices
