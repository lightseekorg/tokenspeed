from tokenspeed_scheduler import (
    PagedCacheGroupConfig,
    PagedCacheRetention,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)

PAGE_SIZE = 16
SWA_GROUP_ID = "v4.swa_kv"
SWA_ROWS_PER_PAGE = 8
SWA_WINDOW_TOKENS = 32
PROMPT_LEN = 96
MAX_SCHEDULED_TOKENS = 32


def _scheduler_config() -> SchedulerConfig:
    cfg = SchedulerConfig()
    cfg.page_size = PAGE_SIZE
    cfg.max_scheduled_tokens = MAX_SCHEDULED_TOKENS
    cfg.max_batch_size = 4
    cfg.num_device_pages = 64
    cfg.disable_l2_cache = True
    cfg.paged_cache_groups = [
        PagedCacheGroupConfig(
            group_id=SWA_GROUP_ID,
            rows_per_page=SWA_ROWS_PER_PAGE,
            entry_stride_tokens=1,
            total_pages=16,
            retention=PagedCacheRetention.SlidingWindow,
            sliding_window_tokens=SWA_WINDOW_TOKENS,
        )
    ]
    return cfg


def _swa_table_for_request(plan, request_id: str) -> tuple[list[int], int]:
    """Return (compact_page_ids, base_logical_page) for the request's SWA row."""
    for op in plan.forward:
        if request_id not in list(op.request_ids):
            continue
        row = list(op.request_ids).index(request_id)
        table = dict(op.paged_cache_block_tables).get(SWA_GROUP_ID)
        if table is None:
            return [], 0
        offsets = dict(op.paged_cache_block_table_base_offsets).get(SWA_GROUP_ID)
        base = int(offsets[row]) if offsets is not None else 0
        return list(table[row]), base
    return [], 0


def _absolute_page_id(compact: list[int], base: int, abs_idx: int) -> int:
    """Resolve absolute logical page idx; -1 if released or beyond live tail."""
    rel = abs_idx - base
    if rel < 0 or rel >= len(compact):
        return -1
    return int(compact[rel])


def test_chunked_prefill_keeps_overlap_swa_pages_for_each_chunk():
    scheduler = Scheduler(_scheduler_config())
    spec = RequestSpec()
    spec.request_id = "r0"
    spec.tokens = list(range(PROMPT_LEN))
    scheduler.submit_requests([spec])

    raw_pos = 0
    non_empty_chunks = 0
    for _ in range(8):
        plan = scheduler.next_execution_plan()
        table, base = _swa_table_for_request(plan, "r0")
        if not table:
            break
        non_empty_chunks += 1

        raw_per_page = SWA_ROWS_PER_PAGE
        page_index = raw_pos // raw_per_page
        if page_index >= base:
            page_id = _absolute_page_id(table, base, page_index)
            assert (
                page_id >= 0
            ), f"chunk_start={raw_pos} found a released absolute page {page_index} (compact={table}, base={base})"

        raw_pos += MAX_SCHEDULED_TOKENS
        if raw_pos >= PROMPT_LEN:
            break

    assert non_empty_chunks >= 2
