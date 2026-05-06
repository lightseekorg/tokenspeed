from tokenspeed_scheduler import (
    PagedCacheGroupConfig,
    PagedCacheRetention,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)

PAGE_SIZE = 16
SWA_GROUP_ID = "v4.swa"
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


def _swa_table_for_request(plan, request_id: str) -> list[int]:
    for op in plan.forward:
        if request_id not in list(op.request_ids):
            continue
        row = list(op.request_ids).index(request_id)
        table = dict(op.paged_cache_block_tables).get(SWA_GROUP_ID)
        if table is None:
            return []
        return list(table[row])
    return []


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
        table = _swa_table_for_request(plan, "r0")
        if not table:
            break
        non_empty_chunks += 1

        raw_per_page = SWA_ROWS_PER_PAGE
        page_index = raw_pos // raw_per_page
        if page_index < len(table):
            assert table[page_index] >= 0

        raw_pos += MAX_SCHEDULED_TOKENS
        if raw_pos >= PROMPT_LEN:
            break

    assert non_empty_chunks >= 2
