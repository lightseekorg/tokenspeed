"""Traffic-simulation stress harness for TokenSpeed.

Run as a module from the repo root: ``python -m test.stress run ...``.

Modules:
    events   - structured event dataclasses + JSONL sink
    metrics  - aggregation, percentiles, post-run summary
    client   - aiohttp client for /v1/chat/completions (streaming + cancel-aware)
    runner   - arrival processes + per-request lifecycle + circuit breaker
    launcher - optional ``--launch-cmd`` server lifecycle manager
    workloads/ - pluggable traffic generators (shared_prefix, long_context, cancel_mix, ...)
    monitors/  - background probes (health, rss)
"""
