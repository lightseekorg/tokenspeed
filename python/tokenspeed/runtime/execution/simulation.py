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

"""Allocator-owned decode fixtures for direct simulated-forward benchmarks.

Only the benchmark driver calls this: it never publishes synthetic prefixes to
an engine serving real requests. Device cache initialization and request-state
seeding must finish before executing ``decode_plan``.
"""

from dataclasses import dataclass

from tokenspeed_scheduler import (
    ExecutionEvent,
    ExecutionPlan,
    ForwardEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)


@dataclass(frozen=True)
class SimulatedDecodeFixture:
    # Retain the allocator for as long as a forward can address these pages.
    scheduler: Scheduler
    allocation_rounds: int
    decode_plan: ExecutionPlan
    context_lengths: tuple[int, ...]
    next_token_ids: tuple[int, ...]
    prompt_tokens: tuple[tuple[int, ...], ...]

    def ngram_inputs(self, context_len: int):
        """Build the ordinary Engram snapshot at the synthetic decode boundary."""
        from tokenspeed.runtime.execution.types import NGramInputs

        if context_len < 0:
            raise ValueError("Engram context length must be nonnegative")
        rows = []
        for prompt, token in zip(self.prompt_tokens, self.next_token_ids, strict=True):
            tail = tuple(reversed(prompt[-context_len:])) if context_len else ()
            rows.append((token, *tail, *([-1] * (context_len - len(tail)))))
        return NGramInputs(tokens=tuple(rows), positions=self.context_lengths)


def allocate_simulated_decode(
    config: SchedulerConfig,
    prompts: list[list[int]],
    next_token_ids: list[int],
    *,
    max_new_tokens: int,
) -> SimulatedDecodeFixture:
    """Allocate a complete synthetic prefix through the normal scheduler.

    ``config`` must describe the device arena's actual cache groups. Its token
    budget bounds each CPU-only allocation round, exactly as chunked prefill
    would. These rounds never enter the model or allocate prefill activations.
    Holding final feedback until all prompts finish prevents early requests
    from decoding while later requests are still being allocated.

    The caller sanitizes all newly allocated cache payloads, seeds per-request
    token and length state, and then executes ``decode_plan`` through the same
    device handle as a real forward. This function supplies no invented page
    addresses and leaves SWA, compression and recurrent-state geometry to LCM.
    """
    if not prompts or len(prompts) != len(next_token_ids):
        raise ValueError("Each nonempty prompt batch needs one next token per request")
    if any(not prompt for prompt in prompts):
        raise ValueError("Simulation needs a positive context length")
    if config.role != SchedulerConfig.Role.Fused or config.overlap_schedule_depth != 0:
        raise ValueError(
            "Direct-forward fixtures require fused scheduling without overlap"
        )
    if not config.disable_prefix_cache:
        raise ValueError("Synthetic fixtures require private prefixes")
    if config.max_batch_size < len(prompts):
        raise ValueError("Fixture allocation must admit the whole decode batch")
    # DeviceSpecs uses zero when L2 is disabled; a null-only host pool is also
    # empty. Snapshot pools always include their null page, even when disabled.
    if (
        config.num_host_pages not in (0, 1)
        or config.num_snapshot_pages != 1
        or config.enable_l3_storage
    ):
        raise ValueError(
            "Synthetic fixtures require device-only cache without retraction; "
            f"got host_pages={config.num_host_pages}, "
            f"snapshot_pages={config.num_snapshot_pages}, "
            f"l3_storage={config.enable_l3_storage}"
        )
    if max_new_tokens < config.decode_input_tokens:
        raise ValueError("Output reservation must cover the decode width")

    scheduler = Scheduler(config)
    specs = []
    for i, prompt in enumerate(prompts):
        spec = RequestSpec()
        spec.request_id = f"simulate-{i}"
        spec.tokens = prompt
        spec.max_new_tokens = max_new_tokens
        specs.append(spec)
    scheduler.submit_requests(specs)
    lengths = {
        spec.request_id: len(prompt)
        for spec, prompt in zip(specs, prompts, strict=True)
    }
    tokens = {
        spec.request_id: token
        for spec, token in zip(specs, next_token_ids, strict=True)
    }
    prompt_by_id = {
        spec.request_id: tuple(prompt)
        for spec, prompt in zip(specs, prompts, strict=True)
    }
    completed = {}
    rounds = 0
    while len(completed) < len(prompts):
        plan = scheduler.next_execution_plan()
        if len(plan.forward) != 1 or not plan.forward[0].request_ids:
            raise RuntimeError(
                "Cache capacity could not admit complete simulated prefixes"
            )
        rounds += 1
        batch = plan.forward[0]
        if batch.num_extends() != len(batch.request_ids):
            raise RuntimeError(
                "Cache capacity or scheduling policy prevented complete prefix allocation"
            )
        event = ExecutionEvent()
        for rid, prefix, count in zip(
            batch.request_ids,
            batch.extend_prefix_lens,
            batch.input_lengths,
            strict=True,
        ):
            result = ForwardEvent.ExtendResult()
            result.request_id = rid
            if prefix + count == lengths[rid]:
                result.tokens = [tokens[rid]]
                completed[rid] = result
            else:
                result.tokens = []
                event.add_event(result)
        scheduler.advance(event)
    event = ExecutionEvent()
    for result in completed.values():
        event.add_event(result)
    scheduler.advance(event)
    decode = scheduler.next_execution_plan()
    if len(decode.forward) != 1:
        raise RuntimeError("Cache capacity could not reserve a simulated decode")
    batch = decode.forward[0]
    if batch.num_extends() or len(batch.request_ids) != len(prompts):
        raise RuntimeError("Simulated decode did not retain the complete batch")
    return SimulatedDecodeFixture(
        scheduler=scheduler,
        allocation_rounds=rounds,
        decode_plan=decode,
        context_lengths=tuple(lengths[rid] for rid in batch.request_ids),
        next_token_ids=tuple(tokens[rid] for rid in batch.request_ids),
        prompt_tokens=tuple(prompt_by_id[rid] for rid in batch.request_ids),
    )


def seed_simulated_index_history(
    pool,
    fixture: SimulatedDecodeFixture,
    *,
    index_format: str | None,
    seed: int,
    rows_per_chunk: int,
) -> int:
    """Fill allocated index pages through their ordinary format-aware writers.

    The arena must already be cleared. Null and unallocated pages remain zero.
    Only a bounded chunk of BF16 keys is materialized at a time, with no model
    prefill. Random index keys avoid an all-tied sparse score distribution.
    ``index_format`` is the V4.1 backend's configured format, or None for DSA,
    whose pool owns its format. Dense/sliding MHA needs no index keys and
    requires None. Returns the number of index rows initialized.
    """
    import torch
    from tokenspeed_kernel.ops.attention.dsv41 import cache_scatter

    from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v41 import (
        DeepseekV41CachePool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool
    from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool

    if rows_per_chunk <= 0:
        raise ValueError("Synthetic cache chunks must contain at least one row")
    if pool.arena.offload is not None or not pool.arena.storage_plan.monolithic:
        raise NotImplementedError("Synthetic fixtures require device-resident history")
    if any(spec.shard_count != 1 for spec in pool.arena.cache_group_specs):
        raise NotImplementedError("Synthetic index setup requires replicated history")
    if isinstance(pool, MHATokenToKVPool):
        if any(plane == "index_k" for _, plane in pool.layer_fields):
            raise NotImplementedError("Synthetic MHA sparse index setup is unsupported")
        if index_format is not None:
            raise ValueError("MHA history has no sparse index encoding")
        # Dense and sliding attention consume the arena's zeroed K/V directly;
        # unlike sparse attention, they have no selection keys to randomize.
        return 0
    if isinstance(pool, DeepseekV41CachePool):
        if index_format not in ("index", "index_v4"):
            raise ValueError("V4.1 requires its configured index encoding")
        head_dim = 128
    elif isinstance(pool, DSATokenToKVPool):
        if index_format is not None:
            raise ValueError("DSA index encoding is owned by its cache pool")
        head_dim = pool.index_head_dim
    else:
        raise NotImplementedError(
            f"Synthetic index setup does not support {type(pool)}"
        )

    tables = fixture.decode_plan.forward[0].block_tables
    initialized = 0
    for (layer, plane), field_id in sorted(pool.layer_fields.items()):
        if plane != "index_k":
            continue
        field = pool.arena.plan.field(field_id)
        pages = sorted(
            {page for row in tables[field.group_id] for page in row if page > 0}
        )
        page_rows = field.shape[0]
        pages_per_chunk = max(1, rows_per_chunk // page_rows)
        device = pool.arena.field(field_id).device
        generator = torch.Generator(device=device).manual_seed(seed + layer)
        for start in range(0, len(pages), pages_per_chunk):
            page_ids = torch.tensor(
                pages[start : start + pages_per_chunk], device=device, dtype=torch.int64
            )
            slots = (
                page_ids[:, None] * page_rows + torch.arange(page_rows, device=device)
            ).flatten()
            values = torch.empty(
                (slots.numel(), head_dim), dtype=torch.bfloat16, device=device
            )
            values.uniform_(-1, 1, generator=generator)
            if isinstance(pool, DeepseekV41CachePool):
                cache_scatter(values, pool.index_k(layer), slots, index_format)
            else:
                pool.set_index_k_buffer(layer, slots, values, write_mask=None)
            initialized += slots.numel()
    return initialized
