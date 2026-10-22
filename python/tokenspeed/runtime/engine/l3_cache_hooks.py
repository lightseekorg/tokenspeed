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

"""Agree on bounded Host-I/O facts; C++ owns readiness and memory lifetime."""

from __future__ import annotations

import time
from typing import NamedTuple

import torch
import torch.distributed as dist
from tokenspeed_scheduler import Cache, ForwardEvent, StorageConfig, StoragePressure

from tokenspeed.runtime.cache.l3.metrics import L3Metrics


class _ReportHeader(NamedTuple):
    """Fixed CPU control fields exchanged before optional completion payloads."""

    now_us: int
    allow_forward: int
    allow_storage: int
    completion_count: int
    has_error: int
    queued_operations: int
    queued_bytes: int
    oldest_operation_us: int


_HEADER_WIDTH = len(_ReportHeader._fields)


class L3CacheHooks:
    def __init__(
        self,
        scheduler,
        l3,
        *,
        config: StorageConfig | None,
        attn_tp_size: int,
        attn_tp_cpu_group,
        pp_size: int,
        pp_cpu_group,
        metrics: L3Metrics | None,
    ):
        if l3 is not None and (config is None or pp_size != 1):
            raise ValueError(
                "Async L3 requires explicit limits and currently supports PP=1"
            )
        if l3 is not None and (
            attn_tp_size < 1 or (attn_tp_size > 1 and attn_tp_cpu_group is None)
        ):
            raise ValueError("Async L3 requires the explicit attention-TP CPU group")
        self._scheduler = scheduler
        self._l3 = l3
        self._config = config
        self._size = attn_tp_size
        self._group = attn_tp_cpu_group
        # Delivered owner facts live here until all owners complete. They are
        # never retransmitted just because a different owner is still pending.
        self._reports: dict[
            int, dict[int, tuple[tuple[bool, ...], bool, tuple[bool, ...]]]
        ] = {}
        self._cancellations: dict[str, object] = {}
        self._metrics = metrics
        self._last_metrics_time = 0.0
        self._last_metrics_pending = False
        self._header: torch.Tensor | None = None
        self._headers: torch.Tensor | None = None
        if l3 is not None:
            self._header = torch.empty(_HEADER_WIDTH, dtype=torch.int64, device="cpu")
            self._headers = torch.empty(
                _HEADER_WIDTH * attn_tp_size, dtype=torch.int64, device="cpu"
            )

    def _exchange_reports(self, report):
        if self._size == 1:
            return [report]
        assert self._header is not None and self._headers is not None
        now_us, allow_forward, allow_storage, completions, error, pressure = report
        local = _ReportHeader(
            now_us,
            int(allow_forward),
            int(allow_storage),
            len(completions),
            int(error is not None),
            *pressure,
        )
        for index, value in enumerate(local):
            self._header[index] = value
        dist.all_gather_into_tensor(self._headers, self._header, group=self._group)
        headers = [
            _ReportHeader(*row)
            for row in self._headers.view(self._size, _HEADER_WIDTH).tolist()
        ]
        # Every owner still agrees the clock and permissions in idle rounds.
        # Only completion/error facts require a variable-size object exchange.
        if any(header.completion_count or header.has_error for header in headers):
            payloads = [None] * self._size
            dist.all_gather_object(payloads, (completions, error), group=self._group)
        else:
            payloads = [([], None) for _ in headers]
        return [
            (
                header.now_us,
                bool(header.allow_forward),
                bool(header.allow_storage),
                payload[0],
                payload[1],
                (
                    header.queued_operations,
                    header.queued_bytes,
                    header.oldest_operation_us,
                ),
            )
            for header, payload in zip(headers, payloads)
        ]

    def record_metrics(self) -> None:
        """Sample at most once a second, plus transitions into/out of pending.

        Called after head feedback, including frozen rounds. Observations do
        not participate in replica agreement or alter scheduler permissions.
        """
        if self._l3 is None or self._metrics is None:
            return
        now = time.monotonic()
        pending = self._scheduler.storage_pending()
        if now - self._last_metrics_time < 1 and pending == self._last_metrics_pending:
            return
        snapshot = self._l3.snapshot()
        if snapshot is not None:
            self._metrics.record(snapshot, self._scheduler.storage_stats())
        self._last_metrics_time = now
        self._last_metrics_pending = pending

    def cancel_request(self, request_id: str) -> bool:
        """Queue head feedback; return whether no future forward will reap output.

        C++ decides resource eligibility. PD retains its existing transport
        fence; cancellation still detaches L3 without waiting for its timeout.
        """
        if self._l3 is None or self._scheduler.request_token_size(request_id) < 0:
            return False
        can_finish = self._scheduler.can_abort_before_admission(request_id)
        if can_finish:
            event = ForwardEvent.Abort()
            event.request_id = request_id
        else:
            event = Cache.StorageDetach(request_id)
        self._cancellations[request_id] = event
        return can_finish

    def poll_ready_events(self, *, allow_forward: bool, allow_storage: bool) -> list:
        if self._l3 is None:
            return []
        config = self._config
        assert config is not None
        error = None
        pressure = (0, 0, 0)
        completions = []
        try:
            poll = self._l3.poll()
            local = poll.completions
            pressure = (
                poll.queued_operations,
                poll.queued_bytes,
                poll.oldest_operation_us,
            )
            if any(
                type(value) is not int or value < 0 or value > 2**63 - 1
                for value in pressure
            ):
                raise ValueError("Invalid L3 pressure observation")
            if len(local) > config.max_operations:
                raise RuntimeError("L3 completion window exceeded operation credits")
            # Normalize every local fact before joining the collective. A bad
            # record must not strand peers already exchanging their headers.
            for item in local:
                if (
                    type(item.op_id) is not int
                    or not 0 < item.op_id < 2**64
                    or type(item.retired) is not bool
                    or len(item.success) > config.max_keys_per_operation
                    or any(type(flag) is not bool for flag in item.success)
                    or len(item.object_failures) != len(item.success)
                    or any(type(flag) is not bool for flag in item.object_failures)
                    or any(
                        ok and failed
                        for ok, failed in zip(item.success, item.object_failures)
                    )
                ):
                    raise ValueError("Malformed L3 completion")
                completions.append(
                    (
                        item.op_id,
                        tuple(item.success),
                        item.retired,
                        tuple(item.object_failures),
                    )
                )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - every owner must join on polling failure
            # Even a local polling fault must participate in the control
            # exchange, so all owners fail the protocol together.
            completions = []
            pressure = (0, 0, 0)
            error = str(exc)[:512]
        # The complete window is bounded by max_operations * max_keys. V1
        # sends this finite window once, with no prefix-of-pending truncation.
        report = (
            time.monotonic_ns() // 1000,
            allow_forward,
            allow_storage,
            completions,
            error,
            pressure,
        )
        reports = self._exchange_reports(report)
        if any(x[4] is not None for x in reports):
            raise RuntimeError(f"L3 replica polling failed: {[x[4] for x in reports]}")
        for rank, (_, _, _, completions, _, _) in enumerate(reports):
            if len(completions) > config.max_operations:
                raise RuntimeError("L3 peer exceeded report credits")
            for op_id, success, retired, object_failures in completions:
                if len(success) > config.max_keys_per_operation:
                    raise RuntimeError("L3 peer exceeded key credits")
                row = self._reports.setdefault(op_id, {})
                value = (tuple(success), retired, tuple(object_failures))
                if rank in row and row[rank] != value:
                    raise RuntimeError("Conflicting L3 owner report")
                row[rank] = value
        if len(self._reports) > config.max_operations:
            raise RuntimeError("L3 aggregation exceeded operation credits")
        events = [
            Cache.StorageTick(
                reports[0][0],
                all(x[1] for x in reports),
                all(x[2] for x in reports),
                StoragePressure(
                    *(max(row[5][index] for row in reports) for index in range(3))
                ),
            )
        ]
        events.extend(self._cancellations.values())
        self._cancellations.clear()
        for op_id in sorted(self._reports):
            row = self._reports[op_id]
            if len(row) != self._size:
                continue
            results = [row[rank] for rank in range(self._size)]
            count = len(results[0][0])
            if any(len(x[0]) != count for x in results):
                raise RuntimeError("L3 mirrored operation shapes differ")
            success = [int(all(x[0][i] for x in results)) for i in range(count)]
            # A confirmed missing/invalid shard on any owner invalidates the
            # object; a peer that rejected work contributes no such evidence.
            object_failures = [int(any(x[2][i] for x in results)) for i in range(count)]
            events.append(
                Cache.StorageDone(
                    op_id, success, all(x[1] for x in results), object_failures
                )
            )
        for op_id in [
            op for op, owners in self._reports.items() if len(owners) == self._size
        ]:
            del self._reports[op_id]
        return events
