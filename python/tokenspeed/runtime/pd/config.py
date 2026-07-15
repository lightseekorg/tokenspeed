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

"""Typed runtime configuration shared by PD and EPD transports."""

from __future__ import annotations

import dataclasses
import math
import os


@dataclasses.dataclass(frozen=True)
class EpdRuntimeConfig:
    """Memory and sharding configuration for encode-to-prefill transfer."""

    encode_ring_slots: int = 64
    encode_ring_slot_mb: int = 256
    encode_embedding_cache_mb: int = 4096
    encode_embedding_cache_dram_mb: int = 0
    recv_pool_slots: int = 16
    recv_pool_slot_mb: int = 256
    embedding_shard: bool = True

    def __post_init__(self) -> None:
        positive = {
            "epd_encode_ring_slots": self.encode_ring_slots,
            "epd_encode_ring_slot_mb": self.encode_ring_slot_mb,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        nonnegative = {
            "epd_encode_embedding_cache_mb": self.encode_embedding_cache_mb,
            "epd_encode_embedding_cache_dram_mb": self.encode_embedding_cache_dram_mb,
            "epd_recv_pool_slots": self.recv_pool_slots,
            "epd_recv_pool_slot_mb": self.recv_pool_slot_mb,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")


@dataclasses.dataclass(frozen=True)
class DisaggregationRuntimeConfig:
    """Explicit process-local configuration for PD/EPD transport workers."""

    queue_size: int = 4
    thread_pool_size: int | None = None
    bootstrap_timeout_s: int = 120
    waiting_timeout_s: int = 300
    failed_session_ttl_s: int = 30
    heartbeat_interval_s: float = 5.0
    heartbeat_max_failures: int = 2
    layerwise_debug: bool = False
    prefill_metadata_wait_log_interval_s: float = 5.0
    epd: EpdRuntimeConfig = dataclasses.field(default_factory=EpdRuntimeConfig)

    def __post_init__(self) -> None:
        if self.queue_size <= 0:
            raise ValueError(
                f"disaggregation_queue_size must be > 0, got {self.queue_size}"
            )
        if (
            self.thread_pool_size is not None
            and self.thread_pool_size < self.queue_size
        ):
            raise ValueError(
                "disaggregation_thread_pool_size must be greater than or equal "
                f"to disaggregation_queue_size ({self.queue_size}), got "
                f"{self.thread_pool_size}"
            )
        if self.bootstrap_timeout_s <= 0:
            raise ValueError(
                "disaggregation_bootstrap_timeout must be > 0, got "
                f"{self.bootstrap_timeout_s}"
            )
        if self.waiting_timeout_s <= 0:
            raise ValueError(
                "disaggregation_waiting_timeout must be > 0, got "
                f"{self.waiting_timeout_s}"
            )
        if self.failed_session_ttl_s < 0:
            raise ValueError(
                "disaggregation_failed_session_ttl must be >= 0, got "
                f"{self.failed_session_ttl_s}"
            )
        finite_positive = {
            "disaggregation_heartbeat_interval": self.heartbeat_interval_s,
            "pd_prefill_metadata_wait_log_interval": (
                self.prefill_metadata_wait_log_interval_s
            ),
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if self.heartbeat_interval_s < 2.0:
            raise ValueError(
                "disaggregation_heartbeat_interval must be >= 2 seconds, got "
                f"{self.heartbeat_interval_s}"
            )
        if self.heartbeat_max_failures < 1:
            raise ValueError(
                "disaggregation_heartbeat_max_failures must be >= 1, got "
                f"{self.heartbeat_max_failures}"
            )

    def resolved_thread_pool_size(self, cpu_count: int | None = None) -> int:
        """Return the configured pool size or the CPU-scaled default."""

        if self.thread_pool_size is not None:
            return self.thread_pool_size
        available_cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 8)
        automatic = min(max(4, int(0.75 * available_cpus) // 8), 12)
        return max(automatic, self.queue_size)
