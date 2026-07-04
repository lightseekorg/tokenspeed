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

from __future__ import annotations

"""Shared helpers for disaggregation runtime components."""

import ctypes
import dataclasses
import os
import random
import threading
import warnings
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import requests
import torch
import torch.distributed as dist

from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.network import get_ip

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.request import Req

FAKE_BOOTSTRAP_HOST = "2.2.2.2"

# env var for testing failure, convert to float explicitly
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))
logger = get_colorful_logger(__name__)


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


def poll_and_all_reduce(pollers, gloo_group):
    """Poll transfer state and all-reduce the result across the gloo group."""
    # At a certain probability, mark the poll as failed to simulate failure.
    if FAILURE_PROB > 0:
        from tokenspeed.runtime.pd.base import KVPoll

        polls = [
            int(KVPoll.Failed) if random.random() < FAILURE_PROB else int(poller.poll())
            for poller in pollers
        ]
    else:
        polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


class ReqToMetadataIdxAllocator:
    """A memory pool that maps a request to its first output token location."""

    def __init__(
        self,
        size: int,
    ):
        self.size = size
        self.free_slots = deque(range(size))

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self) -> int | None:
        if not self.free_slots:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int) -> None:
        self.free_slots.append(free_index)


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MOONCAKE_ASYNC = "mooncake_async"
    FAKE = "fake"
    COMMON = "common"


class KVClassType(Enum):
    MANAGER_PREFILL = "manager_prefill"
    MANAGER_DECODE = "manager_decode"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


def get_kv_class(transfer_backend: TransferBackend, class_type: KVClassType):
    if transfer_backend == TransferBackend.MOONCAKE:
        from tokenspeed.runtime.pd.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManagerDecode,
            MooncakeKVManagerPrefill,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.MANAGER_PREFILL: MooncakeKVManagerPrefill,
            KVClassType.MANAGER_DECODE: MooncakeKVManagerDecode,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)

    if transfer_backend == TransferBackend.MOONCAKE_ASYNC:
        from tokenspeed.runtime.pd.mooncake import (
            MooncakeAsyncKVManager,
            MooncakeKVBootstrapServer,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.MANAGER: MooncakeAsyncKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # 1. The page is guaranteed to be full except the last page.
    # 2. page index = kv_index // page_size
    # The return vector is kv_indices[::page_size] // page_size
    if page_size == 1:  # shortcut
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    # ceil(num_kv_indices / page_size)
    return (num_kv_indices + page_size - 1) // page_size


@dataclasses.dataclass
class PDRegistryRequest:
    """A request to register a machine itself to the LB."""

    mode: str
    registry_url: str
    bootstrap_port: int | None = None

    def __post_init__(self):
        if self.mode == "prefill" and self.bootstrap_port is None:
            raise ValueError("Bootstrap port must be set in PREFILL mode.")
        if self.mode == "decode" and self.bootstrap_port is not None:
            raise ValueError("Bootstrap port must not be set in DECODE mode.")
        if self.mode not in {"prefill", "decode"}:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'prefill' or 'decode'."
            )


def register_disaggregation_server(
    mode: str, server_port: int, bootstrap_port: int, pdlb_url: str
):
    pdlb_url = pdlb_url.rstrip("/")
    registered_bootstrap_port = bootstrap_port if mode == "prefill" else None
    registry_request = PDRegistryRequest(
        mode=mode,
        registry_url=f"http://{get_ip()}:{server_port}",
        bootstrap_port=registered_bootstrap_port,
    )
    res = requests.post(
        f"{pdlb_url}/register",
        json=dataclasses.asdict(registry_request),
    )
    if res.status_code != 200:
        warnings.warn(
            f"Failed to register disaggregation server: {res.status_code} {res.text}"
        )
    else:
        logger.info(
            "Registered disaggregation server with %s: status_code=%s text=%s",
            pdlb_url,
            res.status_code,
            res.text,
        )


def is_mla_backend(target_kv_pool) -> bool:
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool

    return isinstance(target_kv_pool, MLATokenToKVPool)


def prepare_abort(req: Req, error_message: str, status_code=None, err_type=None):
    from tokenspeed.runtime.engine.request_types import ABORT_CODE, FINISH_ABORT

    if err_type is None:
        err_type = ABORT_CODE.UnknownError

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(
        error_message, status_code=status_code, err_type=err_type
    )

    if req.return_logprob:
        req.input_token_logprobs_val = []
        req.input_token_logprobs_idx = []
        req.input_top_logprobs_val = []
        req.input_top_logprobs_idx = []
        req.input_token_ids_logprobs_val = []
        req.input_token_ids_logprobs_idx = []


class MetadataBuffers:
    def __init__(self, size: int, max_top_logprobs_num: int = 128, device: str = "cpu"):

        # We transfer the metadata of first output token to decode
        # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
        self.device = device
        self.output_ids = torch.zeros((size, 16), dtype=torch.int32, device=device)
        self.output_token_logprobs_val = torch.zeros(
            (size, 16), dtype=torch.float32, device=device
        )
        self.output_token_logprobs_idx = torch.zeros(
            (size, 16), dtype=torch.int32, device=device
        )
        self.output_top_logprobs_val = torch.zeros(
            (size, max_top_logprobs_num), dtype=torch.float32, device=device
        )
        self.output_top_logprobs_idx = torch.zeros(
            (size, max_top_logprobs_num), dtype=torch.int32, device=device
        )
        self.cached_tokens = torch.zeros((size, 1), dtype=torch.int32, device=device)

    def get_buf_infos(self):
        ptrs = [
            self.output_ids.data_ptr(),
            self.output_token_logprobs_val.data_ptr(),
            self.output_token_logprobs_idx.data_ptr(),
            self.output_top_logprobs_val.data_ptr(),
            self.output_top_logprobs_idx.data_ptr(),
            self.cached_tokens.data_ptr(),
        ]
        data_lens = [
            self.output_ids.nbytes,
            self.output_token_logprobs_val.nbytes,
            self.output_token_logprobs_idx.nbytes,
            self.output_top_logprobs_val.nbytes,
            self.output_top_logprobs_idx.nbytes,
            self.cached_tokens.nbytes,
        ]
        item_lens = [
            self.output_ids[0].nbytes,
            self.output_token_logprobs_val[0].nbytes,
            self.output_token_logprobs_idx[0].nbytes,
            self.output_top_logprobs_val[0].nbytes,
            self.output_top_logprobs_idx[0].nbytes,
            self.cached_tokens[0].nbytes,
        ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        return (
            self.output_ids[idx],
            self.output_token_logprobs_val[idx],
            self.output_token_logprobs_idx[idx],
            self.output_top_logprobs_val[idx],
            self.output_top_logprobs_idx[idx],
            self.cached_tokens[idx],
        )

    def set_buf_by_batch(
        self,
        output_ids: torch.Tensor,
        output_buffer_indices: list[int],
        logits_output: LogitsProcessorOutput,
        output_token_logprobs_indices: list[tuple[int, int]] | None = None,
        output_top_logprobs_indices: list[tuple[int, int]] | None = None,
        cached_tokens: torch.Tensor | None = None,
    ):
        self.output_ids[
            torch.tensor(output_buffer_indices).to(self.device, non_blocking=True), 0
        ] = output_ids
        if cached_tokens is not None:
            self.cached_tokens[
                torch.tensor(output_buffer_indices).to(self.device, non_blocking=True),
                0,
            ] = cached_tokens

        if output_token_logprobs_indices:
            for src_idx, dst_idx in output_token_logprobs_indices:
                self.output_token_logprobs_val[dst_idx][0] = (
                    logits_output.next_token_top_logprobs_val[src_idx]
                )
                self.output_token_logprobs_idx[dst_idx][0] = (
                    logits_output.next_token_top_logprobs_idx[src_idx]
                )

        if output_top_logprobs_indices:
            for src_idx, dst_idx in output_top_logprobs_indices:
                self.output_top_logprobs_val[dst_idx][
                    : len(logits_output.next_token_top_logprobs_val[src_idx][0])
                ] = torch.tensor(
                    logits_output.next_token_top_logprobs_val[src_idx][0],
                    dtype=torch.float32,
                    device="cpu",
                )
                self.output_top_logprobs_idx[dst_idx][
                    : len(logits_output.next_token_top_logprobs_idx[src_idx][0])
                ] = torch.tensor(
                    logits_output.next_token_top_logprobs_idx[src_idx][0],
                    dtype=torch.int32,
                    device="cpu",
                )

    def set_buf(self, req: Req):
        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        if req.return_logprob:
            if req.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_val[0]
                )
            if req.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_idx[0]
                )

            if req.output_top_logprobs_val:  # not none or empty list
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.output_top_logprobs_val[0], dtype=torch.float32, device="cpu"
                )
            if req.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.output_top_logprobs_idx[0], dtype=torch.int32, device="cpu"
                )


class FastQueue:
    class Empty(Exception):
        """Exception raised when the queue is empty."""

        pass

    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            self._buf.append(item)
            # wake up a thread of wait()
            self._cond.notify()

    def get(self):
        with self._cond:
            # if queue is empty  ,block until is notified()
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()

    def get_nowait(self):
        with self._cond:
            if not self._buf:
                raise FastQueue.Empty()
            return self._buf.popleft()


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int64], dst_indices: npt.NDArray[np.int64]
) -> tuple[list[npt.NDArray[np.int64]], list[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


class StepCounter:
    COUNT_NUM_MAX: int = 2**62

    @classmethod
    def is_step_ready(cls, current_step: int, target_step: int) -> bool:
        # because COUNT_NUM_MAX is very large, we can make sure that if diff is > COUNT_NUM_MAX / 2 means the flush is finished
        # and if the current_sent_count == task_stop_count also means the flush is not finished
        # so if current_sent_count != task_stop_count and diff < COUNT_NUM_MAX / 2, the flush is not finished
        return (
            target_step != current_step
            and (target_step + cls.COUNT_NUM_MAX - current_step) % cls.COUNT_NUM_MAX
            > cls.COUNT_NUM_MAX / 2
        )

    def __init__(self, device: str, gpu_id: int):
        # utilities for cache step
        self.d_ready_cache_step = torch.tensor(0, dtype=torch.int64).cuda(gpu_id)
        self.h_ready_cache_step = torch.tensor(0, dtype=torch.int64, pin_memory=True)
        self.cache_step: int = 0

        # utilities for aux step
        self.d_ready_aux_step = torch.tensor(0, dtype=torch.int64).cuda(gpu_id)
        self.h_ready_aux_step = torch.tensor(0, dtype=torch.int64, pin_memory=True)
        self.aux_step: int = 0

    def current_step(self) -> tuple[int, int]:
        return self.cache_step, self.aux_step

    def advance_step(self, delta_cache_step: int, delta_aux_step: int):
        self.cache_step = (self.cache_step + delta_cache_step) % self.COUNT_NUM_MAX
        self.aux_step = (self.aux_step + delta_aux_step) % self.COUNT_NUM_MAX

    def record_cache(self):
        self.d_ready_cache_step = (self.d_ready_cache_step + 1) % self.COUNT_NUM_MAX
        self.h_ready_cache_step.copy_(self.d_ready_cache_step, non_blocking=True)

    def record_aux(self):
        self.d_ready_aux_step = (self.d_ready_aux_step + 1) % self.COUNT_NUM_MAX
        self.h_ready_aux_step.copy_(self.d_ready_aux_step, non_blocking=True)

    def query_ready_cache_step(self) -> int:
        return ctypes.c_int64.from_address(self.h_ready_cache_step.data_ptr()).value

    def query_ready_aux_step(self) -> int:
        return ctypes.c_int64.from_address(self.h_ready_aux_step.data_ptr()).value


@dataclasses.dataclass
class PageTransferMetadata:
    indices_are_local: bool
    page_transfer_mask: npt.NDArray[np.bool_]
    page_local_indices: npt.NDArray[np.int64] | None = None
