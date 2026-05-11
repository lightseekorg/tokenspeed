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

import dataclasses
import threading
import time
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from tokenspeed.runtime.pd.base import KVArgs, KVPoll
from tokenspeed.runtime.pd.mooncake.entities import (
    TransferInfo,
    TransferKVChunk,
)
from tokenspeed.runtime.pd.mooncake.prefill import (
    MooncakeKVManagerPrefill as MooncakeKVManager,
)
from tokenspeed.runtime.pd.utils import (
    DisaggregationMode,
    FastQueue,
    StepCounter,
    group_concurrent_contiguous,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.server_args import ServerArgs

logger = get_colorful_logger(__name__)


@dataclasses.dataclass
class WriteRequest:
    trans_info: TransferInfo
    dst_ranks_info: Tuple[str, int, int]
    prefill_kv_blocks: npt.NDArray[np.int64]
    dst_kv_blocks: npt.NDArray[np.int64]
    submit_bids: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class LayerWiseTask:
    kv_chunk: TransferKVChunk
    begin_cache_step: int
    aux_step: int
    next_layer_id: int = 0
    write_requests: List[WriteRequest] = dataclasses.field(default_factory=list)
    polls: List[bool] = dataclasses.field(default_factory=list)


class MooncakeAsyncKVManager(MooncakeKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
        draft_is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(
            args, disaggregation_mode, server_args, is_mla_backend, draft_is_mla_backend
        )
        self.target_layer_num = self.kv_args.target_layer_num
        self.draft_layer_num = self.kv_args.draft_layer_num
        self.layer_num = self.target_layer_num + self.draft_layer_num
        self.is_mla_backend = is_mla_backend
        self.draft_is_mla_backend = draft_is_mla_backend
        self.kv_cache_quant_method = server_args.kv_cache_quant_method
        self.submit_interval = server_args.disaggregation_layerwise_interval
        assert self.submit_interval > 0, "submit_interval must be positive"
        self.current_transfer_batch: List[Tuple[TransferKVChunk, int]] = []
        self.offsets = args.offsets

    def _init_offsets(self):
        if self.is_mla_backend:
            if self.kv_cache_quant_method == "per_token_head":
                target_offsets = [
                    [i * self.target_layer_num + layer_id for i in range(3)]
                    for layer_id in range(self.target_layer_num)
                ]
                draft_offsets = [
                    [3 * self.target_layer_num + layer_id]
                    for layer_id in range(self.draft_layer_num)
                ]
            else:
                target_offsets = [
                    [layer_id] for layer_id in range(self.target_layer_num)
                ]
                draft_offsets = [
                    [self.target_layer_num + layer_id]
                    for layer_id in range(self.draft_layer_num)
                ]
        else:
            target_offsets = [
                [i * self.layer_num + layer_id for i in range(2)]
                for layer_id in range(self.target_layer_num)
            ]
            draft_offsets = [
                [
                    i * self.layer_num + self.target_layer_num + layer_id
                    for i in range(2)
                ]
                for layer_id in range(self.draft_layer_num)
            ]
        self.offsets = target_offsets + draft_offsets

    def start_transfer_thread(
        self, transfer_thread_pool_size: int, transfer_queue_size: int
    ):
        self.transfer_queues: List[FastQueue] = [FastQueue()]
        for queue in self.transfer_queues:
            threading.Thread(
                target=self.async_transfer_worker, args=(queue,), daemon=True
            ).start()

    def register_step_counter(self, step_counter: StepCounter):
        self.step_counter = step_counter

    @contextmanager
    def add_batch(self, is_idle: bool):
        yield  # add transfer request

        if not is_idle:
            begin_cache_step, begin_aux_step = self.step_counter.current_step()
            for kv_chunk, shard_idx in self.current_transfer_batch:
                self.transfer_queues[shard_idx].put(
                    LayerWiseTask(
                        kv_chunk=kv_chunk,
                        begin_cache_step=begin_cache_step,
                        aux_step=begin_aux_step if kv_chunk.is_last else None,
                    )
                )

            # advance to step of next batch no matter if batch is empty
            self.step_counter.advance_step(
                delta_cache_step=self.layer_num, delta_aux_step=1
            )

        self.current_transfer_batch.clear()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
        mla_l1_5_args: Optional[Tuple[Any, Any]] = None,
    ):
        logger.debug("async manager add_transfer_request")
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        #  sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.split(":")[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        kv_chunk = TransferKVChunk(
            room=bootstrap_room,
            prefill_kv_indices=kv_indices,
            index_slice=index_slice,
            is_last=is_last,
            prefill_aux_index=aux_index,
            mla_l1_5_args=mla_l1_5_args,
        )
        self.current_transfer_batch.append((kv_chunk, shard_idx))

    def submit_aux(
        self,
        mooncake_session_id: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
    ):
        # Submit transfer for all aux buffers (output_ids, logprobs, cached_tokens, etc.)
        batch_ids = []
        for aux_data_ptr, aux_item_len, dst_aux_ptr in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_item_lens, dst_aux_ptrs
        ):
            prefill_aux_addr = aux_data_ptr + prefill_aux_index * aux_item_len
            decode_aux_addr = dst_aux_ptr + dst_aux_index * aux_item_len
            bid = self.engine.transfer_submit_write(
                mooncake_session_id, prefill_aux_addr, decode_aux_addr, aux_item_len
            )
            batch_ids.append(bid)
        # The public interface returns a single batch id, so expose the last
        # submitted id even though this helper may issue multiple writes.
        return batch_ids[-1] if batch_ids else -1

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )

    def submit_layer_cache(
        self,
        mooncake_session_id: str,
        begin_layer_id: int,  # [begin_layer_id, end_layer_id)
        end_layer_id: int,
        prefill_kv_blocks: npt.NDArray[np.int64],
        dst_kv_blocks: npt.NDArray[np.int64],
    ) -> int:
        dst_kv_ptrs = self.decode_kv_args_table[mooncake_session_id].dst_kv_ptrs
        transfer_blocks = []

        def submit_one_cache(ptr_offset):
            src_ptr = self.kv_args.kv_data_ptrs[ptr_offset]
            dst_ptr = dst_kv_ptrs[ptr_offset]
            item_len = self.kv_args.kv_item_lens[ptr_offset]
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))

        for layer_id in range(begin_layer_id, end_layer_id):
            for offset in self.offsets[layer_id]:
                submit_one_cache(offset)
        return self._transfer_data(mooncake_session_id, transfer_blocks)

    def async_transfer_worker(self, transfer_queue: FastQueue):
        def disard_finished_bid_inplace(submit_bids: List[int]):
            finished_cnt = 0
            failed = False
            for bid in submit_bids:
                status = self.engine.transfer_check_status(bid)
                if status == 1:
                    finished_cnt += 1
                elif status == -1:
                    failed = True
                    if self.kv_transfer_metrics:
                        self.kv_transfer_metrics.record_kv_transfer_timeout()
                    logger.error("Transfer timeout detected!")
                    break
                elif status == -2:
                    failed = True
                    if self.kv_transfer_metrics:
                        self.kv_transfer_metrics.record_kv_transfer_failure()
                    logger.error("Transfer failed detected")
                    break
                else:
                    failed = status != 0
                    break
            submit_bids[:] = submit_bids[finished_cnt:]
            return not failed

        def discard_tasks(
            tasks: List[LayerWiseTask], dropped: List[LayerWiseTask]
        ) -> List[LayerWiseTask]:
            dropped_rooms = set(id(task) for task in dropped)
            return [task for task in tasks if id(task) not in dropped_rooms]

        def query_ready_step(task: LayerWiseTask) -> Tuple[int, int]:
            if task.next_layer_id < self.layer_num:
                ready_cache_step = self.step_counter.query_ready_cache_step()
                if not StepCounter.is_step_ready(
                    ready_cache_step, task.begin_cache_step + task.next_layer_id
                ):
                    time.sleep(1e-3)
            elif task.kv_chunk.is_last:
                ready_aux_step = self.step_counter.query_ready_aux_step()
                if not StepCounter.is_step_ready(ready_aux_step, task.aux_step):
                    time.sleep(1e-3)

            ready_cache_step = self.step_counter.query_ready_cache_step()
            ready_aux_step = self.step_counter.query_ready_aux_step()
            return ready_cache_step, ready_aux_step

        def get_new_tasks(blocking: bool) -> List[LayerWiseTask]:
            new_tasks: List[LayerWiseTask] = []
            if blocking:
                new_tasks.append(transfer_queue.get())
            while True:
                try:
                    new_tasks.append(transfer_queue.get_nowait())
                except FastQueue.Empty:
                    break

            return new_tasks

        def initialize(tasks: List[LayerWiseTask]) -> List[LayerWiseTask]:
            abort_tasks: List[LayerWiseTask] = []

            for task in tasks:
                kv_chunk = task.kv_chunk
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )

                for req in reqs_to_be_processed:
                    if req.is_dummy:
                        task.polls.append(True)
                        abort_tasks.append(task)
                        break

                    with self.session_lock:
                        if self._is_session_failed(req.mooncake_session_id):
                            logger.info(
                                "Blocked transfer due to failed session (room=%s, session=%s).",
                                kv_chunk.room,
                                req.mooncake_session_id,
                            )
                            self.record_failure(
                                kv_chunk.room,
                                f"Decode instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                            )
                            task.polls.append(False)
                            abort_tasks.append(task)
                            break

                    resolved = self.resolve_transfer_indices(kv_chunk, req)

                    # Group by indices
                    prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
                        resolved.src_indices, resolved.dst_indices
                    )
                    task.write_requests.append(
                        WriteRequest(
                            trans_info=req,
                            dst_ranks_info=(req.endpoint, req.dst_port, req.room),
                            prefill_kv_blocks=prefill_kv_blocks,
                            dst_kv_blocks=dst_kv_blocks,
                        )
                    )

            return abort_tasks

        def submit_transfer(
            tasks: List[LayerWiseTask], ready_cache_step: int, ready_aux_step: int
        ) -> Tuple[List[LayerWiseTask], List[LayerWiseTask]]:
            abort_tasks: List[LayerWiseTask] = []
            complete_tasks: List[LayerWiseTask] = []

            for task in tasks:
                kv_chunk = task.kv_chunk
                # submit layer cache
                if task.next_layer_id < self.layer_num and StepCounter.is_step_ready(
                    ready_cache_step, task.begin_cache_step + task.next_layer_id
                ):
                    for req in task.write_requests:
                        if (
                            (task.next_layer_id + 1) % self.submit_interval == 0
                            or task.next_layer_id == self.layer_num - 1
                        ):
                            ret = self.submit_layer_cache(
                                req.trans_info.mooncake_session_id,
                                (task.next_layer_id // self.submit_interval)
                                * self.submit_interval,
                                task.next_layer_id + 1,
                                req.prefill_kv_blocks,
                                req.dst_kv_blocks,
                            )

                            if ret != 0:
                                if self.kv_transfer_metrics:
                                    self.kv_transfer_metrics.record_kv_transfer_failure()
                                    logger.error("Transfer failed detected!")
                                task.polls.append(False)
                                abort_tasks.append(task)

                                with self.session_lock:
                                    self.session_failures[
                                        req.trans_info.mooncake_session_id
                                    ] += 1
                                    # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                    if (
                                        self.session_failures[
                                            req.trans_info.mooncake_session_id
                                        ]
                                        >= 1
                                    ):
                                        self._mark_session_failed(
                                            req.trans_info.mooncake_session_id,
                                            reason="submit_layer_cache",
                                        )
                                        logger.error(
                                            "Session %s failed.",
                                            req.trans_info.mooncake_session_id,
                                        )
                                        self.record_failure(
                                            kv_chunk.room,
                                            f"Failed to send kv chunk of {kv_chunk.room} to {req.trans_info.endpoint}:{req.trans_info.dst_port}",
                                        )
                                break

                    task.next_layer_id += 1

                # submit aux data
                if (
                    kv_chunk.is_last
                    and task.aux_step is not None
                    and StepCounter.is_step_ready(ready_aux_step, task.aux_step)
                ):
                    task.aux_step = None  # reset to None to mark aux has been submitted
                    if kv_chunk.is_last:
                        for req in task.write_requests:
                            aux_bid = self.submit_aux(
                                req.trans_info.mooncake_session_id,
                                kv_chunk.prefill_aux_index,
                                self.decode_kv_args_table[
                                    req.trans_info.mooncake_session_id
                                ].dst_aux_ptrs,
                                req.trans_info.dst_aux_index,
                            )
                            req.submit_bids.append(aux_bid)

                if task.next_layer_id == self.layer_num and task.aux_step is None:
                    complete_tasks.append(task)

            return complete_tasks, abort_tasks

        def pop_transferred(tasks: List[LayerWiseTask]) -> List[LayerWiseTask]:
            complete_tasks: List[LayerWiseTask] = []
            for task in tasks:
                kv_chunk = task.kv_chunk
                for req in task.write_requests[
                    len(task.polls) :
                ]:  # only check the uncompleted requests
                    success = disard_finished_bid_inplace(req.submit_bids)
                    if success:
                        if len(req.submit_bids) == 0:
                            task.polls.append(True)
                    else:
                        task.polls.append(False)
                        with self.session_lock:
                            self.session_failures[
                                req.trans_info.mooncake_session_id
                            ] += 1
                            # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                            if (
                                self.session_failures[
                                    req.trans_info.mooncake_session_id
                                ]
                                >= 1
                            ):
                                self._mark_session_failed(
                                    req.trans_info.mooncake_session_id,
                                    reason="submit_status",
                                )
                                logger.error(
                                    "Session %s failed.",
                                    req.trans_info.mooncake_session_id,
                                )
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Failed to send kv chunk of {kv_chunk.room} to {req.trans_info.endpoint}:{req.trans_info.dst_port}",
                                )
                                break

                # all finished or any failed
                if len(task.polls) == len(task.write_requests) or (
                    len(task.polls) > 0 and not all(task.polls)
                ):
                    complete_tasks.append(task)

            return complete_tasks

        def finalize(tasks: List[LayerWiseTask]) -> None:
            for task in tasks:
                kv_chunk = task.kv_chunk
                status = KVPoll.Success if all(task.polls) else KVPoll.Failed
                if (
                    status == KVPoll.Failed or kv_chunk.is_last
                ):  # last chunk or any failed
                    self.update_status(kv_chunk.room, status)
                    for packed_req in task.write_requests:
                        endpoint, dst_port, room = packed_req.dst_ranks_info
                        self.sync_status_to_decode_endpoint(
                            endpoint, dst_port, room, status, self.attn_tp_rank
                        )

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

        pending_tasks: List[LayerWiseTask] = []
        inflight_tasks: List[LayerWiseTask] = []
        while True:
            try:
                if (
                    len(
                        new_tasks := get_new_tasks(
                            blocking=(len(pending_tasks) + len(inflight_tasks) == 0)
                        )
                    )
                    > 0
                ):
                    if len(abort_tasks := initialize(new_tasks)) > 0:
                        finalize(abort_tasks)
                        new_tasks = discard_tasks(new_tasks, abort_tasks)

                    pending_tasks.extend(new_tasks)

                if len(pending_tasks) > 0:
                    ready_cache_step, ready_aux_step = query_ready_step(
                        pending_tasks[0]
                    )  # only wait the first task
                    submited_tasks, abort_tasks = submit_transfer(
                        pending_tasks, ready_cache_step, ready_aux_step
                    )
                    finalize(abort_tasks)
                    pending_tasks = discard_tasks(
                        pending_tasks, submited_tasks + abort_tasks
                    )
                    inflight_tasks.extend(submited_tasks)

                if len(complete_tasks := pop_transferred(inflight_tasks)) > 0:
                    finalize(complete_tasks)
                    inflight_tasks = discard_tasks(inflight_tasks, complete_tasks)

            except Exception as e:
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )
