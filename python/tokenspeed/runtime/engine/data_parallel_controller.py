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

"""A controller that dispatches requests to multiple data parallel workers."""

import copy
import multiprocessing as mp
import os
import signal
import threading
from collections import deque
from enum import Enum, auto

import psutil
import setproctitle
import zmq

from tokenspeed.runtime.engine.event_loop import run_event_loop
from tokenspeed.runtime.engine.io_struct import (
    BlockReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    WatchLoadUpdateReq,
)
from tokenspeed.runtime.engine.request import Req
from tokenspeed.runtime.utils import (
    configure_logger,
    get_colorful_logger,
    get_zmq_socket,
)
from tokenspeed.runtime.utils.dispatch import TypeBasedDispatcher
from tokenspeed.runtime.utils.exceptions import get_exception_traceback
from tokenspeed.runtime.utils.process import register_usr_signal
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

logger = get_colorful_logger(__name__)


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()
    MINIMUM_CACHE_USAGE = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DPBudget:
    def __init__(self, method: LoadBalanceMethod):
        # Use different metrics for load balancing:
        # - SHORTEST_QUEUE: by num_reqs (running + waiting)
        # - MINIMUM_CACHE_USAGE: by num_pages (page usage)
        self.method = method
        self.budget_queue = deque()

    def update_budget(self, load_update: WatchLoadUpdateReq):
        """Update the budget queue.

        For SHORTEST_QUEUE, use num_reqs instead of num_waiting_reqs to balance decode running batch.
        For MINIMUM_CACHE_USAGE, use num_pages as cache usage metric.
        """
        # method update_budget and method dispatch happen in the same thread, so clearing budget_queue is safe
        self.budget_queue.clear()

        loads = load_update.loads
        if not loads:
            return

        if self.method == LoadBalanceMethod.MINIMUM_CACHE_USAGE:
            metrics = [load.num_pages for load in loads]
        else:
            metrics = [load.num_reqs for load in loads]

        max_metric = max(metrics)
        if all(x == max_metric for x in metrics):
            return

        while any(x != metrics[0] for x in metrics):
            min_load = min(metrics)
            min_indices = [i for i, x in enumerate(metrics) if x == min_load]
            second_min_load = min(x for x in metrics if x > min_load)
            self.budget_queue.extend(
                [loads[i].dp_rank for i in min_indices] * (second_min_load - min_load)
            )
            for idx in min_indices:
                metrics[idx] = second_min_load

    def dispatch(self):
        if self.budget_queue:
            return self.budget_queue.popleft()
        return None


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(self, server_args: ServerArgs, port_args: PortArgs) -> None:
        # Parse args
        self.max_total_num_tokens = None
        self.max_req_input_len = None
        self.max_num_seqs = None
        self.chunked_prefill_size = None
        self.max_model_len = None
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.mapping.attn.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
        # dp_worker for fixed data dispatch can be set by SINGLE_WORKER_ID environment variable
        robin_scheduler = (
            self.round_robin_scheduler
            if os.environ.get("SINGLE_WORKER_ID", "-1") == "-1"
            else self.single_robin_scheduler
        )
        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.budget_scheduler,
            LoadBalanceMethod.MINIMUM_CACHE_USAGE: self.budget_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Load balance budget
        self.dp_budget = DPBudget(self.load_balance_method)

        # Launch data parallel workers
        self.scheduler_procs = []
        self.workers = [None] * server_args.mapping.attn.dp_size

        self.launch_dp_schedulers(server_args, port_args)

        # Workers are already created in launch_dp_schedulers before starting scheduler threads

        if server_args.mapping.has_attn_dp:
            self.control_message_step = server_args.mapping.attn.tp_size
        else:
            self.control_message_step = 1

        self.init_dispatcher()

    def send_to_all_workers(self, obj):
        for worker in self.workers:
            worker.send_pyobj(obj)

    def send_control_message(self, obj):
        # Send control messages to first worker of tp group
        for worker in self.workers[:: self.control_message_step]:
            worker.send_pyobj(obj)

    def handle_load_update_req(self, obj):
        self.dp_budget.update_budget(obj)

    def init_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching),
                (TokenizedEmbeddingReqInput, self.dispatching),
                (BlockReqInput, self.send_to_all_workers),
                (WatchLoadUpdateReq, self.handle_load_update_req),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        threads = []
        dp_port_args = []

        # Parse dist_init_addr from port_args to create per-dp-rank ports
        # Extract base info from the passed port_args
        base_scheduler_port = int(port_args.scheduler_input_ipc_name.split(":")[-1])
        dist_init_host = port_args.scheduler_input_ipc_name.split("//")[1].split(":")[0]

        # port_args.scheduler_input_ipc_name (base_scheduler_port) is used by:
        # TokenizerManager -> DataParallelController
        #
        # For DataParallelController -> Scheduler[dp_rank], we need different ports.
        # Following the same logic as PortArgs.init_new with dp_rank parameter:
        # scheduler_input_port = port_base + 4 + dp_rank
        # Since base_scheduler_port = port_base + 4, we have:
        # scheduler_input_port = base_scheduler_port + dp_rank
        #
        # But we need to avoid conflict with TokenizerManager's port (base_scheduler_port).
        # So we start from base_scheduler_port + 1 for dp_rank=0.

        for dp_rank in range(server_args.mapping.attn.dp_size):
            # Create port_args for each dp_rank by adjusting scheduler_input_port
            # This avoids calling PortArgs.init_new which might use default port
            # Use base_scheduler_port + 1 + dp_rank to avoid conflict with TokenizerManager
            scheduler_input_port = base_scheduler_port + 1 + dp_rank
            tmp_port_args = PortArgs(
                tokenizer_ipc_name=port_args.tokenizer_ipc_name,
                scheduler_input_ipc_name=f"tcp://{dist_init_host}:{scheduler_input_port}",
                nccl_port=port_args.nccl_port,
                rpc_ipc_name=port_args.rpc_ipc_name,
                metrics_ipc_name=port_args.metrics_ipc_name,
                tokenizer_worker_ipc_name=port_args.tokenizer_worker_ipc_name,
            )
            dp_port_args.append(tmp_port_args)

            # Bind to scheduler_input_ipc_name BEFORE starting scheduler threads
            # This ensures the port is available when scheduler tries to connect
            if server_args.node_rank == 0:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,  # bind
                )

        if not server_args.mapping.attn.has_dp:
            dp_rank_range = range(0, 1)
        else:
            dp_ranks_per_node = (
                server_args.mapping.attn.dp_size // server_args.mapping.nnodes
            )
            dp_rank_range = range(
                dp_ranks_per_node * server_args.node_rank,
                dp_ranks_per_node * (server_args.node_rank + 1),
            )
        for dp_rank in dp_rank_range:
            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group,
                args=(server_args, dp_port_args[dp_rank], dp_rank),
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return dp_port_args

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        dp_rank: int,
    ):
        scheduler_pipe_readers = []
        mapping_template = server_args.mapping
        attn_tp_size = mapping_template.attn.tp_size

        if attn_tp_size > mapping_template.nprocs_per_node:
            attn_tp_ranks_per_node = attn_tp_size // mapping_template.nnodes
            attn_tp_rank_range = range(
                attn_tp_ranks_per_node * server_args.node_rank,
                attn_tp_ranks_per_node * (server_args.node_rank + 1),
            )
        else:
            attn_tp_rank_range = range(0, attn_tp_size)
        for attn_tp_rank in attn_tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            global_rank = dp_rank * attn_tp_size + attn_tp_rank

            # Create per-rank server_args with rank-initialized mapping
            rank_server_args = copy.copy(server_args)
            rank_server_args.mapping = copy.deepcopy(mapping_template)
            rank_server_args.mapping.rank = global_rank

            proc = mp.Process(
                target=run_event_loop,
                args=(
                    rank_server_args,
                    port_args,
                    writer,
                ),
            )
            proc.start()
            self.scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)
        # Wait for model to finish loading
        scheduler_info = [reader.recv() for reader in scheduler_pipe_readers]

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]
        self.max_num_seqs = scheduler_info[0]["max_num_seqs"]
        self.chunked_prefill_size = scheduler_info[0]["chunked_prefill_size"]
        self.max_model_len = scheduler_info[0]["max_model_len"]

    def round_robin_scheduler(self, req: Req):
        if self.server_args.disaggregation_mode == "null":
            self.workers[self.round_robin_counter].send_pyobj(req)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )
        else:
            self.workers[req.bootstrap_room % len(self.workers)].send_pyobj(req)

    def single_robin_scheduler(self, req):
        worker_id = int(os.environ.get("SINGLE_WORKER_ID", "-1"))
        if not 0 <= worker_id < self.server_args.mapping.attn.dp_size - 1:
            raise ValueError(f"Invalid SINGLE_WORKER_ID:{worker_id}")
        self.workers[worker_id].send_pyobj(req)

    def budget_scheduler(self, req):
        target_worker = self.dp_budget.dispatch()
        if target_worker is None:
            self.round_robin_scheduler(req)
        else:
            self.workers[target_worker].send_pyobj(req)

    def event_loop(self):
        while True:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    setproctitle.setproctitle("tokenspeed::data_parallel_controller")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()
    register_usr_signal()

    try:
        controller = DataParallelController(server_args, port_args)
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
                "max_num_seqs": controller.max_num_seqs,
                "chunked_prefill_size": controller.chunked_prefill_size,
                "max_model_len": controller.max_model_len,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                "Scheduler or DataParallelController %s terminated with %s",
                proc.pid,
                proc.exitcode,
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error("DataParallelController hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGUSR1)
