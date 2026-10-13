"""Compact Host cache executor tests."""

from __future__ import annotations

import inspect
import os
import sys
import threading
import time
import unittest
from concurrent.futures import Future
from contextlib import nullcontext
from importlib import import_module, util
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

# Pure Python, no runtime dependencies: the knobs of an L2-only executor.
from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest  # noqa: E402

NO_POOL = RetractionPoolRequest(
    host_gb=0.0, ratio=0.0, max_retracted_requests=0, tail_lcm_blocks_per_request=0
)


class _LoadEvents(SimpleNamespace):
    def set_completion(self, event):
        self.layer_done_events[:] = [event] * len(self.layer_done_events)


def _replicated_contract(layout, *, num_lcm_blocks=None):
    """An arena contract for ``layout`` with every group replicated (shard 1)."""
    if num_lcm_blocks is None:
        num_lcm_blocks = layout.num_lcm_blocks
    specs = tuple(
        SimpleNamespace(group_id=group.group_id, shard_count=1)
        for group in layout.groups
    )
    return SimpleNamespace(
        group_specs=specs,
        virtual_block_counts={
            group.group_id: 1 + num_lcm_blocks * group.cache_blocks_per_lcm_block
            for group in layout.groups
        },
        num_lcm_blocks=num_lcm_blocks,
        token_capacity=num_lcm_blocks,
    )


def _synthetic_arena(layout):
    contract = _replicated_contract(layout)
    return SimpleNamespace(
        cache_group_specs=contract.group_specs, runtime_contract=contract
    )


class _SyntheticPool:
    def __init__(self, layout, arena=None):
        self._layout = layout
        if arena is None:
            arena = _synthetic_arena(layout)
        self.arena = arena

    def cache_transfer_layout(self):
        return self._layout

    def register_layerwise_load_tracker(self, tracker):
        self.load_tracker = tracker


def _load_executor_module_without_triton(*, force_isolated=False):
    """Load executor orchestration when optional Triton is not installed."""

    # Keep real dependencies outside the temporary sys.modules snapshot.
    # Otherwise the first isolated load removes psutil again on exit.
    import_module("psutil")

    if not force_isolated:
        executor_name = "tokenspeed.runtime.cache.l2.executor"
        if executor_name in sys.modules or util.find_spec("tokenspeed_triton"):
            return import_module("tokenspeed.runtime.cache.l2.executor")
    host_transfer = ModuleType("tokenspeed_kernel.ops.kvcache.host_transfer")
    host_transfer.HostTransferWorkspace = Mock
    host_transfer.build_host_transfer_geometry = Mock()
    host_transfer.transfer_cache_blocks = Mock()
    host_transfer.wait_layer_ready = Mock()
    ownership = ModuleType("tokenspeed.runtime.cache.transfer.ownership")
    ownership.BlockOwnerTranslation = Mock
    lanes = ModuleType("tokenspeed.runtime.cache.transfer.lanes")
    lanes.CompletionQueue = Mock
    lanes.HostTransferLane = Mock
    lanes.build_transfer_geometry = Mock()
    lanes.check_host_memory = Mock()
    lanes.load_stream_priority = Mock(return_value=None)
    lanes.new_cache_stream = Mock()
    scheduler = ModuleType("tokenspeed_scheduler")

    class Cache:
        class WriteBackOp:
            pass

        class LoadBackOp:
            pass

        class PrefetchOp:
            pass

        class WriteBackDoneEvent:
            pass

        class LoadBackDoneEvent:
            def __init__(self, op_id):
                self.op_id = op_id

        class PrefetchDoneEvent:
            def __init__(self, op_id, landed_pages):
                self.op_id = op_id
                self.landed_pages = landed_pages

    scheduler.Cache = Cache
    layerwise_load = ModuleType("tokenspeed.runtime.cache.l2.layerwise_load")
    layerwise_load.LayerwiseLoadTracker = Mock
    storage = ModuleType("tokenspeed.runtime.cache.l2.storage")
    storage.HostCacheStorage = Mock
    storage.compute_host_lcm_block_bytes = Mock(return_value=1)
    layout = ModuleType("tokenspeed.runtime.cache.transfer.layout")
    layout.combine_cache_transfer_layouts = lambda target, draft, group_ids=None: (
        target if draft is None else draft
    )
    forward_step = ModuleType("tokenspeed.runtime.execution.forward_step")
    forward_step.get_is_capture_mode = Mock(return_value=False)
    runtime_utils = ModuleType("tokenspeed.runtime.utils")
    runtime_utils.get_colorful_logger = Mock(return_value=Mock())
    runtime_utils.get_device_module = Mock(return_value=Mock())
    fake_modules = {
        "tokenspeed_kernel.ops.kvcache.host_transfer": host_transfer,
        "tokenspeed_scheduler": scheduler,
        "tokenspeed.runtime.cache.l2.layerwise_load": layerwise_load,
        "tokenspeed.runtime.cache.l2.storage": storage,
        "tokenspeed.runtime.cache.transfer.lanes": lanes,
        "tokenspeed.runtime.cache.transfer.layout": layout,
        "tokenspeed.runtime.cache.transfer.ownership": ownership,
        "tokenspeed.runtime.execution.forward_step": forward_step,
        "tokenspeed.runtime.utils": runtime_utils,
    }
    executor_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "python",
            "tokenspeed",
            "runtime",
            "cache",
            "l2",
            "executor.py",
        )
    )
    spec = util.spec_from_file_location("_isolated_l2_executor", executor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load isolated executor from {executor_path}")
    executor_module = util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules, clear=False):
        spec.loader.exec_module(executor_module)
    return executor_module


class CacheEventPayloadTest(unittest.TestCase):
    def setUp(self):
        try:
            from tokenspeed_scheduler import Cache

            from tokenspeed.runtime.engine.scheduler_utils import (
                cache_event_from_payload,
                cache_event_to_payload,
                pop_common_cache_event_payloads,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")
        self.Cache = Cache
        self.from_payload = cache_event_from_payload
        self.to_payload = cache_event_to_payload
        self.pop_common = pop_common_cache_event_payloads

    def test_cache_completion_payload_round_trips_load_back_and_prefetch(self):
        write_back = self.Cache.WriteBackDoneEvent()
        write_back.op_id = 7
        write_payload = self.to_payload(write_back)
        self.assertEqual(write_payload, {"kind": "WriteBackDoneEvent", "op_id": 7})

        # A load-back lands or does not happen: the ACK carries no outcome.
        load_back = self.Cache.LoadBackDoneEvent(8)
        load_payload = self.to_payload(load_back)
        self.assertEqual(load_payload, {"kind": "LoadBackDoneEvent", "op_id": 8})
        restored = self.from_payload(load_payload)
        self.assertIsInstance(restored, self.Cache.LoadBackDoneEvent)
        self.assertEqual(int(restored.op_id), 8)

        # An L3 prefetch's ACK carries the pages it landed, replica-converged.
        prefetch = self.Cache.PrefetchDoneEvent(9, 3)
        prefetch_payload = self.to_payload(prefetch)
        self.assertEqual(
            prefetch_payload,
            {"kind": "PrefetchDoneEvent", "op_id": 9, "landed_pages": 3},
        )
        restored = self.from_payload(prefetch_payload)
        self.assertIsInstance(restored, self.Cache.PrefetchDoneEvent)
        self.assertEqual((int(restored.op_id), int(restored.landed_pages)), (9, 3))
        with self.assertRaises(KeyError):
            self.from_payload({"kind": "PrefetchDoneEvent", "op_id": 9})
        self.assertEqual(
            self.pop_common([[load_payload], [dict(load_payload)]]), [load_payload]
        )


def _lanes_module():
    """The shared Host-transfer lane module the L2 write path delegates to."""
    return import_module("tokenspeed.runtime.cache.transfer.lanes")


def _identity_owners(num_groups=1, bound=64):
    """A replicated (shard 1) owner translation: local ids are scheduler ids."""
    from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation

    return BlockOwnerTranslation(
        shard_counts=[1] * num_groups,
        device_virtual_counts=[bound] * num_groups,
        host_virtual_counts=[bound] * num_groups,
        rank=0,
    )


class GroupAwareWireTest(unittest.TestCase):
    def _executor_module(self):
        try:
            return _load_executor_module_without_triton()
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

    def _make_load_executor(
        self,
        *,
        consumers,
        layer_slices,
        backend="auto",
        device_rows=None,
        load_stream=None,
    ):
        executor_module = self._executor_module()
        executor = executor_module.HostCacheExecutor.__new__(
            executor_module.HostCacheExecutor
        )
        executor._ack_lock = threading.Lock()
        executor.attn_tp_rank = 0
        executor._ready_load_acks = []
        executor._completions = _lanes_module().CompletionQueue()
        executor._load_poisoned = False
        executor.l3_store = None
        executor._l3_prefetch_lane = None
        executor._prefetch_jobs = {}
        executor._prefetch_acks = []
        executor._l3_prefetch_timeout_base_s = 10.0
        executor._l3_prefetch_timeout_per_page_s = 0.0
        executor._l3_prefetch_batch_pages = 2
        executor.load_stream = object() if load_stream is None else load_stream
        executor.transfer_backend = backend
        device = SimpleNamespace(type="cuda")
        executor.layout = SimpleNamespace(
            buffers=(SimpleNamespace(device=device),),
            consumers=consumers,
        )
        executor.host_storage = SimpleNamespace(host_buffer="host")
        geometry = SimpleNamespace(
            layer_slices=layer_slices,
            device_rows=device_rows,
            num_field_rows=sum(count for _, count in layer_slices),
        )
        executor._transfer_geometry = geometry
        workspace = MagicMock()
        workspace.load_block_transfers.return_value = (1, (0, 1))
        workspace.prepare_backend.return_value = SimpleNamespace(
            uses_device_tables=device_rows is not None,
            layer_ready=device_rows is not None,
        )
        executor._load_workspaces = (workspace,)
        return executor_module, executor, device, geometry, workspace

    def test_hybrid_state_access_waits_for_layer_load(self):
        try:
            from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
                HybridKDATokenToKVPool,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
                HybridMHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        for pool_type in (HybridMHATokenToKVPool, HybridKDATokenToKVPool):
            with self.subTest(pool_type=pool_type.__name__):
                tracker = Mock()
                pool = pool_type.__new__(pool_type)
                pool.layerwise_load_tracker = tracker
                pool._state_buffers_by_layer = {3: ("conv", "recurrent")}
                if pool_type is HybridMHATokenToKVPool:
                    pool._state_layer_ids = (3,)

                self.assertEqual(pool.get_component(3, "conv_state"), "conv")
                tracker.wait_for_layer.assert_called_once_with(3)

    def test_pool_transfer_layout_matches_scheduler_group_order(self):
        try:
            from cache_pool_test_utils import MinimalCacheView
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        pool = MinimalCacheView.__new__(MinimalCacheView)
        pool.layer_num = 2
        pool._field_layer_offset = 0
        # The arena owns the buffer, the field views and the published specs;
        # the pool only answers for them.
        pool.arena = SimpleNamespace(
            buffer=object(),
            cache_group_specs=(
                SimpleNamespace(group_id="state"),
                SimpleNamespace(group_id="full"),
            ),
        )
        pool.arena.plan = SimpleNamespace(
            num_lcm_blocks=4,
            planes=(
                SimpleNamespace(
                    plane_id="shared",
                    bytes_per_lcm_block=4096,
                    arena_offset_bytes=0,
                ),
            ),
            groups=(
                SimpleNamespace(
                    group_id="full",
                    cache_blocks_per_lcm_block=32,
                ),
                SimpleNamespace(
                    group_id="state",
                    cache_blocks_per_lcm_block=1,
                ),
            ),
            fields=(
                SimpleNamespace(
                    group_id="full",
                    field_id="layer.1.k",
                    plane_id="shared",
                    field_offset_bytes=0,
                    page_stride_bytes=128,
                    payload_bytes=128,
                ),
                SimpleNamespace(
                    group_id="state",
                    field_id="layer.0.state",
                    plane_id="shared",
                    field_offset_bytes=0,
                    page_stride_bytes=4096,
                    payload_bytes=4096,
                ),
            ),
        )

        layout = pool.cache_transfer_layout()

        self.assertEqual(
            tuple(group.group_id for group in layout.groups),
            ("state", "full"),
        )

    def test_submit_load_backs_clears_layerwise_waits_without_load(self):
        HostCacheExecutor = self._executor_module().HostCacheExecutor

        tracker = Mock()
        executor = HostCacheExecutor.__new__(HostCacheExecutor)
        executor._ack_lock = threading.Lock()
        executor._load_trackers = [(tracker, 1)]
        executor._load_poisoned = False
        executor.block_owners = _identity_owners()

        executor.submit_load_backs([], prerequisite_stream=object())

        tracker.set_consumers.assert_called_once_with(-1)

    def _prefetch_executor(self, *, batch_pages=2, timeout_base_s=10.0, rank=0):
        """A bare executor with the prefetch lane's state and a mock L3 store."""
        module = _load_executor_module_without_triton(force_isolated=True)
        executor = module.HostCacheExecutor.__new__(module.HostCacheExecutor)
        executor.attn_tp_rank = 0
        executor._ack_lock = threading.Lock()
        executor._ready_load_acks = []
        executor._completions = _lanes_module().CompletionQueue()
        executor._backup_futures = []
        executor._backup_poll_failed = False
        executor._l3_workers = None
        executor.l3_store = Mock()
        executor._l3_prefetch_lane = None
        executor._prefetch_jobs = {}
        executor._prefetch_acks = []
        executor._l3_prefetch_timeout_base_s = timeout_base_s
        executor._l3_prefetch_timeout_per_page_s = 0.0
        executor._l3_prefetch_batch_pages = batch_pages
        executor.block_owners = _identity_owners(num_groups=2)
        return module, executor

    @staticmethod
    def _prefetch_op(module, op_id, pages, *, groups=(0,)):
        """One prefetch op of ``pages`` prefix pages, page-major, one row per group."""
        ops_module = import_module("tokenspeed.runtime.cache.transfer.ops")
        rows = []
        for page in range(pages):
            for group in groups:
                rows.append(
                    ops_module.PrefetchRow(
                        group_id=group,
                        host_page=1 + page * len(groups) + groups.index(group),
                        content_hash=f"h{page}",
                        page_offset=page,
                        page_index=page,
                    )
                )
        return ops_module.PrefetchOp(
            op_id=op_id,
            request_id=f"r{op_id}",
            first_page=0,
            num_pages=pages,
            rows=tuple(rows),
        )

    @staticmethod
    def _settle(executor, op_id):
        for _ in range(200):
            progress = executor.l3_prefetch_progress()
            if progress[op_id][0]:
                return progress[op_id][1]
            time.sleep(0.01)
        raise AssertionError("prefetch lane did not finish")

    def test_prefetch_lane_lands_the_prefix_before_the_first_missing_page(self):
        """The lane fetches in prefix order, a batch at a time, and stops at
        the first page with a missing object (in any group); the landed count
        is a prefix length and the op is acknowledged once, with the count
        the hooks converged, never before they did."""
        module, executor = self._prefetch_executor(batch_pages=2)
        fetched = []

        def prefetch(pages):
            fetched.append([page[2] for page in pages])
            # Page 2's second group is missing: pages 0 and 1 land.
            return [not (page[2] == "h2" and page[0] == 1) for page in pages]

        executor.l3_store.prefetch.side_effect = prefetch
        op = self._prefetch_op(module, 5, 4, groups=(0, 1))
        executor.submit_prefetches([op, SimpleNamespace()])
        landed = self._settle(executor, 5)

        self.assertEqual(landed, 2)
        # Batches of two pages (four rows); the fetch stopped after the batch
        # holding the miss, so pages 3 were never requested.
        self.assertEqual(fetched, [["h0", "h0", "h1", "h1"], ["h2", "h2", "h3", "h3"]])
        self.assertEqual(executor.poll_results(), [])
        executor.complete_l3_prefetch(5, 1)  # the replica's common prefix
        (ack,) = executor.poll_results()
        self.assertEqual(
            (type(ack).__name__, ack.op_id, ack.landed_pages),
            ("PrefetchDoneEvent", 5, 1),
        )
        self.assertEqual(executor.poll_results(), [])
        self.assertEqual(executor.l3_prefetch_progress(), {})
        with self.assertRaises(KeyError):
            executor.complete_l3_prefetch(5, 1)

    def test_prefetch_lane_stops_at_the_deadline_and_on_a_backend_fault(self):
        module, executor = self._prefetch_executor(batch_pages=1, timeout_base_s=1.0)
        # The lane reads the clock once for the deadline and once before each
        # batch: the first batch is on time, the second past the deadline.
        clock = iter([0.0, 0.0] + [10.0] * 100)
        executor.l3_store.prefetch.side_effect = lambda pages: [True] * len(pages)
        with patch.object(module.time, "monotonic", side_effect=lambda: next(clock)):
            executor.submit_prefetches([self._prefetch_op(module, 1, 3)])
            self.assertEqual(self._settle(executor, 1), 1)
        self.assertEqual(executor.l3_store.prefetch.call_count, 1)

        executor.l3_store.prefetch.side_effect = [[True], RuntimeError("rpc")]
        executor.submit_prefetches([self._prefetch_op(module, 2, 3)])
        self.assertEqual(self._settle(executor, 2), 1)
        with self.assertRaises(ValueError):
            executor.complete_l3_prefetch(2, 4)  # more than the op has
        executor.complete_l3_prefetch(1, 1)
        executor.complete_l3_prefetch(2, 0)
        self.assertEqual(
            [(ack.op_id, ack.landed_pages) for ack in executor.poll_results()],
            [(1, 1), (2, 0)],
        )

    def test_prefetch_submission_validates_the_plan_before_starting_any_job(self):
        module, executor = self._prefetch_executor()
        ops_module = import_module("tokenspeed.runtime.cache.transfer.ops")
        with self.assertRaisesRegex(ValueError, "duplicate prefetch op id"):
            executor.submit_prefetches(
                [self._prefetch_op(module, 1, 1), self._prefetch_op(module, 1, 2)]
            )
        with self.assertRaisesRegex(ValueError, "carries no pages"):
            executor.submit_prefetches(
                [
                    ops_module.PrefetchOp(
                        op_id=3, request_id="r", first_page=0, num_pages=0, rows=()
                    )
                ]
            )
        # Rows must stay inside [first_page, first_page + num_pages) and in order.
        op = self._prefetch_op(module, 5, 2)
        bad = ops_module.PrefetchOp(
            op_id=5, request_id="r5", first_page=1, num_pages=2, rows=op.rows
        )
        with self.assertRaisesRegex(ValueError, "outside .* or out of order"):
            executor.submit_prefetches([bad])
        executor.l3_store = None
        with self.assertRaisesRegex(RuntimeError, "no storage backend"):
            executor.submit_prefetches([self._prefetch_op(module, 4, 1)])
        self.assertEqual(executor._prefetch_jobs, {})

    def test_a_prefetch_that_raised_lands_nothing_and_is_logged_once(self):
        """The hooks poll progress every round until the replica converges;
        a failed lane job is an error once, not a traceback per round."""
        module, executor = self._prefetch_executor(batch_pages=1)
        executor._l3_prefetch_batch_pages = 0  # range(..., 0): the job raises
        module.logger.error.reset_mock()  # the isolated module's logger is a Mock
        executor.submit_prefetches([self._prefetch_op(module, 9, 2)])
        self.assertEqual(self._settle(executor, 9), 0)
        for _ in range(3):
            self.assertEqual(executor.l3_prefetch_progress(), {9: (True, 0)})
        module.logger.error.assert_called_once()
        self.assertIn("prefetch 9 (r9) raised", module.logger.error.call_args.args[0])
        # Draining never re-raises a failed job: it has left the lane.
        executor._wait_l3_prefetches()
        executor.complete_l3_prefetch(9, 0)

    def test_the_weight_version_swap_drains_in_flight_prefetches(self):
        """A prefetch gets under the key prefix, so the prefix is not swapped
        while one is on the lane -- the backups were already drained; the
        prefetches are too, a failed one included."""
        module, executor = self._prefetch_executor()
        executor._l3_prefix_for_weight_version = lambda version: f"prefix/{version}"
        pending, failed = Future(), Future()
        executor._prefetch_jobs = {
            1: module._PrefetchJob(request_id="r1", num_pages=2, future=pending),
            2: module._PrefetchJob(request_id="r2", num_pages=2, future=failed),
        }
        failed.set_exception(RuntimeError("rpc"))
        swapped = threading.Event()
        executor.l3_store.set_key_prefix.side_effect = lambda prefix: swapped.set()
        worker = threading.Thread(
            target=executor.set_l3_weight_version, args=("v2",), daemon=True
        )
        worker.start()
        self.assertFalse(swapped.wait(0.05), "swapped while a prefetch was in flight")
        pending.set_result(2)
        self.assertTrue(swapped.wait(5.0))
        worker.join(5.0)
        executor.l3_store.set_key_prefix.assert_called_once_with("prefix/v2")

    def _owner_executor(self, *, num_groups=2):
        """An executor with replicated (identity) owner translations only."""
        executor_module = self._executor_module()
        executor = executor_module.HostCacheExecutor.__new__(
            executor_module.HostCacheExecutor
        )
        executor.block_owners = _identity_owners(num_groups)
        executor.snapshot_block_owners = _identity_owners(num_groups)
        return executor_module, executor

    def test_submit_preserves_group_identity(self):
        from tokenspeed.runtime.cache.transfer.ops import HostTier

        _, executor = self._owner_executor()
        op_ids = []
        transfers = []
        executor._append_transfers(
            [7],
            [[0, 1]],
            [[5, 5]],
            [[9, 9]],
            collected_op_ids=op_ids,
            transfers=transfers,
            source_is_device=True,
            tier=HostTier.L2,
        )
        self.assertEqual(op_ids, [7])
        self.assertEqual(transfers, [(0, 5, 9), (1, 5, 9)])

    def test_sharded_rows_keep_only_this_ranks_owned_blocks_on_both_ends(self):
        # A group dealt to two owners: rank r keeps the rows whose Device AND
        # Host block have residue r, in local ids; the op is collected even
        # when every row belongs to the other rank.
        from tokenspeed.runtime.cache.transfer.ops import HostTier
        from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation

        executor_module = self._executor_module()
        for rank, expected in ((0, [(0, 1, 2), (0, 2, 3)]), (1, [(0, 1, 1)])):
            with self.subTest(rank=rank):
                executor = executor_module.HostCacheExecutor.__new__(
                    executor_module.HostCacheExecutor
                )
                executor.block_owners = BlockOwnerTranslation(
                    shard_counts=[2],
                    device_virtual_counts=[9],
                    host_virtual_counts=[9],
                    rank=rank,
                )
                op_ids: list[int] = []
                transfers: list[tuple[int, int, int]] = []
                executor._append_transfers(
                    [7],
                    [[0, 0, 0]],
                    [[1, 3, 2]],
                    [[3, 5, 2]],
                    collected_op_ids=op_ids,
                    transfers=transfers,
                    source_is_device=True,
                    tier=HostTier.L2,
                )
                self.assertEqual(op_ids, [7])
                self.assertEqual(transfers, expected)
        with self.assertRaisesRegex(ValueError, "residue class"):
            executor._append_transfers(
                [8],
                [[0]],
                [[1]],
                [[2]],
                collected_op_ids=[],
                transfers=[],
                source_is_device=True,
                tier=HostTier.L2,
            )

    def _make_write_executor(self, executor_module):
        HostCacheExecutor = executor_module.HostCacheExecutor
        lanes_module = _lanes_module()
        executor = HostCacheExecutor.__new__(HostCacheExecutor)
        executor._ack_lock = threading.Lock()
        executor.attn_tp_rank = 0
        device = SimpleNamespace(type="cuda")
        executor.layout = SimpleNamespace(buffers=(SimpleNamespace(device=device),))
        executor.host_storage = SimpleNamespace(host_buffer="host")
        executor.transfer_backend = "auto"
        executor._completions = lanes_module.CompletionQueue()
        executor.block_owners = _identity_owners()
        executor.write_stream = Mock(name="write_stream")
        # Real lanes over mocked workspaces: the staging discipline under test
        # is the lane's, the executor only picks which lane an op rides.
        for lane_name in ("_ordered_write_lane", "_pinned_write_lane"):
            with patch.object(lanes_module, "HostTransferWorkspace", Mock):
                lane = lanes_module.HostTransferLane()
            lane.workspace.load_block_transfers.return_value = (1, (0, 1))
            lane.workspace.prepare_backend.return_value = SimpleNamespace(
                uses_device_tables=True,
            )
            setattr(executor, lane_name, lane)
        executor._transfer_geometry = SimpleNamespace(
            device_rows=object(),
            layer_slices=((0, 2), (2, 1)),
            num_field_rows=3,
        )
        return executor, device

    def test_writeback_rides_the_write_stream_ordered_after_the_caller(self):
        executor_module = self._executor_module()
        lanes_module = _lanes_module()
        executor, device = self._make_write_executor(executor_module)
        lane = executor._pinned_write_lane
        prerequisite_stream = object()
        finish = Mock()
        metadata_done = Mock()
        # Every stream the executor touches is one the caller named; the
        # thread's current stream is never consulted.
        no_current_stream = patch.object(
            lanes_module.device_module,
            "current_stream",
            side_effect=AssertionError("current stream must not be consulted"),
        )

        with (
            no_current_stream,
            patch.object(lanes_module.device_module, "stream") as stream_ctx,
            patch.object(
                lanes_module.device_module,
                "Event",
                side_effect=[metadata_done, finish],
            ),
            patch.object(lanes_module, "transfer_cache_blocks") as transfer,
        ):
            fence = executor._start_writing(
                [7],
                [(0, 5, 9)],
                backup_pages=[],
                lane=lane,
                prerequisite_stream=prerequisite_stream,
            )

        # On the write stream, ordered after the prerequisite stream the
        # caller named (the forwards that wrote the pages). The completion
        # event is handed back so a stream-ordered submission can fence the
        # caller's fence stream on it; a pinned one simply drops it.
        self.assertIs(fence, finish)
        executor.write_stream.wait_stream.assert_called_once_with(prerequisite_stream)
        # The address tables and the metadata H2D are enqueued on the write
        # stream too: the payload kernel reads them from that stream, and a
        # copy left on the caller's stream would land behind the wait above
        # with nothing ordering it ahead of the kernel.
        stream_ctx.assert_called_once_with(executor.write_stream)
        lane.workspace.load_block_transfers.assert_called_once_with(
            [(0, 5, 9)], geometry=executor._transfer_geometry
        )
        lane.workspace.commit_block_transfers.assert_called_once_with(
            1, device, non_blocking=True
        )
        transfer.assert_called_once_with(
            "d2h",
            executor.layout.buffers,
            executor.host_storage.host_buffer,
            executor._transfer_geometry,
            lane.workspace,
            executor.write_stream,
            num_blocks=1,
            geometry_offset=0,
            num_geometry_rows=3,
            backend="auto",
            grid_cap=None,
            layer_ready_flags=None,
        )
        finish.record.assert_called_once_with(executor.write_stream)
        metadata_done.record.assert_called_once_with(executor.write_stream)
        metadata_done.synchronize.assert_not_called()
        self.assertIsNone(executor._ordered_write_lane.metadata_done)

        # Refill must wait for metadata, but must never wait for payload ACK;
        # the upload and its retirement event sit inside the write-stream
        # context, the payload launch names the stream explicitly.
        for ready in (False, True):
            with self.subTest(metadata_ready=ready):
                metadata_done.reset_mock()
                metadata_done.query.return_value = ready
                order = Mock()
                order.attach_mock(metadata_done.synchronize, "retire")
                order.attach_mock(lane.workspace.load_block_transfers, "refill")
                order.attach_mock(lane.workspace.commit_block_transfers, "upload")
                order.attach_mock(metadata_done.record, "record")
                with (
                    no_current_stream,
                    patch.object(lanes_module.device_module, "stream") as stream_ctx,
                    patch.object(
                        lanes_module.device_module, "Event", return_value=finish
                    ),
                    patch.object(lanes_module, "transfer_cache_blocks") as transfer,
                ):
                    order.attach_mock(stream_ctx, "stream")
                    order.attach_mock(transfer, "payload")
                    executor._start_writing(
                        [8],
                        [(0, 6, 10)],
                        backup_pages=[],
                        lane=lane,
                        prerequisite_stream=prerequisite_stream,
                    )
                names = [call[0] for call in order.mock_calls]
                self.assertEqual(
                    names,
                    ([] if ready else ["retire"])
                    + [
                        "refill",
                        "stream",
                        "stream().__enter__",
                        "upload",
                        "record",
                        "stream().__exit__",
                        "payload",
                    ],
                )
                finish.synchronize.assert_not_called()

    def test_submit_write_backs_fences_only_stream_ordered_ops(self):
        executor_module = self._executor_module()
        lanes_module = _lanes_module()
        executor, _ = self._make_write_executor(executor_module)
        fence_stream = Mock(name="fence_stream")
        prerequisite = object()
        ordered_finish = Mock(name="ordered_finish")
        pinned_finish = Mock(name="pinned_finish")
        events = iter(
            [
                Mock(name="ordered_meta"),
                ordered_finish,
                Mock(name="pinned_meta"),
                pinned_finish,
            ]
        )

        class WriteBackOp:
            def __init__(self):
                self.op_ids = [11, 12, 13]
                self.group_ids = [[0], [0], [0]]
                self.src_pages = [[1], [2], [3]]
                self.dst_pages = [[5], [6], [7]]
                self.source_pinned = [True, False, True]
                self.content_hashes = [["pinned-a"], ["ordered"], ["pinned-b"]]
                self.page_offsets = [[0], [1], [2]]

        with (
            patch.object(
                executor_module.Cache, "WriteBackOp", WriteBackOp, create=True
            ),
            patch.object(
                lanes_module.device_module, "stream", return_value=nullcontext()
            ),
            patch.object(
                lanes_module.device_module,
                "Event",
                side_effect=lambda: next(events),
            ),
            patch.object(lanes_module, "transfer_cache_blocks") as transfer,
        ):
            executor.submit_write_backs(
                [WriteBackOp()],
                prerequisite_stream=prerequisite,
                fence_stream=fence_stream,
            )

        # The stream-ordered op (12) launches first and the fence stream the
        # caller named waits on ITS completion only; the pinned ops (11, 13)
        # follow on the write stream and fence nothing -- the scheduler holds
        # their sources.
        ordered_lane = executor._ordered_write_lane
        pinned_lane = executor._pinned_write_lane
        ordered_lane.workspace.load_block_transfers.assert_called_once_with(
            [(0, 2, 6)], geometry=executor._transfer_geometry
        )
        pinned_lane.workspace.load_block_transfers.assert_called_once_with(
            [(0, 1, 5), (0, 3, 7)], geometry=executor._transfer_geometry
        )
        self.assertEqual(
            [call.args[4] for call in transfer.call_args_list],
            [ordered_lane.workspace, pinned_lane.workspace],
        )
        fence_stream.wait_event.assert_called_once_with(ordered_finish)
        pending = executor._completions._pending
        self.assertEqual(
            [(finish, ack.kind, ack.op_ids) for finish, ack in pending],
            [
                (ordered_finish, executor_module._AckKind.WRITE_BACK, [12]),
                (pinned_finish, executor_module._AckKind.WRITE_BACK, [11, 13]),
            ],
        )
        self.assertEqual(
            [ack.backup_pages for _, ack in pending],
            [[(0, 6, "ordered", 1)], [(0, 5, "pinned-a", 0), (0, 7, "pinned-b", 2)]],
        )
        self.assertEqual(
            executor.write_stream.wait_stream.call_args_list,
            [call(prerequisite), call(prerequisite)],
            "both launches order behind the prerequisite stream the caller named",
        )

    def test_mixed_write_lanes_ack_only_their_completed_l3_put(self):
        executor_module = self._executor_module()
        lanes_module = _lanes_module()
        executor, _ = self._make_write_executor(executor_module)
        executor._ready_load_acks = []
        executor._prefetch_acks = []
        executor._backup_futures = []
        executor._l3_workers = None
        executor.l3_store = Mock()
        executor._write_done = lambda op_id: op_id
        started = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        ordered_pages = [(0, 6, "ordered", 0)]
        pinned_pages = [(0, 5, "pinned", 0)]

        def backup(pages):
            index = 0 if pages == ordered_pages else 1
            self.assertEqual(pages, [ordered_pages, pinned_pages][index])
            started[index].set()
            if not release[index].wait(timeout=10):
                raise TimeoutError("mixed-lane PUT gate was not released")
            return [True]

        executor.l3_store.backup.side_effect = backup
        ordered_finish = Mock()
        pinned_finish = Mock()
        ordered_finish.query.return_value = False
        pinned_finish.query.return_value = False
        events = iter([Mock(), ordered_finish, Mock(), pinned_finish])

        class WriteBackOp:
            def __init__(self):
                self.op_ids = [11, 12]
                self.group_ids = [[0], [0]]
                self.src_pages = [[1], [2]]
                self.dst_pages = [[5], [6]]
                self.source_pinned = [True, False]
                self.content_hashes = [["pinned"], ["ordered"]]
                self.page_offsets = [[0], [0]]

        def poll_until_ready():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                ready = executor.poll_results()
                if ready:
                    return ready
                time.sleep(0.001)
            self.fail("completed L3 PUT did not produce its ACK")

        try:
            with (
                patch.object(executor_module.Cache, "WriteBackOp", WriteBackOp),
                patch.object(lanes_module.device_module, "current_stream"),
                patch.object(
                    lanes_module.device_module,
                    "stream",
                    return_value=nullcontext(),
                ),
                patch.object(
                    lanes_module.device_module,
                    "Event",
                    side_effect=lambda: next(events),
                ),
                patch.object(lanes_module, "transfer_cache_blocks"),
            ):
                executor.submit_write_backs(
                    [WriteBackOp()],
                    prerequisite_stream=object(),
                    fence_stream=Mock(),
                )

            self.assertEqual(executor.poll_results(), [])
            executor.l3_store.backup.assert_not_called()
            ordered_finish.query.return_value = True
            pinned_finish.query.return_value = True
            self.assertEqual(executor.poll_results(), [])
            self.assertTrue(started[0].wait(timeout=10))
            self.assertFalse(started[1].is_set())
            self.assertEqual(executor.poll_results(), [])

            release[0].set()
            self.assertTrue(started[1].wait(timeout=10))
            self.assertEqual(poll_until_ready(), [12])
            self.assertEqual(executor.poll_results(), [])
            self.assertEqual(
                [op_ids for _, op_ids, _ in executor._backup_futures], [[11]]
            )

            release[1].set()
            self.assertEqual(poll_until_ready(), [11])
            self.assertEqual(executor.poll_results(), [])
            self.assertEqual(executor._backup_futures, [])
            executor.l3_store.backup.assert_has_calls(
                [call(ordered_pages), call(pinned_pages)]
            )
        finally:
            for gate in release:
                gate.set()
            if executor._l3_workers is not None:
                executor._l3_workers.shutdown(wait=True)

    def test_submit_write_backs_without_stream_ordered_ops_fences_nothing(self):
        executor_module = self._executor_module()
        lanes_module = _lanes_module()
        executor, _ = self._make_write_executor(executor_module)
        fence_stream = Mock(name="fence_stream")
        prerequisite = object()

        class WriteBackOp:
            def __init__(self):
                self.op_ids = [11]
                self.group_ids = [[0]]
                self.src_pages = [[1]]
                self.dst_pages = [[5]]
                self.source_pinned = [True]
                self.content_hashes = [[""]]
                self.page_offsets = [[0]]

        with (
            patch.object(
                executor_module.Cache, "WriteBackOp", WriteBackOp, create=True
            ),
            patch.object(
                lanes_module.device_module, "stream", return_value=nullcontext()
            ),
            patch.object(lanes_module.device_module, "Event", side_effect=Mock),
            patch.object(lanes_module, "transfer_cache_blocks"),
        ):
            executor.submit_write_backs(
                [WriteBackOp()],
                prerequisite_stream=prerequisite,
                fence_stream=fence_stream,
            )

        fence_stream.wait_event.assert_not_called()
        executor._ordered_write_lane.workspace.load_block_transfers.assert_not_called()

    def test_submit_write_backs_rejects_ragged_guard_vector(self):
        executor_module = self._executor_module()
        executor, _ = self._make_write_executor(executor_module)
        prerequisite = object()

        class WriteBackOp:
            def __init__(self):
                self.op_ids = [11, 12]
                self.group_ids = [[0], [0]]
                self.src_pages = [[1], [2]]
                self.dst_pages = [[5], [6]]
                self.source_pinned = [True]

        with patch.object(
            executor_module.Cache, "WriteBackOp", WriteBackOp, create=True
        ):
            with self.assertRaises(ValueError):
                executor.submit_write_backs(
                    [WriteBackOp()],
                    prerequisite_stream=prerequisite,
                    fence_stream=object(),
                )

    def test_cache_operation_without_transfers_is_refused(self):
        # An op is acknowledged by its copy's completion event, so one with
        # nothing to copy could never be acknowledged; the scheduler never
        # emits one, and the runtime refuses rather than inventing an ack.
        from tokenspeed.runtime.cache.transfer.ops import HostTier

        _, executor = self._owner_executor()
        for source_is_device in (True, False):
            with self.subTest(source_is_device=source_is_device):
                op_ids: list[int] = []
                transfers: list[tuple[int, int, int]] = []
                with self.assertRaisesRegex(ValueError, "operation 11 carries no"):
                    executor._append_transfers(
                        [7, 11],
                        [[0], []],
                        [[1], []],
                        [[5], []],
                        collected_op_ids=op_ids,
                        transfers=transfers,
                        source_is_device=source_is_device,
                        tier=HostTier.L2,
                    )

    def test_loadback_logs_non_empty_batch(self):
        executor_module, executor, _, geometry, workspace = self._make_load_executor(
            consumers=(("field",),),
            layer_slices=((0, 1),),
            backend="dma",
            device_rows=None,
        )
        workspace.load_block_transfers.return_value = (2, (0, 2))
        load_events = SimpleNamespace(start_event=Mock(), layer_done_events=[None])
        tracker = Mock()
        tracker.begin_load.return_value = 0
        tracker.event_sets = [load_events]
        executor._load_trackers = [(tracker, 1)]
        finish = Mock()
        prerequisite_stream = object()

        with (
            patch.object(executor_module, "get_is_capture_mode", return_value=False),
            patch.object(
                executor_module.device_module, "stream", return_value=nullcontext()
            ),
            patch.object(executor_module.device_module, "Event", return_value=finish),
            patch.object(executor_module, "transfer_cache_blocks") as transfer,
            patch.object(executor_module.logger, "info") as log_info,
        ):
            executor._start_loading(
                [9],
                [(0, 2, 1), (0, 5, 4)],
                prerequisite_stream=prerequisite_stream,
            )

        workspace.load_block_transfers.assert_called_once_with(
            [(0, 2, 1), (0, 5, 4)], geometry=geometry
        )
        workspace.commit_block_transfers.assert_not_called()
        transfer.assert_called_once()
        log_info.assert_called_once_with(
            "[L2] load started: operations=1 blocks=2",
        )
        # The load orders after the prerequisite stream the caller named --
        # the one that zeroed its destinations -- not after the current stream.
        load_events.start_event.record.assert_called_once_with(prerequisite_stream)
        load_events.start_event.wait.assert_called_once_with(executor.load_stream)

    def test_resolved_transport_selects_events_even_with_device_geometry(self):
        for uses_device_tables in (False, True):
            with self.subTest(uses_device_tables=uses_device_tables):
                module, executor, _, _, workspace = self._make_load_executor(
                    consumers=(("first",), ("second",)),
                    layer_slices=((0, 1), (1, 1)),
                    device_rows="bound geometry",
                )
                workspace.prepare_backend.return_value = SimpleNamespace(
                    uses_device_tables=uses_device_tables, layer_ready=False
                )
                events = _LoadEvents(start_event=Mock(), layer_done_events=[None, None])
                tracker = Mock()
                tracker.begin_load.return_value = 0
                tracker.event_sets = [events]
                executor._load_trackers = [(tracker, 2)]
                with (
                    patch.object(module, "get_is_capture_mode", return_value=False),
                    patch.object(
                        module.device_module, "stream", return_value=nullcontext()
                    ),
                    patch.object(
                        module.device_module, "Event", side_effect=[Mock(), Mock()]
                    ),
                    patch.object(module, "transfer_cache_blocks") as transfer,
                ):
                    executor._start_loading(
                        [9], [(0, 1, 1)], prerequisite_stream=object()
                    )
                self.assertEqual(
                    workspace.commit_block_transfers.call_count, int(uses_device_tables)
                )
                workspace.prepare_layer_ready.assert_not_called()
                self.assertIsNone(events.layer_ready_flags)
                self.assertEqual(transfer.call_count, 2)
                self.assertTrue(
                    all(event is not None for event in events.layer_done_events)
                )

    def test_kernel_init_builds_consumer_ordered_static_geometry_once(self):
        executor_module = self._executor_module()
        lanes_module = _lanes_module()
        HostCacheExecutor = executor_module.HostCacheExecutor

        device = SimpleNamespace(type="cuda")
        buffer = SimpleNamespace(device=device)
        fields = {
            "target.0.k": SimpleNamespace(
                field_id="target.0.k",
                device_buffer_index=0,
                device_block_zero_offset_bytes=8,
                block_stride_bytes=16,
                payload_bytes=12,
            ),
            "target.2.state": SimpleNamespace(
                field_id="target.2.state",
                device_buffer_index=1,
                device_block_zero_offset_bytes=32,
                block_stride_bytes=64,
                payload_bytes=20,
            ),
            "draft.0.k": SimpleNamespace(
                field_id="draft.0.k",
                device_buffer_index=0,
                device_block_zero_offset_bytes=48,
                block_stride_bytes=16,
                payload_bytes=12,
            ),
        }
        combined_layout = SimpleNamespace(
            num_lcm_blocks=11,
            buffers=(buffer, SimpleNamespace(device=device)),
            groups=(
                SimpleNamespace(
                    group_id="state",
                    cache_blocks_per_lcm_block=4,
                    fields=(fields["target.2.state"],),
                ),
                SimpleNamespace(
                    group_id="full",
                    cache_blocks_per_lcm_block=8,
                    fields=(fields["target.0.k"], fields["draft.0.k"]),
                ),
            ),
            # Target layers precede draft layers, and empty layers are retained.
            consumers=(
                ("target.0.k",),
                (),
                ("target.2.state",),
                ("draft.0.k",),
            ),
        )
        target_layout = SimpleNamespace(
            consumers=(("target.0.k",), (), ("target.2.state",))
        )
        draft_layout = SimpleNamespace(consumers=(("draft.0.k",),))
        target_pool = Mock()
        target_pool.cache_transfer_layout.return_value = target_layout
        target_pool.arena.runtime_contract = _replicated_contract(
            combined_layout, num_lcm_blocks=11
        )
        draft_pool = Mock()
        draft_pool.cache_transfer_layout.return_value = draft_layout
        storage = SimpleNamespace(
            host_cache_block_bytes=(20, 24),
            host_field_offsets=((0,), (0, 12)),
            host_lcm_block_bytes=192,
            num_host_lcm_blocks=3,
            host_buffer="host",
        )
        unbound_geometry = Mock()
        bound_geometry = object()
        unbound_geometry.bind.return_value = bound_geometry
        trackers = []

        def make_tracker(consumer_count):
            tracker = Mock()
            tracker.event_sets = [object(), object()]
            trackers.append((consumer_count, tracker))
            return tracker

        with (
            patch.object(
                executor_module,
                "combine_cache_transfer_layouts",
                return_value=combined_layout,
            ),
            patch.object(
                executor_module,
                "compute_host_lcm_block_bytes",
                return_value=storage.host_lcm_block_bytes,
            ),
            patch.object(executor_module, "HostCacheStorage", return_value=storage),
            patch.object(
                lanes_module.psutil,
                "virtual_memory",
                return_value=SimpleNamespace(available=10**12),
            ),
            patch.object(
                executor_module, "LayerwiseLoadTracker", side_effect=make_tracker
            ),
            patch.object(executor_module, "new_cache_stream", return_value="load"),
            patch.object(executor_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(lanes_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(
                lanes_module,
                "build_host_transfer_geometry",
                return_value=unbound_geometry,
            ) as build_geometry,
        ):
            executor = HostCacheExecutor(
                target_pool,
                draft_pool=draft_pool,
                l2_tier=True,
                host_ratio=1.0,
                host_size_gb=0,
                snapshot_pool=NO_POOL,
                slot_state_exporters=None,
                io_backend="kernel",
                attn_tp_rank=0,
                kvp_rank=0,
            )

        build_geometry.assert_called_once_with(
            rows=(
                (1, 0, 8, 16, 24, 0, 8, 12),
                (0, 1, 32, 64, 20, 0, 4, 20),
                (1, 0, 48, 16, 24, 12, 8, 12),
            ),
            layer_slices=((0, 1), (1, 0), (1, 1), (2, 1)),
            group_packing=(4, 8),
            host_lcm_block_bytes=192,
            num_host_lcm_blocks=3,
            num_device_lcm_blocks=11,
            num_device_buffers=2,
        )
        unbound_geometry.bind.assert_called_once_with(device, non_blocking=False)
        self.assertIs(executor._transfer_geometry, bound_geometry)
        self.assertEqual([count for count, _ in trackers], [3, 1])

    def test_direct_and_npu_init_keep_geometry_on_the_host(self):
        executor_module = self._executor_module()
        lanes_module = _lanes_module()
        HostCacheExecutor = executor_module.HostCacheExecutor

        pool = Mock()
        pool.arena.runtime_contract = SimpleNamespace(
            group_specs=(SimpleNamespace(group_id="group", shard_count=1),),
            virtual_block_counts={"group": 3},
        )
        storage = SimpleNamespace(
            host_cache_block_bytes=(16,),
            host_field_offsets=((0,),),
            host_lcm_block_bytes=64,
            num_host_lcm_blocks=2,
            host_buffer="host",
        )
        tracker = Mock()
        tracker.event_sets = [object()]

        with (
            patch.object(
                executor_module,
                "compute_host_lcm_block_bytes",
                return_value=storage.host_lcm_block_bytes,
            ),
            patch.object(executor_module, "HostCacheStorage", return_value=storage),
            patch.object(
                lanes_module.psutil,
                "virtual_memory",
                return_value=SimpleNamespace(available=10**12),
            ),
            patch.object(executor_module, "LayerwiseLoadTracker", return_value=tracker),
            patch.object(executor_module, "new_cache_stream", return_value="load"),
            patch.object(executor_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(lanes_module, "HostTransferWorkspace", side_effect=Mock),
            patch.object(
                lanes_module,
                "build_host_transfer_geometry",
                side_effect=lambda **_kwargs: SimpleNamespace(
                    device_rows=None,
                    bind=Mock(),
                ),
            ) as build_geometry,
        ):
            for io_backend, device_type in (("direct", "cuda"), ("kernel", "npu")):
                with self.subTest(io_backend=io_backend, device_type=device_type):
                    field = SimpleNamespace(
                        field_id="field",
                        device_buffer_index=0,
                        device_block_zero_offset_bytes=0,
                        block_stride_bytes=16,
                        payload_bytes=16,
                    )
                    layout = SimpleNamespace(
                        num_lcm_blocks=2,
                        buffers=(
                            SimpleNamespace(device=SimpleNamespace(type=device_type)),
                        ),
                        groups=(
                            SimpleNamespace(
                                group_id="group",
                                cache_blocks_per_lcm_block=1,
                                fields=(field,),
                            ),
                        ),
                        consumers=(("field",),),
                    )
                    pool.cache_transfer_layout.return_value = layout
                    executor = HostCacheExecutor(
                        pool,
                        l2_tier=True,
                        host_ratio=1.0,
                        host_size_gb=0,
                        snapshot_pool=NO_POOL,
                        slot_state_exporters=None,
                        io_backend=io_backend,
                        attn_tp_rank=0,
                        kvp_rank=0,
                    )
                    self.assertIsNone(executor._transfer_geometry.device_rows)
                    executor._transfer_geometry.bind.assert_not_called()

        self.assertEqual(build_geometry.call_count, 2)

    def test_optional_dependency_shim_restores_existing_modules(self):
        protected_names = (
            "tokenspeed_kernel.ops.kvcache.host_transfer",
            "tokenspeed_scheduler",
            "tokenspeed.runtime.cache.l2.layerwise_load",
            "tokenspeed.runtime.cache.l2.storage",
            "tokenspeed.runtime.cache.transfer.lanes",
            "tokenspeed.runtime.cache.transfer.layout",
            "tokenspeed.runtime.cache.transfer.ownership",
            "tokenspeed.runtime.execution.forward_step",
            "tokenspeed.runtime.utils",
            "tokenspeed.runtime.cache.l2.executor",
        )
        sentinels = {name: ModuleType(name) for name in protected_names}

        with patch.dict(sys.modules, sentinels, clear=False):
            isolated = _load_executor_module_without_triton(force_isolated=True)

            self.assertIsNot(isolated, sentinels[protected_names[-1]])
            for name, sentinel in sentinels.items():
                self.assertIs(sys.modules[name], sentinel)

    def test_two_isolated_loads_preserve_imported_real_modules(self):
        # psutil is a real dependency of the shared lane module; the isolated
        # loader imports it ahead of its sys.modules snapshot so the snapshot's
        # restore cannot drop it again.
        first = _load_executor_module_without_triton(force_isolated=True)
        first_psutil = sys.modules["psutil"]
        self.assertIs(_lanes_module().psutil, first_psutil)

        second = _load_executor_module_without_triton(force_isolated=True)

        self.assertIsNot(first, second)
        self.assertIs(sys.modules["psutil"], first_psutil)
        self.assertIs(_lanes_module().psutil, first_psutil)

    def test_loadback_commits_block_ids_once_and_launches_one_flagged_kernel(self):
        executor_module, executor, device, geometry, workspace = (
            self._make_load_executor(
                consumers=(("layer.0",), (), ("layer.2",)),
                layer_slices=((0, 2), (2, 0), (2, 1)),
                device_rows=object(),
            )
        )
        flags = Mock()
        flags.__getitem__ = Mock(return_value=flags)
        workspace.prepare_layer_ready.return_value = flags
        load_events = _LoadEvents(
            start_event=Mock(),
            layer_done_events=[None, None, None],
            layer_ready_flags=None,
            wait_layer_ready=None,
            layer_ready_init_event=Mock(),
        )
        tracker = Mock()
        tracker.begin_load.return_value = 0
        tracker.event_sets = [load_events]
        executor._load_trackers = [(tracker, 3)]
        finish = Mock()

        with (
            patch.object(executor_module, "get_is_capture_mode", return_value=False),
            patch.object(
                executor_module.device_module,
                "stream",
                return_value=nullcontext(),
            ),
            patch.object(executor_module.device_module, "Event", return_value=finish),
            patch.object(executor_module, "transfer_cache_blocks") as transfer,
        ):
            executor._start_loading([9], [(0, 2, 1)], prerequisite_stream=object())

        workspace.load_block_transfers.assert_called_once_with(
            [(0, 2, 1)], geometry=geometry
        )
        workspace.commit_block_transfers.assert_called_once_with(
            1, device, non_blocking=True
        )
        workspace.prepare_layer_ready.assert_called_once_with(3, device)
        load_events.layer_ready_init_event.record.assert_called_once_with(
            executor.load_stream
        )
        transfer.assert_called_once_with(
            "h2d",
            executor.layout.buffers,
            executor.host_storage.host_buffer,
            geometry,
            workspace,
            executor.load_stream,
            num_blocks=1,
            geometry_offset=0,
            num_geometry_rows=3,
            backend="auto",
            layer_ready_flags=flags,
            grid_cap=None,
        )
        finish.record.assert_called_once_with(executor.load_stream)
        self.assertEqual(load_events.layer_done_events, [finish, finish, finish])
        self.assertIs(load_events.layer_ready_flags, flags)
        self.assertIs(load_events.wait_layer_ready, executor_module.wait_layer_ready)
        ((pending_finish, ack),) = executor._completions._pending
        self.assertIs(pending_finish, finish)
        self.assertEqual(
            (ack.kind, ack.op_ids), (executor_module._AckKind.LOAD_BACK, [9])
        )

    def test_loadback_launch_failure_retires_all_target_and_draft_events(self):
        executor_module, executor, _, _, _ = self._make_load_executor(
            consumers=(("target.0",), ("target.1",), ("draft.0",)),
            layer_slices=((0, 1), (1, 1), (2, 1)),
            device_rows=object(),
            load_stream=Mock(),
        )
        target_events = _LoadEvents(
            start_event=Mock(),
            layer_done_events=[Mock(), Mock()],
            layer_ready_init_event=Mock(),
        )
        draft_events = _LoadEvents(
            start_event=Mock(),
            layer_done_events=[Mock()],
            layer_ready_init_event=Mock(),
        )
        target_tracker = Mock()
        target_tracker.begin_load.return_value = 0
        target_tracker.event_sets = [target_events]
        draft_tracker = Mock()
        draft_tracker.begin_load.return_value = 0
        draft_tracker.event_sets = [draft_events]
        executor._load_trackers = [(target_tracker, 2), (draft_tracker, 1)]
        retirement = Mock()

        with (
            patch.object(executor_module, "get_is_capture_mode", return_value=False),
            patch.object(
                executor_module.device_module,
                "stream",
                return_value=nullcontext(),
            ),
            patch.object(
                executor_module.device_module,
                "Event",
                return_value=retirement,
            ),
            patch.object(
                executor_module,
                "transfer_cache_blocks",
                side_effect=RuntimeError("layer launch failed"),
            ) as transfer,
        ):
            with self.assertRaisesRegex(RuntimeError, "layer launch failed"):
                executor._start_loading([9], [(0, 2, 1)], prerequisite_stream=object())

        self.assertEqual(transfer.call_count, 1)
        retirement.record.assert_called_once_with(executor.load_stream)
        self.assertEqual(
            target_events.layer_done_events,
            [retirement, retirement],
        )
        self.assertEqual(draft_events.layer_done_events, [retirement])
        executor.load_stream.synchronize.assert_not_called()
        self.assertEqual(executor._completions._pending, [])

    def test_failed_retirement_sync_poisons_executor_and_preserves_original_error(self):
        executor_module, executor, _, _, _ = self._make_load_executor(
            consumers=(("target.0",),),
            layer_slices=((0, 1),),
            device_rows=object(),
            load_stream=Mock(),
        )
        load_events = _LoadEvents(
            start_event=Mock(),
            layer_done_events=[Mock()],
            layer_ready_init_event=Mock(),
        )
        tracker = Mock()
        tracker.begin_load.return_value = 0
        tracker.event_sets = [load_events]
        executor._load_trackers = [(tracker, 1)]
        original_error = RuntimeError("original layer launch failed")
        retirement = Mock()
        retirement.record.side_effect = RuntimeError("retirement record failed")
        executor.load_stream.synchronize.side_effect = RuntimeError(
            "retirement sync failed"
        )

        with (
            patch.object(executor_module, "get_is_capture_mode", return_value=False),
            patch.object(
                executor_module.device_module,
                "stream",
                return_value=nullcontext(),
            ),
            patch.object(
                executor_module.device_module,
                "Event",
                return_value=retirement,
            ),
            patch.object(
                executor_module,
                "transfer_cache_blocks",
                side_effect=original_error,
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                executor._start_loading([9], [(0, 2, 1)], prerequisite_stream=object())

            self.assertIs(raised.exception, original_error)
            self.assertEqual(str(raised.exception), "original layer launch failed")
            self.assertTrue(executor._load_poisoned)
            notes = getattr(raised.exception, "__notes__", ())
            self.assertTrue(any("retirement record failed" in note for note in notes))
            self.assertTrue(any("retirement sync failed" in note for note in notes))

            executor.load_stream.synchronize.side_effect = None
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                executor._start_loading([10], [(0, 3, 2)], prerequisite_stream=object())

        tracker.begin_load.assert_called_once_with()


class L3FlatKvExecutorTest(unittest.TestCase):
    def test_wait_l3_backups_snapshots_under_lock_and_waits_outside(self):
        module = _load_executor_module_without_triton(force_isolated=True)
        lock = threading.Lock()
        observed = []

        class PendingBackups(list):
            def __iter__(self):
                self_test.assertTrue(lock.locked())
                return super().__iter__()

        def completed():
            self.assertFalse(lock.locked())
            observed.append("done")

        self_test = self
        future = Mock()
        future.result.side_effect = completed
        executor = SimpleNamespace(
            _ack_lock=lock,
            _backup_futures=PendingBackups([(future, [7], [(0, 1, "h0", 0)])]),
        )
        module.HostCacheExecutor._wait_l3_backups(executor)
        self.assertEqual(observed, ["done"])
        future.result.side_effect = RuntimeError("backup failed")
        with self.assertRaisesRegex(RuntimeError, "backup failed"):
            module.HostCacheExecutor._wait_l3_backups(executor)
        self.assertFalse(lock.locked())

    def test_storage_pages_list_the_kept_hashed_rows_by_local_host_id(self):
        try:
            from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        operation = SimpleNamespace(
            content_hashes=[["h0", "h1", ""]],
            page_offsets=[[0, 1, 0]],
            group_ids=[[0, 1, 0]],
            src_pages=[[3, 4, 9]],
            dst_pages=[[7, 8, 10]],
        )
        # The kept rows name the Host end by local id (the destination a
        # write-back wrote); an unkeyed row (no content hash) is not backed up.
        write_pages = HostCacheExecutor._storage_pages(
            operation, kept_rows=[(0, [(0, 7), (1, 8), (2, 10)])]
        )
        self.assertEqual(write_pages, [(0, 7, "h0", 0), (1, 8, "h1", 1)])
        # Only the rows this rank kept are listed: a KVP peer's row is not.
        self.assertEqual(
            HostCacheExecutor._storage_pages(operation, kept_rows=[(0, [(1, 2)])]),
            [(1, 2, "h1", 1)],
        )

    def test_ack_requires_backup_pages(self):
        try:
            from tokenspeed.runtime.cache.l2.executor import (
                HostCacheExecutor,
                _Ack,
                _AckKind,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        signature = inspect.signature(_Ack)
        self.assertIs(
            signature.parameters["backup_pages"].default, inspect.Parameter.empty
        )
        with self.assertRaises(TypeError):
            _Ack(_AckKind.WRITE_BACK, [1])
        start_writing = inspect.signature(HostCacheExecutor._start_writing)
        self.assertIs(
            start_writing.parameters["backup_pages"].default, inspect.Parameter.empty
        )
        with self.assertRaises(TypeError):
            HostCacheExecutor._start_writing(object(), [7], [(0, 1, 1)])

    def test_l2_constructor_does_not_attach_l3_from_optional_storage(self):
        try:
            from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        signature = inspect.signature(HostCacheExecutor.__init__)
        self.assertNotIn("storage_backend", signature.parameters)
        self.assertNotIn("storage_key_prefix", signature.parameters)
        self.assertNotIn("storage_rank", signature.parameters)
        self.assertIs(
            signature.parameters["attn_tp_rank"].default, inspect.Parameter.empty
        )

    def test_poll_results_backs_up_host_pages_asynchronously(self):
        try:
            from tokenspeed.runtime.cache.l2.executor import (
                HostCacheExecutor,
                _Ack,
                _AckKind,
            )
            from tokenspeed.runtime.cache.transfer.lanes import CompletionQueue
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        started = threading.Event()
        release = threading.Event()

        def backup(pages):
            del pages
            started.set()
            if not release.wait(timeout=2):
                raise TimeoutError("L3 backup was not released")
            return [True]

        executor = HostCacheExecutor.__new__(HostCacheExecutor)
        executor._ack_lock = threading.Lock()
        executor._completions = CompletionQueue()
        executor._ready_load_acks = []
        executor._prefetch_acks = []
        executor._backup_futures = []
        executor._l3_workers = None
        executor.l3_store = Mock()
        executor.l3_store.backup.side_effect = backup
        finish = Mock()
        finish.query.return_value = True
        executor._completions.push(
            finish,
            _Ack(
                kind=_AckKind.WRITE_BACK,
                op_ids=[7],
                backup_pages=[(0, 1, "h0", 0)],
            ),
        )

        first = executor.poll_results()
        self.assertEqual(first, [])
        self.assertTrue(started.wait(timeout=2))
        executor.l3_store.backup.assert_called_once_with([(0, 1, "h0", 0)])
        release.set()
        deadline = time.monotonic() + 2
        second = []
        try:
            while time.monotonic() < deadline:
                second = executor.poll_results()
                if second:
                    break
                time.sleep(0.01)
            self.assertEqual(len(second), 1)
            self.assertEqual(int(second[0].op_id), 7)
        finally:
            workers = executor._l3_workers
            if workers is not None:
                workers.shutdown(wait=True)

    def test_backup_failure_does_not_ack_writeback(self):
        try:
            from tokenspeed.runtime.cache.l2.executor import (
                HostCacheExecutor,
                _Ack,
                _AckKind,
            )
            from tokenspeed.runtime.cache.transfer.lanes import CompletionQueue
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        executor = HostCacheExecutor.__new__(HostCacheExecutor)
        executor._ack_lock = threading.Lock()
        executor._completions = CompletionQueue()
        executor._ready_load_acks = []
        executor._prefetch_acks = []
        executor._backup_futures = []
        executor._backup_poll_failed = False
        executor._l3_workers = None
        executor.l3_store = Mock()
        executor.l3_store.backup.return_value = [False]
        finish = Mock()
        finish.query.return_value = True
        executor._completions.push(
            finish,
            _Ack(
                kind=_AckKind.WRITE_BACK,
                op_ids=[7],
                backup_pages=[(0, 1, "h0", 0)],
            ),
        )

        first = None
        failed = False
        try:
            first = executor.poll_results()
            self.assertEqual(first, [])
            failed = executor.consume_backup_poll_failure()
            deadline = time.monotonic() + 2
            while not failed and time.monotonic() < deadline:
                self.assertEqual(executor.poll_results(), [])
                failed = executor.consume_backup_poll_failure()
                if not failed:
                    time.sleep(0.01)
        finally:
            workers = executor._l3_workers
            if workers is not None:
                workers.shutdown(wait=True)
        self.assertTrue(failed)

    def test_load_with_every_row_owned_elsewhere_acks_without_h2d(self):
        try:
            from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
            from tokenspeed.runtime.cache.transfer.lanes import CompletionQueue
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        # A KVP rank that owns none of a load's rows copies nothing, arms no
        # layer fence and still acknowledges the op (the hooks' replica
        # intersection waits for the owners).
        executor = HostCacheExecutor.__new__(HostCacheExecutor)
        executor._ack_lock = threading.Lock()
        executor._ready_load_acks = []
        executor._prefetch_acks = []
        executor._load_poisoned = False
        executor._completions = CompletionQueue()
        executor._backup_futures = []
        executor._load_trackers = [(Mock(), 1)]
        executor.l3_store = None
        self.assertIsNone(
            executor._start_loading([9], [], prerequisite_stream=object())
        )
        executor._load_trackers[0][0].begin_load.assert_not_called()
        events = executor.poll_results()
        self.assertEqual(
            [(type(e).__name__, int(e.op_id)) for e in events],
            [("LoadBackDoneEvent", 9)],
        )

    def test_shutdown_persists_completed_d2h_before_closing_l3(self):
        try:
            import tokenspeed.runtime.cache.l2.executor as executor_module
            from tokenspeed.runtime.cache.l2.executor import (
                HostCacheExecutor,
                _Ack,
                _AckKind,
            )
            from tokenspeed.runtime.cache.transfer.lanes import CompletionQueue
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        executor = HostCacheExecutor.__new__(HostCacheExecutor)
        executor._ack_lock = threading.Lock()
        executor._completions = CompletionQueue()
        executor._completions.push(
            Mock(),
            _Ack(
                kind=_AckKind.WRITE_BACK,
                op_ids=[7],
                backup_pages=[(0, 1, "h0", 0)],
            ),
        )
        executor._backup_futures = []
        executor._l3_workers = None
        executor._l3_prefetch_lane = None
        executor._prefetch_jobs = {}
        executor._prefetch_acks = []
        executor.load_stream = Mock()
        executor.write_stream = Mock()
        executor.l3_store = Mock()
        executor.l3_store.backup.return_value = [True]
        default_stream = Mock()
        with patch.object(
            executor_module.device_module,
            "synchronize",
            side_effect=default_stream.synchronize,
        ):
            executor.shutdown()

        default_stream.synchronize.assert_called_once_with()
        executor.l3_store.backup.assert_called_once_with([(0, 1, "h0", 0)])
        executor.l3_store.close.assert_called_once_with()
        self.assertEqual(executor._completions._pending, [])


class CompactLayoutRoundTripTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch

            import tokenspeed.runtime.cache.l2.executor as executor_module
            from tokenspeed.runtime.cache.transfer.layout import (
                CacheField,
                CacheGroupLayout,
                CacheTransferLayout,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.executor_module = executor_module
        self.CacheField = CacheField
        self.CacheGroupLayout = CacheGroupLayout
        self.CacheTransferLayout = CacheTransferLayout

    def _make_executor(self, layout, *, draft_layout=None, io_backend):
        pool = _SyntheticPool(layout)
        draft_pool = (
            _SyntheticPool(draft_layout, pool.arena)
            if draft_layout is not None
            else None
        )
        with patch.object(self.executor_module, "_HOST_MEM_HEADROOM_BYTES", 0):
            executor = self.executor_module.HostCacheExecutor(
                pool,
                draft_pool=draft_pool,
                l2_tier=True,
                host_ratio=1.0,
                host_size_gb=0,
                snapshot_pool=NO_POOL,
                slot_state_exporters=None,
                io_backend=io_backend,
                attn_tp_rank=0,
                kvp_rank=0,
            )
        self.addCleanup(executor.shutdown)
        return executor, pool, draft_pool

    def _single_group_layout(self, buffer, *fields):
        return self.CacheTransferLayout(
            4,
            (self.CacheGroupLayout("full", 1, fields),),
            (buffer,),
            tuple((field.field_id,) for field in fields),
        )

    def test_kernel_executor_round_trip_restores_compact_layout_byte_exactly(self):
        torch = self.torch
        first = torch.full((128,), 0xCC, dtype=torch.uint8, device="cuda")
        second = torch.full((128,), 0xCC, dtype=torch.uint8, device="cuda")
        layout = self.CacheTransferLayout(
            num_lcm_blocks=4,
            groups=(
                self.CacheGroupLayout(
                    group_id="full",
                    cache_blocks_per_lcm_block=2,
                    fields=(
                        self.CacheField("layer.0.k", 0, 8, 8, 4),
                        self.CacheField("layer.0.v", 1, 16, 12, 6),
                    ),
                ),
                self.CacheGroupLayout(
                    group_id="state",
                    cache_blocks_per_lcm_block=1,
                    fields=(self.CacheField("layer.1.state", 0, 64, 10, 5),),
                ),
            ),
            buffers=(first, second),
            consumers=(("layer.0.k", "layer.0.v"), ("layer.1.state",)),
        )

        executor, pool, _ = self._make_executor(layout, io_backend="kernel")

        # Hand-derived Device ranges for blocks (full: 1, 4; state: 3).
        full_k_one = torch.tensor([0x11, 0x12, 0x13, 0x14], dtype=torch.uint8)
        full_v_one = torch.tensor(
            [0x21, 0x22, 0x23, 0x24, 0x25, 0x26], dtype=torch.uint8
        )
        full_k_four = torch.tensor([0x41, 0x42, 0x43, 0x44], dtype=torch.uint8)
        full_v_four = torch.tensor(
            [0x51, 0x52, 0x53, 0x54, 0x55, 0x56], dtype=torch.uint8
        )
        state_three = torch.tensor([0x71, 0x72, 0x73, 0x74, 0x75], dtype=torch.uint8)
        first[16:20].copy_(full_k_one)
        second[28:34].copy_(full_v_one)
        first[40:44].copy_(full_k_four)
        second[64:70].copy_(full_v_four)
        first[94:99].copy_(state_three)
        torch.cuda.synchronize()

        executor._start_writing(  # pylint: disable=protected-access
            [7],
            [(0, 1, 1), (0, 4, 4), (1, 3, 3)],
            backup_pages=[],
            lane=executor._pinned_write_lane,  # pylint: disable=protected-access
            prerequisite_stream=torch.cuda.current_stream(),
        )
        executor.write_stream.synchronize()
        write_results = executor.poll_results()
        self.assertEqual([int(event.op_id) for event in write_results], [7])

        # Destroy every Device byte so stale cache contents cannot make the
        # H2D half of the round trip pass accidentally.
        first.fill_(0xEE)
        second.fill_(0xEE)
        torch.cuda.synchronize()

        load_index = executor._start_loading(  # pylint: disable=protected-access
            [9],
            [(0, 2, 1), (0, 5, 4), (1, 4, 3)],
            prerequisite_stream=torch.cuda.current_stream(),
        )
        self.assertIsNotNone(load_index)
        pool.load_tracker.set_consumers(load_index)
        pool.load_tracker.wait_for_layer(0)
        pool.load_tracker.wait_for_layer(1)
        torch.cuda.synchronize()
        load_results = executor.poll_results()
        self.assertEqual([int(event.op_id) for event in load_results], [9])
        # Hand-derived destination ranges for blocks (full: 2, 5; state: 4).
        expected_first = torch.full((128,), 0xEE, dtype=torch.uint8)
        expected_second = torch.full((128,), 0xEE, dtype=torch.uint8)
        expected_first[24:28].copy_(full_k_one)
        expected_second[40:46].copy_(full_v_one)
        expected_first[48:52].copy_(full_k_four)
        expected_second[76:82].copy_(full_v_four)
        expected_first[104:109].copy_(state_three)
        self.assertTrue(torch.equal(first.cpu(), expected_first))
        self.assertTrue(torch.equal(second.cpu(), expected_second))

    def test_async_write_metadata_reuse_keeps_batches_distinct(self):
        torch = self.torch
        device = torch.zeros((128,), dtype=torch.uint8, device="cuda")
        layout = self._single_group_layout(
            device, self.CacheField("layer.0.k", 0, 8, 8, 4)
        )
        executor, pool, _ = self._make_executor(layout, io_backend="kernel")
        for generation in range(3):
            # Three back-to-back submissions, each with its own Device source
            # and Host block, with no caller synchronization between them: if
            # a later batch's block table overwrote an earlier one's before its
            # payload ran, the wrong pair would be copied. The sources are
            # distinct on purpose -- a pinned store's source is never rewritten
            # while its copy is in flight, so the test must not rewrite one
            # either.
            for block in range(1, 4):
                offset = 8 + block * 8
                device[offset : offset + 4].fill_(generation * 16 + block)
                executor._start_writing(
                    [block],
                    [(0, block, block)],
                    backup_pages=[],
                    lane=executor._pinned_write_lane,
                    prerequisite_stream=torch.cuda.current_stream(),
                )
            # The load below has no scheduler ACK to wait for, so order it
            # behind the copies the way a stream-ordered submission would.
            torch.cuda.current_stream().wait_stream(executor.write_stream)
            device.fill_(0xEE)
            load_index = executor._start_loading(
                [9],
                [(0, block, block) for block in range(1, 4)],
                prerequisite_stream=torch.cuda.current_stream(),
            )
            pool.load_tracker.set_consumers(load_index)
            pool.load_tracker.wait_for_layer(0)
            torch.cuda.current_stream().synchronize()
            for block in range(1, 4):
                offset = 8 + block * 8
                self.assertEqual(
                    device[offset : offset + 4].tolist(),
                    [generation * 16 + block] * 4,
                )
            self.assertEqual(
                sorted(int(event.op_id) for event in executor.poll_results()),
                [1, 2, 3, 9],
            )

    def test_real_transfer_restores_merged_owner_draft_subset_once(self):
        torch = self.torch
        device = torch.full((128,), 0xCC, dtype=torch.uint8, device="cuda")
        target_fields = (
            self.CacheField("layer.0.k", 0, 8, 8, 4),
            self.CacheField("layer.1.k", 0, 48, 8, 4),
        )
        target_layout = self._single_group_layout(device, *target_fields)
        draft_layout = self._single_group_layout(device, target_fields[1])
        executor, target_pool, draft_pool = self._make_executor(
            target_layout, draft_layout=draft_layout, io_backend="kernel"
        )

        device[16:20].fill_(0x11)
        device[56:60].fill_(0x12)
        torch.cuda.synchronize()
        executor._start_writing(  # pylint: disable=protected-access
            [7],
            [(0, 1, 1)],
            backup_pages=[],
            lane=executor._pinned_write_lane,
            prerequisite_stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        self.assertEqual([int(event.op_id) for event in executor.poll_results()], [7])

        device.fill_(0xEE)
        torch.cuda.synchronize()
        load_index = executor._start_loading(  # pylint: disable=protected-access
            [9],
            [(0, 2, 1)],
            prerequisite_stream=torch.cuda.current_stream(),
        )
        self.assertIsNotNone(load_index)
        target_pool.load_tracker.set_consumers(load_index)
        draft_pool.load_tracker.set_consumers(load_index)
        target_pool.load_tracker.wait_for_layer(0)
        target_pool.load_tracker.wait_for_layer(1)
        draft_pool.load_tracker.wait_for_layer(0)
        torch.cuda.synchronize()
        self.assertEqual([int(event.op_id) for event in executor.poll_results()], [9])
        self.assertEqual(device[24:28].tolist(), [0x11] * 4)
        self.assertEqual(device[64:68].tolist(), [0x12] * 4)


if __name__ == "__main__":
    unittest.main()
