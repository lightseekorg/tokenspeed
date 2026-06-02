"""Phase 2: WeightTransferManager state machine, get_world_size, pause/resume."""

import asyncio
import types

import pytest

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.engine.io_struct import (
    InitWeightsUpdateGroupReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from tokenspeed.runtime.engine.weight_transfer.config import WeightTransferConfig
from tokenspeed.runtime.engine.weight_transfer.manager import (
    WeightTransferManager,
    WeightTransferStateError,
)
from tokenspeed.runtime.utils.env import envs


class FakeAsyncLLM:
    """Minimal AsyncLLM stand-in implementing the surface the manager uses."""

    def __init__(self, *, world_size=4, tp=4, dp=1, init_ok=True, update_ok=True):
        self.server_args = types.SimpleNamespace(
            mapping=Mapping(
                rank=0, world_size=world_size, attn_tp_size=tp, attn_dp_size=dp
            )
        )
        self._wt_admit = asyncio.Event()
        self._wt_admit.set()
        self.rid_to_state = {}
        self.calls = []
        self._init_ok = init_ok
        self._update_ok = update_ok

    async def init_weights_update_group(self, obj):
        assert isinstance(obj, InitWeightsUpdateGroupReqInput)
        self.calls.append(("init", obj))
        return (self._init_ok, "ok" if self._init_ok else "boom")

    async def update_weights_from_distributed(self, obj):
        assert isinstance(obj, UpdateWeightsFromDistributedReqInput)
        self.calls.append(("update", obj))
        return (self._update_ok, "ok" if self._update_ok else "boom")

    async def flush_cache(self):
        self.calls.append(("flush",))

    # Admission gate (mirrors AsyncLLM).
    def weight_transfer_admission_paused(self):
        return not self._wt_admit.is_set()

    def weight_transfer_block_admission(self):
        self._wt_admit.clear()

    def weight_transfer_allow_admission(self):
        self._wt_admit.set()

    def weight_transfer_abort_inflight(self):
        self.calls.append(("abort", list(self.rid_to_state)))
        self.rid_to_state.clear()

    async def weight_transfer_drain_inflight(self):
        self.calls.append(("drain",))


def _mgr(backend="nccl", **kw):
    a = FakeAsyncLLM(**kw)
    return a, WeightTransferManager(a, WeightTransferConfig(backend=backend))


def _nccl_update(**over):
    d = {"names": ["w"], "dtype_names": ["bfloat16"], "shapes": [[2, 2]]}
    d.update(over)
    return d


class TestGetWorldSize:
    @pytest.mark.parametrize(
        "world_size,tp,dp,inc_dp,expected",
        [
            (4, 4, 1, True, 4),
            (4, 4, 1, False, 4),
            (8, 2, 2, True, 8),  # TP=2 * DP=2 -> 8 across dp
            (8, 2, 2, False, 4),  # one replica -> TP*CP = 4
            (1, 1, 1, True, 1),
            (1, 1, 1, False, 1),
        ],
    )
    def test_matrix(self, world_size, tp, dp, inc_dp, expected):
        _, mgr = _mgr(world_size=world_size, tp=tp, dp=dp)
        assert mgr.get_world_size(include_dp=inc_dp) == expected


class TestLifecycleOrdering:
    def test_start_before_init_raises(self):
        _, mgr = _mgr()
        with pytest.raises(WeightTransferStateError):
            asyncio.run(mgr.start_update())

    def test_update_before_start_raises(self):
        _, mgr = _mgr()
        with pytest.raises(WeightTransferStateError):
            asyncio.run(mgr.update(_nccl_update()))

    def test_finish_before_start_raises(self):
        _, mgr = _mgr()
        with pytest.raises(WeightTransferStateError):
            asyncio.run(mgr.finish_update())

    def test_double_start_raises(self):
        a, mgr = _mgr()

        async def body():
            await mgr.init_engine(_nccl_init())
            await mgr.start_update()
            await mgr.start_update()

        with pytest.raises(WeightTransferStateError):
            asyncio.run(body())

    def test_happy_path(self):
        a, mgr = _mgr()

        async def body():
            await mgr.init_engine(_nccl_init())
            await mgr.start_update(is_checkpoint_format=False)
            await mgr.update(_nccl_update())
            await mgr.finish_update()
            # a second cycle is allowed after finishing
            await mgr.start_update()
            await mgr.finish_update()

        asyncio.run(body())
        kinds = [c[0] for c in a.calls]
        assert kinds == ["init", "update"]

    def test_init_failure_propagates(self):
        _, mgr = _mgr(init_ok=False)
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(mgr.init_engine(_nccl_init()))


def _nccl_init(**over):
    d = {
        "master_address": "127.0.0.1",
        "master_port": 1234,
        "rank_offset": 1,
        "world_size": 2,
    }
    d.update(over)
    return d


class TestNcclParsing:
    def test_init_builds_struct(self):
        a, mgr = _mgr()
        asyncio.run(mgr.init_engine(_nccl_init(group_name="grp")))
        obj = a.calls[0][1]
        assert obj.master_address == "127.0.0.1" and obj.master_port == 1234
        assert obj.rank_offset == 1 and obj.world_size == 2
        assert obj.group_name == "grp" and obj.backend == "nccl"

    def test_init_missing_key_raises(self):
        _, mgr = _mgr()
        bad = _nccl_init()
        del bad["world_size"]
        with pytest.raises(ValueError, match="missing required key"):
            asyncio.run(mgr.init_engine(bad))

    def test_init_unknown_key_raises(self):
        _, mgr = _mgr()
        with pytest.raises(ValueError, match="unknown key"):
            asyncio.run(mgr.init_engine(_nccl_init(bogus=1)))

    def test_update_defaults_and_packed(self):
        a, mgr = _mgr()

        async def body():
            await mgr.init_engine(_nccl_init())
            await mgr.start_update()
            await mgr.update(_nccl_update(packed=True, packed_num_buffers=3))

        asyncio.run(body())
        obj = [c for c in a.calls if c[0] == "update"][0][1]
        assert obj.packed is True and obj.packed_num_buffers == 3
        assert obj.flush_cache is True and obj.group_name == "weight_update_group"

    def test_update_length_mismatch_raises(self):
        a, mgr = _mgr()

        async def body():
            await mgr.init_engine(_nccl_init())
            await mgr.start_update()
            await mgr.update(_nccl_update(shapes=[[2, 2], [3, 3]]))

        with pytest.raises(ValueError, match="equal length"):
            asyncio.run(body())

    def test_update_kind_dense_accepted(self):
        a, mgr = _mgr()

        async def body():
            await mgr.init_engine(_nccl_init())
            await mgr.start_update()
            await mgr.update(_nccl_update(update_kind="dense"))

        asyncio.run(body())
        assert any(c[0] == "update" for c in a.calls)

    def test_update_kind_sparse_rejected(self):
        a, mgr = _mgr()

        async def body():
            await mgr.init_engine(_nccl_init())
            await mgr.start_update()
            await mgr.update(_nccl_update(update_kind="sparse_flat"))

        with pytest.raises(ValueError, match="update_kind"):
            asyncio.run(body())


class TestIpcParsing:
    def test_init_noop(self):
        _, mgr = _mgr(backend="ipc")
        asyncio.run(mgr.init_engine({}))

    def test_init_rejects_keys(self):
        _, mgr = _mgr(backend="ipc")
        with pytest.raises(ValueError):
            asyncio.run(mgr.init_engine({"foo": 1}))

    def _started(self):
        a, mgr = _mgr(backend="ipc")

        async def start():
            await mgr.init_engine({})
            await mgr.start_update()

        asyncio.run(start())
        return a, mgr

    def test_update_requires_handles(self):
        _, mgr = self._started()
        with pytest.raises(ValueError, match="must be provided"):
            asyncio.run(mgr.update(_nccl_update()))

    def test_update_both_handles_raises(self):
        _, mgr = self._started()
        upd = _nccl_update(ipc_handles=[{"u": ()}], ipc_handles_pickled="x")
        with pytest.raises(ValueError, match="Cannot specify both"):
            asyncio.run(mgr.update(upd))

    def test_update_kind_sparse_rejected(self):
        _, mgr = self._started()
        upd = _nccl_update(ipc_handles=[{"u": ()}], update_kind="sparse_flat")
        with pytest.raises(ValueError, match="update_kind"):
            asyncio.run(mgr.update(upd))

    def test_pickled_requires_insecure_optin(self):
        _, mgr = self._started()
        upd = _nccl_update(ipc_handles_pickled="x")
        with pytest.raises(ValueError, match="INSECURE_SERIALIZATION"):
            asyncio.run(mgr.update(upd))

    def test_packed_requires_tensor_sizes(self):
        _, mgr = self._started()
        upd = _nccl_update(ipc_handles=[{"u": ()}], packed=True)
        with pytest.raises(ValueError, match="tensor_sizes"):
            asyncio.run(mgr.update(upd))

    def test_load_not_implemented(self):
        _, mgr = self._started()
        upd = _nccl_update(ipc_handles=[{"u": ()}])
        with pytest.raises(NotImplementedError):
            asyncio.run(mgr.update(upd))

    def test_pickled_with_optin_reaches_load(self):
        _, mgr = self._started()
        upd = _nccl_update(ipc_handles_pickled="x")
        with envs.TOKENSPEED_ALLOW_INSECURE_SERIALIZATION.override(True):
            # gate passes; load is deferred
            with pytest.raises(NotImplementedError):
                asyncio.run(mgr.update(upd))


class TestPauseResume:
    def test_pause_resume_gate(self):
        a, mgr = _mgr()

        async def body():
            assert mgr.is_paused() is False
            await mgr.pause(mode="keep")
            assert mgr.is_paused() is True
            await mgr.resume()
            assert mgr.is_paused() is False

        asyncio.run(body())

    def test_wait_drains_and_flushes(self):
        a, mgr = _mgr()
        asyncio.run(mgr.pause(mode="wait", clear_cache=True))
        kinds = [c[0] for c in a.calls]
        assert "drain" in kinds and "flush" in kinds

    def test_abort_aborts_inflight(self):
        a, mgr = _mgr()
        a.rid_to_state = {"r1": 1, "r2": 2}
        asyncio.run(mgr.pause(mode="abort"))
        assert a.rid_to_state == {}
        assert ("abort", ["r1", "r2"]) in a.calls

    def test_keep_preserves_inflight_no_flush(self):
        a, mgr = _mgr()
        a.rid_to_state = {"r1": 1}
        asyncio.run(mgr.pause(mode="keep", clear_cache=True))
        assert a.rid_to_state == {"r1": 1}
        assert all(c[0] != "flush" for c in a.calls)
        assert all(c[0] != "drain" for c in a.calls)

    def test_invalid_mode_raises(self):
        _, mgr = _mgr()
        with pytest.raises(ValueError, match="Invalid pause mode"):
            asyncio.run(mgr.pause(mode="bogus"))
