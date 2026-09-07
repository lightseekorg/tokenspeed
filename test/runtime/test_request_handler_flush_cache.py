import inspect
import unittest
from unittest import mock

import torch

from tokenspeed.runtime.engine.io_struct import (
    FlushCacheReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
)
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.entrypoints.engine import Engine


class TestRequestHandlerFlushCache(unittest.TestCase):
    def _handler(self, can_clear, clear_result):
        handler = RequestHandler.__new__(RequestHandler)
        handler.send_func = mock.Mock()
        handler.can_clear_cache_fn = mock.Mock(return_value=can_clear)
        handler.clear_cache_fn = mock.Mock(return_value=clear_result)
        handler.clear_l1_cache_fn = mock.Mock(return_value=not clear_result)
        handler._replica_tp_size = 1
        handler._replica_tp_cpu_group = None
        handler.attn_cp_size = 1
        handler.attn_cp_cpu_group = None
        handler.pp_size = 1
        handler.pp_cpu_group = None
        handler._replica_decision_buf = torch.zeros(1, dtype=torch.int32)
        return handler

    def test_returns_scheduler_clear_result(self):
        for success in (True, False):
            with self.subTest(success=success):
                handler = self._handler(can_clear=True, clear_result=success)

                handler.process_requests([FlushCacheReqInput()])

                handler.can_clear_cache_fn.assert_called_once_with()
                handler.clear_cache_fn.assert_called_once_with()
                handler.clear_l1_cache_fn.assert_not_called()
                output = handler.send_func.send_pyobj.call_args.args[0]
                self.assertEqual(output.success, success)

    def test_failed_preflight_does_not_clear(self):
        groups_seen = []

        def fake_all_reduce(buf, op=None, group=None):
            del buf, op
            groups_seen.append(group)

        handler = self._handler(can_clear=False, clear_result=True)
        handler._replica_tp_size = 2
        handler._replica_tp_cpu_group = "tp"

        with mock.patch.object(torch.distributed, "all_reduce", fake_all_reduce):
            handler.process_requests([FlushCacheReqInput()])

        self.assertEqual(groups_seen, ["tp"])
        handler.can_clear_cache_fn.assert_called_once_with()
        handler.clear_cache_fn.assert_not_called()
        handler.clear_l1_cache_fn.assert_not_called()
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertFalse(output.success)

    def test_min_reduces_tp_then_cp_then_pp_before_clear(self):
        groups_seen = []

        def fake_all_reduce(buf, op=None, group=None):
            del op
            groups_seen.append(group)
            buf.fill_(0)

        handler = self._handler(can_clear=True, clear_result=True)
        handler._replica_tp_size = 2
        handler._replica_tp_cpu_group = "tp"
        handler.attn_cp_size = 2
        handler.attn_cp_cpu_group = "cp"
        handler.pp_size = 2
        handler.pp_cpu_group = "pp"

        with mock.patch.object(torch.distributed, "all_reduce", fake_all_reduce):
            handler.process_requests([FlushCacheReqInput()])

        self.assertEqual(groups_seen, ["tp", "cp", "pp"])
        handler.can_clear_cache_fn.assert_called_once_with()
        handler.clear_cache_fn.assert_not_called()
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertFalse(output.success)

    def test_agreed_preflight_clears(self):
        groups_seen = []

        def fake_all_reduce(buf, op=None, group=None):
            del buf, op
            groups_seen.append(group)

        handler = self._handler(can_clear=True, clear_result=True)
        handler._replica_tp_size = 2
        handler._replica_tp_cpu_group = "tp"

        with mock.patch.object(torch.distributed, "all_reduce", fake_all_reduce):
            handler.process_requests([FlushCacheReqInput()])

        self.assertEqual(groups_seen, ["tp"])
        handler.clear_cache_fn.assert_called_once_with()
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertTrue(output.success)


class TestRequestHandlerL3WeightVersion(unittest.TestCase):
    def _handler(self):
        handler = RequestHandler.__new__(RequestHandler)
        handler.send_func = mock.Mock()
        handler.server_args = mock.Mock(
            weight_version="v1", kvstore_storage_backend=None
        )
        handler._device = mock.Mock()
        handler._replica_tp_size = 1
        handler._replica_tp_cpu_group = None
        handler.attn_cp_size = 1
        handler.attn_cp_cpu_group = None
        handler.pp_size = 1
        handler.pp_cpu_group = None
        handler._replica_decision_buf = torch.zeros(1, dtype=torch.int32)
        handler.can_clear_cache_fn = mock.Mock(return_value=True)
        return handler

    def test_successful_update_flushes_then_rebuilds_l3_prefix(self):
        handler = self._handler()
        order = []
        handler.can_clear_cache_fn = mock.Mock(
            side_effect=lambda: order.append("preflight") or True
        )
        handler.clear_cache_fn = mock.Mock(
            side_effect=lambda: order.append("flush") or True
        )

        def _update_weights(req):
            del req
            order.append("gpu")
            return True, "ok"

        handler._device.update_weights.side_effect = _update_weights
        handler._device.set_l3_weight_version.side_effect = (
            lambda version: order.append(("prefix", version))
        )
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=True,
            weight_version="v2",
        )

        handler.process_requests([req])

        self.assertEqual(order, ["preflight", "flush", "gpu", ("prefix", "v2")])
        self.assertEqual(handler.server_args.weight_version, "v2")
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertIsInstance(output, UpdateWeightsFromDistributedReqOutput)
        self.assertTrue(output.success)

    def test_failed_update_keeps_the_old_l3_prefix(self):
        handler = self._handler()
        handler.clear_cache_fn = mock.Mock(return_value=True)
        handler._device.update_weights.return_value = (False, "nccl failed")
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=True,
            weight_version="v2",
        )

        handler.process_requests([req])

        handler.clear_cache_fn.assert_called_once_with()
        handler._device.set_l3_weight_version.assert_not_called()
        self.assertEqual(handler.server_args.weight_version, "v1")

    def test_skips_flush_when_more_updates_are_coming(self):
        handler = self._handler()
        handler.clear_cache_fn = mock.Mock(return_value=True)
        handler._device.update_weights.return_value = (True, "ok")
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=False,
            weight_version="v2",
        )

        handler.process_requests([req])

        handler.clear_cache_fn.assert_not_called()
        handler._device.set_l3_weight_version.assert_called_once_with("v2")
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertTrue(output.success)

    def test_l3_rejects_version_switch_without_flush(self):
        handler = self._handler()
        handler.server_args.kvstore_storage_backend = "memory"
        handler.clear_cache_fn = mock.Mock(return_value=True)
        handler._device.update_weights.return_value = (True, "ok")
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=False,
            weight_version="v2",
        )

        handler.process_requests([req])

        handler._device.update_weights.assert_not_called()
        handler.clear_cache_fn.assert_not_called()
        handler._device.set_l3_weight_version.assert_not_called()
        self.assertEqual(handler.server_args.weight_version, "v1")
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertFalse(output.success)
        self.assertIn("cannot change without flush_cache", output.message)

    def test_l3_intermediate_update_keeps_namespace_when_version_omitted(self):
        handler = self._handler()
        handler.server_args.kvstore_storage_backend = "memory"
        handler.clear_cache_fn = mock.Mock(return_value=True)
        handler._device.update_weights.return_value = (True, "ok")
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=False,
            weight_version=None,
        )

        handler.process_requests([req])

        handler._device.update_weights.assert_called_once_with(req)
        handler.clear_cache_fn.assert_not_called()
        handler._device.set_l3_weight_version.assert_not_called()
        self.assertEqual(handler.server_args.weight_version, "v1")
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertTrue(output.success)

    def test_rejected_flush_does_not_switch_l3_prefix(self):
        handler = self._handler()
        handler.can_clear_cache_fn = mock.Mock(return_value=False)
        handler.clear_cache_fn = mock.Mock(return_value=True)
        handler._device.update_weights.return_value = (True, "ok")
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=True,
            weight_version="v2",
        )

        handler.process_requests([req])

        handler.can_clear_cache_fn.assert_called_once_with()
        handler.clear_cache_fn.assert_not_called()
        handler._device.update_weights.assert_not_called()
        handler._device.set_l3_weight_version.assert_not_called()
        self.assertEqual(handler.server_args.weight_version, "v1")
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertIsInstance(output, UpdateWeightsFromDistributedReqOutput)
        self.assertFalse(output.success)
        self.assertIn("cache flush failed", output.message)

    def test_missing_flush_handler_does_not_switch_l3_prefix(self):
        handler = self._handler()
        handler.clear_cache_fn = None
        handler._device.update_weights.return_value = (True, "ok")
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=True,
            weight_version="v2",
        )

        handler.process_requests([req])

        handler.can_clear_cache_fn.assert_called_once_with()
        handler._device.update_weights.assert_not_called()
        handler._device.set_l3_weight_version.assert_not_called()
        self.assertEqual(handler.server_args.weight_version, "v1")
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertFalse(output.success)
        self.assertIn("cache flush failed", output.message)

    def test_can_clear_cache_fn_is_required(self):
        param = inspect.signature(RequestHandler.__init__).parameters[
            "can_clear_cache_fn"
        ]
        self.assertIs(param.default, inspect.Parameter.empty)

    def test_l3_flush_without_version_is_rejected(self):
        handler = self._handler()
        handler.server_args.kvstore_storage_backend = "memory"
        handler.clear_cache_fn = mock.Mock(return_value=True)
        handler._device.update_weights.return_value = (True, "ok")
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=True,
            weight_version=None,
        )

        handler.process_requests([req])

        handler._device.update_weights.assert_not_called()
        handler.clear_cache_fn.assert_not_called()
        handler._device.set_l3_weight_version.assert_not_called()
        self.assertEqual(handler.server_args.weight_version, "v1")
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertFalse(output.success)
        self.assertIn("require weight_version", output.message)

    def test_flush_min_reduces_tp_then_cp_then_pp(self):
        groups_seen = []

        def fake_all_reduce(buf, op=None, group=None):
            del op
            groups_seen.append(group)
            buf.fill_(0)

        handler = self._handler()
        handler._replica_tp_size = 2
        handler._replica_tp_cpu_group = "tp"
        handler.attn_cp_size = 2
        handler.attn_cp_cpu_group = "cp"
        handler.pp_size = 2
        handler.pp_cpu_group = "pp"
        handler.clear_cache_fn = mock.Mock(return_value=True)
        handler._device.update_weights.return_value = (True, "ok")
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=True,
            weight_version="v2",
        )

        with mock.patch.object(torch.distributed, "all_reduce", fake_all_reduce):
            handler.process_requests([req])

        self.assertEqual(groups_seen, ["tp", "cp", "pp"])
        handler.can_clear_cache_fn.assert_called_once_with()
        handler.clear_cache_fn.assert_not_called()
        handler._device.update_weights.assert_not_called()
        handler._device.set_l3_weight_version.assert_not_called()
        self.assertEqual(handler.server_args.weight_version, "v1")
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertFalse(output.success)
        self.assertIn("cache flush failed", output.message)

    def test_enable_cp_flush_min_uses_cp_group_when_tp_is_one(self):
        groups_seen = []

        def fake_all_reduce(buf, op=None, group=None):
            del buf, op
            groups_seen.append(group)

        handler = self._handler()
        handler._replica_tp_size = 1
        handler._replica_tp_cpu_group = "tp"
        handler.attn_cp_size = 4
        handler.attn_cp_cpu_group = "cp"
        handler.clear_cache_fn = mock.Mock(return_value=True)
        handler._device.update_weights.return_value = (True, "ok")
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=True,
            weight_version="v2",
        )

        with mock.patch.object(torch.distributed, "all_reduce", fake_all_reduce):
            handler.process_requests([req])

        self.assertEqual(groups_seen, ["cp"])
        handler._device.update_weights.assert_called_once_with(req)
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertTrue(output.success)

    def test_failed_flush_still_min_reduces_before_skipping_nccl(self):
        groups_seen = []

        def fake_all_reduce(buf, op=None, group=None):
            del buf, op
            groups_seen.append(group)

        handler = self._handler()
        handler._replica_tp_size = 2
        handler._replica_tp_cpu_group = "tp"
        handler.can_clear_cache_fn = mock.Mock(return_value=False)
        handler.clear_cache_fn = mock.Mock(return_value=True)
        handler._device.update_weights.return_value = (True, "ok")
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=True,
            weight_version="v2",
        )

        with mock.patch.object(torch.distributed, "all_reduce", fake_all_reduce):
            handler.process_requests([req])

        self.assertEqual(groups_seen, ["tp"])
        handler.can_clear_cache_fn.assert_called_once_with()
        handler.clear_cache_fn.assert_not_called()
        handler._device.update_weights.assert_not_called()
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertFalse(output.success)
        self.assertIn("cache flush failed", output.message)

    def test_without_l3_omitted_version_keeps_the_startup_namespace(self):
        handler = self._handler()
        handler.clear_cache_fn = mock.Mock(return_value=True)
        handler._device.update_weights.return_value = (True, "ok")
        req = UpdateWeightsFromDistributedReqInput(
            names=["w"],
            dtype_names=["float16"],
            shapes=[[1]],
            flush_cache=True,
            weight_version=None,
        )

        handler.process_requests([req])

        handler._device.set_l3_weight_version.assert_not_called()
        self.assertEqual(handler.server_args.weight_version, "v1")


class TestEngineStampsL3Version(unittest.TestCase):
    def _engine(self, *, storage_backend):
        engine = Engine.__new__(Engine)
        engine.server_args = mock.Mock(
            weight_version="v1", kvstore_storage_backend=storage_backend
        )
        engine.tokenizer_manager = mock.Mock()
        engine.llm = mock.Mock()
        return engine

    def test_l3_flush_without_version_does_not_send(self):
        engine = self._engine(storage_backend="memory")
        engine.llm.run.return_value = (True, "ok")

        success, message = engine.update_weights_from_distributed(
            names=["w"],
            dtypes=["float16"],
            shapes=[[1]],
            group_name="weight_update_group",
            flush_cache=True,
            weight_version=None,
        )

        self.assertFalse(success)
        self.assertIn("require weight_version", message)
        self.assertEqual(engine.server_args.weight_version, "v1")
        engine.llm.run.assert_not_called()

    def test_successful_update_persists_explicit_namespace(self):
        engine = self._engine(storage_backend="memory")
        engine.llm.run.return_value = (True, "ok")

        engine.update_weights_from_distributed(
            names=["w"],
            dtypes=["float16"],
            shapes=[[1]],
            group_name="weight_update_group",
            flush_cache=True,
            weight_version="v2",
        )

        self.assertEqual(engine.server_args.weight_version, "v2")
        req = engine.tokenizer_manager.update_weights_from_distributed.call_args.args[0]
        self.assertEqual(req.weight_version, "v2")

    def test_failed_update_does_not_persist_explicit_namespace(self):
        engine = self._engine(storage_backend="memory")
        engine.llm.run.return_value = (False, "nccl failed")

        engine.update_weights_from_distributed(
            names=["w"],
            dtypes=["float16"],
            shapes=[[1]],
            group_name="weight_update_group",
            flush_cache=True,
            weight_version="v2",
        )

        self.assertEqual(engine.server_args.weight_version, "v1")

    def test_without_l3_omitted_version_does_not_stamp(self):
        engine = self._engine(storage_backend=None)
        engine.llm.run.return_value = (True, "ok")

        engine.update_weights_from_distributed(
            names=["w"],
            dtypes=["float16"],
            shapes=[[1]],
            group_name="weight_update_group",
            flush_cache=True,
            weight_version=None,
        )

        self.assertEqual(engine.server_args.weight_version, "v1")

    def test_weight_version_has_no_default(self):
        param = inspect.signature(Engine.update_weights_from_distributed).parameters[
            "weight_version"
        ]
        self.assertIs(param.default, inspect.Parameter.empty)
        engine = self._engine(storage_backend="memory")
        with self.assertRaises(TypeError):
            engine.update_weights_from_distributed(
                names=["w"],
                dtypes=["float16"],
                shapes=[[1]],
                group_name="weight_update_group",
                flush_cache=True,
            )
        with self.assertRaises(TypeError):
            UpdateWeightsFromDistributedReqInput(
                names=["w"],
                dtype_names=["float16"],
                shapes=[[1]],
            )


if __name__ == "__main__":
    unittest.main()
