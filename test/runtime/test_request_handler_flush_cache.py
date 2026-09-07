import unittest
from unittest import mock

from tokenspeed.runtime.engine.io_struct import (
    FlushCacheReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
)
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.entrypoints.engine import Engine


class TestRequestHandlerFlushCache(unittest.TestCase):
    def test_returns_scheduler_clear_result(self):
        for success in (True, False):
            with self.subTest(success=success):
                handler = RequestHandler.__new__(RequestHandler)
                handler.send_func = mock.Mock()
                handler.clear_cache_fn = mock.Mock(return_value=success)
                handler.clear_l1_cache_fn = mock.Mock(return_value=not success)

                handler.process_requests([FlushCacheReqInput()])

                handler.clear_cache_fn.assert_called_once_with()
                handler.clear_l1_cache_fn.assert_not_called()
                output = handler.send_func.send_pyobj.call_args.args[0]
                self.assertEqual(output.success, success)


class TestRequestHandlerL3WeightVersion(unittest.TestCase):
    def _handler(self):
        handler = RequestHandler.__new__(RequestHandler)
        handler.send_func = mock.Mock()
        handler.server_args = mock.Mock(
            weight_version="v1", kvstore_storage_backend=None
        )
        handler._device = mock.Mock()
        return handler

    def test_successful_update_flushes_then_rebuilds_l3_prefix(self):
        handler = self._handler()
        order = []
        handler.clear_cache_fn = mock.Mock(
            side_effect=lambda: order.append("flush") or True
        )
        handler._device.update_weights.return_value = (True, "ok")
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

        self.assertEqual(order, ["flush", ("prefix", "v2")])
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

        handler.clear_cache_fn.assert_not_called()
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
        handler.clear_cache_fn = mock.Mock(return_value=False)
        handler._device.update_weights.return_value = (True, "ok")
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

        handler._device.set_l3_weight_version.assert_not_called()
        self.assertEqual(handler.server_args.weight_version, "v1")
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertFalse(output.success)
        self.assertIn("cache flush failed", output.message)

    def test_l3_flush_without_version_derives_a_new_namespace(self):
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

        handler._device.set_l3_weight_version.assert_called_once_with("v1-u1")
        self.assertEqual(handler.server_args.weight_version, "v1-u1")
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertTrue(output.success)

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


class TestEngineStampsDerivedL3Version(unittest.TestCase):
    def _engine(self, *, storage_backend):
        engine = Engine.__new__(Engine)
        engine.server_args = mock.Mock(
            weight_version="v1", kvstore_storage_backend=storage_backend
        )
        engine.tokenizer_manager = mock.Mock()
        engine.llm = mock.Mock()
        return engine

    def test_successful_update_persists_derived_namespace(self):
        engine = self._engine(storage_backend="memory")
        engine.llm.run.return_value = (True, "ok")

        engine.update_weights_from_distributed(
            names=["w"],
            dtypes=["float16"],
            shapes=[[1]],
            group_name="weight_update_group",
            flush_cache=True,
            weight_version=None,
        )

        self.assertEqual(engine.server_args.weight_version, "v1-u1")

        engine.update_weights_from_distributed(
            names=["w"],
            dtypes=["float16"],
            shapes=[[1]],
            group_name="weight_update_group",
            flush_cache=True,
            weight_version=None,
        )

        self.assertEqual(engine.server_args.weight_version, "v1-u2")
        req = engine.tokenizer_manager.update_weights_from_distributed.call_args.args[0]
        self.assertEqual(req.weight_version, "v1-u2")

    def test_failed_update_does_not_persist_derived_namespace(self):
        engine = self._engine(storage_backend="memory")
        engine.llm.run.return_value = (False, "nccl failed")

        engine.update_weights_from_distributed(
            names=["w"],
            dtypes=["float16"],
            shapes=[[1]],
            group_name="weight_update_group",
            flush_cache=True,
            weight_version=None,
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


if __name__ == "__main__":
    unittest.main()
