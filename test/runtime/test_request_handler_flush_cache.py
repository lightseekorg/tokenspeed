import unittest
from unittest import mock

from tokenspeed.runtime.engine.io_struct import (
    FlushCacheReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
)
from tokenspeed.runtime.engine.request_handler import RequestHandler


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
        handler.server_args = mock.Mock(weight_version="v1")
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


if __name__ == "__main__":
    unittest.main()
