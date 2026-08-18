import unittest
from unittest import mock

from tokenspeed.runtime.engine.io_struct import UpdateWeightsFromDistributedReqInput
from tokenspeed.runtime.engine.request_handler import RequestHandler


class TestRequestHandlerWeightUpdate(unittest.TestCase):
    def test_deferred_state_settles_before_the_weights_load(self):
        """A weight update can share a socket drain with a pause(keep),
        ahead of the loop's pause-fence flush; deferred state must settle
        before the new weights load."""
        order = []
        handler = RequestHandler.__new__(RequestHandler)
        handler.send_func = mock.Mock()
        handler.settle_deferred_fn = mock.Mock(
            side_effect=lambda: order.append("settle")
        )
        handler.model_runner = mock.Mock()

        def load(req):
            order.append("load")
            return True, "ok"

        handler.model_runner.update_weights_from_distributed = mock.Mock(
            side_effect=load
        )

        handler.process_requests(
            [UpdateWeightsFromDistributedReqInput(names=[], dtype_names=[], shapes=[])]
        )

        self.assertEqual(order, ["settle", "load"])
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertTrue(output.success)

    def test_weight_update_survives_a_handler_without_the_hook(self):
        """Engines that never defer state wire no callback; the update must
        proceed unchanged."""
        handler = RequestHandler.__new__(RequestHandler)
        handler.send_func = mock.Mock()
        handler.settle_deferred_fn = None
        handler.model_runner = mock.Mock()
        handler.model_runner.update_weights_from_distributed = mock.Mock(
            return_value=(True, "ok")
        )

        handler.process_requests(
            [UpdateWeightsFromDistributedReqInput(names=[], dtype_names=[], shapes=[])]
        )

        handler.model_runner.update_weights_from_distributed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
