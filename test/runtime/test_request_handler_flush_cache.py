import unittest
from unittest import mock

from tokenspeed.runtime.engine.io_struct import FlushCacheReqInput
from tokenspeed.runtime.engine.request_handler import RequestHandler


class TestRequestHandlerFlushCache(unittest.TestCase):
    def test_returns_scheduler_clear_result(self):
        for success in (True, False):
            with self.subTest(success=success):
                handler = RequestHandler.__new__(RequestHandler)
                handler.send_func = mock.Mock()
                handler.clear_l1_cache_fn = mock.Mock(return_value=success)

                handler.process_requests([FlushCacheReqInput()])

                handler.clear_l1_cache_fn.assert_called_once_with()
                output = handler.send_func.send_pyobj.call_args.args[0]
                self.assertEqual(output.success, success)


if __name__ == "__main__":
    unittest.main()
