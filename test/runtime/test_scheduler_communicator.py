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

import asyncio
import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.engine.scheduler_communicator import _Communicator


class _RecordingSender:
    def __init__(self):
        self.sent: list[object] = []

    def send_pyobj(self, obj: object):
        self.sent.append(obj)


class TestWatchingCommunicator(unittest.IsolatedAsyncioTestCase):
    async def test_overlapping_watchers_share_one_complete_result(self):
        sender = _RecordingSender()
        communicator = _Communicator(sender, fan_out=2, mode="watching")

        first = asyncio.create_task(communicator("request"))
        second = asyncio.create_task(communicator("request"))
        await asyncio.sleep(0)
        self.assertEqual(sender.sent, ["request"])

        communicator.handle_recv({"rank": 0})
        self.assertFalse(first.done())
        self.assertFalse(second.done())
        communicator.handle_recv({"rank": 1})

        first_result, second_result = await asyncio.gather(first, second)
        self.assertEqual(first_result, [{"rank": 0}, {"rank": 1}])
        self.assertEqual(second_result, first_result)
        self.assertIsNot(first_result, second_result)
        self.assertIsNot(first_result[0], second_result[0])

    async def test_cancelled_watcher_does_not_cancel_other_watcher_or_reuse(self):
        sender = _RecordingSender()
        communicator = _Communicator(sender, fan_out=2, mode="watching")

        cancelled = asyncio.create_task(communicator("first request"))
        survivor = asyncio.create_task(communicator("first request"))
        await asyncio.sleep(0)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled

        communicator.handle_recv({"rank": 0})
        communicator.handle_recv({"rank": 1})
        self.assertEqual(
            await survivor,
            [{"rank": 0}, {"rank": 1}],
        )

        reused = asyncio.create_task(communicator("second request"))
        await asyncio.sleep(0)
        self.assertEqual(sender.sent, ["first request", "second request"])
        communicator.handle_recv({"rank": 0, "flight": 2})
        communicator.handle_recv({"rank": 1, "flight": 2})
        self.assertEqual(
            await reused,
            [{"rank": 0, "flight": 2}, {"rank": 1, "flight": 2}],
        )

    async def test_cancelled_sole_watcher_preserves_and_drains_active_flight(self):
        sender = _RecordingSender()
        communicator = _Communicator(sender, fan_out=2, mode="watching")

        cancelled = asyncio.create_task(communicator("first request"))
        await asyncio.sleep(0)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled

        observer = asyncio.create_task(communicator("duplicate request"))
        await asyncio.sleep(0)
        self.assertEqual(sender.sent, ["first request"])
        communicator.handle_recv({"rank": 0})
        communicator.handle_recv({"rank": 1})
        self.assertEqual(
            await observer,
            [{"rank": 0}, {"rank": 1}],
        )

        fresh = asyncio.create_task(communicator("fresh request"))
        await asyncio.sleep(0)
        self.assertEqual(sender.sent, ["first request", "fresh request"])
        communicator.handle_recv({"rank": 0, "flight": 2})
        communicator.handle_recv({"rank": 1, "flight": 2})
        self.assertEqual(
            await fresh,
            [{"rank": 0, "flight": 2}, {"rank": 1, "flight": 2}],
        )


if __name__ == "__main__":
    unittest.main()
