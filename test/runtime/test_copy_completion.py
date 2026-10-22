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

"""Causal tests for per-result copy completion without an accelerator."""

import gc
import unittest
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

from tokenspeed.runtime.execution.copy_completion import CopyStreamPool


class QueueStream:
    def __init__(self):
        self.work = []
        self.done = 0
        self.sync_calls = 0
        self.fail_sync = False

    def submit(self, fn):
        self.work.append(fn)

    def run_until(self, count):
        while self.done < count:
            self.work[self.done]()
            self.done += 1

    def wait_stream(self, producer):
        count = len(producer.work)
        self.submit(lambda: producer.run_until(count))

    def synchronize(self):
        self.sync_calls += 1
        if self.fail_sync:
            raise RuntimeError("injected stream failure")
        self.run_until(len(self.work))


class DeviceModule:
    def __init__(self):
        self.current = None
        self.streams = []
        self.devices = []

    def Stream(self, *, device):
        stream = QueueStream()
        self.streams.append(stream)
        return stream

    @contextmanager
    def stream(self, stream):
        before = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = before

    @contextmanager
    def device(self, device):
        self.devices.append(device)
        yield


class Source:
    pass


class CopyCompletionTest(unittest.TestCase):
    def setUp(self):
        self.device = DeviceModule()
        self.producer = QueueStream()
        self.pool = CopyStreamPool(self.device, "npu:3")

    def issue(self, source, generation):
        self.producer.submit(lambda: source.update(value=generation))
        destination = {"value": "stale pinned input metadata"}
        completion = self.pool.copy_after(self.producer, (source,))
        with completion:
            self.device.current.submit(lambda: destination.update(source))
        return completion, destination

    def test_early_event_wait_cannot_publish_old_host_data(self):
        source = {}
        completion, destination = self.issue(source, 271)
        # Reproduce the observed event behavior: the wait returns while the
        # copy has not executed. The leased stream is a separate boundary.
        event = SimpleNamespace(synchronize=lambda: None)
        event.synchronize()
        self.assertEqual(destination["value"], "stale pinned input metadata")
        completion.synchronize()
        self.assertEqual(destination["value"], 271)
        self.assertEqual(self.device.devices, ["npu:3"])

    def test_three_pending_generations_do_not_wait_for_future_collective(self):
        source = {}
        pending = [self.issue(source, generation) for generation in (11, 22, 33)]
        self.assertEqual(len({id(c.stream) for c, _ in pending}), 3)
        self.assertEqual(self.pool._available, [])

        def undispatched_collective():
            raise AssertionError("commit waited for a future forward")

        self.producer.submit(undispatched_collective)
        for generation, (completion, destination) in zip((11, 22, 33), pending):
            completion.synchronize()
            self.assertEqual(destination["value"], generation)
        self.assertEqual(self.producer.sync_calls, 0)
        # Later source writes cannot change already committed CPU results.
        self.assertEqual([d["value"] for _, d in pending], [11, 22, 33])

    def test_reverse_commits_and_reuse_preserve_generations(self):
        source = {}
        pending = [self.issue(source, generation) for generation in range(3)]
        for completion, _ in reversed(pending):
            completion.synchronize()
        again = [self.issue(source, generation) for generation in range(3, 6)]
        self.assertEqual(len(self.device.streams), 3)
        for completion, _ in again:
            completion.synchronize()
        self.assertEqual([d["value"] for _, d in pending + again], list(range(6)))

    def test_sources_survive_until_copy_completion(self):
        source = Source()
        ref = weakref.ref(source)
        completion = self.pool.copy_after(self.producer, (source,))
        with completion:
            pass
        del source
        gc.collect()
        self.assertIsNotNone(ref())
        completion.synchronize()
        gc.collect()
        self.assertIsNone(ref())

    def test_failed_copy_wait_does_not_recycle_stream_or_release_source(self):
        source = Source()
        ref = weakref.ref(source)
        completion = self.pool.copy_after(self.producer, (source,))
        with completion:
            pass
        del source
        completion.stream.fail_sync = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            completion.synchronize()
        self.assertIsNotNone(ref())
        self.assertEqual(self.pool._available, [])

    def test_copy_submission_error_cleans_up_before_recycling(self):
        destination = []
        completion = self.pool.copy_after(self.producer, ())
        with self.assertRaisesRegex(ValueError, "submission"):
            with completion:
                self.device.current.submit(lambda: destination.append("copied"))
                raise ValueError("submission")
        self.assertEqual(destination, ["copied"])
        self.assertEqual(self.pool._available, [completion.stream])
        self.assertIsNone(self.device.current)

    def test_copy_scope_must_finish_and_can_only_sync_once(self):
        completion = self.pool.copy_after(self.producer, ())
        with self.assertRaisesRegex(RuntimeError, "finished copy scope"):
            completion.synchronize()
        with completion:
            pass
        completion.synchronize()
        with self.assertRaisesRegex(RuntimeError, "finished copy scope"):
            completion.synchronize()


if __name__ == "__main__":
    unittest.main()
