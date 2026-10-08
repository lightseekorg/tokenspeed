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


import json
import queue

from tokenspeed.runtime.engine.cache_trace import CacheTraceWriter


def test_cache_trace_close_finishes_when_sentinel_enqueue_times_out(
    tmp_path, monkeypatch
):
    writer = CacheTraceWriter(str(tmp_path / "capture"), {"global_rank": 0})
    original_put = writer._queue.put

    def timeout_on_shutdown(item, *args, **kwargs):
        if item is None:
            raise queue.Full
        return original_put(item, *args, **kwargs)

    monkeypatch.setattr(writer._queue, "put", timeout_on_shutdown)
    try:
        writer.publish([{"kind": "admitted"}])
        writer.close()
        assert not writer._thread.is_alive()
        with open(writer.path) as stream:
            records = [json.loads(line) for line in stream]
        assert [record["kind"] for record in records] == [
            "capture_start",
            "admitted",
            "capture_end",
        ]
    finally:
        if writer._thread.is_alive():
            original_put(None, timeout=1)
            writer._thread.join(timeout=1)


def test_cache_trace_history_requires_native_empty_start_and_no_gap(tmp_path):
    writer = CacheTraceWriter(str(tmp_path / "capture"), {"global_rank": 0})
    writer.publish(
        [
            {"kind": "start", "reason": "empty", "sequence": 1},
            {"kind": "admitted", "sequence": 2},
            {"kind": "gap", "sequence": 4, "reason": "native_buffer_full"},
            {"kind": "admitted", "sequence": 5},
        ]
    )
    writer.close()
    with open(writer.path) as stream:
        records = [json.loads(line) for line in stream]
    assert [record["history_complete"] for record in records] == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]
