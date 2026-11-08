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

"""Opt-in full-model regression against an isolated DP8 V4.1 test server."""

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_dp_cached_and_open_prefill_chunks_complete():
    url = os.environ.get("TOKENSPEED_DSV41_DP_TEST_URL")
    if not url:
        pytest.skip("requires an isolated DP8 V4.1 service with prefix cache enabled")

    def generate(tokens, output_tokens):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        payload = json.dumps(
            {
                "input_ids": tokens,
                "stream": False,
                "sampling_params": {
                    "max_new_tokens": output_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            }
        ).encode()
        request = urllib.request.Request(
            url.rstrip("/") + "/generate",
            payload,
            {"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=300) as response:
            result = json.load(response)
        meta = result["meta_info"]
        assert meta["prompt_tokens"] == len(tokens)
        assert meta["completion_tokens"] == output_tokens
        assert meta["finish_reason"]
        return meta

    # Longer than the test server's 4096-token chunk: cold ranks have open
    # CED chunks while a cached peer already needs its final decoder tail.
    prompts = [[101 + rank] * 8256 for rank in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        cold = list(pool.map(lambda tokens: generate(tokens, 16), prompts))
    assert all(meta["cached_tokens"] == 0 for meta in cold), "use a fresh test service"
    with ThreadPoolExecutor(max_workers=8) as pool:
        mixed = list(pool.map(lambda _: generate(prompts[0], 1), range(8)))
    cached = [meta["cached_tokens"] for meta in mixed]
    assert min(cached) == 0 and max(cached) > 4096, (
        "routing did not exercise both cold and cached ranks",
        cached,
    )
