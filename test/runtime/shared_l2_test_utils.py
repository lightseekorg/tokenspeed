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

"""Shared pieces of the shared L2 service tests: the real owner, run from its
command line in a process of its own over the scheduler's ``SharedL2Pool``,
and what a rank sends it."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from array import array

import pytest

from tokenspeed.runtime.cache.l3.identity import (
    L3_RUNTIME_COMPAT,
    storage_deployment_id,
)
from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection, OwnerError
from tokenspeed.runtime.utils.network import get_free_port

LCM_BYTES = 64
GEOMETRY = {
    "prefix_granularity": 2,
    "lcm_block_bytes": LCM_BYTES,
    "groups": [
        {
            "group_id": "kv",
            "attn_kind": "Full",
            "block_granularity": 2,
            "cache_blocks_per_lcm_block": 2,
            "host_group_packing": 2,
            "shard_count": 1,
            "sliding_window_tokens": None,
            "sink_tokens": 0,
            "replayable": False,
            "checkpoint_granularity": 1,
        },
        {
            "group_id": "state",
            "attn_kind": "State",
            "block_granularity": 0,
            "cache_blocks_per_lcm_block": 1,
            "host_group_packing": 1,
            "shard_count": 1,
            "sliding_window_tokens": None,
            "sink_tokens": 0,
            "replayable": False,
            "checkpoint_granularity": 1,
        },
    ],
}


def deployment_id(revision: str = "abc") -> str:
    """The deployment identity of a rank that loaded ``revision``."""
    return storage_deployment_id(
        "org/model",
        revision=revision,
        model_overrides={},
        cache_signature="layout",
        pipeline_rank=0,
        attn_tp_size=1,
        group_shard_counts=(1, 1),
        draft_model="",
        draft_revision="",
        cache_quantization="",
        runtime_compat=L3_RUNTIME_COMPAT,
        attention_backend="mha",
        draft_attention_backend="",
        skip_softmax_threshold=0.0,
        eagle3_layers_to_capture=[],
        kv_producer={"numerics": "resolved", "attn_kvp_size": 1},
    )


def token_bytes(prompt: list[int]) -> bytes:
    return array("i", prompt).tobytes()


def prompt_keys(prompt: list[int]) -> tuple[list[str], tuple | None, list[tuple]]:
    """A prompt's page hashes (two tokens a page), its tail, and the keys of
    its blocks: a KV block per page and the State block at its end."""
    hashes = [
        "-".join(str(token) for token in prompt[:end])
        for end in range(2, len(prompt) + 1, 2)
    ]
    tail = (1, "-".join(str(token) for token in prompt)) if len(prompt) % 2 else None
    end = hashes[-1] if tail is None else tail[1]
    return hashes, tail, [(0, 0, page, 0) for page in hashes] + [(0, 1, end, 0)]


def write_fields(prompt: list[int]) -> tuple[list, list, list, list]:
    """The keys, buckets, Device blocks and checkpoints of the write that
    stores ``prompt`` with the checkpoint at its end."""
    hashes, tail, keys = prompt_keys(prompt)
    return (
        keys,
        [0] * len(keys),
        list(range(1, len(keys) + 1)),
        [(token_bytes(prompt), hashes, tail, 7)],
    )


def lookup_fields(prompt: list[int]) -> tuple[list[str], bytes, int]:
    return prompt_keys(prompt)[0], token_bytes(prompt), len(prompt)


def wait_until(predicate, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "the awaited state never came"
        time.sleep(0.002)


class ServedOwner:
    """The owner service, run from its command line in a process of its
    own."""

    def __init__(
        self,
        address: str | None = None,
        *,
        num_providers: int = 1,
        joined_providers: int = 1,
        provider_blocks: int = 8,
        client_timeout_s: float = 60.0,
    ):
        """
        Args:
            address: Where to bind; None picks a free local port.
            num_providers: Providers the owner waits for.
            joined_providers: Providers that join right away.
            provider_blocks: LCM blocks of each.
            client_timeout_s: The owner's client timeout.
        """
        self.address = address or f"tcp://127.0.0.1:{get_free_port()}"
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tokenspeed.runtime.cache.shared_l2",
                "owner",
                f"--address={self.address}",
                f"--num-providers={num_providers}",
                "--headroom-lcm-blocks=0",
                f"--client-timeout-s={client_timeout_s}",
                "--checkpoint-retention-requests=0",
                "--checkpoint-retention-leaf-distance=0",
            ]
        )
        self.admin = OwnerConnection(self.address, timeout_ms=30000)
        # The process serves once it answers; it has no pool yet.
        with pytest.raises(OwnerError, match="pool is not ready"):
            self.admin.request("stats")
        for index in range(joined_providers):
            self.add_provider(f"node{index}", provider_blocks)

    def add_provider(self, session: str, blocks: int) -> None:
        self.admin.request(
            "add_provider",
            session=session,
            base_address=1 << 40,
            bytes=blocks * LCM_BYTES,
        )

    def stats(self) -> dict:
        return self.admin.request("stats")["stats"]

    def stall(self) -> None:
        """Stop answering. Requests wait in the socket for ``resume``."""
        self.process.send_signal(signal.SIGSTOP)

    def resume(self) -> None:
        self.process.send_signal(signal.SIGCONT)

    def close(self) -> None:
        self.admin.close()
        # A stalled process is killed as a running one is.
        self.process.kill()
        self.process.wait()
