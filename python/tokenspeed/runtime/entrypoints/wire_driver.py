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

"""Drive a headless engine over its msgpack wire, for tests and tools.

``WireDriver`` starts the headless launcher (``entrypoints/engine.py``) and
binds the frontend sockets its ranks dial with the Rust client
(``tokenspeed_engine.FrontendListener``, ``ts-client``). It submits token-id
requests and gathers their outputs, and runs control operations on every
engine. It is no serving frontend: it has no tokenizer, detokenizer, stop
strings, n > 1 or HTTP surface, which are SMG's and Dynamo's.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

from tokenspeed_engine import FrontendListener

from tokenspeed.runtime.engine import zmq_wire
from tokenspeed.runtime.engine.io_struct import (
    MsgpackEncoder,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

# How long one message may wait for an engine's socket.
_SEND_TIMEOUT_S = 60.0
# How long the launcher may take to stop its ranks before it is killed with
# them (it gives a rank 30 s to exit itself).
_SHUTDOWN_GRACE_S = 60.0


def generate_request(
    input_ids: list[int], sampling_params: dict, **fields
) -> TokenizedGenerateReqInput:
    """A generate request as the engines take it, with a fresh request id.

    Args:
        input_ids: The prompt.
        sampling_params: ``SamplingParams``' fields.
        **fields: Any other ``TokenizedGenerateReqInput`` field: logprobs,
            the PD bootstrap, multimodal inputs, a DP pin, ...
    """
    return TokenizedGenerateReqInput(
        rid=uuid.uuid4().hex,
        input_ids=list(input_ids),
        sampling_params=SamplingParams(**sampling_params),
        **fields,
    )


@dataclasses.dataclass
class Completion:
    """A finished request's outputs, gathered from its chunks."""

    output_ids: list[int] = dataclasses.field(default_factory=list)
    # "stop", "length" or "abort".
    finish_reason: str | None = None
    # The finish reason in full: the matched token, an abort's message.
    finish_details: dict | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Prompt tokens served from the prefix cache.
    cached_tokens: int = 0
    output_token_logprobs: list[float] = dataclasses.field(default_factory=list)
    # The prompt logprobs (None for the first position) and their tokens.
    input_token_logprobs: tuple[list[float | None], list[int]] | None = None
    # With speculative decoding: verify rounds, and accepted tokens per round.
    spec_verify_ct: int | None = None
    accept_draft_tokens: float | None = None

    def add(self, chunk: dict) -> None:
        self.output_ids.extend(chunk["output_ids"])
        self.output_token_logprobs.extend(chunk["output_token_logprobs"])
        self.finish_reason = chunk["finish_reason"]
        self.finish_details = chunk["finish_details"]
        self.prompt_tokens = chunk["prompt_tokens"]
        self.completion_tokens = chunk["completion_tokens"]
        self.cached_tokens = chunk["cached_tokens"]
        if chunk["input_token_logprobs"] is not None:
            self.input_token_logprobs = chunk["input_token_logprobs"]
        self.spec_verify_ct = chunk["spec_verify_ct"]
        self.accept_draft_tokens = chunk["accept_draft_tokens"]


class WireDriver:
    """A headless engine and the frontend connection to it.

    Not thread-safe: one call at a time per driver. Drivers of the two
    roles of a disaggregated deployment run their calls on two threads.
    """

    def __init__(
        self,
        server_args: list[str],
        *,
        num_engines: int,
        routing: str,
        startup_timeout_s: float,
        env: dict[str, str] | None = None,
    ):
        """Start the engine and wait until every engine registered.

        Args:
            server_args: The engine's command line (``ServerArgs``), without
                the frontend address, which the driver adds.
            num_engines: The engine's attention-DP ranks.
            routing: How requests are placed on them: ``"round_robin"``,
                ``"least_in_flight"`` or ``"minimum_cache_usage"``, or
                ``"bootstrap_room"`` for a disaggregated role.
            startup_timeout_s: How long the engine may take to register.
            env: Variables to set for the engine's processes over this
                process's environment, such as ``CUDA_VISIBLE_DEVICES``.

        Raises:
            RuntimeError: the engine exited, or did not register in time.
        """
        listener = FrontendListener.bind(
            handshake_address="tcp://127.0.0.1:0",
            input_address="tcp://127.0.0.1:0",
            output_address="tcp://127.0.0.1:0",
            first_engine_index=0,
            num_engines=num_engines,
            routing=routing,
            handshake_timeout_s=startup_timeout_s,
            send_timeout_s=_SEND_TIMEOUT_S,
        )
        handshake_port = listener.handshake_address.rsplit(":", 1)[1]
        pending = listener.accept()
        self._encoder = MsgpackEncoder()
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tokenspeed.runtime.entrypoints.engine",
                *server_args,
                "--data-parallel-address",
                "127.0.0.1",
                "--data-parallel-rpc-port",
                handshake_port,
                "--zmq-engine-index",
                "0",
            ],
            env={**os.environ, **(env or {})},
            # Its own process group, so a shutdown reaches every rank.
            start_new_session=True,
        )
        accepted: dict = {}

        def accept() -> None:
            try:
                accepted["frontend"] = pending.wait()
            except Exception as error:  # noqa: BLE001 - re-raised below
                accepted["error"] = error

        waiter = threading.Thread(target=accept, daemon=True)
        waiter.start()
        try:
            while waiter.is_alive():
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"the engine exited with rc={self.process.returncode} "
                        "before it registered"
                    )
                waiter.join(timeout=0.5)
            if "error" in accepted:
                raise RuntimeError(
                    f"the engine did not register: {accepted['error']}"
                ) from accepted["error"]
        except BaseException:
            self._stop_process()
            raise
        self.frontend = accepted["frontend"]

    def engines(self) -> list[dict]:
        """The engines as they registered: index, ``max_model_len``, DP rank
        and size, ``kv_cache_size_tokens`` (the tokens its cache holds)."""
        return list(self.frontend.engines())

    def generate(self, requests: list[TokenizedGenerateReqInput]) -> list[Completion]:
        """Run ``requests`` together; their completions, in order.

        A request the engine refuses completes with an ``"abort"`` finish
        reason and the refusal in its ``finish_details``.
        """
        return asyncio.run(self._generate(requests))

    async def _generate(self, requests):
        completions = {request.rid: Completion() for request in requests}
        for request in requests:
            frames = [bytes(frame) for frame in self._encoder.encode(request)]
            await self.frontend.generate(frames)
        unfinished = set(completions)
        while unfinished:
            for rid, chunk in await self.frontend.next_outputs():
                if rid not in completions:
                    # A request of an earlier call that did not finish.
                    continue
                if isinstance(chunk, Exception):
                    raise chunk
                completions[rid].add(chunk)
                if chunk["finish_reason"] is not None:
                    unfinished.discard(rid)
        return [completions[request.rid] for request in requests]

    def control(self, operation) -> list:
        """Run a control operation (``zmq_wire.CONTROL_REQUESTS``) on every
        engine; each engine's reply."""

        async def run():
            replies = await self.frontend.control([zmq_wire.encode_control(operation)])
            return [zmq_wire.decode_control_output(frames[0]) for frames in replies]

        return asyncio.run(run())

    def shutdown(self) -> None:
        """Stop the engine; its ranks stop with the launcher."""
        self._stop_process()

    def _stop_process(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            deadline = time.monotonic() + _SHUTDOWN_GRACE_S
            while self.process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.2)
        # Whatever is left of the process group: ranks the launcher could not
        # stop, or all of it when the launcher itself did not stop.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()

    def __enter__(self) -> WireDriver:
        return self

    def __exit__(self, *exc_info) -> None:
        self.shutdown()
