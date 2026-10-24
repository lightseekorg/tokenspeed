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

"""TokenSpeed's Rust engine: the rank binary and the frontend's client.

* ``host_binary()`` is ``ts-host``, one engine rank: the Rust control plane
  with the data plane in an embedded interpreter (``tokenspeed`` launches
  one per rank).
* ``FrontendListener`` / ``Frontend`` connect a Python frontend
  (``AsyncLLM``) to the ranks over the msgpack ZMQ wire (``ts-client``).
* ``L3Executor`` is the L3 storage tier's executor (``ts-l3``) for callers
  outside a rank, such as tests over a data plane's Host pool; a rank's
  host runs its own.
* ``BootstrapServer`` is a disaggregated deployment's HTTP rendezvous
  (``ts-pd``), which the launcher hosts.
"""

from pathlib import Path

from tokenspeed_engine._native import (
    BootstrapServer,
    EngineError,
    Frontend,
    FrontendListener,
    L3Executor,
    PendingFrontend,
)

__all__ = [
    "BootstrapServer",
    "EngineError",
    "Frontend",
    "FrontendListener",
    "L3Executor",
    "PendingFrontend",
    "host_binary",
]


def host_binary() -> Path:
    """The ``ts-host`` executable installed with this package."""
    return Path(__file__).resolve().parent / "bin" / "ts-host"
