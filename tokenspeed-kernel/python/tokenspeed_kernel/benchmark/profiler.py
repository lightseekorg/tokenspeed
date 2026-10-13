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

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, Protocol, Self

__all__ = ["BenchmarkProfiler", "ProfilePhase", "NullProfiler"]

ProfilePhase = Literal[
    "eager_metadata",
    "measurement",
    "graph_replay",
]


class BenchmarkProfiler(Protocol):
    """Profiler boundary used by the benchmark harness."""

    profiles: dict[str, dict[str, Any]]

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None: ...

    def profile(
        self,
        case_id: str,
        phase: ProfilePhase,
        invocation_index: int | None,
    ) -> AbstractContextManager[None]: ...


class NullProfiler:
    """Profiler implementation for runs without diagnostic profiling."""

    def __init__(self) -> None:
        self.profiles: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        pass

    def profile(
        self,
        case_id: str,
        phase: ProfilePhase,
        invocation_index: int | None,
    ) -> AbstractContextManager[None]:
        return nullcontext()
