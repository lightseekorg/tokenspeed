"""Protocol-conformance guard: ``AsyncLLM`` explicitly inherits
``EngineClient``.

Locks the inheritance as a class-level invariant so a future
refactor cannot silently break protocol conformance by dropping a
base class, renaming a method, or removing an attribute the OpenAI
serving layer depends on.

The check uses ``EngineClient in AsyncLLM.__mro__`` rather than
``issubclass(AsyncLLM, EngineClient)``: ``runtime_checkable``
``Protocol`` classes with attribute members raise ``TypeError:
Protocols with non-method members don't support issubclass()``, and
``EngineClient`` mixes attributes with methods. A direct MRO
membership check works regardless of protocol shape.

Registered on ``runtime-1gpu`` only because the tokenspeed
import graph currently requires a CUDA-capable environment
(``triton`` import); the test itself does not touch the GPU.
"""

import os
import sys

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.engine.protocol import EngineClient  # noqa: E402


class TestAsyncLLMSatisfiesEngineClient:
    """D.3 invariant: ``AsyncLLM`` inherits ``EngineClient`` explicitly.

    MRO membership check keeps this test purely structural — no
    live engine needed — and sidesteps ``issubclass``'s refusal to
    operate on ``@runtime_checkable Protocol`` classes that carry
    attribute members.
    """

    def test_async_llm_mro_contains_engine_client(self) -> None:
        mro_names = [cls.__name__ for cls in AsyncLLM.__mro__]
        _pytest_member_1 = EngineClient
        _pytest_container_2 = AsyncLLM.__mro__
        _pytest_msg_3 = (
            "AsyncLLM must explicitly inherit EngineClient so the "
            "OpenAI serving layer's EngineClient-typed callsites stay "
            "structurally sound. If this fails, check AsyncLLM's base "
            "class declaration in runtime/engine/async_llm.py. "
            f"Current MRO: {mro_names}"
        )
        assert _pytest_member_1 in _pytest_container_2, _pytest_msg_3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
