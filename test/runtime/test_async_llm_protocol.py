"""Guards that ``AsyncLLM`` explicitly inherits ``EngineClient``.

Uses ``EngineClient in AsyncLLM.__mro__`` because ``issubclass`` raises
``TypeError`` on runtime-checkable protocols with non-method members.

Registered on ``runtime-1gpu`` only because the tokenspeed import graph
requires a CUDA-capable environment; the test itself does not touch the GPU.
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
    def test_async_llm_mro_contains_engine_client(self) -> None:
        mro_names = [cls.__name__ for cls in AsyncLLM.__mro__]
        member = EngineClient
        container = AsyncLLM.__mro__
        msg = (
            "AsyncLLM must explicitly inherit EngineClient so the "
            "OpenAI serving layer's EngineClient-typed callsites stay "
            "structurally sound. If this fails, check AsyncLLM's base "
            "class declaration in runtime/engine/async_llm.py. "
            f"Current MRO: {mro_names}"
        )
        assert member in container, msg


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
