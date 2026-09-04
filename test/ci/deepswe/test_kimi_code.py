"""Tests for the Kimi Code Pier integration."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("pier")

from kimi_code_environment import (  # noqa: E402
    KIMI_CODE_CONTAINER_BINARY,
    KimiCodeDockerEnvironment,
)
from kimi_code_pier_agent import KimiCodeAgent  # noqa: E402
from pier.environments.docker.docker import DockerEnvironment  # noqa: E402


def test_environment_preserves_log_mounts_and_adds_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    binary = tmp_path / "kimi"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PIER_KIMI_CODE_BINARY", str(binary))

    def fake_init(self, *args, mounts_json=None, **kwargs) -> None:
        del args, kwargs
        self._mounts_json = mounts_json or [
            {"type": "bind", "source": "/logs", "target": "/logs"}
        ]

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    environment = KimiCodeDockerEnvironment()

    assert environment._mounts_json[0]["target"] == "/logs"
    assert environment._mounts_json[1] == {
        "type": "bind",
        "source": str(binary),
        "target": KIMI_CODE_CONTAINER_BINARY,
        "read_only": True,
        "bind": {"create_host_path": False},
    }


def test_agent_uses_kvv_runtime_parameters(monkeypatch, tmp_path) -> None:
    agent = KimiCodeAgent(
        logs_dir=tmp_path,
        model_name="kimi/kimi-k3",
        version="0.29.0",
        extra_env={"KIMI_MODEL_BASE_URL": "http://10.42.0.10/v1"},
    )
    captured = {}

    async def capture_exec(environment, command, env) -> None:
        captured.update(environment=environment, command=command, env=env)

    monkeypatch.setattr(agent, "exec_as_agent", capture_exec)
    asyncio.run(agent.run("solve it", object(), object()))

    assert captured["env"]["KIMI_MODEL_MAX_CONTEXT_SIZE"] == "1048576"
    assert captured["env"]["KIMI_MODEL_MAX_COMPLETION_TOKENS"] == "32768"
    assert captured["env"]["KIMI_MODEL_THINKING_EFFORT"] == "max"
    assert "KIMI_MODEL_TEMPERATURE" not in captured["env"]
    assert "KIMI_MODEL_TOP_P" not in captured["env"]
    assert KIMI_CODE_CONTAINER_BINARY in captured["command"]

    install_spec = agent.install_spec()
    assert install_spec.version == "0.29.0"
    assert install_spec.steps[0].run == "true"
