"""Pier adapter for running Kimi Code inside an isolated DeepSWE task."""

from __future__ import annotations

import json
import shlex
from urllib.parse import urlparse

from pier.agents.installed.base import BaseInstalledAgent, with_prompt_template
from pier.agents.network import allowlist_from_urls
from pier.environments import agent_setup as pier_agent_setup
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist

KIMI_CODE_VERSION = "0.23.6"
_DEFAULT_SQUID_BOOTSTRAP = pier_agent_setup.squid_bootstrap_command


def _squid_bootstrap_with_http_connect() -> str:
    """Allow authenticated CONNECT tunnels to allowlisted HTTP endpoints."""
    bootstrap = _DEFAULT_SQUID_BOOTSTRAP()
    original = "http_access deny CONNECT !SSL_ports"
    replacement = "http_access deny CONNECT !Safe_ports"
    if bootstrap.count(original) != 1:
        raise RuntimeError("Pier Squid CONNECT policy has changed")
    return bootstrap.replace(original, replacement)


# Kimi Code uses Node's undici EnvHttpProxyAgent, which tunnels both HTTP and
# HTTPS requests with CONNECT. Pier 0.3.1 permits safe HTTP destinations but
# restricts CONNECT to port 443, so extend the existing authenticated,
# allowlisted proxy policy to the full Safe_ports set (80 and 443).
pier_agent_setup.squid_bootstrap_command = _squid_bootstrap_with_http_connect


class KimiCodeAgent(BaseInstalledAgent):
    """Install and run the Kimi Code CLI through Pier's Docker environment."""

    @staticmethod
    def name() -> str:
        return "kimi-code"

    def get_version_command(self) -> str | None:
        return '"$HOME/.kimi-code/bin/kimi" --version'

    def install_spec(self) -> AgentInstallSpec:
        version = self._version or KIMI_CODE_VERSION
        install_command = (
            "set -euo pipefail; "
            "command -v curl >/dev/null; "
            "curl -fsSL https://code.kimi.com/kimi-code/install.sh | "
            f"KIMI_VERSION={shlex.quote(version)} "
            'KIMI_INSTALL_DIR="$HOME/.kimi-code" '
            "KIMI_NO_MODIFY_PATH=1 bash; "
            '"$HOME/.kimi-code/bin/kimi" --version'
        )
        return AgentInstallSpec(
            agent_name=self.name(),
            version=version,
            steps=[InstallStep(user="agent", run=install_command)],
            verification_command=self.get_version_command(),
        )

    def network_allowlist(self) -> NetworkAllowlist:
        base_url = self._get_env("KIMI_MODEL_BASE_URL")
        if not base_url:
            return NetworkAllowlist()
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError(
                "KIMI_MODEL_BASE_URL must be an http URL reachable through "
                "Pier's port-80 egress proxy"
            )
        if parsed.port not in (None, 80):
            raise ValueError("KIMI_MODEL_BASE_URL must use port 80")
        return allowlist_from_urls([base_url])

    def populate_context_post_run(self, context: AgentContext) -> None:
        output_path = self.logs_dir / "kimi-code.jsonl"
        if not output_path.is_file():
            return

        steps = 0
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("role") == "assistant":
                steps += 1
        context.n_agent_steps = steps or None

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError("model_name is required")

        model_name = self.model_name.split("/", 1)[-1]
        base_url = self._get_env("KIMI_MODEL_BASE_URL")
        if not base_url:
            raise ValueError("KIMI_MODEL_BASE_URL is required")

        env = self.build_process_env(
            {
                "KIMI_DISABLE_TELEMETRY": "1",
                "KIMI_MODEL_NAME": model_name,
                "KIMI_MODEL_API_KEY": "EMPTY_TOKEN",
                "KIMI_MODEL_PROVIDER_TYPE": "kimi",
                "KIMI_MODEL_BASE_URL": base_url,
                "KIMI_MODEL_MAX_CONTEXT_SIZE": "80000",
                "KIMI_MODEL_CAPABILITIES": "thinking,always_thinking,tool_use",
                "KIMI_MODEL_TEMPERATURE": self._get_env("KIMI_MODEL_TEMPERATURE")
                or "1.0",
                "KIMI_MODEL_TOP_P": self._get_env("KIMI_MODEL_TOP_P") or "0.95",
                "KIMI_MODEL_THINKING_EFFORT": self._get_env(
                    "KIMI_MODEL_THINKING_EFFORT"
                )
                or "max",
                "KIMI_MODEL_THINKING_KEEP": self._get_env("KIMI_MODEL_THINKING_KEEP")
                or "all",
            }
        )
        await self.exec_as_agent(
            environment,
            command=(
                'trap \'cp "$HOME/.kimi-code/logs/kimi-code.log" '
                "/logs/agent/kimi-code.log 2>/dev/null || true' EXIT; "
                '"$HOME/.kimi-code/bin/kimi" '
                f"--prompt {shlex.quote(instruction)} --output-format stream-json "
                "2>&1 </dev/null | tee /logs/agent/kimi-code.jsonl"
            ),
            env=env,
        )
