import shlex
from pathlib import Path

from pier.agents.installed.base import BaseInstalledAgent
from pier.agents.network import allowlist_from_urls
from pier.environments.base import BaseEnvironment
from pier.environments.docker.docker import DockerEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec
from pier.models.agent.network import NetworkAllowlist


class KimiCodeEnvironment(DockerEnvironment):
    def __init__(self, *args, kimi_binary_path: str, **kwargs):
        binary = Path(kimi_binary_path).resolve()
        if not binary.is_file():
            raise ValueError(f"kimi-code binary not found: {binary}")
        super().__init__(*args, **kwargs)
        self._mounts_json.append(
            {
                "type": "bind",
                "source": binary.as_posix(),
                "target": "/usr/local/bin/kimi",
                "read_only": True,
            }
        )


class KimiCodeAgent(BaseInstalledAgent):
    VERSION = "0.23.6"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("version", self.VERSION)
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "kimi-code"

    def get_version_command(self) -> str:
        return "kimi --version"

    def install_spec(self) -> AgentInstallSpec | None:
        if self._version != self.VERSION:
            raise ValueError(f"unsupported kimi-code version: {self._version}")
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        await self.exec_as_agent(environment, self.get_version_command())

    def network_allowlist(self) -> NetworkAllowlist:
        base_url = self._get_env("KIMI_MODEL_BASE_URL")
        return allowlist_from_urls([base_url] if base_url else [])

    def populate_context_post_run(self, context: AgentContext) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self._get_env("KIMI_MODEL_BASE_URL"):
            raise ValueError("KIMI_MODEL_BASE_URL is required")
        env = self.build_process_env(
            {
                "KIMI_CODE_HOME": "/logs/agent/kimi-code",
                "KIMI_DISABLE_TELEMETRY": "1",
            }
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -o pipefail; mkdir -p /logs/agent/kimi-code; "
                f"kimi --prompt {shlex.quote(instruction)} "
                "--output-format stream-json 2>&1 | "
                "tee /logs/agent/kimi-code.jsonl"
            ),
            env=env,
            cwd="/app",
        )
