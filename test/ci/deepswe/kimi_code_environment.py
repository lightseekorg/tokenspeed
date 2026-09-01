"""Pier Docker environment with a pinned host Kimi Code binary."""

from __future__ import annotations

import os
from pathlib import Path

from pier.environments.docker.docker import DockerEnvironment
from pier.models.trial.config import ServiceVolumeConfig

KIMI_CODE_CONTAINER_BINARY = "/opt/kimi-code/bin/kimi"


class KimiCodeDockerEnvironment(DockerEnvironment):
    """Mount the runner's pinned Kimi Code binary into every trial."""

    def __init__(
        self,
        *args,
        mounts_json: list[ServiceVolumeConfig] | None = None,
        **kwargs,
    ) -> None:
        binary_value = os.environ.get("PIER_KIMI_CODE_BINARY")
        if not binary_value:
            raise ValueError("PIER_KIMI_CODE_BINARY is required")

        binary = Path(binary_value).resolve(strict=True)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(
                f"PIER_KIMI_CODE_BINARY must be an executable file: {binary}"
            )

        resolved_mounts = None if mounts_json is None else list(mounts_json)
        super().__init__(*args, mounts_json=resolved_mounts, **kwargs)
        assert self._mounts_json is not None
        self._mounts_json.append(
            {
                "type": "bind",
                "source": binary.as_posix(),
                "target": KIMI_CODE_CONTAINER_BINARY,
                "read_only": True,
                "bind": {"create_host_path": False},
            }
        )
