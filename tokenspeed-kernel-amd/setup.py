from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from setuptools import setup


BASE_VERSION = "0.1.1"
ROOT = Path(__file__).resolve().parent


def _version_date() -> str:
    override = os.environ.get("TOKENSPEED_KERNEL_VERSION_DATE")
    if override:
        return override

    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        return datetime.fromtimestamp(int(source_date_epoch), tz=timezone.utc).strftime(
            "%Y%m%d"
        )

    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _git_sha() -> str:
    override = os.environ.get("TOKENSPEED_KERNEL_GIT_SHA") or os.environ.get(
        "GIT_COMMIT"
    )
    if override:
        return override[:8].ljust(8, "0")

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short=8", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()[:8]
            .ljust(8, "0")
        )
    except (OSError, subprocess.CalledProcessError):
        return "00000000"


def _git_branch() -> str:
    for env_name in (
        "TOKENSPEED_KERNEL_GIT_BRANCH",
        "GITHUB_REF_NAME",
    ):
        branch = os.environ.get(env_name)
        if branch:
            return branch.removeprefix("refs/heads/")

    github_ref = os.environ.get("GITHUB_REF")
    if github_ref:
        return github_ref.removeprefix("refs/heads/")

    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _package_version() -> str:
    if _git_branch().startswith("release/"):
        return BASE_VERSION

    return f"{BASE_VERSION}.dev{_version_date()}+git{_git_sha()}"


setup(version=_package_version())
