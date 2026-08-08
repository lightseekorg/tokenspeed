# Copyright (c) 2026 LightSeek Foundation

"""Capability helpers for native latent-space MoE execution."""

from __future__ import annotations

from tokenspeed_kernel.platform import current_platform


def native_latent_moe_available() -> bool:
    """Return whether the active backend provides native latent-space MoE."""

    return current_platform().is_amd


__all__ = ["native_latent_moe_available"]
