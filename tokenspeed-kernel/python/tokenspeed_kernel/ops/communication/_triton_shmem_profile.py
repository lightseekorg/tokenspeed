"""Architecture-specific launch policy for the Triton-shmem AR+RMSNorm backend.

This keeps measured host-side choices separate from the device kernels without
introducing a multi-family profile registry. Only the AR+RMSNorm family is
integrated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ArchProfile:
    """Host-side launch tuning for one GPU architecture."""

    name: str
    grid_caps: dict[str, dict[int, int]]
    num_warps: dict[str, int]
    oneshot_max_ws: int
    block_n_bytes: int = 1024
    block_n_min: int = 128
    default_num_warps: int = 4
    tuned: bool = True

    def grid_cap(self, kernel: str, ws: int) -> int:
        caps = self.grid_caps.get(kernel) or self.grid_caps["twoshot_blocked"]
        return caps.get(ws) or caps[min(caps, key=lambda value: abs(value - ws))]

    def warps(self, kernel: str) -> int:
        return self.num_warps.get(kernel, self.default_num_warps)


_MI350X = ArchProfile(
    name="MI350X (gfx950)",
    grid_caps={
        "twoshot_blocked": {2: 256, 4: 256, 8: 256},
        "oneshot_blocked": {2: 128, 4: 256, 8: 256},
        "oneshot_wholerow": {2: 64, 4: 64, 8: 64},
    },
    num_warps={"twoshot_blocked": 4, "oneshot_blocked": 4, "oneshot_wholerow": 8},
    oneshot_max_ws=2,
)

_DEFAULT_PROFILE = ArchProfile(
    name="unsupported architecture (MI350X launch defaults)",
    grid_caps=_MI350X.grid_caps,
    num_warps=_MI350X.num_warps,
    oneshot_max_ws=_MI350X.oneshot_max_ws,
    tuned=False,
)

_PROFILES: dict[str, ArchProfile] = {"gfx950": _MI350X}


def detect_arch(device: int | None = None) -> str:
    """Return the base ``gfxNNN`` token for a CUDA/HIP device."""
    index = torch.cuda.current_device() if device is None else device
    return torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]


def get_arch_profile(arch: str | ArchProfile | None = None) -> ArchProfile:
    """Resolve an explicit profile or the current device's profile."""
    if isinstance(arch, ArchProfile):
        return arch
    if arch is None:
        try:
            arch = detect_arch()
        except Exception:  # noqa: BLE001 - device detection may fail during import
            return _DEFAULT_PROFILE
    return _PROFILES.get(arch, _DEFAULT_PROFILE)


def recommended_grid(
    kernel: str,
    ws: int,
    work_rows: int,
    num_cus: int,
    *,
    profile: str | ArchProfile | None = None,
) -> int:
    """Return the tuned grid bounded by work and available CUs."""
    cap = get_arch_profile(profile).grid_cap(kernel, ws)
    return max(1, min(cap, work_rows, num_cus))


def recommended_num_warps(
    kernel: str,
    *,
    profile: str | ArchProfile | None = None,
) -> int:
    """Return the tuned launch warp count."""
    return get_arch_profile(profile).warps(kernel)


def recommended_block_n(
    dtype: torch.dtype,
    hidden_size: int,
    *,
    profile: str | ArchProfile | None = None,
) -> int:
    """Return the tuned blocked-kernel width."""
    selected = get_arch_profile(profile)
    return min(
        hidden_size,
        max(selected.block_n_min, selected.block_n_bytes // dtype.itemsize),
    )


def recommended_kernel(
    ws: int,
    hidden_size: int,
    *,
    profile: str | ArchProfile | None = None,
) -> str:
    """Return the preferred AR+RMSNorm variant for ``(ws, hidden_size)``."""
    if ws <= get_arch_profile(profile).oneshot_max_ws:
        return (
            "oneshot_wholerow"
            if (hidden_size & (hidden_size - 1)) == 0
            else "oneshot_blocked"
        )
    return "twoshot_blocked"
