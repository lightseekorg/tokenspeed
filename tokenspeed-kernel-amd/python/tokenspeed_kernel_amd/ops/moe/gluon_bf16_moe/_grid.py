"""Shared grid / PID-remap helper (from the a16w16 v9 tutorial).

XCD-aware PID remapping + GROUP_SIZE_M workgroup swizzling for L2 cache
locality on MI350/MI355 (8 XCDs, one L2 per XCD). By default adjacent
workgroups land on different XCDs, destroying cache reuse of the shared
operand (in a MoE the reused operand is the per-expert weight tile, which
is shared across the consecutive sorted M-blocks of one expert). Remapping
so adjacent tiles share an XCD, plus GROUP_SIZE_M grouping, recovers that
reuse. See ``v9_beyond_hotloop`` in ROCm/gfx950-gluon-tutorials.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def get_pids(
    num_pid_m,
    num_pid_n,
    GRID_MN,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """Map ``program_id(0)`` to ``(pid_m, pid_n)`` with XCD remap + M-grouping."""
    pid = gl.program_id(axis=0)

    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (
                tall_xcds * pids_per_xcd
                + (xcd - tall_xcds) * (pids_per_xcd - 1)
                + local_pid
            )

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n
