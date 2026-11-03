# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Device helpers for persistent gfx950 kernels.

A persistent kernel keeps one workgroup on every CU for its whole launch and
hands work between its workgroups through memory instead of kernel
boundaries. These helpers carry no model or operation knowledge.
"""

from tokenspeed_kernel_amd.ops.gfx950.persistent._intrinsics import (
    MAX_VMCNT_WAIT,
    memrealtime,
    persistent_compile_options,
    release_drain,
    scalar_load_glc,
    trap,
    vote,
    wait_vmcnt,
    wave_acquire,
    xcc_id,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent.sync import (
    ENTRY_WORDS,
    ERROR_ABORT,
    ERROR_CALLER,
    ERROR_MIRROR,
    ERROR_RECORD,
    ERROR_RECORD_FIELDS,
    ERROR_WORDS,
    FIRST_KERNEL_SITE,
    GRID,
    JOIN_SET_WORDS,
    JOIN_WORDS,
    LINE_WORDS,
    SITE_ABORTED,
    SITE_CLAIM,
    SITE_JOIN,
    SLOTS,
    XCDS,
    Entry,
    Launch,
    enter_check,
    enter_issue,
    fail_stop,
    grid_join,
    launch_tag,
    logical_cta,
    poll_expired,
    poll_pairs,
    poll_stop_site,
    poll_tags16,
    poll_until,
    poll_until_all,
    store_pairs,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent.timing import (
    REALTIME_TICK_NS,
    realtime_ticks,
    trace_stamp,
)

__all__ = [
    "ENTRY_WORDS",
    "ERROR_ABORT",
    "ERROR_CALLER",
    "ERROR_MIRROR",
    "ERROR_RECORD",
    "ERROR_RECORD_FIELDS",
    "ERROR_WORDS",
    "Entry",
    "FIRST_KERNEL_SITE",
    "GRID",
    "JOIN_SET_WORDS",
    "JOIN_WORDS",
    "LINE_WORDS",
    "Launch",
    "MAX_VMCNT_WAIT",
    "REALTIME_TICK_NS",
    "SITE_ABORTED",
    "SITE_CLAIM",
    "SITE_JOIN",
    "SLOTS",
    "XCDS",
    "enter_check",
    "enter_issue",
    "fail_stop",
    "grid_join",
    "launch_tag",
    "logical_cta",
    "memrealtime",
    "persistent_compile_options",
    "poll_expired",
    "poll_pairs",
    "poll_stop_site",
    "poll_tags16",
    "poll_until",
    "poll_until_all",
    "realtime_ticks",
    "release_drain",
    "scalar_load_glc",
    "store_pairs",
    "trace_stamp",
    "trap",
    "vote",
    "wait_vmcnt",
    "wave_acquire",
    "xcc_id",
]
