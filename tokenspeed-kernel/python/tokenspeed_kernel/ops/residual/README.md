<!--
Copyright (c) 2026 LightSeek Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Residual kernels

## Fused hyperconnection mix

The Blackwell CuTe implementation fuses the down projection, SiLU, up
projection, gating, and residual reduction. Automatic dispatch uses it for one
through 256 rows; an exact kernel override permits diagnostic shapes through
1024 rows.

The wrapper separates code generation from launch-context capacity:

- A one-worker probe is compiled once for each static tactic.
- Occupancy is queried per CUDA stream because a stream can identify a reduced
  green context.
- The resident cluster count determines `workers`. Final plans are keyed by
  the static tactic plus `workers` and scheduling `rounds`, not by stream.
  Streams with equivalent effective capacity therefore share compiled code.
- Occupancy results remain keyed by stream and plan so every cooperative launch
  is checked against the context in which it runs.
- CuTe derives launch shared memory directly from the kernel allocations. The
  occupancy query uses the loaded function's conservative opt-in limit; every
  supported tactic already has one-CTA-per-SM shared-memory residency.

Plans and occupancy entries live for the process lifetime. Production uses a
small fixed set of tactics and streams; diagnostic callers must not scan
unbounded projection scales or stream handles.

Activation and epoch storage is persistent per device and workspace layout:
`(projection_rows, workers, clusters, slot_rows)`. It is intentionally not
duplicated for every stream. Epoch values are Lamport-style protocol state, so
this storage cannot be borrowed from the generic scratch pool and overwritten
between calls.

Each CTA initializes its shared barriers once, then alternates their phases
across projection and token tiles. Down-stage phases account for the number of
K tiles assigned to each stage, including tactics where stages receive unequal
numbers of tiles.

Shared workspaces follow the main-stream scratch ownership contract documented
in `docs/design/event-loop.md`. Calls on different streams must have an explicit
ordering edge before they use the same layout; concurrent side-stream launches
require caller-owned isolated storage and are not supported by this wrapper.
CUDA graph capture records the workspace addresses, so warm all needed layouts
and each capture stream's occupancy entries before capture. Graphs that share a
layout must likewise be replayed serially.
