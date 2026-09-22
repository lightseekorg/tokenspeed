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
through 256 rows; an exact kernel override also permits larger diagnostic
shapes.

The wrapper separates code generation from launch-context capacity:

- A one-worker probe is compiled once for each static tactic.
- Occupancy is queried per CUDA stream because a stream can identify a reduced
  green context.
- The resident cluster count determines `workers`. Final plans are keyed by
  the static tactic plus `workers` and scheduling `rounds`, not by stream.
  Streams with equivalent effective capacity therefore share compiled code.
- Occupancy results remain keyed by stream and plan so every cooperative launch
  is checked against the context in which it runs.

Activation and epoch storage is persistent per device and workspace layout:
`(projection_rows, workers, clusters, slot_rows)`. It is intentionally not
duplicated for every stream. Epoch values are Lamport-style protocol state, so
this storage cannot be borrowed from the generic scratch pool and overwritten
between calls.

Shared workspaces follow TokenSpeed's single model execution-lane contract.
Calls on different streams must have an explicit ordering edge before they use
the same layout; concurrent side-stream launches require caller-owned isolated
storage and are not supported by this wrapper. CUDA graph capture records the
workspace addresses, so warm all needed layouts and each capture stream's
occupancy entries before capture. Graphs that share a layout must likewise be
replayed serially.
