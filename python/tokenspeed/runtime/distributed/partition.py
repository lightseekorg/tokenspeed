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

"""Pure execution-layer partitioning shared by model construction and PD."""

from __future__ import annotations


def target_execution_stage_windows(
    num_execution_layers: int,
    pp_size: int,
    partition: tuple[int, ...] | None,
) -> list[tuple[int, int]]:
    """Partition target execution blocks, independently of cache storage.

    Args:
        num_execution_layers: Target execution block count, excluding draft layers.
        pp_size: Number of nonempty pipeline stages.
        partition: Explicit target layers per stage, or None for even cuts.

    Returns:
        Ordered half-open windows covering exactly the target layers.
    """
    if (
        isinstance(num_execution_layers, bool)
        or not isinstance(num_execution_layers, int)
        or num_execution_layers < 1
    ):
        raise ValueError("target layer count must be a positive integer")
    if (
        isinstance(pp_size, bool)
        or not isinstance(pp_size, int)
        or not 1 <= pp_size <= num_execution_layers
    ):
        raise ValueError("pipeline stages must each own at least one target layer")
    if partition is not None:
        if len(partition) != pp_size:
            raise ValueError(
                f"pp layer partition {partition} has {len(partition)} entries "
                f"for {pp_size} pipeline stages"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in partition
        ):
            raise ValueError(
                f"pp layer partition {partition} must give every stage at least one layer"
            )
        if sum(partition) != num_execution_layers:
            raise ValueError(
                f"pp layer partition {partition} sums to {sum(partition)} "
                f"but the model has {num_execution_layers} execution layers"
            )
        counts = partition
    else:
        base, remainder = divmod(num_execution_layers, pp_size)
        counts = tuple(base + (stage < remainder) for stage in range(pp_size))
    windows = []
    start = 0
    for length in counts:
        windows.append((start, start + length))
        start += length
    return windows
