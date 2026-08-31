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

"""KDA convolution-state geometry shared by cache producers and consumers."""

from __future__ import annotations


def kda_conv_state_channel_axis(
    shape: tuple[int, ...],
    *,
    channels: int | None = None,
    history: int | None = None,
) -> int:
    """Return the channel axis of a physical KDA convolution-state row."""
    if (channels is None) == (history is None):
        raise ValueError("provide exactly one of channels or history")
    expected = channels if channels is not None else history
    if (
        expected is None
        or expected <= 0
        or len(shape) != 2
        or any(dim <= 0 for dim in shape)
    ):
        raise ValueError(
            "invalid KDA convolution state geometry: "
            f"shape={shape}, channels={channels}, history={history}"
        )
    matches = tuple(axis for axis, dim in enumerate(shape) if dim == expected)
    if len(matches) != 1:
        raise ValueError(
            "KDA convolution state must be [channels, history] or "
            "[history, channels], got "
            f"shape={shape}, channels={channels}, history={history}"
        )
    return matches[0] if channels is not None else 1 - matches[0]
