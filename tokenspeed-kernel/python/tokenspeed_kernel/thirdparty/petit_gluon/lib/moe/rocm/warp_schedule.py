# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass


@dataclass(frozen=True)
class WarpSchedule:
    __triton_builtin__ = True
    Config: object

    def __post_init__(self):
        c = self.Config
        assert c.kGroupN % c.kNumWarps == 0
        tile_n = c.kGroupN // c.kNumWarps
        assert tile_n % 16 == 0
        object.__setattr__(self, "kWarpTileM", c.kGroupM)
        object.__setattr__(self, "kWarpTileN", tile_n)
        object.__setattr__(self, "kMmaTileN", 16)
        object.__setattr__(self, "kAccumulatorRows", 2)
        object.__setattr__(self, "kColumnIters", tile_n // self.kMmaTileN)
        object.__setattr__(
            self, "kAccumulatorFragments", self.kAccumulatorRows * self.kColumnIters
        )
        assert self.kAccumulatorFragments % 2 == 0
