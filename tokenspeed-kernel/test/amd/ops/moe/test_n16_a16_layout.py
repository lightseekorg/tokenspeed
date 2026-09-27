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

"""CPU address/layout contracts; no tensor, GPU, or compiler imports."""

import ast
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[5]
_SOURCE = (
    _ROOT / "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4"
)
_Layout = namedtuple("Layout", "size_per_thread threads_per_warp warps_per_cta order")


def _definition(file, name):
    path = _SOURCE / file
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    namespace = {
        "gl": SimpleNamespace(
            BlockedLayout=lambda *axes: _Layout(*(tuple(a) for a in axes))
        )
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


class N16A16LayoutTests(unittest.TestCase):
    def test_non_n16_warp_layouts_are_unchanged(self):
        layouts = _definition("situ_decode.py", "_a16_weight_layouts")
        for n, k, waves in (
            (2, 1024, 1),
            (8, 1024, 4),
            (4, 256, 4),
            (8, 128, 1),
            (8, 512, 8),
        ):
            with self.subTest(n=n, k=k, waves=waves):
                packed, expanded, compute = layouts(n, k, waves, False)
                rows = (n + waves - 1) // waves
                self.assertEqual(
                    packed, _Layout((rows, k // 64), (1, 64), (waves, 1), (1, 0))
                )
                self.assertEqual(
                    expanded, _Layout((rows, k // 32), (1, 64), (waves, 1), (1, 0))
                )
                self.assertEqual(compute, expanded)

    def test_n16_loads_cover_cells_with_original_compute_k_ownership(self):
        layouts = _definition("situ_decode.py", "_a16_weight_layouts")
        offset = _definition("n16_weights.py", "_n16_weight_offset")
        for n, k, waves in ((8, 1024, 4), (8, 512, 8), (4, 512, 4), (4, 128, 4)):
            packed, expanded, compute = layouts(n, k, waves, True)
            self.assertEqual(compute, layouts(n, k, waves, False)[2])
            self.assertEqual(compute.threads_per_warp, (1, 64))
            self.assertEqual(compute.size_per_thread[1], 2 * k // 64)
            n_lanes = min(compute.size_per_thread[0], 2)
            k_lanes = 64 // n_lanes
            self.assertEqual(packed.threads_per_warp, (n_lanes, k_lanes))
            self.assertEqual(packed.warps_per_cta, (waves, 1))
            self.assertEqual(expanded.size_per_thread[1], 2 * packed.size_per_thread[1])
            per_lane = packed.size_per_thread[1]
            k_tile = per_lane * k_lanes
            seen = set()
            for wave in range(waves):
                for lane in range(64):
                    row = wave * n_lanes + lane % n_lanes
                    for base in range(0, k, k_tile):
                        column = (base + lane // n_lanes * per_lane) % k
                        addresses = [
                            offset(row, column + r, k) for r in range(per_lane)
                        ]
                        self.assertEqual(
                            addresses,
                            list(range(addresses[0], addresses[0] + per_lane)),
                        )
                        for r in range(per_lane):
                            seen.add((row, column + r))
            self.assertEqual(
                seen, {(row, column) for row in range(n) for column in range(k)}
            )

    def test_n16_load_and_upcast_keep_each_compute_waves_rows(self):
        layouts = _definition("situ_decode.py", "_a16_weight_layouts")
        for n, k, waves in (
            (8, 1024, 4),
            (4, 512, 4),
            (8, 512, 8),
            (4, 128, 4),
            (8, 128, 8),
            (2, 1024, 1),
            (8, 256, 1),
        ):
            with self.subTest(n=n, k=k, waves=waves):
                packed, expanded, compute = layouts(n, k, waves, True)
                rows = compute.size_per_thread[0]
                for layout in (packed, expanded):
                    self.assertEqual(layout.warps_per_cta, (waves, 1))
                    n_lanes = layout.threads_per_warp[0]
                    self.assertEqual(n_lanes, min(rows, 2))
                    per_lane = layout.size_per_thread[0]
                    for wave in range(waves):
                        actual = {
                            (wave * n_lanes + lane) * per_lane + reg
                            for lane in range(n_lanes)
                            for reg in range(per_lane)
                        }
                        self.assertEqual(
                            actual, set(range(wave * rows, (wave + 1) * rows))
                        )

    def test_masked_small_k_loads_keep_whole_upcast_groups(self):
        layouts = _definition("situ_decode.py", "_a16_weight_layouts")
        for n, waves in ((4, 4), (8, 8)):
            packed, expanded, compute = layouts(n, 128, waves, True)
            self.assertEqual(packed.size_per_thread[1], 4)
            self.assertEqual(expanded.size_per_thread[1], 8)
            self.assertEqual(compute.size_per_thread[1], 4)
            self.assertEqual(layouts(n, 128, waves, False)[1].size_per_thread[1], 4)

    def test_masked_partial_cells_never_address_outside_the_bank(self):
        offset = _definition("n16_weights.py", "_n16_weight_offset")
        for dimension in (256, 512, 768, 1024, 3072, 3584):
            packed_k = dimension // 2
            seen = set()
            for base in range(0, packed_k, 256):
                for n in range(16):
                    for k in range(base, base + 256):
                        if k < packed_k:
                            address = offset(n, k, packed_k)
                            self.assertTrue(0 <= address < 16 * packed_k)
                            seen.add(address)
            self.assertEqual(seen, set(range(16 * packed_k)))

    def test_gate_up_cells_and_group32_scales_keep_their_roles(self):
        weight = _definition("n16_weights.py", "_n16_weight_offset")
        scale = _definition("n16_weights.py", "_n32_scale_offset")
        for n in range(64):
            gate = n // 16 * 32 + n % 16
            up = gate + 16
            for group in range(24):
                self.assertEqual(
                    weight(up, group * 16, 384) - weight(gate, group * 16, 384),
                    16 * 384,
                )
                self.assertEqual(scale(up, group, 24) - scale(gate, group, 24), 1)


if __name__ == "__main__":
    unittest.main()
