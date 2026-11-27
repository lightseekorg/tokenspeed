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

"""TokenSpeed startup requires explicit policies before managing NPU weights."""

import importlib.util
import unittest
from pathlib import Path
from test.ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, suite="runtime-1gpu")

# This check also runs on a host without PyTorch or the NPU allocator.
_source = (
    Path(__file__).resolve().parents[2]
    / "python/tokenspeed/runtime/utils/ascend_memory_saver_state.py"
)
_spec = importlib.util.spec_from_file_location("ascend_saver_state_under_test", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate_memory_saver_scope = _module.validate_memory_saver_scope


class MemorySaverScopeTests(unittest.TestCase):
    def test_parameter_and_draft_restore_policies_are_explicit(self):
        for has_draft, weight, draft, valid in (
            (False, "cpu_backup", None, True),
            (False, "discard", None, True),
            (False, None, None, False),
            (True, "cpu_backup", None, True),
            (True, "discard", "retain", True),
            (True, "discard", "refresh", True),
            (True, "discard", None, False),
        ):
            with self.subTest(has_draft=has_draft, weight=weight, draft=draft):
                args = dict(
                    has_draft_weights=has_draft,
                    weight_policy=weight,
                    draft_policy=draft,
                )
                if valid:
                    validate_memory_saver_scope(**args)
                else:
                    with self.assertRaises(ValueError):
                        validate_memory_saver_scope(**args)


if __name__ == "__main__":
    unittest.main()
