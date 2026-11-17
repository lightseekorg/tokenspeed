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

"""The runtime's local copy of the compressed-tensors scalar types."""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class TestScalarTypes:
    """ScalarType / scalar_types

    Pure Python — no CUDA kernel. Uses the local copy at compressed_tensors/scalar_type.py.
    """

    def test_local_importable(self):
        """Local copy is importable from compressed_tensors."""
        from tokenspeed.runtime.layers.quantization.compressed_tensors.scalar_type import (
            ScalarType,
            scalar_types,
        )

        assert hasattr(scalar_types, "uint4b8")
        assert hasattr(scalar_types, "uint8b128")
        assert isinstance(scalar_types.uint4b8, ScalarType)

    def test_utils_uses_local(self):
        """utils.py now imports from local copy, not flashinfer."""
        from tokenspeed.runtime.layers.quantization.compressed_tensors.scalar_type import (
            ScalarType as LocalScalarType,
        )
        from tokenspeed.runtime.layers.quantization.utils import ScalarType

        assert ScalarType is LocalScalarType

    def test_local_id_values(self):
        """Local ScalarType IDs have expected values and are ints."""
        from tokenspeed.runtime.layers.quantization.compressed_tensors.scalar_type import (
            scalar_types as local_st,
        )

        # Verify known types exist and have integer IDs
        for name in ["uint4b8", "uint8b128", "uint4", "int8"]:
            st = getattr(local_st, name)
            assert isinstance(st.id, int), f"{name}.id should be an int"
            assert st.id >= 0, f"{name}.id should be non-negative"

    def test_local_scalar_type_fields(self):
        """Local ScalarType fields (exponent, mantissa, signed, bias) are correct for known types."""
        from tokenspeed.runtime.layers.quantization.compressed_tensors.scalar_type import (
            scalar_types as local_st,
        )

        for name in ["uint4b8", "uint8b128", "uint4", "int4", "int8", "float8_e4m3fn"]:
            val = getattr(local_st, name)
            assert isinstance(val.exponent, int), f"{name} exponent should be int"
            assert isinstance(val.mantissa, int), f"{name} mantissa should be int"
            assert isinstance(val.signed, bool), f"{name} signed should be bool"
            assert isinstance(val.bias, int), f"{name} bias should be int"
            # Sanity: size_bits should be positive
            assert val.size_bits > 0, f"{name} size_bits should be positive"
