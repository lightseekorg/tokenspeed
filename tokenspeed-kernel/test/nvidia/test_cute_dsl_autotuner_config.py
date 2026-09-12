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

import pytest

autotuner = pytest.importorskip("flashinfer.autotuner")


def _sentinel(*args):
    return args


def test_nvfp4_tuning_config_uses_indexed_flashinfer_initializers() -> None:
    packed_initializer = _sentinel
    dynamic_spec = autotuner.DynamicTensorSpec(
        (0, 6),
        (0, 0),
        _sentinel,
        _sentinel,
    )
    config = autotuner.TuningConfig(
        dynamic_tensor_specs=(dynamic_spec,),
        tensor_initializers=(
            (0, packed_initializer),
            (6, autotuner.autotuner_initializer_empty),
        ),
        use_cold_l2_cache=True,
    )

    assert config.dynamic_tensor_specs[0].input_idx == (0, 6)
    assert config.tensor_initializers[0] == (0, packed_initializer)
    assert config.tensor_initializers[1] == (
        6,
        autotuner.autotuner_initializer_empty,
    )
