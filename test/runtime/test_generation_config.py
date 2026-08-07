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

"""Tests for ``get_generation_config`` (vendor-neutral ``generation_config.json``
load, replacing ``GenerationConfig.from_pretrained``)."""

import json

from tokenspeed.runtime.configs.utils import get_generation_config


def test_get_generation_config_loads_local_dir_as_dict(tmp_path) -> None:
    raw = {"eos_token_id": [1, 2], "pad_token_id": 0}
    (tmp_path / "generation_config.json").write_text(json.dumps(raw), encoding="utf-8")

    assert get_generation_config(str(tmp_path)) == raw


def test_get_generation_config_returns_none_when_missing(tmp_path) -> None:
    assert get_generation_config(str(tmp_path)) is None
