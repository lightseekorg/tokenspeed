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

"""Unit tests for LoraRegistry — no GPU required."""

from __future__ import annotations

import pytest

from tokenspeed.runtime.lora.lora_config import LoraConfig
from tokenspeed.runtime.lora.lora_registry import NO_LORA_ID, LoraRegistry


def _config(name: str, pinned: bool = False, r: int = 16) -> LoraConfig:
    return LoraConfig(name=name, path=f"/fake/{name}", r=r, pinned=pinned)


class TestLoraRegistry:
    def test_register_returns_unique_nonzero_ids(self):
        reg = LoraRegistry(max_loras=4)
        id_a = reg.register(_config("a"))
        id_b = reg.register(_config("b"))
        assert id_a != NO_LORA_ID
        assert id_b != NO_LORA_ID
        assert id_a != id_b

    def test_get_id_round_trips(self):
        reg = LoraRegistry(max_loras=4)
        lora_id = reg.register(_config("sql"))
        assert reg.get_id("sql") == lora_id
        assert reg.get_id("missing") is None

    def test_get_config_round_trips(self):
        reg = LoraRegistry(max_loras=4)
        cfg = _config("sql", r=32)
        reg.register(cfg)
        retrieved = reg.get_config("sql")
        assert retrieved is not None
        assert retrieved.r == 32

    def test_duplicate_registration_raises(self):
        reg = LoraRegistry(max_loras=4)
        reg.register(_config("a"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_config("a"))

    def test_capacity_enforced_for_non_pinned(self):
        reg = LoraRegistry(max_loras=2)
        reg.register(_config("a"))
        reg.register(_config("b"))
        with pytest.raises(ValueError, match="full"):
            reg.register(_config("c"))

    def test_pinned_does_not_count_toward_capacity(self):
        reg = LoraRegistry(max_loras=1)
        reg.register(_config("pinned", pinned=True))
        # max_loras=1 for non-pinned; this should succeed
        reg.register(_config("evictable"))
        # Second non-pinned should fail
        with pytest.raises(ValueError, match="full"):
            reg.register(_config("evictable2"))

    def test_unregister_frees_slot(self):
        reg = LoraRegistry(max_loras=1)
        reg.register(_config("a"))
        reg.unregister("a")
        assert reg.get_id("a") is None
        # Slot is now free
        reg.register(_config("b"))

    def test_unregister_unknown_raises(self):
        reg = LoraRegistry(max_loras=4)
        with pytest.raises(KeyError):
            reg.unregister("nonexistent")

    def test_contains(self):
        reg = LoraRegistry(max_loras=4)
        reg.register(_config("x"))
        assert "x" in reg
        assert "y" not in reg

    def test_len(self):
        reg = LoraRegistry(max_loras=4)
        assert len(reg) == 0
        reg.register(_config("a"))
        assert len(reg) == 1
        reg.register(_config("b"))
        assert len(reg) == 2
        reg.unregister("a")
        assert len(reg) == 1

    def test_lora_scaling(self):
        cfg = LoraConfig(name="t", path="/p", r=8, lora_alpha=16)
        assert cfg.scaling == 2.0
