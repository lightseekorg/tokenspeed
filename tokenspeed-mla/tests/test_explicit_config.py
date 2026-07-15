from pathlib import Path

import pytest

from tokenspeed_mla import fmha_binary
from tokenspeed_mla import mla_prefill


def test_cutedsl_is_the_stable_default_even_with_legacy_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TOKENSPEED_MLA_PREFILL_BACKEND", "binary")
    monkeypatch.setenv("TOKENSPEED_MLA_FMHA_BINARY_SO", "/legacy/env.so")

    assert mla_prefill._resolve_backend("cutedsl") == "cutedsl"


def test_binary_backend_receives_explicit_so_path(monkeypatch, tmp_path) -> None:
    so_path = tmp_path / "fmha.so"
    seen = []

    def has_binary_prefill(path=None):
        seen.append(path)
        return True

    monkeypatch.setattr(
        mla_prefill._fmha_binary,
        "has_binary_prefill",
        has_binary_prefill,
    )

    assert mla_prefill._resolve_backend("binary", so_path) == "binary"
    assert seen == [so_path]


def test_binary_path_requires_binary_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires backend='binary'"):
        mla_prefill._resolve_backend("cutedsl", tmp_path / "fmha.so")


def test_explicit_binary_path_does_not_read_legacy_environment(
    monkeypatch, tmp_path
) -> None:
    explicit = tmp_path / "explicit.so"
    monkeypatch.setenv("TOKENSPEED_MLA_FMHA_BINARY_SO", "/legacy/env.so")

    assert fmha_binary._resolve_so_path(explicit) == Path(explicit)


def test_has_binary_prefill_loads_the_explicit_path(monkeypatch, tmp_path) -> None:
    explicit = tmp_path / "explicit.so"
    loaded = []
    monkeypatch.setattr(
        fmha_binary,
        "_load_module",
        lambda path: loaded.append(path),
    )

    assert fmha_binary.has_binary_prefill(explicit)
    assert loaded == [str(explicit)]
