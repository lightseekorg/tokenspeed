from __future__ import annotations

import json

import pytest

from tokenspeed.runtime.model_loader.weight_utils import kv_cache_scales_loader


def _scale_payload(*, model_type: str = "minimax_m3_text") -> dict:
    return {
        "model_type": model_type,
        "kv_cache": {
            "dtype": "float8_e4m3fn",
            "scaling_factor": {"0": {"0": 0.25, "1": 0.5}},
        },
    }


def _load(filename: str, *, strict: bool = False):
    return list(
        kv_cache_scales_loader(
            filename,
            tp_rank=0,
            tp_size=1,
            num_hidden_layers=2,
            model_type="minimax_m3_text",
            strict=strict,
        )
    )


@pytest.mark.parametrize("strict", [False, True], ids=["legacy", "strict"])
def test_kv_cache_scale_loader_accepts_valid_file(tmp_path, strict: bool) -> None:
    scale_path = tmp_path / "scales.json"
    scale_path.write_text(json.dumps(_scale_payload()))

    assert _load(str(scale_path), strict=strict) == [(0, 0.25), (1, 0.5)]


def test_kv_cache_scale_loader_rejects_missing_file(tmp_path) -> None:
    scale_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        _load(str(scale_path), strict=True)


def test_kv_cache_scale_loader_rejects_invalid_json(tmp_path) -> None:
    scale_path = tmp_path / "scales.json"
    scale_path.write_text("{not-json")

    with pytest.raises(ValueError, match="not valid JSON"):
        _load(str(scale_path), strict=True)


@pytest.mark.parametrize(
    "payload",
    [
        {"model_type": "minimax_m3_text"},
        _scale_payload(model_type="another_model"),
    ],
    ids=["invalid-schema", "wrong-model"],
)
def test_kv_cache_scale_loader_rejects_schema_or_model_mismatch(
    tmp_path, payload
) -> None:
    scale_path = tmp_path / "scales.json"
    scale_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="failed schema validation"):
        _load(str(scale_path), strict=True)


def test_kv_cache_scale_loader_legacy_default_accepts_missing_file(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    scale_path = tmp_path / "missing.json"

    assert _load(str(scale_path)) == []
    assert "Defaulting to KV cache scaling factors = 1.0" in caplog.text


def test_kv_cache_scale_loader_legacy_default_accepts_invalid_json(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    scale_path = tmp_path / "scales.json"
    scale_path.write_text("{not-json")

    assert _load(str(scale_path)) == []
    assert "Defaulting to KV cache scaling factors = 1.0" in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        {"model_type": "minimax_m3_text"},
        _scale_payload(model_type="another_model"),
    ],
    ids=["invalid-schema", "wrong-model"],
)
def test_kv_cache_scale_loader_legacy_default_accepts_schema_or_model_mismatch(
    tmp_path, caplog: pytest.LogCaptureFixture, payload
) -> None:
    scale_path = tmp_path / "scales.json"
    scale_path.write_text(json.dumps(payload))

    assert _load(str(scale_path)) == []
    assert "Defaulting to KV cache scaling factors = 1.0" in caplog.text
