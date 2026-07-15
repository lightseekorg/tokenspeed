import json
from pathlib import Path

import pytest
from runtime_env_preflight import audit_environment, main


def test_clean_environment_allows_vendor_and_packaging_plumbing(tmp_path: Path) -> None:
    result = audit_environment(
        {
            "HOME": str(tmp_path),
            "PATH": "/usr/bin",
            "HF_TOKEN": "redacted",
            "NCCL_CUMEM_ENABLE": "1",
            "CUDA_MODULE_LOADING": "LAZY",
            "PYTHONPATH": "/workspace/source",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        },
        home=tmp_path,
    )

    assert result["ok"] is True
    assert result["forbidden_environment_keys"] == []


@pytest.mark.parametrize(
    "key",
    [
        "TOKENSPEED_MM_SKIP_COMPUTE_HASH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "FLASHINFER_WORKSPACE_SIZE",
        "TORCHINDUCTOR_ENABLE_PDL",
        "TRTLLM_ENABLE_PDL",
        "TLLM_LOG_LEVEL",
        "SINGLE_WORKER_ID",
        "REQUEST_TIMEOUT",
    ],
)
def test_exact_forbidden_key_fails_even_when_empty(tmp_path: Path, key: str) -> None:
    result = audit_environment({key: ""}, home=tmp_path)

    assert result["ok"] is False
    assert result["forbidden_environment_keys"] == [key]
    assert not any("=" in failure for failure in result["failures"])


@pytest.mark.parametrize(
    "key",
    [
        "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DECODE",
        "TOKENSPEED_KERNEL_PROFILE_OUTPUT",
        "TOKENSPEED_KERNEL_CAPTURE_SHAPES_OUTPUT",
        "TOKENSPEED_EPD_ENCODE_RING_SLOTS",
        "TOKENSPEED_FUTURE_UNLISTED_TOGGLE",
        "SMG_MM_PIXEL_RDMA",
        "SMG_FUTURE_UNLISTED_TOGGLE",
        "EPD_PIXEL_SHM",
        "EPD_FUTURE_UNLISTED_TOGGLE",
        "TS_DSA_DECODE_TOPK_CUTEDSL",
        "TS_FUTURE_UNLISTED_TOGGLE",
    ],
)
def test_forbidden_prefix_key_fails(tmp_path: Path, key: str) -> None:
    result = audit_environment({key: "secret-value"}, home=tmp_path)

    assert result["ok"] is False
    assert result["forbidden_environment_keys"] == [key]
    assert "secret-value" not in json.dumps(result)


def test_default_kernel_override_file_and_broken_symlink_fail(tmp_path: Path) -> None:
    override = tmp_path / ".config/tokenspeed-kernel/overrides.yaml"
    override.parent.mkdir(parents=True)
    override.symlink_to(tmp_path / "missing.yaml")

    result = audit_environment({}, home=tmp_path)

    assert result["ok"] is False
    assert result["kernel_override_file"] == {
        "path": str(override),
        "present": True,
    }


def test_main_writes_name_only_failure_artifact(tmp_path: Path, capsys) -> None:
    output = tmp_path / "audit/result.json"
    rc = main(
        ["--output", str(output)],
        env={
            "HOME": str(tmp_path),
            "TOKENSPEED_KERNEL_PROFILE": "do-not-record",
        },
    )

    artifact = json.loads(output.read_text())
    assert rc == 1
    assert artifact["forbidden_environment_keys"] == ["TOKENSPEED_KERNEL_PROFILE"]
    assert "do-not-record" not in output.read_text()
    assert "do-not-record" not in capsys.readouterr().out
