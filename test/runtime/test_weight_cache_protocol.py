"""CPU-only unit tests for the weight cache daemon protocol helpers.

Covers the fingerprint (:class:`CacheConfig`), the quantization allowlist gate,
and the length-prefixed socket message protocol / daemon file lifecycle. These
tests never touch CUDA, so they run on any host.
"""

import os
import socket

import pytest

from tokenspeed.runtime.weight_cache import protocol
from tokenspeed.runtime.weight_cache.protocol import (
    CacheConfig,
    UnsupportedQuantForIPCError,
    check_ipc_quant_support,
    cleanup_stale_daemon_files,
    get_quant_method_name,
    get_ready_path,
    get_socket_path,
    hash_quant_config,
    is_ipc_quant_supported,
    recv_msg,
    send_msg,
)


def _make_config(**overrides):
    base = dict(
        model_path="/models/foo",
        model_arch="FooForCausalLM",
        world_size=2,
        rank=0,
        attn_tp_size=2,
        attn_tp_rank=0,
        dense_tp_size=2,
        moe_ep_size=1,
        moe_ep_rank=0,
        dp_size=1,
        quant_method="fp8",
        quant_config_hash="deadbeef",
        dtype="torch.bfloat16",
        revision="",
        device_capability="9.0",
        torch_version="2.4.0",
    )
    base.update(overrides)
    return CacheConfig(**base)


# ---------------------------------------------------------------------------
# CacheConfig
# ---------------------------------------------------------------------------


def test_cache_config_dict_round_trip():
    cfg = _make_config()
    restored = CacheConfig.from_dict(cfg.to_dict())
    assert restored == cfg
    assert cfg.matches(restored)


def test_cache_config_mismatch_on_rank():
    a = _make_config(rank=0, attn_tp_rank=0)
    b = _make_config(rank=1, attn_tp_rank=1)
    assert not a.matches(b)


def test_cache_config_mismatch_on_env_stamp():
    a = _make_config(device_capability="9.0")
    b = _make_config(device_capability="8.0")
    assert not a.matches(b)


def test_cache_config_from_dict_ignores_extra_keys():
    d = _make_config().to_dict()
    d["unexpected"] = "ignored"
    restored = CacheConfig.from_dict(d)
    assert restored == _make_config()


# ---------------------------------------------------------------------------
# hash_quant_config / get_quant_method_name
# ---------------------------------------------------------------------------


def test_hash_quant_config_none_is_empty():
    assert hash_quant_config(None) == ""


def test_hash_quant_config_dict_is_order_independent():
    h1 = hash_quant_config({"a": 1, "b": 2})
    h2 = hash_quant_config({"b": 2, "a": 1})
    assert h1 == h2
    assert h1 != hash_quant_config({"a": 1, "b": 3})


def test_hash_quant_config_object_with_to_dict_is_stable():
    class QC:
        def to_dict(self):
            return {"weight_block_size": [128, 128]}

    assert hash_quant_config(QC()) == hash_quant_config(QC())


def test_hash_quant_config_object_ignores_memory_address():
    """Two distinct instances with equal data must hash the same (no repr addr)."""

    class QC:
        def __init__(self):
            self.weight_block_size = [128, 128]
            self._private = object()

    assert hash_quant_config(QC()) == hash_quant_config(QC())


def test_get_quant_method_name_variants():
    assert get_quant_method_name(None) == ""
    assert get_quant_method_name("fp8") == "fp8"
    assert get_quant_method_name({"quant_method": "fp8"}) == "fp8"
    assert get_quant_method_name({"quant_algo": "nvfp4"}) == "nvfp4"
    assert get_quant_method_name({}) == ""

    class Named:
        name = "awq"

    assert get_quant_method_name(Named()) == "awq"

    class GetName:
        def get_name(self):
            return "gptq"

    assert get_quant_method_name(GetName()) == "gptq"


# ---------------------------------------------------------------------------
# IPC quant allowlist
# ---------------------------------------------------------------------------


def test_unquantized_is_supported():
    assert is_ipc_quant_supported("", None)
    check_ipc_quant_support("", None, where="test")  # no raise


def test_block_wise_fp8_is_supported():
    assert is_ipc_quant_supported("fp8", {"weight_block_size": [128, 128]})
    check_ipc_quant_support("fp8", {"weight_block_size": [128, 128]}, where="test")


def test_per_tensor_fp8_is_rejected():
    assert not is_ipc_quant_supported("fp8", {"weight_block_size": None})
    with pytest.raises(UnsupportedQuantForIPCError):
        check_ipc_quant_support("fp8", {}, where="daemon")


def test_unknown_quant_method_is_rejected():
    assert not is_ipc_quant_supported("awq", {})
    with pytest.raises(UnsupportedQuantForIPCError):
        check_ipc_quant_support("gptq", {}, where="client")


# ---------------------------------------------------------------------------
# Socket message protocol
# ---------------------------------------------------------------------------


def test_send_recv_msg_round_trip():
    a, b = socket.socketpair()
    try:
        payload = {"cmd": "fetch_state", "config": _make_config().to_dict()}
        send_msg(a, payload)
        assert recv_msg(b) == payload
    finally:
        a.close()
        b.close()


def test_recv_msg_rejects_oversized_frame(monkeypatch):
    monkeypatch.setattr(protocol, "MAX_MSG_SIZE", 8)
    a, b = socket.socketpair()
    try:
        send_msg(a, {"big": "x" * 1024})
        with pytest.raises(ValueError):
            recv_msg(b)
    finally:
        a.close()
        b.close()


def test_recv_msg_raises_on_closed_connection():
    a, b = socket.socketpair()
    a.close()
    try:
        with pytest.raises(ConnectionError):
            recv_msg(b)
    finally:
        b.close()


# ---------------------------------------------------------------------------
# Path helpers + daemon file lifecycle
# ---------------------------------------------------------------------------


def test_socket_and_ready_paths_include_rank():
    assert get_socket_path(3).endswith("rank3.sock")
    assert get_ready_path(3).endswith("rank3.ready")


def _point_templates_at(tmp_path, monkeypatch):
    monkeypatch.setattr(
        protocol,
        "WEIGHT_CACHE_SOCKET_TEMPLATE",
        os.path.join(str(tmp_path), "rank{rank}.sock"),
    )
    monkeypatch.setattr(
        protocol,
        "WEIGHT_CACHE_READY_TEMPLATE",
        os.path.join(str(tmp_path), "rank{rank}.ready"),
    )


def test_cleanup_removes_stale_files(tmp_path, monkeypatch):
    _point_templates_at(tmp_path, monkeypatch)
    ready = get_ready_path(0)
    sock_path = get_socket_path(0)
    # A dead PID leaves stale files that must be reclaimed.
    with open(ready, "w") as f:
        f.write("pid=2147483647\n")
    open(sock_path, "w").close()

    cleanup_stale_daemon_files(0)

    assert not os.path.exists(ready)
    assert not os.path.exists(sock_path)


def test_cleanup_refuses_live_daemon(tmp_path, monkeypatch):
    _point_templates_at(tmp_path, monkeypatch)
    ready = get_ready_path(0)
    # Our own PID is alive -> cleanup must refuse without force.
    with open(ready, "w") as f:
        f.write(f"pid={os.getpid()}\n")

    with pytest.raises(RuntimeError):
        cleanup_stale_daemon_files(0)

    assert os.path.exists(ready)


def test_cleanup_noop_when_no_files(tmp_path, monkeypatch):
    _point_templates_at(tmp_path, monkeypatch)
    cleanup_stale_daemon_files(0)  # must not raise
