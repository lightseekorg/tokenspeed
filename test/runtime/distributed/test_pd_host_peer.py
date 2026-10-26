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

"""CPU-only tests for the data plane's half of PD under the Rust host
(``pd/host_peer.py``, ``pd/factory.py``): the KV arena facts and settings the
host starts its transfer peer from, and the device fences that hand it each
plan's remote stream. The host's bootstrap server (``ts-pd``) stands in for
the Rust side that reads the contract this side publishes."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from dataclasses import replace
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from test.runtime.cache_pd_test_utils import group as make_group  # noqa: E402
from test.runtime.cache_pd_test_utils import layout as make_layout  # noqa: E402
from test.runtime.cache_pd_test_utils import segment as make_segment  # noqa: E402

from tokenspeed_engine import BootstrapServer  # noqa: E402

from tokenspeed.runtime.pd.cache_protocol import (  # noqa: E402
    build_cache_fields_by_producer_step,
)
from tokenspeed.runtime.pd.factory import KVArgs  # noqa: E402
from tokenspeed.runtime.pd.host_peer import (  # noqa: E402
    DecodeSettings,
    HostDecodePeer,
    HostPrefillPeer,
    PeerSettings,
    _prefill_settings,
)
from tokenspeed.runtime.pd.topology import PDParallelTopology  # noqa: E402

_SOURCE = 0x1000


def _layout():
    return make_layout(
        make_group(
            "history",
            make_segment(
                "layer.0.kv", dtype="bfloat16", shape=(8,), offset=0, stride=32
            ),
        ),
        make_group(
            "state",
            make_segment(
                "layer.1.state", dtype="bfloat16", shape=(8,), offset=16, stride=32
            ),
            attn_kind="state",
        ),
        capacity=16,
        page_bytes=32,
    )


def _field_ids(layout) -> tuple[str, ...]:
    return tuple(field.field_id for field in layout.plan.fields)


def _kv_args(rank: int, *, schedule: bool = True) -> KVArgs:
    """A rank's KV args: its arena's landing addresses are its own."""
    layout = _layout()
    bound = replace(
        layout,
        field_addresses={
            name: 0x1000 * (rank + 1) + position * 128
            for position, name in enumerate(_field_ids(layout))
        },
    )
    return KVArgs(
        engine_rank=rank,
        kv_data_ptr=_SOURCE,
        ib_device="",
        gpu_id=rank,
        cache_layout=bound,
        cache_fields_by_stage=(_field_ids(layout),),
        cache_producer_schedule=(
            build_cache_fields_by_producer_step(
                layout.plan,
                producer_fields_by_step=tuple((field,) for field in _field_ids(layout)),
            )
            if schedule
            else None
        ),
    )


def _topology(tp_rank: int) -> PDParallelTopology:
    return PDParallelTopology(
        tp_size=2,
        tp_rank=tp_rank,
        dp_size=1,
        dp_rank=0,
        world_size=2,
        global_rank=tp_rank,
    )


def _settings() -> PeerSettings:
    return PeerSettings(
        library="libtokenspeed_transfer_engine.so",
        local_ip="127.0.0.1",
        protocol="rdma",
        ib_device="",
        engine_rank=0,
        regions=((_SOURCE, 512),),
    )


def _server_args(bootstrap_port: int) -> SimpleNamespace:
    # ``_prefill_settings`` reads only where the bootstrap server listens.
    return SimpleNamespace(
        dist_init_addr="127.0.0.1:5000",
        disaggregation_bootstrap_port=bootstrap_port,
    )


@pytest.mark.parametrize(
    ("vendor", "protocol"),
    [("nvidia", "rdma"), ("amd", "rdma"), ("ascend", "ascend_direct")],
)
def test_mooncake_protocol_uses_the_device_transport(monkeypatch, vendor, protocol):
    from tokenspeed_kernel.platform import current_platform

    from tokenspeed.runtime.pd import host_peer

    # Hardware discovery is unavailable on the host running this test.
    platform = replace(current_platform(), vendor=vendor)
    monkeypatch.setattr(host_peer, "current_platform", lambda: platform)
    assert host_peer.mooncake_protocol() == protocol


def _http(url: str, *, body: dict | None = None) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        method="GET" if body is None else "PUT",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


class _Outbox:
    """Stands in for the Rust host: records what the peer hands it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def remote_stream_ready(self) -> None:
        self.calls.append("remote_stream_ready")

    def release_registration(self) -> None:
        self.calls.append("release_registration")

    def restore_registration(self) -> None:
        self.calls.append("restore_registration")


def test_prefill_settings_publish_the_logical_layout_and_keep_the_landing_one():
    server = BootstrapServer(0, cache_layouts=True)
    try:
        payloads = []
        for rank in range(2):
            kv_args = _kv_args(rank)
            settings = _prefill_settings(
                _server_args(server.port), kv_args, _topology(rank), "127.0.0.1"
            )
            # The host plans copies against this rank's landing addresses;
            # every rank registers the same logical layout without them.
            assert settings.layout == kv_args.cache_layout.to_wire_bytes()
            assert json.loads(settings.wire_layout)["field_addresses"] is None
            assert settings.fields_by_stage == (_field_ids(_layout()),)
            assert settings.fields_by_step == (("layer.0.kv",), ("layer.1.state",))
            # Stage-major: the rank Decode counts this one's completion as.
            assert (settings.tp_rank, settings.pp_rank, settings.status_rank) == (
                rank,
                0,
                rank,
            )
            payload = json.loads(settings.bootstrap_payload)
            assert payload["cache_layout"] == settings.wire_layout.decode("ascii")
            payloads.append(payload)
            assert settings.bootstrap_url == f"http://127.0.0.1:{server.port}/route"
            # The host's bootstrap server parses the contract and accepts the
            # rank's registration once the host added its control port.
            status, _ = _http(
                settings.bootstrap_url, body={**payload, "rank_port": 7000 + rank}
            )
            assert status == 200
        assert payloads[0]["cache_layout"] == payloads[1]["cache_layout"]
        status, info = _http(
            f"http://127.0.0.1:{server.port}/route?engine_rank=-1&target_dp_group=-1"
        )
        assert status == 200
        info = json.loads(info)
        assert (info["prefill_tp_size"], info["prefill_dp_size"]) == (2, 1)
        assert info["cache_layout"] == payloads[0]["cache_layout"]
        assert info["cache_fields_by_stage"] == [list(_field_ids(_layout()))]
        status, rank = _http(
            f"http://127.0.0.1:{server.port}/route?engine_rank=1&target_dp_group=0"
        )
        assert (status, json.loads(rank)) == (
            200,
            {"rank_ip": "127.0.0.1", "rank_port": 7001},
        )
    finally:
        server.close()


def test_a_peer_hands_its_streams_and_sleeps_to_the_host():
    kv_args = _kv_args(0)
    outbox = _Outbox()
    prefill = HostPrefillPeer(
        outbox,
        _settings(),
        _prefill_settings(_server_args(8998), kv_args, _topology(0), "127.0.0.1"),
    )
    assert prefill.mode == "prefill"
    # The device fence passed: the host reads the stream off the plan it
    # staged.
    prefill.execute(object())
    prefill.release_cache_registration()
    prefill.restore_cache_registration()
    decode = HostDecodePeer(
        outbox,
        _settings(),
        DecodeSettings(
            arena_base=_SOURCE,
            layout=kv_args.cache_layout.to_wire_bytes(),
            tp_size=2,
            tp_rank=0,
            waiting_timeout_s=300.0,
            heartbeat_interval_s=5.0,
            heartbeat_max_failures=2,
        ),
    )
    assert decode.mode == "decode"
    decode.execute(object())
    assert outbox.calls == [
        "remote_stream_ready",
        "release_registration",
        "restore_registration",
        "remote_stream_ready",
    ]


def test_layerwise_streaming_registers_its_step_counter():
    settings = _prefill_settings(
        _server_args(8998), _kv_args(0), _topology(0), "127.0.0.1"
    )
    peer = HostPrefillPeer(_Outbox(), _settings(), settings)
    assert (peer.layerwise_step_address, peer.layerwise_interval) == (None, 0)
    counter = SimpleNamespace(
        h_ready_cache_step=SimpleNamespace(data_ptr=lambda: 0xBEEF)
    )
    peer.register_layerwise_step_counter(counter, 0)
    assert (peer.layerwise_step_address, peer.layerwise_interval) == (0xBEEF, 1)

    unscheduled = _prefill_settings(
        _server_args(8998), _kv_args(0, schedule=False), _topology(0), "127.0.0.1"
    )
    peer = HostPrefillPeer(_Outbox(), _settings(), unscheduled)
    with pytest.raises(ValueError, match="producer schedule"):
        peer.register_layerwise_step_counter(counter, 2)


def test_cache_factory_exposes_only_typed_arena() -> None:
    import torch

    from tokenspeed.runtime.pd.factory import get_kv_args

    layout = _layout()
    buffer = torch.zeros(layout.plan.arena_bytes, dtype=torch.uint8)
    pool = SimpleNamespace(
        arena=SimpleNamespace(
            supports_disaggregation=True,
            plan=layout.plan,
            cache_group_specs=layout.group_specs,
            contract_binding=lambda: buffer,
            field_address=lambda name: (
                buffer.data_ptr() + layout.plan.field_page_byte_offset(name, 0)
            ),
            registration_regions=lambda: ((buffer.data_ptr(), buffer.nbytes),),
        ),
    )

    kv_args = get_kv_args(
        0,
        0,
        "mlx5_0",
        pool,
        draft_model_config=None,
        cache_fields_by_stage=(_field_ids(layout),),
        producer_fields_by_step=tuple((field,) for field in _field_ids(layout)),
        logical_plan=None,
        model_config=SimpleNamespace(
            num_attention_layers=2,
            num_key_value_heads=1,
            hf_config=SimpleNamespace(),
            hf_text_config=SimpleNamespace(),
        ),
    )

    assert kv_args.engine_rank == 0
    assert kv_args.kv_data_ptr == buffer.data_ptr()
    assert kv_args.ib_device == "mlx5_0"
    assert kv_args.gpu_id == 0
    assert kv_args.cache_layout.plan == layout.plan
    assert kv_args.cache_layout.group_specs == layout.group_specs
    assert kv_args.cache_layout.transfer_schema == layout.transfer_schema
    assert kv_args.cache_layout.field_addresses == {
        field.field_id: buffer.data_ptr()
        + layout.plan.field_page_byte_offset(field.field_id, 0)
        for field in layout.plan.fields
    }
    assert kv_args.wire_layout.field_addresses is None
    assert kv_args.cache_producer_schedule.fields_by_step == (
        ("layer.0.kv",),
        ("layer.1.state",),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
