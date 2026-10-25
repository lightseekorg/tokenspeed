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

"""Golden wire fixtures shared by the Python encoder and ``ts-wire``.

Each fixture is one ZMQ message exactly as the Python side puts it on the
wire, stored as one hex-encoded frame per line (``<name>.hex``). The
Python test ``test/runtime/test_ts_wire_fixtures.py`` checks that these files
still match what the current Python structs encode; the Rust conformance test
(``tests/conformance.rs``) decodes every file and re-encodes it byte for byte.
A schema change on either side therefore fails one of the two tests.

Run this script to rewrite the files after an intentional schema change:

    python tokenspeed-engine/crates/ts-wire/tests/fixtures/generate.py
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import torch

from tokenspeed.runtime.engine import zmq_wire
from tokenspeed.runtime.engine.io_struct import (
    BatchTokenIDOutSlim,
    FlushCacheReqOutput,
    GetInternalStateReqOutput,
    MsgpackEncoder,
    PauseSchedulerReqInput,
    ProfileReq,
    ProfileReqType,
    SessionParams,
    TokenizedGenerateReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from tokenspeed.runtime.epd.encode_worker import EncodeRequest
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

FIXTURE_DIR = Path(__file__).resolve().parent


# The frontend's handshake frames, map-encoded as its `rmp_serde::to_vec_named`
# codec writes them.
class WireReadyMessage(msgspec.Struct):
    """Engine -> frontend handshake status frame (``status`` = HELLO | READY)."""

    status: str | None = None
    local: bool | None = None
    headless: bool | None = None
    parallel_config_hash: str | None = None


class WireHandshakeAddresses(msgspec.Struct):
    """Frontend-owned data-plane addresses delivered to the engine in INIT."""

    inputs: list[str] = []
    outputs: list[str] = []
    coordinator_input: str | None = None
    coordinator_output: str | None = None
    frontend_stats_publish_address: str | None = None


class WireHandshakeInitMessage(msgspec.Struct):
    """Frontend -> engine INIT payload (sent in reply to HELLO)."""

    addresses: WireHandshakeAddresses
    parallel_config: dict = {}


_ENCODER = MsgpackEncoder()


def _add(request: TokenizedGenerateReqInput) -> list[bytes]:
    return [zmq_wire.REQ_TYPE_ADD, *(bytes(f) for f in _ENCODER.encode(request))]


def _generate_minimal() -> list[bytes]:
    # Unnormalized defaults: the decoder must not invent normalized values.
    return _add(
        TokenizedGenerateReqInput(
            rid="r-min", input_ids=[1, 2, 3], sampling_params=SamplingParams()
        )
    )


def _generate_text() -> list[bytes]:
    # Every field gets a distinct non-default value: a decoder that swapped
    # two fields of one type would still re-encode byte for byte, so only
    # per-field value checks catch it. The combination need not be a
    # request the engine would accept.
    sampling_params = SamplingParams(
        max_new_tokens=16,
        stop=["</s>", "\n\n"],
        stop_token_ids=[7],
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        min_p=0.05,
        frequency_penalty=0.1,
        presence_penalty=-0.2,
        repetition_penalty=1.1,
        min_new_tokens=2,
        json_schema='{"type": "object"}',
        regex="[a-z]+",
        ebnf='root ::= "x"',
        structural_tag='{"tag": 1}',
        ignore_eos=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        no_stop_trim=True,
        thinking_budget=128,
        custom_params={"k": [1, "v", None, 2.5], "flag": True},
        stream_interval=4,
        logit_bias={"42": -1.5},
        logprobs=0,
        n=3,
    )
    sampling_params.resolve_seed("r-text")
    sampling_params.normalize(tokenizer=None)
    return _add(
        TokenizedGenerateReqInput(
            rid="r-text",
            http_worker_ipc="ipc:///tmp/worker",
            input_text="héllo wörld",
            input_ids=[151644, 872, 198, 0, 300000],
            sampling_params=sampling_params,
            return_logprob=True,
            logprob_start_len=3,
            top_logprobs_num=2,
            token_ids_logprob=[5, 6],
            stream=True,
            input_embeds=[[0.25, -0.5]],
            session_params=SessionParams(id="s1", rid="r0", offset=3, replace=False),
            custom_logit_processor="processor",
            return_hidden_states=True,
            created_time=1.5,
            bootstrap_host="10.0.0.1",
            bootstrap_port=8998,
            bootstrap_room=2**40 + 5,
            input_multi_ids=[[1, 2], [3]],
            input_extra_infos=[{"a": 1, "b": [-1, -200, 70000]}],
            input_ids_unpadded=[151644, 872, 198],
            validation_error="rejected",
            data_parallel_rank=1,
        )
    )


def _generate_multimodal() -> list[bytes]:
    image = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=2**64 - 3,
        pad_value=151700,
        offsets=[(0, 4), (10, 14)],
        # 32 bytes: inlined as an ext raw view.
        feature=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        model_specific_data={
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64)
        },
        encode_handshake={
            "bootstrap_room": 7,
            "bootstrap_host": "encoder-0",
            "bootstrap_port": 9001,
        },
    )
    video = MultimodalDataItem(
        modality=Modality.VIDEO,
        hash=11,
        pad_value=151701,
        offsets=[(20, 52)],
        # 512 bytes: moved to an out-of-band frame.
        feature=torch.ones(16, 16, dtype=torch.bfloat16),
        # Engine-local on real traffic; set here so field order is checked.
        encoded=torch.zeros(1, 2, dtype=torch.float16),
        encoded_deepstack=torch.zeros(1, 3, dtype=torch.float16),
    )
    audio = MultimodalDataItem(
        modality=Modality.AUDIO,
        feature_shm=ShmTensorHandle(
            shm_name="/ts_shm_0", shape=(3, 4), dtype=torch.float16
        ),
    )
    multimodal_inputs = MultimodalInputs(
        mm_items=[image, video, audio],
        im_token_id=151655,
        video_token_id=151656,
        # 480 bytes: a second out-of-band frame.
        mrope_positions=torch.arange(60, dtype=torch.int64).reshape(3, 20),
        mrope_position_delta=torch.tensor([[-3]], dtype=torch.int64),
        mrope_position_delta_scalar=-3,
        mrope_position_delta_repeated_cache=torch.tensor([5], dtype=torch.int32),
    )
    sampling_params = SamplingParams(max_new_tokens=8, temperature=0.0)
    sampling_params.resolve_seed("r-mm")
    sampling_params.normalize(tokenizer=None)
    return _add(
        TokenizedGenerateReqInput(
            rid="r-mm",
            input_ids=list(range(60)),
            sampling_params=sampling_params,
            stream=True,
            multimodal_inputs=multimodal_inputs,
        )
    )


def _encode_request() -> list[bytes]:
    request = EncodeRequest(
        request_id="e-1",
        bootstrap_host="10.0.0.5",
        bootstrap_port=9001,
        # Rooms are random 64-bit values.
        bootstrap_room=2**63 + 5,
        items=[
            MultimodalDataItem(
                modality=Modality.IMAGE,
                hash=2**64 - 3,
                pad_value=151700,
                offsets=[(3, 66), (70, 73)],
                # 512 bytes: moved to an out-of-band frame.
                feature=torch.arange(128, dtype=torch.float32).reshape(8, 16),
                model_specific_data={
                    "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.int64)
                },
            )
        ],
    )
    return [bytes(f) for f in _ENCODER.encode(request)]


def _abort() -> list[bytes]:
    return [zmq_wire.REQ_TYPE_ABORT, msgspec.msgpack.encode(["r1", "r2"])]


def _control(request) -> list[bytes]:
    return [zmq_wire.REQ_TYPE_CONTROL, zmq_wire.encode_control(request)]


def _control_pause() -> list[bytes]:
    return _control(PauseSchedulerReqInput(mode="wait"))


def _control_profile() -> list[bytes]:
    return _control(
        ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir="/traces",
            start_step=2,
            num_steps=5,
            activities=["CPU", "GPU"],
            profile_by_stage=True,
            with_stack=False,
            record_shapes=True,
            profile_id="p1",
        )
    )


def _control_update_distributed() -> list[bytes]:
    return _control(
        UpdateWeightsFromDistributedReqInput(
            names=["w.a", "w.b"],
            dtype_names=["bfloat16", "float32"],
            shapes=[[2, 3], [4]],
            group_name="g",
            flush_cache=False,
            weight_version=None,
        )
    )


def _control_output_flush() -> list[bytes]:
    return [zmq_wire.encode_control(FlushCacheReqOutput(success=True))]


def _control_output_internal_state() -> list[bytes]:
    return [
        zmq_wire.encode_control(
            GetInternalStateReqOutput(
                internal_state={"max_num_seqs": 16, "nested": {"a": [1.5]}}
            )
        )
    ]


def _output_slim() -> list[bytes]:
    batch = BatchTokenIDOutSlim(
        rids=["a", "b", "c"],
        output_ids=[[1, 2], [], [300000]],
        finished_reasons=["", "stop", "abort"],
        prompt_tokens=[5, 6, 70000],
        completion_tokens=[2, 9, 1],
        cached_tokens=[0, 4, 65536],
        output_token_logprobs_val=[[-0.5, -1.25], [], [-0.0]],
        output_token_logprobs_idx=[[11, 12], [], [300001]],
        engine_index=3,
        num_running=17,
        num_waiting=2,
        kv_active_pages=1000,
        kv_total_pages=4096,
    )
    return [bytes(f) for f in _ENCODER.encode(batch)]


def _output_slim_tail() -> list[bytes]:
    # The columns a frontend renders beyond the tokens: finish details,
    # speculation, prompt logprobs and top-k alternatives.
    batch = BatchTokenIDOutSlim(
        rids=["a", "b"],
        output_ids=[[7], [8]],
        finished_reasons=["stop", "abort"],
        prompt_tokens=[3, 2],
        completion_tokens=[1, 1],
        cached_tokens=[0, 0],
        output_token_logprobs_val=[[-0.25], []],
        output_token_logprobs_idx=[[7], []],
        finish_reason_details=[
            {"type": "stop", "matched": 7},
            {
                "type": "abort",
                "message": "Invalid request: too many",
                "err_type": 522,
                "status_code": 400,
            },
        ],
        spec_verify_ct=[2, 0],
        accept_draft_tokens=[1.5, None],
        input_token_logprobs_val=[[None, -1.0, -2.5], []],
        input_token_logprobs_idx=[[1, 2, 3], []],
        input_top_logprobs_val=[[None, [-0.5, -1.0], [-0.75, -2.5]], []],
        input_top_logprobs_idx=[[None, [9, 2], [4, 3]], []],
        output_top_logprobs_val=[[[-0.25, -3.0]], []],
        output_top_logprobs_idx=[[[7, 1]], []],
    )
    return [bytes(f) for f in _ENCODER.encode(batch)]


def _output_slim_prefix9() -> list[bytes]:
    # An older producer: the tag plus the eight required fields, without the
    # engine index and load tail. Decoders fill the tail with zeros.
    return [
        msgspec.msgpack.encode(
            [
                "BatchTokenIDOutSlim",
                ["a"],
                [[5]],
                ["length"],
                [3],
                [1],
                [0],
                [[]],
                [[]],
            ]
        )
    ]


def _handshake_hello() -> list[bytes]:
    return [
        zmq_wire.encode(
            WireReadyMessage(
                status="HELLO", local=True, headless=False, parallel_config_hash="abc"
            )
        )
    ]


def _handshake_init() -> list[bytes]:
    return [
        zmq_wire.encode(
            WireHandshakeInitMessage(
                addresses=WireHandshakeAddresses(
                    inputs=["tcp://127.0.0.1:21000"],
                    outputs=["tcp://127.0.0.1:21001"],
                    coordinator_input="tcp://127.0.0.1:21003",
                    coordinator_output="tcp://127.0.0.1:21004",
                    frontend_stats_publish_address="tcp://127.0.0.1:21002",
                ),
                parallel_config={"data_parallel_size": 2, "nested": {"x": [1.5]}},
            )
        )
    ]


def _handshake_ready_response() -> list[bytes]:
    return [
        zmq_wire.encode(
            zmq_wire.WireEngineCoreReadyResponse(
                max_model_len=32768,
                num_gpu_blocks=12345,
                prefix_granularity=64,
                dp_stats_address="tcp://127.0.0.1:21005",
                dtype="float16",
                multimodal_encoder_dtype="bfloat16",
                vllm_version="tokenspeed-0.0.0",
                world_size=8,
                data_parallel_size=2,
                tensor_parallel_size=4,
                pipeline_parallel_size=3,
                decode_context_parallel_size=5,
                data_parallel_rank=1,
                max_num_seqs=256,
                max_num_batched_tokens=8192,
                instance_id="model-a",
                kv_cache_size_tokens=790080,
                kv_cache_max_concurrency=24.1,
                kv_events_config={"publisher": "zmq", "topic": ""},
            )
        )
    ]


FIXTURES = {
    "generate_minimal": _generate_minimal,
    "generate_text": _generate_text,
    "generate_multimodal": _generate_multimodal,
    "encode_request": _encode_request,
    "abort": _abort,
    "control_pause": _control_pause,
    "control_profile": _control_profile,
    "control_update_distributed": _control_update_distributed,
    "control_output_flush": _control_output_flush,
    "control_output_internal_state": _control_output_internal_state,
    "output_slim": _output_slim,
    "output_slim_tail": _output_slim_tail,
    "output_slim_prefix9": _output_slim_prefix9,
    "handshake_hello": _handshake_hello,
    "handshake_init": _handshake_init,
    "handshake_ready_response": _handshake_ready_response,
}


def build_fixtures() -> dict[str, str]:
    """Return each fixture file's expected content, keyed by file name."""
    return {
        f"{name}.hex": "".join(f"{frame.hex()}\n" for frame in build())
        for name, build in FIXTURES.items()
    }


def read_frames(file_name: str) -> list[bytes]:
    """The frames stored in a fixture file."""
    text = (FIXTURE_DIR / file_name).read_text()
    return [bytes.fromhex(line) for line in text.splitlines()]


def main() -> None:
    for file_name, content in build_fixtures().items():
        (FIXTURE_DIR / file_name).write_text(content)
        print(f"wrote {file_name}")


if __name__ == "__main__":
    main()
