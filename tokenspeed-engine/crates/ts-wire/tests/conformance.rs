// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Conformance against messages written by the Python encoder.
//!
//! `tests/fixtures/generate.py` writes each fixture from the real Python
//! structs. Every fixture must decode to exactly the values the generator
//! used (compared as whole structs, so a field read into the wrong slot
//! fails) and re-encode to the same bytes, frame for frame.

use bytes::Bytes;
use ts_wire::{
    control::{
        FlushCacheReqOutput, GetInternalStateReqOutput, PauseSchedulerReqInput, ProfileReq,
        UpdateWeightsFromDistributedReqInput, PROFILE_START,
    },
    decode_frames, encode_frames,
    handshake::STATUS_HELLO,
    BatchTokenIDOutSlim, ControlOutput, ControlRequest, Dict, EncodeRequest,
    EngineCoreReadyResponse, EngineOutput, EngineRequest, HandshakeAddresses, HandshakeInitMessage,
    Modality, MultimodalDataItem, MultimodalInputs, ReadyMessage, SamplingParams, SessionParams,
    ShmTensorHandle, StrOrList, Tensor, TokenizedGenerateReqInput, Value, Wire,
};

/// A fixture's ZMQ frames: one hex-encoded frame per line.
fn fixture(name: &str) -> Vec<Bytes> {
    let path = format!("{}/tests/fixtures/{name}.hex", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.lines()
        .map(|line| {
            let bytes = (0..line.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&line[i..i + 2], 16).expect("hex"))
                .collect::<Vec<u8>>();
            Bytes::from(bytes)
        })
        .collect()
}

/// Decode an ADD message, check it against `expected`, and check that it
/// re-encodes to the fixture's frames.
fn check_add(name: &str, expected: TokenizedGenerateReqInput) {
    let frames = fixture(name);
    let decoded = EngineRequest::from_frames(&frames).unwrap();
    assert_eq!(decoded, EngineRequest::Add(Box::new(expected)));
    assert_eq!(decoded.to_frames(), frames, "{name} re-encodes differently");
}

/// Decode a single-message fixture, check it, and check the re-encoding.
fn check<T: Wire + std::fmt::Debug + PartialEq>(name: &str, expected: T) {
    let frames = fixture(name);
    assert_eq!(decode_frames::<T>(&frames).unwrap(), expected);
    assert_eq!(
        encode_frames(&expected),
        frames,
        "{name} re-encodes differently"
    );
}

fn s(value: &str) -> String {
    value.to_owned()
}

fn int(value: i64) -> Value {
    Value::from(value)
}

fn tensor(dtype: &str, shape: &[i64], data: Vec<u8>) -> Tensor {
    Tensor {
        dtype: s(dtype),
        shape: shape.to_vec(),
        data: Bytes::from(data),
    }
}

fn le_bytes<const N: usize, T>(
    values: impl IntoIterator<Item = T>,
    to_le: fn(T) -> [u8; N],
) -> Vec<u8> {
    values.into_iter().flat_map(to_le).collect()
}

#[test]
fn generate_minimal() {
    // The Python constructor turns `top_k = -1` into the disabled sentinel;
    // everything else is the unnormalized default.
    let sampling_params = SamplingParams {
        top_k: 1 << 30,
        ..Default::default()
    };
    check_add(
        "generate_minimal",
        TokenizedGenerateReqInput {
            input_ids: Some(vec![1, 2, 3]),
            ..TokenizedGenerateReqInput::new("r-min", sampling_params)
        },
    );
}

#[test]
fn generate_text() {
    let stops = Some(StrOrList::Many(vec![s("</s>"), s("\n\n")]));
    let sampling_params = SamplingParams {
        max_new_tokens: Some(16),
        stop: stops.clone(),
        stop_token_ids: Some(vec![7]),
        temperature: 0.7,
        top_p: 0.9,
        top_k: 20,
        min_p: 0.05,
        frequency_penalty: 0.1,
        presence_penalty: -0.2,
        repetition_penalty: 1.1,
        min_new_tokens: 2,
        json_schema: Some(s(r#"{"type": "object"}"#)),
        regex: Some(s("[a-z]+")),
        ebnf: Some(s(r#"root ::= "x""#)),
        structural_tag: Some(s(r#"{"tag": 1}"#)),
        ignore_eos: true,
        skip_special_tokens: false,
        spaces_between_special_tokens: false,
        no_stop_trim: true,
        thinking_budget: Some(128),
        custom_params: Some(Dict(vec![
            (
                s("k"),
                Value::Array(vec![int(1), "v".into(), Value::Nil, 2.5.into()]),
            ),
            (s("flag"), true.into()),
        ])),
        stream_interval: Some(4),
        logit_bias: Some(Dict(vec![(s("42"), -1.5)])),
        // resolve_seed derives a missing seed from the rid: crc32("r-text").
        seed: Some(3568737799),
        logprobs: Some(0),
        n: 3,
        stop_strs: stops,
        stop_str_max_len: 4,
        is_normalized: true,
    };
    check_add(
        "generate_text",
        TokenizedGenerateReqInput {
            rid: Some(StrOrList::One(s("r-text"))),
            http_worker_ipc: Some(s("ipc:///tmp/worker")),
            input_text: Some(s("héllo wörld")),
            input_ids: Some(vec![151644, 872, 198, 0, 300000]),
            sampling_params,
            return_logprob: true,
            logprob_start_len: 3,
            top_logprobs_num: 2,
            token_ids_logprob: Some(vec![5, 6]),
            stream: true,
            input_embeds: Some(vec![Value::Array(vec![0.25.into(), (-0.5).into()])]),
            session_params: Some(SessionParams {
                id: Some(s("s1")),
                rid: Some(s("r0")),
                offset: Some(3),
                replace: Some(false),
            }),
            custom_logit_processor: Some(s("processor")),
            return_hidden_states: true,
            created_time: 1.5,
            bootstrap_host: Some(s("10.0.0.1")),
            bootstrap_port: Some(8998),
            bootstrap_room: Some((1 << 40) + 5),
            input_multi_ids: Some(vec![vec![1, 2], vec![3]]),
            input_extra_infos: Some(vec![Dict(vec![
                ("a".into(), int(1)),
                (
                    "b".into(),
                    Value::Array(vec![int(-1), int(-200), int(70000)]),
                ),
            ])]),
            input_ids_unpadded: Some(vec![151644, 872, 198]),
            multimodal_inputs: None,
            validation_error: Some(s("rejected")),
            data_parallel_rank: Some(1),
        },
    );
}

#[test]
fn generate_multimodal() {
    let item = |modality| MultimodalDataItem {
        modality,
        hash: None,
        pad_value: None,
        offsets: None,
        feature: None,
        feature_shm: None,
        model_specific_data: Dict::default(),
        encoded: None,
        encoded_deepstack: None,
        encode_handshake: None,
    };
    let image = MultimodalDataItem {
        hash: Some(u64::MAX - 2),
        pad_value: Some(151700),
        offsets: Some(vec![(0, 4), (10, 14)]),
        feature: Some(tensor(
            "float32",
            &[2, 4],
            le_bytes((0..8).map(|v| v as f32), f32::to_le_bytes),
        )),
        model_specific_data: Dict(vec![(
            s("image_grid_thw"),
            tensor("int64", &[1, 3], le_bytes([1i64, 2, 2], i64::to_le_bytes)),
        )]),
        encode_handshake: Some(Dict(vec![
            ("bootstrap_room".into(), int(7)),
            ("bootstrap_host".into(), "encoder-0".into()),
            ("bootstrap_port".into(), int(9001)),
        ])),
        ..item(Modality::Image)
    };
    let video = MultimodalDataItem {
        hash: Some(11),
        pad_value: Some(151701),
        offsets: Some(vec![(20, 52)]),
        // bfloat16 1.0 is 0x3f80.
        feature: Some(tensor("bfloat16", &[16, 16], [0x80, 0x3f].repeat(256))),
        encoded: Some(tensor("float16", &[1, 2], vec![0; 4])),
        encoded_deepstack: Some(tensor("float16", &[1, 3], vec![0; 6])),
        ..item(Modality::Video)
    };
    let audio = MultimodalDataItem {
        feature_shm: Some(ShmTensorHandle {
            shm_name: s("/ts_shm_0"),
            shape: vec![3, 4],
            dtype: s("float16"),
        }),
        ..item(Modality::Audio)
    };
    // `temperature=0.0` normalizes to greedy: top_k 1 with the seed pinned.
    let sampling_params = SamplingParams {
        max_new_tokens: Some(8),
        top_k: 1,
        seed: Some(0),
        stop_strs: Some(StrOrList::Many(Vec::new())),
        is_normalized: true,
        ..Default::default()
    };
    let expected = TokenizedGenerateReqInput {
        input_ids: Some((0..60).collect()),
        stream: true,
        multimodal_inputs: Some(MultimodalInputs {
            mm_items: vec![image, video, audio],
            im_token_id: Some(151655),
            video_token_id: Some(151656),
            mrope_positions: Some(tensor(
                "int64",
                &[3, 20],
                le_bytes(0..60i64, i64::to_le_bytes),
            )),
            mrope_position_delta: Some(tensor(
                "int64",
                &[1, 1],
                le_bytes([-3i64], i64::to_le_bytes),
            )),
            mrope_position_delta_scalar: Some(-3),
            mrope_position_delta_repeated_cache: Some(tensor(
                "int32",
                &[1],
                le_bytes([5i32], i32::to_le_bytes),
            )),
        }),
        ..TokenizedGenerateReqInput::new("r-mm", sampling_params)
    };
    check_add("generate_multimodal", expected);

    // The two tensors of at least 256 bytes travel as their own frames.
    let frames = fixture("generate_multimodal");
    let sizes: Vec<_> = frames.iter().map(Bytes::len).collect();
    assert_eq!(sizes[2..], [512, 480]);
}

#[test]
fn encode_request() {
    let expected = EncodeRequest {
        rid: None,
        http_worker_ipc: None,
        request_id: s("e-1"),
        bootstrap_host: s("10.0.0.5"),
        bootstrap_port: 9001,
        bootstrap_room: (1 << 63) + 5,
        items: vec![MultimodalDataItem {
            modality: Modality::Image,
            hash: Some(u64::MAX - 2),
            pad_value: Some(151700),
            offsets: Some(vec![(3, 66), (70, 73)]),
            feature: Some(tensor(
                "float32",
                &[8, 16],
                le_bytes((0..128).map(|v| v as f32), f32::to_le_bytes),
            )),
            feature_shm: None,
            model_specific_data: Dict(vec![(
                s("image_grid_thw"),
                tensor("int64", &[1, 3], le_bytes([1i64, 4, 4], i64::to_le_bytes)),
            )]),
            encoded: None,
            encoded_deepstack: None,
            encode_handshake: None,
        }],
    };
    check("encode_request", expected);
    // The feature travels as its own frame.
    let sizes: Vec<_> = fixture("encode_request").iter().map(Bytes::len).collect();
    assert_eq!(sizes[1..], [512]);
}

#[test]
fn abort() {
    let frames = fixture("abort");
    let decoded = EngineRequest::from_frames(&frames).unwrap();
    assert_eq!(decoded, EngineRequest::Abort(vec![s("r1"), s("r2")]));
    assert_eq!(decoded.to_frames(), frames);
}

/// Decode a CONTROL message, check it, and check the re-encoding.
fn check_control(name: &str, expected: ControlRequest) {
    let frames = fixture(name);
    let decoded = EngineRequest::from_frames(&frames).unwrap();
    assert_eq!(decoded, EngineRequest::Control(Box::new(expected)));
    assert_eq!(decoded.to_frames(), frames, "{name} re-encodes differently");
}

/// Decode a control reply as the output socket carries it.
fn check_control_output(name: &str, expected: ControlOutput) {
    let frames = fixture(name);
    let decoded = EngineOutput::from_frames(&frames).unwrap();
    assert_eq!(decoded, EngineOutput::Control(Box::new(expected)));
    assert_eq!(decoded.to_frames(), frames, "{name} re-encodes differently");
}

#[test]
fn control_requests() {
    check_control(
        "control_pause",
        ControlRequest::PauseScheduler(PauseSchedulerReqInput {
            rid: None,
            http_worker_ipc: None,
            mode: s("wait"),
        }),
    );
    check_control(
        "control_profile",
        ControlRequest::Profile(ProfileReq {
            rid: None,
            http_worker_ipc: None,
            kind: PROFILE_START,
            output_dir: Some(s("/traces")),
            start_step: Some(2),
            num_steps: Some(5),
            activities: Some(vec![s("CPU"), s("GPU")]),
            profile_by_stage: true,
            with_stack: Some(false),
            record_shapes: Some(true),
            profile_id: Some(s("p1")),
        }),
    );
    check_control(
        "control_update_distributed",
        ControlRequest::UpdateWeightsFromDistributed(UpdateWeightsFromDistributedReqInput {
            rid: None,
            http_worker_ipc: None,
            names: vec![s("w.a"), s("w.b")],
            dtype_names: vec![s("bfloat16"), s("float32")],
            shapes: vec![vec![2, 3], vec![4]],
            group_name: s("g"),
            flush_cache: false,
            weight_version: None,
        }),
    );
}

#[test]
fn control_outputs() {
    check_control_output(
        "control_output_flush",
        ControlOutput::FlushCache(FlushCacheReqOutput {
            rid: None,
            http_worker_ipc: None,
            success: true,
        }),
    );
    check_control_output(
        "control_output_internal_state",
        ControlOutput::GetInternalState(GetInternalStateReqOutput {
            rid: None,
            http_worker_ipc: None,
            internal_state: Dict(vec![
                (Value::from("max_num_seqs"), int(16)),
                (
                    Value::from("nested"),
                    Value::Map(vec![(
                        Value::from("a"),
                        Value::Array(vec![Value::Float(1.5)]),
                    )]),
                ),
            ]),
        }),
    );
}

#[test]
fn output_slim() {
    let expected = BatchTokenIDOutSlim {
        rids: Some(vec![s("a"), s("b"), s("c")]),
        output_ids: vec![vec![1, 2], vec![], vec![300000]],
        finished_reasons: vec![s(""), s("stop"), s("abort")],
        prompt_tokens: vec![5, 6, 70000],
        completion_tokens: vec![2, 9, 1],
        cached_tokens: vec![0, 4, 65536],
        output_token_logprobs_val: vec![vec![-0.5, -1.25], vec![], vec![-0.0]],
        output_token_logprobs_idx: vec![vec![11, 12], vec![], vec![300001]],
        engine_index: 3,
        num_running: 17,
        num_waiting: 2,
        kv_active_pages: 1000,
        kv_total_pages: 4096,
        ..no_tail()
    };
    let frames = fixture("output_slim");
    assert_eq!(
        EngineOutput::from_frames(&frames).unwrap(),
        EngineOutput::Batch(Box::new(expected.clone()))
    );
    assert_eq!(
        EngineOutput::Batch(Box::new(expected.clone())).to_frames(),
        frames
    );

    let rows: Vec<_> = expected.requests().unwrap().collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(
        (rows[0].rid, rows[0].output_ids, rows[0].finish_reason),
        ("a", &[1, 2][..], None)
    );
    assert_eq!(rows[0].output_token_logprobs, [-0.5, -1.25]);
    assert_eq!(
        (rows[1].finish_reason, rows[1].completion_tokens),
        (Some("stop"), 9)
    );
    assert_eq!(
        (rows[2].finish_reason, rows[2].cached_tokens),
        (Some("abort"), 65536)
    );
}

#[test]
fn output_from_an_older_producer_defaults_the_load_tail() {
    let batch: BatchTokenIDOutSlim = decode_frames(&fixture("output_slim_prefix9")).unwrap();
    assert_eq!(
        batch,
        BatchTokenIDOutSlim {
            rids: Some(vec![s("a")]),
            output_ids: vec![vec![5]],
            finished_reasons: vec![s("length")],
            prompt_tokens: vec![3],
            completion_tokens: vec![1],
            cached_tokens: vec![0],
            output_token_logprobs_val: vec![vec![]],
            output_token_logprobs_idx: vec![vec![]],
            engine_index: 0,
            num_running: 0,
            num_waiting: 0,
            kv_active_pages: 0,
            kv_total_pages: 0,
            ..no_tail()
        }
    );
}

/// A batch whose required columns are empty and whose tail is the default.
fn no_tail() -> BatchTokenIDOutSlim {
    BatchTokenIDOutSlim {
        rids: None,
        output_ids: vec![],
        finished_reasons: vec![],
        prompt_tokens: vec![],
        completion_tokens: vec![],
        cached_tokens: vec![],
        output_token_logprobs_val: vec![],
        output_token_logprobs_idx: vec![],
        engine_index: 0,
        num_running: 0,
        num_waiting: 0,
        kv_active_pages: 0,
        kv_total_pages: 0,
        finish_reason_details: vec![],
        spec_verify_ct: vec![],
        accept_draft_tokens: vec![],
        input_token_logprobs_val: vec![],
        input_token_logprobs_idx: vec![],
        input_top_logprobs_val: vec![],
        input_top_logprobs_idx: vec![],
        output_top_logprobs_val: vec![],
        output_top_logprobs_idx: vec![],
    }
}

#[test]
fn output_slim_tail() {
    let detail = |entries: Vec<(&str, Value)>| {
        Dict(
            entries
                .into_iter()
                .map(|(key, value)| (s(key), value))
                .collect(),
        )
    };
    let expected = BatchTokenIDOutSlim {
        rids: Some(vec![s("a"), s("b")]),
        output_ids: vec![vec![7], vec![8]],
        finished_reasons: vec![s("stop"), s("abort")],
        prompt_tokens: vec![3, 2],
        completion_tokens: vec![1, 1],
        cached_tokens: vec![0, 0],
        output_token_logprobs_val: vec![vec![-0.25], vec![]],
        output_token_logprobs_idx: vec![vec![7], vec![]],
        finish_reason_details: vec![
            Some(detail(vec![
                ("type", Value::from("stop")),
                ("matched", Value::from(7)),
            ])),
            Some(detail(vec![
                ("type", Value::from("abort")),
                ("message", Value::from("Invalid request: too many")),
                ("err_type", Value::from(522)),
                ("status_code", Value::from(400)),
            ])),
        ],
        spec_verify_ct: vec![2, 0],
        accept_draft_tokens: vec![Some(1.5), None],
        input_token_logprobs_val: vec![vec![None, Some(-1.0), Some(-2.5)], vec![]],
        input_token_logprobs_idx: vec![vec![1, 2, 3], vec![]],
        input_top_logprobs_val: vec![
            vec![None, Some(vec![-0.5, -1.0]), Some(vec![-0.75, -2.5])],
            vec![],
        ],
        input_top_logprobs_idx: vec![vec![None, Some(vec![9, 2]), Some(vec![4, 3])], vec![]],
        output_top_logprobs_val: vec![vec![vec![-0.25, -3.0]], vec![]],
        output_top_logprobs_idx: vec![vec![vec![7, 1]], vec![]],
        ..no_tail()
    };
    let frames = fixture("output_slim_tail");
    assert_eq!(
        EngineOutput::from_frames(&frames).unwrap(),
        EngineOutput::Batch(Box::new(expected.clone()))
    );
    assert_eq!(
        EngineOutput::Batch(Box::new(expected.clone())).to_frames(),
        frames
    );
    let rows: Vec<_> = expected.requests().unwrap().collect();
    assert_eq!(rows[0].spec_verify_ct, Some(2));
    assert_eq!(rows[0].accept_draft_tokens, Some(1.5));
    assert_eq!(
        rows[0].input_token_logprobs,
        Some((&[None, Some(-1.0), Some(-2.5)][..], &[1, 2, 3][..]))
    );
    assert!(rows[1].input_token_logprobs.is_none() && rows[1].output_top_logprobs.is_none());
    assert_eq!(
        rows[1].finish_details.map(|details| details.0.len()),
        Some(4)
    );
}

#[test]
fn ragged_output_columns_are_rejected() {
    let mut batch: BatchTokenIDOutSlim = decode_frames(&fixture("output_slim")).unwrap();
    batch.cached_tokens.pop();
    let error = batch.requests().err().unwrap();
    assert!(error.to_string().contains("cached_tokens"), "{error}");
}

#[test]
fn engine_dead_sentinel() {
    let frames = EngineOutput::Dead.to_frames();
    assert_eq!(frames, [Bytes::from_static(b"ENGINE_CORE_DEAD")]);
    assert_eq!(
        EngineOutput::from_frames(&frames).unwrap(),
        EngineOutput::Dead
    );
}

#[test]
fn handshake_hello() {
    check(
        "handshake_hello",
        ReadyMessage {
            status: Some(s(STATUS_HELLO)),
            local: Some(true),
            headless: Some(false),
            parallel_config_hash: Some(s("abc")),
        },
    );
}

#[test]
fn handshake_init() {
    check(
        "handshake_init",
        HandshakeInitMessage {
            addresses: HandshakeAddresses {
                inputs: vec![s("tcp://127.0.0.1:21000")],
                outputs: vec![s("tcp://127.0.0.1:21001")],
                coordinator_input: Some(s("tcp://127.0.0.1:21003")),
                coordinator_output: Some(s("tcp://127.0.0.1:21004")),
                frontend_stats_publish_address: Some(s("tcp://127.0.0.1:21002")),
            },
            parallel_config: Dict(vec![
                ("data_parallel_size".into(), int(2)),
                (
                    "nested".into(),
                    Value::Map(vec![("x".into(), Value::Array(vec![1.5.into()]))]),
                ),
            ]),
        },
    );
}

#[test]
fn handshake_ready_response() {
    check(
        "handshake_ready_response",
        EngineCoreReadyResponse {
            max_model_len: 32768,
            num_gpu_blocks: 12345,
            prefix_granularity: 64,
            dp_stats_address: Some(s("tcp://127.0.0.1:21005")),
            dtype: s("float16"),
            multimodal_encoder_dtype: Some(s("bfloat16")),
            vllm_version: s("tokenspeed-0.0.0"),
            world_size: 8,
            data_parallel_size: 2,
            tensor_parallel_size: 4,
            pipeline_parallel_size: 3,
            decode_context_parallel_size: 5,
            data_parallel_rank: 1,
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            instance_id: s("model-a"),
            kv_cache_size_tokens: Some(790080),
            kv_cache_max_concurrency: Some(24.1),
            kv_events_config: Some(Dict(vec![
                ("publisher".into(), "zmq".into()),
                ("topic".into(), "".into()),
            ])),
        },
    );
}
