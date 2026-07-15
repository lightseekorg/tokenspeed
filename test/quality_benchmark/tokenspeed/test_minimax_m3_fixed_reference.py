import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("minimax_m3_fixed_reference.py")
MODULE_NAME = "minimax_m3_fixed_reference"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
quality = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = quality
SPEC.loader.exec_module(quality)


def _reference_payload():
    results = []
    for prompt_index in range(5):
        input_ids = [1000 + prompt_index, 2000 + prompt_index]
        generated_ids = [prompt_index * 100 + step + 1 for step in range(8)]
        results.append(
            {
                "prompt": f"prompt {prompt_index}",
                "input_ids": input_ids,
                "generated_ids": generated_ids,
                "generated_text": "fixture",
                "steps": [
                    {
                        "token_id": token_id,
                        "top_ids": [
                            token_id,
                            token_id + 10_000,
                            token_id + 20_000,
                            token_id + 30_000,
                            token_id + 40_000,
                        ],
                        "top_logprobs": [-0.1, -0.5, -1.0, -2.0, -3.0],
                    }
                    for token_id in generated_ids
                ],
            }
        )
    return {
        "snapshot": "/models/minimax-m3",
        "dtype": "bfloat16",
        "tp_size": 4,
        "results": results,
    }


def _write_reference(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_reference_payload()))
    return path


class FakeHttpClient:
    def __init__(self, reference, kv_cache_dtype, logprob_offset=0.0):
        self.kv_cache_dtype = kv_cache_dtype
        self.logprob_offset = logprob_offset
        self.posts = []
        self.autoregressive = {
            tuple(prompt.input_ids): list(prompt.generated_ids)
            for prompt in reference.prompts
        }
        self.teacher = {}
        for prompt in reference.prompts:
            for step_index, step in enumerate(prompt.steps):
                context = (*prompt.input_ids, *prompt.generated_ids[:step_index])
                self.teacher[context] = step.token_id

    def get_json(self, url, timeout_seconds):
        assert url.endswith("/get_server_info")
        assert timeout_seconds == 7
        return {
            "server_args": {
                "enable_output_logprobs": True,
                "kv_cache_dtype": self.kv_cache_dtype,
                "port": 48100 if self.kv_cache_dtype == "auto" else 48200,
                "rl_control_port": (49100 if self.kv_cache_dtype == "auto" else 49200),
                "model": "/models/minimax-m3",
                "revision": "c5454eb0",
                "attn_tp_size": 4,
                "max_model_len": 1_048_576,
                "chunked_prefill_size": 8192,
                "block_size": 128,
                "attention_backend": "mha",
                "moe_backend": "triton",
                "sampling_backend": "greedy",
                "enable_prefix_caching": False,
                "disable_overlap_schedule": False,
                "enforce_eager": False,
                "max_cudagraph_capture_size": 160,
                "disable_prefill_graph": False,
            },
            "tokenspeed_version": "0.1.0",
        }

    def post_json(self, url, body, timeout_seconds):
        assert url.endswith("/generate")
        assert timeout_seconds == 11
        self.posts.append(deepcopy(body))
        input_ids = tuple(body["input_ids"])
        max_new_tokens = body["sampling_params"]["max_new_tokens"]
        if max_new_tokens == 1:
            output_ids = [self.teacher[input_ids]]
        else:
            output_ids = list(self.autoregressive[input_ids])
        logprobs = [
            [-0.2 - position * 0.01 + self.logprob_offset, float(token_id), None]
            for position, token_id in enumerate(output_ids)
        ]
        return {
            "output_ids": output_ids,
            "meta_info": {"output_token_logprobs": logprobs},
        }


def _collect(tmp_path, arm, dtype, reference_path, logprob_offset=0.0):
    reference = quality.load_reference(reference_path)
    client = FakeHttpClient(reference, dtype, logprob_offset)
    output_path = tmp_path / f"{arm}.json"
    config = quality.CollectConfig(
        arm=arm,
        model="minimax-m3",
        reference_path=reference_path,
        output_path=output_path,
        base_url="http://127.0.0.1:8123",
        generate_path="/generate",
        server_info_path="/get_server_info",
        server_sha="abc123",
        request_timeout_seconds=11,
        server_info_timeout_seconds=7,
        autoregressive_repeats=3,
        seed=20260715,
        server_info_base_url="http://127.0.0.1:8124",
    )
    artifact = quality.collect_arm(config, client)
    return output_path, artifact, client


def _rewrite(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_loads_fixed_reference_and_singleton_generation_response(tmp_path):
    reference_path = _write_reference(tmp_path)

    reference = quality.load_reference(reference_path)
    parsed = quality.parse_generation_response(
        [
            {
                "output_ids": [17],
                "meta_info": {"output_token_logprobs": [[-0.25, 17.0, None]]},
            }
        ]
    )

    assert len(reference.prompts) == 5
    assert sum(len(prompt.steps) for prompt in reference.prompts) == 40
    assert len(reference.sha256) == 64
    assert parsed["output_ids"] == [17]
    assert parsed["sampled_logprobs"] == [{"token_id": 17, "logprob": -0.25}]

    with pytest.raises(quality.BenchmarkError, match="not integral"):
        quality.parse_generation_response(
            {
                "output_ids": [17],
                "meta_info": {"output_token_logprobs": [[-0.25, 17.5, None]]},
            }
        )


def test_collect_checkpoints_all_requests_with_fixed_sampling_and_provenance(tmp_path):
    reference_path = _write_reference(tmp_path)

    output_path, artifact, client = _collect(tmp_path, "bf16", "auto", reference_path)
    persisted = json.loads(output_path.read_text())

    assert artifact["status"] == persisted["status"] == "complete"
    assert len(client.posts) == 5 * 3 + 40
    assert len(persisted["autoregressive"]) == 15
    assert len(persisted["teacher_forced"]) == 40
    assert persisted["server"]["sha"] == "abc123"
    assert persisted["server"]["args"]["kv_cache_dtype"] == "auto"
    assert "port" not in persisted["server"]["args"]
    assert "rl_control_port" not in persisted["server"]["args"]
    assert persisted["server"]["raw_info"]["server_args"]["port"] == 48100
    assert persisted["server"]["raw_info"]["server_args"]["rl_control_port"] == 49100
    assert persisted["reference"]["sha256"] == quality._sha256(reference_path)
    assert persisted["client_config"] == {
        "autoregressive_repeats": 3,
        "base_url": "http://127.0.0.1:8123",
        "generate_url": "http://127.0.0.1:8123/generate",
        "ignore_eos": True,
        "model": "minimax-m3",
        "request_timeout_seconds": 11,
        "seed": 20260715,
        "server_info_timeout_seconds": 7,
        "server_info_base_url": "http://127.0.0.1:8124",
        "server_info_url": "http://127.0.0.1:8124/get_server_info",
        "temperature": 0,
        "top_k": 1,
    }
    for body in client.posts:
        assert body["model"] == "minimax-m3"
        assert body["sampling_params"]["temperature"] == 0
        assert body["sampling_params"]["top_k"] == 1
        assert body["sampling_params"]["ignore_eos"] is True
        assert body["sampling_params"]["seed"] == 20260715
        assert body["return_logprob"] is True
    first_response = persisted["autoregressive"][0]["response"]
    assert first_response["output_ids"]
    assert first_response["sampled_logprobs"]
    assert first_response["output_token_logprobs_raw"]
    assert first_response["raw_body"]


def test_compare_passes_fixed_gates_and_records_both_arm_identities(tmp_path):
    reference_path = _write_reference(tmp_path)
    bf16_path, _, _ = _collect(tmp_path, "bf16", "auto", reference_path)
    fp8_path, _, _ = _collect(
        tmp_path, "fp8", "fp8_e4m3", reference_path, logprob_offset=0.05
    )
    report_path = tmp_path / "comparison.json"

    report = quality.compare_arms(bf16_path, fp8_path, report_path)

    assert report["status"] == "pass"
    assert report["gates"]["intrarm_logprob_atol"] == 1e-6
    assert report["metrics"]["teacher_matches"] == 40
    assert report["metrics"]["bf16_hf_top5"] == 40
    assert report["metrics"]["fp8_hf_top5"] == 40
    assert report["metrics"]["autoregressive_exact_prompts"] == 5
    assert report["metrics"]["autoregressive_common_prefix_tokens"] == 40
    assert report["sources"]["bf16"]["server_sha"] == "abc123"
    assert report["sources"]["fp8"]["server_sha"] == "abc123"
    assert report["sources"]["bf16"]["kv_cache_dtype"] == "auto"
    assert report["sources"]["fp8"]["kv_cache_dtype"] == "fp8_e4m3"
    assert json.loads(report_path.read_text())["status"] == "pass"


def test_compare_rejects_different_server_sha_and_wrong_cache_arm(tmp_path):
    reference_path = _write_reference(tmp_path)
    bf16_path, _, _ = _collect(tmp_path, "bf16", "auto", reference_path)
    fp8_path, fp8, _ = _collect(tmp_path, "fp8", "fp8_e4m3", reference_path)

    fp8["server"]["sha"] = "different"
    _rewrite(fp8_path, fp8)
    with pytest.raises(quality.BenchmarkError, match="same server SHA"):
        quality.compare_arms(bf16_path, fp8_path, tmp_path / "sha-report.json")

    fp8["server"]["sha"] = "abc123"
    fp8["server"]["args"]["kv_cache_dtype"] = "auto"
    _rewrite(fp8_path, fp8)
    with pytest.raises(quality.BenchmarkError, match="FP8 cache arm"):
        quality.compare_arms(bf16_path, fp8_path, tmp_path / "dtype-report.json")


def test_compare_rejects_server_config_drift_outside_cache_dtype(tmp_path):
    reference_path = _write_reference(tmp_path)
    bf16_path, _, _ = _collect(tmp_path, "bf16", "auto", reference_path)
    fp8_path, fp8, _ = _collect(tmp_path, "fp8", "fp8_e4m3", reference_path)
    fp8["server"]["args"]["max_model_len"] = 131_072
    _rewrite(fp8_path, fp8)

    with pytest.raises(
        quality.BenchmarkError, match="server args drift.*max_model_len"
    ):
        quality.compare_arms(bf16_path, fp8_path, tmp_path / "drift-report.json")


def test_compare_reports_gate_failures(tmp_path):
    reference_path = _write_reference(tmp_path)
    bf16_path, _, _ = _collect(tmp_path, "bf16", "auto", reference_path)
    fp8_path, fp8, _ = _collect(tmp_path, "fp8", "fp8_e4m3", reference_path)

    repeat = fp8["autoregressive"][1]["response"]
    repeat["sampled_logprobs"][0]["logprob"] += 2e-6
    canonical = fp8["autoregressive"][3]["response"]
    canonical["output_ids"][0] += 999
    canonical["sampled_logprobs"][0]["token_id"] = canonical["output_ids"][0]
    teacher = fp8["teacher_forced"][0]["response"]
    teacher["output_ids"][0] += 999
    teacher["sampled_logprobs"][0]["token_id"] = teacher["output_ids"][0]
    _rewrite(fp8_path, fp8)

    report = quality.compare_arms(bf16_path, fp8_path, tmp_path / "failed-report.json")
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["status"] == "fail"
    assert "fp8_intrarm_logprobs_deterministic" in failed
    assert "fp8_intrarm_ids_deterministic" in failed
    assert "fp8_teacher_tokens_in_hf_top5" in failed
    assert "teacher_mismatches_have_small_hf_margin" in failed
    assert "autoregressive_common_prefix_tokens" in failed
    assert "autoregressive_no_step0_divergence" in failed
