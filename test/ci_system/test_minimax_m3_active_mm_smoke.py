import json
from pathlib import Path

import pytest
from minimax_m3_active_mm_smoke import (
    ENCODER_GRAPH_BUDGETS,
    ActiveMMSmokeConfig,
    parse_args,
    run,
    summarize_encoder_graph_log,
)


def _graph_log() -> str:
    lines = []
    budgets = ", ".join(str(value) for value in ENCODER_GRAPH_BUDGETS)
    for rank in range(4):
        lines.append(
            "EncoderCudaGraphWrapper initialized: "
            f"modality=image, budgets=[{budgets}], max_batch_size=10, "
            "max_metadata_sequences_per_batch=encoder_output_token_budget, encoder_tp=4"
        )
        for budget in ENCODER_GRAPH_BUDGETS:
            lines.append(
                f"Captured encoder cudagraph: modality=image, budget={budget}, "
                "max_batch_size=10, metadata_sequence_budget=1, buffers={}"
            )
        lines.append(
            "Encoder CUDA graph capture complete: modality=image, 9 budget graphs."
        )
        lines.append(
            "Installed encoder CUDA graphs for MiniMaxM3ForConditionalGeneration: "
            "['image_encoder']"
        )
    return "\n".join(lines) + "\n"


def _reference() -> dict:
    return {
        "prompt": "What animal is in this image? Reply with one word.",
        "input_contract": {"input_ids": [1, 254]},
        "reference_logits": {
            "top_tokens": ["Dog"],
            "top_logprobs": [-0.001004667836241424],
        },
    }


def test_graph_summary_requires_four_ranks_and_every_budget() -> None:
    summary = summarize_encoder_graph_log(_graph_log())
    recaptured = summarize_encoder_graph_log(
        _graph_log() + "Captured encoder cudagraph: modality=image, budget=16, "
        "max_batch_size=10, metadata_sequence_budget=1, buffers={}\n"
    )

    assert summary["failures"] == []
    assert summary["initialized"] == 4
    assert summary["captured_total"] == 36
    assert all(value == 4 for value in summary["captures_by_budget"].values())
    assert recaptured["captured_total"] == 37
    assert recaptured["failures"]


class _FakeResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body


def _chat_body(content, *, prompt_tokens=None, logprob=None):
    choice = {"message": {"content": content}}
    if logprob is not None:
        choice["logprobs"] = {
            "content": [{"token": content, "logprob": logprob, "top_logprobs": []}]
        }
    body = {"choices": [choice]}
    if prompt_tokens is not None:
        body["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": 1}
    return body


class _FakeSession:
    def __init__(self, *, bad_video_header: bool = False) -> None:
        video_code = "wrong" if bad_video_header else "invalid_multimodal_request"
        self.responses = [
            _FakeResponse(200, _chat_body("text path ok")),
            _FakeResponse(200, _chat_body("2")),
            _FakeResponse(200, _chat_body("TokenSpeed")),
            _FakeResponse(200, _chat_body("Pug")),
            _FakeResponse(
                200,
                _chat_body("Dog", prompt_tokens=254, logprob=-0.001),
            ),
            _FakeResponse(
                400,
                {
                    "error": {
                        "code": "invalid_multimodal_request",
                        "message": (
                            "Invalid multimodal request: modality video is not supported "
                            "by model spec minimax_m3_vl"
                        ),
                    }
                },
                headers={"X-SMG-Error-Code": video_code},
            ),
        ]
        self.payloads = []

    def post(self, url, *, json, timeout):
        assert url.endswith("/v1/chat/completions")
        assert timeout == 30
        self.payloads.append(json)
        return self.responses.pop(0)

    def get(self, _url, *, timeout):
        assert timeout == 30
        return _FakeResponse(200, {})


def _config(tmp_path: Path) -> ActiveMMSmokeConfig:
    fixtures = {}
    for name, suffix in (("dog", ".jpg"), ("pug", ".jpg"), ("banner", ".png")):
        path = tmp_path / f"{name}{suffix}"
        path.write_bytes(f"fixture-{name}".encode())
        fixtures[name] = path
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(_reference()))
    server_log = tmp_path / "server.log"
    server_log.write_text(_graph_log())
    return ActiveMMSmokeConfig(
        base_url="http://gateway",
        control_url="http://control",
        model="minimax-m3",
        dog=fixtures["dog"],
        pug=fixtures["pug"],
        banner=fixtures["banner"],
        reference=reference,
        server_log=server_log,
        output_dir=tmp_path / "artifacts",
        request_timeout_seconds=30,
    )


def test_six_request_contract_passes_without_recapture(tmp_path: Path) -> None:
    config = _config(tmp_path)
    session = _FakeSession()

    result = run(config, session=session)  # type: ignore[arg-type]

    assert result["ok"] is True
    assert result["request_order"] == [
        "text",
        "two_image",
        "banner",
        "pug",
        "dog",
        "video",
    ]
    assert result["encoder_graph"]["counts_unchanged"] is True
    assert session.payloads[0]["messages"][0]["content"][0]["type"] == "text"
    assert [item["type"] for item in session.payloads[1]["messages"][0]["content"]] == [
        "image_url",
        "image_url",
        "text",
    ]
    assert session.payloads[4]["max_tokens"] == 1
    assert session.payloads[4]["logprobs"] is True
    assert session.payloads[5]["messages"][0]["content"][0]["type"] == "video_url"
    assert "data:image" not in (config.output_dir / "validation.json").read_text()


def test_video_header_contract_is_blocking(tmp_path: Path) -> None:
    result = run(
        _config(tmp_path),
        session=_FakeSession(bad_video_header=True),  # type: ignore[arg-type]
    )

    assert result["ok"] is False
    assert result["cases"]["video"]["header_error_code"] == "wrong"
    assert any("video error header" in failure for failure in result["failures"])


def test_request_time_recapture_is_blocking(tmp_path: Path) -> None:
    config = _config(tmp_path)
    session = _FakeSession()
    original_post = session.post

    def post_and_recapture(url, *, json, timeout):
        response = original_post(url, json=json, timeout=timeout)
        if len(session.payloads) == 2:
            config.server_log.write_text(
                config.server_log.read_text()
                + "Captured encoder cudagraph: modality=image, budget=16, "
                "max_batch_size=10, metadata_sequence_budget=1, buffers={}\n"
            )
        return response

    session.post = post_and_recapture  # type: ignore[method-assign]
    result = run(config, session=session)  # type: ignore[arg-type]

    assert result["ok"] is False
    assert result["encoder_graph"]["counts_unchanged"] is False
    assert any("counts changed" in failure for failure in result["failures"])


def test_all_fixture_and_reference_paths_are_required() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--base-url",
                "http://gateway",
                "--control-url",
                "http://control",
                "--model",
                "minimax-m3",
            ]
        )
