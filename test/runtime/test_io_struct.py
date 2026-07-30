import pytest

from tokenspeed.runtime.engine.io_struct import GenerateReqInput


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"text": "prompt", "input_ids": [1]},
        {"text": "prompt", "input_embeds": [[0.0]]},
        {"input_ids": [1], "input_embeds": [[0.0]]},
    ],
)
def test_multiple_prompt_sources_are_rejected(request_kwargs):
    with pytest.raises(
        ValueError, match="Exactly one of text, input_ids, or input_embeds"
    ):
        GenerateReqInput(**request_kwargs).normalize_batch_and_arguments()
