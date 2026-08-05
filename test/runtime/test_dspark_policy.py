import pytest

from tokenspeed.runtime.execution.dspark_policy import route_kimi_k3_dspark


@pytest.mark.parametrize("input_tokens", [32768, 131072])
def test_long_context_routes_to_no_spec(input_tokens: int) -> None:
    route = route_kimi_k3_dspark(
        input_tokens=input_tokens,
        max_new_tokens=1024,
        concurrency=16,
        workload="reasoning",
    )
    assert route.mode == "no-spec"


def test_short_generation_routes_to_w4() -> None:
    route = route_kimi_k3_dspark(
        input_tokens=1024,
        max_new_tokens=512,
        concurrency=1,
    )
    assert route.mode == "w4"


def test_batched_aime_routes_to_w8() -> None:
    route = route_kimi_k3_dspark(
        input_tokens=512,
        max_new_tokens=32768,
        concurrency=16,
        workload="aime",
    )
    assert route.mode == "w8"


def test_uncalibrated_confidence_cannot_override_validated_route() -> None:
    route = route_kimi_k3_dspark(
        input_tokens=1024,
        max_new_tokens=512,
        concurrency=1,
        confidence=0.01,
        confidence_calibrated=False,
    )
    assert route.mode == "w4"


def test_calibrated_low_confidence_routes_to_no_spec() -> None:
    route = route_kimi_k3_dspark(
        input_tokens=1024,
        max_new_tokens=512,
        concurrency=1,
        confidence=0.1,
        confidence_calibrated=True,
    )
    assert route.mode == "no-spec"
