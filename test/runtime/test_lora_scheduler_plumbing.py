"""LoRA plumbing between the Python front-end and the C++ scheduler.

The scheduling behaviour itself is covered by the C++ suite
(tests/cpp/test_lora_namespace.cpp); these tests only pin the translation at
the boundary, where the front-end's ``None``-for-base-model convention meets
the scheduler's empty-string one.
"""

from __future__ import annotations

from tokenspeed.runtime.engine.scheduler_utils import make_config, make_spec

BASE_CONFIG_KWARGS = dict(
    num_device_pages=8,
    max_scheduled_tokens=64,
    max_batch_size=4,
    page_size=2,
    num_host_pages=8,
    disable_l2_cache=True,
    enable_l3_storage=False,
    role="fused",
)


def test_make_spec_forwards_lora_id():
    spec = make_spec("r1", [1, 2, 3], lora_id="adapter-a")
    assert spec.lora_id == "adapter-a"


def test_make_spec_defaults_to_base_model():
    # Callers that predate LoRA must keep producing base-model requests.
    assert make_spec("r1", [1, 2, 3]).lora_id == ""


def test_make_spec_maps_none_to_empty_string():
    # The front-end spells "no adapter" as None; the scheduler spells it "".
    # An accidental "None" string here would silently create an adapter
    # namespace shared by every base-model request.
    assert make_spec("r1", [1, 2, 3], lora_id=None).lora_id == ""


def test_make_config_forwards_max_loras():
    assert make_config(**BASE_CONFIG_KWARGS, max_loras=3).max_loras == 3


def test_make_config_defaults_max_loras_to_disabled():
    assert make_config(**BASE_CONFIG_KWARGS).max_loras == 0
