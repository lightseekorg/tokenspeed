import json
from types import SimpleNamespace

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.quantization import get_quantization_config
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config


def test_modelopt_sidecar_augments_config_json_quantization(tmp_path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt"},
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_quant_algo": "FP8",
                    "quantized_layers": {
                        "backbone.layers.1.mixer.experts.0.up_proj": {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                        },
                    },
                },
            }
        )
    )

    config_json_quant = {
        "config_groups": {
            "fp4": {
                "input_activations": {"type": "float", "num_bits": 4},
                "weights": {"type": "float", "num_bits": 4, "group_size": 16},
                "targets": ["backbone.layers.1.mixer.experts.0.up_proj"],
            },
            "fp8": {
                "input_activations": {"type": "float", "num_bits": 8},
                "weights": {"type": "float", "num_bits": 8},
                "targets": ["backbone.layers.0.mixer.in_proj"],
            },
        },
    }
    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(quantization_config=config_json_quant)
    model_config.model_path = str(tmp_path)
    model_config.revision = None

    parsed = model_config._parse_quant_hf_config()

    assert parsed["quant_method"] == "modelopt_mixed"
    assert parsed["quant_algo"] == "MIXED_PRECISION"
    assert parsed["kv_cache_quant_algo"] == "FP8"
    assert parsed["config_groups"] is config_json_quant["config_groups"]
    assert "kv_cache_quant_algo" not in config_json_quant

    nvfp4_config = Nvfp4Config.from_config(parsed)
    assert nvfp4_config.kv_cache_quant_algo == "FP8"
    assert nvfp4_config.is_layer_nvfp4("model.layers.1.mixer.experts.0.up_proj")
    assert nvfp4_config.is_layer_fp8("model.layers.0.mixer.in_proj")
    assert Nvfp4Config.override_quantization_method(parsed, "nvfp4") == "modelopt_mixed"
    assert get_quantization_config("modelopt_mixed") is Nvfp4Config
