from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.models.qwen3_5_nextn import _resolve_mtp_quant_config


def test_unquantized_mtp_checkpoint_disables_draft_quantization():
    quant_config = Nvfp4Config(exclude_modules=["mtp.layers.0*"])

    assert _resolve_mtp_quant_config(quant_config) is None


def test_quantized_mtp_checkpoint_keeps_draft_quantization():
    quant_config = Nvfp4Config()

    assert _resolve_mtp_quant_config(quant_config) is quant_config
