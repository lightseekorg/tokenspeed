from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.qwen3 import Qwen3Attention


def test_attention_uses_layer_cache_group():
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention", "sliding_attention"],
    )

    attention = Qwen3Attention(
        config=config,
        mapping=Mapping(rank=0, world_size=1),
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        layer_id=1,
        rms_norm_eps=config.rms_norm_eps,
    )

    assert attention.attn.group_id == "sliding_attention"
