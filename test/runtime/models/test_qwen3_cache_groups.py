from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.qwen3 import Qwen3Attention


def _attention(config: Qwen3Config, layer_id: int) -> Qwen3Attention:
    return Qwen3Attention(
        config=config,
        mapping=Mapping(rank=0, world_size=1),
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        layer_id=layer_id,
        rms_norm_eps=config.rms_norm_eps,
    )


def test_attention_declares_visibility_and_leaves_storage_to_the_plan():
    """The layer's compute mask follows its layer_type; its cache group is
    not the model's to name (bind_cache_groups stamps it from the plan)."""
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        use_sliding_window=True,
        sliding_window=64,
        layer_types=["full_attention", "sliding_attention"],
    )

    full, sliding = _attention(config, 0).attn, _attention(config, 1).attn

    assert full.sliding_window_size == -1
    assert sliding.sliding_window_size == 63
    for layer in (full, sliding):
        try:
            layer.group_id
        except RuntimeError as exc:
            assert "no cache group bound" in str(exc)
        else:
            raise AssertionError("the model must not name its cache group")
