"""Test fixtures for the Inkling model: dummy checkpoint directories.

The tiny variant is fully synthetic (safe to commit). The full-size variant
reads the confidential reference ``config.json``/tokenizer from the directory
named by the ``INKLING_REF_DIR`` env var at runtime and is never embedded here.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

# Tiny synthetic config: 6 layers with the real 5:1 SWA/full pattern
# (local = all but layer 5) and asymmetric KV heads like the real model
# (full layers 2 heads, SWA layers 4 heads -> uniform 4 after replication).
TINY_TEXT_CONFIG = {
    "model_type": "inkling_model",
    "vocab_size": 2048,
    "unpadded_vocab_size": 2000,
    "hidden_size": 256,
    "intermediate_size": 64,
    "dense_intermediate_size": 256,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "d_rel": 8,
    "rel_extent": 64,
    "local_layer_ids": [0, 1, 2, 3, 4],
    "sliding_window_size": 32,
    "swa_num_attention_heads": 8,
    "swa_num_key_value_heads": 4,
    "swa_head_dim": 32,
    "rms_norm_eps": 1e-6,
    "use_embed_norm": True,
    "use_sconv": True,
    "sconv_kernel_size": 4,
    "dense_mlp_idx": 2,
    "n_routed_experts": 8,
    "n_shared_experts": 2,
    "num_experts_per_tok": 2,
    "route_scale": 8.0,
    "use_gate_bias": True,
    # The real checkpoints always ship gate.global_scale / mlp.global_scale
    # and the fused gate kernel now requires it, so the tiny fixture matches.
    "use_global_scale": True,
    "norm_after_topk": True,
    "gate_activation": "sigmoid",
    "shared_expert_sink": True,
    "inference_moe_w13_interleaved": True,
    "log_scaling_n_floor": None,
    "log_scaling_alpha": 0.1,
    "logits_mup_width_multiplier": 4.0,
    "model_max_length": 4096,
}

TINY_MM_CONFIG = {
    "model_type": "inkling_mm_model",
    "architectures": ["InklingForConditionalGeneration"],
    "text_config": TINY_TEXT_CONFIG,
    "audio_config": {"model_type": "inkling_audio_model"},
    "vision_config": {"model_type": "inkling_vision_model"},
    "eos_token_id": 1,
    "dtype": "bfloat16",
}

# Tiny audio/vision tower configs (decoder_dmodel set => towers ON). Kept out
# of TINY_MM_CONFIG so existing text-only tests exercise the towers-absent
# path; make_inkling_dummy_checkpoint(mm_towers=True) merges them in.
TINY_AUDIO_TOWER_CONFIG = {
    "model_type": "inkling_audio_model",
    "decoder_dmodel": TINY_TEXT_CONFIG["hidden_size"],
    "n_mel_bins": 8,
    "mel_vocab_size": 4,
    "dmel_min_value": -1.5,
    "dmel_max_value": 2.0,
    "use_audio_norm": True,
    "audio_mode": "dmel",
}
TINY_VISION_TOWER_CONFIG = {
    "model_type": "inkling_vision_model",
    "vision_encoder_type": "hmlp",
    "decoder_dmodel": TINY_TEXT_CONFIG["hidden_size"],
    "patch_size": 4,
    "temporal_patch_size": 1,
    "n_channels": 3,
    "n_layers": 1,
    "use_vision_norm": True,
}
# Placeholder ids the gateway expands per media item. Any real tokenizer id
# works engine-side (the engine consumes pre-expanded runs + offsets); keep
# them below the unpadded vocab (2000) — ids in the padded tail are masked to
# -inf in the model's logits and make confusing fixtures.
TINY_IMAGE_PLACEHOLDER_TOKEN_ID = 1998
TINY_AUDIO_PLACEHOLDER_TOKEN_ID = 1999

TINY_MM_TOWERS_CONFIG = {
    **TINY_MM_CONFIG,
    "audio_config": TINY_AUDIO_TOWER_CONFIG,
    "vision_config": TINY_VISION_TOWER_CONFIG,
    "image_placeholder_token_id": TINY_IMAGE_PLACEHOLDER_TOKEN_ID,
    "audio_placeholder_token_id": TINY_AUDIO_PLACEHOLDER_TOKEN_ID,
}


def _write_synthetic_tokenizer(target_dir: Path, vocab_size: int) -> None:
    """Write a minimal byte-level BPE tokenizer trained in-process.

    Dummy-weight generations are meaningless text anyway; the tokenizer only
    needs to round-trip prompts and cover ids < vocab_size.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|pad|>", "<|eos|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "hello world, this is a tiny tokenizer fixture",
        "0123456789 !@#$%^&*()",
    ] * 50
    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.save(str(target_dir / "tokenizer.json"))
    (target_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "<|eos|>",
                "pad_token": "<|pad|>",
                "model_max_length": TINY_TEXT_CONFIG["model_max_length"],
            },
            indent=2,
        )
    )


def make_inkling_dummy_checkpoint(
    target_dir: str | Path, *, tiny: bool = True, mm_towers: bool = False
) -> Path:
    """Create a weightless Inkling checkpoint dir usable with --load-format dummy.

    Args:
        target_dir: Directory to create (parents included).
        tiny: If True, write the committed synthetic tiny config + synthetic
            tokenizer. If False, copy the confidential full-size config.json
            and tokenizer files from ``$INKLING_REF_DIR`` (raises if unset and the
            default path is absent).
        mm_towers: Tiny variant only — if True, enable the audio/vision
            towers (``decoder_dmodel`` set + placeholder token ids). Default
            False keeps the towers off so text-only tests are unaffected.

    Returns:
        The checkpoint directory path.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    if tiny:
        config = TINY_MM_TOWERS_CONFIG if mm_towers else TINY_MM_CONFIG
        (target / "config.json").write_text(json.dumps(config, indent=2))
        _write_synthetic_tokenizer(target, TINY_TEXT_CONFIG["vocab_size"])
    else:
        ref_env = os.environ.get("INKLING_REF_DIR")
        if not ref_env or not (Path(ref_env) / "config.json").exists():
            raise FileNotFoundError(
                "Full-size Inkling fixture needs INKLING_REF_DIR set to a "
                f"directory holding the reference config.json (got {ref_env!r}); "
                "the confidential reference files are never committed."
            )
        ref_dir = Path(ref_env)
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            src = ref_dir / name
            if src.exists():
                shutil.copy(src, target / name)
    return target
