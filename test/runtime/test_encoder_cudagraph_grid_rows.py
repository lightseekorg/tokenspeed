import torch

from tokenspeed.runtime.models.qwen3_vision import (
    Qwen3VLMoeVisionModel,
)
from tokenspeed.runtime.multimodal.encoder_cudagraph import (
    EncoderCudaGraphWrapper,
    VisionEncoderBatch,
    VisionEncoderCudaGraphAdapter,
)


def test_vision_encoder_batch_selects_single_and_contiguous_items():
    tokens = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    grid = torch.tensor([[1, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=torch.int32)
    grid_rows = [[1, 1, 1], [1, 2, 1], [1, 1, 2]]
    batch = VisionEncoderBatch(tokens, grid, out_div=1, grid_rows=grid_rows)

    assert batch.encoder_output_tokens == [1, 2, 2]
    contiguous = batch.select([0, 1])
    assert torch.equal(contiguous.tokens, tokens[:3])
    assert torch.equal(contiguous.grid, grid[:2])
    assert contiguous.grid_rows == [[1, 1, 1], [1, 2, 1]]

    single = batch.select([2])
    assert torch.equal(single.tokens, tokens[3:5])
    assert torch.equal(single.grid, grid[2:3])
    assert single.grid_rows == [[1, 1, 2]]


def test_vision_encoder_batch_selects_noncontiguous_items():
    tokens = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    grid = torch.tensor(
        [[1, 1, 1], [1, 2, 1], [1, 3, 1], [1, 4, 1]],
        dtype=torch.int32,
    )
    grid_rows = grid.tolist()
    batch = VisionEncoderBatch(tokens, grid, out_div=1, grid_rows=grid_rows)

    selected = batch.select([2, 0])

    assert torch.equal(selected.tokens, torch.cat([tokens[3:6], tokens[:1]], dim=0))
    assert torch.equal(selected.grid, grid[[2, 0]])
    assert selected.grid_rows == [grid_rows[2], grid_rows[0]]


def test_qwen3_vision_rot_pos_emb_repeats_uniform_rows_exactly():
    class Rotary:
        cos_sin_cache = torch.empty((1, 2), dtype=torch.float32)

    class DummyTower:
        device = torch.device("cpu")
        rotary_pos_emb = Rotary()

        def _rot_pos_emb_one(self, *_args):
            return (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            )

    cos, sin = Qwen3VLMoeVisionModel.rot_pos_emb(
        DummyTower(),
        [[1, 2, 2], [1, 2, 2]],
    )

    assert torch.equal(
        cos,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]]),
    )
    assert torch.equal(
        sin,
        torch.tensor([[5.0, 6.0], [7.0, 8.0], [5.0, 6.0], [7.0, 8.0]]),
    )


def test_qwen3_vision_pos_embed_repeats_uniform_rows_exactly():
    class PosEmbed:
        embedding_dim = 2

    class DummyTower:
        num_grid_per_side = 8
        spatial_merge_size = 2
        pos_embed = PosEmbed()
        device = torch.device("cpu")
        dtype = torch.float32

        def _pos_embed_interpolate_one(self, *_args):
            return torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    tower = DummyTower()

    out = Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list(
        tower,
        [[1, 2, 2], [1, 2, 2]],
    )

    assert torch.equal(
        out,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]]),
    )


def test_vision_encoder_adapter_passes_precomputed_grid_rows_to_metadata():
    tokens = torch.zeros((2, 1), dtype=torch.float32)
    grid = torch.tensor([[1, 2, 1]], dtype=torch.int32)
    grid_rows = [[1, 2, 1]]

    class DummyTower:
        def __init__(self):
            self.seen_grid = None

        def prepare_metadata(self, grid_arg):
            self.seen_grid = grid_arg
            return {}

    tower = DummyTower()

    def pre_encode(_items):
        return tokens, grid, grid_rows

    adapter = VisionEncoderCudaGraphAdapter(
        tower=tower,
        pre_encode=pre_encode,
        post_encode=lambda encoder_outs, _grid: torch.cat(encoder_outs, dim=0),
        out_div=1,
        merge=1,
        input_feature_shape=(1,),
    )

    batch = adapter.batch_from_items([object()])
    adapter.prepare_metadata(
        batch, encoder_output_token_budget=None, metadata_sequence_budget=1
    )

    assert tower.seen_grid is grid_rows


def test_vision_encoder_adapter_can_skip_replay_cu_seqlens_padding():
    tokens = torch.zeros((2, 1), dtype=torch.float32)
    grid = torch.tensor([[1, 2, 1]], dtype=torch.int32)

    class DummyTower:
        def prepare_metadata(self, _grid_arg):
            return {"cu_seqlens": torch.tensor([0, 2], dtype=torch.int32)}

    adapter = VisionEncoderCudaGraphAdapter(
        tower=DummyTower(),
        pre_encode=lambda _items: (tokens, grid),
        post_encode=lambda encoder_outs, _grid: torch.cat(encoder_outs, dim=0),
        out_div=1,
        merge=1,
        input_feature_shape=(1,),
    )
    batch = VisionEncoderBatch(tokens, grid, out_div=1)

    capture_metadata = adapter.prepare_metadata(
        batch, encoder_output_token_budget=4, metadata_sequence_budget=4
    )
    replay_metadata = adapter.prepare_metadata(
        batch,
        encoder_output_token_budget=4,
        metadata_sequence_budget=4,
        pad_cu_seqlens=False,
    )

    assert torch.equal(capture_metadata["cu_seqlens"], torch.tensor([0, 2, 2, 2, 2]))
    assert torch.equal(replay_metadata["cu_seqlens"], torch.tensor([0, 2]))


def test_encoder_cudagraph_copy_prefix_zero_tail_only_clears_unused_rows():
    buf = torch.full((4, 2), 9.0)
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    EncoderCudaGraphWrapper._copy_prefix_zero_tail(buf, src)

    assert torch.equal(
        buf,
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )
    exact = torch.full_like(src, 9.0)
    EncoderCudaGraphWrapper._copy_prefix_zero_tail(exact, src)
    assert torch.equal(exact, src)


def test_encoder_cudagraph_copy_prefix_repeat_last_tail_matches_padding():
    buf = torch.full((5,), -1, dtype=torch.int32)
    src = torch.tensor([0, 2], dtype=torch.int32)

    EncoderCudaGraphWrapper._copy_prefix_repeat_last_tail(buf, src)

    assert torch.equal(buf, torch.tensor([0, 2, 2, 2, 2], dtype=torch.int32))
    exact = torch.full_like(src, -1)
    EncoderCudaGraphWrapper._copy_prefix_repeat_last_tail(exact, src)
    assert torch.equal(exact, src)


def test_encoder_cudagraph_scatter_clones_once_then_scatters_views():
    output = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    dest = {}

    EncoderCudaGraphWrapper._scatter_cloned_output_slices(
        output,
        [0, 1, 2],
        [1, 2, 3],
        dest,
    )

    assert torch.equal(dest[0], output[0:1])
    assert torch.equal(dest[1], output[1:3])
    assert torch.equal(dest[2], output[3:6])
    assert dest[0].untyped_storage().data_ptr() != output.untyped_storage().data_ptr()
    assert (
        dest[0].untyped_storage().data_ptr()
        == dest[1].untyped_storage().data_ptr()
        == dest[2].untyped_storage().data_ptr()
    )
