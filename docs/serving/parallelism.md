# Parallelism

TokenSpeed exposes familiar `--tensor-parallel-size` and `--tp` entry points
plus additional split parallelism controls for attention, dense, and MoE layers.

## Quick Start

Use this form when the same tensor-parallel group is acceptable for the model:

```bash
tokenspeed serve <model> \
  --tensor-parallel-size 8
```

`--tensor-parallel-size` maps to TokenSpeed attention tensor parallelism and
cannot be used together with `--attn-tp-size`.

## Split Parallelism

Use split knobs when different layer families need different process groups:

```bash
tokenspeed serve <model> \
  --world-size 8 \
  --attn-tp-size 4 \
  --dense-tp-size 4 \
  --moe-tp-size 4
```

| Parameter | Use |
| --- | --- |
| `--world-size` | Total worker processes across all nodes. |
| `--nprocs-per-node` | Worker processes launched on each node. |
| `--attn-tp-size` | Attention tensor parallel size. |
| `--dense-tp-size` | Dense layer tensor parallel size. |
| `--moe-tp-size` | MoE layer tensor parallel size. |
| `--data-parallel-size` | Replicated data-parallel groups. |
| `--enable-expert-parallel` | Expert parallelism across the selected world size. |
| `--expert-parallel-size` | Explicit expert parallel size. |

## MoE Deployments

Large MoE models usually choose one of these shapes:

- TP only: simplest startup path, often best for smaller MoE checkpoints.
- TP + EP: tensor parallelism within a replica, expert parallelism across ranks.
- DP + EP: multiple replicated decode groups with experts distributed inside each group.

Start with the recipe closest to your model family, then tune:

- `--tensor-parallel-size` or split TP values
- `--enable-expert-parallel`
- `--moe-backend`
- `--all2all-backend`
- `--deepep-mode`

## Multi-Node

Set these explicitly:

```bash
tokenspeed serve <model> \
  --nnodes 2 \
  --node-rank 0 \
  --nprocs-per-node 8 \
  --world-size 16 \
  --dist-init-addr <rank0-host>:25000
```

Each node must use the same model, backend, precision, and scheduler settings.
Only `--node-rank` should differ between nodes.

## Validation

Before benchmarking:

- verify every rank starts and joins the distributed group
- verify the API responds before sending load
- confirm GPU visibility and process placement
- compare output correctness before tuning throughput
- keep the full launch command with benchmark results
