# 03. Parallel Strategy and Placement

## Core Claim Under Evaluation

The important question is not "does TokenSpeed support TP/DP/EP." Most modern serving stacks do.

The more important question is:

> Can TokenSpeed express attention, dense, MoE, latent-KV, DP, TP, EP, and possibly CP as different execution domains, then safely move hidden/residual tensors between those domains?

This is the potential moat candidate for V4/Kimi-like MoE models. Attention, dense layers, routed MoE, shared experts, latent-KV, and all-to-all traffic do not have the same optimal parallel shape.

## Evidence Already Read

### Split parallelism controls

Docs and runtime arguments expose separate controls such as:

- `--attn-tp-size`
- `--dense-tp-size`
- `--moe-tp-size`
- `--data-parallel-size`
- expert parallel enablement

This is only the surface. The more relevant code path is how these choices enter model execution.

### `Mapping`

`Mapping` is the topology object that makes parallel choices available inside runtime/model code.

Important source:

- `python/tokenspeed/runtime/distributed/mapping.py`

The current analysis treats `Mapping` as the first step from "launch arguments" to "model-structure-aware execution plan."

### `CommManager`

DeepSeek V4 currently appears to rely heavily on model-specific `CommManager` wiring.

Important source:

- `python/tokenspeed/runtime/distributed/comm_manager.py`
- `python/tokenspeed/runtime/distributed/comm_ops.py`
- `python/tokenspeed/runtime/models/deepseek_v4.py`

The current working hypothesis is:

```text
DeepSeek V4 actual path = model-specific CommManager more than generic placement compiler.
```

That means any report must be careful:

- do not claim V4 is fully compilerized;
- do analyze how `CommManager` handles attention/dense/MoE domain transitions;
- verify whether this path can be adapted to Ascend/HCCL without losing the intended strategy.

## Placement Compiler

TokenSpeed also has a base decoder-layer placement compiler:

- `python/tokenspeed/runtime/models/base/placement.py`
- `python/tokenspeed/runtime/models/base/compiler.py`
- `python/tokenspeed/runtime/models/base/comm_ops.py`

The compiler-level idea is:

```text
ModuleSpec(input placement, output placement)
  -> compiler tracks hidden/residual placement
  -> inserts collective ops at module boundaries
  -> returns a compiled step list
```

This is not full-system SPMD. It is a lightweight decoder-layer compiler for hidden/residual placement transitions across attention and MLP/MoE domains.

Observed concepts:

- `REPLICATE`
- `SHARD`
- `PARTIAL`
- attention TP group
- dense TP group
- MoE TP/EP group
- inserted `AllGather`, `ReduceScatter`, `AllReduce`, residual gather/slice, fused reduce-norm style ops

## Relationship Between Parallel Strategy and local-SPMD

The correct positioning is:

```text
parallel strategy = what to split, per layer family
local-SPMD / placement compiler = how to safely materialize those splits in the model graph
```

So local-SPMD is not an independent moat by itself. It is a force multiplier for the parallel strategy moat. If V4/Kimi requires rapid trial of different attention/dense/MoE placements, compilerized placement can reduce manual communication wiring and correctness risk.

## Current Limitations and Cautions

1. **DeepSeek V4 is not clearly on the generic placement compiler path**
   The read code points to explicit `CommManager` usage. This reduces the strength of any claim that TokenSpeed has a mature compilerized V4 path.

2. **MoE TP + EP may not be fully free**
   Prior reading found constraints around MoE TP and EP combinations in the DeepSeek V4 / MegaMoE path. This needs exact re-verification before making final claims.

3. **DP/CP are not necessarily first-class placement compiler dimensions**
   `Mapping` may encode DP/CP, but the base placement compiler appears focused on hidden/residual transitions across TP-like groups.

4. **Performance value must be measured in communication counters**
   The abstraction only matters if it reduces unnecessary collectives, improves token-aware communication, or enables a better MoE execution plan.

## What a Deep Analysis Still Needs

The next code-reading pass should build a token movement ledger for DeepSeek V4:

| Layer Family | What to Read | Questions |
|---|---|---|
| Attention / latent-KV | `models/deepseek_v4.py`, `layers/attention/backends/deepseek_v4.py`, `layers/deepseek_v4_mhc.py` | How are attention TP, latent-KV layout, compressed states, and output projection coordinated? |
| Dense / shared experts | `distributed/comm_manager.py`, linear layers in model path | Where are all-reduce / reduce-scatter / all-gather inserted? |
| Routed MoE | `layers/moe/`, `tokenspeed-kernel/ops/moe/`, DeepEP dispatchers | How are tokens permuted, routed, combined, and synchronized? |
| Expert placement | `runtime/moe/expert_location.py`, EPLB algorithms | Is expert location dynamic/static, and how does it interact with DP/EP groups? |
| Placement compiler | `models/base/compiler.py` | Can the generic path express the V4 path, or is it mainly a separate framework? |

## Copyability

| Capability | Copy Difficulty | Notes |
|---|---:|---|
| expose split TP flags | low-medium | Config surface only |
| model-specific communication wiring | medium-high | Requires model patch and backend collectives |
| generic placement contract | high | Requires model abstraction and correctness validation |
| token-aware MoE communication ledger | high | Requires dispatcher/backend/runtime integration |
| architecture-level parallel-domain compiler | high | Would touch model graph, comm ops, scheduling assumptions |

## Current Research Posture

Parallel strategy remains a core moat candidate, but the current evidence is not yet as strong as Scheduler/KV. The next step is not another high-level summary. It is a source-level execution plan analysis for DeepSeek V4:

```text
input hidden placement
  -> attention placement
  -> attention output placement
  -> dense/MoE expected placement
  -> inserted communication
  -> routed token movement
  -> residual/final norm placement
```

