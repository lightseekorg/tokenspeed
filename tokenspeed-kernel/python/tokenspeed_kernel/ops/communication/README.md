# Communication operations

## Ordinary producer-direct Iris all-reduce

TP8 BF16 payloads from 96 KiB through 16 MiB use a register-based reduce-scatter
and pull gather. Each rank reduces its contiguous element partition, then pulls
the eight reduced partitions into caller-owned output storage. Inputs remain
intact. The path requires partitions aligned to eight BF16 elements; other
shapes, dtypes, group sizes and larger payloads retain the existing kernels.
The operation reuses the prepared input, one-partition scratch and 84-row flag
array. It adds no symmetric allocation and does not overwrite the borrowed
attention/MoE result.

The 1–16 MiB range uses the same kernel with up to 84 workgroups. This covers
the ordinary attention and paired MoE reductions between the small fused
collectives and the token-sharded prefill operations, including uneven rows.
Above 16 MiB the original two-stage kernel remains selected: the measured
advantage of the register reduction shrinks as peer bandwidth dominates.

Each workgroup has one subgroup and processes 512 elements per tile. Up to 84
workgroups run; partitions requiring 85–128 tiles use 64 workgroups. The eight
peer loads are issued together and summed with the existing even/odd FP32 tree
before rounding to BF16. The gather batches its peer loads and output stores
as well. Both payload phases use system-scope ".cv" loads, which bypass
non-coherent caches while retaining the last-level cache's temporal policy.
The inspected CDNA4 instructions are 16-byte buffer loads with `sc0 sc1`.
Scratch stores use ".wt"; the gather writes ordinary owned output with ".wb".
The single-subgroup kernel needs no workgroup barriers or shared-memory
transpose. Peer payload reads and writes remain batched without intermediate
waits between peers.

Calls use the same stream-ordered state as the existing producer-direct kernels.
Each workgroup exclusively owns its diagonal epoch counter. It loads that
counter at entry, publishes entry and completion into non-diagonal incoming
flags, and stores the advanced counter before returning. Polls compare signed
32-bit differences, including across wraparound. Entry uses release/acquire
ordering; completion drains all subgroup stores before publishing and acquiring
the peer flags. Completion also joins input readers before a subsequent producer
can overwrite the input. The next scratch writer's entry rendezvous protects
the preceding gather, including when ordinary and prefill kernels alternate.

Validation includes the ordinary TP2/4/8 correctness tests and
`test_iris_all_reduce_interleaved_protocols`, which delays producers while mixing
one-stage, register, original two-stage and Lamport reductions. It checks inputs
and retained outputs in eager and graph execution across epoch rollover.

## Iris attention prefill and MoE storage

`iris_attention_prefill_mix` reduces a BF16 attention projection and performs
AttnRes plus output RMSNorm on token shards. It returns an owned residual shard
`[M/8, 7168]` and a borrowed, replicated activation `[M, 7168]`. Its contract
requires an existing eight-rank CDNA4 Iris group, a prepared producer buffer,
positive `M` divisible by eight, and up to eleven history snapshots. Epsilon,
history count and group are explicit inputs. Unsupported calls return `None`
before allocation or collective publication.

Kimi K3 enables the operation for aligned prefills from 512 through 8192 tokens,
matching attention and MoE TP8 groups, EP1, PP1, and an unbiased, unquantized BF16
output projection. The model writes its projection directly into prepared Iris
storage through the public GEMM operation. In the same configuration, 16–511-token
prefills use ordinary producer-direct Iris all-reduce and the registered AttnRes
mixer. Their residual stays replicated, avoiding the extra gather required below
the MoE sharding window. The ordinary reduction returns owned local storage and
preserves the symmetric input; a block-write layer can retain the result across
subsequent producers. Other configurations retain their existing collective and
AttnRes selection. No scheduler or cache metadata changes are needed; these
buffers hold transient model activations, not request state.

Uneven prefills through 8192 tokens also use the ordinary producer-direct
reduction and retain a replicated residual. The attention and MoE sharding
operations still require a token count divisible by eight; ordinary Iris
partitions elements instead, so the 7168-wide projection supports every row
count in this window without padding or another symmetric buffer.

### Sequence and numerics

Each rank pulls its contiguous token partition from all eight producers. Loads
from the peer buffers are issued together, then reduced with the existing Iris
even/odd FP32 tree. The sum rounds to BF16 before the optional replicated residual
is added and rounded again. Only the owned local residual is written in this
phase; the symmetric projection remains intact.

The local rows mix the replicated history snapshots and the new residual using
the shared gfx950 AttnRes implementation. The mixed value rounds to BF16 before
output RMSNorm and its final BF16 conversion. The normalized rows are pushed to
the same token partition on every rank. Launch selection fuses mixing with the
push where that avoids overhead without excessive register pressure; other
shapes use the registered AttnRes kernel followed by a separate push gather.
For 512–1016 and 4096–8192 tokens, mixing and gathering fuse through six history
snapshots. At 7680 tokens and above they also fuse seven or eight snapshots.
The 1024–4088 range and other history depths use the separate mixer.

The reduction uses 24 workgroups, four subgroups per workgroup and 2048-element
tiles. The fused mixer uses up to 128 workgroups and four or eight subgroups; the
separate gather uses 32 workgroups. Small partitions cap the grids to useful
work. These launches fit the flags already reserved for the MoE tail.

### Ownership and synchronization

The push reuses `IrisAllReduce._moe_tail_output_buf`, which already holds
112 MiB per rank at 8192 tokens. Attention adds no symmetric allocation or flag
storage. Its owned residual needs 14 MiB per rank at that size and can survive
the next reuse of Iris input and scratch. The borrowed activation is valid until
the next attention prefill mix or MoE tail on the group. A caller retaining it
must clone it.

The previous MoE result may exactly alias attention's replicated residual. Each
token owner finishes reading its residual rows before pushing replacements for
those rows. Shifted overlaps, history/weight aliases with collective storage,
and a sharded MoE residual aliasing the result buffer are rejected. The MoE
receives `prefix_is_sharded` explicitly: it consumes local residual rows directly,
or gathers them before a fallback requiring replicated rows. A declined
attention mix uses ordinary producer-direct Iris reduction, which preserves the
symmetric projection and returns owned storage for the retained residual.

All calls sharing the state must be ordered on one stream. Side-stream consumers
of the borrowed activation must join before the next result overwrite. The
runtime's MoE producer fork joins before its tail runs. Graph replay advances
device-side epochs, including signed and unsigned wraparound, rather than
capturing a host counter. Ordinary Iris reductions also poll the actual peer
flags and compare signed 32-bit differences, so a zero or negative wrapped
epoch cannot skip a producer and a peer in the next stage can be recognized.

The reduce-scatter entry uses system-scope release/acquire flags. Acquire lowering
provides the workgroup rendezvous, so it needs no additional barrier. The final
push drains each subgroup's stores and explicitly joins the workgroup before
publishing completion. Every rank waits for all peer completions. This also
orders all ranks' earlier projection reads before the next producer can reuse
the input, so the reduce-scatter needs no exit rendezvous. No barriers occur
between the eight payload loads or between the eight payload stores.

Peer payload loads use `.cg`, peer stores use `.wt`, and flag polling uses
volatile `.cv` loads. On CDNA4 the inspected payload instructions use `sc0 nt`
for loads and `sc0 sc1` for stores. The publication and acquire protocol provides
visibility; a cache hint alone does not synchronize ranks.

### Validation

On an idle eight-GPU CDNA4 host, run:

```sh
python -m pytest -q tokenspeed-kernel/test/amd/ops/test_iris_attention_prefill.py \
  tokenspeed-kernel/test/amd/ops/test_iris_communication.py::test_iris_all_reduce_correctness_world8 \
  tokenspeed-kernel/test/amd/ops/test_iris_communication.py::test_iris_all_reduce_epoch_rollover \
  tokenspeed-kernel/test/amd/ops/test_iris_moe_tail.py \
  tokenspeed-kernel/test/amd/ops/test_kimi3_prefill_gluon_amd.py
```

The tests cover tail and maximum sizes, history depths, an independent FP64
mixing oracle, exact Iris residual reduction, rejected aliases, borrowed-buffer
lifetime, collective interleaving, a real MoE tail, and capture/replay across
epoch rollover. Ordinary one-stage and two-stage reductions also delay a
producer across entry and intermediate-stage wraps. Runtime tests exercise
automatic eligibility and ownership-safe fallbacks through the Kimi decoder
and communication interfaces.
