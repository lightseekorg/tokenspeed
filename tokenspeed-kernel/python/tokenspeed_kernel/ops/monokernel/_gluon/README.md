# Fused decoder layers on gfx950 (Gluon)

`gluon_monokernel_muse_spark_gfx950` (`muse_spark/layer.py`) runs one Muse
Spark decoder layer as one persistent launch: 256 workgroups x 8 waves, one
workgroup per CU (each reserves 96 KB of LDS). `muse_spark/prepared.py` holds
the host side (state, weights read in place, checks, launch);
`../gluon.py` registers it.

## Roles

A workgroup's roles come from its XCD (hardware) and its slot (program id //
8), and run in order:

| Slots per XCD | Front | MoE | Tail |
|---|---|---|---|
| 0-15 | o_proj sender (SENDER); slot 0 also owns 128 columns of the all-reduce (AR_OWNER); router K part (ROUTER) | stream | slot 0: route owner (ROUTE_OWNER), finisher (FINISHER), prologue head (PROLOGUE); others: QKV task |
| 16-23 | pooler K part (POOLER); slot 16 + t on XCD t reduces pooled tile t | stream | post-expert sender (POST_SENDER); slot 16: sum reducer (REDUCER); QKV task |
| 24-31 | sliding attention splits on XCDs 0 .. 2 x requests - 1 (split = slot - 24); top-k (slot 24, XCD 7 - g), pooled-row norm + MXFP4 (POOLED_NORM, slot 25+) | stream | slots 24-30: ROUTE_OWNER, FINISHER, PROLOGUE; slot 31: QKV task |

Full-attention layers run the balanced decode's items on every workgroup
(program indices `cta` and `cta + 256`) and merge on slots 24 onward. When
the next layer is regular, the FINISHER workgroups write the regular boundary
(BOUNDARY) instead of QKV and PROLOGUE. `muse_spark/plan.py` models these roles and
checks every wait: no cycle, and no vector poll behind the role's own loads
unless the loads land first.

## Hand-offs

The rules in the design doc apply; on gfx950:

- Same-XCD flags (attention records) are polled with scalar `glc` loads.
- Words written on one XCD and polled on another (routed ids, xq rows, h',
  S rows, MoE readiness) live in uncached memory (`UncachedBuffer`), polled
  with scalar loads.
- Cross-GPU areas (Iris heap) are Lamport words (data that replaces an empty
  marker) or 64-bit (value, tag) words, read with `.cv` vector loads; two
  copies by epoch parity, the 128 senders re-arm the next copy at entry.
- Every wait checks the deadline (60 s by default) and the abort word every
  64 retries; on expiry the workgroup records the failure (also in a host
  page), sets every rank's abort word and traps.

## Plan fields

`DEFAULT_PLAN_FIELDS` (prepared.py) are compile-time choices the benchmark
varies one at a time: the MoE ring depth, its wave and down-unit orders and
address form, the tail-role share of the expert bytes, the attention
pipeline depth, the QKV weight window, who merges a sliding block
(`att_merge`), and the launch entry's form (`entry`). Serving uses the
defaults.

## Checks

- Prepare compiles every plan variant offline (`offline_compile_args`,
  `offline_attrs`) and checks one resident workgroup per CU: LDS above half
  the CU's, no scratch, at most 256 VGPRs + AGPRs, no calls. The first launch
  of each variant checks its own binary again.
- `STAMPS` (compiled out by default) stores each workgroup's clock at its
  phase boundaries for benchmarks.
