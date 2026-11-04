# Monokernels on gfx950 (Gluon)

### gfx950 Muse Spark decoder layer

`gluon_monokernel_muse_spark_gfx950` (`muse_spark/layer.py`) runs one Muse
Spark decoder layer, and the next layer's QKV projection and attention
prologue, as one persistent launch. `muse_spark/prepared.py` holds the host
side (state, weights read in place, checks, launch); `../gluon.py` registers
it. `layer.py`'s module docstring defines the roles and exchanges named
below.

#### Contract

- 256 workgroups x 8 waves, one workgroup per CU: each reserves 96 KB of
  LDS, more than half of a CU's 160 KB.
- Prepare compiles every plan variant offline (`offline_compile_args`,
  `offline_attrs`) and accepts a variant only with no scratch, at most 256
  VGPRs + AGPRs and no calls. The first launch of each variant checks its
  own binary again.
- Every wait checks the deadline (60 s by default) and the abort word every
  64 retries. On expiry the workgroup records the failure (also in a host
  page), sets every rank's abort word and traps.
- `DEFAULT_PLAN_FIELDS` (`prepared.py`) are the compile-time choices that
  benchmarks vary one at a time: the MoE ring depth, its wave and down-unit
  orders and address form, the tail workgroups' share of the expert bytes,
  the attention pipeline depth, the QKV weight window and who merges a
  sliding block (`att_merge`). Serving uses the defaults.
- `STAMPS` (compiled out by default) stores each workgroup's clock at its
  phase boundaries, for benchmarks.

#### Algorithm

A workgroup's roles come from its XCD (hardware) and its slot (program id //
8), and run in this order:

| Slots per XCD | Pre-MoE | MoE | Post-MoE |
|---|---|---|---|
| 0-15 | o_proj sender; slot 0 also owns 128 columns of the all-reduce; router K part | stream | slot 0: route owner, finisher, prologue task; others: QKV task |
| 16-23 | pooler K part; slot 16 + t on XCD t sums pooled tile t | stream | post-expert sender; slot 16: reducer; QKV task |
| 24-31 | sliding attention splits on XCDs 0 .. 2 x requests - 1 (split = slot - 24); top-k (slot 24, XCD 7 - g); pooled-row norm (slot 25+) | stream | slots 24-30: route owner, finisher, prologue task; slot 31: QKV task |

Slots 0 and 24-30 are the tail workgroups. Full-attention layers run the
balanced decode's items on every workgroup (program indices `cta` and
`cta + 256`) and merge on slots 24 onward. When the next layer is regular,
the tail workgroups write the regular boundary instead of the QKV and
prologue tasks. `muse_spark/plan.py` models these roles and checks every
wait: no cycle, and no vector poll behind the role's own loads unless those
loads land first.

The hand-off rules of `docs/design/monokernel.md` apply. On gfx950:

- Same-XCD flags (attention records) are polled with scalar `glc` loads.
- Words written on one XCD and polled on another (routed ids, xq rows, h',
  the routed sums, MoE readiness) live in uncached memory (`UncachedBuffer`)
  and are polled with scalar loads.
- Cross-GPU areas (Iris heap) hold Lamport words (data that replaces an
  empty marker) or 64-bit (value, tag) words, read with `.cv` vector loads.
  Each area has two copies by epoch parity; the 128 o_proj senders re-arm
  the next copy at entry.
