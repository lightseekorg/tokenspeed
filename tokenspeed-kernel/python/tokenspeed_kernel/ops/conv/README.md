# Short causal convolutions

`inkling_ring_sconv` uses the same raw-input ring and checkpoint fields for static and dynamic weights.
Prefill reads pre-chunk values from the input checkpoint.
Decode and verification read pre-chunk values from the ring.
The output multiplier never changes stored raw inputs.

Callers explicitly supply `dynamic_weights` and `output_scale`.
Static callers pass `None` and their selected scale.
Legacy unscaled callers pass `1.0`.
Dynamic coefficients have shape `[tokens, heads, width]`, where `heads` divides the channel count.
Each coefficient applies to all channels in one head.

Weights retain shape `[channels, width]` and can use explicit strides.
Static taps run from oldest to current.
Dynamic taps instead use causal lag: tap zero is current, followed by earlier positions.
A dynamic coefficient rounds to the input dtype as it loads, then adds to the static coefficient in FP32 before multiplication.
A native `[width, channels]` dynamic static-weight tensor can pass as its transpose view.

Static accumulation starts at zero and uses FP32 fused multiply-add for every tap.
Dynamic accumulation uses a multiply for tap zero and fused multiply-add for subsequent taps.
The kernel then adds the optional current-input residual and applies `output_scale` before the output store.
The source and destination dtype determine materialization.
These operations preserve the supplied reference's accumulation order.
A `[tokens, slots, channels]` input is MoE route rows: every load adds the slots in FP32 slot order from +0.0 and rounds once, and that sum is the input everywhere, the ring and checkpoints included.
`inkling_ring_sconv_mailbox` reads its slots from a Lamport mailbox (`ops.communication.mailbox`): the ranks' producers publish their partials, and the decode kernel polls each slot until it holds no armed `-0.0`, sums the slots in rank order the same way, and re-arms them.
One program then covers every token of its channels, so it is the slots' only reader, and it waits on its PDL predecessor only before its stores.

Prefill restoration addresses raw checkpoint rows by causal lag.
Prefill launches one extra program row per request beside the token tiles.
Only those edge programs touch the checkpoint: they compute a warm request's first `W - 1` rows, restore its window into the ring, and publish.
The token tiles compute every other row from the chunk alone, which keeps them lean.
With a `token_range`, a prefill runs the tiles of that range plus, per request, the tiles around its last `R` rows that the range does not hold.
A sequence-parallel rank then computes its share of the output rows and still writes every request's ring and published window, reading only its rows, the `W - 1` rows before them and each request's last `R` rows.
Changing coefficient order does not change those rows' physical order.
Padded request slots never write ring or checkpoint state.
The runtime helper selects the existing K, V, or adjacent K+V checkpoint fields.
Unsupported channel spans raise an error.

Under PDL, the `inkling_ring_sconv` decode kernel triggers its dependents before its own wait.
It loads its request metadata and pre-chunk ring taps before that wait; it follows projections and collectives, which all wait first.
