These MessagePack vectors come from TokenSpeed's native `MsgpackEncoder`,
using its declared `TokenizedGenerateReqInput` and `BatchTokenIDOutSlim` types.
They are mirrored under TokenSpeed `test/runtime/fixtures` and exercised on both
sides of the wire.

- `tokenspeed_pd_epd_request`: normalized greedy seed, PD host/port/63-bit room,
  DP rank 2, original and pad-substituted IDs, image grid tensor, EPD per-item
  rendezvous and intentionally omitted encoder feature.
- `tokenspeed_weight_output`: legacy 15-element output carrying weight identity,
  preserving compatibility when newer metadata tails are absent.
- `tokenspeed_finish_metadata_output`: older 18-element output carrying weight identity,
  full stop details (matched token 3), accepted draft count 4 and draft count 6.

Request and legacy output tests compare Rust encoding byte-for-byte. Finish
metadata tests check semantic dictionary fields; map ordering is immaterial.

- `tokenspeed_prompt_output`: 20-element output carrying nullable prompt scores
  `[None, -0.25, -0.5]` and prompt ids `[1, 2, 3]`, alongside prior metadata.
