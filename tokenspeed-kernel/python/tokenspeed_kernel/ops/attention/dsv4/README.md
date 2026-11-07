# DeepSeek V4 selected attention

`dsv4_prefill` attends BF16 queries shaped `[tokens, heads, 512]` over selected
rows of a shared dense K/V workspace. The Triton solution accepts strided
query and output rows.

Token row offsets for queries, selected indices, outputs, and optional LSE
are computed in int64, with the token index widened before multiplication.
Selected K/V row indices are also widened before multiplying by their stride.
This keeps large prefill tensors and strided views addressable beyond the
signed int32 element range without changing the attention calculation.
