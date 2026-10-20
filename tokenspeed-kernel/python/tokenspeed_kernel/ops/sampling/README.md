# Vocabulary-parallel argmax

Packed keys order each shard's selected maximum before its inverse global
vocabulary index. Among those candidates, equal maxima select the lowest
global index, including ties between negative and positive zero. The pack
kernel canonicalizes both zero signs with integer bit operations before
encoding their value order. NaN handling and invalid-index sentinels follow
the local argmax provider as before.
