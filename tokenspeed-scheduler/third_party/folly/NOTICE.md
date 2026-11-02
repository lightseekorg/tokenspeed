# Folly stable radix sort adaptation

`tokenspeed-scheduler/csrc/cache/coordinator/folly_position_radix.h` derives
from Meta Folly's `folly/algorithm/StableRadixSort.h`, commit
`83f525dc2b583925b857ac153e3befd51eb4879e`:
https://github.com/facebook/folly/blob/83f525dc2b583925b857ac153e3befd51eb4879e/folly/algorithm/StableRadixSort.h

Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under Apache 2.0;
the full license is included in LICENSE in this directory. No upstream root
NOTICE file was present at this revision.

Changes: specialize for cache position ranks in [-INT32_MAX, 0], replace Folly
type utilities with standard C++ headers, and remove unused floating-point,
descending and small-input APIs. Retain stable byte counting/scatter, uniform
byte skipping and alternating buffers. No Folly package dependency is added.
