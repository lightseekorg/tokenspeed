# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""``--dsa-kvp-prefill-levers``: faster forms of the KVP sparse (DSA) prefill.

Under ``--kv-parallel-size`` every rank scores and attends only the pages it
owns, and the ranks exchange index candidates and attention partials. Each
lever replaces one step of that prefill with a cheaper form that keeps the
step's result:

- ``merge``: each rank merges the index candidates of the query rows it owns
  (an all-to-all of the candidates, then an all-gather of the selection)
  instead of every rank gathering every candidate and merging every row.
- ``qlayout``: the query heads are gathered rank-major and the sparse core
  writes its partials directly in the layout the partials exchange sends,
  instead of a relayout before the core and a repack after it. An FP8 KV
  cache only, whose query the NCCL gather relays out; a BF16 query keeps the
  ordinary gather, which takes the multicast kernels where they reach
  (``DSABackend._probe_dcp_records_core``).
- ``owned``: the index candidate tiles span the pages a rank owns instead of
  the whole page table.

The server argument names a subset. A DSA leaf applies the subset its numerics
allow (:func:`effective_dsa_kvp_prefill_levers`), and the model reads the
leaf's set, so the indexer and the sparse core always agree.
"""

from __future__ import annotations

# In the order the canonical spelling lists them.
DSA_KVP_PREFILL_LEVERS = ("merge", "qlayout", "owned")

# Spelling of the empty set besides the empty string, for launchers that drop
# empty arguments.
_NO_LEVERS = "none"

# Levers that run their own sparse core in place of the leaf's prefill kernel
# plan; a pinned plan (a bitwise envelope's) keeps its core, so they are off.
_PLAN_BYPASSING_LEVERS = frozenset({"qlayout"})

# Levers whose result matches today's only within a tolerance, which a
# batch-invariant leaf refuses. None of the current levers: ``merge`` and
# ``owned`` keep every row's selection and ``qlayout`` is bitwise.
_TOLERANCE_LEVERS: frozenset[str] = frozenset()


def parse_dsa_kvp_prefill_levers(value: str) -> frozenset[str]:
    """The lever set a ``--dsa-kvp-prefill-levers`` value names.

    Args:
        value: Comma-separated lever names from :data:`DSA_KVP_PREFILL_LEVERS`
            in any order, whitespace around a name ignored; the empty string
            or ``none`` names no lever.

    Returns:
        The named levers.

    Raises:
        ValueError: On an unknown name, or ``none`` together with a lever.
    """
    names = [name.strip() for name in value.split(",") if name.strip()]
    if names == [_NO_LEVERS]:
        return frozenset()
    unknown = sorted(set(names).difference(DSA_KVP_PREFILL_LEVERS))
    if unknown:
        raise ValueError(
            "--dsa-kvp-prefill-levers takes a comma-separated subset of "
            f"{list(DSA_KVP_PREFILL_LEVERS)}, or an empty value or "
            f"{_NO_LEVERS!r} for none; got unknown {unknown} in {value!r}"
        )
    return frozenset(names)


def canonical_dsa_kvp_prefill_levers(value: str) -> str:
    """The one spelling of the lever set a ``--dsa-kvp-prefill-levers`` value
    names: its levers in :data:`DSA_KVP_PREFILL_LEVERS` order, comma-joined;
    the empty string for none.

    Args:
        value: As for :func:`parse_dsa_kvp_prefill_levers`.

    Returns:
        Equal strings for values naming equal sets.

    Raises:
        ValueError: As :func:`parse_dsa_kvp_prefill_levers` does.
    """
    levers = parse_dsa_kvp_prefill_levers(value)
    return ",".join(name for name in DSA_KVP_PREFILL_LEVERS if name in levers)


def effective_dsa_kvp_prefill_levers(
    levers: frozenset[str],
    *,
    batch_invariant: bool,
    prefill_kernel_plan_pinned: bool,
) -> frozenset[str]:
    """The levers a DSA leaf applies to its KVP prefill.

    Args:
        levers: The requested levers (:func:`parse_dsa_kvp_prefill_levers`).
        batch_invariant: Whether the leaf serves a bitwise numerics envelope;
            levers whose result matches only within a tolerance are dropped.
        prefill_kernel_plan_pinned: Whether the leaf's sparse prefill runs a
            pinned kernel plan; levers that would run their own core instead
            are dropped.

    Returns:
        The subset of ``levers`` the leaf applies.
    """
    effective = levers
    if batch_invariant:
        effective = effective - _TOLERANCE_LEVERS
    if prefill_kernel_plan_pinned:
        effective = effective - _PLAN_BYPASSING_LEVERS
    return effective
