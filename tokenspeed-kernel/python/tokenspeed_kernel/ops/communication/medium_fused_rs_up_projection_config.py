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

"""Unqualified medium-M geometry experiments; no runtime dispatch changes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MediumFusedRsUpProjectionTuning:
    """Explicit geometry and overlap controls for the medium-M experiment.

    Args:
        tile_m: Whole MMA M extent, including both CTAs for two-CTA MMA.
        tile_n: Owner-local MMA N extent, exactly dividing the 896 columns.
        two_cta: Whether to use tcgen05 CTA-group-two instructions.
        cluster_m: One for single CTA or two for a cooperating CTA pair.
        cluster_n: One; inter-pair input multicast is a separate experiment.
        c_stages: Output ring depth, zero for the unchanged automatic heuristic.
        ab_stages: Zero for the unchanged A/B heuristic, or an exact pipeline
            depth in [2,21]. An explicit depth also requires explicit C stages;
            the resulting actual layout must fit the device shared-memory budget.
        addend_stages: One or two reduction/residual shared-memory stages.
        reduce_vectors: Independent BF16x8 load-reduces issued as one group.
        cluster_cap: Static-persistent positive cap, or None for all-rank hardware
            capacity. full_grid requires None because it launches every work tile.
        scheduler_type: static_persistent or full_grid. Both retain the same
            static device scheduler; full_grid gives each cluster one work tile.

    Epilogues retain the entire per-CTA M extent and 64 columns. All variants
    retain early accumulator release, acquire-before-store, direct symmetric
    TMA multicast, and the unchanged external publication/completion barriers.
    """

    tile_m: int
    tile_n: int
    two_cta: bool
    cluster_m: int
    cluster_n: int
    c_stages: int
    ab_stages: int
    addend_stages: int
    reduce_vectors: int
    cluster_cap: int | None
    scheduler_type: str

    @property
    def enable_pdl(self):
        """Return the frozen PDL-off policy consumed by the inherited exit."""
        return False

    @property
    def epilogue_m(self):
        """Return full per-CTA accumulator rows; no partial-M partition."""
        return self.tile_m // (2 if self.two_cta else 1)

    @property
    def epilogue_n(self):
        """Return the fixed 64-column epilogue transaction width."""
        return 64

    def launch_geometry(self, num_tokens, hardware_capacity):
        """Return problem/launch cluster counts and the cluster-shaped grid.

        Args:
            num_tokens: Explicit diagnostic token count in [33,8192].
            hardware_capacity: Positive all-rank minimum cluster capacity. This
                limits the persistent worker population, not queued full-grid work.

        Returns:
            A mapping of exact planned launch geometry. Counts are not observed
            occupancy; compiled CUBIN residency is separately queried and admitted.
        """
        self.validate()
        if type(num_tokens) is not int or not 33 <= num_tokens <= 8192:
            raise ValueError("launch geometry requires explicit M in [33,8192]")
        if type(hardware_capacity) is not int or hardware_capacity < 1:
            raise ValueError("hardware capacity must be a positive integer")
        cap = hardware_capacity if self.cluster_cap is None else self.cluster_cap
        if cap > hardware_capacity:
            raise ValueError("requested cap exceeds all-rank hardware capacity")
        problem_clusters = ((num_tokens + self.tile_m - 1) // self.tile_m) * (
            896 // self.tile_n
        )
        launched_clusters = (
            problem_clusters
            if self.scheduler_type == "full_grid"
            else min(problem_clusters, cap)
        )
        return {
            "problem_clusters": problem_clusters,
            "launched_clusters": launched_clusters,
            "grid": [self.cluster_m, self.cluster_n, launched_clusters],
        }

    def validate(self):
        """Reject unsupported or implicitly coerced controls before compilation."""
        for name in (
            "tile_m",
            "tile_n",
            "cluster_m",
            "cluster_n",
            "c_stages",
            "ab_stages",
            "addend_stages",
            "reduce_vectors",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an explicit integer")
        if type(self.two_cta) is not bool:
            raise ValueError("two_cta must be an explicit bool")
        geometries = (
            {(128, 64), (128, 128), (256, 64), (256, 128)}
            if self.two_cta
            else {(64, 64), (64, 128), (128, 64), (128, 128)}
        )
        if (self.tile_m, self.tile_n) not in geometries:
            raise ValueError("unsupported experimental MMA geometry")
        if (self.cluster_m, self.cluster_n) != ((2, 1) if self.two_cta else (1, 1)):
            raise ValueError("cluster must exactly match the one/two-CTA MMA group")
        if self.c_stages not in (0, 2, 3, 4, 5, 6, 7, 8):
            raise ValueError("c_stages must be zero or an explicit integer in [2,8]")
        if self.ab_stages != 0 and not 2 <= self.ab_stages <= 21:
            raise ValueError("ab_stages must be zero or an explicit integer in [2,21]")
        if self.ab_stages and self.c_stages == 0:
            raise ValueError("explicit A/B stages require explicit C stages")
        if self.addend_stages not in (1, 2):
            raise ValueError("fused reduction requires one or two addend stages")
        if self.reduce_vectors not in (1, 2, 4, 8):
            raise ValueError("reduce_vectors must be 1,2,4 or8")
        vectors_per_thread = self.epilogue_m * self.epilogue_n // (128 * 8)
        if vectors_per_thread % self.reduce_vectors:
            raise ValueError("request group must divide each thread's epilogue vectors")
        if self.addend_stages == 2 and self.reduce_vectors != 1:
            raise ValueError(
                "two addend stages currently require single-vector requests"
            )
        if self.cluster_cap is not None and (
            type(self.cluster_cap) is not int or self.cluster_cap < 1
        ):
            raise ValueError("cluster_cap must be None or a positive integer")
        if self.scheduler_type not in ("static_persistent", "full_grid"):
            raise ValueError("scheduler_type must be static_persistent or full_grid")
        if self.scheduler_type == "full_grid" and self.cluster_cap is not None:
            raise ValueError(
                "full_grid requires cluster_cap=None; every work tile is launched"
            )
