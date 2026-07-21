"""EngineDeltaMap: the sole seam converting sampling-frame deltas into a full
engine-order delta vector for ``engine.residual_delta``.

Two modes share one implementation:

- proper mode (``for_proper``): input covers every proper axis (sampled and
  z-marginalized) in proper order; delta-flat slots stay at zero. Used by
  ``build_linearization`` (autodiff/stencil differentiate through this map, so
  physical-chart Jacobians enter W_s/W_m automatically).
- sampled mode (``for_sampled``): input covers the sampled axes only;
  z-marginalized slots are pinned at their fixed sampling-frame expansion
  deltas; delta-flat slots stay at zero. Used by every likelihood delay path.

xp-generic: numpy for plain evaluation, jax.numpy inside traced code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _scatter(vec, idx, values, xp):
    if hasattr(vec, "at"):  # jax array
        return vec.at[xp.asarray(np.asarray(idx, dtype=int))].set(values)
    vec[np.asarray(idx, dtype=int)] = values  # numpy: in-place on fresh array
    return vec


def apply_charts(vec, charts, xp):
    """Apply every PhysicalChart to a full-length sampling-frame delta vector
    (protocol only — this module never imports concrete chart types). Also
    used directly by ``_exact_flat_columns`` (§6.1) to differentiate the
    composed map at arbitrary points, including nonzero delta-flat slots."""
    for ch in charts:
        vec = ch.apply_delta(vec, xp)
    return vec


@dataclass(frozen=True)
class EngineDeltaMap:
    nfit: int
    input_names: tuple[str, ...]  # sampling-frame axis names of the input
    input_slots: tuple[int, ...]  # their fitpar slots
    fixed_slots: tuple[int, ...]  # z-marg slots (sampled mode), else ()
    fixed_values: tuple[float, ...]  # sampling-frame deltas for fixed slots
    charts: tuple = ()  # tuple[PhysicalChart, ...] (protocol)

    @classmethod
    def for_proper(cls, plan, charts=()):
        proper = [
            a for a in plan.axes if a.disposition in ("sample", "marginalize_z_prior")
        ]
        return cls(
            nfit=len(plan.fitpars),
            input_names=tuple(a.name for a in proper),
            input_slots=tuple(a.fitpar_index for a in proper),
            fixed_slots=(),
            fixed_values=(),
            charts=tuple(charts),
        )

    @classmethod
    def for_sampled(cls, plan, charts, linearization):
        proper = [
            a for a in plan.axes if a.disposition in ("sample", "marginalize_z_prior")
        ]
        zm = [
            (a, i)
            for i, a in enumerate(proper)
            if a.disposition == "marginalize_z_prior"
        ]
        return cls(
            nfit=len(plan.fitpars),
            input_names=plan.sampled,
            input_slots=plan.indices("sample"),
            fixed_slots=tuple(a.fitpar_index for a, _ in zm),
            fixed_values=tuple(float(linearization.delta_expansion[i]) for _, i in zm),
            charts=tuple(charts),
        )

    def full_engine_delta(self, values, xp):
        values = xp.asarray(values)
        vec = xp.zeros((self.nfit,), dtype=values.dtype)
        vec = _scatter(vec, self.input_slots, values, xp)
        if self.fixed_slots:
            vec = _scatter(
                vec,
                self.fixed_slots,
                xp.asarray(self.fixed_values, dtype=values.dtype),
                xp,
            )
        return apply_charts(vec, self.charts, xp)
