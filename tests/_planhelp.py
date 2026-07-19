"""Shared test helper: build a TimingParameterPlan without a full model.

Replaces the old ``resolve_partition(pulsar, analytically_marginalize=...)`` /
``PartitionResult(...)`` scaffolding in tests that feed a plan-shaped object into
the metric/whitening/prior helpers.
"""

from __future__ import annotations

from typing import Sequence

from nltiming.coordinates import TimingCoordinatePolicy
from nltiming.inference import TimingInference, resolve_inference_plan
from nltiming.linearity import resolve_linearity


def plan_for(
    pulsar,
    *,
    delta_flat: Sequence[str] = (),
    z_prior: Sequence[str] = (),
    sample_all: bool = False,
    identically_linear: Sequence[str] | None = None,
):
    """Resolve a plan: ``delta_flat`` marginalized, everything else sampled."""
    if sample_all or (not delta_flat and not z_prior):
        inference = TimingInference.sample_all()
    else:
        inference = TimingInference.groups(delta_flat=delta_flat, z_prior=z_prior)
    linearity = resolve_linearity(pulsar, None, identically_linear=identically_linear)
    return resolve_inference_plan(
        pulsar,
        inference=inference,
        linearity=linearity,
        coordinate_policy=TimingCoordinatePolicy(),
    )
