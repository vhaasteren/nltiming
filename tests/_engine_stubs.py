"""Test-only protocol fakes for nltiming math tests (no backend sessions)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from psrdata import ResidualCentering

from nltiming.engine_support import LinearModel, LinearModelEngine


def centering_for_compatibility(compatibility: str) -> ResidualCentering:
    """What a host of that par/tim compatibility normally reports.

    tempo2 removes an unweighted mean; the PINT family a weighted one. The
    stored residuals of a synthetic engine are unknown.
    """
    weighted = not str(compatibility).lower().startswith("tempo2")
    return ResidualCentering(
        stored_residuals="unknown",
        standard_output="mean_removed",
        standard_weighted=weighted,
    )


def gauge_free_centering() -> ResidualCentering:
    """Residuals stored with no centering, by a PINT-family package."""
    return ResidualCentering(
        stored_residuals="none",
        standard_output="mean_removed",
        standard_weighted=True,
    )


class LinearTestEngine(LinearModelEngine):
    """Thin linear TimingEngine for nltiming tests."""

    @classmethod
    def from_linear_model(
        cls,
        model: LinearModel,
        *,
        residual_centering: ResidualCentering | None = None,
        compatibility: str = "auto",
        precision_critical: frozenset[str] | set[str] = frozenset(),
    ):
        _ = precision_critical
        if residual_centering is None:
            residual_centering = centering_for_compatibility(compatibility)
        return cls(model, residual_centering=residual_centering)


class JaxLinearTestEngine(LinearModelEngine):
    """Linear TimingEngine with a JAX residual_delta surface."""

    def __init__(
        self,
        model: LinearModel,
        *,
        residual_centering: ResidualCentering | None = None,
        compatibility: str = "auto",
        precision_critical: frozenset[str] | set[str] = frozenset(),
    ):
        if residual_centering is None:
            residual_centering = centering_for_compatibility(compatibility)
        super().__init__(model, residual_centering=residual_centering)
        self.compatibility = compatibility
        self._precision_critical = frozenset(precision_critical)

    @classmethod
    def from_linear_model(
        cls,
        model: LinearModel,
        *,
        residual_centering: ResidualCentering | None = None,
        compatibility: str = "auto",
        precision_critical: frozenset[str] | set[str] = frozenset(),
    ):
        return cls(
            model,
            residual_centering=residual_centering,
            compatibility=compatibility,
            precision_critical=precision_critical,
        )

    def residual_delta_jax(self, delta_theta: Any) -> Any:
        import jax.numpy as jnp

        design = jnp.asarray(self.design_matrix(), dtype=jnp.asarray(delta_theta).dtype)
        delta = jnp.asarray(delta_theta)
        return -(design @ delta)

    def precision_critical_fitpars(self) -> frozenset[str]:
        return self._precision_critical


@dataclass(frozen=True)
class TestContribution:
    """Contribution metadata for gauge/context tests."""

    name: str
    row_indices: np.ndarray
    engine: Any


TestContribution.__test__ = False  # not a pytest test class


class CompositeView:
    """Lightweight container used by gauge-column tests.

    Its ``residual_centering`` is the union of its leaves', keyed by
    contribution name, which is what a real composite exposes.
    """

    def __init__(self, contributions):
        self.contributions = list(contributions)

    @property
    def residual_centering(self) -> dict[str, ResidualCentering]:
        out = {}
        for contribution in self.contributions:
            leaf = getattr(contribution.engine, "residual_centering", None) or {}
            for value in dict(leaf).values():
                out[str(contribution.name)] = value
        return out
