"""Test-only protocol fakes for nltiming math tests (no backend sessions)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nltiming.engine_support import LinearModel, LinearTimingEngine
from nltiming.protocols import GaugeProvenance


def _reporting_from_compatibility(compatibility: str) -> tuple[str, bool | None]:
    if str(compatibility).lower().startswith("tempo2"):
        return "mean", False
    return "mean", True


class LinearTestEngine(LinearTimingEngine):
    """Thin linear TimingEngine for nltiming tests."""

    @classmethod
    def from_linear_model(
        cls,
        model: LinearModel,
        *,
        gauge_provenance: GaugeProvenance | None = None,
        compatibility: str = "auto",
        precision_critical: frozenset[str] | set[str] = frozenset(),
    ):
        _ = precision_critical
        if gauge_provenance is None:
            reporting_mode, reporting_weighted = _reporting_from_compatibility(
                compatibility
            )
            gauge_provenance = GaugeProvenance(
                export="none",
                reference_mode="unknown",
                reporting_mode=reporting_mode,
                reporting_weighted=reporting_weighted,
            )
        return cls(model, gauge_provenance=gauge_provenance)


class JaxLinearTestEngine(LinearTimingEngine):
    """Linear TimingEngine with a JAX residual_delta surface."""

    def __init__(
        self,
        model: LinearModel,
        *,
        gauge_provenance: GaugeProvenance,
        compatibility: str = "auto",
        precision_critical: frozenset[str] | set[str] = frozenset(),
    ):
        super().__init__(model, gauge_provenance=gauge_provenance)
        self.compatibility = compatibility
        self._precision_critical = frozenset(precision_critical)

    @classmethod
    def from_linear_model(
        cls,
        model: LinearModel,
        *,
        gauge_provenance: GaugeProvenance | None = None,
        compatibility: str = "auto",
        precision_critical: frozenset[str] | set[str] = frozenset(),
    ):
        if gauge_provenance is None:
            reporting_mode, reporting_weighted = _reporting_from_compatibility(
                compatibility
            )
            gauge_provenance = GaugeProvenance(
                export="none",
                reference_mode="unknown",
                reporting_mode=reporting_mode,
                reporting_weighted=reporting_weighted,
            )
        return cls(
            model,
            gauge_provenance=gauge_provenance,
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
    """Lightweight container used by gauge-column tests."""

    def __init__(self, contributions):
        self.contributions = list(contributions)
