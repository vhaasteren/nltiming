"""Helpers for gauge-column fixtures.

Production MetaPulsar always injects per-PTA ``Offset_<pta>`` / ``Offset``
columns. Unit-test pulsars must do the same before context construction.
"""

from __future__ import annotations

import numpy as np

from _engine_stubs import JaxLinearTestEngine
from nltiming.engine_support import LinearModel
from nltiming.protocols import GaugeProvenance


def gauge_free_provenance() -> GaugeProvenance:
    return GaugeProvenance(
        export="none",
        reference_mode="none",
        reporting_mode="mean",
        reporting_weighted=True,
    )


def with_leading_offset(
    fitpars: tuple[str, ...],
    design: np.ndarray,
    theta_exact: dict[str, str],
) -> tuple[tuple[str, ...], np.ndarray, dict[str, str]]:
    """Prepend an ``Offset`` constant column when absent."""
    if any(name == "Offset" or name.startswith(("Offset_", "PHOFF")) for name in fitpars):
        return fitpars, np.asarray(design, dtype=float), dict(theta_exact)
    n = int(np.asarray(design).shape[0])
    fitpars = ("Offset",) + tuple(fitpars)
    design = np.column_stack([np.ones(n), np.asarray(design, dtype=float)])
    theta_exact = {"Offset": "0.0", **dict(theta_exact)}
    return fitpars, design, theta_exact


def linearized_jug_from_design(
    *,
    fitpars: tuple[str, ...],
    design: np.ndarray,
    theta_exact: dict[str, str],
) -> tuple[tuple[str, ...], np.ndarray, JaxLinearTestEngine]:
    fitpars, design, theta_exact = with_leading_offset(fitpars, design, theta_exact)
    model = LinearModel.from_design(
        fitpars=fitpars, design=design, theta_exact=theta_exact
    )
    return fitpars, design, JaxLinearTestEngine.from_linear_model(
        model, gauge_provenance=gauge_free_provenance()
    )
