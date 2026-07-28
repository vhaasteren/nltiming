"""Runtime validators and shared engine primitives."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Mapping

import numpy as np

from nltiming.protocols import EnterprisePulsarLike, GaugeProvenance, TimingEngine


def _as_1d_float(arr, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return out


def validate_pulsar_surface(pulsar: EnterprisePulsarLike) -> None:
    """Validate basic frozen pulsar-surface shape invariants."""
    toas = _as_1d_float(pulsar.toas, name="toas")
    residuals = _as_1d_float(pulsar.residuals, name="residuals")
    toaerrs = _as_1d_float(pulsar.toaerrs, name="toaerrs")
    freqs = _as_1d_float(pulsar.freqs, name="freqs")
    backend_flags = np.asarray(pulsar.backend_flags)
    mmat = np.asarray(pulsar.Mmat, dtype=float)

    nrows = len(toas)
    if len(residuals) != nrows or len(toaerrs) != nrows or len(freqs) != nrows:
        raise ValueError(
            "Pulsar arrays toas/residuals/toaerrs/freqs must be same length"
        )
    if len(backend_flags) != nrows:
        raise ValueError("backend_flags length mismatch with pulsar rows")
    if mmat.shape[0] != nrows:
        raise ValueError("Mmat row count must match pulsar arrays")
    if mmat.shape[1] != len(pulsar.fitpars):
        raise ValueError("Mmat column count must match fitpars length")


def zero_delta_tolerance(engine: TimingEngine, requested: float) -> float:
    """Return the strict caller-requested residual-delta tolerance.

    JUG's former tempo2-specific relaxation represented a closed
    reference-state gap. Current tempo2 compatibility has picosecond-tier
    zero-delta behavior and is validated by the same checks as other engines.
    """
    _ = engine
    return float(requested)


def is_exact_linear_param(param_name: str) -> bool:
    """Return true for fitpars that should use exact design-matrix columns.

    Note: PTA-suffixed aliases such as ``Offset_epta`` currently do *not* match
    bare ``Offset`` (unlike ``JUMP1_epta``, which matches via ``startswith``).
    Changing that alters the MCMC residual path; do it only together with a
    chain re-run, not when overlaying GPs on already-sampled chains.
    """
    name = param_name.upper()
    if name == "OFFSET":
        return True
    if name.startswith(("DMX", "JUMP", "FD")):
        return True
    return False


def validate_engine_zero_delta(engine: TimingEngine, tol: float = 1e-12) -> None:
    """Check residual_delta(0) = 0 invariant."""
    effective_tol = zero_delta_tolerance(engine, tol)
    zero = np.zeros(len(engine.fitpars), dtype=float)
    residual = np.asarray(engine.residual_delta(zero), dtype=float)
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    if max_abs <= effective_tol:
        return
    raise ValueError("residual_delta(0) must equal 0")


def validate_engine_shapes(engine: TimingEngine) -> None:
    """Check engine fitpar/dmatrix shape invariants."""
    design = np.asarray(engine.design_matrix(), dtype=float)
    if design.ndim != 2:
        raise ValueError("design_matrix must be 2D")
    if design.shape[1] != len(engine.fitpars):
        raise ValueError("design_matrix columns must match fitpars")
    ref_exact = engine.reference_theta_exact()
    missing = [name for name in engine.fitpars if name not in ref_exact]
    if missing:
        raise ValueError(f"reference_theta_exact missing fitpars: {missing}")


def validate_engine_against_pulsar(
    engine: TimingEngine, pulsar: EnterprisePulsarLike, tol: float = 1e-12
) -> None:
    """Validate engine outputs against pulsar canonical row and column ordering."""
    validate_pulsar_surface(pulsar)
    validate_engine_shapes(engine)
    validate_engine_zero_delta(engine, tol=tol)
    design = np.asarray(engine.design_matrix(), dtype=float)
    pulsar_design = np.asarray(pulsar.Mmat, dtype=float)
    nrows = len(pulsar.toas)
    if design.shape[0] != nrows:
        raise ValueError("Engine row count must match pulsar rows")
    if tuple(pulsar.fitpars) != tuple(engine.fitpars):
        raise ValueError("Engine fitpars must match pulsar fitpars in canonical order")
    if design.shape != pulsar_design.shape:
        raise ValueError("Engine design_matrix shape must match pulsar.Mmat")
    if not np.allclose(design, pulsar_design, atol=tol, rtol=0.0):
        raise ValueError(
            "Engine design_matrix must match pulsar.Mmat in canonical row order"
        )


@dataclass(frozen=True)
class LinearModel:
    """Simple linearized residual model around reference theta."""

    fitpars: tuple[str, ...]
    design: np.ndarray
    theta_exact: Mapping[str, str]
    native_units: Mapping[str, str]

    @classmethod
    def from_design(
        cls,
        *,
        fitpars: tuple[str, ...],
        design: np.ndarray,
        theta_exact: Mapping[str, str] | None = None,
        native_units: Mapping[str, str] | None = None,
    ) -> "LinearModel":
        if theta_exact is None:
            theta_exact = {name: "0.0" for name in fitpars}
        if native_units is None:
            native_units = {name: "native" for name in fitpars}
        return cls(
            fitpars=fitpars,
            design=np.asarray(design, dtype=float),
            theta_exact=dict(theta_exact),
            native_units=dict(native_units),
        )

    def reference_theta(self) -> np.ndarray:
        with localcontext() as ctx:
            ctx.prec = 50
            return np.asarray(
                [float(Decimal(self.theta_exact[name])) for name in self.fitpars],
                dtype=float,
            )

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        # Fitter sign: Δr ≈ -M δ
        return -(self.design @ delta)


class LinearTimingEngine:
    """Concrete TimingEngine wrapper around a LinearModel."""

    def __init__(self, model: LinearModel, *, gauge_provenance: GaugeProvenance):
        self._model = model
        self._gauge_provenance = gauge_provenance
        self.fitpars = model.fitpars
        self.native_units = dict(model.native_units)

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def identically_linear_fitpars(self) -> frozenset[str]:
        """A linear model is affine in every delta, so every fitpar qualifies."""
        return frozenset(self.fitpars)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        return self._model.residual_delta(delta_theta)

    def residual_jacobian(self) -> np.ndarray:
        """J = -M for the linearized model."""
        return -np.asarray(self._model.design, dtype=float)

    def design_matrix(self, params=None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)

    def gauge_provenance(self) -> GaugeProvenance:
        return self._gauge_provenance

    @property
    def gauge_applied(self) -> bool:
        return self.gauge_provenance().export != "none"
