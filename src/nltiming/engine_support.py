"""Runtime validators and shared engine primitives.

Two linear engines meet here, and they are different things:

- :class:`psrdata.LinearTimingEngine` is *the record's own* calculation,
  ``Δr = -Mmat @ δ`` over a complete ``PulsarData`` (psrdata SPEC section 5).
  It is re-exported unchanged, together with ``LinearContribution`` and
  ``linear_engine``.
- :class:`LinearModelEngine` is this package's reference implementation of
  its own ``TimingEngine`` protocol over a bare :class:`LinearModel`
  (fit parameters, one design matrix, reference strings, units). It exists
  for engine adapters that linearize one block of a larger pulsar and for
  tests that need an engine without a record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
from psrdata import ResidualCentering
from psrdata.linear import LinearContribution, LinearTimingEngine, linear_engine

from nltiming.protocols import EnterprisePulsarLike, TimingEngine


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

    A former JUG tempo2-specific relaxation represented a closed
    reference-state gap. Current engines, including JUG's tempo2 path, have
    picosecond-tier zero-delta behavior and are validated by the same checks.
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
    return name.startswith(("DMX", "JUMP", "FD"))


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


def validate_engine_residual_centering(engine: TimingEngine) -> None:
    """Check the engine exposes a nonempty ``residual_centering`` mapping."""
    mapping = getattr(engine, "residual_centering", None)
    if mapping is None or callable(mapping):
        raise ValueError(
            f"Engine {type(engine).__name__} must expose a residual_centering "
            "mapping (data-set key -> psrdata.ResidualCentering)"
        )
    if not mapping:
        raise ValueError("residual_centering must have one entry per data set")
    for key, value in mapping.items():
        if not isinstance(value, ResidualCentering):
            raise TypeError(
                f"residual_centering[{key!r}] must be a psrdata.ResidualCentering, "
                f"got {type(value).__name__}"
            )


def validate_engine_against_pulsar(
    engine: TimingEngine, pulsar: EnterprisePulsarLike, tol: float = 1e-12
) -> None:
    """Validate engine outputs against pulsar canonical row and column ordering."""
    validate_pulsar_surface(pulsar)
    validate_engine_shapes(engine)
    validate_engine_zero_delta(engine, tol=tol)
    validate_engine_residual_centering(engine)
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


# --- a linear engine over a bare design matrix ---------------------------------

#: Unit label of a :class:`LinearModel` column whose caller supplied none.
#: Production engines take their units from the record's parameter facts
#: (PINT units, psrdata R-3.5.3); this placeholder is for test doubles.
UNKNOWN_UNIT = "native"


@dataclass(frozen=True)
class LinearModel:
    """One linear timing block: fit parameters, matrix, references and units.

    ``design`` is in fitter sign, ``Δr ≈ -design @ δ``. ``theta_exact`` holds
    the reference value of each fit parameter as a decimal string and
    ``native_units`` its unit label, both keyed by fit parameter.
    """

    fitpars: tuple[str, ...]
    design: np.ndarray
    theta_exact: Mapping[str, str] = field(default_factory=dict)
    native_units: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fitpars = tuple(str(name) for name in self.fitpars)
        design = np.asarray(self.design, dtype=float)
        if design.ndim != 2 or design.shape[1] != len(fitpars):
            raise ValueError(
                f"design must have shape (n, {len(fitpars)}) for fitpars "
                f"{fitpars}; got {design.shape}"
            )
        theta = {name: str(self.theta_exact.get(name, "0.0")) for name in fitpars}
        units = {
            name: str(self.native_units.get(name, UNKNOWN_UNIT)) for name in fitpars
        }
        object.__setattr__(self, "fitpars", fitpars)
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "theta_exact", theta)
        object.__setattr__(self, "native_units", units)

    @classmethod
    def from_design(
        cls,
        *,
        fitpars,
        design,
        theta_exact: Mapping[str, str] | None = None,
        native_units: Mapping[str, str] | None = None,
    ) -> LinearModel:
        return cls(
            fitpars=tuple(fitpars),
            design=design,
            theta_exact=dict(theta_exact or {}),
            native_units=dict(native_units or {}),
        )

    def reference_theta(self) -> np.ndarray:
        """float64 reference vector, each entry the correctly rounded decimal."""
        with localcontext() as ctx:
            ctx.prec = 60
            return np.array(
                [float(Decimal(self.theta_exact[name])) for name in self.fitpars],
                dtype=float,
            )


class LinearModelEngine:
    """This package's reference ``TimingEngine`` over a :class:`LinearModel`.

    ``residual_delta(δ) = -design @ δ``; every fit parameter is identically
    linear. ``residual_centering`` is a one-entry mapping unless the caller
    supplies a mapping; a caller that knows nothing leaves it ``"unknown"``.
    """

    engine_name = "linear"
    nonlinear_params: str | None = None

    def __init__(
        self,
        model: LinearModel,
        *,
        residual_centering: (
            ResidualCentering | Mapping[str, ResidualCentering] | None
        ) = None,
        key: str = "single",
    ):
        self._model = model
        self.fitpars = tuple(model.fitpars)
        self.native_units = dict(model.native_units)
        if residual_centering is None:
            residual_centering = ResidualCentering(stored_residuals="unknown")
        if isinstance(residual_centering, ResidualCentering):
            residual_centering = {key: residual_centering}
        self.residual_centering: dict[str, ResidualCentering] = dict(residual_centering)
        validate_engine_residual_centering(self)

    @classmethod
    def from_linear_model(
        cls,
        model: LinearModel,
        *,
        residual_centering: (
            ResidualCentering | Mapping[str, ResidualCentering] | None
        ) = None,
        **_ignored: Any,
    ) -> LinearModelEngine:
        return cls(model, residual_centering=residual_centering)

    @property
    def model(self) -> LinearModel:
        return self._model

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def residual_delta(self, delta_theta) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.shape != (len(self.fitpars),):
            raise ValueError(
                f"delta_theta must have shape ({len(self.fitpars)},); "
                f"got {delta.shape}"
            )
        return -(self._model.design @ delta)

    def design_matrix(self, params: Any | None = None) -> np.ndarray:
        _ = params
        return self._model.design

    def residual_jacobian(self) -> np.ndarray:
        return -self._model.design

    def identically_linear_fitpars(self) -> frozenset[str]:
        return frozenset(self.fitpars)

    def binary_chart_capability(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} fitpars={list(self.fitpars)}>"


__all__ = [
    "UNKNOWN_UNIT",
    "LinearContribution",
    "LinearModel",
    "LinearModelEngine",
    "LinearTimingEngine",
    "is_exact_linear_param",
    "linear_engine",
    "validate_engine_against_pulsar",
    "validate_engine_residual_centering",
    "validate_engine_shapes",
    "validate_engine_zero_delta",
    "validate_pulsar_surface",
    "zero_delta_tolerance",
]
