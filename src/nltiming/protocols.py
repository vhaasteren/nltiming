"""Protocols for pulsars and timing engines.

Stack layering:
- **Timing engine** — residuals and design matrix (JUG, PINT, tempo2).
- **Likelihood interface** — Enterprise / Discovery signal assembly (``likelihoods/*``).
- **Sampler** — user-owned posterior driver (PTMCMC, NumPyro NUTS, …); not imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class BinaryChartCapability:
    """Engine self-declaration consumed by physical-chart candidacy/activation.

    kepler_convention: Kepler parameter convention family ("dd" covers
        DD/T2(DD-mode)/DDH; anything else is not a chart candidate in v1).
    epoch_shift_exact: whether (OM+360 deg, T0+PB) is an EXACT identity for
        this model as configured — False whenever any secular or derived
        epoch-coupled evolution is active (explicit OMDOT/PBDOT/EDOT/A1DOT,
        DDGR-derived terms, T2 epoch dependence, ...).
    secular_terms: canonical names of the epoch-coupled terms the engine
        knows to be active FOR THIS GROUP (informational; goes to the
        manifest).
    origin_certified: **empirical** certification that this backend passed
        the §12.6 real-DD-engine full-likelihood origin gate (no 1/e
        blow-through, leapfrog stability, Discovery/Enterprise density) — a
        regression statement scoped to (backend implementation + version,
        binary model configuration, float precision, differentiation path),
        NOT a proof of bounded behavior arbitrarily near an included origin.
        (The surrogate finiteness check is only a weak gate; the strong cert
        is §12.6, not the §12.3 coordinate-map suite.) Set True per adapter
        only by the PR that lands its passing certification run — never by
        default — with ``certification_ref`` pointing at that run/PR.
    supports_domain: the backend accepts every point of the chart's declared
        physical domain (e < 1) without internal clamping or NaN.
    """

    kepler_convention: str
    epoch_shift_exact: bool
    secular_terms: tuple[str, ...]
    origin_certified: bool
    supports_domain: bool
    certification_ref: str | None = None

    def __post_init__(self) -> None:
        # Review fix: certification without auditable provenance is invalid.
        if self.origin_certified and not self.certification_ref:
            raise ValueError(
                "BinaryChartCapability: origin_certified=True requires a "
                "certification_ref (the recorded §12.6 certification run/PR)"
            )


@runtime_checkable
class EphemerisExtras(Protocol):
    """Optional ephemeris-related pulsar fields used by some signal blocks."""

    @property
    def pos_t(self) -> np.ndarray: ...

    @property
    def planetssb(self) -> np.ndarray: ...

    @property
    def sunssb(self) -> np.ndarray: ...

    @property
    def theta(self) -> float: ...

    @property
    def phi(self) -> float: ...

    @property
    def pdist(self) -> tuple[float, float] | Any: ...

    @property
    def dm(self) -> float | Any: ...

    @property
    def dmx(self) -> Mapping[str, Any]: ...

    @property
    def telescope(self) -> np.ndarray: ...


@runtime_checkable
class PulsarData(Protocol):
    """Duck-typed frozen pulsar-data surface consumed by likelihood interfaces."""

    name: str
    fitpars: list[str] | tuple[str, ...]

    @property
    def toas(self) -> np.ndarray: ...

    @property
    def residuals(self) -> np.ndarray: ...

    @property
    def toaerrs(self) -> np.ndarray: ...

    @property
    def freqs(self) -> np.ndarray: ...

    @property
    def Mmat(self) -> np.ndarray: ...

    @property
    def flags(self) -> Mapping[str, np.ndarray]: ...

    @property
    def backend_flags(self) -> np.ndarray: ...


@runtime_checkable
class TimingEngine(Protocol):
    """Timing engine over theta-native engines in canonical pulsar order."""

    fitpars: tuple[str, ...]
    native_units: Mapping[str, str]

    def reference_theta(self) -> np.ndarray: ...

    def reference_theta_exact(self) -> Mapping[str, str]: ...

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray: ...

    def design_matrix(self, params: Any | None = None) -> np.ndarray: ...


@runtime_checkable
class JaxTimingEngine(TimingEngine, Protocol):
    """JAX-capable timing engine for traced residuals on the NumPyro NUTS tier."""

    def residual_delta_jax(self, delta_theta: Any) -> Any: ...

    def precision_critical_fitpars(self) -> frozenset[str]: ...


@runtime_checkable
class TimingPulsar(PulsarData, Protocol):
    """Pulsar protocol: frozen arrays plus timing-engine accessors."""

    def pint_model(self) -> Any: ...

    def timing_engine(self, engines="jug") -> TimingEngine: ...

    def can_use_engines(self, engines="jug") -> bool: ...

    def state_id(self) -> str | None: ...


# Descriptive alias retained for integrations that use the original name.
EnterprisePulsarLike = PulsarData
