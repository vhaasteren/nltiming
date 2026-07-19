"""Whitening configuration, local posterior metric, and transport records.

This module owns the provenance-carrying inputs to the posterior whitening
transform (§5.1) and the immutable transport record that a conditioned
:class:`~nltiming.nonlinear_timing_model.TimingContext` stores (§7.3).

- :class:`WhiteningConfig` is the frozen configuration a
  ``NonLinearTimingModel`` carries (reference-noise class, expansion point,
  origin policy) — no numerical floor, no likelihood-only mode (§5.4).
- :class:`LocalPosteriorMetric` is the typed, fingerprinted metric a likelihood
  interface (or the built-in TOA-errors/frozen-white helpers) hands to
  ``ctx.with_transport``; it declares which precision model produced it.
- :class:`StaticTransportRecord` is the serializable static-affine transport
  the conditioned context exposes and the run manifest persists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Provenance classes (§5.1). These map one-to-one onto the discovery transport
# reference-noise constructors: class 1 -> reference_noise(psr); classes 2-3 ->
# reference_noise_frozen(kernel, params0); class 4 is the dynamic joint
# transport owned by Discovery (Track J), not a static nltiming metric.
REFERENCE_NOISE_TOA_ERRORS = "toa_errors"
REFERENCE_NOISE_FROZEN_WHITE = "frozen_white"
REFERENCE_NOISE_ASSEMBLED = "assembled_likelihood"

_REFERENCE_NOISE_CLASSES = frozenset(
    {
        REFERENCE_NOISE_TOA_ERRORS,
        REFERENCE_NOISE_FROZEN_WHITE,
        REFERENCE_NOISE_ASSEMBLED,
    }
)
_ORIGIN_POLICIES = frozenset({"auto", "reference", "local_posterior"})
_EXPANSION_POINTS = frozenset({"reference"})


@dataclass(frozen=True)
class WhiteningConfig:
    """Frozen whitening configuration (§5.1).

    ``reference_noise`` is the *default* provenance class used when a likelihood
    interface does not supply its own :class:`LocalPosteriorMetric`. The
    posterior metric ``F_z + I`` is the only metric; there is deliberately no
    ``metric=`` choice and no ``numerical_floor`` (§5.4).
    """

    reference_noise: str = REFERENCE_NOISE_TOA_ERRORS
    expansion_point: str = "reference"
    origin: str = "auto"

    def __post_init__(self) -> None:
        if self.reference_noise not in _REFERENCE_NOISE_CLASSES:
            raise ValueError(
                "WhiteningConfig.reference_noise must be one of "
                f"{sorted(_REFERENCE_NOISE_CLASSES)}; got {self.reference_noise!r}"
            )
        if self.expansion_point not in _EXPANSION_POINTS:
            raise ValueError(
                "WhiteningConfig.expansion_point must be 'reference'; "
                f"got {self.expansion_point!r}"
            )
        if self.origin not in _ORIGIN_POLICIES:
            raise ValueError(
                "WhiteningConfig.origin must be one of "
                f"{sorted(_ORIGIN_POLICIES)}; got {self.origin!r}"
            )

    def as_dict(self) -> dict[str, str]:
        return {
            "reference_noise": self.reference_noise,
            "expansion_point": self.expansion_point,
            "origin": self.origin,
        }


def _array_digest(hasher: "hashlib._Hash", array: np.ndarray | None) -> None:
    if array is None:
        hasher.update(b"\x00none")
        return
    arr = np.ascontiguousarray(np.asarray(array, dtype=float))
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(arr.tobytes())


@dataclass(frozen=True)
class LocalPosteriorMetric:
    """Immutable, provenance-carrying local posterior metric input (§5.1).

    ``fisher_delta`` is the Schur likelihood Fisher in the sampled-block delta
    coordinates (already marginalized over analytically marginalized timing
    columns); ``expansion_delta`` is where it was evaluated (the reference, all
    zeros, by default). ``score_delta`` is the optional ``d(-log L)/d(delta)``
    at the reference — supplying it enables local-posterior centering (§4.2).
    ``reference_noise`` declares the provenance class; a raw ``toa_errors``
    metric is only an approximate preconditioner and must say so.
    """

    fisher_delta: np.ndarray
    sampled: tuple[str, ...]
    expansion_delta: np.ndarray
    reference_noise: str
    source: str
    source_description: str
    score_delta: np.ndarray | None = None
    noise_snapshot: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fisher = np.asarray(self.fisher_delta, dtype=float)
        ndim = len(self.sampled)
        if fisher.shape != (ndim, ndim):
            raise ValueError(
                f"fisher_delta shape {fisher.shape} does not match "
                f"{ndim} sampled parameters {self.sampled}"
            )
        if np.asarray(self.expansion_delta, dtype=float).shape != (ndim,):
            raise ValueError("expansion_delta must have one entry per sampled param")
        if self.score_delta is not None:
            if np.asarray(self.score_delta, dtype=float).shape != (ndim,):
                raise ValueError("score_delta must have one entry per sampled param")
        if self.reference_noise not in _REFERENCE_NOISE_CLASSES:
            raise ValueError(f"unknown reference_noise class {self.reference_noise!r}")

    @property
    def approximate(self) -> bool:
        """True when this metric only approximates the full sampled likelihood."""
        return self.reference_noise != REFERENCE_NOISE_ASSEMBLED

    def fingerprint(self) -> str:
        """Stable digest of the metric's precision model and snapshot (§7.4)."""
        hasher = hashlib.sha256()
        meta = {
            "schema": "nlt-local-posterior-metric-v1",
            "sampled": list(self.sampled),
            "reference_noise": self.reference_noise,
            "source": self.source,
            "source_description": self.source_description,
            "has_score": self.score_delta is not None,
            "noise_snapshot": self.noise_snapshot,
        }
        hasher.update(
            json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        _array_digest(hasher, self.fisher_delta)
        _array_digest(hasher, self.expansion_delta)
        _array_digest(hasher, self.score_delta)
        return "sha256:" + hasher.hexdigest()

    def provenance(self) -> dict[str, Any]:
        """Manifest ``metric_source`` payload (no raw arrays)."""
        return {
            "reference_noise": self.reference_noise,
            "source": self.source,
            "source_description": self.source_description,
            "approximate": self.approximate,
            "has_score": self.score_delta is not None,
            "noise_snapshot": self.noise_snapshot,
            "digest": self.fingerprint(),
        }


@dataclass(frozen=True)
class StaticTransportRecord:
    """Serializable static-affine transport record (§7.3).

    A static timing-only ``(C, c)`` is fixed when the context is conditioned and
    is independently decodable, so ``latent_decodable`` is always True here. The
    dynamic joint transport (Track J / §7.3) is a separate record type.
    """

    kind: str = "static_affine"
    latent_decodable: bool = True
    coordinate: str = "x"
    metric_source: dict[str, Any] = field(default_factory=dict)
    origin: str = "reference"
    expansion_point: str = "reference"
    guard_engaged: bool = False
    C_digest: str = ""
    z0_digest: str = ""

    def section(self) -> dict[str, Any]:
        """Manifest ``transport`` section payload."""
        return {
            "kind": self.kind,
            "latent_decodable": self.latent_decodable,
            "coordinate": self.coordinate,
            "metric_source": self.metric_source,
            "origin": self.origin,
            "expansion_point": self.expansion_point,
            "guard_engaged": self.guard_engaged,
            "C_digest": self.C_digest,
            "z0_digest": self.z0_digest,
        }

    def fingerprint(self) -> str:
        payload = self.section()
        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        )


@dataclass(frozen=True)
class DynamicTransportRecord:
    """Serializable dynamic (joint) transport record (Track J, §7.3).

    A dynamic transport ``q = mu(eta, theta_det) + L(eta)^-T xi`` depends on
    sampled hyperparameters, so ``xi`` alone has no physical meaning and the
    record is **not** latent-decodable. It captures the transport structure
    (block names/dimensions/order, reference-noise description, parameter
    dependencies, centering policy) plus the transport digest — never an opaque
    Python closure. Built from a Discovery ``Transport``'s ``diagnostics()`` and
    ``fingerprint()``.
    """

    transport_digest: str
    structure: dict[str, Any]
    dimension: int
    reference_noise: str
    centering: str
    parameter_dependencies: tuple[str, ...]
    kind: str = "dynamic_transport"
    latent_decodable: bool = False
    coordinate: str = "xi"

    def section(self) -> dict[str, Any]:
        """Manifest ``transport`` section payload (§7.3)."""
        return {
            "kind": self.kind,
            "latent_decodable": self.latent_decodable,
            "coordinate": self.coordinate,
            "dimension": self.dimension,
            "reference_noise": self.reference_noise,
            "centering": self.centering,
            "parameter_dependencies": list(self.parameter_dependencies),
            "structure": self.structure,
            "transport_digest": self.transport_digest,
        }

    def fingerprint(self) -> str:
        payload = self.section()
        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        )


def dynamic_transport_record(transport) -> DynamicTransportRecord:
    """Build a :class:`DynamicTransportRecord` from a Discovery ``Transport``.

    Consumes the transport's structural ``diagnostics()`` and ``fingerprint()``
    (Track J); no closure is serialized. Works for a single ``Transport`` (which
    reports ``blocks``) or an ``ArrayTransport`` (``per_pulsar``).
    """
    structure = transport.diagnostics()
    dimension = int(structure.get("dimension", 0))
    if "blocks" in structure:
        blocks = structure["blocks"]
        reference_noise = str(structure.get("reference_noise", ""))
        centering = "centered" if structure.get("center") else "uncentered"
    else:  # ArrayTransport
        per = structure.get("per_pulsar", [])
        blocks = [b for entry in per for b in entry.get("blocks", [])]
        reference_noise = str(per[0].get("reference_noise", "")) if per else ""
        centering = "centered" if (per and per[0].get("center")) else "uncentered"
    params = sorted({p for b in blocks for p in b.get("params", [])})
    return DynamicTransportRecord(
        transport_digest=transport.fingerprint(),
        structure=structure,
        dimension=dimension,
        reference_noise=reference_noise,
        centering=centering,
        parameter_dependencies=tuple(params),
    )


class OneAffineLayerError(ValueError):
    """Raised when a joint dynamic transport composes with a non-identity
    static timing transport (violates the one-affine-layer invariant, §5.5)."""


def assert_static_layer_identity(
    space, *, context: str = "joint dynamic sampling"
) -> None:
    """Assert the nltiming static affine layer is identity (§5.5).

    In joint full-basis Discovery mode the dynamic transport jointly maps timing
    ``z`` and sampled stochastic coefficients, so ``ParameterSpace.linear`` must
    be identity — composing both non-identity affine layers is forbidden.
    """
    linear = space.linear
    C = np.asarray(linear.C, dtype=float)
    z0 = np.asarray(linear.z0, dtype=float)
    ndim = C.shape[0]
    if not (np.allclose(C, np.eye(ndim)) and np.allclose(z0, 0.0)):
        raise OneAffineLayerError(
            f"{context} requires the nltiming static affine layer to be "
            "identity, but the timing ParameterSpace carries a non-identity "
            "(C, c). Build the joint context with whitening=None (the identity "
            "static layer) so exactly one non-identity affine transport is "
            "active (§4.4.1, §5.5)."
        )


def _column_digest(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    _array_digest(hasher, array)
    return "sha256:" + hasher.hexdigest()


def static_transport_record(
    linear,
    *,
    metric: LocalPosteriorMetric,
    coordinate: str,
    origin: str,
    expansion_point: str,
    guard_engaged: bool,
) -> StaticTransportRecord:
    """Build a :class:`StaticTransportRecord` from a conditioned linear layer."""
    return StaticTransportRecord(
        coordinate=coordinate,
        metric_source=metric.provenance(),
        origin=origin,
        expansion_point=expansion_point,
        guard_engaged=guard_engaged,
        C_digest=_column_digest(np.asarray(linear.C, dtype=float)),
        z0_digest=_column_digest(np.asarray(linear.z0, dtype=float)),
    )


def identity_transport_record(linear, *, coordinate: str) -> StaticTransportRecord:
    """Transport record for a no-op (``whitening=None``) identity static layer.

    ``whitening=None`` samples the prior-normal ``z`` directly, so no
    reference-noise metric is computed (avoiding an unnecessary — and possibly
    singular — Fisher solve); the record simply documents the identity map.
    """
    return StaticTransportRecord(
        coordinate=coordinate,
        metric_source={
            "reference_noise": "identity",
            "source": "static_layer_identity",
            "source_description": "identity static layer (whitening=None)",
            "approximate": False,
            "has_score": False,
            "noise_snapshot": {},
            "digest": "sha256:identity",
        },
        origin="reference",
        expansion_point="reference",
        guard_engaged=False,
        C_digest=_column_digest(np.asarray(linear.C, dtype=float)),
        z0_digest=_column_digest(np.asarray(linear.z0, dtype=float)),
    )


# --------------------------------------------------------------------------
# Built-in reference-noise metric constructors (§5.1 classes 1 and 2).
# --------------------------------------------------------------------------


def _schur_metric(
    *,
    pulsar,
    partition,
    variance,
    design_matrix,
    reference_noise,
    source,
    description,
    noise_snapshot=None,
) -> LocalPosteriorMetric:
    from .whitening import schur_delta_wls

    wls = schur_delta_wls(
        pulsar=pulsar,
        partition=partition,
        variance=np.asarray(variance, dtype=float),
        design_matrix=design_matrix,
    )
    ndim = len(partition.sampled)
    return LocalPosteriorMetric(
        fisher_delta=wls.fisher,
        sampled=tuple(partition.sampled),
        expansion_delta=np.zeros(ndim, dtype=float),
        reference_noise=reference_noise,
        source=source,
        source_description=description,
        score_delta=None,
        noise_snapshot=noise_snapshot or {},
    )


def toa_errors_metric(*, pulsar, partition, design_matrix=None) -> LocalPosteriorMetric:
    """Class 1 (§5.1): diagonal ``toaerrs**2`` reference metric.

    Only an approximate preconditioner for a correlated/marginalized
    likelihood; its provenance says so (``reference_noise='toa_errors'``).
    """
    variance = np.asarray(pulsar.toaerrs, dtype=float) ** 2
    return _schur_metric(
        pulsar=pulsar,
        partition=partition,
        variance=variance,
        design_matrix=design_matrix,
        reference_noise=REFERENCE_NOISE_TOA_ERRORS,
        source="toa_errors",
        description="diagonal toaerrs**2 reference (approximate preconditioner)",
    )


def frozen_white_metric(
    *, pulsar, partition, efac=1.0, equad=0.0, design_matrix=None
) -> LocalPosteriorMetric:
    """Class 2 (§5.1): EFAC/EQUAD white-noise reference at declared values."""
    from .whitening import _resolve_noise_value

    labels = np.asarray(pulsar.backend_flags)
    efac_v = _resolve_noise_value(efac, labels, 1.0)
    equad_v = _resolve_noise_value(equad, labels, 0.0)
    toaerrs = np.asarray(pulsar.toaerrs, dtype=float)
    variance = (efac_v * toaerrs) ** 2 + equad_v**2
    snapshot = {
        "efac": efac if isinstance(efac, dict) else float(efac),
        "equad": equad if isinstance(equad, dict) else float(equad),
    }
    return _schur_metric(
        pulsar=pulsar,
        partition=partition,
        variance=variance,
        design_matrix=design_matrix,
        reference_noise=REFERENCE_NOISE_FROZEN_WHITE,
        source="frozen_white",
        description="frozen EFAC/EQUAD white-noise reference",
        noise_snapshot=snapshot,
    )
