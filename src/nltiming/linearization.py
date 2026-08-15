"""Immutable fixed-expansion timing linearization record (§5.2).

``TimingLinearization`` freezes the exact engine waveform and its coordinate
Jacobians at one fixed expansion point ``z_e`` (in prior-normal ``z``). The
sampled block ``W_s`` generalizes the ``local_timing_block`` Jacobian from the
engine reference (``z = 0``) to an arbitrary expansion; the marginalized-``z``
block (``W_m``, ``c_m``) is present in the record but empty until the z-prior
adapters exist (Stage 5). Delta-flat columns come from the engine design matrix
(§5.4) and are not part of this record.

The waveform is ``d(z) = -residual_delta(delta(z))`` — the delay subtracted from
the reference residual (sign as in ``local_timing_block``). The delay *value*
always comes from the engine residual. The delay *tangent* ``W`` follows
``derivative_method``: analytic mode is the sampling-frame design matrix times
the prior-coordinate Jacobian ``∂δ/∂z``; autodiff mode is ``jax.jacfwd`` of the
composed residual. There is no finite-difference route.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .frames import EngineDeltaMap

ExpansionSource = Literal[
    "engine_reference", "prior_center", "explicit_delta", "refined"
]


def _frozen_float(array, *, name: str, shape=None) -> np.ndarray:
    arr = np.array(np.asarray(array, dtype=float))
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    arr.setflags(write=False)
    return arr


def _array_digest(hasher, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(np.asarray(array, dtype=float))
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(arr.tobytes())


@dataclass(frozen=True)
class TimingLinearization:
    """Fixed affine surrogate of the timing waveform at one expansion (§5.2)."""

    proper_names: tuple[str, ...]
    sampled_names: tuple[str, ...]
    marginalized_z_names: tuple[str, ...]
    z_expansion: np.ndarray  # proper order
    delta_expansion: np.ndarray  # proper order
    sampled_z_expansion: np.ndarray
    sampled_waveform_expansion: np.ndarray  # d_e, (n_toa,)
    sampled_basis: np.ndarray  # W_s, (n_toa, k_s)
    marginalized_z_basis: np.ndarray  # W_m, (n_toa, k_m)
    marginalized_z_intercept: np.ndarray  # c_m, (n_toa,)
    source: ExpansionSource

    def __post_init__(self) -> None:
        k_s = len(self.sampled_names)
        k_m = len(self.marginalized_z_names)
        n_toa = int(np.asarray(self.sampled_waveform_expansion).shape[0])
        object.__setattr__(
            self,
            "z_expansion",
            _frozen_float(self.z_expansion, name="z_expansion", shape=(k_s + k_m,)),
        )
        object.__setattr__(
            self,
            "delta_expansion",
            _frozen_float(
                self.delta_expansion, name="delta_expansion", shape=(k_s + k_m,)
            ),
        )
        object.__setattr__(
            self,
            "sampled_z_expansion",
            _frozen_float(
                self.sampled_z_expansion, name="sampled_z_expansion", shape=(k_s,)
            ),
        )
        object.__setattr__(
            self,
            "sampled_waveform_expansion",
            _frozen_float(
                self.sampled_waveform_expansion,
                name="sampled_waveform_expansion",
                shape=(n_toa,),
            ),
        )
        object.__setattr__(
            self,
            "sampled_basis",
            _frozen_float(self.sampled_basis, name="sampled_basis", shape=(n_toa, k_s)),
        )
        object.__setattr__(
            self,
            "marginalized_z_basis",
            _frozen_float(
                self.marginalized_z_basis,
                name="marginalized_z_basis",
                shape=(n_toa, k_m),
            ),
        )
        object.__setattr__(
            self,
            "marginalized_z_intercept",
            _frozen_float(
                self.marginalized_z_intercept,
                name="marginalized_z_intercept",
                shape=(n_toa,),
            ),
        )

    @property
    def n_toa(self) -> int:
        return int(self.sampled_waveform_expansion.shape[0])

    def transport_effective_residual(self, raw_residual: np.ndarray) -> np.ndarray:
        """Fixed dynamic-transport residual anchored at the expansion (§5.7).

        ``y - c_m - d_e + W_s z_s,e``. At the engine reference (``z_e = 0`` for a
        symmetric/Gaussian chart) this reduces to the raw residual.
        """
        return (
            np.asarray(raw_residual, dtype=float)
            - self.marginalized_z_intercept
            - self.sampled_waveform_expansion
            + self.sampled_basis @ self.sampled_z_expansion
        )

    def metadata(self) -> dict[str, object]:
        return {
            "proper_names": list(self.proper_names),
            "sampled_names": list(self.sampled_names),
            "marginalized_z_names": list(self.marginalized_z_names),
            "source": self.source,
            "n_toa": self.n_toa,
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(
            json.dumps(
                {
                    "proper_names": list(self.proper_names),
                    "sampled_names": list(self.sampled_names),
                    "marginalized_z_names": list(self.marginalized_z_names),
                    "source": self.source,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for arr in (
            self.z_expansion,
            self.delta_expansion,
            self.sampled_z_expansion,
            self.sampled_waveform_expansion,
            self.sampled_basis,
            self.marginalized_z_basis,
            self.marginalized_z_intercept,
        ):
            _array_digest(hasher, arr)
        return "sha256:" + hasher.hexdigest()


def _outside_prior_interior(priors, delta_e, names) -> list[str]:
    """Names whose expansion delta is on/outside their proper prior support (§5.3).

    A Gaussian (``normal``) chart has unbounded support and never fails.
    """
    bad: list[str] = []
    for name, prior, d in zip(names, priors, np.asarray(delta_e, dtype=float)):
        family = prior.family
        if family == "normal":
            continue
        if family in ("uniform", "truncated_normal"):
            if not (prior.lower < d < prior.upper):
                bad.append(name)
        elif family == "log_uniform":
            absolute = d + prior.offset
            if not (prior.lower < absolute < prior.upper):
                bad.append(name)
    return bad


def _waveform_of_z(engine, space, engine_map, xp):
    """d(z) = -residual_delta(engine_map(delta(z))) over ``space``'s axes."""

    def d_of_z(z):
        delta_s = space.delta_from_z(z, xp)
        return -engine.residual_delta_jax(engine_map.full_engine_delta(delta_s, xp))

    return d_of_z


def _waveform_value(engine, space, engine_map, z_e) -> np.ndarray:
    """NumPy evaluation of ``d(z_e)`` through the sampling→engine map."""
    delta_s = np.asarray(space.delta_from_z(z_e, np), dtype=float)
    full = engine_map.full_engine_delta(delta_s, np)
    return -np.asarray(engine.residual_delta(full), dtype=float)


def _analytic_proper_jacobian(
    design_matrix, proper_axes, proper_space, z_e
) -> np.ndarray:
    """W = M_s[:, proper] * diag(∂δ/∂z) at ``z_e`` (analytic route)."""
    M = np.asarray(design_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"design_matrix must be 2D, got shape {M.shape}")
    slots = [a.fitpar_index for a in proper_axes]
    if any(s < 0 or s >= M.shape[1] for s in slots):
        raise ValueError(
            f"proper-axis fitpar indices {slots} exceed design_matrix "
            f"column count {M.shape[1]}"
        )
    d_delta_d_z = np.asarray(
        proper_space.prior_bijector.jacobian_diag_delta_from_z(z_e, np),
        dtype=float,
    )
    if d_delta_d_z.shape != (len(slots),):
        raise ValueError(
            f"prior Jacobian has shape {d_delta_d_z.shape}, expected {(len(slots),)}"
        )
    return M[:, slots] * d_delta_d_z[None, :]


def build_linearization(
    *,
    engine,
    plan,
    proper_space,
    delta_expansion: np.ndarray,
    source: ExpansionSource,
    charts: tuple = (),
    derivative_method: str = "analytic",
    design_matrix: np.ndarray | None = None,
) -> TimingLinearization:
    """Build the fixed linearization at ``delta_expansion`` (proper order, §5.2).

    ``d(z)`` is the exact engine waveform. Its proper-axis tangent ``W`` follows
    ``derivative_method`` — the same knob as ``engine_design_matrix`` /
    ``design_matrix``. Analytic mode uses the sampling-frame design matrix and
    ``∂δ/∂z``; autodiff mode uses ``jax.jacfwd`` through the sampling→engine
    composition. Columns are split into the sampled block ``W_s`` and the
    z-marginalized block ``W_m`` (with intercept ``c_m = -W_m z_m,e``).
    Delta-flat axes are held at zero (their improper columns come from the
    engine design matrix, §5.4).
    """
    from .protocols import JaxTimingEngine

    if derivative_method not in ("analytic", "autodiff"):
        raise ValueError(
            "derivative_method must be 'analytic' or 'autodiff'; "
            f"got {derivative_method!r}"
        )

    proper_axes = [
        a for a in plan.axes if a.disposition in ("sample", "marginalize_z_prior")
    ]
    proper_names = tuple(a.name for a in proper_axes)
    sampled_cols = [i for i, a in enumerate(proper_axes) if a.disposition == "sample"]
    zm_cols = [i for i, a in enumerate(proper_axes) if a.disposition != "sample"]
    sampled_names = tuple(proper_names[i] for i in sampled_cols)
    zm_names = tuple(proper_names[i] for i in zm_cols)
    k_p = len(proper_names)
    nfit = len(plan.fitpars)
    engine_map = EngineDeltaMap.for_proper(plan, charts)

    delta_e = np.asarray(delta_expansion, dtype=float)
    if delta_e.shape != (k_p,):
        raise ValueError(
            f"delta_expansion has shape {delta_e.shape}, expected {(k_p,)} "
            "(proper axes in fitpar order)"
        )

    if k_p == 0:
        n_toa = int(np.asarray(engine.residual_delta(np.zeros(nfit))).shape[0])
        return TimingLinearization(
            proper_names=(),
            sampled_names=(),
            marginalized_z_names=(),
            z_expansion=np.zeros(0),
            delta_expansion=np.zeros(0),
            sampled_z_expansion=np.zeros(0),
            sampled_waveform_expansion=np.zeros(n_toa),
            sampled_basis=np.zeros((n_toa, 0)),
            marginalized_z_basis=np.zeros((n_toa, 0)),
            marginalized_z_intercept=np.zeros(n_toa),
            source=source,
        )

    bad = _outside_prior_interior(
        proper_space.prior_bijector.priors, delta_e, proper_names
    )
    if bad:
        raise ExpansionOutsidePriorInteriorError(bad, source)
    z_e = np.asarray(proper_space.z_from_delta(delta_e, np), dtype=float)
    if not np.all(np.isfinite(z_e)):
        raise ExpansionOutsidePriorInteriorError(
            [proper_names[i] for i in range(k_p) if not np.isfinite(z_e[i])], source
        )

    d_e = _waveform_value(engine, proper_space, engine_map, z_e)
    if derivative_method == "analytic":
        if design_matrix is None:
            raise ValueError(
                "derivative_method='analytic' requires the sampling-frame "
                "design_matrix to form the proper-axis linearization"
            )
        W = _analytic_proper_jacobian(design_matrix, proper_axes, proper_space, z_e)
    else:
        if not isinstance(engine, JaxTimingEngine):
            raise ValueError(
                "derivative_method='autodiff' requires a JAX-capable engine "
                "exposing residual_delta_jax(); got "
                f"{type(engine).__name__}. Finite-difference linearization "
                "is not supported."
            )
        import jax
        import jax.numpy as jnp

        d_of_z = _waveform_of_z(engine, proper_space, engine_map, jnp)
        W = np.asarray(jax.jacfwd(d_of_z)(jnp.asarray(z_e)), dtype=float)

    n_toa = int(np.asarray(d_e).shape[0])
    if W.shape != (n_toa, k_p):
        raise ValueError(
            f"proper-axis Jacobian has shape {W.shape}, expected {(n_toa, k_p)}"
        )
    W_s = W[:, sampled_cols] if sampled_cols else np.zeros((n_toa, 0))
    W_m = W[:, zm_cols] if zm_cols else np.zeros((n_toa, 0))
    z_s_e = z_e[sampled_cols] if sampled_cols else np.zeros(0)
    z_m_e = z_e[zm_cols] if zm_cols else np.zeros(0)
    c_m = -(W_m @ z_m_e) if zm_cols else np.zeros(n_toa)

    return TimingLinearization(
        proper_names=proper_names,
        sampled_names=sampled_names,
        marginalized_z_names=zm_names,
        z_expansion=z_e,
        delta_expansion=delta_e,
        sampled_z_expansion=z_s_e,
        sampled_waveform_expansion=d_e,
        sampled_basis=W_s,
        marginalized_z_basis=W_m,
        marginalized_z_intercept=c_m,
        source=source,
    )


class ExpansionOutsidePriorInteriorError(ValueError):
    """A requested expansion delta lies on/outside a proper prior boundary (§5.3).

    Choose ``TimingExpansionSpec.prior_center()``, a complete
    ``explicit_delta(...)``, or revise the physical prior. nltiming never clips or
    nudges the expansion into the box.
    """

    def __init__(self, axes, source: str = "engine_reference"):
        self.axes = tuple(axes)
        self.source = source
        detail = (
            "engine-reference expansion (delta=0)"
            if source == "engine_reference"
            else f"{source} expansion"
        )
        super().__init__(
            f"{detail} lies on/outside the prior interior for axes "
            f"{list(self.axes)}; select prior_center(), a complete "
            "explicit_delta(...), or revise the physical prior."
        )
