"""Immutable fixed-expansion timing linearization record (§5.2).

``TimingLinearization`` freezes the exact engine waveform and its coordinate
Jacobians at one fixed expansion point ``z_e`` (in prior-normal ``z``). The
sampled block ``W_s`` generalizes the ``local_timing_block`` Jacobian from the
engine reference (``z = 0``) to an arbitrary expansion; the marginalized-``z``
block (``W_m``, ``c_m``) is present in the record but empty until the z-prior
adapters exist (Stage 5). Delta-flat columns come from the engine design matrix
(§5.4) and are not part of this record.

The waveform is ``d(z) = -residual_delta(delta(z))`` — the delay subtracted from
the reference residual (sign as in ``local_timing_block``). Derivatives use the
exact engine residual path (``jax.jacfwd`` for a JAX engine, a five-point
stencil otherwise); never ``pulsar.Mmat``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

ExpansionSource = Literal["engine_reference", "prior_center", "explicit_delta", "refined"]


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
            self, "z_expansion",
            _frozen_float(self.z_expansion, name="z_expansion", shape=(k_s + k_m,)))
        object.__setattr__(
            self, "delta_expansion",
            _frozen_float(self.delta_expansion, name="delta_expansion", shape=(k_s + k_m,)))
        object.__setattr__(
            self, "sampled_z_expansion",
            _frozen_float(self.sampled_z_expansion, name="sampled_z_expansion", shape=(k_s,)))
        object.__setattr__(
            self, "sampled_waveform_expansion",
            _frozen_float(self.sampled_waveform_expansion,
                          name="sampled_waveform_expansion", shape=(n_toa,)))
        object.__setattr__(
            self, "sampled_basis",
            _frozen_float(self.sampled_basis, name="sampled_basis", shape=(n_toa, k_s)))
        object.__setattr__(
            self, "marginalized_z_basis",
            _frozen_float(self.marginalized_z_basis,
                          name="marginalized_z_basis", shape=(n_toa, k_m)))
        object.__setattr__(
            self, "marginalized_z_intercept",
            _frozen_float(self.marginalized_z_intercept,
                          name="marginalized_z_intercept", shape=(n_toa,)))

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


def _waveform_of_sampled_z(engine, space, idx_sampled, nfit, xp):
    """Return ``d(z_s) = -residual_delta(delta(z_s))`` as an ``xp`` callable."""

    idx = xp.asarray(np.asarray(idx_sampled, dtype=int))

    def d_of_zs(z_s):
        delta_sampled = space.delta_from_z(z_s, xp)
        full = xp.zeros((nfit,), dtype=delta_sampled.dtype).at[idx].set(delta_sampled)
        return -engine.residual_delta_jax(full)

    return d_of_zs


def _stencil_jacobian(f, z_e, *, h: float) -> np.ndarray:
    """Five-point stencil Jacobian of ``f: R^k -> R^n`` at ``z_e`` (§5.3)."""
    z_e = np.asarray(z_e, dtype=float)
    k = z_e.shape[0]
    cols = []
    for j in range(k):
        e = np.zeros(k)
        e[j] = 1.0
        col = (
            f(z_e - 2 * h * e)
            - 8 * f(z_e - h * e)
            + 8 * f(z_e + h * e)
            - f(z_e + 2 * h * e)
        ) / (12.0 * h)
        cols.append(np.asarray(col, dtype=float))
    return np.stack(cols, axis=1)


def build_linearization(
    *,
    engine,
    plan,
    space,
    delta_expansion: np.ndarray,
    source: ExpansionSource,
) -> TimingLinearization:
    """Build the fixed linearization at ``delta_expansion`` (proper/sampled order).

    z-prior marginalization is not yet wired, so the proper set is exactly the
    sampled set and the marginalized-z blocks are empty.
    """
    from .protocols import JaxTimingEngine

    sampled_names = tuple(plan.sampled)
    k_s = len(sampled_names)
    nfit = len(plan.fitpars)

    delta_e = np.asarray(delta_expansion, dtype=float)
    if delta_e.shape != (k_s,):
        raise ValueError(
            f"delta_expansion has shape {delta_e.shape}, expected {(k_s,)} "
            "(proper/sampled axes in fitpar order)"
        )

    if k_s == 0:
        # No sampled axes: derive n_toa from a zero-delta residual.
        n_toa = int(np.asarray(engine.residual_delta(np.zeros(nfit))).shape[0])
        return TimingLinearization(
            proper_names=(), sampled_names=(), marginalized_z_names=(),
            z_expansion=np.zeros(0), delta_expansion=np.zeros(0),
            sampled_z_expansion=np.zeros(0),
            sampled_waveform_expansion=np.zeros(n_toa),
            sampled_basis=np.zeros((n_toa, 0)),
            marginalized_z_basis=np.zeros((n_toa, 0)),
            marginalized_z_intercept=np.zeros(n_toa),
            source=source)

    bad = _outside_prior_interior(space.prior_bijector.priors, delta_e, sampled_names)
    if bad:
        raise ExpansionOutsidePriorInteriorError(bad, source)
    z_e = np.asarray(space.z_from_delta(delta_e, np), dtype=float)
    if not np.all(np.isfinite(z_e)):
        raise ExpansionOutsidePriorInteriorError(
            [sampled_names[i] for i in range(k_s) if not np.isfinite(z_e[i])], source
        )

    if isinstance(engine, JaxTimingEngine):
        import jax
        import jax.numpy as jnp

        d_of_zs = _waveform_of_sampled_z(engine, space, plan.idx_sampled, nfit, jnp)
        d_e = np.asarray(d_of_zs(jnp.asarray(z_e)), dtype=float)
        W_s = np.asarray(jax.jacfwd(d_of_zs)(jnp.asarray(z_e)), dtype=float)
    else:
        idx = np.asarray(plan.idx_sampled, dtype=int)

        def d_np(z_s):
            delta_sampled = np.asarray(space.delta_from_z(z_s, np), dtype=float)
            full = np.zeros(nfit)
            full[idx] = delta_sampled
            return -np.asarray(engine.residual_delta(full), dtype=float)

        d_e = d_np(z_e)
        W_a = _stencil_jacobian(d_np, z_e, h=1e-4)
        W_b = _stencil_jacobian(d_np, z_e, h=5e-5)
        scale = np.maximum(np.abs(W_a), 1e-300)
        if np.max(np.abs(W_a - W_b) / scale) > 1e-5:
            raise ValueError(
                "five-point stencil timing Jacobian did not converge to relative "
                "tolerance 1e-5 across h=1e-4 and h=5e-5"
            )
        W_s = W_a

    n_toa = int(np.asarray(d_e).shape[0])
    return TimingLinearization(
        proper_names=sampled_names,
        sampled_names=sampled_names,
        marginalized_z_names=(),
        z_expansion=z_e,
        delta_expansion=delta_e,
        sampled_z_expansion=z_e,
        sampled_waveform_expansion=d_e,
        sampled_basis=W_s,
        marginalized_z_basis=np.zeros((n_toa, 0)),
        marginalized_z_intercept=np.zeros(n_toa),
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
