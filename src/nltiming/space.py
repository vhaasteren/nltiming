"""ParameterSpace single-source-of-truth for coordinate transforms."""

from __future__ import annotations

from collections import namedtuple
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .bijectors import AxisPrior, PriorBijector, WhiteningLinear
from .precision import ExactNativeRef
from .units import to_display


_STATIC_LAYERS = {"identity", "whitening"}


def coord_for_static_layer(static_layer: str) -> str:
    """Return the sampler coordinate for a static-layer choice (§4.4.1).

    ``identity`` (``whitening=None``) samples the prior-normal chart coordinate
    ``z``; ``whitening`` (a ``WhiteningConfig``) samples the whitened ``x``.
    """
    if static_layer == "identity":
        return "z"
    if static_layer == "whitening":
        return "x"
    raise ValueError(f"Unsupported static layer: {static_layer}")


DensityParts = namedtuple(
    "DensityParts",
    ["logprior_physical", "logjacobian", "logprior_coord", "loglike", "logpost"],
)


class ParameterSpace:
    """Own x <-> z <-> delta_theta transforms and related densities."""

    def __init__(
        self,
        names: tuple[str, ...],
        theta_ref: ExactNativeRef,
        prior_bijector: PriorBijector,
        linear: WhiteningLinear,
        *,
        static_layer: str = "identity",
        pint_model: Any | None = None,
    ):
        if static_layer not in _STATIC_LAYERS:
            raise ValueError(f"Unknown static layer: {static_layer}")
        if names != theta_ref.names:
            raise ValueError("names must match theta_ref names")
        self.names = names
        self.theta_ref = theta_ref
        self.prior_bijector = prior_bijector
        self.linear = linear
        self.static_layer = static_layer
        self.pint_model = pint_model
        self.ndim = len(names)
        if self.linear.C.shape != (self.ndim, self.ndim):
            raise ValueError("linear transform dimension mismatch")

    @classmethod
    def build(
        cls,
        theta_ref_mapping: dict[str, str | float | int],
        prior_bijector: PriorBijector | None = None,
        *,
        static_layer: str = "identity",
        linear_transform: WhiteningLinear | None = None,
        pint_model: Any | None = None,
    ) -> "ParameterSpace":
        if all(isinstance(v, str) for v in theta_ref_mapping.values()):
            exact = ExactNativeRef.from_mapping(theta_ref_mapping)  # type: ignore[arg-type]
        else:
            exact = ExactNativeRef.from_float_mapping(theta_ref_mapping)  # type: ignore[arg-type]
        names = exact.names
        if prior_bijector is None:
            prior_bijector = PriorBijector.from_normal(
                names=names,
                means=np.zeros(len(names), dtype=float),
                stds=np.ones(len(names), dtype=float),
            )
        if linear_transform is None:
            linear_transform = WhiteningLinear.identity(len(names))
        return cls(
            names=names,
            theta_ref=exact,
            prior_bijector=prior_bijector,
            linear=linear_transform,
            static_layer=static_layer,
            pint_model=pint_model,
        )

    def select(self, names) -> "ParameterSpace":
        """Extract the subspace over ``names`` (a subset of this space's axes).

        Preserves each axis's prior and exact reference. Requires an identity
        static layer — subspaces are selected before static whitening is built,
        so a non-block-diagonal ``WhiteningLinear`` can never be sliced here.
        """
        names = tuple(names)
        pos = {n: i for i, n in enumerate(self.names)}
        missing = [n for n in names if n not in pos]
        if missing:
            raise ValueError(f"select: unknown axes {missing}")
        C = np.asarray(self.linear.C, dtype=float)
        z0 = np.asarray(self.linear.z0, dtype=float)
        if C.size and not (
            np.allclose(C, np.eye(C.shape[0])) and np.allclose(z0, 0.0)
        ):
            raise ValueError(
                "ParameterSpace.select requires an identity static layer; select "
                "subspaces before static whitening is built (§4.5)"
            )
        idx = [pos[n] for n in names]
        sub_priors = tuple(self.prior_bijector.priors[i] for i in idx)
        sub_bijector = PriorBijector(names=names, priors=sub_priors)
        full = self.theta_ref.as_mapping()
        sub_ref = ExactNativeRef.from_mapping({n: full[n] for n in names})
        return ParameterSpace(
            names=names,
            theta_ref=sub_ref,
            prior_bijector=sub_bijector,
            linear=WhiteningLinear.identity(len(idx)),
            static_layer=self.static_layer,
            pint_model=self.pint_model,
        )

    def z_from_x(self, x, xp):
        return self.linear.z_from_x(x, xp)

    def x_from_z(self, z, xp):
        return self.linear.x_from_z(z, xp)

    def delta_from_z(self, z, xp):
        return self.prior_bijector.delta_from_z(z, xp)

    def z_from_delta(self, delta, xp):
        return self.prior_bijector.z_from_delta(delta, xp)

    def delta_from_coord(self, q, xp, coord: str = "x"):
        if coord == "delta":
            return xp.asarray(q)
        if coord == "z":
            return self.delta_from_z(q, xp)
        if coord == "x":
            z = self.z_from_x(q, xp)
            return self.delta_from_z(z, xp)
        raise ValueError(f"Unsupported coord: {coord}")

    def coord_from_delta(self, delta, xp, coord: str = "x"):
        if coord == "delta":
            return xp.asarray(delta)
        if coord == "z":
            return self.z_from_delta(delta, xp)
        if coord == "x":
            z = self.z_from_delta(delta, xp)
            return self.x_from_z(z, xp)
        raise ValueError(f"Unsupported coord: {coord}")

    def delta_from_cube(self, u, xp):
        return self.prior_bijector.delta_from_u(u, xp)

    def coord_from_cube(self, u, xp, coord: str = "x"):
        delta = self.delta_from_cube(u, xp)
        return self.coord_from_delta(delta, xp, coord=coord)

    def logprior_physical(self, delta, xp):
        return self.prior_bijector.logprior_physical(delta, xp)

    def logjacobian(self, q, xp, coord: str = "x"):
        if coord == "delta":
            return xp.asarray(0.0)
        if coord == "z":
            return self.prior_bijector.logabsdet_delta_from_z(q, xp)
        if coord == "x":
            z = self.z_from_x(q, xp)
            return self.prior_bijector.logabsdet_delta_from_z(z, xp) + xp.asarray(
                self.linear.logabsdet
            )
        raise ValueError(f"Unsupported coord: {coord}")

    def logprior_coord(self, q, xp, coord: str = "x"):
        delta = self.delta_from_coord(q, xp, coord=coord)
        return self.logprior_physical(delta, xp) + self.logjacobian(q, xp, coord=coord)

    def density_parts(
        self,
        q,
        loglike_fn: Callable[[np.ndarray], float],
        xp,
        coord: str = "x",
    ) -> DensityParts:
        delta = self.delta_from_coord(q, xp, coord=coord)
        logprior_physical = self.logprior_physical(delta, xp)
        logjacobian = self.logjacobian(q, xp, coord=coord)
        logprior_coord = logprior_physical + logjacobian
        loglike = xp.asarray(loglike_fn(delta))
        logpost = loglike + logprior_coord
        return DensityParts(
            logprior_physical=logprior_physical,
            logjacobian=logjacobian,
            logprior_coord=logprior_coord,
            loglike=loglike,
            logpost=logpost,
        )

    def theta_from_delta(self, delta) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (self.ndim,):
            raise ValueError("delta has wrong shape")
        with localcontext() as ctx:
            ctx.prec = 50
            out = []
            for ref_str, d in zip(self.theta_ref.values, delta, strict=True):
                out.append(float(Decimal(ref_str) + Decimal(str(float(d)))))
        return np.asarray(out, dtype=float)

    def delta_from_theta(self, theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.ndim,):
            raise ValueError("theta has wrong shape")
        with localcontext() as ctx:
            ctx.prec = 50
            out = []
            for ref_str, th in zip(self.theta_ref.values, theta, strict=True):
                out.append(float(Decimal(str(float(th))) - Decimal(ref_str)))
        return np.asarray(out, dtype=float)

    def to_physical(self, samples, units: str = "display", coord: str | None = None):
        if coord is None:
            coord = coord_for_static_layer(self.static_layer)
        arr = np.asarray(samples, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[-1] != self.ndim:
            raise ValueError("samples final dimension mismatch")
        delta = np.stack(
            [self.delta_from_coord(row, np, coord=coord) for row in arr], axis=0
        )
        theta = np.stack([self.theta_from_delta(row) for row in delta], axis=0)
        if units == "native":
            return {name: theta[:, i] for i, name in enumerate(self.names)}
        if units == "display":
            return {
                name: np.asarray(
                    to_display(name, theta[:, i], pint_model=self.pint_model),
                    dtype=float,
                )
                for i, name in enumerate(self.names)
            }
        raise ValueError(f"Unknown units mode: {units}")

    def metadata(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "theta_ref": self.theta_ref.as_mapping(),
            "static_layer": self.static_layer,
        }

    def fingerprint(self) -> str:
        """Stable identity of this decoder (names, refs, transform, priors, C, z0)."""
        hasher = hashlib.sha256()
        meta = {
            "schema": "nlt-parameter-space-v2",
            "names": list(self.names),
            "theta_ref": self.theta_ref.as_mapping(),
            "static_layer": self.static_layer,
            "priors": [
                {
                    "family": p.family,
                    "lower": p.lower,
                    "upper": p.upper,
                    "mean": p.mean,
                    "std": p.std,
                    "offset": p.offset,
                }
                for p in self.prior_bijector.priors
            ],
        }
        hasher.update(
            json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        hasher.update(np.ascontiguousarray(self.linear.C, dtype=np.float64).tobytes())
        hasher.update(np.ascontiguousarray(self.linear.z0, dtype=np.float64).tobytes())
        return "sha256:" + hasher.hexdigest()

    def save(self, path: str | Path) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(base) + ".npz",
            C=self.linear.C,
            z0=self.linear.z0,
        )
        prior_payload = []
        for p in self.prior_bijector.priors:
            prior_payload.append(
                {
                    "family": p.family,
                    "lower": p.lower,
                    "upper": p.upper,
                    "mean": p.mean,
                    "std": p.std,
                    "offset": p.offset,
                }
            )
        payload = self.metadata() | {"priors": prior_payload}
        (Path(str(base) + ".json")).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "ParameterSpace":
        base = Path(path)
        arrays = np.load(str(base) + ".npz")
        payload = json.loads(Path(str(base) + ".json").read_text(encoding="utf-8"))
        names = tuple(payload["names"])
        theta_ref = ExactNativeRef.from_mapping(payload["theta_ref"])
        priors = tuple(AxisPrior(**entry) for entry in payload["priors"])
        prior_bijector = PriorBijector(names=names, priors=priors)
        linear = WhiteningLinear(C=arrays["C"], z0=arrays["z0"])
        return cls(
            names=names,
            theta_ref=theta_ref,
            prior_bijector=prior_bijector,
            linear=linear,
            static_layer=payload["static_layer"],
        )
