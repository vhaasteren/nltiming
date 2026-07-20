"""Timing-coordinate policy and expansion-point specification (§4.4).

``TimingCoordinatePolicy`` records the per-axis prior-normal chart policy: the
default prior scales for unresolved axes, the sigma source, and the warning
policies for the two non-affine situations. ``TimingExpansionSpec`` names the
fixed local linearization point; it is resolved against the proper-prior space
in :mod:`nltiming.linearization`.

No Discovery, Enterprise, NumPyro, pandas, or sampler imports live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

NonAffineLinearPolicy = Literal["warn", "ignore"]

_NONAFFINE_POLICIES = ("warn", "ignore")


class NonAffineIdenticallyLinearWarning(UserWarning):
    """An identically-linear axis carries a non-Gaussian prior, so its
    probability-integral-transform (PIT) chart is only a local surrogate (§4.4).
    The user's prior is honored."""


class LocallyMarginalizedTimingWarning(UserWarning):
    """A marginalized axis is not certified identically linear, so its analytical
    integration uses a fixed local affine likelihood (§4.4). The plan is honored."""


@dataclass(frozen=True)
class TimingCoordinatePolicy:
    """Chart/prior policy for the proper-prior timing axes (§4.4)."""

    linear_scale: float = 50.0
    nonlinear_scale: float = 50.0
    sigma_source: Literal["parfile_then_wls"] = "parfile_then_wls"
    nonaffine_identically_linear: NonAffineLinearPolicy = "warn"
    nonidentically_linear_marginalization: NonAffineLinearPolicy = "warn"

    def __post_init__(self) -> None:
        if not (float(self.linear_scale) > 0.0):
            raise ValueError("linear_scale must be positive")
        if not (float(self.nonlinear_scale) > 0.0):
            raise ValueError("nonlinear_scale must be positive")
        if self.sigma_source != "parfile_then_wls":
            raise ValueError("sigma_source must be 'parfile_then_wls'")
        if self.nonaffine_identically_linear not in _NONAFFINE_POLICIES:
            raise ValueError(
                "nonaffine_identically_linear must be 'warn' or 'ignore'"
            )
        if self.nonidentically_linear_marginalization not in _NONAFFINE_POLICIES:
            raise ValueError(
                "nonidentically_linear_marginalization must be 'warn' or 'ignore'"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "linear_scale": float(self.linear_scale),
            "nonlinear_scale": float(self.nonlinear_scale),
            "sigma_source": self.sigma_source,
            "nonaffine_identically_linear": self.nonaffine_identically_linear,
            "nonidentically_linear_marginalization": (
                self.nonidentically_linear_marginalization
            ),
        }


ExpansionMode = Literal["engine_reference", "prior_center", "explicit_delta"]


@dataclass(frozen=True)
class TimingExpansionSpec:
    """Fixed local expansion point for all proper-prior axes (§5.3)."""

    mode: ExpansionMode
    delta: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("engine_reference", "prior_center", "explicit_delta"):
            raise ValueError(f"unsupported expansion mode: {self.mode!r}")
        if self.mode == "explicit_delta":
            if not self.delta:
                raise ValueError("explicit_delta requires a non-empty delta mapping")
        elif self.delta is not None:
            raise ValueError(f"{self.mode} expansion takes no delta mapping")

    @classmethod
    def engine_reference(cls) -> "TimingExpansionSpec":
        return cls(mode="engine_reference")

    @classmethod
    def prior_center(cls) -> "TimingExpansionSpec":
        return cls(mode="prior_center")

    @classmethod
    def explicit_delta(cls, delta: Mapping[str, float]) -> "TimingExpansionSpec":
        return cls(mode="explicit_delta", delta=dict(delta))

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "delta": None if self.delta is None else dict(self.delta),
        }
