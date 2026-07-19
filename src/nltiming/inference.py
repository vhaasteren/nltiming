"""One per-parameter timing inference plan (§4).

Replaces the old ``PartitionResult`` plus the independent ``sample=`` /
``sample_linear=`` / ``analytically_marginalize=`` switches with a single typed
object. Every fitpar receives exactly one disposition:

    not selected for marginalization -> sample
    Marginalize.delta_flat()          -> marginalize_delta_flat
    Marginalize.z_prior()             -> marginalize_z_prior

Inference disposition is independent of identical linearity
(:mod:`nltiming.linearity`): declaring an axis identically linear never moves it
between sampled and marginalized groups, and choosing a marginalization never
labels an axis identically linear.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from .bijectors import AxisPrior
from .coordinates import TimingCoordinatePolicy
from .linearity import LinearityResolution
from .pint_compat import (
    get_parameters_by_type_from_models,
    resolve_parameter_alias,
)
from .selection import canonical_fitpars, select_fitpars

MarginalCoordinate = Literal["delta", "z"]
Disposition = Literal["sample", "marginalize_delta_flat", "marginalize_z_prior"]
ChartKind = Literal["affine_normal", "prior_pit"]


@dataclass(frozen=True)
class Marginalize:
    """Analytical treatment of selected timing axes."""

    coordinate: MarginalCoordinate

    @classmethod
    def delta_flat(cls) -> "Marginalize":
        """Linearize in physical delta and use an improper flat delta measure."""
        return cls(coordinate="delta")

    @classmethod
    def z_prior(cls) -> "Marginalize":
        """Linearize in prior-normal z and integrate z ~ Normal(0, I)."""
        return cls(coordinate="z")


@dataclass(frozen=True)
class TimingInference:
    """Complete disjoint timing-axis inference plan; unmentioned axes are sampled."""

    marginalize: Mapping[str, Marginalize] = field(default_factory=dict)
    preset: Literal["explicit", "default_delta"] = "explicit"

    def __post_init__(self) -> None:
        if self.preset not in ("explicit", "default_delta"):
            raise ValueError(f"unsupported preset: {self.preset!r}")
        if self.preset == "default_delta" and self.marginalize:
            raise ValueError(
                "the 'default_delta' preset resolves its own delta-flat set; do "
                "not also pass an explicit marginalize mapping"
            )
        object.__setattr__(self, "marginalize", dict(self.marginalize))

    @classmethod
    def sample_all(cls) -> "TimingInference":
        return cls(marginalize={})

    @classmethod
    def default(cls) -> "TimingInference":
        return cls(marginalize={}, preset="default_delta")

    @classmethod
    def groups(
        cls,
        *,
        delta_flat: Sequence[str] = (),
        z_prior: Sequence[str] = (),
    ) -> "TimingInference":
        overlap = set(delta_flat) & set(z_prior)
        if overlap:
            raise ValueError(f"timing inference groups overlap: {sorted(overlap)}")
        return cls(
            marginalize={
                **{name: Marginalize.delta_flat() for name in delta_flat},
                **{name: Marginalize.z_prior() for name in z_prior},
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "marginalize": {
                name: marg.coordinate
                for name, marg in sorted(self.marginalize.items())
            },
        }


# ---------------------------------------------------------------------------
# Inference-default registry (§4.2). Separate from the linearity registry.
# ---------------------------------------------------------------------------

DEFAULT_DELTA_CATEGORIES = (
    "spindown",
    "dispersion",
    "dispersion_constant",
    "dispersion_dmx",
    "phase_jump",
    "fd",
    "frequency_dependent",
)
DEFAULT_DELTA_EXACT = frozenset(
    {"DM", "DM1", "DM2", "OFFSET", "PHOFF",
     "RAJ", "DECJ", "ELONG", "ELAT", "RA", "DEC", "LAMBDA", "BETA"}
)
DEFAULT_DELTA_PREFIXES = ("DMX", "JUMP", "FD")
DEFAULT_DELTA_EXCLUDE = frozenset(
    {"PMRA", "PMDEC", "PMELONG", "PMELAT", "PX"}
)


def _discover_category_params(pint_model, categories) -> set[str]:
    wanted = set(categories)
    discovered: set[str] = set()
    for comp in getattr(pint_model, "components", {}).values():
        if getattr(comp, "category", None) not in wanted:
            continue
        for name in getattr(comp, "params", []):
            discovered.add(name)
            discovered.add(resolve_parameter_alias(name))
    return discovered


def _base_candidates(pulsar, name: str) -> set[str]:
    from .selection import _base_param_candidates

    return _base_param_candidates(pulsar, name)


def _suffix_candidates(pulsar, name: str) -> set[str]:
    from .selection import _suffix_policy_candidates

    return _suffix_policy_candidates(pulsar, name)


def default_delta_flat_fitpars(pulsar) -> tuple[str, ...]:
    """Resolve the versioned default delta-flat marginalization set (§4.2).

    An exclusion (proper motion, parallax) wins over category/name/prefix
    inclusion. A missing PINT model raises. A family match that resolves to an
    empty set raises the broken-suffix error.
    """
    model = pulsar.pint_model()
    if model is None:
        raise ValueError(
            "pulsar.pint_model() is required for TimingInference.default() "
            "(the default delta-flat preset)"
        )

    discovered: set[str] = set()
    for category in DEFAULT_DELTA_CATEGORIES:
        for name in get_parameters_by_type_from_models(category, {"ref": model}):
            discovered.add(name)
            discovered.add(resolve_parameter_alias(name))
    discovered.update(_discover_category_params(model, DEFAULT_DELTA_CATEGORIES))

    exact_upper = {n.upper() for n in DEFAULT_DELTA_EXACT}
    exclude_upper = {n.upper() for n in DEFAULT_DELTA_EXCLUDE}
    prefixes = tuple(p.upper() for p in DEFAULT_DELTA_PREFIXES)

    selected: list[str] = []
    has_family = False
    for raw in canonical_fitpars(pulsar):
        candidates = _base_candidates(pulsar, raw)
        suffix_cands = _suffix_candidates(pulsar, raw)
        canon = {resolve_parameter_alias(c).upper() for c in candidates}
        canon_suffix = {resolve_parameter_alias(c).upper() for c in suffix_cands}
        if canon & exact_upper or any(c.startswith(prefixes) for c in canon):
            has_family = True
        if canon & exclude_upper:
            continue
        if (
            candidates & discovered
            or canon & exact_upper
            or canon_suffix & exact_upper
            or any(c.startswith(prefixes) for c in canon_suffix)
        ):
            selected.append(raw)

    if has_family and not selected:
        raise ValueError(
            "TimingInference.default() resolved to an empty delta-flat set even "
            "though the pulsar carries linear nuisance families (DMX/JUMP/FD). "
            "This usually means fitpar name matching failed (e.g. an unmapped PTA "
            "suffix). Refusing to silently sample every timing parameter."
        )
    return tuple(selected)


# ---------------------------------------------------------------------------
# Resolved plan (§4.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedTimingAxis:
    name: str
    fitpar_index: int
    disposition: Disposition
    linearity_sources: tuple[str, ...]
    prior: AxisPrior | None = None
    prior_source: str | None = None
    chart: ChartKind | None = None


@dataclass(frozen=True)
class TimingParameterPlan:
    fitpars: tuple[str, ...]
    axes: tuple[ResolvedTimingAxis, ...]
    inference: TimingInference
    coordinate_policy: TimingCoordinatePolicy

    def axis(self, name: str) -> ResolvedTimingAxis:
        canonical = resolve_parameter_alias(name)
        for ax in self.axes:
            if ax.name == canonical or ax.name == name:
                return ax
        raise KeyError(f"no timing axis named {name!r}")

    def _names(self, disposition: Disposition) -> tuple[str, ...]:
        return tuple(a.name for a in self.axes if a.disposition == disposition)

    @property
    def sampled(self) -> tuple[str, ...]:
        return self._names("sample")

    @property
    def marginalized_delta(self) -> tuple[str, ...]:
        return self._names("marginalize_delta_flat")

    @property
    def marginalized_z(self) -> tuple[str, ...]:
        return self._names("marginalize_z_prior")

    @property
    def proper(self) -> tuple[str, ...]:
        """Sampled plus z-marginalized axes, in fitpar order (proper-prior set)."""
        return tuple(
            a.name
            for a in self.axes
            if a.disposition in ("sample", "marginalize_z_prior")
        )

    def indices(self, disposition: Disposition) -> tuple[int, ...]:
        return tuple(
            a.fitpar_index for a in self.axes if a.disposition == disposition
        )

    # Transitional index accessors consumed by the metric/whitening/adapter
    # helpers that still take a partition-shaped object. Stage 4/5 rewrite those
    # helpers to call ``indices(...)`` directly and these are removed then.
    @property
    def idx_sampled(self) -> tuple[int, ...]:
        return self.indices("sample")

    @property
    def idx_analytically_marginalized(self) -> tuple[int, ...]:
        return self.indices("marginalize_delta_flat")

    @property
    def analytically_marginalized(self) -> tuple[str, ...]:
        return self.marginalized_delta

    def with_axes(self, axes: Sequence[ResolvedTimingAxis]) -> "TimingParameterPlan":
        """Return a copy with replacement axes (used to fill prior/chart records)."""
        return TimingParameterPlan(
            fitpars=self.fitpars,
            axes=tuple(axes),
            inference=self.inference,
            coordinate_policy=self.coordinate_policy,
        )

    def fingerprint(self) -> str:
        payload = {
            "fitpars": list(self.fitpars),
            "inference": self.inference.as_dict(),
            "coordinate_policy": self.coordinate_policy.as_dict(),
            "axes": [
                {
                    "name": a.name,
                    "fitpar_index": a.fitpar_index,
                    "disposition": a.disposition,
                    "linearity_sources": list(a.linearity_sources),
                    "prior": None if a.prior is None else vars(a.prior),
                    "prior_source": a.prior_source,
                    "chart": a.chart,
                }
                for a in self.axes
            ],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def resolve_inference_plan(
    pulsar,
    *,
    inference: TimingInference,
    linearity: LinearityResolution,
    coordinate_policy: TimingCoordinatePolicy,
) -> TimingParameterPlan:
    """Resolve one exhaustive, disjoint per-fitpar inference plan (§4.2, §4.5).

    Prior/chart records are left unset here; the model fills them for the
    proper-prior axes after prior resolution (§4.4).
    """
    fitpars = canonical_fitpars(pulsar)
    index = {name: i for i, name in enumerate(fitpars)}

    disposition: dict[str, Disposition] = {name: "sample" for name in fitpars}

    if inference.preset == "default_delta":
        for name in default_delta_flat_fitpars(pulsar):
            disposition[name] = "marginalize_delta_flat"
    else:
        claimed: dict[str, str] = {}
        for selector, marg in inference.marginalize.items():
            hits = select_fitpars(pulsar, [selector], what="marginalize")
            target: Disposition = (
                "marginalize_delta_flat"
                if marg.coordinate == "delta"
                else "marginalize_z_prior"
            )
            for hit in hits:
                if hit in claimed:
                    raise ValueError(
                        "timing inference selectors overlap after suffix "
                        f"expansion: {hit!r} claimed by {claimed[hit]!r} and "
                        f"{selector!r}"
                    )
                claimed[hit] = selector
                disposition[hit] = target

    axes = tuple(
        ResolvedTimingAxis(
            name=name,
            fitpar_index=index[name],
            disposition=disposition[name],
            linearity_sources=linearity.sources_for(name),
        )
        for name in fitpars
    )
    return TimingParameterPlan(
        fitpars=fitpars,
        axes=axes,
        inference=inference,
        coordinate_policy=coordinate_policy,
    )
