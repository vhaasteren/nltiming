"""Canonical/suffix-aware fit-parameter selector expansion.

This module knows only how to turn a user selector (a base name like ``"JUMP"``
or an exact possibly-PTA-suffixed name like ``"JUMP2_epta"``) into the set of
matching fitpars on a pulsar, preserving canonical fitpar order. It has **no**
knowledge of sampling, marginalization, priors, or linearity — those live in
:mod:`nltiming.inference` and :mod:`nltiming.linearity`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .pint_compat import resolve_parameter_alias

__all__ = [
    "ParameterMappingError",
    "ParameterMappingView",
    "parameter_mapping_view",
    "validated_parameter_mapping_view",
    "canonical_fitpars",
    "match_fitpars",
    "select_fitpars",
    "fitpar_suffix",
]


class ParameterMappingError(ValueError):
    """A pulsar's timing-parameter mapping violates the composite-name contract."""


@dataclass(frozen=True)
class ParameterMappingView:
    mapping: Mapping[str, Mapping[str, str]] | None
    source: str


def parameter_mapping_view(pulsar) -> ParameterMappingView:
    """Return the public timing mapping, with a legacy private fallback."""
    provider = getattr(pulsar, "timing_parameter_mapping", None)
    if provider is not None:
        if not callable(provider):
            raise ParameterMappingError(
                "timing_parameter_mapping exists but is not callable"
            )
        mapping = provider()
        source = "timing_parameter_mapping()"
        if not isinstance(mapping, Mapping):
            raise ParameterMappingError(
                f"{source} must return a mapping, got {type(mapping).__name__}"
            )
        return ParameterMappingView(mapping, source)

    if not hasattr(pulsar, "_fitparameters"):
        return ParameterMappingView(None, "none")
    mapping = getattr(pulsar, "_fitparameters")
    if mapping is None:
        return ParameterMappingView(None, "none")
    if not isinstance(mapping, Mapping):
        raise ParameterMappingError(
            "legacy _fitparameters must be a mapping when present"
        )
    return ParameterMappingView(mapping, "legacy _fitparameters")


def validated_parameter_mapping_view(
    pulsar,
    fitpars: tuple[str, ...] | None = None,
) -> ParameterMappingView:
    """Fetch the mapping and enforce the total coherent contract over ``fitpars``.

    When a mapping is supplied it must cover every actual fitpar with a
    well-formed merged-or-local owner entry. Call once at consumer boundaries
    and reuse the returned view.
    """
    view = parameter_mapping_view(pulsar)
    if view.mapping is None:
        return view
    names = fitpars if fitpars is not None else canonical_fitpars(pulsar)
    for fitpar in names:
        fitpar_suffix(pulsar, fitpar, mapping_view=view)
    return view


def canonical_fitpars(pulsar) -> tuple[str, ...]:
    """Pulsar fitpars with aliases normalized, in canonical order."""
    fitpars = tuple(resolve_parameter_alias(p) for p in pulsar.fitpars)
    if len(set(fitpars)) != len(fitpars):
        raise ValueError("Duplicate fit parameters after alias normalization")
    return fitpars


def _owners_for_fitpar(
    view: ParameterMappingView, name: str
) -> Mapping[str, str] | None:
    """Return validated owners for ``name``, or ``None`` if absent from mapping."""
    mapping = view.mapping
    if mapping is None or name not in mapping:
        return None
    owners = mapping[name]
    if not isinstance(owners, Mapping) or not owners:
        raise ParameterMappingError(
            f"fitpar {name!r} has an empty or non-mapping owner entry "
            f"from {view.source}: {owners!r}"
        )
    return owners


def _base_param_candidates(
    pulsar,
    name: str,
    *,
    mapping_view: ParameterMappingView | None = None,
) -> set[str]:
    """Canonical base-name candidates for a (possibly suffixed) fitpar.

    Composite pulsars expose PTA-suffixed fitpars (e.g. ``RAJ_ng5``) while PINT
    category discovery yields unsuffixed canonical names (e.g. ``RAJ``). The
    pulsar carries the suffixed -> per-PTA base mapping via the public timing
    parameter mapping (or legacy ``_fitparameters``).
    """
    candidates: set[str] = {name, resolve_parameter_alias(name)}
    view = parameter_mapping_view(pulsar) if mapping_view is None else mapping_view
    owners = _owners_for_fitpar(view, name)
    if owners is None:
        return candidates
    for base in owners.values():
        candidates.add(base)
        candidates.add(resolve_parameter_alias(base))
    return candidates


def _suffix_policy_candidates(
    pulsar,
    name: str,
    *,
    mapping_view: ParameterMappingView | None = None,
) -> set[str]:
    """Candidates eligible for exact-name/prefix registries.

    Composite pulsars with PTA-suffixed names must provide a timing-parameter
    mapping; without one, broad prefix matching would silently treat
    ``DMX_0001_ng5`` as a valid DMX parameter instead of surfacing broken suffix
    resolution.
    """
    view = parameter_mapping_view(pulsar) if mapping_view is None else mapping_view
    if view.mapping is None:
        return {name, resolve_parameter_alias(name)}
    owners = _owners_for_fitpar(view, name)
    if owners is None:
        return set()
    candidates: set[str] = set()
    for base in owners.values():
        candidates.add(base)
        candidates.add(resolve_parameter_alias(base))
    return candidates


def match_fitpars(
    pulsar,
    name: str,
    fitpars: tuple[str, ...],
    *,
    mapping_view: ParameterMappingView | None = None,
) -> tuple[str, ...]:
    """Fitpars matching a requested base or exact (possibly suffixed) name.

    ``"PB"`` matches the canonical fitpar ``PB`` and every PTA-suffixed variant
    (``PB_epta``, ``PB_ppta``) exposed by a composite pulsar; an exact suffixed
    name matches only itself. Returns matches in canonical fitpar order.

    When ``mapping_view`` is omitted, the pulsar's mapping is fetched and
    validated once against the pulsar's actual fitpars (not against the
    ``fitpars`` matching universe, which may include synthesized names).
    """
    canonical = resolve_parameter_alias(name)
    view = (
        validated_parameter_mapping_view(pulsar)
        if mapping_view is None
        else mapping_view
    )
    hits = []
    for fitpar in fitpars:
        if fitpar == canonical:
            hits.append(fitpar)
            continue
        candidates = {
            resolve_parameter_alias(c)
            for c in _base_param_candidates(pulsar, fitpar, mapping_view=view)
        }
        if canonical in candidates:
            hits.append(fitpar)
    return tuple(hits)


def fitpar_suffix(
    pulsar,
    fitpar: str,
    *,
    mapping_view: ParameterMappingView | None = None,
) -> str:
    """Return the single PTA suffix carried by a fitpar (``""`` if merged)."""
    view = parameter_mapping_view(pulsar) if mapping_view is None else mapping_view
    if not isinstance(fitpar, str) or not fitpar:
        raise ParameterMappingError(f"fitpar must be a non-empty string: {fitpar!r}")
    if view.mapping is None:
        return ""
    if fitpar not in view.mapping:
        raise ParameterMappingError(
            f"fitpar {fitpar!r} is absent from the total parameter mapping "
            f"provided by {view.source}"
        )
    owners = view.mapping[fitpar]
    if not isinstance(owners, Mapping) or not owners:
        raise ParameterMappingError(
            f"fitpar {fitpar!r} has an empty or non-mapping owner entry "
            f"from {view.source}: {owners!r}"
        )
    if any(
        not isinstance(pta, str) or not pta or not isinstance(native, str) or not native
        for pta, native in owners.items()
    ):
        raise ParameterMappingError(
            f"fitpar {fitpar!r} has non-string or empty owner/native names "
            f"from {view.source}: {dict(owners)!r}"
        )

    canonical = resolve_parameter_alias(fitpar)
    if all(resolve_parameter_alias(native) == canonical for native in owners.values()):
        return ""

    if len(owners) != 1:
        raise ParameterMappingError(
            f"fitpar {fitpar!r} mixes merged/local identities or has multiple "
            f"local owners in {view.source}: {dict(owners)!r}"
        )
    pta, native = next(iter(owners.items()))
    suffix = f"_{pta}"
    if not fitpar.endswith(suffix):
        raise ParameterMappingError(
            f"fitpar {fitpar!r} does not end in owner suffix {suffix!r}; "
            f"{view.source} entry: {dict(owners)!r}"
        )
    stem = fitpar[: -len(suffix)]
    if resolve_parameter_alias(stem) != resolve_parameter_alias(native):
        raise ParameterMappingError(
            f"fitpar {fitpar!r} stem {stem!r} is not an alias of native "
            f"{native!r}; {view.source} entry: {dict(owners)!r}"
        )
    return suffix


def select_fitpars(pulsar, names, *, what: str = "selector") -> tuple[str, ...]:
    """Resolve a sequence of base/exact names to fitpars, preserving fitpar order.

    Raises if any requested name matches nothing — a silent miss would move a
    parameter into a different inference disposition than the user intended.
    """
    fitpars = canonical_fitpars(pulsar)
    view = validated_parameter_mapping_view(pulsar, fitpars)
    selected: set[str] = set()
    for name in names:
        hits = match_fitpars(pulsar, name, fitpars, mapping_view=view)
        if not hits:
            raise ValueError(
                f"{what} entry {name!r} matches no fit parameter on this pulsar; "
                f"fitpars: {list(fitpars)}"
            )
        selected.update(hits)
    return tuple(p for p in fitpars if p in selected)
