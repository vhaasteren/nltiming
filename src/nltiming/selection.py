"""Canonical/suffix-aware fit-parameter selector expansion.

This module knows only how to turn a user selector (a base name like ``"JUMP"``
or an exact possibly-PTA-suffixed name like ``"JUMP2_epta"``) into the set of
matching fitpars on a pulsar, preserving canonical fitpar order. It has **no**
knowledge of sampling, marginalization, priors, or linearity — those live in
:mod:`nltiming.inference` and :mod:`nltiming.linearity`.
"""

from __future__ import annotations

from .pint_compat import resolve_parameter_alias

__all__ = [
    "canonical_fitpars",
    "match_fitpars",
    "select_fitpars",
    "fitpar_suffixes",
]


def canonical_fitpars(pulsar) -> tuple[str, ...]:
    """Pulsar fitpars with aliases normalized, in canonical order."""
    fitpars = tuple(resolve_parameter_alias(p) for p in pulsar.fitpars)
    if len(set(fitpars)) != len(fitpars):
        raise ValueError("Duplicate fit parameters after alias normalization")
    return fitpars


def _base_param_candidates(pulsar, name: str) -> set[str]:
    """Canonical base-name candidates for a (possibly suffixed) fitpar.

    Composite pulsars expose PTA-suffixed fitpars (e.g. ``RAJ_ng5``) while PINT
    category discovery yields unsuffixed canonical names (e.g. ``RAJ``). The
    pulsar carries the suffixed -> per-PTA base mapping in ``_fitparameters``.
    """
    candidates: set[str] = {name, resolve_parameter_alias(name)}
    mapping = getattr(pulsar, "_fitparameters", None) or {}
    for base in mapping.get(name, {}).values():
        candidates.add(base)
        candidates.add(resolve_parameter_alias(base))
    return candidates


def _suffix_policy_candidates(pulsar, name: str) -> set[str]:
    """Candidates eligible for exact-name/prefix registries.

    Composite pulsars with PTA-suffixed names must provide ``_fitparameters``
    mappings; without one, broad prefix matching would silently treat
    ``DMX_0001_ng5`` as a valid DMX parameter instead of surfacing broken suffix
    resolution.
    """
    mapping = getattr(pulsar, "_fitparameters", None)
    if mapping is not None:
        bases = mapping.get(name, {})
        candidates: set[str] = set()
        for base in bases.values():
            candidates.add(base)
            candidates.add(resolve_parameter_alias(base))
        return candidates
    return {name, resolve_parameter_alias(name)}


def match_fitpars(pulsar, name: str, fitpars: tuple[str, ...]) -> tuple[str, ...]:
    """Fitpars matching a requested base or exact (possibly suffixed) name.

    ``"PB"`` matches the canonical fitpar ``PB`` and every PTA-suffixed variant
    (``PB_epta``, ``PB_ppta``) exposed by a composite pulsar; an exact suffixed
    name matches only itself. Returns matches in canonical fitpar order.
    """
    canonical = resolve_parameter_alias(name)
    hits = []
    for fitpar in fitpars:
        if fitpar == canonical:
            hits.append(fitpar)
            continue
        candidates = {
            resolve_parameter_alias(c) for c in _base_param_candidates(pulsar, fitpar)
        }
        if canonical in candidates:
            hits.append(fitpar)
    return tuple(hits)


def fitpar_suffixes(pulsar, fitpar: str) -> set[str]:
    """PTA suffixes carried by a composite fitpar name (``{""}`` if unsuffixed)."""
    mapping = getattr(pulsar, "_fitparameters", None) or {}
    bases = set(mapping.get(fitpar, {}).values())
    if not bases:
        return {""}
    suffixes: set[str] = set()
    for base in bases:
        if fitpar == base:
            suffixes.add("")
        elif fitpar.startswith(base):
            suffixes.add(fitpar[len(base) :])
    return suffixes or {""}


def select_fitpars(pulsar, names, *, what: str = "selector") -> tuple[str, ...]:
    """Resolve a sequence of base/exact names to fitpars, preserving fitpar order.

    Raises if any requested name matches nothing — a silent miss would move a
    parameter into a different inference disposition than the user intended.
    """
    fitpars = canonical_fitpars(pulsar)
    selected: set[str] = set()
    for name in names:
        hits = match_fitpars(pulsar, name, fitpars)
        if not hits:
            raise ValueError(
                f"{what} entry {name!r} matches no fit parameter on this pulsar; "
                f"fitpars: {list(fitpars)}"
            )
        selected.update(hits)
    return tuple(p for p in fitpars if p in selected)
