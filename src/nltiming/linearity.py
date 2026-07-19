"""Identical-linearity authority: nltiming's engine-independent fallback registry.

An *identically linear* parameter is one whose engine waveform is affine in
delta. This is an intrinsic modeling assertion, **not** an inference disposition
(see :mod:`nltiming.inference`): declaring a parameter identically linear never
moves it between sampled and marginalized groups, and choosing a marginalization
never labels a parameter identically linear.

Authority order (§4.3):

- ``identically_linear=None`` -> effective = fallback registry union engine
  declarations;
- an explicit sequence (including an empty sequence) -> effective = exactly the
  resolved user sequence. Registry/engine candidates the explicit list omits are
  recorded under ``suppressed_candidates`` but are not treated as identically
  linear.

The exact-name and prefix lists below are scientifically certified and
normative. A change requires a separately reviewed proposal and a registry
version increment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Sequence

from .pint_compat import resolve_parameter_alias
from .selection import canonical_fitpars, match_fitpars, select_fitpars

FALLBACK_IDENTICALLY_LINEAR_EXACT = frozenset(
    {"DM", "DM1", "DM2", "OFFSET", "PHOFF"}
)
FALLBACK_IDENTICALLY_LINEAR_PREFIXES = ("DMX", "JUMP", "FD")

FALLBACK_REGISTRY_VERSION = 1

LinearitySource = Literal["engine", "fallback", "user"]
# Canonical emission order for a declaration's sources (matches manifest schema).
_SOURCE_ORDER = ("fallback", "engine", "user")


def _order_sources(sources) -> tuple[str, ...]:
    present = set(sources)
    return tuple(s for s in _SOURCE_ORDER if s in present)


def fallback_registry_digest() -> str:
    payload = {
        "version": FALLBACK_REGISTRY_VERSION,
        "exact": sorted(FALLBACK_IDENTICALLY_LINEAR_EXACT),
        "prefixes": list(FALLBACK_IDENTICALLY_LINEAR_PREFIXES),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _suffix_policy_candidates(pulsar, name: str) -> set[str]:
    from .selection import _suffix_policy_candidates as _spc

    return _spc(pulsar, name)


def fallback_identically_linear_fitpars(pulsar) -> frozenset[str]:
    """Fitpars matched by nltiming's engine-independent fallback registry.

    Applies whether or not any timing engine is installed or selected.
    """
    exact_upper = {n.upper() for n in FALLBACK_IDENTICALLY_LINEAR_EXACT}
    prefixes = tuple(p.upper() for p in FALLBACK_IDENTICALLY_LINEAR_PREFIXES)
    out: set[str] = set()
    for fitpar in canonical_fitpars(pulsar):
        cands = {
            resolve_parameter_alias(c).upper()
            for c in _suffix_policy_candidates(pulsar, fitpar)
        }
        if cands & exact_upper or any(c.startswith(prefixes) for c in cands):
            out.add(fitpar)
    return frozenset(out)


def _engine_identically_linear_fitpars(pulsar, engine) -> frozenset[str]:
    if engine is None:
        return frozenset()
    method = getattr(engine, "identically_linear_fitpars", None)
    if method is None:
        return frozenset()
    declared = method()
    fitpars = canonical_fitpars(pulsar)
    out: set[str] = set()
    for name in declared:
        out.update(match_fitpars(pulsar, name, fitpars))
    return frozenset(out)


@dataclass(frozen=True)
class LinearityDeclaration:
    name: str
    sources: tuple[LinearitySource, ...]


@dataclass(frozen=True)
class LinearityResolution:
    mode: Literal["defaults", "explicit_user"]
    effective: tuple[LinearityDeclaration, ...]
    suppressed_candidates: tuple[LinearityDeclaration, ...]
    fallback_registry_version: int
    fallback_registry_digest: str

    @property
    def effective_names(self) -> frozenset[str]:
        return frozenset(d.name for d in self.effective)

    def sources_for(self, name: str) -> tuple[LinearitySource, ...]:
        for d in self.effective:
            if d.name == name:
                return d.sources
        return ()


def resolve_linearity(
    pulsar,
    engine=None,
    *,
    identically_linear: Sequence[str] | None = None,
) -> LinearityResolution:
    """Resolve the effective identical-linearity set and its provenance (§4.3)."""
    fitpars = canonical_fitpars(pulsar)
    fallback = fallback_identically_linear_fitpars(pulsar)
    engine_set = _engine_identically_linear_fitpars(pulsar, engine)

    def _sources(name, *, user: bool) -> tuple[LinearitySource, ...]:
        srcs: set[str] = set()
        if user:
            srcs.add("user")
        if name in fallback:
            srcs.add("fallback")
        if name in engine_set:
            srcs.add("engine")
        return _order_sources(srcs)

    if identically_linear is None:
        effective_set = fallback | engine_set
        effective = tuple(
            LinearityDeclaration(name, _sources(name, user=False))
            for name in fitpars
            if name in effective_set
        )
        return LinearityResolution(
            mode="defaults",
            effective=effective,
            suppressed_candidates=(),
            fallback_registry_version=FALLBACK_REGISTRY_VERSION,
            fallback_registry_digest=fallback_registry_digest(),
        )

    user_set = set(
        select_fitpars(pulsar, identically_linear, what="identically_linear")
    )
    effective = tuple(
        LinearityDeclaration(name, _sources(name, user=True))
        for name in fitpars
        if name in user_set
    )
    candidate_set = (fallback | engine_set) - user_set
    suppressed = tuple(
        LinearityDeclaration(name, _sources(name, user=False))
        for name in fitpars
        if name in candidate_set
    )
    return LinearityResolution(
        mode="explicit_user",
        effective=effective,
        suppressed_candidates=suppressed,
        fallback_registry_version=FALLBACK_REGISTRY_VERSION,
        fallback_registry_digest=fallback_registry_digest(),
    )
