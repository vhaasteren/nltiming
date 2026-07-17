"""Timing-engine builders and public timing-engine classes."""

from __future__ import annotations

from .composite import PtaContribution, build_composite_engine
from .jug import JugEngine, LinearizedJugEngine
from .pint import LinearizedPintEngine, PintEngine
from .tempo2 import LinearizedLibstempoEngine, LibstempoEngine
from .vela import VelaEngine

_ENGINE_CHOICES = {"tempo2": ("libstempo", "jug"), "pint": ("pint", "jug", "vela")}
_IMPL_FAMILY = {"libstempo": "tempo2", "pint": "pint", "jug": "jug", "vela": "vela"}


def normalize_engines(engines):
    """Return ``{'tempo2': impl, 'pint': impl}`` for an engine selection."""
    if isinstance(engines, str):
        engines = {"tempo2": engines, "pint": engines}
    else:
        engines = dict(engines)
    extra = set(engines) - set(_ENGINE_CHOICES)
    if extra:
        raise ValueError(f"Unknown engine compatibility keys: {sorted(extra)}")
    out = {}
    for native, choices in _ENGINE_CHOICES.items():
        impl = engines.get(native, "jug")
        if impl not in choices:
            raise ValueError(
                f"engines[{native!r}] must be one of {choices}, got {impl!r}"
            )
        out[native] = impl
    return out


def build_engine(
    *,
    fitpars: tuple[str, ...],
    nrows: int,
    contributions: list[PtaContribution],
    design_matrix=None,
):
    """Build the per-pulsar composite over per-PTA engine contributions."""
    return build_composite_engine(
        fitpars=fitpars,
        nrows=nrows,
        contributions=contributions,
        design_matrix=design_matrix,
    )


__all__ = [
    "PtaContribution",
    "JugEngine",
    "VelaEngine",
    "LinearizedJugEngine",
    "LinearizedPintEngine",
    "LinearizedLibstempoEngine",
    "PintEngine",
    "LibstempoEngine",
    "build_engine",
    "build_composite_engine",
    "normalize_engines",
    "_ENGINE_CHOICES",
    "_IMPL_FAMILY",
]
