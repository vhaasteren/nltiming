"""Backend-neutral timing-engine selection vocabulary."""

from __future__ import annotations

_ENGINE_CHOICES = {
    "tempo2": ("libstempo", "jug"),
    "pint": ("pint", "jug", "vela"),
}
_IMPL_FAMILY = {
    "libstempo": "tempo2",
    "pint": "pint",
    "jug": "jug",
    "vela": "vela",
}


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


__all__ = [
    "_ENGINE_CHOICES",
    "_IMPL_FAMILY",
    "normalize_engines",
]
