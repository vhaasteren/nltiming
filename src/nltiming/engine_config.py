"""Backend-neutral timing-engine selection vocabulary."""

from __future__ import annotations

# Default implementation when a native package is omitted from ``engines=``.
# ``vela_jax`` is the JAX delay kernel: PINT or tempo2 reads the files, Vela's
# ported chain evaluates the residual. JUG remains a legal choice via
# ``engines="jug"``; it is not the default.
DEFAULT_ENGINE = "vela_jax"

# Which implementations may serve each native timing package. The key is the
# package a leg's files were *written* by; the value set is the implementations
# able to evaluate them.
#
# ``vela_jax`` appears under both because it separates the two concerns: PINT
# or tempo2 reads the files (``Engine.from_files`` / ``Engine.from_tempo2``)
# and Vela's ported component chain evaluates the delay either way.
_ENGINE_CHOICES = {
    "tempo2": ("libstempo", "jug", "vela_jax"),
    "pint": ("pint", "jug", "vela", "vela_jax"),
}

# The physics family an implementation belongs to. ``vela_jax`` is PINT-family
# even when tempo2 read the file: its residual is Vela's, its design matrix is
# PINT's, and its units are PINT's.
_IMPL_FAMILY = {
    "libstempo": "tempo2",
    "pint": "pint",
    "jug": "jug",
    "vela": "vela",
    "vela_jax": "pint",
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
        impl = engines.get(native, DEFAULT_ENGINE)
        if impl not in choices:
            raise ValueError(
                f"engines[{native!r}] must be one of {choices}, got {impl!r}"
            )
        out[native] = impl
    return out


__all__ = [
    "DEFAULT_ENGINE",
    "_ENGINE_CHOICES",
    "_IMPL_FAMILY",
    "normalize_engines",
]
