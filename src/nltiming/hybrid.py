"""Hybrid residual linearisation: the closed mode set and the binary-axis registry.

``nonlinear_params`` is a residual-formula choice every engine family executes:
``"binary"`` keeps the binary axes on the nonlinear path and serves every other
axis as ``-M δ``; ``"binary+"`` adds ``PX``. Under a hybrid mode astrometry
sits at the par-file reference *inside* the binary delay, which is what makes
the mode a model choice rather than an optimisation.

This module is the single owner of the vocabulary and of what "binary axis"
means. Engines execute the mode; they do not define it. That ownership used to
sit in JUG, which meant a vela-jax leg through MetaPulsar imported JUG to ask
what a binary parameter is -- a hard dependency on an optional engine. The
registry below is the union of JUG's ``_BINARY_PARAMS`` and vela-jax's
``BINARY_AXES``, so neither engine loses an axis it can evaluate; JUG keeps
its own copy for JUG-internal use, and there is no import cycle in either
direction.
"""

from __future__ import annotations

import re

from .pint_compat import resolve_parameter_alias

NONLINEAR_PARAMS_BINARY = "binary"
NONLINEAR_PARAMS_BINARY_PLUS = "binary+"
NONLINEAR_PARAMS_MODES = frozenset(
    {NONLINEAR_PARAMS_BINARY, NONLINEAR_PARAMS_BINARY_PLUS}
)

#: PINT canonical names of every binary-model axis any supported engine
#: evaluates: the union of JUG's ``_BINARY_PARAMS`` (which lacks ``LNEDOT``)
#: and vela-jax's ``BINARY_AXES`` (which lacks ``A0``/``B0``/``H4``/``MTOT``/
#: ``XOMDOT``). An engine that cannot evaluate one of these refuses
#: it at build time; the registry's job is to say what *kind* of axis it is.
#:
#: ``COSI`` and ``GGAMMA`` come from ``BINARY DDR`` (vela-jax v2.6): the DT92
#: inclination cosine, which DDR uses in place of ``SINI``/``KIN``, and the
#: regular Einstein coefficient it reads under ``DDRPK N``. ``COSI`` is signed
#: on ``(-1, 1)`` -- see ``units._SIGNED_UNIT_INTERVAL`` -- rather than a
#: non-negative amplitude.
# fmt: off
BINARY_AXES = frozenset({
    "A0", "A1", "A1DOT", "B0", "COSI", "DR", "DTH", "ECC", "EDOT",
    "EPS1", "EPS1DOT", "EPS2", "EPS2DOT", "GAMMA", "GGAMMA", "H3", "H4",
    "KIN", "KOM", "LNEDOT", "M2", "MTOT", "OM", "OMDOT", "PB", "PBDOT",
    "SHAPMAX", "SINI", "STIGMA", "T0", "TASC", "XOMDOT", "XPBDOT",
})
# fmt: on

#: ``FB0``, ``FB1``, ... -- the orbital-frequency parametrisation of ``PB``.
_FB_PREFIX = re.compile(r"^FB\d+$")


def validate_nonlinear_params(value: str | None) -> str | None:
    """``None``, or the normalized mode string; ``ValueError`` otherwise."""
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or value.strip().lower() not in NONLINEAR_PARAMS_MODES
    ):
        raise ValueError(
            f"Unknown nonlinear_params={value!r}; expected None or one of "
            f"{', '.join(repr(s) for s in sorted(NONLINEAR_PARAMS_MODES))}"
        )
    return value.strip().lower()


def is_binary_axis(name: str) -> bool:
    """``name`` (any PINT/tempo2 spelling, unsuffixed) is a binary-model axis.

    Aliases are resolved first, so ``XDOT``, ``E`` and ``STIG`` classify with
    ``A1DOT``, ``ECC`` and ``STIGMA``. PTA-suffixed fitpar names must be mapped
    to their engine spelling before they get here.
    """
    canonical = resolve_parameter_alias(name)
    return canonical in BINARY_AXES or bool(_FB_PREFIX.match(canonical))


def is_hybrid_engine_axis(engine_param: str, mode: str | None) -> bool:
    """Whether ``engine_param`` stays on the engine residual path under ``mode``."""
    resolved = validate_nonlinear_params(mode)
    if resolved is None:
        return True
    if is_binary_axis(engine_param):
        return True
    return resolved == NONLINEAR_PARAMS_BINARY_PLUS and (
        resolve_parameter_alias(engine_param) == "PX"
    )


def hybrid_linearized_fitpars(
    fitpars, engine_names, mode: str | None
) -> frozenset[str]:
    """Fitpars the hybrid mode moves onto the design-matrix path."""
    resolved = validate_nonlinear_params(mode)
    if resolved is None:
        return frozenset()
    engine_names = dict(engine_names or {})
    return frozenset(
        name
        for name in fitpars
        if not is_hybrid_engine_axis(engine_names.get(name, name), resolved)
    )


def resolve_hybrid_partition(
    *,
    fitpars,
    param_mapping,
    mode: str | None,
    engine_fitpars,
    exact_linear_fitpars,
):
    """Resolve one adapter's ``(engine, exact_linear)`` split for a mode.

    Used by the adapter constructors so a directly built engine can never
    carry a hybrid mode it does not execute: with no explicit
    ``engine_fitpars`` the hybrid partition *is* the default, and an explicit
    engine list that contains a linearized axis is refused rather than
    silently evaluated by the engine. ``mode=None`` keeps today's defaults
    (every fitpar on the engine unless the caller said otherwise).
    """
    resolved = validate_nonlinear_params(mode)
    fitpars = tuple(fitpars)
    mapping = dict(param_mapping or {})
    exact = frozenset(exact_linear_fitpars or frozenset())
    if resolved is None:
        on_engine = fitpars if engine_fitpars is None else tuple(engine_fitpars)
        return on_engine, exact

    linearized = hybrid_linearized_fitpars(fitpars, mapping, resolved)
    if engine_fitpars is None:
        on_engine = tuple(name for name in fitpars if name not in linearized)
        return on_engine, exact | linearized

    on_engine = tuple(engine_fitpars)
    offenders = sorted(set(on_engine) & linearized)
    if offenders:
        raise ValueError(
            f"nonlinear_params={resolved!r} linearizes {offenders}, but they "
            "were passed as engine_fitpars; a stamped mode must match the "
            "partition the engine actually evaluates"
        )
    # An explicit engine list may name only the axes the engine can evaluate;
    # the mode still owns every linearized axis, so fold them into the
    # design-matrix set rather than leaving their deltas unevaluated.
    exact = exact | linearized
    dropped = sorted(set(fitpars) - set(on_engine) - exact)
    if dropped:
        raise ValueError(
            f"fitpars {dropped} are neither on the engine nor exact-linear "
            f"under nonlinear_params={resolved!r}; their deltas would be "
            "silently dropped from the residual"
        )
    return on_engine, exact


__all__ = [
    "NONLINEAR_PARAMS_BINARY",
    "NONLINEAR_PARAMS_BINARY_PLUS",
    "NONLINEAR_PARAMS_MODES",
    "BINARY_AXES",
    "validate_nonlinear_params",
    "is_binary_axis",
    "is_hybrid_engine_axis",
    "hybrid_linearized_fitpars",
    "resolve_hybrid_partition",
]
