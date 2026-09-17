"""Unit conversion helpers for timing-space parameters."""

from __future__ import annotations

from typing import Any

import astropy.units as u
import numpy as np

from .pint_compat import (
    get_aliases_for_parameter,
    pint_parameter_name,
    resolve_parameter_alias,
)

# Physical domains for prior clipping (not unit metadata).
# PBDOT is signed: observed orbital-period derivatives are often negative
# (GR decay, kinematics). PX and H3 are non-negative physical amplitudes;
# placeholder PX=0 / out-of-domain starts are nudged inside the prior by the
# validation host builder, not by widening these domains.
_NON_NEGATIVE = {"PB", "FB0", "A1", "M2", "MTOT", "MP", "STIGMA", "PX", "H3"}
# STIGMA is not unit-interval: sigma = tan(i/2) exceeds 1 for i > 90 deg
# (e.g. J1902-5105, STIG = 1.154); its physical domain is (0, inf).
_UNIT_INTERVAL = {"ECC", "E", "SINI"}
# COSI (DDR's DT92 inclination cosine) is signed: cos i runs over the whole of
# (-1, 1), and the two halves are physically distinct orbital orientations
# rather than a sign convention. Clipping it to [0, 1] would silently halve the
# prior and make Vela/pyvela's isotropic Uniform(-1, 1) unreachable. The engine
# still enforces the strict |COSI| < 1: an endpoint has zero measure and
# returns an invalid-state NaN rather than being clamped.
_SIGNED_UNIT_INTERVAL = {"COSI"}
_KIN_INTERVAL_DEG = (0.0, 180.0)
# Angles the delay depends on only through their sine and cosine, so the
# likelihood is *exactly* invariant under a full turn. Measured on a
# geometry-on DDR fixture, KOM + 360 deg moves the binary delay by 0.0 s
# bit-for-bit (KOM + 180 deg moves it by 1.6e-7 s, so the period is 360, not
# 180); OM + 360 deg moves DD residuals by 1.9 ns against 40 s for OM + 180,
# the residue being float64 `sin(x + 2pi)` on DD's unwrapped anomaly.
#
# These are *periodic*, not *bounded*, and the difference is the whole reason
# they are not in the sets above. KIN's (0, 180) is a real physical range for
# an inclination; [0, 360) for KOM is only a choice of branch. Left unbounded,
# a +-100 sigma box spans several turns and the posterior has infinitely many
# identical modes, which reads as a huge sigma rather than as a bug.
_PERIODIC_DEG = {"KOM": 360.0, "OM": 360.0}


def _qualified_name_candidates(name: str) -> list[str]:
    text = str(name)
    parts = text.split("_")
    candidates = [text]
    if len(parts) > 1:
        candidates.append("_".join(parts[:-1]))  # Strip PTA suffix, e.g. M2_ng5.
        candidates.append(parts[-1])  # Strip signal prefix, e.g. psr_ntm_M2.
        candidates.extend(reversed(parts))
        for start in range(len(parts)):
            candidates.append("_".join(parts[start:]))
        for start in range(len(parts)):
            for stop in range(start + 1, len(parts) + 1):
                candidates.append("_".join(parts[start:stop]))
    return candidates


def normalize_param_name(name: str) -> str:
    """Normalize qualified fit-parameter names to a canonical timing token."""
    seen: set[str] = set()
    for candidate in _qualified_name_candidates(name):
        resolved = resolve_parameter_alias(candidate)
        key = resolved.upper()
        if key in seen:
            continue
        seen.add(key)
        canonical = pint_parameter_name(resolved)
        if canonical is not None:
            return canonical.upper()
    return resolve_parameter_alias(name).upper()


def _lookup_pint_param(pint_model: Any | None, name: str):
    if pint_model is None:
        return None
    canonical = normalize_param_name(name)
    candidates: list[str] = []
    for key in (name, canonical, resolve_parameter_alias(name)):
        if key not in candidates:
            candidates.append(key)
    for alias in get_aliases_for_parameter(canonical):
        if alias not in candidates:
            candidates.append(alias)
    for key in candidates:
        if hasattr(pint_model, key):
            return getattr(pint_model, key)
    return None


def _coerce_unit(units: Any) -> u.Unit | None:
    if units is None:
        return None
    if units == "":
        return None
    try:
        unit = u.Unit(units)
    except (TypeError, ValueError):
        return None
    if unit == u.dimensionless_unscaled:
        return None
    return unit


def lookup_pint_param(pint_model: Any | None, name: str):
    """Return the PINT parameter object for a fit name, if available."""
    return _lookup_pint_param(pint_model, name)


def storage_unit(name: str, pint_model: Any | None = None) -> u.Unit | None:
    """Return the timing-engine storage unit for a fit parameter from PINT."""
    param = _lookup_pint_param(pint_model, name)
    if param is None:
        return None
    return _coerce_unit(getattr(param, "units", None))


def _unit_label(unit: u.Unit | None) -> str:
    if unit is None:
        return "native"
    if unit == u.deg:
        return "deg"
    if unit == u.hourangle:
        return "hourangle"
    if unit == u.day:
        return "MJD"
    if unit.is_equivalent(u.deg):
        return "deg"
    if unit.is_equivalent(u.hourangle):
        return "hourangle"
    if unit.is_equivalent(u.day):
        return "MJD"
    return unit.to_string()


def display_unit(name: str, pint_model: Any | None = None) -> str:
    """Human-readable display unit label."""
    return _unit_label(storage_unit(name, pint_model))


def native_unit_label(name: str, pint_model: Any | None = None) -> str:
    """Engine-native storage unit label for a fit parameter."""
    unit = storage_unit(name, pint_model)
    return "native" if unit is None else unit.to_string()


def units_map(
    names, pint_model: Any | None = None, *, kind: str = "display"
) -> dict[str, str]:
    """Map fitpar name -> unit label. kind in {'display', 'native'}."""
    if kind == "display":
        return {name: display_unit(name, pint_model) for name in names}
    if kind == "native":
        return {name: native_unit_label(name, pint_model) for name in names}
    raise ValueError(f"kind must be 'display' or 'native'; got {kind!r}")


def to_native(name: str, display_value, pint_model: Any | None = None):
    """Convert display-unit magnitudes to timing-engine storage units."""
    _ = storage_unit(name, pint_model)
    return np.asarray(display_value)


def to_display(name: str, native_value, pint_model: Any | None = None):
    """Convert timing-engine storage units to display-unit magnitudes."""
    _ = storage_unit(name, pint_model)
    return np.asarray(native_value)


def native_period(name: str) -> float | None:
    """Period of a cyclic axis in storage units, or ``None`` if aperiodic.

    Deliberately separate from :func:`native_physical_bounds`. A bound says
    "outside this the model is unphysical"; a period says "outside this the
    model repeats". The prior builder needs the second for cyclic angles,
    because it centres its box on the par value: one period *about the
    reference* keeps the axis identifiable while putting the box edge as far
    from the starting point as it can go. Clipping to an absolute ``[0, 360)``
    would instead drop a hard edge right next to any pulsar whose ``KOM`` sits
    near zero, where the likelihood is perfectly smooth.
    """
    return _PERIODIC_DEG.get(normalize_param_name(name))


def native_physical_bounds(name: str) -> tuple[float | None, float | None]:
    """Return ``(lower, upper)`` physical bounds in storage units, ``None`` if unbounded."""
    key = normalize_param_name(name)
    if key == "KIN":
        return _KIN_INTERVAL_DEG
    if key in _UNIT_INTERVAL:
        return 0.0, 1.0
    if key in _SIGNED_UNIT_INTERVAL:
        return -1.0, 1.0
    if key in _NON_NEGATIVE:
        return 0.0, None
    return None, None
