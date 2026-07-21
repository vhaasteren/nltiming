"""Physical (frame) charts: block reparameterizations between the sampling
frame and the engine frame (Kepler <-> Laplace-Lagrange for low-e binaries).

Distinct from the per-axis *prior charts* (``affine_normal`` / ``prior_pit``)
that map delta <-> z: a physical chart changes which physical coordinates the
plan names, priors, and sampler see, while the engine delay model and its
fitpar frame stay untouched. Slot-preserving: each sampling axis occupies the
fitpar slot of the engine axis it replaces, so ``fitpar_index`` is valid in
both frames.

No PINT/tempo2 imports here; the Kepler<->Laplace identities are copied as
pure NumPy (tests cross-check against PINT).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import ClassVar, Literal, Mapping, Protocol, runtime_checkable

import numpy as np

from .pint_compat import resolve_parameter_alias
from .selection import canonical_fitpars, fitpar_suffixes

PI = float(np.pi)  # plain Python floats: xp-generic code below mixes
DEG2RAD = PI / 180.0  # them with numpy or traced jax arrays
RAD2DEG = 180.0 / PI
TWO_PI = 2.0 * PI
_PI_50 = Decimal("3.14159265358979323846264338327950288419716939937511")

ENGINE_TRIPLE = ("ECC", "OM", "T0")
SAMPLE_TRIPLE = ("EPS1", "EPS2", "TASC")

BinaryChartMode = Literal["off", "auto", "on"]

_SKIP_REASONS = (
    "mode_off",
    "incomplete_triple",
    "already_laplace",
    "e_ref_not_positive",
    "e_ref_above_e_max",
    "no_sampled_axis",
    "split_ecc_om_dispositions",
    "prior_on_kepler_axis",
    "pint_prior_on_kepler_axis",
    "seam_reachable_with_secular_terms",
    "origin_uncertified_backend",
    "unsupported_binary_model",
)

# Fallback list of seam-relevant secular terms, used ONLY when the engine
# does not provide a per-group binary_chart_capability() (§2.4). The capability
# descriptor is authoritative — a name search cannot see derived evolution
# (DDGR) or model-specific epoch dependence (T2 families).
SECULAR_SEAM_PARAMS = ("OMDOT", "PBDOT", "EDOT", "A1DOT", "XDOT")

# Binary-model families whose secular (post-Keplerian) evolution is DERIVED from
# the model — unambiguously from the model NAME — rather than exposed as explicit
# fitpars, so a name-search over SECULAR_SEAM_PARAMS cannot see it. When the
# fallback recognizes such a model on the pulsar's PINT object it conservatively
# marks the GR-derived rates present so the seam guard engages.
#
# T2 is deliberately NOT in this set (review): T2 is a *wrapper* whose secular
# behavior depends on its resolved sub-model — a bare T2-DD binary has no
# derived secular evolution, so flagging every T2 would spuriously demote many
# healthy binaries. A T2 with active secular rates is caught by the explicit /
# populated-value name search below when PINT exposes the values; a definitive
# T2 (or any sub-model) determination needs binary-instance introspection, which
# is the job of the authoritative per-group ``binary_chart_capability`` (§2.4)
# on the engine adapter — this name-based fallback is the stopgap.
_GR_DERIVED_BINARY_MODELS = frozenset({"DDGR"})
_GR_DERIVED_SECULAR_TERMS = ("OMDOT", "PBDOT")

# Strict-support margin: every accepted prior support must satisfy
# e <= 1 - DISK_MARGIN (the model needs e < 1 strictly, not e <= 1), and the
# seam/origin guards use the same closed supports. Sampled values are NEVER
# numerically clipped anywhere — supports are shrunk before sampling.
DISK_MARGIN = 1e-6

# Named, versioned identifier of the default prior package the chart applies
# under `auto`. Bump the version on ANY change to the default-box recipe (WLS
# source, scale, shrink rule, margin).
DEFAULT_PRIOR_PACKAGE = "nlt-eps-wls-boxes-v1"

BinaryChartPrior = Literal["sampling_frame"]  # "pushforward" reserved (I.4)


@dataclass(frozen=True)
class KeplerLaplacePolicy:
    mode: BinaryChartMode = "auto"
    e_max: float = 0.1
    prior: BinaryChartPrior = "sampling_frame"

    def __post_init__(self) -> None:
        if self.mode not in ("off", "auto", "on"):
            raise ValueError(
                f"binary_chart mode must be off|auto|on, got {self.mode!r}"
            )
        if not (0.0 < float(self.e_max) <= 1.0):
            raise ValueError("e_max must satisfy 0 < e_max <= 1")
        if self.prior != "sampling_frame":
            raise ValueError(
                "binary_chart prior='pushforward' is reserved for a future "
                "release (it requires a joint prior bijector; see design doc "
                "I.4); v1 supports prior='sampling_frame' only"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "e_max": float(self.e_max),
            "prior": self.prior,
            "default_prior_package": DEFAULT_PRIOR_PACKAGE,
        }


def coerce_binary_chart_policy(value) -> KeplerLaplacePolicy:
    if value is None or value == "auto":
        return KeplerLaplacePolicy(mode="auto")
    if value == "on":
        return KeplerLaplacePolicy(mode="on")
    if value == "off":
        return KeplerLaplacePolicy(mode="off")
    if isinstance(value, KeplerLaplacePolicy):
        return value
    raise TypeError(
        "binary_chart must be None, 'off'|'auto'|'on', or KeplerLaplacePolicy; "
        f"got {type(value).__name__}"
    )


# ---------------------------------------------------------------------------
# Pure maps (scalar SoT; used for references, decode, and tests)
# ---------------------------------------------------------------------------


def laplace_from_kepler(ecc, om_deg, t0, pb):
    """(ECC, OM_deg, T0) -> (EPS1, EPS2, TASC). Units: dimensionless/deg/days."""
    om = float(om_deg) * DEG2RAD
    return (
        float(ecc) * np.sin(om),
        float(ecc) * np.cos(om),
        float(t0) - float(pb) * om / TWO_PI,
    )


def kepler_from_laplace(eps1, eps2, tasc, pb):
    """(EPS1, EPS2, TASC) -> (ECC, OM_deg in [0, 360), T0)."""
    ecc = float(np.hypot(eps1, eps2))
    om = float(np.arctan2(eps1, eps2))
    if om < 0.0:
        om += TWO_PI
    return ecc, om * RAD2DEG, float(tasc) + float(pb) * om / TWO_PI


def kepler_from_laplace_vec(eps1, eps2, tasc, pb):
    """Vectorized decode variant; identical branch convention ([0, 2pi))."""
    eps1 = np.asarray(eps1, dtype=float)
    eps2 = np.asarray(eps2, dtype=float)
    ecc = np.hypot(eps1, eps2)
    om = np.arctan2(eps1, eps2)
    om = np.where(om < 0.0, om + TWO_PI, om)
    return (
        ecc,
        om * RAD2DEG,
        np.asarray(tasc, dtype=float) + np.asarray(pb) * om / TWO_PI,
    )


def unwrap_om_delta_deg(om_abs_deg, om_ref_deg):
    return (float(om_abs_deg) - float(om_ref_deg) + 180.0) % 360.0 - 180.0


def logabsdet_kepler_from_laplace(eps1, eps2):
    """Diagnostics/tests only — never a posterior-density term (Part I.3)."""
    e = float(np.hypot(eps1, eps2))
    if e == 0.0:
        return float(-np.inf)
    return float(-np.log(e) + np.log(RAD2DEG))


def tasc_ref_decimal(t0_ref: str, pb_ref: str, om_ref_deg: str) -> str:
    """Exact-decimal TASC reference string (prec 50): T0 - PB*omega/(2*pi)."""
    with localcontext() as ctx:
        ctx.prec = 50
        omega = Decimal(om_ref_deg) * _PI_50 / Decimal(180)
        tasc = Decimal(t0_ref) - Decimal(pb_ref) * omega / (2 * _PI_50)
        return str(tasc)


# ---------------------------------------------------------------------------
# Generic physical-chart protocol (the extensible seam contract)
# ---------------------------------------------------------------------------


def _set_slot(vec, i, value):
    """xp-generic single-slot write (jax .at path or in-place numpy)."""
    if hasattr(vec, "at"):
        return vec.at[i].set(value)
    vec[i] = value
    return vec


@runtime_checkable
class PhysicalChart(Protocol):
    name: str  # chart family, e.g. "kepler_laplace"
    suffix: str
    engine_names: tuple[str, ...]  # engine fitpars this chart replaces
    sample_names: tuple[str, ...]  # sampling axes introduced (same slots)
    engine_slots: tuple[int, ...]  # fitpar slots of the replaced block
    dependency_slots: tuple[int, ...]  # read-only slots the map uses (e.g. PB)

    @property
    def chart_id(self) -> tuple[str, str]:
        """Stable identifier ``(family, suffix)`` for manifests/registries."""

    def in_domain(self, vec) -> bool:
        """Whether a full-length sampling-frame delta vector lies strictly
        inside this chart's physical domain (e.g. ``e <= 1 - DISK_MARGIN``)."""

    def apply_delta(self, vec, xp):
        """Given the full-length sampling-frame delta vector (already
        scattered), return it with this chart's engine slots overwritten by
        engine-frame deltas. xp-generic and jax-traceable; must not write any
        slot outside ``engine_slots``."""

    def write_frame_block(self, B: np.ndarray, delta_full: np.ndarray) -> None:
        """Write this chart's analytic Jacobian block and dependency-coupling
        entries into the frame-change matrix ``B`` in place, evaluated at the
        sampling point ``delta_full``."""

    def decode(
        self,
        samples: Mapping[str, np.ndarray],
        dependency: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Derived engine-frame posterior columns from absolute sampling-frame
        arrays, on the SAME reference-local branch as ``apply_delta``."""

    def record(self, **kwargs) -> dict: ...


def check_chart_compatibility(charts) -> None:
    """**v1 disjoint-composition rule**: charts compose only if their engine
    slots are pairwise disjoint AND no chart's engine slots intersect another
    chart's dependency slots (that would make application order semantic).
    ValueError otherwise, naming the offending charts and slots.
    """
    claimed: dict[int, str] = {}
    for ch in charts:
        for slot in ch.engine_slots:
            if slot in claimed:
                raise ValueError(
                    f"physical charts {claimed[slot]!r} and {ch.name!r} "
                    f"(suffix={ch.suffix!r}) both claim engine slot {slot}"
                )
            claimed[slot] = ch.name
    for ch in charts:
        overlap = set(ch.dependency_slots) & set(claimed) - set(ch.engine_slots)
        for slot in sorted(overlap):
            if claimed[slot] != ch.name:
                raise ValueError(
                    f"physical chart {ch.name!r} (suffix={ch.suffix!r}) "
                    f"depends on slot {slot}, which chart {claimed[slot]!r} "
                    "rewrites; application order would be semantic"
                )


# ---------------------------------------------------------------------------
# Kepler-Laplace chart (first concrete PhysicalChart)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeplerLaplaceChart:
    """One activated group. All reference floats are engine-native units.

    Slot convention (frozen): sample axes occupy the engine slots in the fixed
    role order EPS1@ECC-slot, EPS2@OM-slot, TASC@T0-slot. ``omega_ref_rad`` is
    the **normalized** OM reference (par-file OM reduced mod 360 into [0, 360)
    at Decimal precision) in radians. The reference identity
    ``T0_ref = TASC_ref + PB_ref * omega_ref / (2*pi)`` defines TASC_ref via
    ``tasc_ref_decimal`` with the same normalized OM and the exact-decimal PB
    string. ``om_ref_raw_str`` preserves the unnormalized par-file string for
    the manifest record only.
    """

    suffix: str
    engine_names: tuple[str, str, str]  # actual fitpar strings (suffixed)
    sample_names: tuple[str, str, str]  # ("EPS1","EPS2","TASC") + suffix
    slots: tuple[int, int, int]  # fitpar indices, (ECC, OM, T0) order
    pb_name: str | None  # free-PB fitpar name in this group
    pb_slot: int | None
    a1_name: str | None  # free A1 fitpar name, if any
    a1_ref: float | None  # A1 reference (free OR fixed); xe2 only
    e_ref: float
    omega_ref_rad: float  # in [0, 2*pi)
    eps1_ref: float  # e_ref * sin(omega_ref)
    eps2_ref: float  # e_ref * cos(omega_ref)
    pb_ref: float
    t0_ref_str: str  # exact decimal strings
    tasc_ref_str: str
    ecc_ref_str: str
    om_ref_raw_str: str  # par-file OM string, unnormalized
    om_ref_norm_str: str  # Decimal OM % 360, in [0, 360)

    def engine_delta_from_sample_delta(self, d_eps1, d_eps2, d_tasc, *, d_pb, xp):
        """Delta-form conversion (precision-critical; see §3.2). xp-generic
        (numpy or jax.numpy); scalars or same-shape arrays."""
        eps1 = self.eps1_ref + d_eps1
        eps2 = self.eps2_ref + d_eps2
        e = xp.sqrt(eps1 * eps1 + eps2 * eps2)
        domega = xp.arctan2(eps1, eps2) - self.omega_ref_rad
        domega = (domega + PI) % TWO_PI - PI
        d_ecc = e - self.e_ref
        d_om = domega * RAD2DEG
        d_t0 = (
            d_tasc
            + (self.pb_ref * domega + d_pb * (self.omega_ref_rad + domega)) / TWO_PI
        )
        return d_ecc, d_om, d_t0

    def _omega_at(self, eps1: float, eps2: float) -> float:
        """omega on the reference branch: omega_ref + wrapped delta."""
        domega = float(np.arctan2(eps1, eps2)) - self.omega_ref_rad
        domega = (domega + PI) % TWO_PI - PI
        return self.omega_ref_rad + domega

    def jacobian_at(
        self, d_eps1: float = 0.0, d_eps2: float = 0.0, d_pb: float = 0.0
    ) -> np.ndarray:
        """Analytic J = d(ECC, OM_deg, T0)/d(EPS1, EPS2, TASC) at the sampling
        point ref + delta, fixed PB = pb_ref + d_pb. Defaults give the engine
        reference. |det J| = (180/pi)/e at the evaluation point. Raises at
        e == 0 exactly (never clip)."""
        eps1 = self.eps1_ref + float(d_eps1)
        eps2 = self.eps2_ref + float(d_eps2)
        e = float(np.hypot(eps1, eps2))
        if e == 0.0:
            raise ValueError(
                f"binary_chart {self.suffix!r}: Jacobian requested at "
                "eps1 = eps2 = 0 (singular point)"
            )
        s, c = eps1 / e, eps2 / e
        pb = self.pb_ref + float(d_pb)
        return np.array(
            [
                [s, c, 0.0],
                [RAD2DEG * c / e, -RAD2DEG * s / e, 0.0],
                [pb * c / (TWO_PI * e), -pb * s / (TWO_PI * e), 1.0],
            ]
        )

    def pb_coupling_at(self, d_eps1: float = 0.0, d_eps2: float = 0.0) -> float:
        """dT0/dPB at fixed (EPS1, EPS2, TASC): omega/(2*pi) on the reference
        branch, at ref + delta (defaults: the reference, omega_ref/(2*pi))."""
        return (
            self._omega_at(self.eps1_ref + float(d_eps1), self.eps2_ref + float(d_eps2))
            / TWO_PI
        )

    # -- PhysicalChart protocol conformance ---------------------------------
    # ClassVar, not dataclass fields (must not enter __init__):

    name: ClassVar[str] = "kepler_laplace"
    DOMAIN_MAX_E: ClassVar[float] = 1.0  # bound Keplerian orbit: e < 1 (§5.4)

    @property
    def engine_slots(self) -> tuple[int, ...]:
        return self.slots

    @property
    def dependency_slots(self) -> tuple[int, ...]:
        return () if self.pb_slot is None else (self.pb_slot,)

    @property
    def chart_id(self) -> tuple[str, str]:
        return (self.name, self.suffix)

    def in_domain(self, vec) -> bool:
        s1, s2 = self.slots[0], self.slots[1]
        e = float(np.hypot(self.eps1_ref + vec[s1], self.eps2_ref + vec[s2]))
        return e <= self.DOMAIN_MAX_E - DISK_MARGIN

    def apply_delta(self, vec, xp):
        s1, s2, s3 = self.slots
        d_pb = vec[self.pb_slot] if self.pb_slot is not None else 0.0
        d_ecc, d_om, d_t0 = self.engine_delta_from_sample_delta(
            vec[s1], vec[s2], vec[s3], d_pb=d_pb, xp=xp
        )
        vec = _set_slot(vec, s1, d_ecc)
        vec = _set_slot(vec, s2, d_om)
        return _set_slot(vec, s3, d_t0)

    def write_frame_block(self, B: np.ndarray, delta_full: np.ndarray) -> None:
        s1, s2, s3 = self.slots
        d_pb = delta_full[self.pb_slot] if self.pb_slot is not None else 0.0
        B[np.ix_([s1, s2, s3], [s1, s2, s3])] = self.jacobian_at(
            delta_full[s1], delta_full[s2], d_pb
        )
        if self.pb_slot is not None:
            B[s3, self.pb_slot] = self.pb_coupling_at(delta_full[s1], delta_full[s2])

    def decode(self, samples, dependency=None):
        """Derived (ECC, OM, T0) columns on the SAME reference-local branch as
        the likelihood (never the global [0, 360) normalization). ``samples``:
        absolute sampling-frame arrays keyed by ``sample_names``; ``dependency``:
        optional {pb_name: array} for a sampled PB.
        """
        d1 = np.asarray(samples[self.sample_names[0]], float) - self.eps1_ref
        d2 = np.asarray(samples[self.sample_names[1]], float) - self.eps2_ref
        dt = np.asarray(samples[self.sample_names[2]], float) - float(self.tasc_ref_str)
        d_pb = (
            np.asarray(dependency[self.pb_name], float) - self.pb_ref
            if dependency and self.pb_name in (dependency or {})
            else 0.0
        )
        d_ecc, d_om, d_t0 = self.engine_delta_from_sample_delta(
            d1, d2, dt, d_pb=d_pb, xp=np
        )
        return {
            self.engine_names[0]: self.e_ref + d_ecc,
            self.engine_names[1]: float(self.om_ref_norm_str) + d_om,
            self.engine_names[2]: float(self.t0_ref_str) + d_t0,
        }

    def record(
        self,
        *,
        enabled: bool,
        reason: str | None,
        dispositions: Mapping[str, str] | None,
        xe2_us: float | None,
        seam_guard: Mapping[str, object] | None = None,
        origin_guard: Mapping[str, object] | None = None,
        capability_source: str | None = None,
    ) -> dict:
        return {
            "suffix": self.suffix,
            "enabled": bool(enabled),
            "reason": reason,
            "engine_names": list(self.engine_names),
            "sample_names": list(self.sample_names) if enabled else None,
            "dispositions": dict(dispositions) if dispositions else None,
            "e_ref": float(self.e_ref),
            "pb_ref": float(self.pb_ref),
            "pb_fitpar": self.pb_name,
            "theta_ref_engine": {
                "ECC": self.ecc_ref_str,
                "OM": self.om_ref_raw_str,
                "OM_normalized": self.om_ref_norm_str,
                "T0": self.t0_ref_str,
            },
            "theta_ref_sample": (
                {
                    "EPS1": repr(self.eps1_ref),
                    "EPS2": repr(self.eps2_ref),
                    "TASC": self.tasc_ref_str,
                }
                if enabled
                else None
            ),
            "xe2_us": None if xe2_us is None else float(xe2_us),
            "domain": {"max_e": self.DOMAIN_MAX_E, "margin": DISK_MARGIN},
            "seam_guard": None if seam_guard is None else dict(seam_guard),
            "origin_guard": None if origin_guard is None else dict(origin_guard),
            "capability_source": capability_source,
        }


# ---------------------------------------------------------------------------
# Candidacy (§2.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartCandidate:
    """A suffix group inspected for charting; ``chart`` is None when the group
    is structurally ineligible (record-only)."""

    suffix: str
    engine_names: tuple[str | None, str | None, str | None]  # found fitpars
    chart: KeplerLaplaceChart | None
    skip_reason: str | None  # candidacy-stage reason, else None
    e_ref: float | None
    capability: "object | None" = None
    secular_terms: tuple[str, ...] = ()


def _group_fitpars(pulsar) -> dict[str, dict[str, str]]:
    """suffix -> {base_canonical_name: fitpar_string}, using canonical_fitpars
    + fitpar_suffixes (no new suffix parser). A group exists only if it carries
    at least one of the six chart names (ENGINE_TRIPLE | SAMPLE_TRIPLE); PB and
    A1 are recorded as accessories of such groups and never create a group by
    themselves."""
    core = set(ENGINE_TRIPLE) | set(SAMPLE_TRIPLE)
    wanted = core | {"PB", "A1"}
    groups: dict[str, dict[str, str]] = {}
    for fitpar in canonical_fitpars(pulsar):
        for suffix in fitpar_suffixes(pulsar, fitpar):
            base = fitpar[: len(fitpar) - len(suffix)] if suffix else fitpar
            base = resolve_parameter_alias(base)
            if base in wanted:
                groups.setdefault(suffix, {})[base] = fitpar
    return {s: g for s, g in groups.items() if set(g) & core}


def resolve_chart_candidates(pulsar, engine, policy) -> tuple[ChartCandidate, ...]:
    """Structural candidacy (Part I.2.1 rows 8-11) + warnings W1-W3.

    Emits W1/W2 under "auto" and "on"; W3 under "on" only. The auto e_max skip
    is silent. Returns one ChartCandidate per suffix group that carries at least
    one of ECC/OM/T0/EPS1/EPS2/TASC as a free fitpar; PB/A1-only groups are
    omitted entirely. Resolves the engine's capability descriptor (§2.4) per
    suffix group.
    """
    if policy.mode == "off":
        return ()
    refs = engine.reference_theta_exact()
    cap_fn = getattr(engine, "binary_chart_capability", None)
    groups = _group_fitpars(pulsar)
    single_group = len(groups) == 1
    out: list[ChartCandidate] = []
    for suffix, found in sorted(groups.items()):
        capability = (
            cap_fn("kepler_laplace", suffix) if cap_fn is not None else None
        )  # per group (§2.4)
        allow_unsuffixed = (not suffix) or single_group  # accessory rule
        triple = tuple(found.get(b) for b in ENGINE_TRIPLE)
        names3 = (found.get("ECC"), found.get("OM"), found.get("T0"))
        if any(found.get(b) for b in SAMPLE_TRIPLE):
            if policy.mode == "on":
                warnings.warn(  # W3
                    f"nltiming binary_chart: group suffix={suffix!r} already "
                    "carries Laplace fitpars (EPS1/EPS2/TASC); the chart is a "
                    "no-op and stays off for this group.",
                    UserWarning,
                    stacklevel=3,
                )
            out.append(
                ChartCandidate(
                    suffix,
                    names3,
                    None,
                    "already_laplace",
                    None,
                    capability=capability,
                )
            )
            continue
        if not all(triple):
            om, t0, ecc = found.get("OM"), found.get("T0"), found.get("ECC")
            if om and t0 and not ecc:
                warnings.warn(  # W1
                    "nltiming binary_chart: OM and T0 are free fit parameters "
                    f"but ECC is not (suffix={suffix!r}). The OM-T0 degeneracy "
                    "cannot be removed by a Kepler<->Laplace chart without a "
                    "free ECC; T0 sampling is expected to remain poorly "
                    "conditioned. Set ECC to fit in the par file, or pass "
                    "binary_chart='off'.",
                    UserWarning,
                    stacklevel=3,
                )
            elif t0 and not om:
                warnings.warn(  # W2
                    "nltiming binary_chart: T0 is a free fit parameter but OM "
                    f"is not (suffix={suffix!r}). Free OM and ECC as well, or "
                    "pass binary_chart='off'.",
                    UserWarning,
                    stacklevel=3,
                )
            out.append(
                ChartCandidate(
                    suffix,
                    names3,
                    None,
                    "incomplete_triple",
                    None,
                    capability=capability,
                )
            )
            continue
        ecc_fp, om_fp, t0_fp = triple
        if capability is not None and (
            capability.kepler_convention != "dd" or not capability.supports_domain
        ):
            if policy.mode == "on":
                warnings.warn(
                    f"nltiming binary_chart: suffix={suffix!r} — the engine "
                    "declares an unsupported binary model or domain "
                    f"(convention={capability.kepler_convention!r}, "
                    f"supports_domain={capability.supports_domain}); the "
                    "chart stays off.",
                    UserWarning,
                    stacklevel=3,
                )
            out.append(
                ChartCandidate(
                    suffix,
                    names3,
                    None,
                    "unsupported_binary_model",
                    None,
                    capability=capability,
                )
            )
            continue
        try:
            e_ref = float(refs[ecc_fp])
        except (KeyError, ValueError):
            e_ref = float("nan")
        if not np.isfinite(e_ref) or e_ref <= 0.0:
            if policy.mode == "on":
                warnings.warn(
                    f"nltiming binary_chart: suffix={suffix!r} has "
                    f"e_ref={e_ref!r}; the Kepler<->Laplace map is singular at "
                    "e=0 and the chart stays off. Refit with a small nonzero "
                    "ECC to use the chart.",
                    UserWarning,
                    stacklevel=3,
                )
            out.append(
                ChartCandidate(
                    suffix,
                    names3,
                    None,
                    "e_ref_not_positive",
                    e_ref,
                    capability=capability,
                )
            )
            continue
        if policy.mode == "auto" and e_ref >= policy.e_max:
            # Silent by design: the manifest records the skip. A warning here
            # would fire once per pulsar on PTA ensembles with moderate-e
            # binaries — pure noise.
            out.append(
                ChartCandidate(
                    suffix,
                    names3,
                    None,
                    "e_ref_above_e_max",
                    e_ref,
                    capability=capability,
                )
            )
            continue
        fitpars = canonical_fitpars(pulsar)
        index = {name: i for i, name in enumerate(fitpars)}
        pb_fp = found.get("PB")
        pb_ref_str = (
            str(refs[pb_fp])
            if pb_fp
            else _accessory_ref_string(
                pulsar,
                refs,
                suffix,
                "PB",
                required=True,
                allow_unsuffixed=allow_unsuffixed,
            )
        )
        a1_ref_str = _accessory_ref_string(
            pulsar,
            refs,
            suffix,
            "A1",
            required=False,
            allow_unsuffixed=allow_unsuffixed,
        )
        om_raw_str = str(refs[om_fp])
        om_norm_str = _normalize_om_deg(om_raw_str)
        omega = float(om_norm_str) * DEG2RAD
        chart = KeplerLaplaceChart(
            suffix=suffix,
            engine_names=(ecc_fp, om_fp, t0_fp),
            sample_names=tuple(b + suffix for b in SAMPLE_TRIPLE),
            slots=(index[ecc_fp], index[om_fp], index[t0_fp]),
            pb_name=pb_fp,
            pb_slot=index[pb_fp] if pb_fp else None,
            a1_name=found.get("A1"),
            a1_ref=None if a1_ref_str is None else float(a1_ref_str),
            e_ref=e_ref,
            omega_ref_rad=omega,
            eps1_ref=e_ref * float(np.sin(omega)),
            eps2_ref=e_ref * float(np.cos(omega)),
            pb_ref=float(pb_ref_str),
            t0_ref_str=str(refs[t0_fp]),
            tasc_ref_str=tasc_ref_decimal(str(refs[t0_fp]), pb_ref_str, om_norm_str),
            ecc_ref_str=str(refs[ecc_fp]),
            om_ref_raw_str=om_raw_str,
            om_ref_norm_str=om_norm_str,
        )
        secular = (
            tuple(sorted(capability.secular_terms))
            if capability is not None
            else tuple(sorted(_present_secular_terms(pulsar, suffix)))
        )
        out.append(
            ChartCandidate(
                suffix,
                triple,
                chart,
                None,
                e_ref,
                capability=capability,
                secular_terms=secular,
            )
        )
    return tuple(out)


def _normalize_om_deg(om_str: str) -> str:
    """Exact-decimal OM normalization into [0, 360) at prec 50."""
    with localcontext() as ctx:
        ctx.prec = 50
        om = Decimal(om_str) % Decimal(360)
        if om < 0:
            om += Decimal(360)
        return str(om)


def _accessory_ref_string(
    pulsar, refs, suffix: str, base: str, *, required: bool, allow_unsuffixed: bool
) -> str | None:
    """Exact reference string for an accessory parameter (PB, A1) that need not
    be a free fitpar (a fixed-but-present A1 must still feed ``xe2_us``).

    Fixed fallback chain (frozen):
      1. ``reference_theta_exact()['{base}{suffix}']``;
      2. the suffixed PINT-model value via ``lookup_pint_param``, stringified
         with ``repr(float(value))``;
      3. the UNSUFFIXED forms of 1-2, but only when ``allow_unsuffixed``;
      4. ``required=True`` (PB): raise; ``required=False`` (A1): return None.
    """
    keys = [f"{base}{suffix}"] if suffix else [base]
    if suffix and allow_unsuffixed:
        keys.append(base)
    for key in keys:
        if key in refs:
            return str(refs[key])
    from .units import lookup_pint_param  # no PINT import in this module

    model = pulsar.pint_model()
    if model is not None:
        for key in keys:
            param = lookup_pint_param(model, key)
            value = getattr(param, "value", None) if param is not None else None
            if value is not None:
                return repr(float(value))
    if required:
        raise ValueError(
            f"binary_chart: no {base} reference available for "
            f"suffix={suffix!r} (neither engine.reference_theta_exact() nor "
            f"the PINT model exposes it); free {base} or pass "
            "binary_chart='off'"
        )
    return None


# ---------------------------------------------------------------------------
# Selector normalization and activation (§2.2)
# ---------------------------------------------------------------------------


def normalize_inference_selectors(inference, candidates):
    """Rewrite TimingInference selectors before plan resolution (§I.2).

    - 'TASC' (base or exact suffixed) -> the group's T0 fitpar, for groups with
      a structural chart. Base 'TASC' rewrites for every such group; if none
      exists the key is left untouched.
    - 'EPS1'/'EPS2' (any suffix) -> ValueError (dispositions are declared on
      engine names; ECC and OM must share one disposition for the chart).
    Returns a TimingInference with rewritten keys; presets pass through.
    """
    if inference.preset == "default_delta" or not inference.marginalize:
        return inference
    by_suffix = {c.suffix: c for c in candidates if c.chart is not None}
    rewritten = {}
    for key, marg in inference.marginalize.items():
        base, suffix = _split_chart_key(key, by_suffix)
        if base in ("EPS1", "EPS2"):
            raise ValueError(
                f"inference selector {key!r}: dispositions are declared on "
                "engine names. Use ECC/OM (they must share one disposition "
                "for the Kepler<->Laplace chart to activate); see the "
                "binary_chart documentation."
            )
        if base == "TASC":
            targets = (
                [by_suffix[suffix]] if suffix is not None else list(by_suffix.values())
            )
            if targets:
                for cand in targets:
                    t0_fitpar = cand.chart.engine_names[2]
                    if t0_fitpar in rewritten:
                        raise ValueError(
                            f"inference selectors overlap: {key!r} and a "
                            f"T0-form selector both target {t0_fitpar!r} "
                            "(T0 and TASC are exact aliases; use one)."
                        )
                    rewritten[t0_fitpar] = marg
                continue
        if key in rewritten:
            raise ValueError(
                f"inference selectors overlap: {key!r} targets an axis already "
                "claimed by a TASC-form selector (T0 and TASC are exact "
                "aliases; use one)."
            )
        rewritten[key] = marg
    from .inference import TimingInference

    return TimingInference(marginalize=rewritten)


def _split_chart_key(key: str, by_suffix) -> tuple[str, str | None]:
    """('TASC_epta') -> ('TASC', '_epta') when that suffix group exists;
    ('TASC') -> ('TASC', None); anything else -> (canonical(key), None)."""
    canonical = resolve_parameter_alias(key)
    for base in ("TASC", "EPS1", "EPS2"):
        if canonical == base:
            return base, None
        for suffix in by_suffix:
            if suffix and canonical == base + suffix:
                return base, suffix
    return canonical, None


def activate_charts(
    plan,
    candidates,
    policy,
    *,
    prior_overrides,
    pint_model,
    pulsar,
    engine_design_matrix,
    nonlinear_scale,
    engine_refs,
    prior_policy,
):
    """Activation + slot-preserving rename (Part I.2.1 rows 1-7d).

    Returns ``(plan, resolved, records)`` where ``resolved`` is a tuple of
    ResolvedPhysicalChart. Calls check_chart_compatibility on the activated set
    before returning.
    """
    from dataclasses import replace as dc_replace

    axes = list(plan.axes)
    by_name = {a.name: i for i, a in enumerate(axes)}
    resolved, records = [], []
    for cand in candidates:
        if cand.chart is None:
            records.append(_skip_record(cand))
            continue
        ch = cand.chart
        disp = {en: axes[by_name[en]].disposition for en in ch.engine_names}
        d_ecc, d_om, d_t0 = (disp[n] for n in ch.engine_names)
        reason = None
        seam_record = None
        origin_record = None
        eps_supports = None
        reach_rect = None
        if "sample" not in disp.values():
            reason = "no_sampled_axis"
        elif d_ecc != d_om:
            reason = "split_ecc_om_dispositions"
        elif _kepler_prior_conflict(ch, frozenset(prior_overrides)):
            reason = "prior_on_kepler_axis"
        elif _pint_prior_conflict(pint_model, ch, prior_policy=prior_policy):
            reason = "pint_prior_on_kepler_axis"
        else:
            # §5 support resolution: raises the strict-support ValueError on
            # unbounded / disk-crossing user EPS priors (never demotes those).
            reach_rect, eps_supports = resolved_eps_reachability(
                pulsar,
                ch,
                plan,
                engine_design_matrix,
                nonlinear_scale=nonlinear_scale,
                prior_overrides=prior_overrides,
                engine_refs=engine_refs,
            )
            epoch_exact = (
                cand.capability.epoch_shift_exact
                if cand.capability is not None
                else not cand.secular_terms
            )
            seam_ok, seam_record = _seam_guard(
                ch,
                reach_rect,
                epoch_shift_exact=epoch_exact,
                secular_terms=cand.secular_terms,
            )
            if not seam_ok:
                reason = "seam_reachable_with_secular_terms"
            else:
                origin_ok, origin_record = _origin_guard(
                    ch,
                    reach_rect,
                    origin_certified=(
                        cand.capability.origin_certified
                        if cand.capability is not None
                        else False
                    ),
                    certification_ref=(
                        cand.capability.certification_ref
                        if cand.capability is not None
                        else None
                    ),
                )
                if not origin_ok:
                    reason = "origin_uncertified_backend"
        cap_source = "engine" if cand.capability is not None else "fallback"
        if reason is not None:
            _skip_notice(reason, ch, policy)
            records.append(
                ch.record(
                    enabled=False,
                    reason=reason,
                    dispositions=None,
                    xe2_us=None,
                    seam_guard=seam_record,
                    origin_guard=origin_record,
                    capability_source=cap_source,
                )
            )
            continue
        new_disp = {
            ch.sample_names[0]: d_ecc,
            ch.sample_names[1]: d_om,
            ch.sample_names[2]: d_t0,
        }
        for engine_name, sample_name in zip(ch.engine_names, ch.sample_names):
            i = by_name[engine_name]
            axes[i] = dc_replace(
                axes[i],
                name=sample_name,
                disposition=new_disp[sample_name],
                engine_name=engine_name,
                physical_chart="kepler_laplace",
                linearity_sources=(),
            )
        resolved.append(
            ResolvedPhysicalChart(
                chart=ch,
                eps_supports=tuple(eps_supports),
                reachability_rect=reach_rect,
            )
        )
        records.append(
            ch.record(
                enabled=True,
                reason=None,
                dispositions=new_disp,
                xe2_us=(None if ch.a1_ref is None else 1e6 * ch.a1_ref * ch.e_ref**2),
                seam_guard=seam_record,
                origin_guard=origin_record,
                capability_source=cap_source,
            )
        )
    check_chart_compatibility(tuple(r.chart for r in resolved))
    return plan.with_axes(axes), tuple(resolved), tuple(records)


def _kepler_prior_conflict(chart, prior_override_keys) -> bool:
    """True iff a stored prior-override key targets this group's ECC, OM, or T0
    (base name or the exact suffixed fitpar). 'TASC'/'EPS1'/'EPS2' keys never
    conflict — they are the first-class sampling-frame declarations.
    """
    conflicts = {"ECC", "OM", "T0"}
    for engine_name in chart.engine_names:
        conflicts.add(engine_name)
        conflicts.add(resolve_parameter_alias(engine_name))
    return bool(set(prior_override_keys) & conflicts)


def _pint_prior_conflict(pint_model, chart, *, prior_policy) -> bool:
    """True iff ECC, OM, or T0 would have resolved an informative PINT-model
    prior chart-off (row 7b). Mirrors ``PriorBlock.from_fitpars`` discovery
    exactly — including the policy gate: under ``prior_policy == "explicit"``
    PINT priors are never consulted chart-off, so they never conflict here.
    """
    if pint_model is None or prior_policy == "explicit":
        return False
    from .priors import axis_prior_from_object

    for engine_name in chart.engine_names:  # ECC, OM, and T0
        param = getattr(pint_model, engine_name, None)
        if (
            param is not None
            and axis_prior_from_object(getattr(param, "prior", None)) is not None
        ):
            return True
    return False


def disk_shrink_factor(a: float, b: float, h1: float, h2: float, r_max: float) -> float:
    """Largest c in (0, 1] with hypot(a + c*h1, b + c*h2) <= r_max, for
    a, b >= 0 (use |eps_ref| components). Closed form: if the c=1 corner is
    inside, 1; else the positive root of
    (h1^2+h2^2)c^2 + 2(a*h1+b*h2)c + (a^2+b^2-r_max^2) = 0."""
    if float(np.hypot(a + h1, b + h2)) <= r_max:
        return 1.0
    A = h1 * h1 + h2 * h2
    Bq = 2.0 * (a * h1 + b * h2)
    C = a * a + b * b - r_max * r_max
    if C >= 0.0:  # reference itself outside r_max — cannot happen post-candidacy
        raise ValueError("reference eccentricity outside the physical disk")
    return float((-Bq + np.sqrt(Bq * Bq - 4.0 * A * C)) / (2.0 * A))


@dataclass(frozen=True)
class EpsAxisSupport:
    """Resolved support interval for one charted slot, with provenance.

    **Delta-frame bounds are authoritative**: the prior machinery consumes
    ``(lo_delta, hi_delta)`` directly, and absolute corners are *derived* via
    ``ref + delta`` — the identical runtime expression
    ``engine_delta_from_sample_delta`` evaluates.

    kind: "default_box" (WLS box, shrinkable), "user_bounded" (validated, never
    shrunk), or "fixed" (delta-flat axis pinned at its reference — degenerate
    interval). z-marginalized axes are NOT "fixed": they carry their full prior
    interval.
    """

    kind: str
    ref: float
    lo_delta: float
    hi_delta: float

    def abs_bounds(self) -> tuple[float, float]:
        """Absolute corners via the runtime expression ``ref + delta``."""
        return (self.ref + self.lo_delta, self.ref + self.hi_delta)

    @property
    def worst(self) -> float:
        lo, hi = self.abs_bounds()
        return max(abs(lo), abs(hi))


@dataclass(frozen=True)
class ResolvedPhysicalChart:
    """One activated chart PLUS its resolved support geometry — the immutable
    handoff object between activation and prior construction. The guards and
    `_fill_wls_cheat_priors` consume THESE stored intervals; nothing downstream
    recomputes them.
    """

    chart: KeplerLaplaceChart
    eps_supports: tuple[EpsAxisSupport, EpsAxisSupport]
    reachability_rect: tuple[tuple[float, float], tuple[float, float]]

    def default_box_delta_bounds(self) -> dict[str, tuple[float, float]]:
        """Delta-frame (lower, upper) per sampling-axis name for every
        "default_box" support — exactly what `_fill_wls_cheat_priors` installs
        verbatim as the axis's uniform cheat box (§5). No arithmetic."""
        return {
            self.chart.sample_names[pos]: (sup.lo_delta, sup.hi_delta)
            for pos, sup in enumerate(self.eps_supports)
            if sup.kind == "default_box"
        }


def resolved_eps_reachability(
    pulsar,
    chart,
    plan,
    engine_design_matrix,
    *,
    nonlinear_scale,
    prior_overrides,
    engine_refs,
):
    """Closed rectangle of absolute EPS values **reachable by the nonlinear
    engine decode**, plus per-axis provenance: returns
    ``(rect, (axis1, axis2))`` with ``rect = ((lo1, hi1), (lo2, hi2))``.

    Sampled and z-marginalized EPS axes contribute their full resolved prior
    intervals; delta-flat axes contribute their reference point only. The
    returned supports are stored on the ``ResolvedPhysicalChart`` activation
    object; the guards consume its rect and ``_fill_wls_cheat_priors`` installs
    its ``default_box_delta_bounds()`` verbatim.
    """
    from .whitening import schur_delta_wls

    slots = (chart.slots[0], chart.slots[1])
    refs = (chart.eps1_ref, chart.eps2_ref)
    axis_by_slot = {a.fitpar_index: a for a in plan.axes}
    proper_idx = tuple(
        sorted(plan.indices("sample") + plan.indices("marginalize_z_prior"))
    )
    wls_std_by_slot: dict[int, float] | None = None

    def default_sigma(slot: int) -> float:
        # The ONLY place charted-EPS default sigmas are ever computed. Same
        # recipe as the generic cheat path: synthesized EPS names never resolve
        # a PINT parameter, so the WLS marginal is always operative here. M_s
        # uses THIS chart's B only.
        nonlocal wls_std_by_slot
        if wls_std_by_slot is None:
            m_s = engine_design_matrix @ frame_change_matrix(
                len(plan.fitpars), (chart,)
            )
            wls = schur_delta_wls(
                pulsar=pulsar,
                partition=plan,
                variance=np.asarray(pulsar.toaerrs, dtype=float) ** 2,
                design_matrix=m_s,
                idx_kept=proper_idx,
                idx_marginalized=plan.indices("marginalize_delta_flat"),
            )
            wls_std_by_slot = dict(zip(proper_idx, np.sqrt(np.diag(wls.covariance))))
        return float(wls_std_by_slot[slot])

    supports: list[EpsAxisSupport] = []
    for pos, slot in enumerate(slots):
        axis = axis_by_slot[slot]
        ref = refs[pos]
        if axis.disposition == "marginalize_delta_flat":
            supports.append(EpsAxisSupport("fixed", ref, 0.0, 0.0))
            continue
        prior = materialize_eps_override(
            prior_overrides,
            chart,
            pos,
            pulsar=pulsar,
            plan=plan,
            engine_refs=engine_refs,
        )
        if prior is not None:
            lo_d, hi_d = _delta_bounds_from_axis_prior(prior, chart, pos)
            supports.append(EpsAxisSupport("user_bounded", ref, lo_d, hi_d))
            continue
        h = nonlinear_scale * default_sigma(slot)
        supports.append(EpsAxisSupport("default_box", ref, -h, +h))

    # Common-factor shrink of the default boxes only (§5): user-bounded and
    # fixed axes are constants in the corner condition.
    r_max = chart.DOMAIN_MAX_E - DISK_MARGIN
    d_idx = [i for i, s in enumerate(supports) if s.kind == "default_box"]
    if d_idx:
        h = [0.0, 0.0]
        const = [supports[i].worst for i in range(2)]
        for i in d_idx:
            h[i] = supports[i].hi_delta
            const[i] = abs(refs[i])
        c = disk_shrink_factor(const[0], const[1], h[0], h[1], r_max)
        if c < 1.0:
            for i in d_idx:
                supports[i] = EpsAxisSupport(
                    "default_box", refs[i], -c * h[i], +c * h[i]
                )

    # Absolute corners derived ONLY via EpsAxisSupport.abs_bounds() — the
    # runtime `ref + delta` expression (precision fix).
    rect = tuple(s.abs_bounds() for s in supports)
    if (
        float(
            np.hypot(
                max(abs(rect[0][0]), abs(rect[0][1])),
                max(abs(rect[1][0]), abs(rect[1][1])),
            )
        )
        > r_max
    ):
        raise ValueError(
            f"binary_chart {chart.suffix!r}: the joint EPS prior support "
            f"reaches e > {r_max:.6f} (outside the physical eccentricity "
            "disk). Narrow the EPS1/EPS2 priors, or use the joint disk "
            "prior when available (design doc §15)."
        )
    return rect, tuple(supports)


def materialize_eps_override(prior_overrides, chart, pos, *, pulsar, plan, engine_refs):
    """THE single materialization path for a charted EPS axis's user override
    — shared verbatim by ``resolved_eps_reachability`` (guard bounds) and by
    ``_resolve_prior_overrides`` (the sampled prior). Returns the final
    delta-frame ``AxisPrior`` or None.
    """
    from .priors import PriorBuildContext, resolve_prior_override, spec_for_target

    base = ("EPS1", "EPS2")[pos]
    hits = [
        key
        for key in prior_overrides
        if _chart_key_base(resolve_parameter_alias(key), chart.suffix) == base
    ]
    if not hits:
        return None
    if len(hits) > 1:
        raise ValueError(
            f"binary_chart {chart.suffix!r}: overlapping prior overrides "
            f"{sorted(hits)!r} both target {chart.sample_names[pos]!r}; "
            "declare exactly one (base or exact-suffixed, not both)."
        )
    target = chart.sample_names[pos]
    spec = spec_for_target(
        pulsar,
        prior_overrides[hits[0]],
        target,
        plan.fitpars,
        target_suffix=chart.suffix,
    )
    rename = dict(zip(chart.engine_names, chart.sample_names))
    proper_names = tuple(rename.get(name, name) for name in plan.proper)
    refs_ctx = PriorBuildContext(
        refs={**engine_refs, **sampling_reference_strings((chart,))},
        fitpars=plan.fitpars,
        sampled=proper_names,
    )
    return resolve_prior_override(target, spec, refs_ctx)


def _delta_bounds_from_axis_prior(prior, chart, pos):
    """Delta-frame (lo, hi) from a materialized delta-frame AxisPrior — passed
    through verbatim, no reference arithmetic.

    Raises the §5 strict-support ValueError for family 'normal' (unbounded
    support) and for 'log_uniform' (absolute positive-valued support, a
    semantics incompatibility with a signed EPS component); accepts
    'uniform'/'truncated_normal' bounds verbatim.
    """
    if prior.family == "normal":
        raise ValueError(
            f"binary_chart {chart.suffix!r}: prior for "
            f"{chart.sample_names[pos]!r} has unbounded support (family "
            "'normal'); EPS priors must be bounded inside the eccentricity "
            "disk (use uniform/truncated_normal, or the joint disk prior "
            "when available, design doc §15)."
        )
    if prior.family == "log_uniform":
        raise ValueError(
            f"binary_chart {chart.suffix!r}: prior family 'log_uniform' "
            "describes an absolute positive-valued support and cannot "
            f"apply to the signed component {chart.sample_names[pos]!r}."
        )
    return float(prior.lower), float(prior.upper)


def _rect_ray_intersects(rect, direction) -> bool:
    """Exact closed-rectangle vs ray-from-origin intersection (slab method).

    rect = ((x0, x1), (y0, y1)); ray = {r*direction : r >= 0}, origin included.
    """
    lo, hi = 0.0, float("inf")
    for (a0, a1), d in zip(rect, direction):
        if d == 0.0:
            if not (a0 <= 0.0 <= a1):
                return False
            continue
        r0, r1 = sorted((a0 / d, a1 / d))
        lo, hi = max(lo, r0), min(hi, r1)
        if lo > hi:
            return False
    return True


def _seam_guard(chart, reach_rect, *, epoch_shift_exact, secular_terms):
    """Row 7c — exact support geometry, not a probability statement.

    Inactive (returns (True, None)) when ``epoch_shift_exact``. Otherwise the
    guard passes iff the closed support rectangle does NOT intersect the seam
    ray { -r*(sin w_ref, cos w_ref) : r >= 0 } (origin included).
    """
    if epoch_shift_exact:
        return True, None
    direction = (
        -float(np.sin(chart.omega_ref_rad)),
        -float(np.cos(chart.omega_ref_rad)),
    )
    intersects = _rect_ray_intersects(reach_rect, direction)
    return not intersects, {
        "secular": sorted(secular_terms),
        "support": [list(map(float, iv)) for iv in reach_rect],
        "seam_direction": list(direction),
        "passed": not intersects,
    }


def _origin_guard(chart, reach_rect, *, origin_certified, certification_ref=None):
    """Row 7d. Passes when the backend is (empirically) origin-certified OR the
    closed support rectangle excludes the origin (then the certification is
    never needed and record is None)."""
    (x0, x1), (y0, y1) = reach_rect
    origin_in = (x0 <= 0.0 <= x1) and (y0 <= 0.0 <= y1)
    if not origin_in:
        return True, None
    return bool(origin_certified), {
        "origin_in_support": True,
        "origin_certified": bool(origin_certified),
        "certification_ref": certification_ref,
        "passed": bool(origin_certified),
    }


def _skip_notice(reason, chart, policy) -> None:
    if reason == "no_sampled_axis":
        if policy.mode == "on":
            warnings.warn(
                "nltiming binary_chart='on': every charted axis "
                f"(suffix={chart.suffix!r}) is analytically marginalized; a "
                "chart cannot help marginalized directions and would only "
                "alter the marginalization measure, so it stays off. Pass "
                "binary_chart='off' or sample at least one of ECC/OM/T0.",
                UserWarning,
                stacklevel=4,
            )
        return  # auto: silent by design; the manifest records the skip
    messages = {
        "split_ecc_om_dispositions": (
            f"suffix={chart.suffix!r}: ECC and OM carry different inference "
            "dispositions, which is not representable in Laplace coordinates. "
            "Give ECC and OM one common disposition to activate the chart."
        ),
        "prior_on_kepler_axis": (
            f"suffix={chart.suffix!r}: a prior override targets ECC, OM, or "
            "T0, which do not exist as sampling axes under the chart (and a "
            "T0 density does not transfer to TASC — it would couple TASC to "
            "the eccentricity vector). Deliberate physical priors always "
            "win: the chart is demoted. Re-declare the prior on "
            "EPS1/EPS2/TASC to activate the chart."
        ),
        "pint_prior_on_kepler_axis": (
            f"suffix={chart.suffix!r}: ECC, OM, or T0 carries an informative "
            "PINT-model prior, which cannot be re-expressed on the sampling "
            "axes automatically (dEPS1 dEPS2 = e de dOM: the measures "
            "differ). The chart is demoted so the deliberate prior is "
            "honored; re-declare it on EPS1/EPS2/TASC via priors= to "
            "activate the chart."
        ),
        "seam_reachable_with_secular_terms": (
            f"suffix={chart.suffix!r}: the epoch-shift identity is not exact "
            "(secular/derived evolution present) and the joint EPS prior "
            "support intersects the omega-branch seam ray; the seam carries "
            "an O(rate x PB) likelihood discontinuity, which HMC must never "
            "encounter. The chart is demoted. Narrow the EPS priors to a "
            "branch-safe sector, or wait for the secular eccentricity-vector "
            "chart (design doc §15)."
        ),
        "origin_uncertified_backend": (
            f"suffix={chart.suffix!r}: the EPS prior support contains the "
            "eccentricity origin and this engine backend has not passed the "
            "full-likelihood origin certification (§12.3); origin smoothness "
            "is a per-backend numerical property, not a transferable fixture "
            "result. The chart is demoted. Narrow the EPS priors away from "
            "the origin, or use a certified backend."
        ),
    }
    if policy.mode == "on":
        raise ValueError("nltiming binary_chart='on': " + messages[reason])
    warnings.warn(
        "nltiming binary_chart: chart demoted — " + messages[reason],
        UserWarning,
        stacklevel=4,
    )


def _skip_record(cand: ChartCandidate) -> dict:
    return {
        "suffix": cand.suffix,
        "enabled": False,
        "reason": cand.skip_reason,
        "engine_names": [n for n in cand.engine_names if n],
        "sample_names": None,
        "dispositions": None,
        "e_ref": (
            None
            if cand.e_ref is None or not np.isfinite(cand.e_ref)
            else float(cand.e_ref)
        ),
        "pb_ref": None,
        "pb_fitpar": None,
        "theta_ref_engine": None,
        "theta_ref_sample": None,
        "xe2_us": None,
        "domain": None,
        "seam_guard": None,
        "origin_guard": None,
        "capability_source": ("engine" if cand.capability is not None else "fallback"),
    }


# ---------------------------------------------------------------------------
# Frame-change matrix and sampling references (§2.3)
# ---------------------------------------------------------------------------


def frame_change_matrix(nfit: int, charts, *, delta=None) -> np.ndarray:
    """B with M_s = M_e @ B: column j of M_s is d(residual)/d(sampling axis j).

    Chart-agnostic: delegates every block to ``PhysicalChart.write_frame_block``.
    **Production always calls this at the reference** (``delta=None``). The
    ``delta`` kwarg exists for diagnostics/tests only. Identity when charts is
    empty.
    """
    B = np.eye(nfit)
    d = np.zeros(nfit) if delta is None else np.asarray(delta, dtype=float)
    for ch in charts:
        ch.write_frame_block(B, d)
    return B


def sampling_reference_strings(charts) -> dict[str, str]:
    """Exact-decimal reference strings for the synthesized sampling axes."""
    out: dict[str, str] = {}
    for ch in charts:
        out[ch.sample_names[0]] = repr(ch.eps1_ref)
        out[ch.sample_names[1]] = repr(ch.eps2_ref)
        out[ch.sample_names[2]] = ch.tasc_ref_str
    return out


# ---------------------------------------------------------------------------
# Engine-capability fallback (§2.4) — the capability descriptor is authoritative
# ---------------------------------------------------------------------------


def _present_secular_terms(pulsar, suffix: str) -> set[str]:
    """Fallback-only par inspection (capability descriptor is authoritative).
    Strictly suffix-isolated on BOTH paths: a suffixed group is checked only
    against its own suffixed names — it never inherits another group's
    unsuffixed value; a fixed value counts only if finite and nonzero."""
    fitpars = set(canonical_fitpars(pulsar))
    present: set[str] = set()
    for base in SECULAR_SEAM_PARAMS:
        canonical = resolve_parameter_alias(base)
        key = canonical + suffix if suffix else canonical
        if key in fitpars:
            present.add(canonical)
    model = pulsar.pint_model()
    if model is not None:
        from .units import lookup_pint_param

        for base in SECULAR_SEAM_PARAMS:
            canonical = resolve_parameter_alias(base)
            if canonical in present:
                continue
            key = canonical + suffix if suffix else canonical
            param = lookup_pint_param(model, key)
            value = getattr(param, "value", None) if param is not None else None
            if value is not None and np.isfinite(float(value)) and float(value) != 0.0:
                present.add(canonical)
        # GR-derived binary models (e.g. DDGR) compute their post-Keplerian
        # secular rates internally, so the name search above cannot see them
        # (review correctness fix). Recognize the model family from the loaded
        # PINT object (attribute access only — no PINT class import) and mark the
        # derived rates present so the seam guard engages conservatively.
        binary_param = getattr(model, "BINARY", None)
        binary_type = getattr(binary_param, "value", None)
        if (
            isinstance(binary_type, str)
            and binary_type.upper() in _GR_DERIVED_BINARY_MODELS
        ):
            present.update(
                resolve_parameter_alias(b) for b in _GR_DERIVED_SECULAR_TERMS
            )
    return present


# ---------------------------------------------------------------------------
# Prior-override key expansion (§5.1) — defined here; consumed by the model
# ---------------------------------------------------------------------------


def expand_override_key(pulsar, name, plan, charts) -> tuple[str, ...]:
    """Expand one user prior-override key to target axis names on the plan.

    Union of:
      1. generic engine-name matching (match_fitpars against plan.axis_names) —
         identical to today for every uncharted axis; finds nothing for
         charted-away engine names;
      2. 'TASC'/'EPS1'/'EPS2' (base or exact suffixed) -> the synthesized
         sampling axes of active charts — the first-class sampling-frame prior
         declarations.
    De-duplicated, ordered by fitpar slot.
    """
    from .selection import match_fitpars

    canonical = resolve_parameter_alias(name)
    targets = set(match_fitpars(pulsar, name, plan.axis_names))  # rule 1

    for ch in charts:  # rule 2
        base = _chart_key_base(canonical, ch.suffix)
        if base == "TASC":
            targets.add(ch.sample_names[2])
        elif base == "EPS1":
            targets.add(ch.sample_names[0])
        elif base == "EPS2":
            targets.add(ch.sample_names[1])

    order = {n: i for i, n in enumerate(plan.axis_names)}
    return tuple(sorted(targets, key=order.__getitem__))


def _chart_key_base(canonical: str, suffix: str) -> str | None:
    """'TASC'|'EPS1'|'EPS2' when ``canonical`` addresses that base for a group
    with this ``suffix`` — either the bare base (applies to every group) or the
    exact suffixed form ``base + suffix``. Else None."""
    for base in ("TASC", "EPS1", "EPS2"):
        if canonical == base or (suffix and canonical == base + suffix):
            return base
    return None
