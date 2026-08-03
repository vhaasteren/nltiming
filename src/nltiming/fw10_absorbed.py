"""``fw10_absorbed`` physical chart: sampling-path conditioning for DDH+STIGMA.

Maps absorbed-gauge chart coordinates
``(A1_ABS, EPS1_ABS, EPS2_ABS, TASC_ABS, H3, STIGMA)`` to engine DDH coordinates
``(A1, ECC, OM, T0, H3, STIGMA)`` via the exact inverse of the Freire & Wex
(2010) absorbed-gauge transfer (dot-free; PB frozen at reference). Sibling of
``kepler_laplace``; see feature contract §10.8.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping

import numpy as np

# Local constants — do not import physical_charts here (activation imports us).
PI = float(np.pi)
DEG2RAD = PI / 180.0
RAD2DEG = 180.0 / PI
TWO_PI = 2.0 * PI
DAY_SEC = 86400.0
ENGINE_SEXTET = ("A1", "ECC", "OM", "T0", "H3", "STIGMA")
CHART_ABS_BASES = ("A1_ABS", "EPS1_ABS", "EPS2_ABS", "TASC_ABS")
# Secular terms that break encode∘decode invertibility (§10.8.2 / §7.6).
FW10_SECULAR_PARAMS = ("PBDOT", "A1DOT", "EDOT", "OMDOT")
_ROUNDTRIP_RTOL = 1e-12
_DEFAULT_EPS_EXCURSION = 1e-10


def _set_slot(vec, i, value):
    """xp-generic single-slot write (jax .at path or in-place numpy)."""
    if hasattr(vec, "at"):
        return vec.at[i].set(value)
    vec[i] = value
    return vec


def mean_motion_rad_s(pb_days: float) -> float:
    """Orbital mean motion ``nb = 2π / (PB_days · 86400)`` [rad/s]."""
    return TWO_PI / (float(pb_days) * DAY_SEC)


def fw10_encode(
    a1: float,
    ecc: float,
    om_rad: float,
    t0: float,
    h3: float,
    stig: float,
    pb_days: float,
) -> tuple[float, float, float, float, float, float]:
    """Engine (intrinsic DDH) → chart (absorbed). Exact inverse of decode."""
    a1 = float(a1)
    ecc = float(ecc)
    om_rad = float(om_rad)
    t0 = float(t0)
    h3 = float(h3)
    stig = float(stig)
    pb_days = float(pb_days)
    nb = mean_motion_rad_s(pb_days)
    e1i = ecc * np.sin(om_rad)
    e2i = ecc * np.cos(om_rad)
    x_p = a1 + 4.0 * h3 / stig**2
    e1p = (a1 * e1i + 4.0 * h3 / stig) / x_p
    e2p = (a1 * e2i + 8.0 * nb * (h3 / stig**2) * a1) / x_p
    tasc = t0 - pb_days * om_rad / TWO_PI - (1.5 * a1 * e1i + h3 / stig) / DAY_SEC
    return float(x_p), float(e1p), float(e2p), float(tasc), h3, stig


def fw10_decode(
    x_p: float,
    e1p: float,
    e2p: float,
    tasc_days: float,
    h3: float,
    stig: float,
    pb_days: float,
    *,
    omega_ref_rad: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Chart (absorbed) → engine (intrinsic DDH).

    Returns ``(A1, ECC, OM_rad, T0, H3, STIGMA)``. When ``omega_ref_rad`` is
    given, ω is unwrapped onto the branch nearest that reference.
    """
    x_p = float(x_p)
    e1p = float(e1p)
    e2p = float(e2p)
    tasc_days = float(tasc_days)
    h3 = float(h3)
    stig = float(stig)
    pb_days = float(pb_days)
    nb = mean_motion_rad_s(pb_days)
    x_i = x_p - 4.0 * h3 / stig**2
    p1 = x_p * e1p - 4.0 * h3 / stig
    p2 = x_p * e2p - 8.0 * nb * (h3 / stig**2) * x_i
    e1i = p1 / x_i
    e2i = p2 / x_i
    ecc = float(np.hypot(e1i, e2i))
    om_rad = float(np.arctan2(e1i, e2i))
    if omega_ref_rad is not None:
        domega = om_rad - float(omega_ref_rad)
        domega = (domega + PI) % TWO_PI - PI
        om_rad = float(omega_ref_rad) + domega
    dt0_s = 1.5 * x_i * e1i + h3 / stig
    t0 = tasc_days + pb_days * om_rad / TWO_PI + dt0_s / DAY_SEC
    return float(x_i), ecc, om_rad, float(t0), h3, stig


def fw10_in_domain(
    x_p: float, e1p: float, e2p: float, h3: float, stig: float, pb_days: float
) -> bool:
    """Decode-time domain: ``0 < ς ≤ 1``, ``4H3/ς² < 0.01 x_p``, ``e_i > 0``."""
    if not (0.0 < float(stig) <= 1.0):
        return False
    if not (float(x_p) > 0.0 and float(h3) >= 0.0):
        return False
    if 4.0 * float(h3) / float(stig) ** 2 >= 0.01 * float(x_p):
        return False
    x_i = float(x_p) - 4.0 * float(h3) / float(stig) ** 2
    if x_i <= 0.0:
        return False
    nb = mean_motion_rad_s(pb_days)
    p1 = float(x_p) * float(e1p) - 4.0 * float(h3) / float(stig)
    p2 = float(x_p) * float(e2p) - 8.0 * nb * (float(h3) / float(stig) ** 2) * x_i
    e_i = float(np.hypot(p1 / x_i, p2 / x_i))
    return e_i > 0.0


def fw10_jacobian(
    x_p: float,
    e1p: float,
    e2p: float,
    tasc_days: float,
    h3: float,
    stig: float,
    pb_days: float,
) -> np.ndarray:
    """Analytic 6×6 J = ∂(A1, ECC, OM_deg, T0, H3, STIGMA)/∂chart.

    Column order: ``(A1_ABS, EPS1_ABS, EPS2_ABS, TASC_ABS, H3, STIGMA)``.
    ``tasc_days`` is unused (affine in TASC) but kept for API symmetry with
    decode. PB is a frozen parameter (not a Jacobian column).
    """
    del tasc_days  # TASC column is the unit vector on the T0 row only
    x_p = float(x_p)
    e1p = float(e1p)
    e2p = float(e2p)
    h3 = float(h3)
    s = float(stig)
    pb = float(pb_days)
    nb = mean_motion_rad_s(pb)
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    s5 = s4 * s

    # Intermediates and their gradients w.r.t. (xp, e1p, e2p, tasc, h3, s)
    xi = x_p - 4.0 * h3 / s2
    d_xi = np.array([1.0, 0.0, 0.0, 0.0, -4.0 / s2, 8.0 * h3 / s3])

    p1 = x_p * e1p - 4.0 * h3 / s
    d_p1 = np.array([e1p, x_p, 0.0, 0.0, -4.0 / s, 4.0 * h3 / s2])

    p2 = x_p * e2p - 8.0 * nb * (h3 / s2) * xi
    d_p2 = np.array(
        [
            e2p - 8.0 * nb * h3 / s2,
            0.0,
            x_p,
            0.0,
            -8.0 * nb * xi / s2 + 32.0 * nb * h3 / s4,
            16.0 * nb * h3 * xi / s3 - 64.0 * nb * h3 * h3 / s5,
        ]
    )

    inv_xi = 1.0 / xi
    inv_xi2 = inv_xi * inv_xi
    e1i = p1 * inv_xi
    e2i = p2 * inv_xi
    d_e1i = d_p1 * inv_xi - p1 * d_xi * inv_xi2
    d_e2i = d_p2 * inv_xi - p2 * d_xi * inv_xi2

    ecc = float(np.hypot(e1i, e2i))
    if ecc == 0.0:
        raise ValueError("fw10_absorbed Jacobian singular at e_i = 0")
    d_ecc = (e1i * d_e1i + e2i * d_e2i) / ecc

    # ∂atan2(e1i, e2i)/∂e1i = e2i/e², ∂/∂e2i = -e1i/e²
    d_om = (e2i * d_e1i - e1i * d_e2i) / (ecc * ecc)
    d_om_deg = RAD2DEG * d_om

    # dt0 = 1.5*p1 + h3/s  (since xi*e1i = p1)
    d_dt0 = 1.5 * d_p1 + np.array([0.0, 0.0, 0.0, 0.0, 1.0 / s, -h3 / s2])

    d_t0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    d_t0 = d_t0 + (pb / TWO_PI) * d_om + d_dt0 / DAY_SEC

    J = np.zeros((6, 6), dtype=float)
    J[0, :] = d_xi  # A1 = xi
    J[1, :] = d_ecc
    J[2, :] = d_om_deg
    J[3, :] = d_t0
    J[4, 4] = 1.0  # H3
    J[5, 5] = 1.0  # STIGMA
    return J


def assert_fw10_roundtrip(
    a1: float,
    ecc: float,
    om_rad: float,
    t0: float,
    h3: float,
    stig: float,
    pb_days: float,
    *,
    rtol: float = _ROUNDTRIP_RTOL,
) -> None:
    """Raise AssertionError unless encode∘decode and decode∘encode are id."""
    chart = fw10_encode(a1, ecc, om_rad, t0, h3, stig, pb_days)
    back = fw10_decode(*chart, pb_days, omega_ref_rad=om_rad)
    eng = (a1, ecc, om_rad, t0, h3, stig)
    for name, a, b in zip(("A1", "ECC", "OM", "T0", "H3", "STIGMA"), eng, back):
        scale = max(1.0, abs(float(a)), abs(float(b)))
        if abs(float(a) - float(b)) > rtol * scale:
            raise AssertionError(f"fw10 encode∘decode failed on {name}: {a!r} vs {b!r}")
    eng2 = fw10_decode(*chart, pb_days, omega_ref_rad=om_rad)
    chart2 = fw10_encode(eng2[0], eng2[1], eng2[2], eng2[3], eng2[4], eng2[5], pb_days)
    for name, a, b in zip(
        ("A1_ABS", "EPS1_ABS", "EPS2_ABS", "TASC_ABS", "H3", "STIGMA"), chart, chart2
    ):
        scale = max(1.0, abs(float(a)), abs(float(b)))
        if abs(float(a) - float(b)) > rtol * scale:
            raise AssertionError(f"fw10 decode∘encode failed on {name}: {a!r} vs {b!r}")


@dataclass(frozen=True)
class FW10AbsorbedChart:
    """Activated ``fw10_absorbed`` group. Slot-preserving six-axis chart."""

    suffix: str
    engine_names: tuple[str, str, str, str, str, str]  # A1,ECC,OM,T0,H3,STIGMA fitpars
    sample_names: tuple[str, str, str, str, str, str]
    slots: tuple[int, int, int, int, int, int]
    # Chart-frame references (absorbed)
    x_p_ref: float
    eps1p_ref: float
    eps2p_ref: float
    tasc_ref: float
    # Engine-frame references (intrinsic)
    a1_ref: float
    e_ref: float
    omega_ref_rad: float
    t0_ref: float
    h3_ref: float
    stigma_ref: float
    pb_ref: float  # days; frozen dependency (never a dependency_slot)
    # Exact-decimal strings for the manifest
    a1_ref_str: str
    ecc_ref_str: str
    om_ref_str: str
    t0_ref_str: str
    h3_ref_str: str
    stigma_ref_str: str
    pb_ref_str: str

    name: ClassVar[str] = "fw10_absorbed"

    def __post_init__(self) -> None:
        assert_fw10_roundtrip(
            self.a1_ref,
            self.e_ref,
            self.omega_ref_rad,
            self.t0_ref,
            self.h3_ref,
            self.stigma_ref,
            self.pb_ref,
        )

    @classmethod
    def from_engine_refs(
        cls,
        *,
        suffix: str,
        engine_names: tuple[str, str, str, str, str, str],
        slots: tuple[int, int, int, int, int, int],
        a1: float,
        ecc: float,
        om_deg: float,
        t0: float,
        h3: float,
        stig: float,
        pb_days: float,
        a1_ref_str: str,
        ecc_ref_str: str,
        om_ref_str: str,
        t0_ref_str: str,
        h3_ref_str: str,
        stigma_ref_str: str,
        pb_ref_str: str,
    ) -> "FW10AbsorbedChart":
        """Build from absolute engine references via encode."""
        om_rad = float(om_deg) * DEG2RAD
        x_p, e1p, e2p, tasc, _, _ = fw10_encode(a1, ecc, om_rad, t0, h3, stig, pb_days)
        sample_names = (
            f"A1_ABS{suffix}",
            f"EPS1_ABS{suffix}",
            f"EPS2_ABS{suffix}",
            f"TASC_ABS{suffix}",
            engine_names[4],  # H3 fitpar (identity axis)
            engine_names[5],  # STIGMA fitpar (identity axis)
        )
        return cls(
            suffix=suffix,
            engine_names=engine_names,
            sample_names=sample_names,
            slots=slots,
            x_p_ref=float(x_p),
            eps1p_ref=float(e1p),
            eps2p_ref=float(e2p),
            tasc_ref=float(tasc),
            a1_ref=float(a1),
            e_ref=float(ecc),
            omega_ref_rad=float(om_rad),
            t0_ref=float(t0),
            h3_ref=float(h3),
            stigma_ref=float(stig),
            pb_ref=float(pb_days),
            a1_ref_str=a1_ref_str,
            ecc_ref_str=ecc_ref_str,
            om_ref_str=om_ref_str,
            t0_ref_str=t0_ref_str,
            h3_ref_str=h3_ref_str,
            stigma_ref_str=stigma_ref_str,
            pb_ref_str=pb_ref_str,
        )

    @property
    def engine_slots(self) -> tuple[int, ...]:
        return self.slots

    @property
    def dependency_slots(self) -> tuple[int, ...]:
        # PB is a frozen decode constant read from pb_ref only (§10.8.2).
        return ()

    @property
    def chart_id(self) -> tuple[str, str]:
        return (self.name, self.suffix)

    def chart_point_from_delta(
        self, vec
    ) -> tuple[float, float, float, float, float, float]:
        s = self.slots
        return (
            self.x_p_ref + float(vec[s[0]]),
            self.eps1p_ref + float(vec[s[1]]),
            self.eps2p_ref + float(vec[s[2]]),
            self.tasc_ref + float(vec[s[3]]),
            self.h3_ref + float(vec[s[4]]),
            self.stigma_ref + float(vec[s[5]]),
        )

    def in_domain(self, vec) -> bool:
        x_p, e1p, e2p, _tasc, h3, stig = self.chart_point_from_delta(vec)
        return fw10_in_domain(x_p, e1p, e2p, h3, stig, self.pb_ref)

    def apply_delta(self, vec, xp):
        """Sampling-frame deltas → engine-frame deltas (xp-generic).

        Out-of-domain proposals yield NaN, not a finite number. This is not a
        runtime *guard* (activation-time certification remains the framework's
        contract) — it is the honest value of a point outside the map's domain.
        The dangerous case is ς < 0: every division stays finite, so the decode
        would otherwise hand the likelihood a plausible-looking but meaningless
        orbit instead of a rejected sample.
        """
        s = self.slots
        x_p = self.x_p_ref + vec[s[0]]
        e1p = self.eps1p_ref + vec[s[1]]
        e2p = self.eps2p_ref + vec[s[2]]
        tasc = self.tasc_ref + vec[s[3]]
        h3 = self.h3_ref + vec[s[4]]
        stig_raw = self.stigma_ref + vec[s[5]]
        in_domain = (stig_raw > 0.0) & (stig_raw <= 1.0)
        # Keep the arithmetic away from stig == 0 so the NaN comes from the
        # explicit mask below, not from an incidental division.
        stig = xp.where(in_domain, stig_raw, 1.0)
        nb = TWO_PI / (self.pb_ref * DAY_SEC)
        x_i = x_p - 4.0 * h3 / (stig * stig)
        p1 = x_p * e1p - 4.0 * h3 / stig
        p2 = x_p * e2p - 8.0 * nb * (h3 / (stig * stig)) * x_i
        e1i = p1 / x_i
        e2i = p2 / x_i
        ecc = xp.sqrt(e1i * e1i + e2i * e2i)
        om_rad = xp.arctan2(e1i, e2i)
        domega = om_rad - self.omega_ref_rad
        domega = (domega + PI) % TWO_PI - PI
        om_rad = self.omega_ref_rad + domega
        dt0_s = 1.5 * x_i * e1i + h3 / stig
        t0 = tasc + self.pb_ref * om_rad / TWO_PI + dt0_s / DAY_SEC
        nan = xp.nan if hasattr(xp, "nan") else float("nan")
        vec = _set_slot(vec, s[0], xp.where(in_domain, x_i - self.a1_ref, nan))
        vec = _set_slot(vec, s[1], xp.where(in_domain, ecc - self.e_ref, nan))
        vec = _set_slot(vec, s[2], xp.where(in_domain, domega * RAD2DEG, nan))
        vec = _set_slot(vec, s[3], xp.where(in_domain, t0 - self.t0_ref, nan))
        vec = _set_slot(vec, s[4], xp.where(in_domain, h3 - self.h3_ref, nan))
        vec = _set_slot(vec, s[5], xp.where(in_domain, stig_raw - self.stigma_ref, nan))
        return vec

    def jacobian_at(
        self,
        d_xp: float = 0.0,
        d_e1p: float = 0.0,
        d_e2p: float = 0.0,
        d_tasc: float = 0.0,
        d_h3: float = 0.0,
        d_stig: float = 0.0,
    ) -> np.ndarray:
        return fw10_jacobian(
            self.x_p_ref + float(d_xp),
            self.eps1p_ref + float(d_e1p),
            self.eps2p_ref + float(d_e2p),
            self.tasc_ref + float(d_tasc),
            self.h3_ref + float(d_h3),
            self.stigma_ref + float(d_stig),
            self.pb_ref,
        )

    def write_frame_block(self, B: np.ndarray, delta_full: np.ndarray) -> None:
        s = self.slots
        J = self.jacobian_at(
            delta_full[s[0]],
            delta_full[s[1]],
            delta_full[s[2]],
            delta_full[s[3]],
            delta_full[s[4]],
            delta_full[s[5]],
        )
        B[np.ix_(list(s), list(s))] = J

    def decode(self, samples, dependency=None):
        """Absolute engine columns from absolute chart-frame sample arrays."""
        del dependency  # PB is frozen at pb_ref
        x_p = np.asarray(samples[self.sample_names[0]], float)
        e1p = np.asarray(samples[self.sample_names[1]], float)
        e2p = np.asarray(samples[self.sample_names[2]], float)
        tasc = np.asarray(samples[self.sample_names[3]], float)
        h3 = np.asarray(samples[self.sample_names[4]], float)
        stig = np.asarray(samples[self.sample_names[5]], float)
        # Broadcast-safe elementwise decode on the reference ω branch.
        x_p_f = np.atleast_1d(x_p)
        out_a1 = np.empty_like(x_p_f, dtype=float)
        out_ecc = np.empty_like(x_p_f, dtype=float)
        out_om = np.empty_like(x_p_f, dtype=float)
        out_t0 = np.empty_like(x_p_f, dtype=float)
        out_h3 = np.empty_like(x_p_f, dtype=float)
        out_stig = np.empty_like(x_p_f, dtype=float)
        for i in range(x_p_f.size):
            a1, ecc, om_rad, t0, h3_i, stig_i = fw10_decode(
                float(np.atleast_1d(x_p)[i]),
                float(np.atleast_1d(e1p)[i]),
                float(np.atleast_1d(e2p)[i]),
                float(np.atleast_1d(tasc)[i]),
                float(np.atleast_1d(h3)[i]),
                float(np.atleast_1d(stig)[i]),
                self.pb_ref,
                omega_ref_rad=self.omega_ref_rad,
            )
            out_a1[i] = a1
            out_ecc[i] = ecc
            out_om[i] = om_rad * RAD2DEG
            out_t0[i] = t0
            out_h3[i] = h3_i
            out_stig[i] = stig_i
        if np.ndim(x_p) == 0:
            return {
                self.engine_names[0]: float(out_a1[0]),
                self.engine_names[1]: float(out_ecc[0]),
                self.engine_names[2]: float(out_om[0]),
                self.engine_names[3]: float(out_t0[0]),
                self.engine_names[4]: float(out_h3[0]),
                self.engine_names[5]: float(out_stig[0]),
            }
        return {
            self.engine_names[0]: out_a1,
            self.engine_names[1]: out_ecc,
            self.engine_names[2]: out_om,
            self.engine_names[3]: out_t0,
            self.engine_names[4]: out_h3,
            self.engine_names[5]: out_stig,
        }

    def sampling_reference_strings(self) -> dict[str, str]:
        return {
            self.sample_names[0]: repr(self.x_p_ref),
            self.sample_names[1]: repr(self.eps1p_ref),
            self.sample_names[2]: repr(self.eps2p_ref),
            self.sample_names[3]: repr(self.tasc_ref),
            # H3/STIGMA keep engine names; their refs already live in engine_refs.
        }

    def log_abs_det_jacobian_at_ref(self) -> float:
        sign, logabsdet = np.linalg.slogdet(self.jacobian_at())
        if sign == 0.0:
            return float(-np.inf)
        return float(logabsdet)

    def record(
        self,
        *,
        enabled: bool,
        reason: str | None,
        dispositions: Mapping[str, str] | None = None,
    ) -> dict:
        return {
            "suffix": self.suffix,
            "enabled": bool(enabled),
            "reason": reason,
            "family": self.name,
            "gauge": "absorbed",
            "engine_names": list(self.engine_names),
            "chart_names": list(self.sample_names),
            "sample_names": list(self.sample_names) if enabled else None,
            "dispositions": dict(dispositions) if dispositions else None,
            "h3_ref": float(self.h3_ref),
            "stigma_ref": float(self.stigma_ref),
            "pb_ref": float(self.pb_ref),
            "e_ref": float(self.e_ref),
            "theta_ref_engine": {
                "A1": self.a1_ref_str,
                "ECC": self.ecc_ref_str,
                "OM": self.om_ref_str,
                # The reference-local omega branch the decode actually uses.
                # `om_ref_str` is the raw par string, which may sit outside
                # [0, 360); a decoder keying off it would unwrap onto a branch
                # 2*pi away and shift T0 by a whole PB. Mirrors the
                # kepler_laplace record's OM_normalized.
                "OM_normalized": repr(float(self.omega_ref_rad * RAD2DEG)),
                "T0": self.t0_ref_str,
                "H3": self.h3_ref_str,
                "STIGMA": self.stigma_ref_str,
            },
            "theta_ref_sample": (
                {
                    "A1_ABS": repr(self.x_p_ref),
                    "EPS1_ABS": repr(self.eps1p_ref),
                    "EPS2_ABS": repr(self.eps2p_ref),
                    "TASC_ABS": repr(self.tasc_ref),
                    "H3": self.h3_ref_str,
                    "STIGMA": self.stigma_ref_str,
                }
                if enabled
                else None
            ),
            "log_abs_det_jacobian_at_ref": (
                self.log_abs_det_jacobian_at_ref() if enabled else None
            ),
            "active": bool(enabled),
        }
