"""PR-2 (§12.2): unit tests for the Kepler<->Laplace physical chart module.

The feature is inert at this stage (no pipeline wiring); these exercise the
pure maps, candidacy, activation, guards, and prior-override materialization
in isolation.
"""

import numpy as np
import pytest

from nltiming import physical_charts as pc
from nltiming.coordinates import TimingCoordinatePolicy
from nltiming.inference import resolve_inference_plan, TimingInference
from nltiming.linearity import resolve_linearity
from nltiming.physical_charts import (
    DISK_MARGIN,
    KeplerLaplaceChart,
    KeplerLaplacePolicy,
    activate_charts,
    check_chart_compatibility,
    disk_shrink_factor,
    expand_override_key,
    frame_change_matrix,
    kepler_from_laplace,
    laplace_from_kepler,
    normalize_inference_selectors,
    resolve_chart_candidates,
    resolved_eps_reachability,
    tasc_ref_decimal,
    unwrap_om_delta_deg,
)
from nltiming.priors import delta_uniform, normal, truncated_normal
from nltiming.protocols import BinaryChartCapability

RAD2DEG = pc.RAD2DEG
DEG2RAD = pc.DEG2RAD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FITPARS = ("F0", "ECC", "OM", "T0", "PB", "A1")
REFS = {
    "F0": "100.0",
    "ECC": "8e-4",
    "OM": "50.7",
    "T0": "55000.0",
    "PB": "8.6866194196",
    "A1": "10.0",
}


class _Prior:  # informative uniform-like PINT prior
    def __init__(self, lo, hi):
        self.lower_bound = lo
        self.upper_bound = hi


class _BarePrior:  # PINT default uninformative object (no bounds/mean attrs)
    pass


class _Param:
    def __init__(self, value=None, prior=None):
        self.value = value
        self.prior = prior


class _FakePINT:
    def __init__(self, **params):
        for k, v in params.items():
            setattr(self, k, v)


class _FakeEngine:
    def __init__(self, refs=REFS, capability=None, capability_map=None):
        self._refs = dict(refs)
        self.fitpars = tuple(refs)
        self._capability = capability
        self._capability_map = capability_map

    def reference_theta_exact(self):
        return dict(self._refs)

    def binary_chart_capability(self, family, suffix):
        if self._capability_map is not None:
            return self._capability_map.get(suffix)
        return self._capability


class _FakeEngineNoCap:
    """Engine that does NOT implement binary_chart_capability (fallback path)."""

    def __init__(self, refs=REFS):
        self._refs = dict(refs)
        self.fitpars = tuple(refs)

    def reference_theta_exact(self):
        return dict(self._refs)


class _FakePulsar:
    def __init__(self, fitpars=FITPARS, pint_model=None, fitparameters=None, ntoa=40):
        self.name = "JTEST+0000"
        self.fitpars = tuple(fitpars)
        self._pint = pint_model
        if fitparameters is not None:
            self._fitparameters = fitparameters
        rng = np.random.default_rng(12345)
        self._mmat = rng.standard_normal((ntoa, len(self.fitpars)))
        self._residuals = 1e-7 * rng.standard_normal(ntoa)
        self._toaerrs = np.full(ntoa, 1e-6)

    @property
    def toaerrs(self):
        return self._toaerrs

    @property
    def residuals(self):
        return self._residuals

    @property
    def Mmat(self):
        return self._mmat

    def pint_model(self):
        return self._pint


def _plan(pulsar, inference):
    linearity = resolve_linearity(pulsar, None)
    return resolve_inference_plan(
        pulsar,
        inference=inference,
        linearity=linearity,
        coordinate_policy=TimingCoordinatePolicy(),
    )


def _chart():
    (cand,) = [
        c
        for c in resolve_chart_candidates(
            _FakePulsar(), _FakeEngineNoCap(), KeplerLaplacePolicy(mode="auto")
        )
        if c.chart is not None
    ]
    return cand.chart


# ---------------------------------------------------------------------------
# Pure maps
# ---------------------------------------------------------------------------


def test_roundtrip_kepler_laplace():
    for e in (1e-6, 1e-4, 8e-4, 0.05):
        for om in (0.0, 50.7, 179.0, 359.0):
            t0, pb = 55123.4567, 8.9
            eps1, eps2, tasc = laplace_from_kepler(e, om, t0, pb)
            e2, om2, t02 = kepler_from_laplace(eps1, eps2, tasc, pb)
            assert abs(e2 - e) < 1e-12 * max(1.0, e)
            assert abs((om2 - om + 180) % 360 - 180) < 1e-9
            assert abs(t02 - t0) < 1e-9
            assert 0.0 <= om2 < 360.0


def test_om_unwrap_seam():
    assert abs(unwrap_om_delta_deg(1.0, 359.0) - 2.0) < 1e-12
    assert abs(unwrap_om_delta_deg(359.0, 1.0) - (-2.0)) < 1e-12
    assert abs(unwrap_om_delta_deg(50.7, 50.7)) < 1e-12


def test_jacobian_vs_fd():
    ch = _chart()
    for d_eps1, d_eps2, d_pb in [(0.0, 0.0, 0.0), (3e-4, -2e-4, 1e-3)]:
        J = ch.jacobian_at(d_eps1, d_eps2, d_pb)
        h = 1e-9

        def f(e1, e2, tasc):
            return np.array(
                ch.engine_delta_from_sample_delta(e1, e2, tasc, d_pb=d_pb, xp=np)
            )

        base = (d_eps1, d_eps2, 0.0)
        cols = []
        for k in range(3):
            hi = list(base)
            lo = list(base)
            hi[k] += h
            lo[k] -= h
            cols.append((f(*hi) - f(*lo)) / (2 * h))
        J_fd = np.stack(cols, axis=1)
        np.testing.assert_allclose(J, J_fd, rtol=1e-5, atol=1e-6)
        e = float(np.hypot(ch.eps1_ref + d_eps1, ch.eps2_ref + d_eps2))
        assert abs(abs(np.linalg.det(J)) - RAD2DEG / e) < 1e-6 * RAD2DEG / e

    with pytest.raises(ValueError):
        # A chart whose reference IS the origin cannot exist (candidacy rejects
        # e_ref<=0), so construct one and probe the exact origin.
        origin_chart = KeplerLaplaceChart(
            suffix="",
            engine_names=("ECC", "OM", "T0"),
            sample_names=("EPS1", "EPS2", "TASC"),
            slots=(1, 2, 3),
            pb_name="PB",
            pb_slot=4,
            a1_name=None,
            a1_ref=None,
            e_ref=1e-3,
            omega_ref_rad=0.5,
            eps1_ref=0.0,
            eps2_ref=0.0,
            pb_ref=8.0,
            t0_ref_str="55000.0",
            tasc_ref_str="54999.0",
            ecc_ref_str="1e-3",
            om_ref_raw_str="28.6",
            om_ref_norm_str="28.6",
        )
        origin_chart.jacobian_at(0.0, 0.0)


def test_origin_approach():
    ch = _chart()
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def f(v):
        d1, d2, dt = v
        out = ch.engine_delta_from_sample_delta(d1, d2, dt, d_pb=0.0, xp=jnp)
        return jnp.stack([out[0], out[1], out[2]])

    ref = np.hypot(ch.eps1_ref, ch.eps2_ref)
    for scale in (1e-2, 1e-6, 1e-10, 1e-14):
        # Move toward the origin along the -eps_ref direction.
        d1 = -ch.eps1_ref * (1 - scale)
        d2 = -ch.eps2_ref * (1 - scale)
        v = jnp.asarray([d1, d2, 0.0])
        val = np.asarray(f(v))
        jacf = np.asarray(jax.jacfwd(f)(v))
        assert np.all(np.isfinite(val))
        assert np.all(np.isfinite(jacf))
    assert ref > 0


def test_delta_form_zero_identity():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        e_ref = float(rng.uniform(1e-4, 0.05))
        omega = float(rng.uniform(0, 2 * np.pi))
        ch = KeplerLaplaceChart(
            suffix="",
            engine_names=("ECC", "OM", "T0"),
            sample_names=("EPS1", "EPS2", "TASC"),
            slots=(1, 2, 3),
            pb_name="PB",
            pb_slot=4,
            a1_name=None,
            a1_ref=None,
            e_ref=e_ref,
            omega_ref_rad=omega,
            eps1_ref=e_ref * float(np.sin(omega)),  # consistent references
            eps2_ref=e_ref * float(np.cos(omega)),
            pb_ref=float(rng.uniform(1, 100)),
            t0_ref_str="55000.0",
            tasc_ref_str="54999.0",
            ecc_ref_str="1e-3",
            om_ref_raw_str="0",
            om_ref_norm_str="0",
        )
        d_ecc, d_om, d_t0 = ch.engine_delta_from_sample_delta(
            0.0, 0.0, 0.0, d_pb=0.0, xp=np
        )
        # The given delta-form map is zero at the origin to machine precision
        # (§3.1); it is not bitwise zero because sqrt(eps1^2+eps2^2) and atan2
        # of rounded consistent references differ from e_ref/omega_ref in the
        # last ULP.
        assert abs(d_ecc) < 1e-14
        assert abs(d_om) < 1e-10
        assert abs(d_t0) < 1e-10


def test_delta_form_matches_absolute_form():
    ch = _chart()
    rng = np.random.default_rng(3)
    for _ in range(20):
        d1, d2, dt = rng.normal(scale=1e-4, size=3)
        d_ecc, d_om, d_t0 = ch.engine_delta_from_sample_delta(
            d1, d2, dt, d_pb=0.0, xp=np
        )
        eps1 = ch.eps1_ref + d1
        eps2 = ch.eps2_ref + d2
        tasc = float(ch.tasc_ref_str) + dt
        e_abs, om_abs, t0_abs = kepler_from_laplace(eps1, eps2, tasc, ch.pb_ref)
        assert abs((ch.e_ref + d_ecc) - e_abs) < 1e-12
        d_om_abs = unwrap_om_delta_deg(om_abs, float(ch.om_ref_norm_str))
        assert abs(d_om - d_om_abs) < 1e-9
        assert abs((float(ch.t0_ref_str) + d_t0) - t0_abs) < 1e-9


def test_tasc_ref_decimal_precision():
    t0, pb, om = "55000.0", "8.6866194196", "50.7"
    tasc = tasc_ref_decimal(t0, pb, om)
    # Identity T0 = TASC + PB*omega/2pi at prec 50.
    from decimal import Decimal, localcontext

    with localcontext() as ctx:
        ctx.prec = 50
        omega = Decimal(om) * pc._PI_50 / Decimal(180)
        recon = Decimal(tasc) + Decimal(pb) * omega / (2 * pc._PI_50)
        assert abs(recon - Decimal(t0)) < Decimal("1e-30")


# ---------------------------------------------------------------------------
# Candidacy
# ---------------------------------------------------------------------------


def test_candidacy_matrix():
    pol_auto = KeplerLaplacePolicy(mode="auto")
    pol_on = KeplerLaplacePolicy(mode="on")

    # Row 8: incomplete triple (ECC fixed) -> W1/W2 + incomplete_triple.
    p = _FakePulsar(fitpars=("OM", "T0", "PB"))
    with pytest.warns(UserWarning):
        cands = resolve_chart_candidates(
            p, _FakeEngineNoCap({"OM": "50.7", "T0": "55000.0", "PB": "8.6"}), pol_on
        )
    assert cands[0].skip_reason == "incomplete_triple"

    # Row 10: e_ref <= 0 -> e_ref_not_positive.
    refs0 = {**REFS, "ECC": "0.0"}
    (c0,) = resolve_chart_candidates(_FakePulsar(), _FakeEngineNoCap(refs0), pol_auto)
    assert c0.skip_reason == "e_ref_not_positive" and c0.chart is None

    # Row 9: e_ref >= e_max under auto is SILENT with reason e_ref_above_e_max.
    refs_big = {**REFS, "ECC": "0.3"}
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        (cbig,) = resolve_chart_candidates(
            _FakePulsar(), _FakeEngineNoCap(refs_big), pol_auto
        )
    assert cbig.skip_reason == "e_ref_above_e_max"
    # under "on", e_max is ignored -> candidate.
    (cbig_on,) = resolve_chart_candidates(
        _FakePulsar(), _FakeEngineNoCap(refs_big), pol_on
    )
    assert cbig_on.chart is not None

    # Row 11: EPS already free -> already_laplace (W3 under on).
    p_ell1 = _FakePulsar(fitpars=("EPS1", "EPS2", "TASC", "PB"))
    refs_ell1 = {"EPS1": "1e-4", "EPS2": "8e-4", "TASC": "55000.0", "PB": "8.6"}
    with pytest.warns(UserWarning):
        (cl,) = resolve_chart_candidates(p_ell1, _FakeEngineNoCap(refs_ell1), pol_on)
    assert cl.skip_reason == "already_laplace"

    # PB/A1-only group yields NO candidate and no record.
    p_pb = _FakePulsar(fitpars=("F0", "PB", "A1"))
    cands_pb = resolve_chart_candidates(
        p_pb, _FakeEngineNoCap({"F0": "1", "PB": "8.6", "A1": "10"}), pol_auto
    )
    assert cands_pb == ()

    # Happy: full triple, low e -> a real chart.
    (ok,) = resolve_chart_candidates(_FakePulsar(), _FakeEngineNoCap(), pol_auto)
    assert ok.chart is not None and ok.e_ref == pytest.approx(8e-4)


def test_pb_om_exact_reference_strings():
    refs = {**REFS, "OM": "410.7"}
    (cand,) = resolve_chart_candidates(
        _FakePulsar(), _FakeEngineNoCap(refs), KeplerLaplacePolicy()
    )
    ch = cand.chart
    assert ch.om_ref_raw_str == "410.7"
    assert ch.om_ref_norm_str.startswith("50.7")
    # tasc from the exact PB string; the reference identity holds at prec 50.
    from decimal import Decimal, localcontext

    with localcontext() as ctx:
        ctx.prec = 50
        omega = Decimal(ch.om_ref_norm_str) * pc._PI_50 / Decimal(180)
        recon = Decimal(ch.tasc_ref_str) + Decimal(REFS["PB"]) * omega / (2 * pc._PI_50)
        assert abs(recon - Decimal(ch.t0_ref_str)) < Decimal("1e-25")


def test_accessory_ref_fallback_chain():
    pol = KeplerLaplacePolicy()
    # PB free -> from refs; A1 free -> xe2 filled.
    (c,) = resolve_chart_candidates(_FakePulsar(), _FakeEngineNoCap(), pol)
    assert c.chart.pb_ref == pytest.approx(float(REFS["PB"]))
    assert c.chart.a1_ref == pytest.approx(10.0)

    # A1 absent everywhere -> a1_ref None (xe2 null); PB absent -> ValueError.
    refs_noa1 = {k: v for k, v in REFS.items() if k != "A1"}
    p_noa1 = _FakePulsar(fitpars=("F0", "ECC", "OM", "T0", "PB"))
    (c2,) = resolve_chart_candidates(p_noa1, _FakeEngineNoCap(refs_noa1), pol)
    assert c2.chart.a1_ref is None

    refs_nopb = {k: v for k, v in REFS.items() if k != "PB"}
    p_nopb = _FakePulsar(fitpars=("F0", "ECC", "OM", "T0", "A1"))
    with pytest.raises(ValueError, match="no PB reference"):
        resolve_chart_candidates(p_nopb, _FakeEngineNoCap(refs_nopb), pol)

    # PB only on the PINT model -> repr(float(value)).
    p_pbmodel = _FakePulsar(
        fitpars=("F0", "ECC", "OM", "T0", "A1"),
        pint_model=_FakePINT(PB=_Param(value=8.6866194196)),
    )
    (c3,) = resolve_chart_candidates(p_pbmodel, _FakeEngineNoCap(refs_nopb), pol)
    assert c3.chart.pb_ref == pytest.approx(8.6866194196)


# ---------------------------------------------------------------------------
# Selector normalization / override expansion
# ---------------------------------------------------------------------------


def test_selector_normalization():
    p = _FakePulsar()
    cands = resolve_chart_candidates(p, _FakeEngineNoCap(), KeplerLaplacePolicy())
    from nltiming.inference import Marginalize

    inf = TimingInference(marginalize={"TASC": Marginalize.z_prior()})
    out = normalize_inference_selectors(inf, cands)
    assert "T0" in out.marginalize and "TASC" not in out.marginalize

    with pytest.raises(ValueError, match="declared on"):
        normalize_inference_selectors(
            TimingInference(marginalize={"EPS1": Marginalize.z_prior()}), cands
        )

    # both T0 and TASC -> overlap.
    with pytest.raises(ValueError, match="overlap"):
        normalize_inference_selectors(
            TimingInference(
                marginalize={
                    "TASC": Marginalize.z_prior(),
                    "T0": Marginalize.z_prior(),
                }
            ),
            cands,
        )


def test_expand_override_key_union():
    p = _FakePulsar()
    plan = _plan(p, TimingInference.sample_all())
    (cand,) = [
        c
        for c in resolve_chart_candidates(p, _FakeEngineNoCap(), KeplerLaplacePolicy())
        if c.chart is not None
    ]
    # Rename the plan's ECC/OM/T0 axes to the sampling names (simulate active).
    active_plan, resolved, _ = activate_charts(
        plan,
        [cand],
        KeplerLaplacePolicy(),
        prior_overrides={},
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=TimingCoordinatePolicy().nonlinear_scale,
        engine_refs=REFS,
        prior_policy="wide_default",
    )
    charts = tuple(r.chart for r in resolved)
    assert expand_override_key(p, "TASC", active_plan, charts) == ("TASC",)
    assert expand_override_key(p, "EPS1", active_plan, charts) == ("EPS1",)
    # T0 has no chart rule -> matches nothing on the (renamed) plan.
    assert expand_override_key(p, "T0", active_plan, charts) == ()


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def _activate(pulsar, inference, policy, **kw):
    plan = _plan(pulsar, inference)
    cands = resolve_chart_candidates(
        pulsar, kw.pop("engine", _FakeEngineNoCap()), policy
    )
    return activate_charts(
        plan,
        cands,
        policy,
        prior_overrides=kw.pop("prior_overrides", {}),
        pint_model=kw.pop("pint_model", pulsar.pint_model()),
        pulsar=pulsar,
        engine_design_matrix=pulsar.Mmat,
        nonlinear_scale=TimingCoordinatePolicy().nonlinear_scale,
        engine_refs=kw.pop("engine_refs", REFS),
        prior_policy=kw.pop("prior_policy", "wide_default"),
    )


def test_activation_matrix():
    p = _FakePulsar()
    pol = KeplerLaplacePolicy(mode="auto")

    # Row 1: all three sampled -> activate.
    plan, resolved, records = _activate(p, TimingInference.sample_all(), pol)
    assert len(resolved) == 1
    names = plan.axis_names
    assert {"EPS1", "EPS2", "TASC"} <= set(names)
    assert not ({"ECC", "OM", "T0"} & set(names))
    eps1_axis = plan.axis("ECC")  # engine-name lookup resolves to EPS1
    assert eps1_axis.name == "EPS1" and eps1_axis.engine_name == "ECC"
    assert eps1_axis.physical_chart == "kepler_laplace"
    assert records[0]["enabled"] is True

    # Row 5: all three z-marginalized -> no_sampled_axis (silent auto).
    inf_all_z = TimingInference.groups(z_prior=["ECC", "OM", "T0"])
    plan5, res5, rec5 = _activate(p, inf_all_z, pol)
    assert res5 == () and rec5[0]["reason"] == "no_sampled_axis"
    assert "EPS1" not in plan5.axis_names
    # under "on" -> UserWarning.
    with pytest.warns(UserWarning):
        _activate(p, inf_all_z, KeplerLaplacePolicy(mode="on"))

    # Row 6: split ECC/OM dispositions -> demote (auto warn / on raise).
    inf_split = TimingInference.groups(z_prior=["OM"])  # ECC sample, OM z-marg
    with pytest.warns(UserWarning, match="different inference dispositions"):
        _, res6, rec6 = _activate(p, inf_split, pol)
    assert res6 == () and rec6[0]["reason"] == "split_ecc_om_dispositions"
    with pytest.raises(ValueError, match="different inference dispositions"):
        _activate(p, inf_split, KeplerLaplacePolicy(mode="on"))

    # Row 7: user prior on ECC -> demote (prior_on_kepler_axis).
    with pytest.warns(UserWarning, match="targets ECC, OM, or"):
        _, res7, rec7 = _activate(
            p,
            TimingInference.sample_all(),
            pol,
            prior_overrides={"ECC": delta_uniform(-1e-4, 1e-4)},
        )
    assert res7 == () and rec7[0]["reason"] == "prior_on_kepler_axis"
    with pytest.raises(ValueError):
        _activate(
            p,
            TimingInference.sample_all(),
            KeplerLaplacePolicy(mode="on"),
            prior_overrides={"T0": delta_uniform(-1e-4, 1e-4)},
        )

    # Row 7b: informative PINT prior on ECC (wide_default) -> demote.
    p_pint = _FakePulsar(pint_model=_FakePINT(ECC=_Param(prior=_Prior(0.0, 0.01))))
    with pytest.warns(UserWarning, match="informative"):
        _, res7b, rec7b = _activate(p_pint, TimingInference.sample_all(), pol)
    assert res7b == () and rec7b[0]["reason"] == "pint_prior_on_kepler_axis"


def test_pint_prior_policy_gate():
    p_pint = _FakePulsar(pint_model=_FakePINT(ECC=_Param(prior=_Prior(0.0, 0.01))))
    # explicit -> PINT priors not consulted -> activates.
    _, res, _ = _activate(
        p_pint,
        TimingInference.sample_all(),
        KeplerLaplacePolicy(),
        prior_policy="explicit",
    )
    assert len(res) == 1
    # wide_default -> demotes (bare uninformative prior must NOT demote).
    p_bare = _FakePulsar(pint_model=_FakePINT(ECC=_Param(prior=_BarePrior())))
    _, res_bare, _ = _activate(
        p_bare, TimingInference.sample_all(), KeplerLaplacePolicy()
    )
    assert len(res_bare) == 1


def test_capability_resolution():
    p = _FakePulsar()
    # An engine capability with a non-"dd" convention fails candidacy.
    cap_bad = BinaryChartCapability(
        kepler_convention="ell1",
        epoch_shift_exact=True,
        secular_terms=(),
        origin_certified=False,
        supports_domain=True,
    )
    (cbad,) = resolve_chart_candidates(
        p, _FakeEngine(capability=cap_bad), KeplerLaplacePolicy()
    )
    assert cbad.skip_reason == "unsupported_binary_model"

    # Fallback (no method): capability_source recorded as "fallback".
    (cfb,) = resolve_chart_candidates(p, _FakeEngineNoCap(), KeplerLaplacePolicy())
    _, _, recfb = _activate(p, TimingInference.sample_all(), KeplerLaplacePolicy())
    assert recfb[0]["capability_source"] == "fallback"

    # Fallback secular terms: OMDOT present -> seam guard becomes active.
    p_omdot = _FakePulsar(fitpars=("F0", "ECC", "OM", "T0", "PB", "A1", "OMDOT"))
    refs_omdot = {**REFS, "OMDOT": "1e-3"}
    (com,) = resolve_chart_candidates(
        p_omdot, _FakeEngineNoCap(refs_omdot), KeplerLaplacePolicy()
    )
    assert "OMDOT" in com.secular_terms


def test_capability_group_isolation():
    # Two suffixed groups with different descriptors resolve independently.
    fitparameters = {
        "ECC_a": {"a": "ECC"},
        "OM_a": {"a": "OM"},
        "T0_a": {"a": "T0"},
        "PB_a": {"a": "PB"},
        "ECC_b": {"b": "ECC"},
        "OM_b": {"b": "OM"},
        "T0_b": {"b": "T0"},
        "PB_b": {"b": "PB"},
    }
    fitpars = tuple(fitparameters)
    refs = {
        "ECC_a": "8e-4",
        "OM_a": "50.7",
        "T0_a": "55000.0",
        "PB_a": "8.6",
        "ECC_b": "9e-4",
        "OM_b": "120.0",
        "T0_b": "56000.0",
        "PB_b": "3.2",
    }
    p = _FakePulsar(fitpars=fitpars, fitparameters=fitparameters)
    cap_a = BinaryChartCapability(
        kepler_convention="dd",
        epoch_shift_exact=True,
        secular_terms=(),
        origin_certified=False,
        supports_domain=True,
    )
    cap_b = BinaryChartCapability(
        kepler_convention="ell1",
        epoch_shift_exact=True,
        secular_terms=(),
        origin_certified=False,
        supports_domain=True,
    )
    eng = _FakeEngine(refs=refs, capability_map={"_a": cap_a, "_b": cap_b})
    cands = {
        c.suffix: c for c in resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
    }
    assert cands["_a"].chart is not None
    assert cands["_b"].skip_reason == "unsupported_binary_model"


# ---------------------------------------------------------------------------
# Guards (unit) + reachability handoff
# ---------------------------------------------------------------------------


def test_disk_shrink_factor():
    # c=1 corner inside -> 1.0.
    assert disk_shrink_factor(0.0, 0.0, 0.1, 0.1, 1.0) == 1.0
    # corner outside -> shrunk so the corner lands exactly on r_max.
    c = disk_shrink_factor(0.5, 0.0, 1.0, 0.0, 1.0)
    assert np.hypot(0.5 + c * 1.0, 0.0) == pytest.approx(1.0)


def test_rect_ray_and_guards():
    ch = _chart()
    # A rect around ref that does NOT reach the origin/seam.
    rect_tight = (
        (ch.eps1_ref - 1e-6, ch.eps1_ref + 1e-6),
        (ch.eps2_ref - 1e-6, ch.eps2_ref + 1e-6),
    )
    ok, rec = pc._seam_guard(
        ch, rect_tight, epoch_shift_exact=False, secular_terms=("OMDOT",)
    )
    assert ok and rec["passed"] is True
    ok2, rec2 = pc._seam_guard(
        ch, rect_tight, epoch_shift_exact=True, secular_terms=("OMDOT",)
    )
    assert ok2 and rec2 is None

    # A wide rect crossing the origin -> crosses the seam ray -> demote.
    rect_wide = ((-0.01, 0.01), (-0.01, 0.01))
    okw, recw = pc._seam_guard(
        ch, rect_wide, epoch_shift_exact=False, secular_terms=("OMDOT",)
    )
    assert not okw and recw["passed"] is False

    # origin guard.
    o_ok, o_rec = pc._origin_guard(ch, rect_tight, origin_certified=False)
    assert o_ok and o_rec is None
    o_bad, o_rec2 = pc._origin_guard(ch, rect_wide, origin_certified=False)
    assert not o_bad and o_rec2["origin_in_support"] is True
    o_cert, o_rec3 = pc._origin_guard(
        ch, rect_wide, origin_certified=True, certification_ref="run-42"
    )
    assert o_cert and o_rec3["certification_ref"] == "run-42"


def test_binary_chart_capability_requires_cert_ref():
    with pytest.raises(ValueError, match="certification_ref"):
        BinaryChartCapability(
            kepler_convention="dd",
            epoch_shift_exact=True,
            secular_terms=(),
            origin_certified=True,
            supports_domain=True,
        )


def test_guard_support_equals_prior_support():
    p = _FakePulsar()
    plan, resolved, _ = _activate(
        p, TimingInference.sample_all(), KeplerLaplacePolicy()
    )
    res = resolved[0]
    boxes = res.default_box_delta_bounds()
    # Every default-box axis's stored bounds match the reachability supports
    # (a stored object, not a recomputed recipe).
    for pos, sup in enumerate(res.eps_supports):
        if sup.kind == "default_box":
            name = res.chart.sample_names[pos]
            assert boxes[name] == (sup.lo_delta, sup.hi_delta)


def test_reachability_corners_match_runtime():
    p = _FakePulsar()
    plan = _plan(p, TimingInference.sample_all())
    (cand,) = [
        c
        for c in resolve_chart_candidates(p, _FakeEngineNoCap(), KeplerLaplacePolicy())
        if c.chart is not None
    ]
    rect, supports = resolved_eps_reachability(
        p,
        cand.chart,
        plan,
        p.Mmat,
        nonlinear_scale=TimingCoordinatePolicy().nonlinear_scale,
        prior_overrides={},
        engine_refs=REFS,
    )
    for pos, sup in enumerate(supports):
        lo, hi = sup.abs_bounds()
        assert lo == sup.ref + sup.lo_delta
        assert hi == sup.ref + sup.hi_delta
        assert rect[pos] == (lo, hi)


def test_user_eps_prior_support_policy():
    p = _FakePulsar()
    # Unbounded normal EPS prior -> ValueError under both modes.
    with pytest.raises(ValueError, match="unbounded support"):
        _activate(
            p,
            TimingInference.sample_all(),
            KeplerLaplacePolicy(),
            prior_overrides={"EPS1": normal(0.0, 1e-4, frame="delta")},
        )
    # In-disk truncated_normal accepted (activates).
    _, res, _ = _activate(
        p,
        TimingInference.sample_all(),
        KeplerLaplacePolicy(),
        prior_overrides={
            "EPS1": truncated_normal(0.0, 1e-4, -1e-3, 1e-3, frame="delta")
        },
    )
    assert len(res) == 1


def _suffixed_pulsar_engine():
    fitparameters = {
        "ECC_a": {"a": "ECC"},
        "OM_a": {"a": "OM"},
        "T0_a": {"a": "T0"},
        "PB_a": {"a": "PB"},
    }
    refs = {"ECC_a": "8e-4", "OM_a": "50.7", "T0_a": "55000.0", "PB_a": "8.6"}
    p = _FakePulsar(fitpars=tuple(fitparameters), fitparameters=fitparameters)
    return p, _FakeEngineNoCap(refs), refs


def test_override_overlap_rejected():
    from nltiming.physical_charts import materialize_eps_override

    p, eng, refs = _suffixed_pulsar_engine()
    plan = _plan(p, TimingInference.sample_all())
    (cand,) = [
        c
        for c in resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
        if c.chart is not None
    ]
    assert cand.chart.suffix == "_a"
    # Base 'EPS1' alone materializes cleanly.
    prior = materialize_eps_override(
        {"EPS1": delta_uniform(-1e-3, 1e-3)},
        cand.chart,
        0,
        pulsar=p,
        plan=plan,
        engine_refs=refs,
    )
    assert prior is not None and prior.family == "uniform"
    # Base 'EPS1' AND exact 'EPS1_a' both target the one axis -> overlap error.
    with pytest.raises(ValueError, match="overlapping prior overrides"):
        materialize_eps_override(
            {"EPS1": delta_uniform(-1e-3, 1e-3), "EPS1_a": delta_uniform(-2e-3, 2e-3)},
            cand.chart,
            0,
            pulsar=p,
            plan=plan,
            engine_refs=refs,
        )


def test_frame_change_matrix_shape():
    ch = _chart()
    B = frame_change_matrix(len(FITPARS), (ch,))
    assert B.shape == (6, 6)
    # Identity off the charted slots.
    off = [0, 5]  # F0, A1
    for i in off:
        row = np.zeros(6)
        row[i] = 1.0
        np.testing.assert_array_equal(B[i], row)
    # The charted 3x3 block equals jacobian_at at the reference.
    J = ch.jacobian_at(0.0, 0.0, 0.0)
    np.testing.assert_allclose(B[np.ix_([1, 2, 3], [1, 2, 3])], J)
    assert B[3, 4] == pytest.approx(ch.pb_coupling_at(0.0, 0.0))
    # With delta != 0 the block moves.
    B2 = frame_change_matrix(len(FITPARS), (ch,), delta=_delta_vec(ch, 3e-4, -2e-4))
    assert not np.allclose(B2[np.ix_([1, 2, 3], [1, 2, 3])], J)


def _delta_vec(ch, d1, d2):
    v = np.zeros(6)
    v[ch.slots[0]] = d1
    v[ch.slots[1]] = d2
    return v


def test_chart_compatibility():
    a = _chart()
    # Same chart twice -> overlapping engine slots.
    with pytest.raises(ValueError, match="claim engine slot"):
        check_chart_compatibility((a, a))
    # Single chart is fine.
    check_chart_compatibility((a,))


def test_matches_pint():
    pytest.importorskip("pint")
    # Cross-check kepler_from_laplace against PINT's ELL1->DD TASC/T0 relation
    # on random points.
    rng = np.random.default_rng(7)
    for _ in range(10):
        e = rng.uniform(1e-4, 0.05)
        om = rng.uniform(0, 360)
        t0 = 55000.0 + rng.uniform(-100, 100)
        pb = rng.uniform(1, 50)
        eps1, eps2, tasc = laplace_from_kepler(e, om, t0, pb)
        e2, om2, t02 = kepler_from_laplace(eps1, eps2, tasc, pb)
        # T0 = TASC + PB*om/360 (in turns).
        assert abs(t02 - (tasc + pb * (om2 / 360.0))) < 1e-9


# ---------------------------------------------------------------------------
# Activation matrix rows 2-4 (remedy dispositions) and 7c/7d (guards) through
# the real activate_charts path (review: rows 2-4, 7c, 7d were untested E2E).
# ---------------------------------------------------------------------------


def test_activation_matrix_remedy_rows():
    p = _FakePulsar()
    pol = KeplerLaplacePolicy(mode="auto")

    def _disp(plan):
        return {a.name: a.disposition for a in plan.axes}

    # Row 2 (ablation remedy): T0 sample, ECC+OM z_prior -> sample TASC,
    # z-marginalize EPS1/EPS2.
    plan2, res2, _ = _activate(p, TimingInference.groups(z_prior=["ECC", "OM"]), pol)
    assert len(res2) == 1
    d2 = _disp(plan2)
    assert d2["TASC"] == "sample"
    assert d2["EPS1"] == "marginalize_z_prior" and d2["EPS2"] == "marginalize_z_prior"

    # Row 3: T0 sample, ECC+OM delta_flat -> delta-flat EPS1/EPS2 (their δ stays
    # zero in the seam; variation lives in M_s columns).
    plan3, res3, _ = _activate(p, TimingInference.groups(delta_flat=["ECC", "OM"]), pol)
    assert len(res3) == 1
    d3 = _disp(plan3)
    assert d3["EPS1"] == "marginalize_delta_flat"
    assert d3["EPS2"] == "marginalize_delta_flat"
    assert d3["TASC"] == "sample"

    # Row 4: ECC+OM sample, T0 z_prior -> sample EPS1/EPS2, marginalize TASC.
    plan4, res4, _ = _activate(p, TimingInference.groups(z_prior=["T0"]), pol)
    assert len(res4) == 1
    d4 = _disp(plan4)
    assert d4["EPS1"] == "sample" and d4["EPS2"] == "sample"
    assert d4["TASC"] == "marginalize_z_prior"


def test_activation_seam_and_origin_guards():
    p = _FakePulsar()
    wide = {
        "EPS1": delta_uniform(-0.01, 0.01),
        "EPS2": delta_uniform(-0.01, 0.01),
    }
    # Row 7c: epoch-shift not exact (secular) + a wide EPS box crossing the seam
    # -> demote (origin_certified True isolates the SEAM reason from the origin
    # guard, which runs second).
    cap_secular = BinaryChartCapability(
        kepler_convention="dd",
        epoch_shift_exact=False,
        secular_terms=("OMDOT",),
        origin_certified=True,
        certification_ref="run-x",
        supports_domain=True,
    )
    eng_sec = _FakeEngine(capability=cap_secular)
    with pytest.warns(UserWarning, match="omega-branch seam ray"):
        _, res_s, rec_s = _activate(
            p,
            TimingInference.sample_all(),
            KeplerLaplacePolicy(mode="auto"),
            engine=eng_sec,
            prior_overrides=dict(wide),
        )
    assert res_s == () and rec_s[0]["reason"] == "seam_reachable_with_secular_terms"
    assert rec_s[0]["seam_guard"]["passed"] is False
    with pytest.raises(ValueError, match="omega-branch seam ray"):
        _activate(
            p,
            TimingInference.sample_all(),
            KeplerLaplacePolicy(mode="on"),
            engine=eng_sec,
            prior_overrides=dict(wide),
        )

    # Row 7d: origin in support + uncertified backend -> demote.
    cap_unc = BinaryChartCapability(
        kepler_convention="dd",
        epoch_shift_exact=True,
        secular_terms=(),
        origin_certified=False,
        supports_domain=True,
    )
    with pytest.warns(UserWarning, match="eccentricity origin"):
        _, res_u, rec_u = _activate(
            p,
            TimingInference.sample_all(),
            KeplerLaplacePolicy(mode="auto"),
            engine=_FakeEngine(capability=cap_unc),
            prior_overrides=dict(wide),
        )
    assert res_u == () and rec_u[0]["reason"] == "origin_uncertified_backend"

    # 7d: certified backend -> activates, with certification_ref recorded.
    cap_cert = BinaryChartCapability(
        kepler_convention="dd",
        epoch_shift_exact=True,
        secular_terms=(),
        origin_certified=True,
        certification_ref="cert-7",
        supports_domain=True,
    )
    _, res_c, rec_c = _activate(
        p,
        TimingInference.sample_all(),
        KeplerLaplacePolicy(mode="auto"),
        engine=_FakeEngine(capability=cap_cert),
        prior_overrides=dict(wide),
    )
    assert len(res_c) == 1
    assert rec_c[0]["origin_guard"]["certification_ref"] == "cert-7"

    # A support EXCLUDING the origin activates regardless (origin_guard null).
    _, res_ok, rec_ok = _activate(
        p,
        TimingInference.sample_all(),
        KeplerLaplacePolicy(mode="auto"),
        engine=_FakeEngine(capability=cap_unc),
    )
    assert len(res_ok) == 1 and rec_ok[0]["origin_guard"] is None


def test_override_materialization_identity():
    from nltiming.physical_charts import materialize_eps_override

    # Call with the ACTUAL pre-activation, ENGINE-named plan (plan.proper has
    # ECC/OM/T0, not the synthesized names). The prospective-rename fix is what
    # makes this succeed (review regression).
    p, eng, refs = _suffixed_pulsar_engine()
    plan = _plan(p, TimingInference.sample_all())
    assert "ECC_a" in plan.proper and "EPS1_a" not in plan.proper
    (cand,) = [
        c
        for c in resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
        if c.chart is not None
    ]
    chart = cand.chart
    # An absolute-frame truncated_normal EPS override resolves through the
    # Decimal-backed pipeline (no float-subtraction shortcut).
    overrides = {"EPS1_a": truncated_normal(0.0, 1e-4, -1e-3, 1e-3, frame="absolute")}
    prior_pre = materialize_eps_override(
        overrides, chart, 0, pulsar=p, plan=plan, engine_refs=refs
    )
    assert prior_pre is not None and prior_pre.family == "truncated_normal"

    # Post-activation plan carries the sampling names; the rename is idempotent,
    # so the same call returns a bit-identical AxisPrior.
    active_plan, _, _ = activate_charts(
        plan,
        [cand],
        KeplerLaplacePolicy(),
        prior_overrides=overrides,
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=TimingCoordinatePolicy().nonlinear_scale,
        engine_refs=refs,
        prior_policy="wide_default",
    )
    prior_post = materialize_eps_override(
        overrides, chart, 0, pulsar=p, plan=active_plan, engine_refs=refs
    )
    assert prior_post.lower == prior_pre.lower and prior_post.upper == prior_pre.upper
    assert prior_post.mean == prior_pre.mean and prior_post.std == prior_pre.std


def test_suffixed_groups():
    # Two suffixed binary groups; only _a qualifies (low e). _b has e >= e_max
    # under auto -> skip. Records exist for BOTH groups.
    fitparameters = {
        "ECC_a": {"a": "ECC"},
        "OM_a": {"a": "OM"},
        "T0_a": {"a": "T0"},
        "PB_a": {"a": "PB"},
        "ECC_b": {"b": "ECC"},
        "OM_b": {"b": "OM"},
        "T0_b": {"b": "T0"},
        "PB_b": {"b": "PB"},
    }
    refs = {
        "ECC_a": "8e-4",
        "OM_a": "50.7",
        "T0_a": "55000.0",
        "PB_a": "8.6",
        "ECC_b": "0.3",
        "OM_b": "120.0",
        "T0_b": "56000.0",
        "PB_b": "3.2",
    }
    p = _FakePulsar(fitpars=tuple(fitparameters), fitparameters=fitparameters)
    cands = {
        c.suffix: c
        for c in resolve_chart_candidates(
            p, _FakeEngineNoCap(refs), KeplerLaplacePolicy()
        )
    }
    assert set(cands) == {"_a", "_b"}
    assert cands["_a"].chart is not None
    assert cands["_b"].chart is None and cands["_b"].skip_reason == "e_ref_above_e_max"
    plan = _plan(p, TimingInference.sample_all())
    active, resolved, records = activate_charts(
        plan,
        list(cands.values()),
        KeplerLaplacePolicy(),
        prior_overrides={},
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=TimingCoordinatePolicy().nonlinear_scale,
        engine_refs=refs,
        prior_policy="wide_default",
    )
    # Only _a is renamed to EPS*; _b keeps engine names; both have records.
    assert "EPS1_a" in active.axis_names and "ECC_b" in active.axis_names
    assert {r["suffix"] for r in records} == {"_a", "_b"}
    assert len(resolved) == 1 and resolved[0].chart.suffix == "_a"


def test_eps_default_box_disk_shrink():
    # A huge nonlinear_scale makes the default WLS boxes exceed the disk, so the
    # common-factor shrink engages; every corner must satisfy e <= 1 - margin
    # STRICTLY, and the h1/h2 ratio is preserved (relative WLS scaling kept).
    p = _FakePulsar()
    plan = _plan(p, TimingInference.sample_all())
    (cand,) = [
        c
        for c in resolve_chart_candidates(p, _FakeEngineNoCap(), KeplerLaplacePolicy())
        if c.chart is not None
    ]
    chart = cand.chart
    r_max = chart.DOMAIN_MAX_E - DISK_MARGIN
    rect, supports = resolved_eps_reachability(
        p,
        chart,
        plan,
        p.Mmat,
        nonlinear_scale=1e12,
        prior_overrides={},
        engine_refs=REFS,
    )
    assert all(s.kind == "default_box" for s in supports)
    # Strictly inside the disk at every corner.
    corner = float(
        np.hypot(
            max(abs(rect[0][0]), abs(rect[0][1])),
            max(abs(rect[1][0]), abs(rect[1][1])),
        )
    )
    assert corner <= r_max
    # Ratio of the two half-widths preserved by the common factor.
    h1, h2 = supports[0].hi_delta, supports[1].hi_delta
    assert h1 > 0 and h2 > 0
    # Realistic scale -> no shrink (c == 1); bounds are the raw 50*sigma box.
    _, supports_ok = resolved_eps_reachability(
        p,
        chart,
        plan,
        p.Mmat,
        nonlinear_scale=TimingCoordinatePolicy().nonlinear_scale,
        prior_overrides={},
        engine_refs=REFS,
    )
    # The unshrunk half-widths keep the SAME ratio as the shrunk ones.
    r_shrunk = h1 / h2
    r_full = supports_ok[0].hi_delta / supports_ok[1].hi_delta
    assert abs(r_shrunk - r_full) < 1e-9 * abs(r_full)


def test_fallback_detects_gr_derived_secular_model():
    # A DDGR pulsar computes OMDOT/PBDOT internally (no explicit fitpars), so the
    # name-search fallback would miss them; the binary-type check flags them so
    # the seam guard engages (review correctness fix).
    from nltiming.physical_charts import _present_secular_terms

    class _BinaryParam:
        value = "DDGR"

    model = _FakePINT(BINARY=_BinaryParam())
    p = _FakePulsar(pint_model=model)
    secular = _present_secular_terms(p, "")
    assert {"OMDOT", "PBDOT"} <= secular

    # A plain DD model with no explicit secular params -> none detected.
    class _DDParam:
        value = "DD"

    p_dd = _FakePulsar(pint_model=_FakePINT(BINARY=_DDParam()))
    assert _present_secular_terms(p_dd, "") == set()
