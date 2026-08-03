"""§10.8.2: unit tests for the ``fw10_absorbed`` physical chart."""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.fw10_absorbed import (
    FW10AbsorbedChart,
    assert_fw10_roundtrip,
    fw10_decode,
    fw10_encode,
    fw10_jacobian,
    mean_motion_rad_s,
)
from nltiming.inference import TimingInference
from nltiming.physical_charts import (
    KeplerLaplaceChart,
    KeplerLaplacePolicy,
    activate_charts,
    resolve_chart_candidates,
)
from nltiming.priors import delta_uniform

from test_physical_charts import (  # reuse fakes
    _FakeEngineNoCap,
    _FakePulsar,
    _activate,
    _plan,
)

# J2145-like intrinsic engine point
_J2145 = dict(
    a1=10.16,
    ecc=2.0e-5,
    om_rad=1.0,
    t0=55000.1,
    h3=1.8e-7,
    pb_days=6.838902511,
)


def _engine_refs(**overrides):
    a1 = overrides.get("a1", _J2145["a1"])
    ecc = overrides.get("ecc", _J2145["ecc"])
    om_rad = overrides.get("om_rad", _J2145["om_rad"])
    t0 = overrides.get("t0", _J2145["t0"])
    h3 = overrides.get("h3", _J2145["h3"])
    stig = overrides.get("stig", 0.5)
    pb = overrides.get("pb_days", _J2145["pb_days"])
    refs = {
        "F0": "100.0",
        "A1": repr(a1),
        "ECC": repr(ecc),
        "OM": repr(om_rad * 180.0 / np.pi),
        "T0": repr(t0),
        "H3": repr(h3),
        "STIGMA": repr(stig),
        "PB": repr(pb),
    }
    return refs


FITPARS_FW10 = ("F0", "A1", "ECC", "OM", "T0", "H3", "STIGMA", "PB")


def _fw10_pulsar(fitpars=FITPARS_FW10, ntoa=40):
    return _FakePulsar(fitpars=fitpars, ntoa=ntoa)


# ---------------------------------------------------------------------------
# Pure maps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stig", [0.5, 0.9])
def test_encode_decode_roundtrip(stig):
    a1, ecc, om, t0 = _J2145["a1"], _J2145["ecc"], _J2145["om_rad"], _J2145["t0"]
    h3, pb = _J2145["h3"], _J2145["pb_days"]
    assert_fw10_roundtrip(a1, ecc, om, t0, h3, stig, pb, rtol=1e-12)
    chart = fw10_encode(a1, ecc, om, t0, h3, stig, pb)
    back = fw10_decode(*chart, pb, omega_ref_rad=om)
    for a, b in zip((a1, ecc, om, t0, h3, stig), back):
        scale = max(1.0, abs(a), abs(b))
        assert abs(a - b) <= 1e-12 * scale


def test_decode_reference_equals_engine():
    refs = _engine_refs(stig=0.5)
    chart = FW10AbsorbedChart.from_engine_refs(
        suffix="",
        engine_names=("A1", "ECC", "OM", "T0", "H3", "STIGMA"),
        slots=(1, 2, 3, 4, 5, 6),
        a1=float(refs["A1"]),
        ecc=float(refs["ECC"]),
        om_deg=float(refs["OM"]),
        t0=float(refs["T0"]),
        h3=float(refs["H3"]),
        stig=float(refs["STIGMA"]),
        pb_days=float(refs["PB"]),
        a1_ref_str=refs["A1"],
        ecc_ref_str=refs["ECC"],
        om_ref_str=refs["OM"],
        t0_ref_str=refs["T0"],
        h3_ref_str=refs["H3"],
        stigma_ref_str=refs["STIGMA"],
        pb_ref_str=refs["PB"],
    )
    samples = {
        chart.sample_names[0]: chart.x_p_ref,
        chart.sample_names[1]: chart.eps1p_ref,
        chart.sample_names[2]: chart.eps2p_ref,
        chart.sample_names[3]: chart.tasc_ref,
        chart.sample_names[4]: chart.h3_ref,
        chart.sample_names[5]: chart.stigma_ref,
    }
    eng = chart.decode(samples)
    assert abs(eng["A1"] - chart.a1_ref) < 1e-12 * max(1.0, chart.a1_ref)
    assert abs(eng["ECC"] - chart.e_ref) < 1e-12 * max(1.0, chart.e_ref)
    assert abs(eng["OM"] - float(refs["OM"])) < 1e-9
    assert abs(eng["T0"] - chart.t0_ref) < 1e-12 * max(1.0, chart.t0_ref)
    assert eng["H3"] == chart.h3_ref
    assert eng["STIGMA"] == chart.stigma_ref


def test_jacobian_vs_finite_differences():
    refs = _engine_refs(stig=0.5)
    chart = FW10AbsorbedChart.from_engine_refs(
        suffix="",
        engine_names=("A1", "ECC", "OM", "T0", "H3", "STIGMA"),
        slots=(0, 1, 2, 3, 4, 5),
        a1=float(refs["A1"]),
        ecc=float(refs["ECC"]),
        om_deg=float(refs["OM"]),
        t0=float(refs["T0"]),
        h3=float(refs["H3"]),
        stig=float(refs["STIGMA"]),
        pb_days=float(refs["PB"]),
        a1_ref_str=refs["A1"],
        ecc_ref_str=refs["ECC"],
        om_ref_str=refs["OM"],
        t0_ref_str=refs["T0"],
        h3_ref_str=refs["H3"],
        stigma_ref_str=refs["STIGMA"],
        pb_ref_str=refs["PB"],
    )
    pb = chart.pb_ref
    om_ref = chart.omega_ref_rad

    def f(q):
        a1, ecc, om, t0, h3, stig = fw10_decode(*q, pb, omega_ref_rad=om_ref)
        return np.array([a1, ecc, om * 180.0 / np.pi, t0, h3, stig], dtype=float)

    q0 = np.array(
        [
            chart.x_p_ref,
            chart.eps1p_ref,
            chart.eps2p_ref,
            chart.tasc_ref,
            chart.h3_ref,
            chart.stigma_ref,
        ],
        dtype=float,
    )
    displaced = [
        np.zeros(6),
        np.array([1e-4, 1e-7, -1e-7, 1e-6, 1e-10, 1e-4]),
        np.array([-2e-4, -5e-8, 2e-7, -1e-5, -5e-11, -2e-4]),
    ]

    # Central-difference steps: T0/TASC live near MJD~5e4 (ulp~1e-11), so
    # A1_ABS/TASC_ABS need larger absolute steps; EPS steps stay small for the
    # stiff 1/e OM/T0 rows. atol covers residual float64 FD noise on tiny entries.
    steps0 = np.array([1e-5, 1e-9, 1e-9, 1e-6, 1e-10, 1e-6])

    for d in displaced:
        q = q0 + d
        q[5] = float(np.clip(q[5], 1e-3, 0.999))
        J = fw10_jacobian(*q, pb)
        cols = []
        for k in range(6):
            h = steps0[k]
            hi = q.copy()
            lo = q.copy()
            hi[k] += h
            lo[k] -= h
            cols.append((f(hi) - f(lo)) / (2.0 * h))
        J_fd = np.stack(cols, axis=1)
        np.testing.assert_allclose(J, J_fd, rtol=1e-6, atol=1e-5)


def test_algebraic_encode_inverse_identity():
    """Pure algebraic identity: encode∘decode = id (gap note: no PINT delay)."""
    for stig in (0.5, 0.9):
        chart = fw10_encode(
            _J2145["a1"],
            _J2145["ecc"],
            _J2145["om_rad"],
            _J2145["t0"],
            _J2145["h3"],
            stig,
            _J2145["pb_days"],
        )
        eng = fw10_decode(*chart, _J2145["pb_days"], omega_ref_rad=_J2145["om_rad"])
        chart2 = fw10_encode(*eng[:4], eng[4], eng[5], _J2145["pb_days"])
        for a, b in zip(chart, chart2):
            scale = max(1.0, abs(a), abs(b))
            assert abs(a - b) <= 1e-12 * scale


# ---------------------------------------------------------------------------
# Activation guards
# ---------------------------------------------------------------------------


def test_activation_success_renames_and_tags():
    refs = _engine_refs(stig=0.5)
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    cands = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
    inf = TimingInference.groups(delta_flat=["PB", "F0"])
    plan, resolved, _, records, fw10_recs = activate_charts(
        _plan(p, inf),
        cands,
        KeplerLaplacePolicy(),
        prior_overrides={},
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=1.0,
        engine_refs=refs,
        prior_policy="wide_default",
    )
    assert len(resolved) == 1
    assert isinstance(resolved[0].chart, FW10AbsorbedChart)
    assert resolved[0].chart.name == "fw10_absorbed"
    names = set(plan.axis_names)
    assert {"A1_ABS", "EPS1_ABS", "EPS2_ABS", "TASC_ABS", "H3", "STIGMA"} <= names
    assert not ({"A1", "ECC", "OM", "T0"} & names)
    assert plan.axis("A1").name == "A1_ABS"
    assert plan.axis("A1").physical_chart == "fw10_absorbed"
    assert plan.axis("H3").physical_chart == "fw10_absorbed"
    assert plan.axis("H3").name == "H3"
    assert fw10_recs[0]["enabled"] is True
    # kepler_laplace records why it did not take the group (manifest
    # completeness); a group is charted by exactly one family.
    assert len(records) == 1
    assert records[0]["enabled"] is False
    assert records[0]["reason"] == "claimed_by_fw10_absorbed"


def test_guard_dependency_sampled():
    refs = _engine_refs()
    # PB free and sample_all → dependency_sampled
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    cands = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
    assert cands[0].fw10_chart is not None
    _, resolved, _, _, fw10_recs = activate_charts(
        _plan(p, TimingInference.sample_all()),
        cands,
        KeplerLaplacePolicy(),
        prior_overrides={},
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=1.0,
        engine_refs=refs,
        prior_policy="wide_default",
    )
    assert fw10_recs and fw10_recs[0]["reason"] == "dependency_sampled"
    assert not any(isinstance(r.chart, FW10AbsorbedChart) for r in resolved)


def test_guard_secular_terms_present():
    refs = _engine_refs()
    refs["OMDOT"] = "1e-4"
    fitpars = FITPARS_FW10 + ("OMDOT",)
    p = _FakePulsar(fitpars=fitpars)
    # Freeze PB by marginalizing it so dependency_sampled does not fire first.
    eng = _FakeEngineNoCap(refs=refs)
    cands = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
    assert cands[0].fw10_chart is not None
    assert "OMDOT" in cands[0].secular_terms
    inf = TimingInference.groups(delta_flat=["PB", "F0"])
    _, resolved, _, _, fw10_recs = activate_charts(
        _plan(p, inf),
        cands,
        KeplerLaplacePolicy(),
        prior_overrides={},
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=1.0,
        engine_refs=refs,
        prior_policy="wide_default",
    )
    assert fw10_recs[0]["reason"] == "secular_terms_present"
    assert not any(isinstance(r.chart, FW10AbsorbedChart) for r in resolved)


def test_guard_unsupported_disposition_mix():
    refs = _engine_refs()
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    cands = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
    # Marginalize STIGMA → mix
    inf = TimingInference.groups(delta_flat=["PB", "F0", "STIGMA"])
    _, resolved, _, _, fw10_recs = activate_charts(
        _plan(p, inf),
        cands,
        KeplerLaplacePolicy(),
        prior_overrides={},
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=1.0,
        engine_refs=refs,
        prior_policy="wide_default",
    )
    assert fw10_recs[0]["reason"] == "unsupported_disposition_mix"
    assert not any(isinstance(r.chart, FW10AbsorbedChart) for r in resolved)


def test_guard_near_circular_reference():
    # e_i = 1e-7 with EPS*_ABS half-width 1e-6 → 10×excursion = 1e-5 > e_i
    refs = _engine_refs(ecc=1e-7, stig=0.5)
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    cands = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
    assert cands[0].fw10_chart is not None
    inf = TimingInference.groups(delta_flat=["PB", "F0"])
    overrides = {"EPS1_ABS": delta_uniform(-1e-6, 1e-6)}
    _, resolved, _, _, fw10_recs = activate_charts(
        _plan(p, inf),
        cands,
        KeplerLaplacePolicy(),
        prior_overrides=overrides,
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=1.0,
        engine_refs=refs,
        prior_policy="wide_default",
    )
    assert fw10_recs[0]["reason"] == "near_circular_reference"
    assert not any(isinstance(r.chart, FW10AbsorbedChart) for r in resolved)


def test_prior_on_engine_name_demotes_fw10():
    refs = _engine_refs()
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    cands = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
    inf = TimingInference.groups(delta_flat=["PB", "F0"])
    _, resolved, _, _, fw10_recs = activate_charts(
        _plan(p, inf),
        cands,
        KeplerLaplacePolicy(),
        prior_overrides={"ECC": delta_uniform(-1e-6, 1e-6)},
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=1.0,
        engine_refs=refs,
        prior_policy="wide_default",
    )
    assert fw10_recs[0]["enabled"] is False
    assert fw10_recs[0]["reason"] == "prior_on_engine_axis"
    assert not any(isinstance(r.chart, FW10AbsorbedChart) for r in resolved)


def test_prior_on_h3_stigma_ok_with_fw10():
    refs = _engine_refs()
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    cands = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
    inf = TimingInference.groups(delta_flat=["PB", "F0"])
    _, resolved, _, _, fw10_recs = activate_charts(
        _plan(p, inf),
        cands,
        KeplerLaplacePolicy(),
        prior_overrides={
            "H3": delta_uniform(-1e-9, 1e-9),
            "STIGMA": delta_uniform(-0.1, 0.1),
        },
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=1.0,
        engine_refs=refs,
        prior_policy="wide_default",
    )
    assert fw10_recs[0]["enabled"] is True
    assert isinstance(resolved[0].chart, FW10AbsorbedChart)


def test_activation_all_six_sample_pb_frozen():
    refs = _engine_refs(stig=0.9)
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    cands = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
    assert cands[0].fw10_chart is not None
    inf = TimingInference.groups(delta_flat=["PB", "F0"])
    plan, resolved, _, records, fw10_recs = activate_charts(
        _plan(p, inf),
        cands,
        KeplerLaplacePolicy(),
        prior_overrides={},
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=1.0,
        engine_refs=refs,
        prior_policy="wide_default",
    )
    assert fw10_recs[0]["enabled"] is True
    assert fw10_recs[0]["gauge"] == "absorbed"
    assert fw10_recs[0]["log_abs_det_jacobian_at_ref"] is not None
    assert isinstance(resolved[0].chart, FW10AbsorbedChart)
    assert plan.axis("A1").name == "A1_ABS"
    assert plan.axis("ECC").name == "EPS1_ABS"
    assert plan.axis("OM").name == "EPS2_ABS"
    assert plan.axis("T0").name == "TASC_ABS"
    # Kepler skipped for this group when fw10 wins, but still recorded.
    assert [(r["enabled"], r["reason"]) for r in records] == [
        (False, "claimed_by_fw10_absorbed")
    ]


def test_kepler_only_binaries_unaffected():
    """No H3/STIGMA → fw10 absent; existing kepler path still works."""
    p = _FakePulsar()
    plan, resolved, _, records = _activate(
        p, TimingInference.sample_all(), KeplerLaplacePolicy(mode="auto")
    )
    assert len(resolved) == 1
    assert isinstance(resolved[0].chart, KeplerLaplaceChart)
    assert {"EPS1", "EPS2", "TASC"} <= set(plan.axis_names)
    assert records[0]["enabled"] is True


def test_mean_motion_units():
    nb = mean_motion_rad_s(1.0)
    assert abs(nb - 2.0 * np.pi / 86400.0) < 1e-18


def test_delay_equivalence_along_stigma_grid():
    """Decoded chart points reproduce the absorbed→DDH identity (ties to §7.5).

    At each ς on a grid, encode the absorbed ELL1H printed parameters into
    engine DDH via ``fw10_decode``, then compare PINT stand-alone delays
    (absorbed ELL1H vs DDH) after mean removal. Residual must stay within the
    Case-B fidelity budget (~1 ns).
    """
    astropy = pytest.importorskip("astropy")
    u = astropy.units
    from pint.models.stand_alone_psr_binaries.ELL1H_model import ELL1Hmodel
    from pint.models.stand_alone_psr_binaries.DDH_model import DDHmodel

    ls = u.lsec
    pb_d = _J2145["pb_days"]
    x_p = _J2145["a1"]
    e1p, e2p = 7.0e-6, -1.8e-5
    tasc = 55000.0
    h3 = _J2145["h3"]
    n = 256
    t = tasc + np.linspace(0.0, pb_d, n, endpoint=False)

    for stig in (0.5, 0.9):
        # Absorbed ELL1H delay at printed chart coordinates
        bm = ELL1Hmodel()
        bm.fit_params = ["H3", "STIGMA"]
        bm.update_input(
            barycentric_toa=np.asarray(t, dtype=np.longdouble),
            PB=pb_d * u.day,
            A1=x_p * ls,
            TASC=np.longdouble(tasc) * u.day,
            EPS1=e1p * u.Unit(""),
            EPS2=e2p * u.Unit(""),
            H3=h3 * u.s,
            STIGMA=stig * u.Unit(""),
        )
        bm.ds_func = bm.delayS3p_H3_STIGMA_exact
        d_abs = bm.ELL1Hdelay().to_value(u.s)

        a1, ecc, om_rad, t0, _, _ = fw10_decode(x_p, e1p, e2p, tasc, h3, stig, pb_d)
        dd = DDHmodel()
        dd.update_input(
            barycentric_toa=np.asarray(t, dtype=np.longdouble),
            PB=pb_d * u.day,
            A1=a1 * ls,
            T0=np.longdouble(t0) * u.day,
            ECC=ecc * u.Unit(""),
            OM=(om_rad * 180.0 / np.pi) * u.deg,
            H3=h3 * u.s,
            STIGMA=stig * u.Unit(""),
        )
        d_ddh = dd.DDdelay().to_value(u.s)
        resid = (d_abs - d_ddh) - np.mean(d_abs - d_ddh)
        # Case-B acceptance: ≤ 1 ns (feature §7.5 / §10.8.2)
        assert (
            np.max(np.abs(resid)) < 1.0e-9
        ), f"stig={stig}: max|centered delay diff| = {np.max(np.abs(resid)):.3e} s"


# ---------------------------------------------------------------------------
# Posterior decode (RunResults.posterior parity with kepler_laplace)
# ---------------------------------------------------------------------------


def test_derived_fw10_columns_decode_matches_chart():
    """Chart-frame posterior samples decode back to engine A1/ECC/OM/T0.

    Without this the chart renames four axes and the run's posterior carries
    only ``*_ABS`` columns, so the engine parameters are unrecoverable — the
    ``derived_kepler_columns`` gap for the six-axis sibling.
    """
    from nltiming.run_io import derived_fw10_columns

    refs = _engine_refs(stig=0.5)
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    cands = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
    chart = cands[0].fw10_chart
    assert chart is not None
    _, _, _, _, fw10_recs = activate_charts(
        _plan(p, TimingInference.groups(delta_flat=["PB", "F0"])),
        cands,
        KeplerLaplacePolicy(),
        prior_overrides={},
        pint_model=None,
        pulsar=p,
        engine_design_matrix=p.Mmat,
        nonlinear_scale=1.0,
        engine_refs=refs,
        prior_policy="wide_default",
    )
    record = fw10_recs[0]
    assert record["enabled"] is True

    # Three draws: the reference itself plus two displaced points.
    rng = np.random.default_rng(0)
    n = 3
    samples = {
        chart.sample_names[0]: chart.x_p_ref
        + np.r_[0.0, rng.normal(size=n - 1) * 1e-6],
        chart.sample_names[1]: chart.eps1p_ref
        + np.r_[0.0, rng.normal(size=n - 1) * 1e-7],
        chart.sample_names[2]: chart.eps2p_ref
        + np.r_[0.0, rng.normal(size=n - 1) * 1e-7],
        chart.sample_names[3]: chart.tasc_ref
        + np.r_[0.0, rng.normal(size=n - 1) * 1e-4],
        chart.sample_names[4]: np.full(n, chart.h3_ref),
        chart.sample_names[5]: np.full(n, chart.stigma_ref),
    }
    out = derived_fw10_columns(samples, {"groups": [record]})
    a1_name, ecc_name, om_name, t0_name = chart.engine_names[:4]
    assert set(out) == {a1_name, ecc_name, om_name, t0_name}

    # Draw 0 is the reference point: decode must return the engine reference.
    assert out[a1_name][0] == pytest.approx(chart.a1_ref, rel=1e-12)
    assert out[ecc_name][0] == pytest.approx(chart.e_ref, rel=1e-12)
    assert out[om_name][0] == pytest.approx(
        chart.omega_ref_rad * 180.0 / np.pi, rel=1e-12
    )
    assert out[t0_name][0] == pytest.approx(chart.t0_ref, rel=1e-12)

    # Every draw agrees with the chart's own decode (the likelihood's path).
    ref = chart.decode(samples)
    for name in (a1_name, ecc_name, om_name, t0_name):
        np.testing.assert_allclose(out[name], ref[name], rtol=1e-12, atol=0.0)


def test_derived_fw10_columns_skip_partial_and_disabled():
    """No fabricated columns from a disabled group or a partial sample dict."""
    from nltiming.run_io import derived_fw10_columns

    refs = _engine_refs(stig=0.5)
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    chart = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())[0].fw10_chart
    disabled = chart.record(enabled=False, reason="unsupported_disposition_mix")
    assert derived_fw10_columns({}, {"groups": [disabled]}) == {}
    assert derived_fw10_columns({}, None) == {}

    enabled = chart.record(enabled=True, reason=None, dispositions=None)
    partial = {chart.sample_names[0]: np.array([chart.x_p_ref])}
    assert derived_fw10_columns(partial, {"groups": [enabled]}) == {}


def test_decode_out_of_domain_stigma_is_nan_not_finite_garbage():
    """ς outside (0, 1] must yield NaN, not a plausible-looking orbit.

    For ς < 0 every division in the map stays finite, so an unguarded decode
    hands the likelihood a meaningless but *finite* orbit — silently wrong
    rather than rejected.
    """
    refs = _engine_refs(stig=0.5)
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    chart = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())[0].fw10_chart
    n = len(p.fitpars)

    base = np.zeros(n)
    assert np.all(np.isfinite(chart.apply_delta(base.copy(), np)))

    for d_stig in (-0.6, -1.0, 0.6):  # -> stigma = -0.1, -0.5, 1.1
        vec = np.zeros(n)
        vec[chart.slots[5]] = d_stig
        out = chart.apply_delta(vec.copy(), np)
        assert np.all(np.isnan(np.asarray(out)[list(chart.slots)])), d_stig
        assert not chart.in_domain(vec)


def test_declared_stigma_prior_outside_domain_refuses_activation():
    """A declared STIGMA prior whose support leaves (0, 1] blocks the chart."""
    refs = _engine_refs(stig=0.5)
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    cands = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())
    inf = TimingInference.groups(delta_flat=["PB", "F0"])

    def _activate_with(overrides):
        _, resolved, _, _, recs = activate_charts(
            _plan(p, inf),
            cands,
            KeplerLaplacePolicy(),
            prior_overrides=overrides,
            pint_model=None,
            pulsar=p,
            engine_design_matrix=p.Mmat,
            nonlinear_scale=1.0,
            engine_refs=refs,
            prior_policy="wide_default",
        )
        return resolved, recs

    # Delta frame: reference 0.5, so ±0.1 -> [0.4, 0.6] is fine ...
    resolved, recs = _activate_with({"STIGMA": delta_uniform(-0.1, 0.1)})
    assert recs[0]["enabled"] is True
    assert isinstance(resolved[0].chart, FW10AbsorbedChart)

    # ... but ±0.8 reaches stigma = -0.3.
    resolved, recs = _activate_with({"STIGMA": delta_uniform(-0.8, 0.8)})
    assert recs[0]["reason"] == "stigma_support_out_of_domain"
    assert not any(isinstance(r.chart, FW10AbsorbedChart) for r in resolved)


def test_record_carries_the_normalized_omega_branch():
    """OM_normalized, not the raw par string, is what the decode keys off.

    An OM outside [0, 360) would otherwise send the posterior decode onto a
    branch 2*pi away, shifting derived T0 by a whole PB.
    """
    from nltiming.run_io import derived_fw10_columns

    raw_om_deg = _J2145["om_rad"] * 180.0 / np.pi
    refs = _engine_refs(stig=0.5)
    refs["OM"] = repr(raw_om_deg + 360.0)  # same orbit, wrapped branch
    p = _fw10_pulsar()
    eng = _FakeEngineNoCap(refs=refs)
    chart = resolve_chart_candidates(p, eng, KeplerLaplacePolicy())[0].fw10_chart
    assert chart is not None
    record = chart.record(enabled=True, reason=None, dispositions=None)
    assert float(record["theta_ref_engine"]["OM"]) > 360.0
    assert 0.0 <= float(record["theta_ref_engine"]["OM_normalized"]) < 360.0

    samples = {
        chart.sample_names[0]: np.array([chart.x_p_ref]),
        chart.sample_names[1]: np.array([chart.eps1p_ref]),
        chart.sample_names[2]: np.array([chart.eps2p_ref]),
        chart.sample_names[3]: np.array([chart.tasc_ref]),
        chart.sample_names[4]: np.array([chart.h3_ref]),
        chart.sample_names[5]: np.array([chart.stigma_ref]),
    }
    out = derived_fw10_columns(samples, {"groups": [record]})
    ref = chart.decode(samples)
    for name in chart.engine_names[:4]:
        np.testing.assert_allclose(out[name], ref[name], rtol=1e-12, atol=0.0)
    # T0 lands on the chart's branch, not one shifted by a whole PB.
    assert abs(float(out[chart.engine_names[3]][0]) - chart.t0_ref) < 1e-9
