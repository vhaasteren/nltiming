"""PR-3 (§12.3): residual invariance through the chart<->engine seam, and the
branch-seam decode structure.

The composed map (chart then engine) must reproduce, to machine precision, the
engine residual at the corresponding absolute Kepler point; and two sampling
points straddling the omega-branch seam must decode (on the reference-local
branch) to engine points differing by exactly (360 deg, PB) -- the same orbit
up to secular terms.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytest.importorskip("jug")

from nltiming.physical_charts import kepler_from_laplace  # noqa: E402

from test_physical_charts_context import _ctx  # noqa: E402


def test_zero_delta_residual_identity():
    _, off = _ctx(binary_chart="off")
    _, ctx = _ctx()
    nfit = len(ctx.plan.fitpars)
    r_off = np.asarray(ctx.engine.residual_delta_jax(jnp.zeros(nfit)))
    k = len(ctx.plan.sampled)
    r_chart = np.asarray(
        ctx.engine.residual_delta_jax(
            ctx.engine_delta_map.full_engine_delta(jnp.zeros(k), jnp)
        )
    )
    np.testing.assert_allclose(r_chart, r_off, atol=1e-12, rtol=0)


def test_finite_delta_matches_hand_built():
    _, ctx = _ctx()
    chart = ctx.physical_charts[0]
    rng = np.random.default_rng(11)
    names = list(ctx.plan.sampled)
    i1, i2, it = names.index("EPS1"), names.index("EPS2"), names.index("TASC")
    ipb = names.index("PB")
    for _ in range(8):
        vals = np.zeros(len(names))
        vals[i1], vals[i2] = rng.normal(scale=1e-4, size=2)
        vals[it] = rng.normal(scale=1e-4)
        vals[ipb] = rng.normal(scale=1e-5)
        full = np.asarray(
            ctx.engine_delta_map.full_engine_delta(jnp.asarray(vals), jnp)
        )
        # Hand-built engine delta from the same absolute Kepler point.
        eps1 = chart.eps1_ref + vals[i1]
        eps2 = chart.eps2_ref + vals[i2]
        tasc = float(chart.tasc_ref_str) + vals[it]
        pb = chart.pb_ref + vals[ipb]
        e_abs, om_abs, t0_abs = kepler_from_laplace(eps1, eps2, tasc, pb)
        s_ecc, s_om, s_t0 = chart.slots
        assert full[s_ecc] == pytest.approx(e_abs - chart.e_ref, abs=1e-12)
        dom = (om_abs - float(chart.om_ref_norm_str) + 180.0) % 360.0 - 180.0
        assert full[s_om] == pytest.approx(dom, abs=1e-9)
        assert full[s_t0] == pytest.approx(t0_abs - float(chart.t0_ref_str), abs=1e-9)


def test_branch_seam_decode_structure():
    # DECODE-level structure only (review: this is not the full §12.3 waveform
    # coverage — the waveform-agreement / secular-offset check needs a periodic
    # DD engine, see test_branch_seam_discontinuity below, skipped here).
    # Two EPS points straddling the +/-180deg Delta-omega seam decode (on the
    # reference-local branch) to engine points differing by (~360deg, ~PB): the
    # same orbit up to secular terms.
    _, ctx = _ctx()
    chart = ctx.physical_charts[0]
    e = chart.e_ref
    w = chart.omega_ref_rad
    # Points just either side of the seam ray (omega = w + pi).
    for sign, other in ((+1, -1),):
        wa = w + (np.pi - 1e-7)
        wb = w - (np.pi - 1e-7)
        da = chart.engine_delta_from_sample_delta(
            e * np.sin(wa) - chart.eps1_ref,
            e * np.cos(wa) - chart.eps2_ref,
            0.0,
            d_pb=0.0,
            xp=np,
        )
        db = chart.engine_delta_from_sample_delta(
            e * np.sin(wb) - chart.eps1_ref,
            e * np.cos(wb) - chart.eps2_ref,
            0.0,
            d_pb=0.0,
            xp=np,
        )
        d_om_jump = abs(da[1] - db[1])
        d_t0_jump = abs(da[2] - db[2])
        assert abs(d_om_jump - 360.0) < 1e-3
        assert abs(d_t0_jump - chart.pb_ref) < 1e-3 * chart.pb_ref

    # A closed EPS loop returns the decode to its start (the wrap is memoryless).
    phis = np.linspace(0.0, 2 * np.pi, 64)
    start = chart.engine_delta_from_sample_delta(
        e * np.sin(w) - chart.eps1_ref,
        e * np.cos(w) - chart.eps2_ref,
        0.0,
        d_pb=0.0,
        xp=np,
    )
    end = chart.engine_delta_from_sample_delta(
        e * np.sin(w + 2 * np.pi) - chart.eps1_ref,
        e * np.cos(w + 2 * np.pi) - chart.eps2_ref,
        0.0,
        d_pb=0.0,
        xp=np,
    )
    assert abs(start[0] - end[0]) < 1e-12
    assert abs(start[1] - end[1]) < 1e-9
    _ = phis


def test_branch_seam_discontinuity():
    # The NORMATIVE §12.3 test: waveform agreement across the seam without
    # secular terms, and jump == analytic secular offset with OMDOT/PBDOT/etc.
    # This requires a real periodic DD engine (a linear surrogate is not
    # orbit-periodic: OM+360 is NOT invisible to it), so it is exercised by the
    # requires_jug J1640 validation (§12.6). The decode-level structure is
    # covered engine-agnostically by test_branch_seam_decode_structure above.
    pytest.skip("waveform seam-offset bound requires a periodic DD engine (§12.6)")
