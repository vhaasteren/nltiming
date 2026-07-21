"""PR-3 (§12.3): the Kepler<->Laplace chart wired through the full pipeline on
a JAX-capable (linear JUG) binary fixture.

The fixture is a *linear* engine, so the composed map is chart(nonlinear) then
engine(linear): the seam, the slot-preserving rename, W_s/W_m, the frame-change
design matrix, and the prior/guard handoff are all exercised end-to-end.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytest.importorskip("jug")

from nltiming import TimingInference  # noqa: E402
from nltiming.engines.base import LinearModel  # noqa: E402
from nltiming.engines.jug import LinearizedJugEngine  # noqa: E402
from nltiming.nonlinear_timing_model import NonLinearTimingModel  # noqa: E402
from nltiming.physical_charts import (  # noqa: E402
    kepler_from_laplace,
    unwrap_om_delta_deg,
)
from nltiming.priors import delta_uniform, normal  # noqa: E402

REF = {
    "F0": "100.0",
    "ECC": "8e-4",
    "OM": "50.7",
    "T0": "55000.0",
    "PB": "8.6866194196",
}


class _BinaryPulsar:
    def __init__(self, seed=1):
        self.name = "JBIN+0000"
        self.fitpars = ("F0", "ECC", "OM", "T0", "PB")
        n = 60
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, 1.0, n)
        design = np.column_stack(
            [np.ones(n), np.sin(2 * t), np.cos(3 * t), t - 0.5, np.sin(5 * t)]
        )
        self._toas = t * 3.15e7 + 5.3e4
        self._residuals = 1e-7 * rng.standard_normal(n)
        self._toaerrs = np.full(n, 1e-7)
        self._freqs = np.full(n, 1400.0)
        self._bf = np.array(["d"] * n, dtype="U8")
        self._flags = {"pta": self._bf}
        model = LinearModel.from_design(
            fitpars=self.fitpars, design=design, theta_exact=dict(REF)
        )
        self._backend = LinearizedJugEngine.from_linear_model(model)

    toas = property(lambda s: s._toas)
    residuals = property(lambda s: s._residuals)
    toaerrs = property(lambda s: s._toaerrs)
    freqs = property(lambda s: s._freqs)
    Mmat = property(lambda s: s._backend.design_matrix())
    flags = property(lambda s: s._flags)
    backend_flags = property(lambda s: s._bf)

    def state_id(self):
        return "bin"

    def pint_model(self):
        return None

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend


def _ctx(inference=None, binary_chart="auto", **kw):
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=inference or TimingInference.sample_all(),
        binary_chart=binary_chart,
        name="t",
        **kw,
    )
    return ntm, ntm.for_pulsar(_BinaryPulsar(), condition=False)


# ---------------------------------------------------------------------------


def test_plan_names_and_slots():
    _, off = _ctx(binary_chart="off")
    assert off.plan.axis_names == ("F0", "ECC", "OM", "T0", "PB")
    assert off.physical_charts == ()

    _, ctx = _ctx()
    assert ctx.plan.axis_names == ("F0", "EPS1", "EPS2", "TASC", "PB")
    for eng, samp in (("ECC", "EPS1"), ("OM", "EPS2"), ("T0", "TASC")):
        off_ax = off.plan.axis(eng)
        ax = ctx.plan.axis(samp)
        assert ax.fitpar_index == off_ax.fitpar_index  # same slot
        assert ax.engine_name == eng
        assert ax.physical_chart == "kepler_laplace"
        assert ax.linearity_sources == ()
        # engine-name lookup resolves to the charted axis.
        assert ctx.plan.axis(eng).name == samp
    # plan.inference stays engine-frame (no rewrite here — sample_all).
    assert ctx.plan.inference.preset in ("explicit",)


def test_delay_keys_and_sites():
    _, ctx = _ctx()
    assert ctx.delay_keys == tuple(
        f"JBIN+0000_t_{n}" for n in ("F0", "EPS1", "EPS2", "TASC", "PB")
    )


def test_cache_key_includes_policy():
    ntm_auto = NonLinearTimingModel(
        engines="jug", inference=TimingInference.sample_all(), binary_chart="auto"
    )
    ntm_off = NonLinearTimingModel(
        engines="jug", inference=TimingInference.sample_all(), binary_chart="off"
    )
    assert ntm_auto._config_fingerprint() != ntm_off._config_fingerprint()


def test_zero_delta_residual_identity():
    _, off = _ctx(binary_chart="off")
    _, ctx = _ctx()
    nfit = len(ctx.plan.fitpars)
    engine = ctx.engine
    # chart-off zero-delta residual.
    r_off = np.asarray(engine.residual_delta_jax(jnp.zeros(nfit)))
    # charted sampled-mode map at zero sampled input.
    k = len(ctx.plan.sampled)
    r_chart = np.asarray(
        engine.residual_delta_jax(
            ctx.engine_delta_map.full_engine_delta(jnp.zeros(k), jnp)
        )
    )
    np.testing.assert_allclose(r_chart, r_off, atol=1e-12, rtol=0)


def test_finite_delta_matches_hand_built():
    _, ctx = _ctx()
    chart = ctx.physical_charts[0]
    # sampled order: F0, EPS1, EPS2, TASC, PB.
    d_f0, d_e1, d_e2, d_ta, d_pb = 1e-9, 2e-4, -1e-4, 3e-4, 5e-5
    vals = np.array([d_f0, d_e1, d_e2, d_ta, d_pb])
    full = np.asarray(ctx.engine_delta_map.full_engine_delta(jnp.asarray(vals), jnp))

    # Hand-built engine delta from the same absolute Kepler point.
    eps1 = chart.eps1_ref + d_e1
    eps2 = chart.eps2_ref + d_e2
    tasc = float(chart.tasc_ref_str) + d_ta
    pb = chart.pb_ref + d_pb
    e_abs, om_abs, t0_abs = kepler_from_laplace(eps1, eps2, tasc, pb)
    d_ecc = e_abs - chart.e_ref
    d_om = unwrap_om_delta_deg(om_abs, float(chart.om_ref_norm_str))
    d_t0 = t0_abs - float(chart.t0_ref_str)
    s_ecc, s_om, s_t0 = chart.slots
    assert full[0] == pytest.approx(d_f0)  # F0 unchanged
    assert full[chart.pb_slot] == pytest.approx(d_pb)  # PB unchanged
    assert full[s_ecc] == pytest.approx(d_ecc, abs=1e-12)
    assert full[s_om] == pytest.approx(d_om, abs=1e-9)
    assert full[s_t0] == pytest.approx(d_t0, abs=1e-9)


def test_all_marginalized_bit_identity():
    inf = TimingInference.groups(delta_flat=["ECC", "OM", "T0"])
    _, off = _ctx(inference=inf, binary_chart="off")
    _, auto = _ctx(inference=inf, binary_chart="auto")
    # Chart skips (no sampled charted axis) -> plan and design identical.
    assert auto.physical_charts == ()
    assert auto.binary_chart_records[0]["reason"] == "no_sampled_axis"
    assert auto.binary_chart_records[0]["enabled"] is False
    assert auto.plan.axis_names == off.plan.axis_names
    np.testing.assert_array_equal(auto.design_matrix, off.design_matrix)
    # Residual at random sampled points identical bit-for-bit.
    rng = np.random.default_rng(0)
    k = len(off.plan.sampled)
    for _ in range(5):
        v = rng.normal(size=k) * 1e-9
        r_off = np.asarray(
            off.engine.residual_delta_jax(
                off.engine_delta_map.full_engine_delta(jnp.asarray(v), jnp)
            )
        )
        r_auto = np.asarray(
            auto.engine.residual_delta_jax(
                auto.engine_delta_map.full_engine_delta(jnp.asarray(v), jnp)
            )
        )
        np.testing.assert_array_equal(r_off, r_auto)
    # under "on", an all-marginalized triple warns.
    with pytest.warns(UserWarning):
        _ctx(inference=inf, binary_chart="on")


def test_partial_marginalization_activates():
    # sample T0, z_prior ECC+OM -> TASC sampled, EPS1/EPS2 z-marginalized.
    inf = TimingInference.groups(z_prior=["ECC", "OM"])
    _, ctx = _ctx(inference=inf)
    assert ctx.physical_charts and ctx.physical_charts[0].suffix == ""
    assert "TASC" in ctx.plan.sampled
    assert set(ctx.plan.marginalized_z) >= {"EPS1", "EPS2"}
    # W_m carries the two z-marginalized charted columns.
    assert ctx.linearization.marginalized_z_basis.shape[1] == len(
        ctx.plan.marginalized_z
    )
    assert np.all(np.isfinite(ctx.linearization.marginalized_z_basis))


def test_delta_flat_charted_columns_reference():
    # sample TASC, delta-flat ECC+OM -> EPS1/EPS2 delta-flat.
    inf = TimingInference.groups(delta_flat=["ECC", "OM"])
    _, ctx = _ctx(inference=inf)
    assert ctx.physical_charts, "chart should activate (T0 sampled)"
    charts = ctx.physical_charts
    # M_s charted columns (M_e(ref) @ B(ref)) vs FD of the composed residual map.
    from nltiming.frames import apply_charts

    def r_of_full(vec):
        return np.asarray(
            ctx.engine.residual_delta_jax(apply_charts(jnp.asarray(vec), charts, jnp))
        )

    nfit = len(ctx.plan.fitpars)
    for slot in (charts[0].slots[0], charts[0].slots[1]):
        h = 1e-6
        e = np.zeros(nfit)
        e[slot] = h
        fd = (r_of_full(e) - r_of_full(-e)) / (2 * h)
        # design_matrix column = d(residual delta)/d(sampling axis); the emitted
        # delay is -residual, so the design column is +d(residual)/d(axis) here.
        np.testing.assert_allclose(ctx.design_matrix[:, slot], fd, rtol=1e-5, atol=1e-8)


def test_with_expansion_rebuilds_map():
    _, ctx = _ctx()
    k = len(ctx.plan.sampled)
    # A moved expansion over proper axes (all sampled here), interior to each
    # axis's prior box.
    proper = ctx.plan.proper
    delta = {name: 0.0 for name in proper}
    tasc_prior = ctx.plan.axis("TASC").prior
    delta["TASC"] = 0.2 * float(tasc_prior.upper)
    moved = ctx.with_expansion(delta=delta)
    # No charted delta-flat axes -> design_matrix bit-unchanged.
    np.testing.assert_array_equal(moved.design_matrix, ctx.design_matrix)
    # The sampled-mode map still round-trips.
    v = np.zeros(k)
    assert np.all(
        np.isfinite(
            np.asarray(moved.engine_delta_map.full_engine_delta(jnp.asarray(v), jnp))
        )
    )


def test_prior_overrides_with_chart():
    # A deliberate T0 prior demotes the chart (auto warns).
    with pytest.warns(UserWarning, match="targets ECC, OM, or"):
        _, ctx = _ctx(priors={"T0": delta_uniform(-1e-4, 1e-4)})
    assert ctx.physical_charts == ()
    assert ctx.plan.axis("T0").name == "T0"  # engine frame kept
    # 'on' raises for a Kepler-axis prior.
    with pytest.raises(ValueError):
        _ctx(binary_chart="on", priors={"ECC": delta_uniform(-1e-4, 1e-4)})
    # A TASC prior is first-class and lands on the sampling axis (activates).
    _, ctx2 = _ctx(priors={"TASC": delta_uniform(-1e-3, 1e-3)})
    assert ctx2.physical_charts
    tasc_axis = ctx2.plan.axis("TASC")
    assert tasc_axis.prior is not None and tasc_axis.prior.family == "uniform"


def test_prior_semantics_disclosure():
    _, ctx = _ctx()
    man = ctx.binary_chart_manifest()
    assert man["policy"]["prior"] == "sampling_frame"
    assert man["policy"]["default_prior_package"] == "nlt-eps-wls-boxes-v1"
    assert man["groups"][0]["enabled"] is True
    # charted EPS axes resolve the cheat_wls source (default boxes).
    eps_axis = ctx.plan.axis("EPS1")
    assert eps_axis.prior_source == "cheat_wls"
    assert eps_axis.prior.family == "uniform"


def test_wls_default_sigmas_finite_at_low_e():
    _, ctx = _ctx()
    for name in ("EPS1", "EPS2", "TASC"):
        prior = ctx.plan.axis(name).prior
        assert prior is not None
        if prior.family == "uniform":
            assert np.isfinite(prior.lower) and np.isfinite(prior.upper)
            assert prior.upper > prior.lower


def test_guard_support_equals_prior_support_no_recompute(monkeypatch):
    # Prove the prior path installs the STORED box and never recomputes WLS for
    # charted EPS axes. Make EPS1/EPS2 the ONLY cheat_wls axes: delta-flat
    # F0/PB, sample ECC/OM/T0, and give TASC an explicit user prior. Then
    # monkeypatch the model-side schur_delta_wls (the prior path's copy) to
    # raise; activation's own reachability schur (physical_charts' copy) is
    # untouched, so the box is still computed there.
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(delta_flat=["F0", "PB"]),
        binary_chart="auto",
        priors={"TASC": delta_uniform(-1e-3, 1e-3)},
        name="t",
    )
    import nltiming.nonlinear_timing_model as M

    def _boom(*a, **k):
        raise AssertionError("prior path must not recompute WLS for charted EPS axes")

    monkeypatch.setattr(M, "schur_delta_wls", _boom)
    ctx = ntm.for_pulsar(_BinaryPulsar(), condition=False)  # must not raise
    assert ctx.physical_charts
    assert ctx.plan.axis("EPS1").prior.family == "uniform"
    assert ctx.plan.axis("EPS2").prior.family == "uniform"


def test_user_eps_normal_prior_rejected():
    with pytest.raises(ValueError, match="unbounded support"):
        _ctx(priors={"EPS1": normal(0.0, 1e-4, frame="delta")})


# ---------------------------------------------------------------------------
# Review gap-fill: real W_m, exact moved-expansion columns, PINT demotion,
# composed-likelihood origin certification, and the geometry smoke.
# ---------------------------------------------------------------------------


class _Prior:
    def __init__(self, lo, hi):
        self.lower_bound = lo
        self.upper_bound = hi


class _Param:
    def __init__(self, value=None, prior=None):
        self.value = value
        self.prior = prior


class _FakePINT:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _BinaryPulsarPINT(_BinaryPulsar):
    def __init__(self, pint_model, seed=1):
        super().__init__(seed=seed)
        self._pint = pint_model

    def pint_model(self):
        return self._pint


def test_partial_marginalization_wm():
    # sample T0 (-> TASC), z_prior ECC+OM (-> EPS1/EPS2 z-marginalized). W_m must
    # be the exact derivative of the composed (chart -> engine) waveform over the
    # z-marg z-axes -- cross-checked by an independent numpy finite difference.
    from nltiming.frames import EngineDeltaMap

    _, ctx = _ctx(inference=TimingInference.groups(z_prior=["ECC", "OM"]))
    charts = ctx.physical_charts
    plan = ctx.plan
    proper = list(plan.proper)
    zm_names = plan.marginalized_z
    W_m = np.asarray(ctx.linearization.marginalized_z_basis)
    assert W_m.shape[1] == len(zm_names) >= 2

    emap = EngineDeltaMap.for_proper(plan, charts)
    ps = ctx.proper_space
    z0 = np.asarray(ps.z_from_delta(np.zeros(len(proper)), np))

    def d_of_z(z):
        delta_s = np.asarray(ps.delta_from_z(z, np), dtype=float)
        full = emap.full_engine_delta(delta_s, np)
        return -np.asarray(ctx.engine.residual_delta(full), dtype=float)

    zm_pos = [proper.index(n) for n in zm_names]
    h = 1e-6
    for col, pos in enumerate(zm_pos):
        e = np.zeros(len(proper))
        e[pos] = h
        fd = (d_of_z(z0 + e) - d_of_z(z0 - e)) / (2 * h)
        np.testing.assert_allclose(W_m[:, col], fd, rtol=1e-5, atol=1e-9)


def _delta_full_for(ctx, delta_map):
    delta_full = np.zeros(len(ctx.plan.fitpars), dtype=float)
    for a in ctx.plan.axes:
        if a.disposition in ("sample", "marginalize_z_prior"):
            delta_full[a.fitpar_index] = float(delta_map[a.name])
    return delta_full


def test_with_expansion_rebuilds_map_charted_delta_flat():
    from nltiming.frames import apply_charts
    from nltiming.nonlinear_timing_model import _exact_flat_columns

    # EPS1/EPS2 delta-flat (charted), TASC sampled.
    _, ctx = _ctx(inference=TimingInference.groups(delta_flat=["ECC", "OM"]))
    charts = ctx.physical_charts
    assert charts
    flat_slots = tuple(
        a.fitpar_index
        for a in ctx.plan.axes
        if a.disposition == "marginalize_delta_flat" and a.physical_chart is not None
    )
    assert len(flat_slots) == 2

    delta = {n: 0.0 for n in ctx.plan.proper}
    delta["TASC"] = 0.2 * float(ctx.plan.axis("TASC").prior.upper)
    moved = ctx.with_expansion(delta=delta)
    delta_full = _delta_full_for(ctx, delta)
    cols = _exact_flat_columns(ctx.engine, ctx.plan, charts, delta_full, flat_slots)

    def r(vec):
        return np.asarray(
            ctx.engine.residual_delta_jax(apply_charts(jnp.asarray(vec), charts, jnp))
        )

    h = 1e-6
    for s in flat_slots:
        # The rebuilt column IS the exact composed jvp (not the M_e(ref)*B(exp)
        # hybrid), and matches a finite difference of the composed map.
        np.testing.assert_array_equal(moved.design_matrix[:, s], cols[s])
        e = np.zeros(len(ctx.plan.fitpars))
        e[s] = h
        fd = (r(delta_full + e) - r(delta_full - e)) / (2 * h)
        np.testing.assert_allclose(moved.design_matrix[:, s], fd, rtol=1e-5, atol=1e-8)
    # Non-charted columns and engine_design_matrix are bit-unchanged.
    for s in range(len(ctx.plan.fitpars)):
        if s not in flat_slots:
            np.testing.assert_array_equal(
                moved.design_matrix[:, s], ctx.design_matrix[:, s]
            )
    np.testing.assert_array_equal(moved.engine_design_matrix, ctx.engine_design_matrix)


def test_pint_prior_demotion_end_to_end():
    # A pulsar whose PINT model carries an informative ECC prior: auto demotes
    # with the pint_prior_on_kepler_axis warning, the plan keeps engine names,
    # and the resolved prior block honors the PINT ECC prior (source "pint").
    model = _FakePINT(ECC=_Param(value=8e-4, prior=_Prior(0.0, 0.02)))
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.sample_all(),
        binary_chart="auto",
        name="t",
    )
    with pytest.warns(UserWarning, match="informative"):
        ctx = ntm.for_pulsar(_BinaryPulsarPINT(model), condition=False)
    assert ctx.physical_charts == ()
    assert ctx.plan.axis("ECC").name == "ECC"
    ecc_axis = ctx.plan.axis("ECC")
    assert ecc_axis.prior_source == "pint"
    assert ecc_axis.prior.family == "uniform"
    assert ctx.binary_chart_records[0]["reason"] == "pint_prior_on_kepler_axis"
    # 'on' raises.
    ntm_on = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.sample_all(),
        binary_chart="on",
        name="t",
    )
    with pytest.raises(ValueError):
        ntm_on.for_pulsar(_BinaryPulsarPINT(model), condition=False)


def test_full_likelihood_origin_certification():
    # The composed likelihood (chart -> engine -> Gaussian) has FINITE value,
    # gradient, and Hessian diagonal over shrinking annuli approaching the
    # eccentricity origin (NaN only at the exact origin, measure zero), and a
    # leapfrog toward the origin produces no NaN.
    #
    # NOTE (review): the stronger "no 1/e blow-through — cancellation survives
    # autodiff" claim is a *per-backend empirical* gate requiring a real DD
    # engine (where d(delay)/d(OM) ~ e·x cancels the 1/e in d(OM)/d(EPS)). The
    # linear surrogate here treats OM linearly, so its gradient legitimately
    # grows ~1/e; that certification lives in the requires_jug §12.6 suite. Here
    # we certify the weaker, engine-agnostic property: finiteness for every
    # e > 0.
    _, ctx = _ctx()  # sample_all -> EPS1/EPS2/TASC sampled
    chart = ctx.physical_charts[0]
    engine = ctx.engine
    emap = ctx.engine_delta_map
    y = jnp.asarray(np.asarray(ctx.pulsar.residuals, dtype=float))
    var = jnp.asarray(np.asarray(ctx.pulsar.toaerrs, dtype=float) ** 2)
    names = list(ctx.plan.sampled)
    i1, i2 = names.index("EPS1"), names.index("EPS2")

    def neglogp(vals):
        delay = -engine.residual_delta_jax(emap.full_engine_delta(vals, jnp))
        r = y - delay
        return 0.5 * jnp.sum(r * r / var)

    grad = jax.grad(neglogp)
    hess = jax.hessian(neglogp)
    for scale in (1e-2, 1e-4, 1e-6, 1e-8):
        v = np.zeros(len(names))
        v[i1] = -chart.eps1_ref * (1 - scale)
        v[i2] = -chart.eps2_ref * (1 - scale)
        vj = jnp.asarray(v)
        val = float(neglogp(vj))
        g = np.asarray(grad(vj))
        Hd = np.diag(np.asarray(hess(vj)))
        assert np.isfinite(val)
        assert np.all(np.isfinite(g)) and np.all(np.isfinite(Hd))
    # (The composed-likelihood leapfrog-stability and no-1/e-blow-through checks
    # are the per-backend §12.6 requires_jug gates: a leapfrog on the linear
    # surrogate diverges precisely because it lacks the physical e-suppression.)


@pytest.mark.slow
def test_gram_conditioning_ratio():
    # §12.5 (best effort): record cond(W_s^T W_s) in the Laplace vs Kepler frame.
    # The low-e conditioning WIN is a property of the real DD posterior geometry
    # (the ω–T0 tube), not of a random linear surrogate — whose raw W_s columns
    # carry the 1/e Jacobian scale directly, so the naive Gram ratio here does
    # NOT show the benefit. The real ratio is measured on the J1640 fixture
    # (§12.6). We only record the surrogate value and assert it is finite.
    _, lap = _ctx()  # EPS1/EPS2/TASC
    _, kep = _ctx(binary_chart="off")  # ECC/OM/T0
    W_lap = np.asarray(lap.linearization.sampled_basis)
    W_kep = np.asarray(kep.linearization.sampled_basis)
    c_lap = float(np.linalg.cond(W_lap.T @ W_lap))
    c_kep = float(np.linalg.cond(W_kep.T @ W_kep))
    print(
        f"surrogate gram cond: Kepler={c_kep:.3e} Laplace={c_lap:.3e} "
        f"ratio(K/L)={c_kep / c_lap:.3e}"
    )
    assert np.isfinite(c_lap) and np.isfinite(c_kep)
