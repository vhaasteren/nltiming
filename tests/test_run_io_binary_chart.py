"""PR-4 (§12.4): binary-chart manifest section + derived Kepler decode."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("jug")

from nltiming import TimingInference  # noqa: E402
from nltiming.nonlinear_timing_model import NonLinearTimingModel  # noqa: E402
from nltiming.physical_charts import kepler_from_laplace_vec  # noqa: E402
from nltiming.run_io import RunResults, derived_kepler_columns  # noqa: E402

from test_physical_charts_context import _BinaryPulsar  # noqa: E402


def _conditioned(binary_chart="auto", inference=None):
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=inference or TimingInference.sample_all(),
        binary_chart=binary_chart,
        name="t",
    )
    base = ntm.for_pulsar(_BinaryPulsar(), condition=False)
    return base.with_transport()  # identity layer -> conditioned


_GROUP_KEYS = {
    "suffix",
    "enabled",
    "reason",
    "engine_names",
    "sample_names",
    "dispositions",
    "e_ref",
    "pb_ref",
    "pb_fitpar",
    "theta_ref_engine",
    "theta_ref_sample",
    "xe2_us",
    "domain",
    "seam_guard",
    "origin_guard",
    "capability_source",
}


def test_manifest_binary_chart_section():
    ctx = _conditioned()
    man = ctx.run_manifest(likelihood="discovery", sampler="nuts")
    assert man.binary_chart["policy"]["prior"] == "sampling_frame"
    assert man.binary_chart["policy"]["default_prior_package"] == "nlt-eps-wls-boxes-v1"
    group = man.binary_chart["groups"][0]
    assert set(group) == _GROUP_KEYS
    assert group["enabled"] is True
    assert group["sample_names"] == ["EPS1", "EPS2", "TASC"]
    # sections() carries a digest; run_meta() carries the raw view.
    sec = man.sections()["binary_chart"]
    assert sec["digest"].startswith("sha256:")
    rm = man.run_meta()
    assert rm["binary_chart"]["policy"]["prior"] == "sampling_frame"

    # mode-off: schema-stable, empty groups.
    ctx_off = _conditioned(binary_chart="off")
    man_off = ctx_off.run_manifest(likelihood="discovery", sampler="nuts")
    assert man_off.binary_chart["policy"]["mode"] == "off"
    assert man_off.binary_chart["groups"] == []
    assert man_off.sections()["binary_chart"]["digest"].startswith("sha256:")


def _synth_samples(ctx, n=64, seed=0, near_seam=False):
    """Absolute sampling-frame draws around the chart reference."""
    chart = ctx.physical_charts[0]
    rng = np.random.default_rng(seed)
    if near_seam:
        # Sweep omega across the reference branch out toward +/- pi.
        phi = np.linspace(-np.pi + 1e-6, np.pi - 1e-6, n)
        rho = chart.e_ref
        eps1 = rho * np.sin(phi)  # NOTE: eps1=e*sin(om), eps2=e*cos(om)
        eps2 = rho * np.cos(phi)
    else:
        eps1 = chart.eps1_ref + rng.normal(scale=1e-4, size=n)
        eps2 = chart.eps2_ref + rng.normal(scale=1e-4, size=n)
    tasc = float(chart.tasc_ref_str) + rng.normal(scale=1e-4, size=n)
    return {
        chart.sample_names[0]: eps1,
        chart.sample_names[1]: eps2,
        chart.sample_names[2]: tasc,
    }


def test_posterior_derived_kepler_columns():
    ctx = _conditioned()
    chart = ctx.physical_charts[0]
    man = ctx.binary_chart_manifest()
    samples = _synth_samples(ctx, seed=1)

    derived = derived_kepler_columns(samples, man)
    live = chart.decode(samples)
    ecc_n, om_n, t0_n = chart.engine_names
    # OM and T0 are pure delta-form in both copies -> bit-for-bit identical.
    np.testing.assert_array_equal(derived[om_n], live[om_n])
    np.testing.assert_array_equal(derived[t0_n], live[t0_n])
    # ECC: derived uses np.hypot while decode routes e_ref + (|eps| - e_ref);
    # the two agree to machine precision (last-ULP hypot-vs-sqrt difference).
    np.testing.assert_allclose(derived[ecc_n], live[ecc_n], rtol=1e-14, atol=0)

    # truths() appends the engine-frame references (OM via OM_normalized).
    rr = RunResults(
        run_dir=ctx.pulsar and __import__("pathlib").Path("."),
        run_meta={"sampled": list(ctx.space.names), "binary_chart": man},
        space=ctx.space,
    )
    truths = rr.truths()
    assert truths[ecc_n] == pytest.approx(float(chart.ecc_ref_str))
    assert truths[om_n] == pytest.approx(float(chart.om_ref_norm_str))
    assert truths[t0_n] == pytest.approx(float(chart.t0_ref_str))


def test_posterior_derived_sampled_pb_branch():
    # With a sampled PB column present, the decode uses the per-draw PB.
    ctx = _conditioned()
    chart = ctx.physical_charts[0]
    man = ctx.binary_chart_manifest()
    samples = _synth_samples(ctx, seed=3)
    rng = np.random.default_rng(4)
    samples[chart.pb_name] = chart.pb_ref + rng.normal(scale=1e-5, size=64)
    derived = derived_kepler_columns(samples, man)
    live = chart.decode(
        {k: v for k, v in samples.items() if k in chart.sample_names},
        dependency={chart.pb_name: samples[chart.pb_name]},
    )
    np.testing.assert_array_equal(
        derived[chart.engine_names[2]], live[chart.engine_names[2]]
    )


def test_near_seam_uses_engine_branch():
    ctx = _conditioned()
    chart = ctx.physical_charts[0]
    man = ctx.binary_chart_manifest()
    samples = _synth_samples(ctx, n=64, near_seam=True)
    derived = derived_kepler_columns(samples, man)
    om = derived[chart.engine_names[1]]
    t0 = derived[chart.engine_names[2]]
    # The reference-local branch lets OM run continuously OUTSIDE [0, 360)
    # near the seam; the global normalization would wrap it.
    glob_e, glob_om, glob_t0 = kepler_from_laplace_vec(
        samples[chart.sample_names[0]],
        samples[chart.sample_names[1]],
        samples[chart.sample_names[2]],
        chart.pb_ref,
    )
    # Somewhere in the sweep the two branches disagree (integer 360/PB jump).
    assert np.any(np.abs(om - glob_om) > 1.0)
    assert np.all(np.isfinite(t0))


def test_posterior_decode_partial_marginalization():
    # Headline config: TASC sampled, EPS1/EPS2 z-marginalized -> NO derived cols.
    ctx = _conditioned(inference=TimingInference.groups(z_prior=["ECC", "OM"]))
    assert ctx.physical_charts, "chart should activate (T0 sampled)"
    man = ctx.binary_chart_manifest()
    chart = ctx.physical_charts[0]
    # Only the TASC column is present in the products.
    samples = {
        chart.sample_names[2]: np.linspace(-1e-4, 1e-4, 16) + float(chart.tasc_ref_str)
    }
    derived = derived_kepler_columns(samples, man)
    assert derived == {}  # no KeyError, no fabricated columns
