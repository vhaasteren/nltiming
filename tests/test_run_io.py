"""Tests for NLT artifact contract (run_meta, manifest, chain bundle)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nltiming import WhiteningConfig
from nltiming import TimingInference
from nltiming.run_io import (
    DISCOVERY_CHECKPOINT_NAME,
    DISCOVERY_FINAL_NAME,
    RunIOError,
    RunResults,
    build_run_manifest,
    derived_param_name,
    decode_physical,
    save_discovery_checkpoint,
)
from nltiming.engines.base import LinearModel
from nltiming.engines.jug import LinearizedJugEngine
from nltiming.nonlinear_timing_model import NonLinearTimingModel
from nltiming.space import ParameterSpace


class _Pulsar:
    def __init__(self, *, state_id: str = "artifact-token"):
        self.name = "J1111+1111"
        self.fitpars = ("F0", "F1")
        self._toas = np.linspace(0.0, 1.0, 5)
        self._residuals = np.zeros(5)
        self._toaerrs = np.full(5, 1.0e-6)
        self._freqs = np.full(5, 1400.0)
        self._flags = {"pta": np.array(["demo"] * 5, dtype="U8")}
        self._backend_flags = np.array(["demo"] * 5, dtype="U8")
        self._state_id = state_id
        model = LinearModel.from_design(
            fitpars=self.fitpars,
            design=np.column_stack([np.ones(5), np.linspace(-0.5, 0.5, 5)]),
            theta_exact={"F0": "100.0", "F1": "1.0"},
        )
        self._backend = LinearizedJugEngine.from_linear_model(model)

    @property
    def toas(self):
        return self._toas

    @property
    def residuals(self):
        return self._residuals

    @property
    def toaerrs(self):
        return self._toaerrs

    @property
    def freqs(self):
        return self._freqs

    @property
    def Mmat(self):
        return self._backend.design_matrix()

    @property
    def flags(self):
        return self._flags

    @property
    def backend_flags(self):
        return self._backend_flags

    def state_id(self):
        return self._state_id

    def pint_model(self):
        return object()

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend


@pytest.fixture
def pulsar():
    return _Pulsar()


@pytest.fixture
def ntm():
    return NonLinearTimingModel(
        engines="jug",
        whitening=WhiteningConfig(),
        inference=TimingInference.groups(delta_flat=["F0"]),
        name="timing",
    )


@pytest.fixture
def manifest(pulsar, ntm):
    return build_run_manifest(
        ntm.for_pulsar(pulsar),
        likelihood="discovery",
        sampler="numpyro-nuts",
        scenario="demo",
        checkpoint={"kind": "npz", "path": DISCOVERY_CHECKPOINT_NAME, "key_name": "x"},
        latent={"kind": "npz", "path": DISCOVERY_FINAL_NAME, "key_name": "x"},
    )


def test_run_meta_written_before_checkpoint(tmp_path, manifest):
    manifest.write(tmp_path)
    x = np.array([[0.1], [0.2]], dtype=float)
    save_discovery_checkpoint(tmp_path, x, manifest, final=False, n_target=4)
    assert (tmp_path / "nlt_run_meta.json").is_file()
    assert (tmp_path / DISCOVERY_CHECKPOINT_NAME).is_file()


def test_space_fingerprint_stable_across_save_load(tmp_path, manifest):
    fp1 = manifest.space.fingerprint()
    manifest.space.save(tmp_path / "nlt_parameter_space")
    loaded = ParameterSpace.load(tmp_path / "nlt_parameter_space")
    fp2 = loaded.fingerprint()
    assert fp1 == fp2


def test_space_fingerprint_changes_when_C_changes(manifest):
    fp_before = manifest.space.fingerprint()
    manifest.space.linear.C[0, 0] *= 1.1
    assert manifest.space.fingerprint() != fp_before


def test_context_digest_changes_with_pulsar_state(pulsar, ntm):
    fp_a = ntm.for_pulsar(pulsar).fingerprint()
    pulsar._state_id = "artifact-token-updated"
    fp_b = ntm.for_pulsar(pulsar).fingerprint()
    assert fp_a != fp_b


def test_bundle_refuses_wrong_space_same_names(tmp_path, manifest):
    manifest.write(tmp_path)
    arrays = np.load(str(tmp_path / "nlt_parameter_space") + ".npz")
    tampered_C = arrays["C"].copy()
    tampered_C[0, 0] *= 2.0
    np.savez(
        str(tmp_path / "nlt_parameter_space") + ".npz",
        C=tampered_C,
        z0=arrays["z0"],
    )
    with pytest.raises(RunIOError, match="digest mismatch"):
        RunResults.load(tmp_path, verify=True)


def test_bundle_force_downgrades_mismatch_to_warning(tmp_path, manifest):
    manifest.write(tmp_path)
    arrays = np.load(str(tmp_path / "nlt_parameter_space") + ".npz")
    tampered_C = arrays["C"].copy()
    tampered_C[0, 0] *= 2.0
    np.savez(
        str(tmp_path / "nlt_parameter_space") + ".npz",
        C=tampered_C,
        z0=arrays["z0"],
    )
    with pytest.warns(UserWarning, match="digest mismatch"):
        RunResults.load(tmp_path, verify=True, force=True)


def test_decode_requires_no_pulsar_rebuild(tmp_path, manifest):
    manifest.write(tmp_path)
    x = np.array([[0.05], [0.15]], dtype=float)
    save_discovery_checkpoint(tmp_path, x, manifest, final=True, n_target=2)
    bundle = RunResults.load(tmp_path)
    loaded_x = bundle.load_latent()
    np.testing.assert_allclose(loaded_x, x)
    display = bundle.load_display()
    assert "F1" in display


def test_discovery_checkpoint_contains_display_deterministics(tmp_path, manifest):
    manifest.write(tmp_path)
    x = np.array([[0.1], [0.2]], dtype=float)
    save_discovery_checkpoint(tmp_path, x, manifest, final=False, n_target=4)
    data = np.load(tmp_path / DISCOVERY_CHECKPOINT_NAME)
    prefix = manifest.name_stem
    assert derived_param_name(prefix, "F1", "display") in data.files


def test_load_display_prefers_latent_over_checkpoint(tmp_path, manifest):
    manifest.write(tmp_path)
    x_ckpt = np.array([[0.1]], dtype=float)
    x_final = np.array([[0.9]], dtype=float)
    save_discovery_checkpoint(tmp_path, x_ckpt, manifest, final=False, n_target=2)
    save_discovery_checkpoint(tmp_path, x_final, manifest, final=True, n_target=2)
    bundle = RunResults.load(tmp_path)
    prefix = manifest.name_stem
    display_key = derived_param_name(prefix, "F1", "display")
    final_data = np.load(tmp_path / DISCOVERY_FINAL_NAME)[display_key]
    loaded = bundle.load_display()["F1"]
    np.testing.assert_allclose(loaded, final_data)


def test_load_latent_prefers_chain_txt_over_npz(tmp_path, pulsar, ntm):
    ndim = len(ntm.for_pulsar(pulsar).sampled)
    manifest = build_run_manifest(
        ntm.for_pulsar(pulsar),
        likelihood="enterprise",
        sampler="ptmcmc",
        chain_layout={
            "kind": "ptmcmc",
            "file": "chains/chain_1.txt",
            "columns": list(range(ndim)),
            "coord": "x",
        },
        latent={"kind": "npz", "path": "enterprise_x.npz", "key_name": "x"},
    )
    manifest.write(tmp_path)
    chains_dir = tmp_path / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    chain_x = np.array([[1.0], [2.0], [3.0]], dtype=float)
    meta = np.column_stack(
        [
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
        ]
    )
    np.savetxt(chains_dir / "chain_1.txt", np.hstack([chain_x, meta]))
    npz_x = np.array([[9.0], [8.0]], dtype=float)
    np.savez(tmp_path / "enterprise_x.npz", x=npz_x)
    bundle = RunResults.load(tmp_path)
    np.testing.assert_allclose(bundle.load_latent(), chain_x)
    np.testing.assert_allclose(bundle.load_latent(prefer_npz=True), npz_x)


def test_load_latent_returns_raw_chain_including_burn(tmp_path, pulsar, ntm):
    ndim = len(ntm.for_pulsar(pulsar).sampled)
    manifest = build_run_manifest(
        ntm.for_pulsar(pulsar),
        likelihood="enterprise",
        sampler="ptmcmc",
        chain_layout={
            "kind": "ptmcmc",
            "file": "chains/chain_1.txt",
            "columns": list(range(ndim)),
            "coord": "x",
        },
    )
    manifest.write(tmp_path)
    chains_dir = tmp_path / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    chain_x = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float)
    meta = np.zeros((4, 4))
    np.savetxt(chains_dir / "chain_1.txt", np.hstack([chain_x, meta]))
    bundle = RunResults.load(tmp_path)
    loaded = bundle.load_latent()
    assert loaded.shape[0] == 4


def test_enterprise_run_meta_omits_latent_by_default(tmp_path, pulsar, ntm):
    ndim = len(ntm.for_pulsar(pulsar).sampled)
    manifest = build_run_manifest(
        ntm.for_pulsar(pulsar),
        likelihood="enterprise",
        sampler="ptmcmc",
        chain_layout={
            "kind": "ptmcmc",
            "file": "chains/chain_1.txt",
            "columns": list(range(ndim)),
            "coord": "x",
        },
    )
    manifest.write(tmp_path)
    run_meta = json.loads((tmp_path / "nlt_run_meta.json").read_text())
    assert "latent" not in run_meta["run_products"]


def test_ptmcmc_chain_decodes_with_run_meta(tmp_path, pulsar, ntm):
    ndim = len(ntm.for_pulsar(pulsar).sampled)
    manifest = build_run_manifest(
        ntm.for_pulsar(pulsar),
        likelihood="enterprise",
        sampler="ptmcmc",
        chain_layout={
            "kind": "ptmcmc",
            "file": "chains/chain_1.txt",
            "columns": list(range(ndim)),
            "coord": "x",
        },
    )
    manifest.write(tmp_path)
    chains_dir = tmp_path / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    x = np.array([[0.01], [0.02]], dtype=float)
    meta = np.zeros((2, 4))
    np.savetxt(chains_dir / "chain_1.txt", np.hstack([x, meta]))
    bundle = RunResults.load(tmp_path)
    phys = bundle.load_display()
    assert "F1" in phys
    assert phys["F1"].shape == (2,)


def test_posterior_applies_burn_and_thin_consistently(tmp_path, pulsar, ntm):
    ndim = len(ntm.for_pulsar(pulsar).sampled)
    manifest = build_run_manifest(
        ntm.for_pulsar(pulsar),
        likelihood="enterprise",
        sampler="ptmcmc",
        chain_layout={
            "kind": "ptmcmc",
            "file": "chains/chain_1.txt",
            "columns": list(range(ndim)),
            "coord": "x",
        },
    )
    manifest.write(tmp_path)
    chains_dir = tmp_path / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.01, 0.08, 8).reshape(-1, 1)
    meta = np.zeros((8, 4))
    np.savetxt(chains_dir / "chain_1.txt", np.hstack([x, meta]))

    bundle = RunResults.load(tmp_path)

    latent = bundle.latent(burn=4, thin=2)
    assert latent.shape == (2, 1)
    np.testing.assert_allclose(latent[:, 0], x[4::2, 0])

    phys = bundle.posterior(burn=4, thin=2)
    expected = bundle.space.to_physical(latent, units="display")
    np.testing.assert_allclose(phys["F1"], expected["F1"])

    frac = bundle.latent(burn=0.5)
    assert frac.shape == (4, 1)

    with pytest.raises(ValueError, match="fractional burn"):
        bundle.latent(burn=1.5)
    with pytest.raises(ValueError, match="thin"):
        bundle.latent(thin=0)


def test_truths_return_reference_values(tmp_path, manifest):
    manifest.write(tmp_path)
    x = np.array([[0.1], [0.2]], dtype=float)
    save_discovery_checkpoint(tmp_path, x, manifest, final=True, n_target=2)
    bundle = RunResults.load(tmp_path)
    truths = bundle.truths()
    # F1 reference is 1.0 in the pulsar's theta_exact; zero delta must decode to it.
    assert truths == {"F1": pytest.approx(1.0)}


def test_run_meta_schema_is_v3_and_code_block_names_owning_package(tmp_path, manifest):
    manifest.write(tmp_path)
    run_meta = json.loads((tmp_path / "nlt_run_meta.json").read_text())
    assert run_meta["schema"] == "nlt-run-meta-v3"
    assert run_meta["code"]["package"] == "nltiming"
    assert run_meta["code"]["version"]


def test_display_units_recorded_without_pint_model(tmp_path, manifest):
    manifest.write(tmp_path)
    run_meta = json.loads((tmp_path / "nlt_run_meta.json").read_text())
    assert run_meta["display_units"]["F1"] == "native"
    assert run_meta["native_units"]["F1"] == "native"


def test_decode_physical_keys(manifest):
    x = np.array([[0.1], [0.2]], dtype=float)
    out = decode_physical(manifest.space, x, manifest.name_stem)
    prefix = manifest.name_stem
    assert derived_param_name(prefix, "F1", "native") in out
    assert derived_param_name(prefix, "F1", "display") in out


def test_arviz_export_has_fingerprint_attrs(tmp_path, manifest):
    pytest.importorskip("arviz")
    manifest.write(tmp_path)
    x = np.array([[0.1], [0.2]], dtype=float)
    save_discovery_checkpoint(tmp_path, x, manifest, final=True, n_target=2)
    bundle = RunResults.load(tmp_path)
    idata = bundle.to_arviz()
    assert idata.attrs["space_digest"] == manifest.space_fingerprint
    assert idata.attrs["context_digest"] == manifest.context_digest
