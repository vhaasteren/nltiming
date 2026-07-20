"""Self-contained NLT run I/O: run metadata + serialized ParameterSpace.

A valid NLT read requires ONLY on-disk run products (nlt_run_meta.json,
nlt_parameter_space.{json,npz}) plus raw sampler output. It MUST NOT require an
Enterprise PTA, Discovery model, MetaPulsar construction, or a PINT reload.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .space import ParameterSpace, coord_for_static_layer
from .units import units_map

RUN_META_SCHEMA = "nlt-run-meta-v3"
RUN_META_FILENAME = "nlt_run_meta.json"
PARAMETER_SPACE_STEM = "nlt_parameter_space"
ENTERPRISE_NPZ_NAME = "enterprise_x.npz"
DISCOVERY_FINAL_NAME = "discovery_x.npz"
DISCOVERY_CHECKPOINT_NAME = "discovery_x_checkpoint.npz"
_NATIVE_SUFFIX = "_theta_native"
_DISPLAY_SUFFIX = "_theta_display"


def _run_meta_tempo2_native(ntm) -> str | dict[str, Any] | None:
    # Record the resolved mode (§18): omitted tempo2_native is fixed_state_stripped.
    resolved = getattr(ntm, "resolved_tempo2_native", None)
    if resolved is not None:
        return resolved
    value = getattr(ntm, "tempo2_native", None)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value):
        return asdict(value)
    return str(value)


class RunIOError(RuntimeError):
    """Raised when NLT run products are missing, mismatched, or unreadable."""


def _slice_draws(x: np.ndarray, *, burn: int | float, thin: int) -> np.ndarray:
    """Apply burn-in (count or leading fraction) and thinning to sample rows."""
    n = int(x.shape[0])
    if isinstance(burn, float):
        if not (0.0 <= burn < 1.0):
            raise ValueError("fractional burn must be in [0, 1)")
        burn = int(burn * n)
    burn = int(burn)
    if burn < 0 or burn > n:
        raise ValueError(f"burn={burn} out of range for {n} draws")
    thin = int(thin)
    if thin < 1:
        raise ValueError("thin must be >= 1")
    return x[burn::thin]


def _code_block(git_commit: str | None) -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    package = __package__.split(".")[0]
    try:
        pkg_version = version(package)
    except PackageNotFoundError:
        pkg_version = "0.0.0"
    return {"package": package, "version": pkg_version, "git_commit": git_commit}


def _section_digest(content: Mapping[str, Any]) -> str:
    """Stable digest of a manifest section's canonical serialization (§7.4)."""
    import hashlib

    payload = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derived_param_name(name_stem: str, fitpar: str, units: str) -> str:
    return f"{name_stem}_{fitpar}_theta_{units}"


def decode_physical(
    space: ParameterSpace,
    x: np.ndarray,
    name_stem: str,
    *,
    units: Sequence[str] = ("native", "display"),
) -> dict[str, np.ndarray]:
    """Decode latent draws x into {name_stem}_{fitpar}_theta_{units} arrays."""
    x = np.asarray(x, dtype=float)
    out: dict[str, np.ndarray] = {}
    for unit_mode in units:
        phys = space.to_physical(x, units=unit_mode)
        for name, arr in phys.items():
            out[derived_param_name(name_stem, name, unit_mode)] = np.asarray(
                arr, dtype=float
            )
    return out


@dataclass
class RunManifest:
    """Write-side snapshot of everything needed to decode a run's samples."""

    space: ParameterSpace
    likelihood: str
    sampler: str
    pulsar_name: str
    latent_name: str
    sampled: tuple[str, ...]
    coord: str
    static_layer: str
    design_matrix_method: str
    engines: dict[str, str]
    native_units: dict[str, str]
    display_units: dict[str, str]
    context_digest: str
    space_fingerprint: str
    name_stem: str
    tempo2_native: str | dict[str, Any] | None = None
    prior_override_policy: str | None = None
    scenario: str | None = None
    latent: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    chain_layout: dict[str, Any] | None = None
    git_commit: str | None = None
    metric_source: dict[str, Any] | None = None
    transport: dict[str, Any] | None = None

    @property
    def latent_decodable(self) -> bool:
        """Whether a saved latent chain is independently decodable (§7.3).

        A static-affine transport is always latent-decodable; a dynamic joint
        transport is not (its map depends on sampled hyperparameters).
        """
        if self.transport is None:
            return True
        return bool(self.transport.get("latent_decodable", True))

    def _chains_section(self) -> dict[str, Any]:
        layout: dict[str, Any] = {}
        for key, spec in (
            ("latent", self.latent),
            ("checkpoint", self.checkpoint),
            ("chain_layout", self.chain_layout),
        ):
            if spec is not None:
                layout[key] = spec
        content = {
            "sample_coord": self.coord,
            "latent_name": self.latent_name,
            "latent_decodable": self.latent_decodable,
            "derived_suffixes": {"native": _NATIVE_SUFFIX, "display": _DISPLAY_SUFFIX},
            "layout": layout,
        }
        return {**content, "digest": _section_digest(content)}

    def sections(self) -> dict[str, Any]:
        """Per-section manifest with one digest per section (§7.4).

        parameter_space, context, metric_source and transport digests come from
        the live objects' fingerprints; chains is digested from its own layout.
        """
        parameter_space = {
            "json": f"{PARAMETER_SPACE_STEM}.json",
            "arrays": f"{PARAMETER_SPACE_STEM}.npz",
            "digest": self.space_fingerprint,
        }
        transport = None if self.transport is None else dict(self.transport)
        metric_source = None if self.metric_source is None else dict(self.metric_source)
        return {
            "parameter_space": parameter_space,
            "context": {"digest": self.context_digest},
            "metric_source": metric_source,
            "transport": transport,
            "chains": self._chains_section(),
        }

    def run_meta(self) -> dict[str, Any]:
        run_products: dict[str, Any] = {"parameter_space_stem": PARAMETER_SPACE_STEM}
        if self.latent is not None:
            run_products["latent"] = self.latent
        if self.checkpoint is not None:
            run_products["checkpoint"] = self.checkpoint
        if self.chain_layout is not None:
            run_products["chain_layout"] = self.chain_layout
        sections = self.sections()
        return {
            "schema": RUN_META_SCHEMA,
            "likelihood": self.likelihood,
            "sampler": self.sampler,
            "pulsar": self.pulsar_name,
            "scenario": self.scenario,
            "sample_coord": self.coord,
            "latent_name": self.latent_name,
            "sampled": list(self.sampled),
            "static_layer": self.static_layer,
            "design_matrix_method": self.design_matrix_method,
            "engines": dict(self.engines),
            "tempo2_native": self.tempo2_native,
            "prior_override_policy": self.prior_override_policy,
            "native_units": self.native_units,
            "display_units": self.display_units,
            "name_stem": self.name_stem,
            "latent_decodable": self.latent_decodable,
            "unit_suffixes": {
                "native": _NATIVE_SUFFIX,
                "display": _DISPLAY_SUFFIX,
            },
            # Back-compatible top-level views (also carried inside `sections`).
            "parameter_space": sections["parameter_space"],
            "context_digest": self.context_digest,
            "sections": sections,
            "run_products": run_products,
            "code": _code_block(self.git_commit),
        }

    def write(self, run_dir: str | Path, *, force: bool = False) -> Path:
        """Write run metadata + parameter space. Call before the first sample.

        Refuses to overwrite an existing *incompatible* run (different context
        or parameter-space digest) unless ``force=True`` (§7.4 rule 3). Writing
        an identical manifest, or into a fresh directory, is always allowed.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        existing_path = run_dir / RUN_META_FILENAME
        if existing_path.is_file() and not force:
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = None
            if existing is not None:
                incompatible = existing.get(
                    "context_digest"
                ) != self.context_digest or (
                    existing.get("parameter_space", {}).get("digest")
                    != self.space_fingerprint
                )
                if incompatible:
                    raise RunIOError(
                        f"refusing to overwrite an incompatible run in {run_dir}: "
                        f"existing context/space digests differ from this manifest. "
                        f"Pass force=True to overwrite (§7.4)."
                    )
        self.space.save(run_dir / PARAMETER_SPACE_STEM)
        existing_path.write_text(
            json.dumps(self.run_meta(), indent=2), encoding="utf-8"
        )
        return existing_path


def build_run_manifest(
    ctx,
    *,
    likelihood: str,
    sampler: str,
    scenario: str | None = None,
    latent: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
    chain_layout: dict[str, Any] | None = None,
    git_commit: str | None = None,
    dynamic_transport=None,
) -> RunManifest:
    """Build a RunManifest snapshot from a ``TimingContext``.

    For a joint full-basis run, pass ``dynamic_transport`` (a
    ``DynamicTransportRecord``); its section replaces the static transport and
    marks the run non-latent-decodable (Track J, §7.3). In that mode the
    nltiming static affine layer must be identity (§5.5), which is asserted.
    """
    ntm = ctx.model
    pulsar = ctx.pulsar
    space = ctx.space
    sampled = tuple(ctx.sampled)
    keys = ctx.timing_param_keys()
    if not keys:
        raise RunIOError(f"pulsar {pulsar.name!r} has no sampled timing parameters")
    pint_model = pulsar.pint_model()
    tempo2_native = _run_meta_tempo2_native(ntm)
    metric_source = ctx.metric.provenance() if ctx.metric is not None else None
    transport = None
    if dynamic_transport is not None:
        from .metric import assert_static_layer_identity

        assert_static_layer_identity(space, context="joint full-basis run manifest")
        # The dynamic transport section owns the joint provenance; the static
        # timing metric is identity and not the whitening authority here.
        metric_source = {
            "reference_noise": dynamic_transport.reference_noise,
            "source": "dynamic_transport",
            "source_description": "joint sampled-coefficient transport (Track J)",
            "approximate": False,
            "digest": dynamic_transport.transport_digest,
        }
        transport = dynamic_transport.section()
        transport["digest"] = dynamic_transport.fingerprint()
    elif ctx.transport is not None:
        transport = ctx.transport.section()
        transport["digest"] = ctx.transport.fingerprint()
    return RunManifest(
        space=space,
        likelihood=likelihood,
        sampler=sampler,
        pulsar_name=pulsar.name,
        latent_name=keys[0],
        sampled=sampled,
        coord=coord_for_static_layer(ntm.static_layer),
        static_layer=ntm.static_layer,
        design_matrix_method=ntm.design_matrix_method,
        engines=dict(ntm.engines),
        tempo2_native=tempo2_native,
        prior_override_policy=ntm.prior_override_policy,
        native_units=units_map(sampled, pint_model, kind="native"),
        display_units=units_map(sampled, pint_model, kind="display"),
        context_digest=ctx.fingerprint(),
        space_fingerprint=space.fingerprint(),
        name_stem=ctx.name_stem,
        scenario=scenario,
        latent=latent,
        checkpoint=checkpoint,
        chain_layout=chain_layout,
        git_commit=git_commit,
        metric_source=metric_source,
        transport=transport,
    )


def save_discovery_checkpoint(
    run_dir: str | Path,
    x: np.ndarray,
    manifest: RunManifest,
    *,
    final: bool,
    n_target: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write Discovery latent x + native/display derived values to an npz."""
    if not manifest.latent_decodable:
        raise RunIOError(
            "save_discovery_checkpoint recomputes physical values by passing x "
            "through ParameterSpace and cannot be used for a dynamic transport "
            "manifest; use save_dynamic_checkpoint for joint xi runs (§7.3)."
        )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=float)
    payload: dict[str, Any] = {
        "x": x,
        "n_draws": int(x.shape[0]),
        "latent_name": manifest.latent_name,
        "space_digest": manifest.space_fingerprint,
        "context_digest": manifest.context_digest,
    }
    if n_target is not None:
        payload["n_target"] = int(n_target)
    payload.update(decode_physical(manifest.space, x, manifest.name_stem))
    if extra:
        payload.update({k: np.asarray(v) for k, v in extra.items()})
    name = DISCOVERY_FINAL_NAME if final else DISCOVERY_CHECKPOINT_NAME
    out = run_dir / name
    np.savez(out, **payload)
    return out


DYNAMIC_FINAL_NAME = "dynamic_xi.npz"
DYNAMIC_CHECKPOINT_NAME = "dynamic_xi_checkpoint.npz"


def save_dynamic_checkpoint(
    run_dir: str | Path,
    manifest: RunManifest,
    *,
    xi: np.ndarray,
    final: bool,
    hyperparameters: Mapping[str, np.ndarray] | None = None,
    timing_delta: np.ndarray | None = None,
    theta_native: Mapping[str, np.ndarray] | None = None,
    theta_display: Mapping[str, np.ndarray] | None = None,
    log_density: np.ndarray | None = None,
    n_target: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically persist a joint (dynamic-transport) checkpoint (Track J, §7.3).

    Unlike the static writer, this cannot recompute physical values by passing
    ``xi`` through ``ParameterSpace`` (the joint map depends on sampled
    hyperparameters). Callers must supply the canonical decoded timing values.
    A *final* checkpoint is refused if those canonical physical values are
    absent — an incomplete dynamic checkpoint must never be promoted to final.
    """
    if manifest.latent_decodable:
        raise RunIOError(
            "save_dynamic_checkpoint requires a dynamic (non-latent-decodable) "
            "transport manifest; use save_discovery_checkpoint for static runs."
        )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    xi = np.asarray(xi, dtype=float)

    has_physical = bool(theta_native) or bool(theta_display)
    if final and not has_physical:
        raise RunIOError(
            "refusing to write a final dynamic checkpoint without canonical "
            "decoded physical values (theta_native/theta_display): xi alone is "
            "not decodable (§7.3). Provide the per-draw decoded timing values."
        )

    payload: dict[str, Any] = {
        "xi": xi,
        "n_draws": int(xi.shape[0]),
        "latent_name": manifest.latent_name,
        "space_digest": manifest.space_fingerprint,
        "context_digest": manifest.context_digest,
        "transport_digest": (manifest.transport or {}).get("digest", ""),
    }
    if n_target is not None:
        payload["n_target"] = int(n_target)
    if timing_delta is not None:
        payload["timing_delta"] = np.asarray(timing_delta, dtype=float)
    if log_density is not None:
        payload["log_density"] = np.asarray(log_density, dtype=float)
    for name, arr in dict(hyperparameters or {}).items():
        payload[f"hyper_{name}"] = np.asarray(arr)
    for fitpar, arr in dict(theta_native or {}).items():
        payload[derived_param_name(manifest.name_stem, fitpar, "native")] = np.asarray(
            arr, dtype=float
        )
    for fitpar, arr in dict(theta_display or {}).items():
        payload[derived_param_name(manifest.name_stem, fitpar, "display")] = np.asarray(
            arr, dtype=float
        )
    if extra:
        payload.update({k: np.asarray(v) for k, v in extra.items()})
    name = DYNAMIC_FINAL_NAME if final else DYNAMIC_CHECKPOINT_NAME
    out = run_dir / name
    np.savez(out, **payload)
    return out


@dataclass
class RunResults:
    """Read-side decoder over existing sampler output. No live model needed."""

    run_dir: Path
    run_meta: dict[str, Any]
    space: ParameterSpace

    @classmethod
    def load(
        cls, run_dir: str | Path, *, verify: bool = True, force: bool = False
    ) -> "RunResults":
        run_dir = Path(run_dir)
        run_meta_path = run_dir / RUN_META_FILENAME
        if not run_meta_path.is_file():
            raise RunIOError(f"missing {RUN_META_FILENAME} in {run_dir}")
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        schema = run_meta.get("schema")
        if schema != RUN_META_SCHEMA:
            raise RunIOError(
                f"unsupported run-metadata schema {schema!r} in {run_dir}; this "
                f"nltiming supports {RUN_META_SCHEMA!r}. Older 'nlt-sidecar-v2' "
                "runs are readable only by their pinned production nltiming "
                "commit; re-run or migrate to load them here (§8.5)."
            )
        stem = run_meta["run_products"]["parameter_space_stem"]
        space = ParameterSpace.load(run_dir / stem)
        results = cls(run_dir=run_dir, run_meta=run_meta, space=space)
        if verify:
            results._verify(force=force)
        return results

    def _fail(self, section: str, msg: str, *, force: bool) -> None:
        """Raise (or warn under ``force``) with the diverging section named."""
        full = f"{section} section: {msg} (in {self.run_dir})"
        if force:
            warnings.warn(full, stacklevel=3)
        else:
            raise RunIOError(full)

    def _verify(self, *, force: bool) -> None:
        """Recompute and compare per-section digests, naming any divergence (§7.4)."""
        run_meta = self.run_meta
        sections = run_meta.get("sections", {})

        # parameter_space: the on-disk ParameterSpace must match its digest.
        expected_ps = run_meta.get("parameter_space", {}).get("digest")
        actual_ps = self.space.fingerprint()
        if expected_ps is not None and actual_ps != expected_ps:
            self._fail(
                "parameter_space",
                f"digest mismatch run-metadata={expected_ps} disk={actual_ps}",
                force=force,
            )

        # dimensionality / sampled-name consistency across sections.
        sampled = tuple(run_meta.get("sampled", ()))
        if sampled and tuple(self.space.names) != sampled:
            self._fail(
                "parameter_space",
                f"sampled names {sampled} disagree with space {self.space.names}",
                force=force,
            )

        # chains: the stored layout digest must match its own content, and any
        # digests embedded in the saved sample arrays must match the manifest.
        chains = sections.get("chains")
        if chains is not None:
            stored = chains.get("digest")
            content = {k: v for k, v in chains.items() if k != "digest"}
            if stored is not None and _section_digest(content) != stored:
                self._fail(
                    "chains", "layout digest does not match content", force=force
                )
        self._verify_embedded_array_digests(force=force)

        # transport: the stored section digest must match its own content.
        transport = sections.get("transport")
        if transport is not None:
            stored = transport.get("digest")
            content = {k: v for k, v in transport.items() if k != "digest"}
            if stored is not None and _section_digest(content) != stored:
                self._fail(
                    "transport", "digest does not match transport content", force=force
                )

    def _verify_embedded_array_digests(self, *, force: bool) -> None:
        run_products = self.run_meta.get("run_products", {})
        expected_space = self.run_meta.get("parameter_space", {}).get("digest")
        expected_context = self.run_meta.get("context_digest")
        for key in ("latent", "checkpoint"):
            spec = run_products.get(key)
            if spec is None or spec.get("kind") != "npz":
                continue
            path = self.run_dir / spec["path"]
            if not path.is_file():
                continue
            data = np.load(path)
            files = set(data.files)
            if "space_digest" in files and expected_space is not None:
                if str(data["space_digest"]) != expected_space:
                    self._fail(
                        "chains",
                        f"{spec['path']} space_digest does not match the manifest",
                        force=force,
                    )
            if "context_digest" in files and expected_context is not None:
                if str(data["context_digest"]) != expected_context:
                    self._fail(
                        "chains",
                        f"{spec['path']} context_digest does not match the manifest",
                        force=force,
                    )

    def assert_consistent_with(self, ctx) -> None:
        """Verify a live context matches this run's persisted decoder (§7.5).

        Raises an actionable :class:`RunIOError` naming the first section that
        diverges. Use before feeding a saved latent point through a rebuilt
        likelihood; for pure decoding use ``run.space`` directly and never
        rebuild a decoder.
        """
        run_meta = self.run_meta
        checks = [
            ("parameter_space", self.space.fingerprint(), ctx.space.fingerprint()),
            (
                "context",
                run_meta.get("context_digest"),
                ctx.fingerprint(),
            ),
            ("chains", run_meta.get("sample_coord"), ctx.coord),
        ]
        for section, expected, actual in checks:
            if expected is not None and expected != actual:
                raise RunIOError(
                    f"{section} section diverged: run={expected} live={actual}. "
                    f"Run was sampled with coord={run_meta.get('sample_coord')!r}, "
                    f"static_layer={run_meta.get('static_layer')!r}. Decode with "
                    "run.space; rebuild a live context only for fresh "
                    "calculations (§7.5)."
                )
        run_sampled = tuple(run_meta.get("sampled", ()))
        if run_sampled and run_sampled != tuple(ctx.sampled):
            raise RunIOError(
                f"context section diverged: sampled parameters run={run_sampled} "
                f"live={tuple(ctx.sampled)}. Decode with run.space (§7.5)."
            )
        # Metric-source and transport identity, when both sides carry them.
        sections = run_meta.get("sections", {})
        run_metric = (sections.get("metric_source") or {}).get("digest")
        if run_metric is not None and ctx.metric is not None:
            if run_metric != ctx.metric.fingerprint():
                raise RunIOError(
                    "metric_source section diverged: the live metric provenance "
                    "differs from the run's. Decode with run.space (§7.5)."
                )
        run_transport = (sections.get("transport") or {}).get("digest")
        if run_transport is not None and ctx.transport is not None:
            if run_transport != ctx.transport.fingerprint():
                raise RunIOError(
                    "transport section diverged: the live transport differs from "
                    "the run's. Decode with run.space (§7.5)."
                )

    @property
    def latent_decodable(self) -> bool:
        """Whether the saved latent chain is independently decodable (§7.3)."""
        return bool(self.run_meta.get("latent_decodable", True))

    @property
    def coord(self) -> str:
        return self.run_meta["sample_coord"]

    def load_latent(self, *, prefer_npz: bool = False) -> np.ndarray:
        """Load latent sample rows. Returns raw chain (burn-in included for PTMCMC).

        Callers must apply burn/thin slicing themselves (e.g. x[burn:]).
        """
        run_products = self.run_meta["run_products"]

        def _load_ptmcmc(spec: dict[str, Any]) -> np.ndarray:
            chain_path = self.run_dir / spec["file"]
            if not chain_path.is_file():
                raise RunIOError(f"missing PTMCMC chain file {chain_path}")
            chain = np.loadtxt(chain_path)
            cols = list(spec["columns"])
            return np.asarray(chain[:, cols], dtype=float)

        def _load_npz(spec: dict[str, Any]) -> np.ndarray:
            data = np.load(self.run_dir / spec["path"])
            return np.asarray(data[spec.get("key_name", "x")], dtype=float)

        chain_spec = run_products.get("chain_layout")
        latent_spec = run_products.get("latent")
        checkpoint_spec = run_products.get("checkpoint")

        if prefer_npz and latent_spec is not None and latent_spec.get("kind") == "npz":
            return _load_npz(latent_spec)

        if chain_spec is not None and chain_spec.get("kind") == "ptmcmc":
            return _load_ptmcmc(chain_spec)

        if latent_spec is not None and latent_spec.get("kind") == "npz":
            return _load_npz(latent_spec)

        if checkpoint_spec is not None and checkpoint_spec.get("kind") == "npz":
            return _load_npz(checkpoint_spec)

        raise RunIOError(
            "run metadata has no readable latent source (chain_layout, latent, or checkpoint)"
        )

    def latent(self, *, burn: int | float = 0, thin: int = 1) -> np.ndarray:
        """Latent sample rows with burn-in and thinning applied.

        ``burn`` is a draw count (int) or a leading fraction (float in [0, 1)).
        """
        x = self.load_latent()
        return _slice_draws(x, burn=burn, thin=thin)

    def posterior(
        self,
        *,
        burn: int | float = 0,
        thin: int = 1,
        units: str = "display",
    ) -> dict[str, np.ndarray]:
        """Physical timing-parameter samples decoded from the latent chain.

        Burn/thin are applied to the latent draws before decoding, so latent
        and physical views stay consistent.
        """
        x = self.latent(burn=burn, thin=thin)
        return self.space.to_physical(x, units=units)

    def truths(self, *, units: str = "display") -> dict[str, float]:
        """Par-file reference values (zero delta) for overlay markers."""
        ndim = len(self.run_meta["sampled"])
        zero = np.zeros((1, ndim), dtype=float)
        phys = self.space.to_physical(zero, units=units, coord="delta")
        return {name: float(np.asarray(arr)[0]) for name, arr in phys.items()}

    def _load_theta(self, unit_mode: str) -> dict[str, np.ndarray]:
        suffix = _DISPLAY_SUFFIX if unit_mode == "display" else _NATIVE_SUFFIX
        name_stem = self.run_meta["name_stem"]
        run_products = self.run_meta["run_products"]
        for key in ("latent", "checkpoint"):
            spec = run_products.get(key)
            if spec is None or spec.get("kind") != "npz":
                continue
            data = np.load(self.run_dir / spec["path"])
            keys = [k for k in data.files if k.endswith(suffix)]
            if keys:
                return {
                    k[len(name_stem) + 1 : -len(suffix)]: np.asarray(
                        data[k], dtype=float
                    )
                    for k in keys
                }
        # For a dynamic transport, xi alone has no physical meaning; the stored
        # decoded physical values are canonical and their absence is a
        # run-record error, never a reason to reinterpret xi as static timing
        # coordinates (§7.3, §8.1).
        if not self.latent_decodable:
            raise RunIOError(
                "this run uses a dynamic transport (latent_decodable=false); its "
                "canonical decoded physical values are required but missing. "
                "Latent xi cannot be decoded through run.space (§7.3)."
            )
        return self.space.to_physical(self.load_latent(), units=unit_mode)

    def load_display(self) -> dict[str, np.ndarray]:
        return self._load_theta("display")

    def load_native(self) -> dict[str, np.ndarray]:
        return self._load_theta("native")

    def to_physical_table(self) -> dict[str, np.ndarray]:
        """Display-unit physical samples keyed by fitpar name."""
        return self.load_display()

    def export_physical_table(self, path: str | Path) -> Path:
        path = Path(path)
        table = self.to_physical_table()
        if path.suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise RunIOError(
                    "parquet export requires pandas; install pandas/pyarrow"
                ) from exc
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(table).to_parquet(path)
            return path
        if path.suffix == ".npz":
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **table)
            return path
        raise RunIOError(f"unsupported physical-table extension {path.suffix!r}")

    def to_arviz(self, *, sample_stats: Mapping[str, np.ndarray] | None = None):
        try:
            import arviz as az
        except ImportError as exc:
            raise RunIOError("to_arviz requires arviz") from exc
        posterior = {k: v[None, ...] for k, v in self.load_display().items()}
        stats = (
            {k: np.asarray(v)[None, ...] for k, v in sample_stats.items()}
            if sample_stats
            else None
        )
        return az.from_dict(
            posterior=posterior,
            sample_stats=stats,
            attrs={
                "nlt_run_meta_schema": self.run_meta["schema"],
                "space_digest": self.run_meta["parameter_space"]["digest"],
                "context_digest": self.run_meta["context_digest"],
                "likelihood": self.run_meta["likelihood"],
                "sample_coord": self.run_meta["sample_coord"],
            },
        )


def load_run(
    run_dir: str | Path, *, verify: bool = True, force: bool = False
) -> RunResults:
    """Load a run directory's metadata + parameter space into ``RunResults``."""
    return RunResults.load(run_dir, verify=verify, force=force)


def save_ptmcmc_decentered_checkpoint(
    run_dir: str | Path,
    chain: np.ndarray,
    ctx,
    transport,
    manifest: RunManifest,
    *,
    hyper_names,
    final: bool,
    n_target: int | None = None,
) -> Path:
    """Decode a decentered PTMCMC chain and persist a dynamic checkpoint (E25).

    ``chain`` is the PTMCMC array (rows in memory or loaded from ``chain_1.txt``);
    the last 4 columns (``lnpost, lnlike, accept, pt-accept``) are stripped by
    count. The parameter block is ``[xi (k) | eta (m)]`` (E20/E23); ``xi``/``eta``
    are decoded row-wise via :func:`nltiming.decentering.decode_decentered_chain`
    (each row reconstructs the transport at its own eta), then handed to the same
    :func:`save_dynamic_checkpoint` writer as the NumPyro dynamic path. No new
    storage layout is invented.
    """
    from .decentering import decode_decentered_chain

    chain = np.asarray(chain, dtype=float)
    hyper_names = tuple(hyper_names)
    k = len(ctx.plan.sampled)
    m = len(hyper_names)
    params_block = chain[:, :-4]  # strip lnpost/lnlike/accept/pt-accept
    chain_xi = params_block[:, :k]
    chain_eta = params_block[:, k : k + m]
    log_density = chain[:, -4]  # lnpost

    timing_delta = decode_decentered_chain(
        chain_xi, chain_eta, hyper_names, transport, ctx.space
    )
    theta_native = ctx.space.to_physical(timing_delta, units="native", coord="delta")
    theta_display = ctx.space.to_physical(timing_delta, units="display", coord="delta")
    hyperparameters = {n: chain_eta[:, i] for i, n in enumerate(hyper_names)}

    return save_dynamic_checkpoint(
        run_dir,
        manifest,
        xi=chain_xi,
        final=final,
        hyperparameters=hyperparameters,
        timing_delta=timing_delta,
        theta_native=theta_native,
        theta_display=theta_display,
        log_density=log_density,
        n_target=n_target,
    )
