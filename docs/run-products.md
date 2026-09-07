# Run products: decode chains anywhere

A persisted run is a **scientific record**: decode it with the exact space it
was sampled with, and build a live model only for fresh calculations. A valid
read needs only the on-disk products, `nlt_run_meta.json` (schema
`nlt-run-meta-v5`) plus the serialized `ParameterSpace` and the raw chain, never
a live PTA, Discovery model, or PINT reload.

```python
import nltiming

run   = nltiming.load_run(outdir)      # RunResults, verified by default
phys  = run.load_display()             # prefer stored decoded physical values
post  = run.posterior(burn=0.25)       # decode latent draws through run.space
lat   = run.load_latent()              # raw latent chain (diagnostic)
truth = run.truths()                   # par-file reference values for overlays
```

`load_run` (sugar for `RunResults.load(outdir, verify=True)`) recomputes every
manifest **section digest**, `parameter_space`, `context`, `metric_source`,
`transport`, `chains`, and a verification failure names the section that
diverged. It refuses an unsupported schema with migration guidance, and
`RunManifest.write` refuses to overwrite an incompatible run without
`force=True`.

**Reconcile before feeding a saved point through a rebuilt likelihood** (e.g. a
GLS diagnostic):

```python
run.assert_consistent_with(timing)   # raises, naming the diverging section, on any mismatch
```

Decode with `run.space`; never rebuild a decoder from pulsar/config for a saved
chain, and never `timing.write(...)` an existing run before loading it.

## Static vs. dynamic transport

The manifest's `transport` section records one of two classes:

- **`static_affine`** (`latent_decodable = true`), a fixed timing-only
  `(C, c)`. The latent chain is independently decodable through `run.space`;
  this is what the [static timing whitening](whitening.md) produces.
- **`dynamic_transport`** (`latent_decodable = false`), a joint full-basis
  transport `q = mu(eta) + L(eta)^{-T} xi` whose map is a *parameterized family*
  of affine maps indexed by sampled hyperparameters η. For each fixed η the map
  is ordinary coordinate transport; jointly, `(xi, eta) ↦ (q, eta)`, so `xi`
  alone has no physical meaning. That hyperparameter dependence makes the
  transport dynamic/conditional, it does not make the per-axis δ↔z charts any
  less exact. `load_display()` reads the **required** stored per-draw physical
  values and refuses to reinterpret `xi` through `run.space`. Joint runs are
  written with `save_dynamic_checkpoint`, which refuses to promote a final
  checkpoint that lacks those canonical decoded values, and the one-affine-layer
  invariant keeps the static timing layer at identity when a dynamic transport
  is active.

