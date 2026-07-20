# Nonlinear timing — introductory notebooks

These notebooks introduce **numerical sampling of pulsar timing parameters**
with `nltiming`. They assume you already know tempo2/PINT, can build a PTA
likelihood in Discovery or Enterprise, and have sampled something like a
continuous-wave (CW) source. They do **not** assume you have ever sampled
timing-model parameters yourself — only that you know the usual practice of
analytically marginalizing a linear timing model.

## Pulsar object: MetaPulsar required today

`nltiming` binds to the `TimingPulsar` protocol. **Right now the only
production implementation is
[MetaPulsar](https://github.com/vhaasteren/metapulsar)** — even for a single
PTA dataset — so every notebook builds the pulsar with `create_metapulsar`.
Once Discovery and/or Enterprise provide a native `TimingPulsar`, that
dependency can be dropped; the `nltiming` API in these notebooks will not
change.

Run top-to-bottom in an environment that has MetaPulsar, JUG, Discovery,
NumPyro, and Enterprise (e.g. the MetaPulsar devcontainer with this package
editable-installed).

## Suggested order

1. **`01_nonlinear_timing_charts.ipynb`** (simulated data) — why sample timing
   at all; the inference plan (sample vs analytically marginalize); coordinate
   charts; a short joint NUTS run; the Enterprise path.
2. **`02_geometry_certification_and_pivot.ipynb`** (simulated data) — checking
   that the sampling geometry is healthy before a long run; declaring linear
   axes; pivot-amplitude red noise.
3. **`03_j1640_decentering_validation.ipynb`** (real IPTA DR2 J1640+2224) —
   full-basis sampling on real data: charts, geometry check, modest NUTS,
   pivot amplitude.
4. **`04_j1640_marginalization_validation.ipynb`** (real J1640) — the two
   analytical-marginalization measures (delta-flat vs z-prior) as distinct
   models.

Notebooks **01** and **02** need no external data (PINT simulates TOAs).
**03** and **04** look for EPTA-DR2 J1640 under a MetaPulsar checkout
(`data/ipta-dr2/`).

Sampling cells use short chains for pedagogy; scale `num_warmup` /
`num_samples` / `num_chains` for science. Outputs are not committed.
