# Nonlinear timing, introductory notebooks

These notebooks introduce **numerical sampling of pulsar timing parameters**
with `nltiming`. They assume you already know tempo2/PINT and can build a PTA
likelihood in Discovery or Enterprise. They do **not** assume you have sampled
timing-model parameters before.

## Pulsar object: a `TimingPulsar`

`nltiming` binds a `TimingSpec` to a pulsar through the `TimingPulsar`
protocol and returns a `TimingSignal`. **Today the only production
implementation is
[MetaPulsar](https://github.com/vhaasteren/metapulsar)**, even for a single
PTA dataset, so every notebook builds the host with `create_metapulsar`.
Once Discovery and/or Enterprise provide a native `TimingPulsar`, that
dependency can be dropped; the `nltiming` API in these notebooks will not
change.

Run top-to-bottom from `examples/notebooks/` in an environment that has
MetaPulsar, JUG, Discovery, NumPyro, Enterprise, PTMCMCSampler, and (for
notebook 5) pyvela / Vela.jl (e.g. the MetaPulsar devcontainer with this
package editable-installed).

Short, non-notebook versions of the first notebook live in
[`../scripts/`](../scripts/): `quickstart_discovery.py` and
`quickstart_enterprise.py`.

## Suggested order

| # | Notebook | Data | Focus |
|---|----------|------|-------|
| 1 | `01_discovery_enterprise_backends.ipynb` | J1721-2457 | Discovery + Enterprise, backends (JUG, libstempo, Vela), chains and corner plots |
| 2 | `02_charts_and_binary.ipynb` | J1022+1001 | Per-axis charts and Kepler↔Laplace (`EPS1/EPS2/TASC` on a DDH engine) |
| 3 | `03_decentering_and_full_basis.ipynb` | J1022+1001 | Default decentered sampling vs `inference="all"` |
| 4 | `04_geometry.ipynb` | J1022+1001 | Certify geometry; `identically_linear` |
| 5 | `05_vela_discovery_sim.ipynb` | J1909-3744-sim | Three-way overlay on a simulated ELL1 pulsar: native Vela/emcee, nltiming Discovery/JUG/NUTS, and Discovery/Vela/`discovery_target` PTMCMC |

Data lives in [`examples/data/`](../data/): AEI-DR2 combined
`par-optimized` pars and INCLUDE `.tim` trees, plus the barycentric ELL1
simulation used by notebook 5. Sampling cells use short chains for pedagogy;
scale `num_warmup` / `num_samples` / `Niter` for science. Outputs are not
committed.
