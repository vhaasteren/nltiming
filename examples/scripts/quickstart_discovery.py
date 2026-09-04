"""nltiming quickstart: sample the nonlinear timing block of one pulsar.

Discovery likelihood, JUG timing engine, NumPyro NUTS. Runs top to bottom on
the J1721-2457 example data shipped with the repository:

    cd examples/scripts && python quickstart_discovery.py
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")  # timing residuals need float64

from pathlib import Path  # noqa: E402

import discovery as ds  # noqa: E402
import jax  # noqa: E402
from metapulsar import create_metapulsar  # noqa: E402
from numpyro.infer import init_to_value  # noqa: E402

from nltiming import TimingSpec  # noqa: E402
import nltiming.sampling as nlts  # noqa: E402

# 1. A pulsar. MetaPulsar reads the par/tim pair (here a single "PTA" leg).
DATA = Path(__file__).resolve().parent.parent / "data" / "J1721-2457"
pulsar = create_metapulsar(
    {
        "combined": [
            {
                "par": DATA / "J1721-2457.par",
                "tim": DATA / "J1721-2457.tim",
                "timing_package": "tempo2",
            }
        ]
    },
    combination_strategy="per_pta",
    use_pulse_numbers="reuse",
)

# 2. The timing model: which fit parameters to sample, which to marginalize.
#    The default plan samples the nonlinear axes and marginalizes the rest.
spec = TimingSpec(engines="jug")
timing = spec.for_pulsar(pulsar)
print("sampled:     ", timing.sampled)
print("marginalized:", timing.marginalized)

# 3. A Discovery likelihood with the timing signals added. The marginalized
#    timing block needs Discovery's "metamath" kernel path.
ds.config(kernels="metamath")
efac = f"{pulsar.name}_efac"
likelihood = ds.PulsarLikelihood(
    [
        pulsar.residuals,
        ds.makenoise_measurement_simple(pulsar, add_equad=False),
        *timing.discovery_signals(),
    ]
)

# 4. A NumPyro model and a short NUTS run.
model = nlts.numpyro.decentered_model(likelihood, timing, priors={efac: (0.1, 10.0)})
init = {**nlts.numpyro.decentered_init_values(timing, model.transport), efac: 1.0}
mcmc = nlts.numpyro.nuts(
    model,
    timing,
    num_warmup=200,
    num_samples=500,
    init_strategy=init_to_value(values=init),
)
mcmc.run(jax.random.PRNGKey(0))

# 5. Physical posterior draws (ArviZ InferenceData; corner.corner(post) works).
post = nlts.numpyro.posterior(mcmc, timing)
print(post)
