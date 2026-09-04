"""nltiming quickstart, Enterprise edition.

Same TimingSpec idea as quickstart_discovery.py, sampled with PTMCMCSampler
through an Enterprise PTA. Uses the libstempo engine, so a tempo2 install is
required. Runs top to bottom on the example data shipped with the repository:

    cd examples/scripts && python quickstart_enterprise.py
"""

import tempfile
from pathlib import Path

import numpy as np
from enterprise.signals import parameter, signal_base, white_signals
from metapulsar import create_metapulsar
from PTMCMCSampler.PTMCMCSampler import PTSampler

from nltiming import TimingSpec, load_run
import nltiming.sampling as nlts

# 1. A pulsar.
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

# 2. The timing model, evaluated by libstempo.
spec = TimingSpec(engines={"tempo2": "libstempo"})
timing = spec.for_pulsar(pulsar)
print("sampled:", timing.sampled)

# 3. An Enterprise PTA with the timing signal added.
white = white_signals.MeasurementNoise(efac=parameter.Uniform(0.1, 10.0))
pta = signal_base.PTA([(white + spec.enterprise_signal())(pulsar)])
print(pta.param_names)

# 4. Sidecar so load_run() can decode the chain, then sample.
outdir = Path(tempfile.mkdtemp(prefix="nlt_quickstart_"))
timing.write(
    outdir,
    likelihood="enterprise",
    sampler="ptmcmc",
    chain_layout=nlts.ptmcmc.chain_layout(timing, pta.param_names),
)

x0 = np.hstack([np.atleast_1d(p.sample()) for p in pta.params])
sampler = PTSampler(
    len(x0),
    pta.get_lnlikelihood,
    pta.get_lnprior,
    np.diag(np.full(len(x0), 0.01)),
    outDir=str(outdir),
)
sampler.sample(x0, Niter=5_000)

# 5. Physical posterior draws.
run = load_run(outdir)
posterior = run.posterior(burn=0.25)
for name in timing.sampled:
    print(name, np.mean(posterior[name]), np.std(posterior[name]))
