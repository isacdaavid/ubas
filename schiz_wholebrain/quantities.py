"""
Functions to compute measurements of interest on `Subject` data.
"""

from typing import TypeVar

SubjectT = TypeVar('SubjectT', bound='Subject')

from neurolib.models.aln import ALNModel
import neurolib.utils.functions
import numpy as np

from .subject import Subject

# TODO: don't hardcode parameters. Receive and unpack variable args.

ATLAS = '4S156'
CORTEX = slice(0, 100)

def structure_function_correlation(subject: SubjectT) -> float:
    structural = subject.structural_connectivity[ATLAS].normalize('raw_count')
    structural_cortex = structural[CORTEX, CORTEX].ravel()
    functional = subject.functional_connectivity[ATLAS].correlation_matrix
    functional_cortex = functional[CORTEX, CORTEX].ravel()
    return float(np.corrcoef(structural_cortex, functional_cortex)[0, 1])

def aln_correlation(subject: SubjectT, mean_structural=False) -> float:
    # Adaptive linear-nonlinear model of exponential integrate-and-fire neurons
    if mean_structural:
        structural = subject.quantities['cohort_mean_raw_count']
        distances =  subject.quantities['cohort_mean_mean_length']
    else:
        structural = subject.structural_connectivity[ATLAS].normalize('raw_count')
        distances = subject.structural_connectivity[ATLAS].normalize('mean_length')

    structural_cortex = structural[CORTEX, CORTEX]
    distances_cortex = distances[CORTEX, CORTEX]

    model = ALNModel(Cmat = structural_cortex, Dmat = distances_cortex)
    # Default values between brackets.
    parameters = {
        'duration': 5 * 60 * 1000, # 5 min x 60 s x 1000 ms.
        # Global constants.
        'signalV': 20, # [20 m/s] global signal speed.
        'c_gl': 0.3,   # [0.3] global current?
        'Ke_gl': 300, # [250] global coupling between E populations.
        # Background input to E and I populations.
        'mue_ext_mean': 1.63,   # [0.4 mV/ms] mean ext input to E.
        'mui_ext_mean': 0.05,   # [0.3 mV/ms] mean ext input to I.
        'ext_exc_rate': 0.0,    # [0.0]
        'ext_inh_rate': 0.0,    # [0.0]
        'ext_exc_current': 0.0, # [0.0]
        'ext_inh_current': 0.0, # [0.0]
        'sigmae_ext': 1.5,      # [1.5 mV/√ms] std dev ext input to E.
        'sigmai_ext': 1.5,      # [1.5 mV/√ms] std dev ext input to I.
        # Number of inputs per neuron.
        'Ke': 800,    # [800] E inputs.
        'Ki': 200,    # [200] I inputs.
        # Synaptic constants.
        'de': 4.0,     # [4.0 ms] delay to E neurons.
        'di': 2.0,     # [2.0 ms] delay to I neurons.
        'tau_se': 2.0, # [2.0 ms] E synapse time constant.
        'tau_si': 5.0, # [5.0 ms] I synapse time constant.
        'tau_de': 1.0, # [1.0 ms] delay to E time constant?
        'tau_di': 1.0, # [1.0 ms] delay to E time constant?
        # Maximum post-synaptic currents.
        'cee': 0.3,   # [0.3 mV/ms] AMPA E to E.
        'cie': 0.3,   # [0.3 mV/ms] AMPA E to I.
        'cei': 0.5,   # [0.5 mV/ms] GABA I to E.
        'cii': 0.5,   # [0.5 mV/ms] GABA I to I.
        # Maximum synaptic currents.
        'Jee_max': 2.43,  # [2.43 mV/ms] E to E limit.
        'Jie_max': 2.6,  # [2.6 mV/ms] E to I limit.
        'Jei_max': -3.3, # [-3.3 mV/ms] I to E limit.
        'Jii_max': -1.64, # [-1.64 mV/ms] I to I limit.
        # Turn on spike-triggered adaptation currents.
        'a': 28.26,   # [0 nS] subthreshold adaptation conductance.
        'b': 24.04,   # [0 pA] spike-triggered current increment.
        'EA': -80,    # [-80 mV] adaptation reversal potential.
        'tauA': 200,  # [200 ms] adaptation time constant.
        # Membrane and other constants.
        'C': 200,      # [200 pF] membrane capacitance.
        'gL': 10,      # [10 nS] leak conductance.
        'EL': -65,     # [-65 mV] leak reversal potential.
        'DeltaT': 1.5, # [1.5 mV] threshold slope factor.
        'VT': -50,     # [-50 mV] threshold voltage.
        'Vr': -70,     # [-70 mV] rate threshold voltage.
        'Vs': -40,     # [-40 mV] spike threshold voltage.
        'Tref': 1.5,   # [1.5 ms] refractory period.
        # Ornstein-Uhlenbeck (brownian walk + mean drift + mean reversion).
        'tau_ou': 5.0,             # [5.0 ms]
        'sigma_ou': 0.19,         # [0] std. dev.
    }
    model.params.update(parameters)
    model.run(chunkwise=True, chunksize=60000, bold=True)

    transient = 12              # seconds
    simulated = neurolib.utils.functions.fc(model.BOLD.BOLD[:, transient:])
    simulated = simulated.ravel()
    empirical = subject.functional_connectivity[ATLAS].correlation_matrix
    empirical_cortex = empirical[CORTEX, CORTEX].ravel()

    return float(np.corrcoef(simulated, empirical_cortex)[0, 1])

def fcs(subject: SubjectT, absolute=True) -> np.ndarray[float]:
    correlation = subject.functional_connectivity[ATLAS].correlation_matrix
    correlation_cortex = correlation[CORTEX, CORTEX]
    fcs = correlation_cortex.mean(axis=0)
    if absolute:
        return fcs
    fisher = np.arctanh(fcs)
    zscores = (fisher - fisher.mean()) / fisher.std()
    return zscores
