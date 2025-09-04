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
    structural_cortex = structural[CORTEX, CORTEX].flatten()
    functional = subject.functional_connectivity[ATLAS].correlation_matrix
    functional_cortex = functional[CORTEX, CORTEX].flatten()
    return float(np.corrcoef(structural_cortex, functional_cortex)[0, 1])

def fcs(subject: SubjectT, absolute=True) -> np.ndarray[float]:
    correlation = subject.functional_connectivity[ATLAS].correlation_matrix
    correlation_cortex = correlation[CORTEX, CORTEX]
    fcs = correlation_cortex.mean(axis=0)
    if absolute:
        return fcs
    fisher = np.arctanh(fcs)
    zscores = (fisher - fisher.mean()) / fisher.std()
    return zscores

def aln_correlation(subject: SubjectT) -> float:
    structural = subject.structural_connectivity[ATLAS].normalize('raw_count')
    structural_cortex = structural[CORTEX, CORTEX]
    distances = subject.structural_connectivity[ATLAS].normalize('mean_length')
    distances_cortex = distances[CORTEX, CORTEX]

    model = ALNModel(Cmat = structural_cortex, Dmat = distances_cortex)

    model.params['duration'] = 5 * 60 * 1000 # 5 minutes x 60 seconds x 1000 ms
    model.params['mue_ext_mean'] = 1.63
    model.params['mui_ext_mean'] = 0.05
    # We set an appropriate level of noise
    model.params['sigma_ou'] = 0.19
    # And turn on adaptation of spike-triggered adaptation currents.
    model.params['a'] = 28.26
    model.params['b'] = 24.04

    model.params['Jee_max'] = 2.4
    model.params['tau_de'] = 4
    model.params['tau_di'] = 2

    model.run(chunkwise=True, chunksize=60000, bold=True)

    transient = 12              # seconds
    functional = neurolib.utils.functions.fc(model.BOLD.BOLD[:, transient:])

    return float(np.corrcoef(structural_cortex, functional)[0, 1])
