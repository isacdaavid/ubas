"""
Functions to compute measurements of interest on `Subject` data.
"""

from typing import Optional, TypeVar

SubjectT = TypeVar('SubjectT', bound='Subject')

from neurolib.models.aln import ALNModel
import neurolib.utils.functions
import numpy as np

from .subject import Subject


# TODO: don't hardcode parameters. Receive and unpack variable args.

ATLAS = '4S156'
CORTEX = slice(0, 100)

def matrix2matrix_correlation(
        subject: SubjectT,
        matrix1: str="",
        matrix2: str="",
) -> float:
    """
    Pearson correlation between two connectivity matrices in a Subject.

    Args:
        subject (SubjectT):
            Subject with structural and functional connectivity matrices.
        matrix1 (str):
            Attribute path of first matrix to collect from this Subject.
        matrix2 (str):
            Attribute path of second matrix to collect from this Subject.

    Returns:
        float:
            Pearson correlation.
    """
    matrix1 = subject.collect(matrix1)
    matrix2 = subject.collect(matrix2)
    matrix1_cortex = matrix1[CORTEX, CORTEX]
    matrix2_cortex = matrix2[CORTEX, CORTEX]
    correlation = np.corrcoef(matrix1_cortex.ravel(), matrix2_cortex.ravel())
    return float(correlation[0, 1])


def aln_functional_connectivity(
        subject: SubjectT,
        transient: Optional[int]=0,
        model_key: Optional[str]='aln_model',
) -> np.ndarray:
    """Obtain functional connectivity matrix from BOLD simulation in ALN model.

    Args:
        subject (SubjectT):
            Subject containing ALN model results.
        transient (Optional[int]):
            Duration of transient dynamics to be discard from data, in seconds.

    Returns:
        np.ndarray:
            Pearson correlation matrix.
    """
    time_series = subject.quantities[model_key].BOLD.BOLD[:, transient:]
    # time_series = bandpass_filter(time_series, 0.5, 0.01, 0.1)
    return np.corrcoef(time_series)


def aln_model(
        subject: SubjectT,
        mean_structural: Optional[bool]=False,
        duration: Optional[int]=300,
) -> ALNModel:
    """Simulation of Adaptive Linear-Nonlinear neural mass model.

    The model simulates cortical dynamics using structural connectivity and
    distance matrices (in mm) to add propagation delays, either from subject
    data or pre-computed cohort-averaged values. Node dynamics are taken from a
    neural-mass reduction of excitatory and inhibitory populations with
    exponential integrate-and-fire neurons.

    Args:
        subject (SubjectT):
            Subject containing structural connectivity data.
        mean_structural (Optional[bool]):
            If True, uses cohort-averaged structural connectivity.
        duration (Optional[int]):
            The duration of the simulation, in seconds.

    Returns:
        ALNModel:
            An initialized and executed ALN model with simulation results.

    """
    if mean_structural:
        structural = subject.quantities['cohort_mean_raw_count']
        distances =  subject.quantities['cohort_mean_mean_length']
    else:
        structural = subject.structural_connectivity[ATLAS].normalize('raw_count')
        # Don't normalize distances, only average with transpose to
        # make symmetrical. Some model parameters are distance-sensitive.
        distances = subject.structural_connectivity[ATLAS].mean_length
        distances = (distances + distances.T) / 2

    structural[structural == 0] = structural[structural != 0].min()
    distances[distances == 0] = distances.mean()

    structural_cortex = structural[CORTEX, CORTEX]
    distances_cortex = distances[CORTEX, CORTEX]

    model = ALNModel(Cmat = structural_cortex, Dmat = distances_cortex)
    # Default values between brackets.
    parameters = {
        'duration': duration * 1000, # 5 min x 60 s x 1000 ms.
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
        'tau_di': 1.0, # [1.0 ms] delay to I time constant?
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
        # Ornstein-Uhlenbeck noise (brownian walk + mean drift + mean reversion).
        'tau_ou': 5.0,            # [5.0 ms]
        'sigma_ou': 0.19,         # [0] std. dev.
    }

    model.params.update(parameters)
    model.run(chunkwise=True, chunksize=60000, bold=True)
    return model

def fcs(subject: SubjectT, absolute=True) -> np.ndarray[float]:
    correlation = subject.functional_connectivity[ATLAS].correlation_matrix
    correlation_cortex = correlation[CORTEX, CORTEX]
    fcs = correlation_cortex.mean(axis=0)
    if absolute:
        return fcs
    fisher = np.arctanh(fcs)
    zscores = (fisher - fisher.mean()) / fisher.std()
    return zscores

def bandpass_filter(
    data: np.ndarray,
    sampling_rate: float,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """
    Apply a bandpass filter to each row of a time-series matrix.

    Args:
        data: 2D array of shape (n_signals, n_samples).
        sampling_rate: Sampling frequency in Hz.
        low_freq: Low cutoff frequency in Hz.
        high_freq: High cutoff frequency in Hz.

    Returns:
        Filtered time-series in the time domain.
    """
    n_signals, n_samples = data.shape
    filtered_data = np.zeros_like(data)

    # Frequency axis
    freqs = np.fft.fftfreq(n_samples, d=1/sampling_rate)

    for i in range(n_signals):
        # FFT of the signal
        fft_signal = np.fft.fft(data[i])
        # Create bandpass mask
        mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
        # Apply mask (zero out frequencies outside the band)
        fft_filtered = fft_signal * mask
        # Inverse FFT to return to time domain
        filtered_data[i] = np.fft.ifft(fft_filtered).real

    return filtered_data
