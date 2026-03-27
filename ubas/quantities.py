"""
Functions to compute measurements of interest on `Member` data.
"""

from typing import Optional, Sequence, TypeVar, Tuple

from neurolib.models.aln import ALNModel
from neurolib.models.hopf import HopfModel
import numpy as np

from .subject import Subject
from .connectivity import FunctionalConnectivity, StructuralConnectivity

SubjectT = TypeVar('SubjectT', bound=Subject)

# TODO: don't hardcode parameters. Receive and unpack variable args.
ATLAS = '4S156'
CORTEX = slice(0, 100)


def structural_connectivity(
        subject: SubjectT,
        atlas: str = ATLAS
) -> StructuralConnectivity:
    session = subject.members.pop()
    dwi = subject[session.label]['dwi']
    file_pattern = 'connectivity.mat'
    dwi = dwi.filter(lambda f: f.label.endswith(file_pattern))

    connectivity_file = None

    # Only take the first run, if many exist.
    for f in dwi:
        if 'run' not in f.entities or f.entities['run'] == '01':
            connectivity_file = f
            break

    return StructuralConnectivity(connectivity_file, atlas)


def functional_connectivity(
        subject: SubjectT,
        atlas: str = ATLAS
) -> FunctionalConnectivity:
    session = subject.members.pop()
    func = subject[session.label]['func']
    file_pattern = f'{atlas}Parcels_stat-mean_timeseries.tsv'
    func = func.filter(lambda f: f.label.endswith(file_pattern))

    connectivity_file = None

    # Only take the first run, if many exist.
    for f in func:
        if 'run' not in f.entities or f.entities['run'] == '01':
            connectivity_file = f
            break

    func = subject[session.label]['func']
    file_pattern = f'_outliers.tsv'
    func = func.filter(lambda f: f.label.endswith(file_pattern))

    outliers_file = None

    # Only take the first run, if many exist.
    for f in func:
        if 'run' not in f.entities or f.entities['run'] == '01':
            outliers_file = f
            break

    return FunctionalConnectivity(connectivity_file, outliers_file)


def sim_functional_connectivity(
        subject: SubjectT,
        transient: int = 0,
        model_key: str = '',
        bandpass: Optional[Sequence[float]] = None,
        sampling_rate: Optional[float] = None,
) -> np.ndarray:
    """Obtain functional connectivity matrix from BOLD simulation in model.

    Args:
        subject (SubjectT):
            Subject containing ALN model results.
        transient (int):
            Duration of transient dynamics to be discard from data, in seconds.
        model_key (str):
            Custom name under which the model should be retrieved.
        bandpass (Optional[Sequence[float]]):
            Filter frequency content of BOLD time series to (min Hz, max Hz).
        sampling_rate (Optional[float]):
            Sampling frequency in Hz.

    Returns:
        np.ndarray:
            Pearson correlation matrix.
    """
    time_series = subject.quantities[model_key].BOLD.BOLD

    if transient < 0 or transient >= time_series.shape[1]:
        raise ValueError(
            f"transient must be between 0 and {time_series.shape[1] - 1}."
        )

    time_series = time_series[:, transient:]

    if bandpass is not None:
        if sampling_rate is None:
            raise ValueError("Missing `sampling_rate` for `bandpass`.")

        time_series = bandpass_filter(
            time_series, sampling_rate, bandpass[0], bandpass[1]
        )

    return np.corrcoef(time_series)


def _get_structural(
        subject: SubjectT,
        mean_structural: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    quantities = subject.quantities

    if mean_structural:
        C = quantities['cohort_mean_raw_count']
        D = quantities['cohort_mean_mean_length']
    else:
        C = quantities['structural_connectivity'].raw_count
        D = quantities['structural_connectivity'].mean_length

    C_cortex = C[CORTEX, CORTEX]
    D_cortex = D[CORTEX, CORTEX]

    np.fill_diagonal(C_cortex, 0)
    np.fill_diagonal(D_cortex, 0)

    C_cortex_norm = C_cortex / np.linalg.norm(C_cortex, ord=2)
    D_cortex_norm = D_cortex / np.linalg.norm(D_cortex, ord=2)

    return C_cortex_norm, D_cortex_norm


def hopf_model(
        subject: SubjectT,
        mean_structural: bool = False,
        duration: int = 300,
) -> HopfModel:
    structural, distances = _get_structural(subject, mean_structural)

    model = HopfModel(Cmat=structural, Dmat=distances)

    parameters = {
        'duration': duration * 1000,  # 5 min x 60 s x 1000 ms.
        # Global parameters.
        'signalV': 20,  # [20 m/s] global signal speed.
        'K_gl': 1,    # [0.6] global coupling between nodes.
        # Local parameters.
        'a': -0.4,                # [0.25] a in Hopf bifurcation (a+iw)z.
        'w': 0.2,                 # [0.2] Hopf oscillator freq. (a+iw)z.
        # Ornstein-Uhlenbeck noise (brownian walk + mean drift + mean reversion).
        'tau_ou': 5.0,            # [5.0 ms]
        'sigma_ou': 0.0,          # [0.0] std. dev.
        'x_ou_mean': 0.0,         # [0.0 mV/ms]
        'y_ou_mean': 0.0,         # [0.0 mV/ms]
    }

    model.params.update(parameters)
    model.run(chunkwise=True, chunksize=20000, bold=True)
    return model


def aln_model(
        subject: SubjectT,
        mean_structural: bool = False,
        duration: int = 300,
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
        mean_structural (bool):
            If True, uses cohort-averaged structural connectivity.
        duration (int):
            The duration of the simulation, in seconds.

    Returns:
        ALNModel:
            An initialized and executed ALN model with simulation results.
    """
    structural, distances = _get_structural(subject, mean_structural)

    model = ALNModel(Cmat=structural, Dmat=distances)
    # Default values between brackets.
    parameters = {
        'duration': duration * 1000,  # 5 min x 60 s x 1000 ms.
        # Global parameters.
        'signalV': 20,  # [20 m/s] global signal speed.
        'c_gl': 0.3,    # [0.3] global current?
        'Ke_gl': 300,   # [250] global coupling between E populations.
        # Background input to E and I populations.
        'mue_ext_mean': 1.63,    # [0.4 mV/ms] mean ext input to E.
        'mui_ext_mean': 0.05,    # [0.3 mV/ms] mean ext input to I.
        'ext_exc_rate': 0.0,     # [0.0]
        'ext_inh_rate': 0.0,     # [0.0]
        'ext_exc_current': 0.0,  # [0.0]
        'ext_inh_current': 0.0,  # [0.0]
        'sigmae_ext': 1.5,       # [1.5 mV/√ms] std dev ext input to E.
        'sigmai_ext': 1.5,       # [1.5 mV/√ms] std dev ext input to I.
        # Number of inputs per neuron.
        'Ke': 800,    # [800] E inputs.
        'Ki': 200,    # [200] I inputs.
        # Synaptic constants.
        'de': 4.0,      # [4.0 ms] delay to E neurons.
        'di': 2.0,      # [2.0 ms] delay to I neurons.
        'tau_se': 2.0,  # [2.0 ms] E synapse time constant.
        'tau_si': 5.0,  # [5.0 ms] I synapse time constant.
        'tau_de': 1.0,  # [1.0 ms] delay to E time constant?
        'tau_di': 1.0,  # [1.0 ms] delay to I time constant?
        # Maximum post-synaptic currents.
        'cee': 0.3,   # [0.3 mV/ms] AMPA E to E.
        'cie': 0.3,   # [0.3 mV/ms] AMPA E to I.
        'cei': 0.5,   # [0.5 mV/ms] GABA I to E.
        'cii': 0.5,   # [0.5 mV/ms] GABA I to I.
        # Maximum synaptic currents.
        'Jee_max': 2.43,   # [2.43 mV/ms] E to E limit.
        'Jie_max': 2.6,    # [2.6 mV/ms] E to I limit.
        'Jei_max': -3.3,   # [-3.3 mV/ms] I to E limit.
        'Jii_max': -1.64,  # [-1.64 mV/ms] I to I limit.
        # Turn on spike-triggered adaptation currents.
        'a': 28.26,   # [0 nS] subthreshold adaptation conductance.
        'b': 24.04,   # [0 pA] spike-triggered current increment.
        'EA': -80,    # [-80 mV] adaptation reversal potential.
        'tauA': 200,  # [200 ms] adaptation time constant.
        # Membrane and other constants.
        'C': 200,       # [200 pF] membrane capacitance.
        'gL': 10,       # [10 nS] leak conductance.
        'EL': -65,      # [-65 mV] leak reversal potential.
        'DeltaT': 1.5,  # [1.5 mV] threshold slope factor.
        'VT': -50,      # [-50 mV] threshold voltage.
        'Vr': -70,      # [-70 mV] rate threshold voltage.
        'Vs': -40,      # [-40 mV] spike threshold voltage.
        'Tref': 1.5,    # [1.5 ms] refractory period.
        # Ornstein-Uhlenbeck noise (brownian walk + mean drift + mean reversion).
        'tau_ou': 5.0,            # [5.0 ms]
        'sigma_ou': 0.19,         # [0] std. dev.
    }

    model.params.update(parameters)
    model.run(chunkwise=True, chunksize=20000, bold=True)
    return model


def bandpass_filter(
    data: np.ndarray,
    sampling_rate: float,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """
    Apply a bandpass filter to each row of a time-series matrix.

    Args:
        data (np.ndarray):
            2D array of shape (n_signals, n_samples).
        sampling_rate (float):
            Sampling frequency in Hz.
        low_freq (float):
            Low cutoff frequency in Hz.
        high_freq (float):
            High cutoff frequency in Hz.

    Returns:
        np.ndarray:
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


def connectivity_strength(
        subject: SubjectT,
        absolute=True,
) -> np.ndarray:
    """Connectivity Strength (Global Brain Connectivity) of connectivity matrix.

    The connectivity strength of a node is defined as its average connectivity
    with all other nodes.

    Args:
        subject (SubjectT):
            Subject containing structural connectivity data.
        absolute (bool):
            Whether to return bare (absolute) strengths or normalize to z-scores.

    Return:
        np.ndarray:
            A 1-dimensional array with the connectivity strength of each node.
    """
    fc = subject.quantities['functional_connectivity']
    correlation = fc.correlation_matrix
    correlation_cortex = correlation[CORTEX, CORTEX]
    fcs = correlation_cortex.mean(axis=0)
    if absolute:
        return fcs
    fisher = np.arctanh(fcs)
    zscores = (fisher - fisher.mean()) / fisher.std()
    return zscores


def matrix2matrix_correlation(
        subject: SubjectT,
        matrix1: str = "",
        matrix2: str = "",
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
    matrix1_cortex = subject.collect_self(matrix1)[CORTEX, CORTEX]
    matrix2_cortex = subject.collect_self(matrix2)[CORTEX, CORTEX]
    correlation = np.corrcoef(matrix1_cortex.ravel(), matrix2_cortex.ravel())
    return float(correlation[0, 1])
