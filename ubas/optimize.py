import logging
from typing import Optional, Sequence

from scipy.optimize import minimize_scalar
import numpy as np

from ubas import quantities
from .subject import Subject

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def optimize_g(
    subject: Subject,
    *,
    empirical_fc_path: str = '',
    maxiter: Optional[int] = 100,
    bounds: tuple[float] = (),
    sampling_period: float,  # seconds
    transient: Optional[int] = 0,      # seconds
    bandpass: Optional[Sequence[float]] = None,  # Hz
    duration: Optional[int] = 5 * 60,        # seconds
) -> dict:
    """
    Optimize the 'g' coupling parameter for a single subject.
    """
    TEMP_MODEL_KEY = '_temp_hopf_model_opt'
    TEMP_FC_KEY = '_temp_sim_fc_opt'

    # Lists to store convergence history
    K_gl_history = []
    corr_history = []

    def objective(K_gl: float) -> float:
        try:
            # Run model
            model = quantities.hopf_model(
                subject,
                mean_structural=False,
                duration=duration,
                K_gl=K_gl,
            )
            subject.quantities[TEMP_MODEL_KEY] = model

            # Compute simulated FC
            sim_fc = quantities.simulation_functional_connectivity(
                subject,
                simulation_key=TEMP_MODEL_KEY,
                transient=transient,
                bandpass=bandpass,
                sampling_period=sampling_period,
            )
            subject.quantities[TEMP_FC_KEY] = sim_fc

            # Compute correlation
            corr = quantities.matrix2matrix_correlation(
                subject,
                matrix1=f'quantities[{TEMP_FC_KEY}]',
                matrix2=empirical_fc_path,
            )

            # Clean up
            del subject.quantities[TEMP_MODEL_KEY]
            del subject.quantities[TEMP_FC_KEY]

            if np.isnan(corr):
                return 1e6  # Penalize NaN

            # Store K_gl and correlation for this iteration
            K_gl_history.append(K_gl)
            corr_history.append(corr)

            # logger.info(f"{subject.label}, K_gl={K_gl:.4f}, corr={corr:.4f}")
            return -corr  # Minimize negative correlation

        except Exception as e:
            # logger.error(f"Failed for K_gl={K_gl:.4f}: {e}")
            return 1e6

    # Run optimization
    result = minimize_scalar(
        objective,
        bounds=bounds,
        method='bounded',
        options={'maxiter': maxiter},
    )

    return {
        'K_gl': float(result.x),
        'correlation': float(-result.fun),
        'success': result.success,
        'message': result.message,
        'K_gl_history': K_gl_history,
        'corr_history': corr_history,
    }
