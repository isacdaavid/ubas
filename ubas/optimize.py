from functools import wraps
from typing import Any, Callable, Optional, Mapping, Sequence, Tuple

from neurolib.models.model import Model
from scipy.optimize import minimize
import numpy as np

from ubas import quantities
from .subject import Subject


def optimizer(objective):
    @wraps(objective)
    def wrapper(
        *posargs,
        parameters: Sequence[str],
        bounds: Sequence[Tuple[float, float]],
        maxiter: Optional[int] = 100,
        n_restarts: Optional[int] = 1,
        method: str = 'COBYLA',
        **kwargs: Mapping[str, Any],
    ) -> dict:
        parameters_history = {p: [] for p in parameters}
        objective_history = []

        def _objective(
            x0: Sequence[float],
            args: Sequence[str],
        ) -> float:
            parameters_with_values = dict(zip(args, x0))
            merged_kwargs = {**kwargs, **parameters_with_values}
            cost = objective(*posargs, **merged_kwargs)

            if np.isnan(cost):
                cost = 1e6

            for p in args:
                parameters_history[p].append(parameters_with_values[p])

            objective_history.append(cost)
            return cost  # Minimize cost

        # Multi-start optimization
        best_minimum = np.inf

        for _ in range(n_restarts):
            # Random initial guess within bounds
            initial_values = [
                np.random.uniform(param_bounds[0], param_bounds[1])
                for param_bounds in bounds
            ]

            result = minimize(
                _objective,
                x0=initial_values,
                args=parameters,
                bounds=bounds,
                method=method,
                options={
                    'maxiter': maxiter,
                    'ftol': 1e-6,
                    'gtol': 1e-5,
                    'disp': False,  # Set to True for debugging
                },
            )

            if result.fun < best_minimum:
                best_minimum = result.fun
                best_result = result

        return {
            'optimal_parameters': dict(zip(parameters, best_result.x)),
            'optimum': best_minimum,
            'success': best_result.success,
            'message': best_result.message,
            'parameters_history': parameters_history,
            'objective_history': objective_history,
        }

    return wrapper


@optimizer
def optimize_whole_brain(
    subject: Subject,
    *,
    empirical_fc_path: str,
    sampling_period: float,
    transient: Optional[int] = 0,
    bandpass: Optional[Tuple[float, float]] = None,
    duration: Optional[int] = 5 * 60,
    model: Callable[[Subject], Model],
    **parameters_with_values,
) -> dict:
    TEMP_MODEL_KEY = '_temp_model_opt'
    TEMP_FC_KEY = '_temp_sim_fc_opt'

    try:
        simulation = model(
            subject,
            mean_structural=False,
            duration=duration,
            **parameters_with_values,
        )

        subject.quantities[TEMP_MODEL_KEY] = simulation

        sim_fc = quantities.simulation_functional_connectivity(
            subject,
            simulation_key=TEMP_MODEL_KEY,
            transient=transient,
            bandpass=bandpass,
            sampling_period=sampling_period,
        )

        subject.quantities[TEMP_FC_KEY] = sim_fc

        correlation = quantities.matrix2matrix_correlation(
            subject,
            matrix1=f'quantities[{TEMP_FC_KEY}]',
            matrix2=empirical_fc_path,
        )

        return -correlation  # Minimize negative correlation

    except Exception as e:
        print(e)
        return 1e6
