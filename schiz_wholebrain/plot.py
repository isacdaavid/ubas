"""
Plotting functions
"""

from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 14


def plottify(func: Callable) -> Callable:
    """Decorator to add routine properties and behaviors to plotting functions.

    Args:
        title (str):
            Title for the plot.
        figsize (tuple):
            Figure size (width, height) in inches.
        output (bool):
            Whether to return plot objects or display them.

    Returns:
        None:
            Display plot as side effect.
        tuple[plt.Figure, plt.Axes]:
            Return plot and axes objects.
    """
    @wraps(func)
    def wrapper(
            *args,
            title: str = "",
            figsize: Optional[Sequence[float]] = None,
            output: bool = False,
            **kwargs,
    ) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
        # Call the actual plotting function.
        fig, ax = func(*args, **kwargs)

        # Set properties common to all plots.
        ax.set_title(title, fontsize=FONTSIZE)
        plt.tight_layout()

        # Show or return the plot.
        if output:
            plt.close(fig)
            return fig, ax

        plt.show()
        plt.close(fig)

    return wrapper


@plottify
def compose(
    plots: Sequence[tuple[plt.Figure, plt.Axes]],
    shape: Sequence[int] = (1, 1),
    supertitle: str = "",
):
    """
    Arrange a list of subplots in a grid.

    Args:
        plots (Sequence[Sequence[plt.Figure, plt.Axes]]):
            List of (figure, axis) tuples.
        shape (Sequence(int)):
            Shape of the grid (rows, columns).
        Plus everything listed in `help(plot.plottify)`.

    Returns:
        See `help(plot.plottify)`.
    """
    # Create a new figure for the composite plot.
    fig = plt.figure()
    if supertitle:
        fig.suptitle(supertitle, fontsize=FONTSIZE)

    for i, (subfig, subax) in enumerate(plots):
        # Create a subplot in the composite figure.
        ax = fig.add_subplot(shape[0], shape[1], i + 1)
        # Copy the image and colorbar from the original plot.
        subim = subax.images[0]
        ax.imshow(subim.get_array())
        ax.set_title(subax.get_title())
        # plt.colorbar(subim, ax=ax)

    return fig, ax


@plottify
def connectivity(
        matrix: np.array,
        labels: Optional[Sequence[str]] = None,
):
    """Visualize a connectivity matrix as a heatmap.

    Args:
        matrix (np.array):
            numerical 2D numpy array to visualize.
        labels (Optional[Sequence[str]]):
            Labels for the x and y axes (brain regions).
        Plus everything listed in `help(plot.plottify)`.

    Returns:
        See `help(plot.plottify)`.

    Raises:
        ValueError:
            If matrix is not 2D and square. If labels don't match matrix size.
        TypeError:
            If matrix is not numerical.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be 2D and square.")
    if labels is not None and len(labels) != matrix.shape[0]:
        raise ValueError("Wrong number of labels.")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("Only numerical values are supported.")

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation='vertical', fontsize=0.2 * FONTSIZE)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=0.2 * FONTSIZE )

    plt.colorbar(im, ax=ax)
    return fig, ax


@plottify
def density(
        controls,
        patients,
        xlabel="",
        ylabel="",
):
    """
    Args:
        Plus everything listed in `help(plot.plottify)`.

    Returns:
        See `help(plot.plottify)`.
    """
    fig, ax = plt.subplots()
    plt.hist(controls, alpha=0.5, bins=40, color='blue', label='HC')
    plt.hist(patients, alpha=0.5, bins=40, color='orange', label='SSD')
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend()
    return fig, ax


def jitter(
        positions: Sequence[float],
        amount: float=0.1,
) -> np.array:
    """Offset numerical positions by some random amount."""
    offsets = np.random.uniform(-amount, amount, size=len(positions))
    return np.array(positions) + offsets


@plottify
def repeated_measures(
        measures: Mapping[str, Mapping[str, Any]],
        subject_colors: Optional[Mapping[str, str]] = None,
        subject_groups: Optional[Mapping[str, str]] = None,
        xlabel: str = "",
        ylabel: str = "",
):
    """
    Plot repeated measures as bars (mean ± std) and individual time series.

    Args:
        measures (Mapping[str, Mapping[str, Any]]):
            A dict of measurement names to subject data ({subject: datum}).
        subject_colors (Optional[Mapping[str, str]]):
            Optional dict of subject labels to colors.
        subject_groups (Optional[Mapping[str, str]]):
            Optional dict of subject labels to group labels.
        xlabel (str):
            Label for the x-axis.
        ylabel (str):
            Label for the y-axis.
        Plus everything listed in `help(plot.plottify)`.

    Returns:
        See `help(plot.plottify)`.
    """
    fig, ax = plt.subplots()
    # Plot bars with mean and std error for each measurement.
    for i, measure in enumerate(measures):
        values = np.array(list(measures[measure].values()))
        plt.bar(i, values.mean(), yerr=values.std(), color='gray')

    # Gather time series for each unique Subject key across measurements.
    time_series = defaultdict(list)

    for measure in measures.values():
        for subject_label, datum in measure.items():
            time_series[subject_label].append(datum)

    # Plot Subject time series on top of measure bars.
    unique_labels = {}

    for subject_label, series in time_series.items():
        color = subject_colors.get(subject_label, 'k') if subject_colors else 'k'
        label = subject_groups.get(subject_label, '') if subject_groups else None
        line, = plt.plot(series, color=color, alpha=.3)

        if label and label not in unique_labels:
            unique_labels[label] = line

    # Final touches.
    if subject_groups is not None and unique_labels:
        plt.legend(unique_labels.values(), unique_labels.keys())

    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.xticks(range(len(measures)), measures, fontsize=FONTSIZE * .8)
    return fig, ax


@plottify
def scatter(
        controls,
        patients,
        xlabel="",
        ylabel="",
):
    """
    Args:
        Plus everything listed in `help(plot.plottify)`.

    Returns:
        See `help(plot.plottify)`.
    """
    fig, ax = plt.subplots()
    x = jitter([1] * len(controls))
    plt.plot(x, controls, 'o', color='blue')
    x = jitter([2] * len(patients))
    plt.plot(x, patients, 'o', color='orange')
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.xticks([1, 2], ["HC", "SSD"], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlim(0.5, 2.5)
    return fig, ax


@plottify
def spectrum(
        controls,
        patients,
        xlim,
        xlabel="",
        ylabel="Power",
):
    """
    Args:
        Plus everything listed in `help(plot.plottify)`.

    Returns:
        See `help(plot.plottify)`.
    """
    sample_rate = 1 / 0.5
    fig, ax = plt.subplots()

    for control in controls:
        freq = np.fft.fftfreq(control.shape[0], sample_rate)
        plt.plot(freq, control, color='blue', alpha=1)

    for patient in patients:
        freq = np.fft.fftfreq(patient.shape[0], sample_rate)
        plt.plot(freq, patient, color='orange', alpha=1)

    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.xlim(*xlim)
    return fig, ax
