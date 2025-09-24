"""
Plotting functions.
"""

from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 14
MINIMUM_ALPHA = 0.25
JITTER = 0.1


def jitter(
        positions: Sequence[float],
        amount: float=JITTER,
) -> np.ndarray[float]:
    """Offset numerical positions by some random amount."""
    offsets = np.random.uniform(-amount, amount, size=len(positions))
    return np.array(positions) + offsets


def min_alpha(n: int) -> float:
    """Dynamically compute the alpha opacity for n overlapping objects."""
    return max(MINIMUM_ALPHA, 1 / n)


def plottify(func: Callable) -> Callable:
    """Decorator to add routine properties and behaviors to plotting functions.

    Args:
        title (str):
            Title for the plot.
        figsize (tuple):
            Figure size (width, height) in inches.
        axislabels (Sequence[str]):
            Ordered labels for the x-axis, y-axis, etc.
        fontsize (float):
            The base font size for the plot, in inches.
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
            axislabels: Sequence[str] = ("", "", ""),
            fontsize: float = FONTSIZE,
            output: bool = False,
            **kwargs,
    ) -> Union[None, tuple[plt.Figure, plt.Axes]]:
        # Call the actual plotting function.
        fig, ax = func(*args, fontsize=fontsize, **kwargs)

        # Set properties common to all plots.

        if figsize is not None:
            fig.set_size_inches(figsize)

        ax.set_title(title, fontsize=fontsize)

        try:
            setters = [ax.set_xlabel, ax.set_ylabel, ax.set_zlabel]
        except AttributeError:
            setters = [ax.set_xlabel, ax.set_ylabel]
        for label, setter in zip(axislabels, setters):
            setter(label, fontsize=fontsize)

        ax.tick_params(labelsize=fontsize * 0.8)

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
    fontsize: float = FONTSIZE,
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
        fig.suptitle(supertitle, fontsize=fontsize)

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
        matrix: np.ndarray,
        labels: Optional[Sequence[str]] = None,
        fontsize: float = FONTSIZE,
):
    """Visualize a connectivity matrix as a heatmap.

    Args:
        matrix (np.ndarray):
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
        ax.set_xticklabels(labels, rotation='vertical', fontsize=0.2 * fontsize)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=0.2 * fontsize )

    plt.colorbar(im, ax=ax)
    return fig, ax


@plottify
def density(
        distributions: Mapping[str, Iterable[float]],
        bins: int = 10,
        fontsize: float = FONTSIZE,
):
    """
    Args:
        distributions (Mapping[str, Iterable[float]]):
            Named samples of measurements
        bins (int):
            Number of bins to divide each histogram.
        Plus everything listed in `help(plot.plottify)`.

    Returns:
        See `help(plot.plottify)`.
    """
    fig, ax = plt.subplots()
    alpha = min_alpha(len(distributions))
    for label, distribution in distributions.items():
        plt.hist(distribution, alpha=alpha, bins=bins, label=label)

    plt.legend(fontsize=fontsize)
    return fig, ax


@plottify
def repeated_measures(
        measures: Mapping[str, Mapping[str, Any]],
        subject_colors: Optional[Mapping[str, str]] = None,
        subject_groups: Optional[Mapping[str, str]] = None,
        fontsize: float = FONTSIZE,
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
        Plus everything listed in `help(plot.plottify)`.

    Returns:
        See `help(plot.plottify)`.
    """
    fig, ax = plt.subplots()
    # Plot bars with mean and std error for each measurement.
    for i, measure in enumerate(measures):
        values = np.array(list(measures[measure].values()))
        ax.bar(i, values.mean(), yerr=values.std(), color='gray')

    # Gather time series for each unique Subject key across measurements.
    time_series = defaultdict(list)

    for measure in measures.values():
        for subject_label, datum in measure.items():
            time_series[subject_label].append(datum)

    # Plot Subject time series on top of measure bars.
    unique_labels = {}
    alpha = min_alpha(len(time_series))
    for subject_label, series in time_series.items():
        color = subject_colors.get(subject_label, 'k') if subject_colors else 'k'
        label = subject_groups.get(subject_label, '') if subject_groups else None
        line, = ax.plot(series, color=color, alpha=alpha)

        if label and label not in unique_labels:
            unique_labels[label] = line

    # Final touches.
    if subject_groups is not None and unique_labels:
        ax.legend(
            unique_labels.values(),
            unique_labels.keys(),
            fontsize=fontsize
        )

    ax.set_xticks(range(len(measures)), measures)
    return fig, ax


@plottify
def scatter(
        groups: Mapping[str, Iterable[float]],
        fontsize: float = FONTSIZE,
):
    """
    Args:
        groups (Mapping[str, Iterable[float]]):
            Named samples of measurements.
        Plus everything listed in `help(plot.plottify)`.

    Returns:
        See `help(plot.plottify)`.
    """
    fig, ax = plt.subplots()
    for i, group in enumerate(groups):
        values = np.array(groups[group])
        ax.bar(i, values.mean(), yerr=values.std(), alpha=0.5)
        x = jitter([i] * len(values), amount=0.2)
        ax.plot(x, values, 'o')

    plt.xticks(range(len(groups)), groups.keys(), fontsize=fontsize)
    plt.xlim(-0.5, len(groups) - 0.5)
    return fig, ax


@plottify
def spectrum(
        controls,
        patients,
        xlim,
        xlabel="",
        ylabel="Power",
        fontsize: float = FONTSIZE,
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

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlim(*xlim)
    return fig, ax
