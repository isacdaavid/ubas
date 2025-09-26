"""
Ready-made generic functions to produce and compose matplotlib plots.
"""

from collections import defaultdict
from functools import wraps
from math import floor
from typing import (
    Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Union
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Constants
FONTSIZE: float = 14
MINIMUM_ALPHA: float = 0.25
JITTER: float = 0.2
DEFAULT_COLORMAP: str = 'plasma'

# Set default matplotlib parameters
plt.rcParams.update({
    'image.cmap': DEFAULT_COLORMAP,
    'font.size': FONTSIZE,
    'axes.labelsize': FONTSIZE,
    'axes.titlesize': FONTSIZE,
    'xtick.labelsize': FONTSIZE * 0.8,
    'ytick.labelsize': FONTSIZE * 0.8,
    'legend.fontsize': FONTSIZE * 0.8,
})


def jitter(
        positions: Sequence[float],
        amount: float=JITTER,
) -> np.ndarray[float]:
    """Offset numerical positions by some random amount.

    Args:
        positions (Sequence[float]):
            Sequence of original numerical positions to jitter.
        amount (float):
            Maximum jitter amount to add or subtract at random.

    Returns:
        np.ndarray[float]:
            Jittered vector, with each original value changed within ±amount.

    Example:
        >>> original = [1.0, 2.0, 3.0]
        >>> jittered = jitter(original, amount=0.2)
        >>> all(abs(j - o) <= 0.2 for j, o in zip(jittered, original))
        True
    """
    offsets = np.random.uniform(-amount, amount, size=len(positions))
    return np.array(positions) + offsets


def min_alpha(
        n: int,
) -> float:
    """Dynamically compute the alpha opacity for n overlapping objects.

    Args:
        n (int):
            The number of objects whose opacity will be decreased.

    Returns:
        float:
            An appropiate alpha level above a baseline of visibility.

    Example:
        >>> min_alpha(2)
        0.5
        >>> min_alpha(4)
        0.25
        >>> min_alpha(40)
        0.25
    """
    if n <= 0:
        return 1
    return max(MINIMUM_ALPHA, 1 / floor(n))


def plottify(
        func: Callable[..., tuple[plt.Figure, plt.Axes]],
) ->  Callable[..., Union[tuple[plt.Figure, plt.Axes], None]]:
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
        tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]:
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
            **kwargs: Mapping[str, Any],
    ) -> Union[None, tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]]:
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

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    return wrapper


@plottify
def connectivity(
        matrix: np.ndarray,
        labels: Sequence[str] = [],
        colorbar: bool = True,
        fontsize: float = FONTSIZE,
) -> tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]:
    """Visualize a connectivity matrix as a heatmap.

    Args:
        matrix (np.ndarray):
            numerical 2D numpy array to visualize.
        labels (Sequence[str]):
            Labels for the x and y axes (brain regions).
        colorbar (bool):
            Whether to display a color bar.
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
    if labels and len(labels) != matrix.shape[0]:
        raise ValueError("Wrong number of labels.")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("Only numerical values are supported.")

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation='vertical', fontsize=0.2 * fontsize)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=0.2 * fontsize )

    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=0.5 * fontsize)

    return fig, ax


@plottify
def density(
        distributions: Mapping[str, Iterable[float]],
        bins: int = 10,
        fontsize: float = FONTSIZE,
) -> tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]:
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
        ax.hist(distribution, alpha=alpha, bins=bins, label=label)

    if len(distributions) > 1:
        ax.legend(fontsize=fontsize)

    return fig, ax


@plottify
def repeated_measures(
        measures: Mapping[str, Mapping[str, Any]],
        subject_colors: Optional[Mapping[str, str]] = None,
        subject_groups: Optional[Mapping[str, str]] = None,
        fontsize: float = FONTSIZE,
) -> tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]:
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
        values = np.array(
            [x if x is not None else np.nan for x in measures[measure].values()]
        )
        ax.bar(i, np.nanmean(values), yerr=np.nanstd(values), color='gray')

    # Gather time series for each unique Subject key across measurements.
    time_series = defaultdict(list)

    for measure in measures.values():
        for subject_label, datum in measure.items():
            if datum:
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
) -> tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]:
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
        x = jitter([i] * len(values))
        ax.plot(x, values, 'o')

    plt.xticks(range(len(groups)), groups.keys(), fontsize=fontsize)
    plt.xlim(-0.5, len(groups) - 0.5)
    return fig, ax


def compose(
    plots: Sequence[tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]],
    shape: Sequence[int] = (1, 1),
    title: str = "",
    figsize: Optional[Sequence[float]] = None,
    fontsize: float = FONTSIZE,
    output: bool = False,
) -> Union[None, tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]]:
    """
    Arrange a list of subplots in a grid.

    Args:
        plots (Sequence[tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]]):
            List of (figure, axis) tuples.
        shape (Sequence(int)):
            Shape of the grid (rows, columns).
        Plus everything listed in `help(plot.plottify)`.

    Returns:
        See `help(plot.plottify)`.
    """
    # Create a new figure for the composite plot.
    fig = plt.figure(figsize=figsize)

    if title:
        fig.suptitle(title, fontsize=fontsize)

    # Iterate over subplots.
    for i, (subfig, subax) in enumerate(plots):
        # Create a subplot in the composite figure.
        ax = fig.add_subplot(shape[0], shape[1], i + 1)

        # Copy title.
        ax.set_title(subax.get_title(), fontsize=fontsize)

        # Copy spines
        for spine_loc in ['top', 'bottom', 'left', 'right']:
            source_spine = subax.spines[spine_loc]
            target_spine = ax.spines[spine_loc]
            target_spine.set_visible(source_spine.get_visible())
            target_spine.set_position(source_spine.get_position())
            target_spine.set_color(source_spine.get_edgecolor())
            target_spine.set_linewidth(source_spine.get_linewidth())

        # Copy ticks, tick labels and axis labels.
        for dim in ('x', 'y', 'z'):
            try:
                eval(f'ax.set_{dim}ticks(subax.get_{dim}ticks())')
                eval(f'ax.set_{dim}ticklabels(subax.get_{dim}ticklabels())')
                eval(f'ax.set_{dim}label(subax.get_{dim}label())')
                eval(f'ax.set_{dim}lim(subax.get_{dim}lim())')
            except AttributeError:
                continue

        # Copy all artists (lines, patches, collections, images, etc.)
        for artist in subax.get_children():
            if isinstance(artist, (matplotlib.text.Text, matplotlib.axis.Axis)):
                continue

            # Lines from plot, scatter, hist.
            elif isinstance(artist, matplotlib.lines.Line2D):
                x, y = artist.get_data()
                ax.plot(x, y, **artist_properties(artist))

            # Patches (rectangles, circles, etc.) from bar, hist, etc.
            elif isinstance(artist, matplotlib.patches.Patch):
                if not isinstance(artist, matplotlib.spines.Spine):
                    properties = artist_properties(artist)
                    properties.pop('edgecolor')
                    patch = type(artist)(**properties)
                    ax.add_patch(patch)

            # Images from imshow.
            elif isinstance(artist, matplotlib.image.AxesImage):
                im = ax.imshow(
                    artist.get_array(),
                    norm=artist.norm,
                    **artist_properties(artist),
                )
                # Check if colorbar exists in original subplot
                if (
                        hasattr(subax, 'images') and
                        subax.images and
                        hasattr(subax.images[0], 'colorbar') and
                        subax.images[0].colorbar is not None
                ):
                    cbar = plt.colorbar(im, ax=ax)
                    original_cbar = subax.images[0].colorbar
                    cbar_labels = original_cbar.ax.yaxis.get_ticklabels()[0]
                    cbar_fontsize = cbar_labels.get_fontsize()
                    cbar.ax.tick_params(labelsize=cbar_fontsize)

            # Legends.
            elif isinstance(artist, matplotlib.legend.Legend):
                handles, labels = subax.get_legend_handles_labels()
                ax.legend(handles, labels)

            # Collections.
            elif isinstance(artist, matplotlib.collections.Collection):
                if hasattr(artist, 'get_facecolor'):
                    properties = artist_properties(artist)
                    properties.pop('fill')
                    properties.pop('segments')
                    if hasattr(artist, 'get_paths'):
                        paths = artist.get_paths()
                        collection = matplotlib.collections.PathCollection(
                            paths, **properties
                        )
                        ax.add_collection(collection)
                    elif hasattr(artist, 'get_offsets'):
                        offsets = artist.get_offsets()
                        if hasattr(artist, 'get_sizes'):
                            sizes = artist.get_sizes()
                            collection = matplotlib.collections.ScatterCollection(
                                sizes, **properties
                            )
                            new_coll.set_offsets(offsets)
                            ax.add_collection(new_coll)

            else:
                raise NotImplementedError(
                    f'Missing implementation for {type(artist)}.'
                )

    if output:
        plt.close(fig)
        return fig, ax

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def artist_properties(
        artist: matplotlib.artist.Artist,
) -> Dict[str, Any]:
    """Extract a dictionary of visual properties from a matplotlib artist.

    Dynamically retrieves common visual properties (e.g., color, linestyle,
    linewidth) from a matplotlib artist object (e.g., Line2D, Rectangle, Patch,
    etc.) by calling the corresponding getter methods. Only includes properties
    for which the getter method exists and returns a non-None value.

    Args:
        artist (matplotlib.artist.Artist):
            A matplotlib artist object from which to extract properties.

    Returns:
        Dict[str, Any]:
            A mapping of property names to their corresponding values.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> line, = ax.plot(
        ...     [1, 2, 3],
        ...     [4, 5, 6],
        ...     color='red',
        ...     linestyle='--',
        ... )
        >>> artist_properties(line)
        {
            'color': 'red',
            'linestyle': '--',
            'linewidth': 1.5,
            'alpha': 1.0
        }
    """
    getters = {
        'visible': 'get_visible',
        'alpha': 'get_alpha',
        'zorder': 'get_zorder',
        'clip_on': 'get_clip_on',
        # 'transform': 'get_transform',
        # 'figure': 'get_figure',
        'rasterized': 'get_rasterized',
        'url': 'get_url',
        'gid': 'get_gid',
        'path_effects': 'get_path_effects',
        # Line2D-specific.
        'color': 'get_color',
        'linestyle': 'get_linestyle',
        'linewidth': 'get_linewidth',
        'marker': 'get_marker',
        'markersize': 'get_markersize',
        'drawstyle': 'get_drawstyle',
        'markeredgecolor': 'get_markeredgecolor',
        'markeredgewidth': 'get_markeredgewidth',
        'markerfacecolor': 'get_markerfacecolor',
        # 'xdata': 'get_xdata',
        # 'ydata': 'get_ydata',
        # Patch-specific.
        'edgecolor': 'get_edgecolor',
        'facecolor': 'get_facecolor',
        'xy': 'get_xy',
        'width': 'get_width',
        'height': 'get_height',
        'hatch': 'get_hatch',
        'fill': 'get_fill',
        # Text-specific.
        'text': 'get_text',
        'position': 'get_position',
        'fontsize': 'get_fontsize',
        'fontname': 'get_fontname',
        'fontweight': 'get_fontweight',
        'horizontalalignment': 'get_horizontalalignment',
        'verticalalignment': 'get_verticalalignment',
        'rotation': 'get_rotation',
        # Image-specific.
        'cmap': 'get_cmap',
        'norm': 'get_norm',
        'extent': 'get_extent',
        'origin': 'get_origin',
        'interpolation': 'get_interpolation',
        # Collection-specific.
        'offsets': 'get_offsets',
        'segments': 'get_segments',
        'sizes': 'get_sizes',
    }
    properties = {
        property: p
        for property, getter in getters.items()
        if (p := run_method(artist, getter)) is not MISSING
    }
    return properties


class _Missing:
    """Sentinel value to distinguish between missing attributes and None."""
    def __repr__(self):
        return "<MISSING>"


MISSING = _Missing()


def run_method(
        obj: matplotlib.artist.Artist,
        method: str,
        *args: Sequence[Any],
        **kwargs: Mapping[str, Any],
) -> Union[Any, _Missing]:
    """Safely call a method on an object by name and return its result.

    Attempts to call a method (specified by its name as a string) on
     the given object.  If the method does not exist or is not
     callable, returns `None` instead of raising an exception.

    Args:
        obj (matplotlib.artist.Artist):
            The object on which to call the method.
        method (str):
            The name of the method to call.
        *args (Sequence[Any]):
            Variable positional arguments passed as `obj.method(*args)`.
        **kwargs (Mapping[str, Any]):
            Variable named arguments passed as `obj.method(**kwargs)`.

    Returns:
        Any:
            The result of the method call, if successful.
        MISSING:
            Otherwise.

    Example:
        >>> class Example:
        ...     def greet(self, message='Hello!'):
        ...         return message
        ...
        >>> example = Example()
        >>> run_method(example, 'greet')
        'Hello!'
        >>> run_method(example, 'greet', message=None)
        None
        >>> run_method(example, 'nonexistent_method')
        '<MISSING>'
    """
    try:
        attr =  getattr(obj, method)
    except AttributeError:
        return MISSING
    if callable(attr):
        return attr(*args, **kwargs)
    return MISSING
