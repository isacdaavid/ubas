"""
Plotting functions
"""

from collections import defaultdict
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 14

def jitter(positions: Sequence[float], amount: float=0.1) -> np.array:
    """Offset numerical positions by some random amount."""
    offsets = np.random.uniform(-amount, amount, size=len(positions))
    return np.array(positions) + offsets

def repeated_measures(
        measures: Mapping[str, Mapping[str, Any]],
        subject_colors: Mapping[str, str]=None,
        subject_groups: Mapping[str, str]=None,
        xlabel: str="",
        ylabel: str="",
) -> None:
    """
    Plot repeated measures as bars (mean ± std) and individual time series.

    Args:
        measures: A dict of measurement names to subject data ({subject: datum}).
        subject_colors: Optional dict of subject labels to colors.
        subject_groups: Optional dict of subject labels to group labels.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
    """
    plt.figure()
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
    plt.show()

def scatter(controls, patients, xlabel="", ylabel=""):
    plt.figure()
    x = jitter([1] * len(controls))
    plt.plot(x, controls, 'o', color='blue')
    x = jitter([2] * len(patients))
    plt.plot(x, patients, 'o', color='orange')
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.xticks([1, 2], ["HC", "SSD"], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlim(0.5, 2.5)
    plt.show()

def density(controls, patients, xlabel="", ylabel=""):
    plt.figure()
    plt.hist(controls, alpha=0.5, bins=40, color='blue', label='HC')
    plt.hist(patients, alpha=0.5, bins=40, color='orange', label='SSD')
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend()
    plt.show()

def spectrum(controls, patients, xlim, xlabel="", ylabel="Power"):
    sample_rate = 1 / 0.5
    plt.figure()
    for control in controls:
        freq = np.fft.fftfreq(control.shape[0], sample_rate)
        plt.plot(freq, control, color='blue', alpha=1)
    for patient in patients:
        freq = np.fft.fftfreq(patient.shape[0], sample_rate)
        plt.plot(freq, patient, color='orange', alpha=1)
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    plt.xlim(*xlim)
    plt.show()
