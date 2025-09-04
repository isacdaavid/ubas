"""
Plotting functions
"""

import matplotlib.pyplot as plt
import numpy as np

def jitter(positions, amount = 0.1):
    offsets = np.random.uniform(-amount, amount, size=len(positions))
    return np.array(positions) + offsets

def bars(controls, patients, xlabel="", ylabel=""):
    plt.figure()
    label_c = "SC vs empirical FC"
    plt.bar(0, controls.mean(), yerr=controls.std(), label=label_c)
    label_p = "Simulated FC vs empirical FC"
    plt.bar(1, patients.mean(), yerr=patients.std(), label=label_p)
    plt.xticks([1, 2], ['', ''], fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend()
    plt.show()

def scatter(controls, patients, xlabel="", ylabel=""):
    plt.figure()
    x = jitter([1] * len(controls))
    plt.plot(x, controls, 'o', color='blue')
    x = jitter([2] * len(patients))
    plt.plot(x, patients, 'o', color='orange')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks([1, 2], ["HC", "SSD"], fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0.5, 2.5)
    plt.show()

def density(controls, patients, xlabel="", ylabel=""):
    plt.figure()
    plt.hist(controls, alpha=0.5, bins=40, color='blue', label='HC')
    plt.hist(patients, alpha=0.5, bins=40, color='orange', label='SSD')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.yticks(fontsize=14)
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
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlim(*xlim)
    plt.show()
